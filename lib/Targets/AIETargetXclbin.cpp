//===- AIETargetXclbin.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements translation from xclbin binaries to MLIR.
// It parses the AXLF xclbin format, extracts PDI (Programmable Device Image)
// sections, extracts CDO (Configuration Data Object) binaries from PDIs,
// decodes CDO commands using bootgen's decoder, and lifts register writes
// to MLIR operations (aiex.npu.write32, aiex.npu.maskwrite32, aiex.npu.blockwrite).
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIEDMABDLifting.h"
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "xrt/detail/xclbin.h"

#ifdef HAVE_BOOTGEN
extern "C" {
#include <cdo-binary.h>
#include <cdo-command.h>
}
#endif

#include <cstring>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// Helper class to manage lifted BD emission state
class LiftedBDEmitter {
public:
  LiftedBDEmitter(OpBuilder &builder, AIE::DeviceOp device)
      : builder(builder), device(device) {}

  /// Get or create a tile operation
  AIE::TileOp getOrCreateTile(int col, int row) {
    TileID id{col, row};
    auto it = tiles.find(id);
    if (it != tiles.end())
      return it->second;

    auto tile = AIE::TileOp::getOrCreate(builder, device, col, row);
    tiles[id] = tile;
    return tile;
  }

  /// Get or create a buffer for a BD (compute tiles)
  Value getOrCreateBuffer(const ParsedBDConfig &bd) {
    // Create a unique buffer name based on tile and BD info
    std::string bufName = llvm::formatv("bd_buf_{0}_{1}_{2}",
                                        bd.column, bd.row, bd.bdIndex);

    auto it = buffers.find(bufName);
    if (it != buffers.end())
      return it->second;

    // Create buffer with anonymous memref type
    auto tile = getOrCreateTile(bd.column, bd.row);
    auto memrefType = MemRefType::get({static_cast<int64_t>(bd.bufferLength)},
                                      builder.getI32Type());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto bufOp = builder.create<AIE::BufferOp>(
        builder.getUnknownLoc(), memrefType, tile,
        builder.getStringAttr(bufName), nullptr, nullptr, nullptr);

    buffers[bufName] = bufOp.getResult();
    return bufOp.getResult();
  }

  /// Get or create an external buffer for a BD (shim tiles)
  Value getOrCreateExternalBuffer(const ParsedBDConfig &bd) {
    // Create a unique buffer name based on tile and BD info
    std::string bufName = llvm::formatv("ext_buf_{0}_{1}_{2}",
                                        bd.column, bd.row, bd.bdIndex);

    auto it = externalBuffers.find(bufName);
    if (it != externalBuffers.end())
      return it->second;

    // Create external buffer with memref type
    auto memrefType = MemRefType::get({static_cast<int64_t>(bd.bufferLength)},
                                      builder.getI32Type());

    OpBuilder::InsertionGuard guard(builder);
    // Insert external buffers at the beginning of the device block
    builder.setInsertionPointToStart(&device.getRegion().front());

    auto extBufOp = builder.create<AIE::ExternalBufferOp>(
        builder.getUnknownLoc(), memrefType,
        builder.getStringAttr(bufName), nullptr);

    externalBuffers[bufName] = extBufOp.getResult();
    return extBufOp.getResult();
  }

  /// Get or create a lock operation
  Value getOrCreateLock(int col, int row, int lockId) {
    // Create a unique key for this lock
    auto lockKey = std::make_tuple(col, row, lockId);
    auto it = locks.find(lockKey);
    if (it != locks.end())
      return it->second;

    // Create lock operation
    auto tile = getOrCreateTile(col, row);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto lockOp = builder.create<AIE::LockOp>(
        builder.getUnknownLoc(),
        builder.getIndexType(),
        tile,
        builder.getI32IntegerAttr(lockId),
        nullptr,  // init
        nullptr   // sym_name
    );

    locks[lockKey] = lockOp.getResult();
    return lockOp.getResult();
  }

  /// Record a BD configuration for later emission
  void recordBD(const ParsedBDConfig &bd) {
    TileID id{bd.column, bd.row};
    tileBDs[id].push_back(bd);
  }

  /// Record a switchbox connection for later emission
  void recordSwitchboxConnection(const SwitchConnectionInfo &conn) {
    SwitchboxAccumulator::SwitchboxKey key{conn.column, conn.row, conn.tileType};

    // Add connection to the switchbox config
    ParsedSwitchboxConfig &config = switchboxes[key];
    config.column = conn.column;
    config.row = conn.row;
    config.tileType = conn.tileType;

    ParsedSwitchboxConfig::Connection newConn;
    newConn.sourceBundle = conn.sourceBundle;
    newConn.sourceChannel = conn.sourceChannel;
    newConn.destBundle = conn.destBundle;
    newConn.destChannel = conn.destChannel;
    newConn.isPacketMode = conn.packetMode;

    config.connections.push_back(newConn);
  }

  /// Emit all collected BDs as aie.mem or aie.shim_dma operations
  void emitAllBDs() {
    auto &targetModel = getTargetModel(device);
    for (const auto &[tileId, bds] : tileBDs) {
      if (targetModel.isShimNOCorPLTile(tileId.col, tileId.row)) {
        // Shim tiles use aie.shim_dma instead of aie.mem
        emitShimDmaOpForTile(tileId, bds);
      } else {
        // Compute and memory tiles use aie.mem
        emitMemOpForTile(tileId, bds);
      }
    }
  }

  /// Emit all collected switchboxes as aie.switchbox operations
  void emitAllSwitchboxes() {
    for (const auto &[key, config] : switchboxes) {
      emitSwitchboxForTile(config);
    }
  }

  /// Check if a BD was lifted (to suppress raw write emission)
  bool wasLifted(uint32_t addr) const {
    return liftedAddresses.count(addr) > 0;
  }

  /// Mark an address as lifted
  void markLifted(uint32_t addr) {
    liftedAddresses.insert(addr);
  }

private:
  void emitMemOpForTile(TileID tileId, const llvm::SmallVector<ParsedBDConfig> &bds) {
    auto tile = getOrCreateTile(tileId.col, tileId.row);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto memOp = builder.create<AIE::MemOp>(builder.getUnknownLoc(),
                                             builder.getIndexType(), tile);
    Block *memBlock = &memOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(memBlock);

    // Emit all BDs for this tile
    for (const auto &bd : bds) {
      emitSingleBD(bd, memBlock);
    }

    // Terminate with aie.end
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  void emitShimDmaOpForTile(TileID tileId, const llvm::SmallVector<ParsedBDConfig> &bds) {
    auto tile = getOrCreateTile(tileId.col, tileId.row);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto shimDmaOp = builder.create<AIE::ShimDMAOp>(builder.getUnknownLoc(),
                                                     builder.getIndexType(), tile);
    Block *shimDmaBlock = &shimDmaOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(shimDmaBlock);

    // Emit all BDs for this shim tile
    for (const auto &bd : bds) {
      emitSingleShimBD(bd, shimDmaBlock);
    }

    // Terminate with aie.end
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  void emitSingleBD(const ParsedBDConfig &bd, Block *memBlock) {
    OpBuilder::InsertionGuard guard(builder);

    // Create a basic block for this BD
    Block *bdBlock = new Block();
    memBlock->getParent()->push_back(bdBlock);
    builder.setInsertionPointToEnd(bdBlock);

    // Emit lock acquire if needed
    emitLockAcquire(bd);

    // Emit the dma_bd operation
    auto buffer = getOrCreateBuffer(bd);

    // Build dimension attributes if needed
    AIE::BDDimLayoutArrayAttr dimensions = buildDimensionAttrs(bd);

    auto bdOp = builder.create<AIE::DMABDOp>(
        builder.getUnknownLoc(),
        buffer,
        bd.baseAddress,
        bd.bufferLength
    );
    if (dimensions)
      bdOp.setDimensionsAttr(dimensions);
    if (bd.bdIndex >= 0)
      bdOp.setBdIdAttr(builder.getI32IntegerAttr(bd.bdIndex));

    // Emit lock release if needed
    emitLockRelease(bd);

    // Terminate the block
    emitBlockTerminator(bd);
  }

  void emitSingleShimBD(const ParsedBDConfig &bd, Block *shimDmaBlock) {
    OpBuilder::InsertionGuard guard(builder);

    // Create a basic block for this BD
    Block *bdBlock = new Block();
    shimDmaBlock->getParent()->push_back(bdBlock);
    builder.setInsertionPointToEnd(bdBlock);

    // Emit lock acquire if needed
    emitLockAcquire(bd);

    // Emit the dma_bd operation using external buffer
    auto buffer = getOrCreateExternalBuffer(bd);

    // Build dimension attributes if needed (same as for compute tiles)
    AIE::BDDimLayoutArrayAttr dimensions = buildDimensionAttrs(bd);

    auto bdOp = builder.create<AIE::DMABDOp>(
        builder.getUnknownLoc(),
        buffer,
        0,  // Offset for external buffers is typically 0
        bd.bufferLength
    );
    if (dimensions)
      bdOp.setDimensionsAttr(dimensions);
    if (bd.bdIndex >= 0)
      bdOp.setBdIdAttr(builder.getI32IntegerAttr(bd.bdIndex));

    // Emit lock release if needed
    emitLockRelease(bd);

    // Terminate the block
    emitBlockTerminator(bd);
  }

  void emitSwitchboxForTile(const ParsedSwitchboxConfig &config) {
    if (!config.hasConnections()) {
      return;
    }

    auto tile = getOrCreateTile(config.column, config.row);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    // Create aie.switchbox operation
    auto switchboxOp = builder.create<AIE::SwitchboxOp>(
        builder.getUnknownLoc(),
        builder.getIndexType(),
        tile
    );

    Block *switchboxBlock = &switchboxOp.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(switchboxBlock);

    // Emit all connections
    for (const auto &conn : config.connections) {
      builder.create<AIE::ConnectOp>(
          builder.getUnknownLoc(),
          conn.sourceBundle,
          conn.sourceChannel,
          conn.destBundle,
          conn.destChannel
      );
    }

    // Terminate with aie.end
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  /// Helper: Emit lock acquire operation if BD has lock acquire
  void emitLockAcquire(const ParsedBDConfig &bd) {
    if (!bd.hasLockAcquire()) {
      return;
    }

    auto lock = getOrCreateLock(bd.column, bd.row, bd.lockAcquire.lockId);

    // Determine lock action based on sign of value
    AIE::LockAction action;
    int32_t lockValue;
    if (bd.lockAcquire.value < 0) {
      action = AIE::LockAction::AcquireGreaterEqual;
      lockValue = -bd.lockAcquire.value;  // Use absolute value
    } else {
      action = AIE::LockAction::Acquire;
      lockValue = bd.lockAcquire.value;
    }

    builder.create<AIE::UseLockOp>(
        builder.getUnknownLoc(),
        lock,
        action,
        lockValue
    );
  }

  /// Helper: Build dimension attributes from BD configuration
  AIE::BDDimLayoutArrayAttr buildDimensionAttrs(const ParsedBDConfig &bd) {
    if (!bd.hasDimensions()) {
      return nullptr;
    }

    SmallVector<Attribute> dimAttrs;

    // Build dimension layout attributes from outermost to innermost
    // Hardware dimension mapping:
    //   - bd.dimensions[0] = D0 (innermost)
    //   - bd.dimensions[1] = D1 (middle)
    //   - bd.dimensions[2] = D2 (outermost)

    // D2 dimension (outermost) - only if D1 has wrap
    if (bd.dimensions[1].wrap != 0) {
      auto dimAttr = AIE::BDDimLayoutAttr::get(
          builder.getContext(),
          bd.dimensions[1].wrap,      // size
          bd.dimensions[2].stepSize   // stride
      );
      dimAttrs.push_back(dimAttr);
    }

    // D1 dimension (middle) - only if D0 has wrap
    if (bd.dimensions[0].wrap != 0) {
      auto dimAttr = AIE::BDDimLayoutAttr::get(
          builder.getContext(),
          bd.dimensions[0].wrap,      // size
          bd.dimensions[1].stepSize   // stride
      );
      dimAttrs.push_back(dimAttr);
    }

    // D0 dimension (innermost) - always included when dimensions are non-trivial
    auto dimAttr = AIE::BDDimLayoutAttr::get(
        builder.getContext(),
        bd.bufferLength,            // size (in 32-bit words)
        bd.dimensions[0].stepSize   // stride
    );
    dimAttrs.push_back(dimAttr);

    if (dimAttrs.empty()) {
      return nullptr;
    }

    SmallVector<AIE::BDDimLayoutAttr> bdDimAttrs;
    for (auto attr : dimAttrs) {
      bdDimAttrs.push_back(llvm::cast<AIE::BDDimLayoutAttr>(attr));
    }
    return AIE::BDDimLayoutArrayAttr::get(builder.getContext(), bdDimAttrs);
  }

  /// Helper: Emit lock release operation if BD has lock release
  void emitLockRelease(const ParsedBDConfig &bd) {
    if (!bd.hasLockRelease()) {
      return;
    }

    auto lock = getOrCreateLock(bd.column, bd.row, bd.lockRelId);

    builder.create<AIE::UseLockOp>(
        builder.getUnknownLoc(),
        lock,
        AIE::LockAction::Release,
        std::abs(bd.lockRelValue)  // Use absolute value
    );
  }

  /// Helper: Emit block termination (EndOp)
  void emitBlockTerminator(const ParsedBDConfig &bd) {
    // For now, create a simple end terminator
    // Full next_bd support requires block references
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  OpBuilder &builder;
  AIE::DeviceOp device;
  llvm::DenseMap<TileID, AIE::TileOp> tiles;
  llvm::StringMap<Value> buffers;
  llvm::StringMap<Value> externalBuffers;  // External buffers for shim tiles
  llvm::DenseMap<std::tuple<int, int, int>, Value> locks;  // (col, row, lockId) -> lock Value
  llvm::DenseMap<TileID, llvm::SmallVector<ParsedBDConfig>> tileBDs;
  llvm::DenseSet<uint32_t> liftedAddresses;

  // Switchbox storage
  std::map<SwitchboxAccumulator::SwitchboxKey, ParsedSwitchboxConfig> switchboxes;
};

#ifdef HAVE_BOOTGEN
/// Extract PDI (Programmable Device Image) section from xclbin binary data.
/// Parses the AXLF format and finds the PDI section.
/// Note: xclbinData is the actual binary content (not a filename), as MLIR's
/// translation framework reads the file and passes the content.
LogicalResult extractPDIFromXclbin(StringRef xclbinData,
                                   std::vector<uint8_t> &pdiData) {
  // xclbinData contains the actual xclbin binary content
  const uint8_t *data =
      reinterpret_cast<const uint8_t *>(xclbinData.data());
  size_t size = xclbinData.size();

  // Parse AXLF header
  if (size < sizeof(axlf)) {
    llvm::errs() << "xclbin file too small to contain valid AXLF header\n";
    return failure();
  }

  const axlf *header = reinterpret_cast<const axlf *>(data);

  // Verify magic
  if (std::memcmp(header->m_magic, "xclbin2\0", 8) != 0) {
    llvm::errs() << "Invalid xclbin magic (expected 'xclbin2\\0')\n";
    return failure();
  }

  // Access section headers
  // Note: axlf has m_sections[1] as a placeholder, but actual count is variable
  // We need to calculate the offset properly
  size_t headerSize = sizeof(axlf) - sizeof(axlf_section_header);
  const axlf_section_header *sections =
      reinterpret_cast<const axlf_section_header *>(data + headerSize);

  // Find PDI section (or AIE_PARTITION section for NPU xclbins)
  uint32_t numSections = header->m_header.m_numSections;

  // First, try to find a PDI section (kind 18)
  for (uint32_t i = 0; i < numSections; i++) {
    if (sections[i].m_sectionKind == PDI) {
      // Found PDI section
      uint64_t offset = sections[i].m_sectionOffset;
      uint64_t len = sections[i].m_sectionSize;

      if (offset + len > size) {
        llvm::errs() << "PDI section offset/size extends beyond file\n";
        return failure();
      }

      pdiData.resize(len);
      std::memcpy(pdiData.data(), data + offset, len);

      return success();
    }
  }

  // For NPU xclbins, try AIE_PARTITION section (kind 32) which contains CDO directly
  for (uint32_t i = 0; i < numSections; i++) {
    if (sections[i].m_sectionKind == AIE_PARTITION) {
      // Found AIE partition section - treat as PDI for NPU
      uint64_t offset = sections[i].m_sectionOffset;
      uint64_t len = sections[i].m_sectionSize;

      if (offset + len > size) {
        llvm::errs() << "AIE_PARTITION section offset/size extends beyond file\n";
        return failure();
      }

      pdiData.resize(len);
      std::memcpy(pdiData.data(), data + offset, len);

      return success();
    }
  }

  llvm::errs() << "No PDI or AIE_PARTITION section found in xclbin\n";
  return failure();
}

/// Extract CDO (Configuration Data Object) from PDI binary.
/// Scans the PDI for CDO magic bytes and extracts the CDO section.
LogicalResult extractCDOFromPDI(const uint8_t *pdiData, size_t pdiSize,
                                std::vector<uint8_t> &cdoData) {
  // CDO magic: "CDO\0" in little-endian (0x004F4443)
  const uint8_t cdoMagic[] = {0x43, 0x44, 0x4F, 0x00};

  // Scan PDI for CDO header
  for (size_t i = 0; i < pdiSize - 20; i++) {
    if (std::memcmp(pdiData + i, cdoMagic, 4) == 0) {
      // Found CDO identification word
      // CDO header structure:
      // uint32_t NumWords (at i-8)
      // uint32_t IdentWord (at i) - this is what we found
      // uint32_t Version (at i+4)
      // uint32_t CDOLength (at i+8)
      // uint32_t CheckSum (at i+12)

      if (i < 4) {
        // Not enough space for NumWords field before ident
        continue;
      }

      const uint32_t *headerPtr =
          reinterpret_cast<const uint32_t *>(pdiData + i - 4);
      uint32_t numWords = *headerPtr;

      // Sanity check - NumWords should be small (typically 4-5)
      if (numWords > 0x100) {
        continue;
      }

      // Read CDO length from header
      const uint32_t *lenPtr =
          reinterpret_cast<const uint32_t *>(pdiData + i + 8);
      uint32_t cdoLen = *lenPtr;  // Length in 32-bit words

      // Sanity check on length
      if (cdoLen > 0x100000) {  // Max 1M words = 4MB
        continue;
      }

      // Calculate total CDO size in bytes
      // Header is (4 + numWords) words, payload is cdoLen words
      size_t totalLen = (4 + numWords + cdoLen) * 4;

      if (i - 4 + totalLen > pdiSize) {
        llvm::errs() << "CDO extends beyond PDI bounds\n";
        continue;
      }

      // Extract full CDO (header + payload)
      cdoData.resize(totalLen);
      std::memcpy(cdoData.data(), pdiData + i - 4, totalLen);

      return success();
    }
  }

  llvm::errs() << "No CDO found in PDI\n";
  return failure();
}
#endif // HAVE_BOOTGEN (for extractPDI and extractCDO)

#ifdef HAVE_BOOTGEN
/// Decode CDO binary using bootgen's decoder.
/// Returns a list of CdoCommand structures.
LogicalResult decodeCDOToCmds(const uint8_t *data, size_t len,
                              std::vector<CdoCommand *> &commands) {
  // Call bootgen's CDO decoder
  CdoSequence *seq = decode_cdo_binary(data, len);
  if (!seq) {
    llvm::errs() << "Failed to decode CDO binary\n";
    return failure();
  }

  // Extract commands from linked list
  // seq->cmds is a LINK, use all2cmds macro to get first CdoCommand
  for (LINK *link = seq->cmds.next; link != &seq->cmds; link = link->next) {
    CdoCommand *cmd = all2cmds(link);
    commands.push_back(cmd);
  }

  return success();
}

/// Emit MLIR operations from decoded CDO commands.
/// Creates aie.device, runtime_sequence, and MLIR operations for register writes.
LogicalResult emitMLIRFromCDO(ModuleOp module,
                              llvm::ArrayRef<CdoCommand *> commands,
                              RegisterDatabase *regDB = nullptr,
                              bool emitLifted = false) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());

  // Create aie.device
  auto deviceOp = AIE::DeviceOp::create(
      builder, builder.getUnknownLoc(),
      AIE::AIEDeviceAttr::get(builder.getContext(), AIE::AIEDevice::npu1_1col),
      mlir::StringAttr::get(builder.getContext(), "xclbin_device"));

  Block *deviceBlock = &deviceOp.getRegion().emplaceBlock();
  builder.setInsertionPointToEnd(deviceBlock);

  // Create runtime_sequence
  auto seqOp = AIE::RuntimeSequenceOp::create(
      builder, builder.getUnknownLoc(), "configure");

  Block *seqBlock = &seqOp.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(seqBlock);

  // Initialize BD semantic lifting utilities
  BDAddressParser bdParser(/*numMemTileRows=*/1);
  BDAccumulator bdAccum;

  // Initialize switchbox semantic lifting utilities
  SwitchAddressParser switchParser(/*numMemTileRows=*/1);
  SwitchboxAccumulator switchAccum;

  std::unique_ptr<LiftedBDEmitter> liftedEmitter;

  if (emitLifted) {
    liftedEmitter = std::make_unique<LiftedBDEmitter>(builder, deviceOp);
  }

  // First pass: collect blockwrite data and create memref.global operations
  int blockwriteIdx = 0;
  llvm::DenseMap<CdoCommand *, int> blockwriteMap;

  for (CdoCommand *cmd : commands) {
    if (cmd->type == CdoCmdSetBlock) {
      // Create memref.global for blockwrite data
      std::string globalName =
          "cdo_blockwrite_" + std::to_string(blockwriteIdx);

      SmallVector<int32_t> dataVec;
      uint32_t *dataPtr = reinterpret_cast<uint32_t *>(cmd->buf);
      for (uint32_t j = 0; j < cmd->count; j++) {
        dataVec.push_back(dataPtr[j]);
      }

      auto memrefType =
          MemRefType::get({static_cast<int64_t>(cmd->count)},
                         builder.getI32Type());
      auto dataAttr = DenseIntElementsAttr::get(memrefType, dataVec);

      builder.setInsertionPointToStart(deviceBlock);
      builder.create<memref::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(globalName),
          builder.getStringAttr("private"), memrefType, dataAttr,
          /*constant=*/true, /*alignment=*/nullptr);

      blockwriteMap[cmd] = blockwriteIdx++;
      builder.setInsertionPointToEnd(seqBlock);
    }
  }

  // Second pass: emit operations in sequence
  for (size_t cmdIdx = 0; cmdIdx < commands.size(); cmdIdx++) {
    CdoCommand *cmd = commands[cmdIdx];
    Location loc = builder.getUnknownLoc();

    switch (cmd->type) {
    case CdoCmdWrite: {
      // aiex.npu.write32
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      uint32_t value = cmd->value;

      // Check if this is a BD register write and accumulate
      auto completedBD = bdAccum.addWrite(addr, value, bdParser);

      // Check if this is a switchbox register write and accumulate
      auto switchConn = switchAccum.addMasterWrite(addr, value, switchParser);

      // If lifted mode and this is a BD write, mark addresses for suppression
      // However, skip shim tiles since they don't have local memory and can't
      // have aie.mem operations - keep them as raw writes
      bool shouldEmitRaw = true;
      if (emitLifted && bdParser.isBDAddress(addr)) {
        auto addrInfo = bdParser.parse(addr);
        auto &targetModel = getTargetModel(deviceOp);
        bool isShimTile = targetModel.isShimNOCorPLTile(addrInfo.column, addrInfo.row);

        if (!isShimTile) {
          shouldEmitRaw = false;  // Don't emit raw write for BD registers (except shim)
          if (liftedEmitter) {
            liftedEmitter->markLifted(addr);
          }
        }
      }

      // If lifted mode and this is a switchbox write, mark for suppression
      if (emitLifted && switchParser.isSwitchboxAddress(addr)) {
        shouldEmitRaw = false;  // Don't emit raw write for switchbox registers
        if (liftedEmitter) {
          liftedEmitter->markLifted(addr);
        }
      }

      // If a BD configuration was completed, record it for lifted emission
      if (completedBD.has_value()) {
        if (emitLifted && liftedEmitter) {
          liftedEmitter->recordBD(*completedBD);
        }
      }

      // If a switchbox connection was configured, record it for lifted emission
      if (switchConn.has_value()) {
        if (emitLifted && liftedEmitter) {
          liftedEmitter->recordSwitchboxConnection(*switchConn);
        }
      }

      // Emit raw write only if not suppressed
      if (shouldEmitRaw) {
        AIEX::NpuWrite32Op::create(builder, loc, addr, value,
                                   nullptr, nullptr, nullptr);
      }
      break;
    }

    case CdoCmdMaskWrite: {
      // aiex.npu.maskwrite32
      // Signature: (address, value, mask, buffer, column, row)
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      uint32_t mask = cmd->mask;
      uint32_t value = cmd->value;

      AIEX::NpuMaskWrite32Op::create(builder, loc, addr, value, mask,
                                     nullptr, nullptr, nullptr);
      break;
    }

    case CdoCmdSetBlock: {
      // aiex.npu.blockwrite
      int idx = blockwriteMap[cmd];
      std::string globalName = "cdo_blockwrite_" + std::to_string(idx);

      auto memrefType =
          MemRefType::get({static_cast<int64_t>(cmd->count)},
                         builder.getI32Type());
      auto getGlobal = builder.create<memref::GetGlobalOp>(
          loc, memrefType, builder.getStringAttr(globalName));

      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);

      AIEX::NpuBlockWriteOp::create(builder, loc, addr,
                                    getGlobal.getResult(),
                                    nullptr, nullptr, nullptr);
      break;
    }

    default:
      // Skip unsupported commands (NOP, etc.)
      break;
    }
  }

  // Flush any incomplete BD configurations at the end
  auto pendingBDs = bdAccum.flush();
  if (!pendingBDs.empty()) {
    if (emitLifted && liftedEmitter) {
      // Record incomplete BDs for lifted emission too
      for (const auto &bd : pendingBDs) {
        liftedEmitter->recordBD(bd);
      }
    } else {
      // Warn about incomplete BDs
      for (const auto &bd : pendingBDs) {
        llvm::errs() << "Warning: Incomplete BD configuration found (tile "
                     << bd.column << "," << bd.row << " BD " << bd.bdIndex << ")\n";
      }
    }
  }

  // Add terminator to runtime_sequence block
  builder.setInsertionPointToEnd(seqBlock);
  builder.create<AIE::EndOp>(builder.getUnknownLoc());

  // Emit all lifted BDs and switchboxes
  if (emitLifted && liftedEmitter) {
    builder.setInsertionPointToStart(deviceBlock);
    liftedEmitter->emitAllBDs();
    liftedEmitter->emitAllSwitchboxes();
  }

  // Add terminator to device block
  builder.setInsertionPointToEnd(deviceBlock);
  builder.create<AIE::EndOp>(builder.getUnknownLoc());

  return success();
}
#endif // HAVE_BOOTGEN

} // namespace

namespace xilinx {
namespace AIE {

/// Main entry point: translate xclbin binary to MLIR module.
LogicalResult AIETranslateFromXclbin(ModuleOp module, StringRef filename,
                                     bool emitLifted) {
  // Step 1: Load register database for AIE2
  auto regDB = RegisterDatabase::loadAIE2();
  if (!regDB) {
    llvm::errs() << "Warning: Failed to load register database. "
                 << "Register names will not be annotated.\n";
  }

#ifdef HAVE_BOOTGEN
  // Step 2: Extract PDI from xclbin
  std::vector<uint8_t> pdiData;
  if (failed(extractPDIFromXclbin(filename, pdiData))) {
    return module.emitError("Failed to extract PDI from xclbin");
  }

  // Step 3: Extract CDO from PDI
  std::vector<uint8_t> cdoData;
  if (failed(extractCDOFromPDI(pdiData.data(), pdiData.size(), cdoData))) {
    return module.emitError("Failed to extract CDO from PDI");
  }
  // Step 4: Decode CDO to commands
  std::vector<CdoCommand *> commands;
  if (failed(
          decodeCDOToCmds(cdoData.data(), cdoData.size(), commands))) {
    return module.emitError("Failed to decode CDO binary");
  }

  // Step 5: Emit MLIR operations (lifted or annotated mode)
  if (failed(emitMLIRFromCDO(module, commands, regDB.get(), emitLifted))) {
    return module.emitError("Failed to emit MLIR from CDO commands");
  }

  return success();
#else
  return module.emitError("CDO decoding not available - bootgen library was not built (OpenSSL required)");
#endif
}

} // namespace AIE
} // namespace xilinx
