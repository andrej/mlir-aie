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
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"
#include "aie/Dialect/AIE/Util/AIEFlowReconstruction.h"
#include "aie/Dialect/AIE/Util/AIELockLifting.h"
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

    // Create buffer with memref type
    // Note: bufferLength may be 0 if BD was not fully configured in CDO
    // (e.g., configured dynamically at runtime via NPU instruction stream).
    // In such cases, use a placeholder size of 1 since aie.buffer requires
    // static dimensions. The user will need to update this with the correct size.
    auto tile = getOrCreateTile(bd.column, bd.row);
    int64_t bufSize = (bd.bufferLength == 0) ? 1 : static_cast<int64_t>(bd.bufferLength);
    auto memrefType = MemRefType::get({bufSize}, builder.getI32Type());

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
    // Note: bufferLength may be 0 if BD was not fully configured in CDO
    // (e.g., configured dynamically at runtime via NPU instruction stream).
    // In such cases, use a placeholder size of 1 since aie.external_buffer requires
    // static dimensions. The user will need to update this with the correct size.
    int64_t bufSize = (bd.bufferLength == 0) ? 1 : static_cast<int64_t>(bd.bufferLength);
    auto memrefType = MemRefType::get({bufSize}, builder.getI32Type());

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
  Value getOrCreateLock(int col, int row, int lockId, std::optional<int32_t> initValue = std::nullopt) {
    // Create a unique key for this lock
    auto lockKey = std::make_tuple(col, row, lockId);
    auto it = locks.find(lockKey);
    if (it != locks.end())
      return it->second;

    // Create lock operation
    auto tile = getOrCreateTile(col, row);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    // Create init attribute if value provided
    IntegerAttr initAttr = nullptr;
    if (initValue.has_value()) {
      initAttr = builder.getI32IntegerAttr(initValue.value());
    }

    auto lockOp = builder.create<AIE::LockOp>(
        builder.getUnknownLoc(),
        builder.getIndexType(),
        tile,
        builder.getI32IntegerAttr(lockId),
        initAttr,  // init value
        nullptr    // sym_name
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

  /// Record a lock configuration for later emission
  void recordLock(const ParsedLockConfig &lock) {
    LockAccumulator::LockKey key{lock.column, lock.row, lock.lockId};
    lockConfigs[key] = lock;
  }

  /// Record a tile reference from a register address
  /// This ensures all tiles (including shim tiles) are emitted even if they
  /// don't have BDs, switchboxes, or locks configured via CDO
  void recordTileReference(int col, int row) {
    TileID id{col, row};
    referencedTiles.insert(id);
  }

  /// Extract tile coordinates from a register address
  /// Returns true if coordinates were successfully extracted
  static bool extractTileCoordinates(uint32_t addr, int &col, int &row) {
    // AIE2 address format: base + (col * 32 + row_offset) * 0x100000
    // Extract tile offset from address
    constexpr uint32_t kTileAddrShift = 20;  // 0x100000 per tile
    uint32_t tileOffset = (addr >> kTileAddrShift) & 0xFFF;

    col = tileOffset / 32;
    int rowPart = tileOffset % 32;

    // Row mapping:
    // rowPart 0 = row 0 (shim)
    // rowPart 1 = row 1 (memory tile)
    // rowPart 2-5 = rows 2-5 (compute tiles)
    row = rowPart;

    // Validate coordinates are reasonable
    // Column should be < 64, row should be < 32
    return (col < 64 && row < 32);
  }

  /// Emit standalone tile declarations for tiles that were referenced
  /// but don't have BDs, switchboxes, or locks
  void emitStandaloneTiles() {
    // Build a set of tiles that already have operations (BDs, switchboxes, locks)
    llvm::DenseSet<TileID> tilesWithOps;

    // Tiles with BDs
    for (const auto &[tileId, _] : tileBDs) {
      tilesWithOps.insert(tileId);
    }

    // Tiles with switchboxes
    for (const auto &[key, _] : switchboxes) {
      TileID id{key.col, key.row};
      tilesWithOps.insert(id);
    }

    // Tiles with locks
    for (const auto &[key, _] : lockConfigs) {
      auto [col, row, lockId] = key;
      TileID id{col, row};
      tilesWithOps.insert(id);
    }

    // Emit standalone tiles for referenced tiles without operations
    for (const auto &tileId : referencedTiles) {
      if (!tilesWithOps.contains(tileId)) {
        // Just ensure the tile exists - getOrCreateTile will emit it
        getOrCreateTile(tileId.col, tileId.row);
      }
    }
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

  /// Reconstruct and emit aie.flow operations from switchbox connections
  void emitAllFlows() {
    // Build flow reconstruction graph from all switchbox configs
    FlowReconstructionGraph flowGraph;
    for (const auto &[key, config] : switchboxes) {
      flowGraph.addSwitchboxConfig(config);
    }

    // Reconstruct end-to-end flows
    auto flows = flowGraph.reconstructFlows();

    if (flows.empty()) {
      return;  // No flows to emit
    }

    // Emit aie.flow operations
    OpBuilder::InsertionGuard guard(builder);

    // Insert flows after all tiles but before other operations
    // Find a good insertion point - after the last tile
    Operation *lastTile = nullptr;
    for (auto &[tileId, tile] : tiles) {
      if (!lastTile || lastTile->isBeforeInBlock(tile)) {
        lastTile = tile;
      }
    }

    if (lastTile) {
      builder.setInsertionPointAfter(lastTile);
    }

    // Emit each flow
    for (const auto &flow : flows) {
      auto srcTile = getOrCreateTile(flow.sourceCol, flow.sourceRow);
      auto dstTile = getOrCreateTile(flow.destCol, flow.destRow);

      AIE::FlowOp::create(
          builder,
          builder.getUnknownLoc(),
          srcTile,
          flow.sourceBundle,
          flow.sourceChannel,
          dstTile,
          flow.destBundle,
          flow.destChannel
      );
    }
  }

  /// Emit all collected locks as aie.lock operations
  void emitAllLocks() {
    for (const auto &[key, lock] : lockConfigs) {
      emitLockForTile(lock);
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

    // Create buffers for all BDs in this tile
    for (const auto &bd : bds) {
      getOrCreateBuffer(bd);
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto memOp = builder.create<AIE::MemOp>(builder.getUnknownLoc(),
                                             builder.getIndexType(), tile);
    Block *memBlock = &memOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(memBlock);

    // Group BDs by DMA channel
    llvm::DenseMap<int, llvm::SmallVector<const ParsedBDConfig*>> bdsByChannel;
    for (const auto &bd : bds) {
      if (bd.dmaChannel >= 0) {
        bdsByChannel[bd.dmaChannel].push_back(&bd);
      }
    }

    // Track blocks for BD chains
    llvm::DenseMap<int, Block*> bdBlocks;  // bdIndex -> Block*
    Block *endBlock = nullptr;

    // Emit DMA channels in order: S2MM_0, S2MM_1, MM2S_0, MM2S_1
    for (int channelIdx = 0; channelIdx < 4; channelIdx++) {
      auto it = bdsByChannel.find(channelIdx);
      if (it == bdsByChannel.end() || it->second.empty()) {
        continue;
      }

      const auto &channelBDs = it->second;

      // Determine channel direction
      AIE::DMAChannelDir channelDir;
      int channelNum;
      if (channelIdx == 0) {  // S2MM_0
        channelDir = AIE::DMAChannelDir::S2MM;
        channelNum = 0;
      } else if (channelIdx == 1) {  // S2MM_1
        channelDir = AIE::DMAChannelDir::S2MM;
        channelNum = 1;
      } else if (channelIdx == 2) {  // MM2S_0
        channelDir = AIE::DMAChannelDir::MM2S;
        channelNum = 0;
      } else {  // MM2S_1
        channelDir = AIE::DMAChannelDir::MM2S;
        channelNum = 1;
      }

      // Create blocks for each BD in this channel
      for (const auto *bd : channelBDs) {
        Block *bdBlock = new Block();
        memBlock->getParent()->push_back(bdBlock);
        bdBlocks[bd->bdIndex] = bdBlock;
      }

      // Find the first BD (one with lowest index, or follow hardware convention)
      const ParsedBDConfig *firstBD = channelBDs[0];

      // Create end block if not already created
      if (!endBlock) {
        endBlock = new Block();
        memBlock->getParent()->push_back(endBlock);
      }

      // Emit dma_start operation
      Block *firstBDBlock = bdBlocks[firstBD->bdIndex];
      Block *nextChain = endBlock;  // Chain to end or next channel

      (void) AIE::DMAStartOp::create(
          builder,
          builder.getUnknownLoc(),
          channelDir,
          channelNum,
          0,  // repeat_count
          firstBDBlock,
          nextChain
      );

      // Now emit each BD block
      for (const auto *bd : channelBDs) {
        Block *bdBlock = bdBlocks[bd->bdIndex];
        builder.setInsertionPointToEnd(bdBlock);

        // Emit lock acquire
        emitLockAcquire(*bd);

        // Emit dma_bd operation
        auto buffer = getOrCreateBuffer(*bd);
        auto dimAttrs = buildDimensionAttrs(*bd);

        AIE::DMABDOp::create(
            builder,
            builder.getUnknownLoc(),
            buffer,
            0,  // offset
            bd->bufferLength,
            dimAttrs
        );

        // Emit lock release
        emitLockRelease(*bd);

        // Emit next_bd terminator
        Block *nextBlock = nullptr;
        if (bd->useNextBd) {
          auto nextIt = bdBlocks.find(bd->nextBd);
          if (nextIt != bdBlocks.end()) {
            nextBlock = nextIt->second;
          }
        }

        if (!nextBlock) {
          // If no valid next_bd, loop back to first BD
          nextBlock = bdBlocks[firstBD->bdIndex];
        }

        AIE::NextBDOp::create(
            builder,
            builder.getUnknownLoc(),
            nextBlock
        );
      }
    }

    // Emit end block
    if (!endBlock) {
      endBlock = new Block();
      memBlock->getParent()->push_back(endBlock);
    }
    builder.setInsertionPointToEnd(endBlock);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
  }

  void emitShimDmaOpForTile(TileID tileId, const llvm::SmallVector<ParsedBDConfig> &bds) {
    auto tile = getOrCreateTile(tileId.col, tileId.row);

    // Create external buffers for all BDs in this shim tile
    for (const auto &bd : bds) {
      getOrCreateExternalBuffer(bd);
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    auto shimDmaOp = builder.create<AIE::ShimDMAOp>(builder.getUnknownLoc(),
                                                     builder.getIndexType(), tile);
    Block *shimDmaBlock = &shimDmaOp.getBody().emplaceBlock();

    // TODO: Properly implement DMA channel reconstruction to emit aie.dma_start operations
    // For now, emit aie.end without DMA BD blocks to avoid unreachable blocks
    builder.setInsertionPointToEnd(shimDmaBlock);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
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
      AIE::ConnectOp::create(
          builder,
          builder.getUnknownLoc(),
          conn.sourceBundle,
          conn.sourceChannel,
          conn.destBundle,
          conn.destChannel
      );
    }

    // Terminate with aie.end
    AIE::EndOp::create(builder, builder.getUnknownLoc());
  }

  void emitLockForTile(const ParsedLockConfig &lock) {
    // Check if lock already exists (might have been created by BD lifting)
    auto lockKey = std::make_tuple(lock.column, lock.row, lock.lockId);
    if (locks.find(lockKey) != locks.end()) {
      // Already created, skip
      return;
    }

    // Create the lock with init value
    getOrCreateLock(lock.column, lock.row, lock.lockId, lock.initValue);
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

    AIE::UseLockOp::create(
        builder,
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

    AIE::UseLockOp::create(
        builder,
        builder.getUnknownLoc(),
        lock,
        AIE::LockAction::Release,
        std::abs(bd.lockRelValue)  // Use absolute value
    );
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

  // Lock storage
  std::map<LockAccumulator::LockKey, ParsedLockConfig> lockConfigs;

  // Track all tiles referenced by any register write (including shim tiles)
  llvm::DenseSet<TileID> referencedTiles;
};

//===----------------------------------------------------------------------===//
// DMA Channel Tracker - Tracks BD assignments to DMA channels
//===----------------------------------------------------------------------===//

class DMAChannelTracker {
public:
  // DMA Start_Queue register offsets (relative to tile base)
  static constexpr uint32_t kDMA_S2MM_0_Start_Queue = 0x1DE04;
  static constexpr uint32_t kDMA_S2MM_1_Start_Queue = 0x1DE0C;
  static constexpr uint32_t kDMA_MM2S_0_Start_Queue = 0x1DE14;
  static constexpr uint32_t kDMA_MM2S_1_Start_Queue = 0x1DE1C;

  // Channel indices
  enum Channel {
    S2MM_0 = 0,
    S2MM_1 = 1,
    MM2S_0 = 2,
    MM2S_1 = 3,
    INVALID = -1
  };

  /// Parse a write to a Start_Queue register and return channel assignment
  /// Returns the BD index and channel, or std::nullopt if not a Start_Queue write
  std::optional<std::pair<int, Channel>> parseStartQueue(uint32_t addr, uint32_t value) {
    // Get offset within tile
    uint32_t offset = addr & 0xFFFFF;

    // Determine which channel based on offset
    Channel channel = INVALID;
    if (offset == kDMA_S2MM_0_Start_Queue) {
      channel = S2MM_0;
    } else if (offset == kDMA_S2MM_1_Start_Queue) {
      channel = S2MM_1;
    } else if (offset == kDMA_MM2S_0_Start_Queue) {
      channel = MM2S_0;
    } else if (offset == kDMA_MM2S_1_Start_Queue) {
      channel = MM2S_1;
    } else {
      return std::nullopt;  // Not a Start_Queue register
    }

    // Extract BD index from value (bits [3:0])
    int bdIndex = value & 0xF;

    return std::make_pair(bdIndex, channel);
  }

  /// Record a BD-to-channel assignment
  void recordAssignment(int col, int row, int bdIndex, Channel channel) {
    BDKey key{col, row, bdIndex};
    bdChannelMap[key] = channel;
  }

  /// Get the channel assigned to a BD, returns INVALID if not assigned
  Channel getChannel(int col, int row, int bdIndex) const {
    BDKey key{col, row, bdIndex};
    auto it = bdChannelMap.find(key);
    if (it != bdChannelMap.end()) {
      return it->second;
    }
    return INVALID;
  }

  /// Check if this is a Start_Queue register address
  bool isStartQueueAddress(uint32_t addr) const {
    uint32_t offset = addr & 0xFFFFF;
    return offset == kDMA_S2MM_0_Start_Queue ||
           offset == kDMA_S2MM_1_Start_Queue ||
           offset == kDMA_MM2S_0_Start_Queue ||
           offset == kDMA_MM2S_1_Start_Queue;
  }

private:
  struct BDKey {
    int col, row, bdIndex;

    bool operator<(const BDKey &other) const {
      if (col != other.col) return col < other.col;
      if (row != other.row) return row < other.row;
      return bdIndex < other.bdIndex;
    }
  };

  std::map<BDKey, Channel> bdChannelMap;
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

  // Initialize lock semantic lifting utilities
  LockAddressParser lockParser(/*numMemTileRows=*/1);
  LockAccumulator lockAccum;

  // Initialize DMA channel tracker
  DMAChannelTracker dmaChannelTracker;

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

      // Track tile reference from this register write
      if (emitLifted && liftedEmitter) {
        int col, row;
        if (LiftedBDEmitter::extractTileCoordinates(addr, col, row)) {
          liftedEmitter->recordTileReference(col, row);
        }
      }

      // Check if this is a BD register write and accumulate
      auto completedBD = bdAccum.addWrite(addr, value, bdParser);

      // Check if this is a DMA Start_Queue write and track channel assignment
      auto channelAssignment = dmaChannelTracker.parseStartQueue(addr, value);
      if (channelAssignment.has_value()) {
        int col = (addr >> 20) & 0xFF;
        int row = ((addr >> 20) >> 8) & 0xFF;
        int bdIndex = channelAssignment->first;
        DMAChannelTracker::Channel channel = channelAssignment->second;
        dmaChannelTracker.recordAssignment(col, row, bdIndex, channel);
      }

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
          // Assign DMA channel to BD before recording
          auto &bd = *completedBD;
          auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
          bd.dmaChannel = static_cast<int>(channel);
          liftedEmitter->recordBD(bd);
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

      // Track tile reference from this register write
      if (emitLifted && liftedEmitter) {
        int col, row;
        if (LiftedBDEmitter::extractTileCoordinates(addr, col, row)) {
          liftedEmitter->recordTileReference(col, row);
        }
      }

      // Check if this is a lock register write and accumulate
      lockAccum.addMaskWrite(addr, value, mask, lockParser);

      // If lifted mode and this is a lock write, mark for suppression
      bool shouldEmitRaw = true;
      if (emitLifted && lockParser.isLockAddress(addr)) {
        shouldEmitRaw = false;  // Don't emit raw write for lock registers
        if (liftedEmitter) {
          liftedEmitter->markLifted(addr);
        }
      }

      // Emit raw write only if not suppressed
      if (shouldEmitRaw) {
        AIEX::NpuMaskWrite32Op::create(builder, loc, addr, value, mask,
                                       nullptr, nullptr, nullptr);
      }
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
      for (auto &bd : pendingBDs) {
        // Assign DMA channel to BD before recording
        auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
        bd.dmaChannel = static_cast<int>(channel);
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
  AIE::EndOp::create(builder, builder.getUnknownLoc());

  // Emit all lifted BDs, switchboxes, and locks
  if (emitLifted && liftedEmitter) {
    builder.setInsertionPointToStart(deviceBlock);

    // Get all accumulated locks and record them for emission
    auto allLocks = lockAccum.getAllLocks();
    for (const auto &[key, lock] : allLocks) {
      liftedEmitter->recordLock(lock);
    }

    // Emit standalone tiles first (tiles referenced but without BDs/switchboxes/locks)
    // This ensures all tiles (including shim tiles) are declared
    liftedEmitter->emitStandaloneTiles();

    liftedEmitter->emitAllLocks();
    liftedEmitter->emitAllBDs();
    liftedEmitter->emitAllFlows();
    liftedEmitter->emitAllSwitchboxes();
  }

  // Add terminator to device block
  builder.setInsertionPointToEnd(deviceBlock);
  AIE::EndOp::create(builder, builder.getUnknownLoc());

  return success();
}
#endif // HAVE_BOOTGEN

} // namespace

namespace xilinx {
namespace AIE {

/// Main entry point: translate xclbin binary to MLIR module.
LogicalResult AIETranslateFromXclbin(ModuleOp module, StringRef filename,
                                     bool emitLifted) {
#ifdef HAVE_BOOTGEN
  // Step 1: Extract PDI from xclbin
  std::vector<uint8_t> pdiData;
  if (failed(extractPDIFromXclbin(filename, pdiData))) {
    return module.emitError("Failed to extract PDI from xclbin");
  }

  // Step 2: Extract CDO from PDI
  std::vector<uint8_t> cdoData;
  if (failed(extractCDOFromPDI(pdiData.data(), pdiData.size(), cdoData))) {
    return module.emitError("Failed to extract CDO from PDI");
  }

  // Step 3: Decode CDO to commands
  std::vector<CdoCommand *> commands;
  if (failed(
          decodeCDOToCmds(cdoData.data(), cdoData.size(), commands))) {
    return module.emitError("Failed to decode CDO binary");
  }

  // Step 4: Emit MLIR operations (lifted or raw mode)
  if (failed(emitMLIRFromCDO(module, commands, emitLifted))) {
    return module.emitError("Failed to emit MLIR from CDO commands");
  }

  return success();
#else
  return module.emitError("CDO decoding not available - bootgen library was not built (OpenSSL required)");
#endif
}

} // namespace AIE
} // namespace xilinx
