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

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIEDMABDLifting.h"
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"
#include "aie/Dialect/AIE/Util/AIEFlowReconstruction.h"
#include "aie/Dialect/AIE/Util/AIELockLifting.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
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

    // Determine buffer size:
    // 1. First check if we have a tracked buffer length (from NPU instructions or CDO)
    // 2. Fall back to bd.bufferLength if available
    // 3. Default to 1 as a placeholder
    int64_t bufSize = 1;  // Default placeholder

    BDBufferKey key{bd.column, bd.row, bd.bdIndex};
    auto lengthIt = bdBufferLengths.find(key);
    if (lengthIt != bdBufferLengths.end()) {
      // Use tracked buffer length from NPU instructions
      bufSize = static_cast<int64_t>(lengthIt->second);
      llvm::errs() << "DEBUG: Using tracked buffer length for " << bd.column << ","
                   << bd.row << " bd[" << bd.bdIndex << "]: " << bufSize << "\n";
    } else if (bd.bufferLength != 0) {
      // Use buffer length from BD config
      bufSize = static_cast<int64_t>(bd.bufferLength);
      llvm::errs() << "DEBUG: Using BD config buffer length for " << bd.column << ","
                   << bd.row << " bd[" << bd.bdIndex << "]: " << bufSize << "\n";
    } else {
      llvm::errs() << "WARNING: No buffer length found for " << bd.column << ","
                   << bd.row << " bd[" << bd.bdIndex << "], using placeholder size 1\n";
    }

    auto tile = getOrCreateTile(bd.column, bd.row);
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

  /// Record shim mux connections for later emission
  void recordShimMuxConnections(int column, const std::vector<ShimMuxConnection> &conns) {
    if (conns.empty()) return;

    ParsedShimMuxConfig &config = shimMuxes[column];
    config.column = column;

    // Add or update connections - if the same stream is configured multiple times,
    // keep only the latest configuration
    for (const auto &newConn : conns) {
      // Check if we already have a connection for this stream
      bool found = false;
      for (auto &existingConn : config.connections) {
        if (existingConn.streamIndex == newConn.streamIndex &&
            existingConn.isInput == newConn.isInput) {
          // Update existing connection
          existingConn = newConn;
          found = true;
          break;
        }
      }
      if (!found) {
        // Add new connection
        config.connections.push_back(newConn);
      }
    }
  }

  /// Record a tile reference from a register address
  /// This ensures all tiles (including shim tiles) are emitted even if they
  /// don't have BDs, switchboxes, or locks configured via CDO
  void recordTileReference(int col, int row) {
    TileID id{col, row};
    referencedTiles.insert(id);
  }

  /// Record a buffer length for a specific BD (from NPU instruction parsing or CDO)
  /// This allows us to infer correct buffer sizes even when BD config is incomplete
  void recordBufferLength(int col, int row, int bdIndex, uint32_t length) {
    BDBufferKey key{col, row, bdIndex};
    bdBufferLengths[key] = length;
    llvm::errs() << "DEBUG: Recorded buffer length for BD " << col << "," << row
                 << " bd[" << bdIndex << "]: " << length << "\n";
  }

  /// Get the maximum column index used in the xclbin
  /// Returns -1 if no tiles were referenced
  int getMaxColumn() const {
    int maxCol = -1;

    // Check referenced tiles
    for (const auto &tileId : referencedTiles) {
      if (tileId.col > maxCol) {
        maxCol = tileId.col;
      }
    }

    // Also check tiles with BDs
    for (const auto &[tileId, _] : tileBDs) {
      if (tileId.col > maxCol) {
        maxCol = tileId.col;
      }
    }

    // Also check tiles with switchboxes
    for (const auto &[key, _] : switchboxes) {
      if (key.col > maxCol) {
        maxCol = key.col;
      }
    }

    return maxCol;
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

    // First pass: Create all buffers for all tiles
    // This ensures buffers are created before any mem/shim_dma ops
    for (const auto &[tileId, bds] : tileBDs) {
      if (!targetModel.isShimNOCorPLTile(tileId.col, tileId.row)) {
        // Only create buffers for non-shim tiles (shim tiles use external buffers)
        for (const auto &bd : bds) {
          getOrCreateBuffer(bd);
        }
      }
    }

    // Second pass: Create mem/shim_dma/memtile_dma ops
    for (const auto &[tileId, bds] : tileBDs) {
      if (targetModel.isShimNOCorPLTile(tileId.col, tileId.row)) {
        // Shim tiles use aie.shim_dma
        emitShimDmaOpForTile(tileId, bds);
      } else if (targetModel.isMemTile(tileId.col, tileId.row)) {
        // Memory tiles use aie.memtile_dma
        emitMemTileDmaOpForTile(tileId, bds);
      } else {
        // Core/compute tiles use aie.mem
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

  /// Emit all collected shim mux configurations as aie.shim_mux operations
  void emitAllShimMuxes() {
    for (const auto &[column, config] : shimMuxes) {
      emitShimMuxForTile(config);
    }
  }

  /// Reconstruct and emit aie.flow operations from switchbox connections
  /// NOTE: For NPU xclbins, switchbox routing configuration is not stored in the
  /// binary format (architectural limitation). This code is preserved for potential
  /// future use with other target architectures or enhanced binary formats.
  void emitAllFlows() {
    // Build flow reconstruction graph from all switchbox configs
    FlowReconstructionGraph flowGraph;
    for (const auto &[key, config] : switchboxes) {
      flowGraph.addSwitchboxConfig(config);
    }

    // Add shim mux configs to the flow graph
    // Shim mux connections represent DMA endpoints (flow sources/sinks)
    for (const auto &[column, config] : shimMuxes) {
      flowGraph.addShimMuxConfig(config);
    }

    // Reconstruct end-to-end flows
    auto flows = flowGraph.reconstructFlows();

    if (flows.empty()) {
      return;  // No flows to emit (expected for NPU xclbins)
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

    // Find the last operation associated with this tile (locks, buffers, etc.)
    // by scanning operations after the tile in the device block
    Operation *lastTileOp = tile;
    Block *deviceBlock = tile->getBlock();
    for (auto it = std::next(Block::iterator(tile)); it != deviceBlock->end(); ++it) {
      Operation *op = &*it;
      // Check if this operation is associated with our tile
      // (BufferOp, LockOp, or other tile-associated ops)
      if (auto bufOp = dyn_cast<AIE::BufferOp>(op)) {
        if (bufOp.getTile() == tile) {
          lastTileOp = op;
        }
      } else if (auto lockOp = dyn_cast<AIE::LockOp>(op)) {
        if (lockOp.getTile() == tile) {
          lastTileOp = op;
        }
      } else if (isa<AIE::TileOp>(op)) {
        // Stop when we hit another tile
        break;
      }
    }

    // Buffers should have been created already in emitAllBDs first pass
    // Just set insertion point after the last tile-associated operation

    OpBuilder::InsertionGuard guard(builder);
    // Set insertion point after the last buffer/lock
    builder.setInsertionPointAfter(lastTileOp);

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

    // If there are no channels with BDs, we may have BDs without channel assignments.
    // Check if we have any BDs at all to emit with inferred channel assignments
    if (bdsByChannel.empty()) {
      // Collect BDs without channel assignments
      llvm::SmallVector<const ParsedBDConfig*> unassignedBDs;
      for (const auto &bd : bds) {
        if (bd.dmaChannel < 0) {
          unassignedBDs.push_back(&bd);
        }
      }

      if (!unassignedBDs.empty()) {
        // Emit BDs with default channel assignment (S2MM_0 is common for input)
        // Channel assignment inference: BDs without explicit channel assignment are
        // assumed to be S2MM_0 (stream-to-memory, channel 0) which is a typical
        // pattern for memory tiles receiving data.
        llvm::errs() << "Warning: Emitting " << unassignedBDs.size()
                     << " BDs with inferred channel S2MM_0 for tile("
                     << tileId.col << "," << tileId.row << ")\n";

        // Assign to channel S2MM_0 (channelIdx = 0)
        bdsByChannel[0] = unassignedBDs;
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
      // Use a unique index for each BD in the vector, not bd->bdIndex,
      // because multiple BDs can have the same bdIndex (reconfiguration)
      for (size_t i = 0; i < channelBDs.size(); i++) {
        Block *bdBlock = new Block();
        memBlock->getParent()->push_back(bdBlock);
        bdBlocks[i] = bdBlock;
      }

      // Create end block if not already created
      if (!endBlock) {
        endBlock = new Block();
        memBlock->getParent()->push_back(endBlock);
      }

      // Emit dma_start operation
      Block *firstBDBlock = bdBlocks[0];  // First BD in the vector
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
      for (size_t i = 0; i < channelBDs.size(); i++) {
        const auto *bd = channelBDs[i];
        Block *bdBlock = bdBlocks[i];
        builder.setInsertionPointToEnd(bdBlock);

        // Emit lock acquire
        emitLockAcquire(*bd);

        // Emit dma_bd operation
        auto buffer = getOrCreateBuffer(*bd);
        auto dimAttrs = buildDimensionAttrs(*bd);

        llvm::errs() << "[DEBUG emitMemOpForTile] Emitting aie.dma_bd for tile("
                     << bd->column << "," << bd->row << ") BD" << bd->bdIndex
                     << " with bufferLength=" << bd->bufferLength << "\n";

        if (dimAttrs) {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,  // offset
              bd->bufferLength,
              dimAttrs
          );
        } else {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,  // offset
              bd->bufferLength
          );
        }

        // Emit lock release
        emitLockRelease(*bd);

        // Emit next_bd terminator
        Block *nextBlock = nullptr;
        if (bd->useNextBd) {
          // Find the BD with the specified bdIndex in our vector
          for (size_t j = 0; j < channelBDs.size(); j++) {
            if (channelBDs[j]->bdIndex == bd->nextBd) {
              nextBlock = bdBlocks[j];
              break;
            }
          }
        }

        if (!nextBlock) {
          // If no valid next_bd, loop back to first BD (index 0)
          nextBlock = bdBlocks[0];
        }

        AIE::NextBDOp::create(
            builder,
            builder.getUnknownLoc(),
            nextBlock
        );
      }
    }

    // If there are no channels with BDs, just end the memBlock directly
    if (bdsByChannel.empty()) {
      builder.setInsertionPointToEnd(memBlock);
      AIE::EndOp::create(builder, builder.getUnknownLoc());
      return;
    }

    // Emit end block
    if (!endBlock) {
      endBlock = new Block();
      memBlock->getParent()->push_back(endBlock);
    }
    builder.setInsertionPointToEnd(endBlock);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
  }

  void emitMemTileDmaOpForTile(TileID tileId, const llvm::SmallVector<ParsedBDConfig> &bds) {
    auto tile = getOrCreateTile(tileId.col, tileId.row);

    // Find the last operation associated with this tile
    Operation *lastTileOp = tile;
    Block *deviceBlock = tile->getBlock();
    for (auto it = std::next(Block::iterator(tile)); it != deviceBlock->end(); ++it) {
      Operation *op = &*it;
      if (auto bufOp = dyn_cast<AIE::BufferOp>(op)) {
        if (bufOp.getTile() == tile) {
          lastTileOp = op;
        }
      } else if (auto lockOp = dyn_cast<AIE::LockOp>(op)) {
        if (lockOp.getTile() == tile) {
          lastTileOp = op;
        }
      } else if (isa<AIE::TileOp>(op)) {
        break;
      }
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(lastTileOp);

    // Create MemTileDMAOp instead of MemOp for memory tiles
    auto memTileDmaOp = builder.create<AIE::MemTileDMAOp>(builder.getUnknownLoc(),
                                                           builder.getIndexType(), tile);
    Block *dmaBlock = &memTileDmaOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(dmaBlock);

    // Group BDs by DMA channel (same logic as MemOp)
    llvm::DenseMap<int, llvm::SmallVector<const ParsedBDConfig*>> bdsByChannel;
    for (const auto &bd : bds) {
      if (bd.dmaChannel >= 0) {
        bdsByChannel[bd.dmaChannel].push_back(&bd);
      }
    }

    if (bdsByChannel.empty()) {
      llvm::SmallVector<const ParsedBDConfig*> unassignedBDs;
      for (const auto &bd : bds) {
        if (bd.dmaChannel < 0) {
          unassignedBDs.push_back(&bd);
        }
      }

      if (!unassignedBDs.empty()) {
        llvm::errs() << "Warning: Emitting " << unassignedBDs.size()
                     << " BDs with inferred channel S2MM_0 for memtile("
                     << tileId.col << "," << tileId.row << ")\n";
        bdsByChannel[0] = unassignedBDs;
      }
    }

    llvm::DenseMap<int, Block*> bdBlocks;
    Block *endBlock = nullptr;

    // Emit DMA channels
    for (int channelIdx = 0; channelIdx < 4; channelIdx++) {
      auto it = bdsByChannel.find(channelIdx);
      if (it == bdsByChannel.end() || it->second.empty()) {
        continue;
      }

      const auto &channelBDs = it->second;

      AIE::DMAChannelDir channelDir;
      int channelNum;
      if (channelIdx == 0) {
        channelDir = AIE::DMAChannelDir::S2MM;
        channelNum = 0;
      } else if (channelIdx == 1) {
        channelDir = AIE::DMAChannelDir::S2MM;
        channelNum = 1;
      } else if (channelIdx == 2) {
        channelDir = AIE::DMAChannelDir::MM2S;
        channelNum = 0;
      } else {
        channelDir = AIE::DMAChannelDir::MM2S;
        channelNum = 1;
      }

      // Use a unique index for each BD in the vector, not bd->bdIndex,
      // because multiple BDs can have the same bdIndex (reconfiguration)
      for (size_t i = 0; i < channelBDs.size(); i++) {
        Block *bdBlock = new Block();
        dmaBlock->getParent()->push_back(bdBlock);
        bdBlocks[i] = bdBlock;
      }

      if (!endBlock) {
        endBlock = new Block();
        dmaBlock->getParent()->push_back(endBlock);
      }

      Block *firstBDBlock = bdBlocks[0];  // First BD in the vector
      Block *nextChain = endBlock;

      (void) AIE::DMAStartOp::create(
          builder,
          builder.getUnknownLoc(),
          channelDir,
          channelNum,
          0,
          firstBDBlock,
          nextChain
      );

      for (size_t i = 0; i < channelBDs.size(); i++) {
        const auto *bd = channelBDs[i];
        Block *bdBlock = bdBlocks[i];
        builder.setInsertionPointToEnd(bdBlock);

        emitLockAcquire(*bd);

        auto buffer = getOrCreateBuffer(*bd);
        auto dimAttrs = buildDimensionAttrs(*bd);

        if (dimAttrs) {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,
              bd->bufferLength,
              dimAttrs
          );
        } else {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,
              bd->bufferLength
          );
        }

        emitLockRelease(*bd);

        Block *nextBlock = nullptr;
        if (bd->useNextBd) {
          // Find the BD with the specified bdIndex in our vector
          for (size_t j = 0; j < channelBDs.size(); j++) {
            if (channelBDs[j]->bdIndex == bd->nextBd) {
              nextBlock = bdBlocks[j];
              break;
            }
          }
        }

        if (!nextBlock) {
          // If no valid next_bd, loop back to first BD (index 0)
          nextBlock = bdBlocks[0];
        }

        AIE::NextBDOp::create(
            builder,
            builder.getUnknownLoc(),
            nextBlock
        );
      }
    }

    if (bdsByChannel.empty()) {
      builder.setInsertionPointToEnd(dmaBlock);
      AIE::EndOp::create(builder, builder.getUnknownLoc());
      return;
    }

    if (!endBlock) {
      endBlock = new Block();
      dmaBlock->getParent()->push_back(endBlock);
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
    builder.setInsertionPointToEnd(shimDmaBlock);

    // Group BDs by DMA channel
    llvm::DenseMap<int, llvm::SmallVector<const ParsedBDConfig*>> bdsByChannel;
    for (const auto &bd : bds) {
      if (bd.dmaChannel >= 0) {
        bdsByChannel[bd.dmaChannel].push_back(&bd);
      }
    }

    // If there are no channels with BDs, we may have BDs without channel assignments.
    // For shim tiles, BDs are often configured dynamically at runtime rather than statically.
    // Check if we have any BDs at all to emit with inferred channel assignments
    if (bdsByChannel.empty()) {
      // Collect BDs without channel assignments
      llvm::SmallVector<const ParsedBDConfig*> unassignedBDs;
      for (const auto &bd : bds) {
        if (bd.dmaChannel < 0) {
          unassignedBDs.push_back(&bd);
        }
      }

      if (!unassignedBDs.empty()) {
        // Emit BDs with default channel assignment (MM2S_0 is most common for shim output)
        // Channel assignment inference: BDs without explicit channel assignment are
        // assumed to be MM2S_0 (memory-to-stream, channel 0) which is the typical
        // pattern for shim tiles outputting data.
        llvm::errs() << "Warning: Emitting " << unassignedBDs.size()
                     << " BDs with inferred channel MM2S_0 for tile("
                     << tileId.col << "," << tileId.row << ")\n";

        // Assign to channel MM2S_0 (channelIdx = 2)
        bdsByChannel[2] = unassignedBDs;
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
      // Use a unique index for each BD in the vector, not bd->bdIndex,
      // because multiple BDs can have the same bdIndex (reconfiguration)
      for (size_t i = 0; i < channelBDs.size(); i++) {
        Block *bdBlock = new Block();
        shimDmaBlock->getParent()->push_back(bdBlock);
        bdBlocks[i] = bdBlock;
      }

      // Create end block if not already created
      if (!endBlock) {
        endBlock = new Block();
        shimDmaBlock->getParent()->push_back(endBlock);
      }

      // Emit dma_start operation
      Block *firstBDBlock = bdBlocks[0];  // First BD in the vector
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
      for (size_t i = 0; i < channelBDs.size(); i++) {
        const auto *bd = channelBDs[i];
        Block *bdBlock = bdBlocks[i];
        builder.setInsertionPointToEnd(bdBlock);

        // Emit lock acquire (if applicable)
        emitLockAcquire(*bd);

        // Emit dma_bd operation
        auto buffer = getOrCreateExternalBuffer(*bd);
        auto dimAttrs = buildDimensionAttrs(*bd);

        if (dimAttrs) {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,  // offset
              bd->bufferLength,
              dimAttrs
          );
        } else {
          AIE::DMABDOp::create(
              builder,
              builder.getUnknownLoc(),
              buffer,
              0,  // offset
              bd->bufferLength
          );
        }

        // Emit lock release (if applicable)
        emitLockRelease(*bd);

        // Emit next_bd terminator
        Block *nextBlock = nullptr;
        if (bd->useNextBd) {
          // Find the BD with the specified bdIndex in our vector
          for (size_t j = 0; j < channelBDs.size(); j++) {
            if (channelBDs[j]->bdIndex == bd->nextBd) {
              nextBlock = bdBlocks[j];
              break;
            }
          }
        }

        if (!nextBlock) {
          // If no valid next_bd, loop back to first BD (index 0)
          nextBlock = bdBlocks[0];
        }

        AIE::NextBDOp::create(
            builder,
            builder.getUnknownLoc(),
            nextBlock
        );
      }
    }

    // If no BDs were emitted, add an end terminator to shimDmaBlock
    if (bdsByChannel.empty()) {
      builder.setInsertionPointToEnd(shimDmaBlock);
      AIE::EndOp::create(builder, builder.getUnknownLoc());
      return;
    }

    // Emit end block
    if (!endBlock) {
      endBlock = new Block();
      shimDmaBlock->getParent()->push_back(endBlock);
    }
    builder.setInsertionPointToEnd(endBlock);
    AIE::EndOp::create(builder, builder.getUnknownLoc());
  }


  void emitSwitchboxForTile(const ParsedSwitchboxConfig &config) {
    if (!config.hasConnections()) {
      return;
    }

    // Note: Shim tiles (row 0) have BOTH aie.shim_mux AND aie.switchbox
    // - shim_mux: handles local DMA/NOC/PL connections
    // - switchbox: handles routing to/from neighboring tiles
    // Do not skip shim tiles here!

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

    // Emit all connections that pass MLIR validation
    // The hardware may support connections that the MLIR dialect rejects
    // due to conservative verifier constraints
    for (const auto &conn : config.connections) {
      // For memtiles, apply additional validation constraints
      if (config.tileType == TileType::MemoryTile) {
        // Constraint 1: South/North -> South/North requires matching channels
        bool srcIsDirectional = (conn.sourceBundle == WireBundle::South ||
                                 conn.sourceBundle == WireBundle::North);
        bool dstIsDirectional = (conn.destBundle == WireBundle::South ||
                                 conn.destBundle == WireBundle::North);
        if (srcIsDirectional && dstIsDirectional &&
            conn.sourceChannel != conn.destChannel) {
          // This is a valid hardware configuration but MLIR verifier rejects it
          // TODO: Fix the MLIR verifier to match hardware capabilities
          continue;
        }

        // Constraint 2: North source channels must be < 4 per MLIR verifier
        // (hardware has 6, but MLIR only supports 4)
        if (conn.sourceBundle == WireBundle::North && conn.sourceChannel >= 4) {
          continue;
        }
        // Similar for South source: verifier allows 6, check hardware limit
        // South seems to be 6 which matches, so no constraint needed
      }

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

  void emitShimMuxForTile(const ParsedShimMuxConfig &config) {
    if (!config.hasConnections()) {
      return;
    }

    // Shim tiles are always at row 0
    auto tile = getOrCreateTile(config.column, 0);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(tile);

    // Create aie.shim_mux operation
    auto shimMuxOp = builder.create<AIE::ShimMuxOp>(
        builder.getUnknownLoc(),
        builder.getIndexType(),
        tile
    );

    Block *shimMuxBlock = &shimMuxOp.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(shimMuxBlock);

    // Emit connections for each configured stream
    // The shim mux connects between local resources (DMA/PL) and the tile array (North)
    for (const auto &conn : config.connections) {
      WireBundle srcBundle, dstBundle;
      int srcChannel, dstChannel;

      // Map ShimMuxSource to WireBundle
      WireBundle localBundle;
      switch (conn.source) {
        case ShimMuxSource::PL:
          localBundle = WireBundle::South;  // PL connects via South
          break;
        case ShimMuxSource::DMA:
          localBundle = WireBundle::DMA;
          break;
        case ShimMuxSource::NOC:
          // NoC uses special addressing - for now skip
          continue;
        default:
          continue;  // Skip invalid sources
      }

      if (conn.isInput) {
        // Mux: Input stream receives from local resource (DMA/PL) and sends North
        // Example: South3 <- DMA means DMA:0 -> North:3
        srcBundle = localBundle;
        srcChannel = 0;  // DMA/PL channel 0
        dstBundle = WireBundle::North;
        dstChannel = conn.streamIndex;
      } else {
        // Demux: Output stream receives from North and sends to local resource
        // Example: South2 -> DMA means North:2 -> DMA:0
        srcBundle = WireBundle::North;
        srcChannel = conn.streamIndex;
        dstBundle = localBundle;
        dstChannel = 0;  // DMA/PL channel 0
      }

      AIE::ConnectOp::create(
          builder,
          builder.getUnknownLoc(),
          srcBundle,
          srcChannel,
          dstBundle,
          dstChannel
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


  /// Key for tracking buffer lengths per BD
  struct BDBufferKey {
    int col, row, bdIndex;

    bool operator==(const BDBufferKey &other) const {
      return col == other.col && row == other.row && bdIndex == other.bdIndex;
    }
  };

  struct BDBufferKeyHash {
    std::size_t operator()(const BDBufferKey &key) const {
      return std::hash<int>()(key.col) ^ (std::hash<int>()(key.row) << 1) ^
             (std::hash<int>()(key.bdIndex) << 2);
    }
  };

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

  // Shim mux storage (indexed by column)
  std::map<int, ParsedShimMuxConfig> shimMuxes;

  // Lock storage
  std::map<LockAccumulator::LockKey, ParsedLockConfig> lockConfigs;

  // Track all tiles referenced by any register write (including shim tiles)
  llvm::DenseSet<TileID> referencedTiles;

  // Track buffer lengths extracted from NPU instructions or CDO (col, row, bdIndex) -> length
  std::unordered_map<BDBufferKey, uint32_t, BDBufferKeyHash> bdBufferLengths;
};

//===----------------------------------------------------------------------===//
// Core Program Memory Extractor - Extracts ELF data from CDO memory writes
//===----------------------------------------------------------------------===//

/// Helper class to extract and save core program memory from CDO writes.
/// Program memory writes are identified by their address range and collected
/// per core, then saved as ELF files for recompilation.
class CoreProgramExtractor {
public:
  /// Program memory offset within each core's address space
  static constexpr uint32_t kProgramMemoryOffset = 0x20000;

  /// Program memory size (16KB per the register spec)
  static constexpr uint32_t kProgramMemorySize = 0x4000;  // 16KB

  /// Structure to hold program memory data for a core
  struct CoreProgram {
    int col;
    int row;
    // Map from offset (relative to program memory base) to 32-bit value
    std::map<uint32_t, uint32_t> memory;

    CoreProgram(int c, int r) : col(c), row(r) {}
  };

  /// Check if an address is a program memory write
  static bool isProgramMemoryAddress(uint32_t addr, int &col, int &row, uint32_t &offset) {
    // Extract tile coordinates (same as LiftedBDEmitter::extractTileCoordinates)
    constexpr uint32_t kTileAddrShift = 20;  // 0x100000 per tile
    uint32_t tileOffset = (addr >> kTileAddrShift) & 0xFFF;

    col = tileOffset / 32;
    int rowPart = tileOffset % 32;
    row = rowPart;

    // Get the offset within the tile's address space
    uint32_t tileLocalOffset = addr & 0xFFFFF;

    // Check if this is within program memory range
    if (tileLocalOffset >= kProgramMemoryOffset &&
        tileLocalOffset < kProgramMemoryOffset + kProgramMemorySize) {
      offset = tileLocalOffset - kProgramMemoryOffset;

      // Only cores (rows 2+) have program memory we care about
      // Row 0 is shim, row 1 is memory tile
      return (row >= 2 && col < 64 && row < 32);
    }

    return false;
  }

  /// Record a program memory write
  void recordWrite(uint32_t addr, uint32_t value) {
    int col, row;
    uint32_t offset;

    if (isProgramMemoryAddress(addr, col, row, offset)) {
      TileID id{col, row};

      auto it = corePrograms.find(id);
      if (it == corePrograms.end()) {
        it = corePrograms.emplace(id, CoreProgram(col, row)).first;
      }

      it->second.memory[offset] = value;
    }
  }

  /// Save all collected core programs as ELF files
  /// Returns the output directory where files were saved
  std::string saveELFFiles(llvm::StringRef outputDir = ".") const {
    for (const auto &[tileId, program] : corePrograms) {
      std::string filename = llvm::formatv("{0}/core_{1}_{2}.elf",
                                           outputDir, program.col, program.row);

      std::error_code EC;
      llvm::raw_fd_ostream file(filename, EC);

      if (EC) {
        llvm::errs() << "Failed to create ELF file " << filename << ": "
                     << EC.message() << "\n";
        continue;
      }

      // Write program memory as raw binary
      // The memory map is sorted by offset, so we write sequentially
      // Fill gaps with zeros to maintain correct offsets
      uint32_t currentOffset = 0;

      for (const auto &[offset, value] : program.memory) {
        // Fill gap with zeros if needed
        while (currentOffset < offset) {
          uint32_t zero = 0;
          file.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
          currentOffset += sizeof(uint32_t);
        }

        // Write the value (in little-endian format, matching the target)
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        currentOffset += sizeof(uint32_t);
      }

      file.close();
    }

    return outputDir.str();
  }

  /// Get all cores that have program memory
  std::vector<TileID> getCoresWithPrograms() const {
    std::vector<TileID> cores;
    for (const auto &[tileId, _] : corePrograms) {
      cores.push_back(tileId);
    }
    return cores;
  }

  /// Check if a core has program memory
  bool hasProgram(int col, int row) const {
    return corePrograms.count(TileID{col, row}) > 0;
  }

private:
  std::map<TileID, CoreProgram> corePrograms;
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

/// Extract AIE_METADATA section from xclbin binary data.
/// This section contains JSON metadata that may include routing/switchbox information.
LogicalResult extractAIEMetadata(StringRef xclbinData,
                                 std::string &metadataJson) {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(xclbinData.data());
  size_t size = xclbinData.size();

  // Parse xclbin header
  if (size < sizeof(axlf)) {
    llvm::errs() << "File too small to be a valid xclbin\n";
    return failure();
  }

  const axlf *header = reinterpret_cast<const axlf *>(data);

  // Get sections array
  size_t headerSize = sizeof(axlf) - sizeof(axlf_section_header);
  const axlf_section_header *sections =
      reinterpret_cast<const axlf_section_header *>(data + headerSize);

  uint32_t numSections = header->m_header.m_numSections;

  // Find AIE_METADATA section (kind 25)
  for (uint32_t i = 0; i < numSections; i++) {
    if (sections[i].m_sectionKind == AIE_METADATA) {
      uint64_t offset = sections[i].m_sectionOffset;
      uint64_t len = sections[i].m_sectionSize;

      if (offset + len > size) {
        llvm::errs() << "AIE_METADATA section offset/size extends beyond file\n";
        return failure();
      }

      // Copy metadata as string (assuming it's JSON text)
      metadataJson.assign(reinterpret_cast<const char *>(data + offset), len);
      return success();
    }
  }

  // Try EMBEDDED_METADATA section (kind 2) as fallback
  for (uint32_t i = 0; i < numSections; i++) {
    if (sections[i].m_sectionKind == EMBEDDED_METADATA) {
      uint64_t offset = sections[i].m_sectionOffset;
      uint64_t len = sections[i].m_sectionSize;

      if (offset + len > size) {
        llvm::errs() << "EMBEDDED_METADATA section offset/size extends beyond file\n";
        return failure();
      }

      // Copy metadata as string
      metadataJson.assign(reinterpret_cast<const char *>(data + offset), len);
      return success();
    }
  }

  llvm::errs() << "No AIE_METADATA or EMBEDDED_METADATA section found in xclbin\n";
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

/// Direct CDO parsing for write32 commands.
/// This is a fallback that scans the raw CDO binary for write32 command patterns
/// that the bootgen decoder may miss due to compressed command handling.
/// Pattern: 03 01 XX XX followed by 4-byte address and 4-byte value
///
/// The bootgen decoder returns disabled (value=0) versions of switchbox writes.
/// This function scans for enabled versions and either replaces or adds them.
void scanForAdditionalWrite32Commands(const uint8_t *data, size_t len,
                                       std::vector<CdoCommand *> &commands,
                                       llvm::DenseSet<uint32_t> &seenAddresses) {
  // Build map from address to command index for replacement
  llvm::DenseMap<uint32_t, size_t> addrToIndex;
  for (size_t idx = 0; idx < commands.size(); idx++) {
    CdoCommand *cmd = commands[idx];
    if (cmd->type == CdoCmdWrite) {
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      seenAddresses.insert(addr);
      addrToIndex[addr] = idx;
    }
  }

  int addedCount = 0;
  int replacedCount = 0;

  // Scan for write32 patterns: 03 01 XX XX ADDR[4] VALUE[4]
  for (size_t i = 0; i + 12 <= len; i += 4) {
    if (data[i] == 0x03 && data[i+1] == 0x01) {
      // Potential write32 command
      uint32_t addr = *reinterpret_cast<const uint32_t*>(&data[i+4]);
      uint32_t value = *reinterpret_cast<const uint32_t*>(&data[i+8]);

      // Sanity checks
      uint32_t col = (addr >> 20) & 0xFF;
      uint32_t row = ((addr >> 20) >> 8) & 0xFF;
      uint32_t offset = addr & 0xFFFFF;

      if (col < 64 && row < 64) {
        // Check if this is a switchbox address with enable bit set
        bool isSwitchbox = (offset >= 0x3F000 && offset <= 0x3F05C) ||
                           (offset >= 0xB0000 && offset <= 0xB0040);
        bool isEnabled = (value & 0x80000000) != 0;

        if (isSwitchbox && isEnabled) {
          auto it = addrToIndex.find(addr);
          if (it != addrToIndex.end()) {
            // Replace existing command's value with the enabled version
            commands[it->second]->value = value;
            replacedCount++;
          } else {
            // Add new command
            CdoCommand *cmd = new CdoCommand();
            cmd->type = CdoCmdWrite;
            cmd->dstaddr = addr;
            cmd->value = value;
            commands.push_back(cmd);
            seenAddresses.insert(addr);
            addrToIndex[addr] = commands.size() - 1;
            addedCount++;
          }
        }
      }
    }
  }

  // Debug output (disabled):
  // if (addedCount > 0 || replacedCount > 0) {
  //   llvm::errs() << "Switchbox fix: added " << addedCount
  //                << ", replaced " << replacedCount << " writes\n";
  // }
}

#ifdef HAVE_BOOTGEN

/// Manually extract program memory blockwrite commands from raw CDO.
/// Bootgen's decoder doesn't properly decode blockwrite commands to core program memory.
/// This function scans the raw CDO for SET_BLOCK commands (0xXXXX0104) and extracts
/// the program memory data directly.
void extractProgramMemoryFromCDO(const uint8_t *data, size_t len,
                                  CoreProgramExtractor &extractor) {
  // CDO header: skip it to get to commands
  if (len < 32) return;

  const uint32_t *words = reinterpret_cast<const uint32_t *>(data);

  // Header structure: [NumWords] [IdentWord="CDO\0"] [Version] [Length] [Checksum]...
  // Commands start after header (NumWords+4 words total in header)
  size_t headerWords = 4 + words[0];
  if (headerWords * 4 >= len) return;

  const uint8_t *cmdData = data + (headerWords * 4);
  size_t cmdLen = len - (headerWords * 4);

  // Find all blockwrite commands to program memory
  // Pattern: [prev_data] [address] [0xXXXX0104] [data...]
  llvm::SmallVector<std::tuple<uint32_t, uint32_t, uint32_t, size_t>, 32> blockwrites;

  for (size_t i = 0; i + 12 < cmdLen; i += 4) {
    uint32_t word = *reinterpret_cast<const uint32_t *>(cmdData + i);

    // Check for SET_BLOCK command (ID 0x0104 in lower 16 bits)
    if ((word & 0xFFFF) == 0x0104 && i >= 4) {
      // Address is 4 bytes before the command word
      uint32_t addr = *reinterpret_cast<const uint32_t *>(cmdData + i - 4);

      // Decode address to check if it's program memory
      uint32_t tileOffset = (addr >> 20) & 0xFFF;
      uint32_t localOffset = addr & 0xFFFFF;

      // Program memory starts at offset 0x20000 within each tile
      if (localOffset == 0x20000 && tileOffset < 128) {
        uint32_t col = tileOffset / 32;
        uint32_t row = tileOffset % 32;

        // Only compute tiles have program memory we care about (row >= 2)
        if (row >= 2 && row < 32) {
          blockwrites.push_back(std::make_tuple(addr, col, row, i));
        }
      }
    }
  }

  // Extract program memory data from each blockwrite
  for (size_t idx = 0; idx < blockwrites.size(); idx++) {
    auto [addr, col, row, cmdPos] = blockwrites[idx];

    // Data starts immediately after the command word
    size_t dataStart = cmdPos + 4;

    // Data ends at the next blockwrite's address field (8 bytes before next command)
    size_t dataEnd;
    if (idx + 1 < blockwrites.size()) {
      size_t nextCmdPos = std::get<3>(blockwrites[idx + 1]);
      dataEnd = nextCmdPos - 8;
    } else {
      // Last blockwrite - limit to reasonable program memory size
      dataEnd = std::min(dataStart + 16384, cmdLen);  // 16KB max
    }

    // Write program memory data word by word
    size_t dataSize = dataEnd - dataStart;
    const uint8_t *programData = cmdData + dataStart;

    for (size_t offset = 0; offset < dataSize && offset < 16384; offset += 4) {
      if (dataStart + offset + 4 <= cmdLen) {
        uint32_t value = *reinterpret_cast<const uint32_t *>(programData + offset);
        uint32_t writeAddr = addr + offset;
        extractor.recordWrite(writeAddr, value);
      }
    }
  }
}

/// Manually extract BD configuration from raw CDO SetBlock commands.
/// Bootgen's decoder doesn't properly decode SetBlock commands (0x0104) to BD regions.
/// This function scans the raw CDO for SET_BLOCK commands and feeds BD register writes
/// into the BDAccumulator for semantic lifting.
void extractBDFromCDO(const uint8_t *data, size_t len,
                      BDAddressParser &bdParser, BDAccumulator &bdAccum) {
  // CDO header: skip it to get to commands
  if (len < 32) return;

  const uint32_t *words = reinterpret_cast<const uint32_t *>(data);

  // Header structure: [NumWords] [IdentWord="CDO\0"] [Version] [Length] [Checksum]...
  // Commands start after header (NumWords+4 words total in header)
  size_t headerWords = 4 + words[0];
  if (headerWords * 4 >= len) return;

  const uint8_t *cmdData = data + (headerWords * 4);
  size_t cmdLen = len - (headerWords * 4);

  llvm::errs() << "[DEBUG extractBDFromCDO] Scanning CDO binary (size=" << len
               << " bytes, cmd region=" << cmdLen << " bytes) for SetBlock commands to BD regions...\n";
  llvm::errs() << "[DEBUG extractBDFromCDO] First 32 bytes of cmd region:";
  for (size_t i = 0; i < 32 && i < cmdLen; i++) {
    if (i % 4 == 0) llvm::errs() << "\n  " << llvm::format("%04zx", i) << ":";
    llvm::errs() << " " << llvm::format("%02X", cmdData[i]);
  }
  llvm::errs() << "\n";

  // BD address ranges (from memory)
  // Compute tile BDs: 0x1D000 - 0x1D200 (within tile local offset)
  // MemTile BDs: 0xA0000 - 0xA0600 (within tile local offset)

  int setBlockCount = 0;
  int bdSetBlockCount = 0;
  int allCmdCount = 0;

  // Find all blockwrite commands to BD regions
  // Pattern: [prev_data] [address] [0xXXXX0104] [data...]
  for (size_t i = 0; i + 12 < cmdLen; i += 4) {
    uint32_t word = *reinterpret_cast<const uint32_t *>(cmdData + i);
    uint16_t cmdId = word & 0xFFFF;

    // Log all command types we see
    if (cmdId != 0 && allCmdCount < 20) {
      llvm::errs() << "[DEBUG extractBDFromCDO] Offset " << llvm::format("%04zx", i)
                   << ": cmdId=0x" << llvm::format("%04X", cmdId)
                   << " word=0x" << llvm::format("%08X", word) << "\n";
      allCmdCount++;
    }

    // Check for SET_BLOCK command (ID 0x0104 in lower 16 bits)
    if ((word & 0xFFFF) == 0x0104 && i >= 4) {
      setBlockCount++;
      // Address is 4 bytes before the command word
      uint32_t addr = *reinterpret_cast<const uint32_t *>(cmdData + i - 4);

      // Check if this address is in a BD region
      if (bdParser.isBDAddress(addr)) {
        bdSetBlockCount++;
        // Data length is in upper 16 bits of command word (in 32-bit words)
        uint32_t dataWords = (word >> 16) & 0xFFFF;

        llvm::errs() << "[DEBUG extractBDFromCDO] Found SetBlock #" << bdSetBlockCount
                     << " to BD region: addr=0x" << llvm::format("%08X", addr)
                     << " dataWords=" << dataWords << "\n";

        // Data starts immediately after the command word
        size_t dataStart = i + 4;

        // Feed each word to bdAccum as a write operation
        for (uint32_t offset = 0; offset < dataWords && (dataStart + offset * 4 < cmdLen); offset++) {
          uint32_t value = *reinterpret_cast<const uint32_t *>(cmdData + dataStart + offset * 4);
          uint32_t writeAddr = addr + (offset * 4);

          // Feed to BD accumulator for semantic lifting
          auto completedBD = bdAccum.addWrite(writeAddr, value, bdParser);

          if (completedBD.has_value()) {
            auto &bd = *completedBD;
            llvm::errs() << "[DEBUG extractBDFromCDO] *** COMPLETED BD from CDO: tile("
                         << bd.column << "," << bd.row << ") BD" << bd.bdIndex
                         << " bufferLength=" << bd.bufferLength
                         << " baseAddress=0x" << llvm::format("%08X", bd.baseAddress) << "\n";
          }
        }

        // Skip past the data we just processed
        i += dataWords * 4;
      }
    }
  }

  llvm::errs() << "[DEBUG extractBDFromCDO] Summary: found " << setBlockCount
               << " total SetBlock commands, " << bdSetBlockCount << " targeting BD regions\n";
}

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

  // Workaround: The bootgen decoder may miss some write32 commands that appear
  // after compressed/DMA sections. Scan the raw CDO binary for additional
  // enabled switchbox configuration writes.
  llvm::DenseSet<uint32_t> seenAddresses;
  scanForAdditionalWrite32Commands(data, len, commands, seenAddresses);

  return success();
}

// Forward declarations
void extractBDsFromTransaction(ModuleOp txnModule, AIE::DeviceOp deviceOp,
                                BDAddressParser &bdParser, BDAccumulator &bdAccum,
                                DMAChannelTracker &dmaChannelTracker,
                                LiftedBDEmitter &emitter);

void liftNPUInstructions(AIE::RuntimeSequenceOp seqOp, AIE::DeviceOp deviceOp,
                         LiftedBDEmitter &emitter);

/// Determine device type from maximum column index
AIE::AIEDevice getDeviceFromMaxColumn(int maxCol) {
  // Default to 1 column if no columns detected
  if (maxCol < 0) {
    llvm::errs() << "Warning: No tile columns detected in xclbin, defaulting to npu1_1col\n";
    return AIE::AIEDevice::npu1_1col;
  }

  // Map max column index to device type
  // Column indices are 0-based, so maxCol=0 means 1 column
  if (maxCol == 0) {
    return AIE::AIEDevice::npu1_1col;
  } else if (maxCol == 1) {
    return AIE::AIEDevice::npu1_2col;
  } else if (maxCol == 2) {
    return AIE::AIEDevice::npu1_3col;
  } else {
    // For more than 3 columns, default to npu1 (generic NPU1 device)
    llvm::errs() << "Warning: Max column " << maxCol
                 << " exceeds npu1_3col, using generic npu1 device\n";
    return AIE::AIEDevice::npu1;
  }
}

/// Scan CDO commands to determine maximum column index
int scanForMaxColumn(llvm::ArrayRef<CdoCommand *> commands) {
  int maxCol = -1;

  for (CdoCommand *cmd : commands) {
    if (cmd->type == CdoCmdWrite || cmd->type == CdoCmdMaskWrite) {
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      int col, row;
      if (LiftedBDEmitter::extractTileCoordinates(addr, col, row)) {
        if (col > maxCol) {
          maxCol = col;
        }
      }
    } else if (cmd->type == CdoCmdSetBlock) {
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      int col, row;
      if (LiftedBDEmitter::extractTileCoordinates(addr, col, row)) {
        if (col > maxCol) {
          maxCol = col;
        }
      }
    }
  }

  return maxCol;
}

/// Check if a maskwrite is a core control operation (enable/disable/reset).
/// Core control register is at offset 0x32000 with:
///   Bit 0 (mask=1): Enable
///   Bit 1 (mask=2): Reset
/// These operations are boilerplate inserted by the compiler and can be omitted.
static bool isCoreControlOperation(uint32_t addr, uint32_t mask) {
  // Extract tile coordinates using AIE2 address format
  uint32_t tileOffset = (addr >> 20) & 0xFFF;
  int col = tileOffset / 32;
  int row = tileOffset % 32;
  uint32_t tileBase = (col * 32 + row) << 20;
  uint32_t offset = addr - tileBase;

  // Core_Control register at offset 0x32000
  // Only applies to compute tiles (row >= 2)
  if (offset == 0x32000 && row >= 2 && (mask == 1 || mask == 2)) {
    return true;
  }

  return false;
}

/// Check if a maskwrite is a DMA control operation (channel enable/reset).
/// Compute Tile DMA control registers are at:
///   0x1DE00: DMA_S2MM_0_Ctrl
///   0x1DE08: DMA_S2MM_1_Ctrl
///   0x1DE10: DMA_MM2S_0_Ctrl
///   0x1DE18: DMA_MM2S_1_Ctrl
/// MemTile DMA control registers are at:
///   0xA0600: DMA_S2MM_0_Ctrl, 0xA0608: DMA_S2MM_1_Ctrl
///   0xA0610: DMA_S2MM_2_Ctrl, 0xA0618: DMA_S2MM_3_Ctrl
///   0xA0620: DMA_S2MM_4_Ctrl, 0xA0628: DMA_S2MM_5_Ctrl
///   0xA0630: DMA_MM2S_0_Ctrl, 0xA0638: DMA_MM2S_1_Ctrl
///   0xA0640: DMA_MM2S_2_Ctrl, 0xA0648: DMA_MM2S_3_Ctrl
///   0xA0650: DMA_MM2S_4_Ctrl, 0xA0658: DMA_MM2S_5_Ctrl
/// These operations enable/reset DMA channels and are derivable from BD configuration.
static bool isDMAControlOperation(uint32_t addr, uint32_t mask, uint32_t value) {
  // Extract tile coordinates using AIE2 address format
  uint32_t tileOffset = (addr >> 20) & 0xFFF;
  int col = tileOffset / 32;
  int row = tileOffset % 32;
  uint32_t tileBase = (col * 32 + row) << 20;
  uint32_t offset = addr - tileBase;

  // Compute tile DMA control registers (row >= 2)
  if (row >= 2) {
    // DMA_S2MM_0_Ctrl, DMA_S2MM_1_Ctrl, DMA_MM2S_0_Ctrl, DMA_MM2S_1_Ctrl
    if (offset == 0x1DE00 || offset == 0x1DE08 ||
        offset == 0x1DE10 || offset == 0x1DE18) {
      // mask=0, value=1: Enable operation
      // mask=2, value=2: Reset operation
      // mask=2, value=0: Clear reset
      if ((mask == 0 && value == 1) || (mask == 2)) {
        return true;
      }
    }
  }

  // MemTile DMA control registers (row == 1)
  if (row == 1) {
    if (offset == 0xA0600 || offset == 0xA0608 ||  // S2MM 0-1
        offset == 0xA0610 || offset == 0xA0618 ||  // S2MM 2-3
        offset == 0xA0620 || offset == 0xA0628 ||  // S2MM 4-5
        offset == 0xA0630 || offset == 0xA0638 ||  // MM2S 0-1
        offset == 0xA0640 || offset == 0xA0648 ||  // MM2S 2-3
        offset == 0xA0650 || offset == 0xA0658) {  // MM2S 4-5
      // mask=0, value=1: Enable operation
      // mask=2: Reset operations
      if ((mask == 0 && value == 1) || (mask == 2)) {
        return true;
      }
    }
  }

  return false;
}

/// Check if a register write is an initialization/reset operation that can be
/// omitted from decompiled MLIR. These are writes with value=0 that clear
/// registers to a known state before configuration.
static bool isInitializationWrite(uint32_t addr, uint32_t value) {
  // Only consider writes with value=0 as initialization
  if (value != 0) {
    return false;
  }

  // Extract tile coordinates using AIE2 address format
  uint32_t tileOffset = (addr >> 20) & 0xFFF;
  int col = tileOffset / 32;
  int row = tileOffset % 32;
  uint32_t tileBase = (col * 32 + row) << 20;
  uint32_t offset = addr - tileBase;

  // DMA BD register ranges (these get cleared before configuration)
  // Core/MemTile: BD registers at 0x1D000-0x1E000 range
  if (offset >= 0x1D000 && offset < 0x1E000) {
    return true;  // DMA BD initialization
  }

  // MemTile DMA BD range: 0x1F000-0x20000
  if (offset >= 0x1F000 && offset < 0x20000) {
    return true;  // MemTile BD initialization
  }

  // Memory module base (offset 0x20000) - data memory base address
  if (offset == 0x20000) {
    return true;  // Memory module initialization
  }

  // MemTile address ranges (row 1, memtile addresses around 0xC0000-0xE0000)
  if (offset >= 0xC0000 && offset < 0xE0000) {
    return true;  // MemTile region initialization
  }

  // MemTile DMA BD region (larger range for memtiles)
  if (offset >= 0xA0000 && offset < 0xB0000) {
    return true;  // MemTile DMA BD initialization
  }

  // DMA control/status registers: 0x1DE00-0x1E000 (Start_Queue, Control, Status)
  if (offset >= 0x1DE00 && offset < 0x1E000) {
    return true;  // DMA control initialization
  }

  // DMA registers in core tile: around 0x300-0x500 range
  if (offset >= 0x300 && offset < 0x600) {
    return true;  // Core DMA initialization
  }

  // Core module base register
  if (offset == 0x0) {
    return true;  // Core base initialization
  }

  // Shim tile initialization (row=0)
  if (row == 0) {
    // Shim DMA BD range: 0x14000-0x15000
    if (offset >= 0x14000 && offset < 0x15000) {
      return true;  // Shim BD initialization
    }
    // Shim configuration registers (various ranges)
    if (offset < 0x100000) {
      return true;  // Shim initialization
    }
  }

  return false;
}

/// Emit MLIR operations from decoded CDO commands.
/// Creates aie.device, runtime_sequence, and MLIR operations for register writes.
LogicalResult emitMLIRFromCDO(ModuleOp module,
                              llvm::ArrayRef<CdoCommand *> commands,
                              bool emitLifted = false,
                              std::optional<ModuleOp> txnModule = std::nullopt,
                              const uint8_t *rawCDO = nullptr,
                              size_t rawCDOSize = 0) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());

  // Pre-scan CDO commands to determine device column count
  int maxCol = scanForMaxColumn(commands);
  AIE::AIEDevice deviceType = getDeviceFromMaxColumn(maxCol);


  // Create aie.device with detected device type
  auto deviceOp = AIE::DeviceOp::create(
      builder, builder.getUnknownLoc(),
      AIE::AIEDeviceAttr::get(builder.getContext(), deviceType),
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

  // Initialize shim mux semantic lifting utilities
  ShimMuxAddressParser shimMuxParser;

  // Initialize lock semantic lifting utilities
  LockAddressParser lockParser(/*numMemTileRows=*/1);
  LockAccumulator lockAccum;

  // Initialize DMA channel tracker
  DMAChannelTracker dmaChannelTracker;

  // Initialize core program extractor (for ELF extraction)
  CoreProgramExtractor coreProgExtractor;

  // Extract program memory from raw CDO (bootgen decoder doesn't handle this)
  if (rawCDO && rawCDOSize > 0) {
    extractProgramMemoryFromCDO(rawCDO, rawCDOSize, coreProgExtractor);
  }

  // Extract BD configuration from raw CDO (bootgen decoder doesn't decode SetBlock commands)
  if (rawCDO && rawCDOSize > 0) {
    extractBDFromCDO(rawCDO, rawCDOSize, bdParser, bdAccum);
  }

  // NEW FIX: Extract BD configuration from decoded CDO commands
  // The bootgen decoder successfully decodes CdoCmdWrite commands, which contain
  // BD register writes. Feed these to the BD accumulator for semantic lifting.
  llvm::errs() << "[DEBUG] Processing " << commands.size() << " decoded CDO commands for BD extraction...\n";
  int bdWriteCount = 0;
  int completedBDsFromCommands = 0;
  int setBlockCount = 0;

  // First check for SetBlock commands which might contain BD data
  for (CdoCommand *cmd : commands) {
    if (cmd->type == CdoCmdSetBlock) {
      setBlockCount++;
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);

      if (setBlockCount <= 5 && bdParser.isBDAddress(addr)) {
        llvm::errs() << "[DEBUG] SetBlock #" << setBlockCount
                     << ": addr=0x" << llvm::format("%08X", addr)
                     << " count=" << cmd->count << " words\n";
      }

      // Process SetBlock as a sequence of writes
      if (bdParser.isBDAddress(addr)) {
        uint32_t *dataPtr = reinterpret_cast<uint32_t *>(cmd->buf);
        for (uint32_t i = 0; i < cmd->count; i++) {
          uint32_t writeAddr = addr + (i * 4);
          uint32_t value = dataPtr[i];

          auto completedBD = bdAccum.addWrite(writeAddr, value, bdParser);
          if (completedBD.has_value()) {
            completedBDsFromCommands++;
            auto &bd = *completedBD;
            llvm::errs() << "[DEBUG] *** COMPLETED BD from SetBlock: tile("
                         << bd.column << "," << bd.row << ") BD" << bd.bdIndex
                         << " bufferLength=" << bd.bufferLength
                         << " baseAddress=0x" << llvm::format("%08X", bd.baseAddress) << "\n";
          }
        }
      }
    }
  }

  llvm::errs() << "[DEBUG] Found " << setBlockCount << " SetBlock commands, "
               << completedBDsFromCommands << " completed BDs from SetBlock\n";

  // Now process regular writes
  for (CdoCommand *cmd : commands) {
    if (cmd->type == CdoCmdWrite || cmd->type == CdoCmdMaskWrite) {
      uint32_t addr = static_cast<uint32_t>(cmd->dstaddr & 0xFFFFFFFF);
      uint32_t value = cmd->value;

      // Check if this write targets a BD register
      if (bdParser.isBDAddress(addr)) {
        bdWriteCount++;

        if (bdWriteCount <= 10) {  // Log first 10 BD writes for debugging
          auto addrInfo = bdParser.parse(addr);
          llvm::errs() << "[DEBUG] BD write #" << bdWriteCount
                       << ": addr=0x" << llvm::format("%08X", addr)
                       << " value=0x" << llvm::format("%08X", value)
                       << " -> tile(" << addrInfo.column << "," << addrInfo.row
                       << ") BD" << addrInfo.bdIndex << " reg" << addrInfo.regIndex << "\n";
        }

        // For mask writes, we need to apply the mask to get the actual value
        // The CDO mask write format: value = (oldValue & ~mask) | (newValue & mask)
        // For our purposes, we use the value as-is since we're reconstructing from scratch

        auto completedBD = bdAccum.addWrite(addr, value, bdParser);

        if (completedBD.has_value()) {
          completedBDsFromCommands++;
          auto &bd = *completedBD;
          llvm::errs() << "[DEBUG] *** COMPLETED BD from decoded CDO command: tile("
                       << bd.column << "," << bd.row << ") BD" << bd.bdIndex
                       << " bufferLength=" << bd.bufferLength
                       << " baseAddress=0x" << llvm::format("%08X", bd.baseAddress)
                       << " lockAcqId=" << (int)bd.lockAcquire.lockId
                       << " lockRelId=" << (int)bd.lockRelId << "\n";
        }
      }
    }
  }

  llvm::errs() << "[DEBUG] BD extraction from decoded commands: " << bdWriteCount
               << " BD register writes, " << completedBDsFromCommands << " completed BDs\n";

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

      // Check if this is a program memory write and record it for ELF extraction
      coreProgExtractor.recordWrite(addr, value);

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

      // Check if this is a shim mux register write
      auto shimMuxConns = shimMuxParser.parseShimMux(addr, value);

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

          // Record buffer length if available
          if (bd.bufferLength > 0) {
            liftedEmitter->recordBufferLength(bd.column, bd.row, bd.bdIndex, bd.bufferLength);
          }
        }
      }

      // If a switchbox connection was configured, record it for lifted emission
      if (switchConn.has_value()) {
        if (emitLifted && liftedEmitter) {
          liftedEmitter->recordSwitchboxConnection(*switchConn);
          shouldEmitRaw = false;  // Don't emit raw write for switchbox registers
          liftedEmitter->markLifted(addr);
        }
      }

      // If shim mux connections were configured, record them for lifted emission
      if (!shimMuxConns.empty()) {
        if (emitLifted && liftedEmitter) {
          int col = ShimMuxAddressParser::getColumn(addr);
          liftedEmitter->recordShimMuxConnections(col, shimMuxConns);
          shouldEmitRaw = false;  // Don't emit raw write for shim mux registers
          liftedEmitter->markLifted(addr);
        }
      }

      // If lifted mode and this is an initialization write (value=0), suppress it
      if (emitLifted && isInitializationWrite(addr, value)) {
        shouldEmitRaw = false;  // Don't emit initialization writes
        if (liftedEmitter) {
          liftedEmitter->markLifted(addr);
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

      // Check if this is a switchbox register write and accumulate
      // Note: maskwrite32 is used for switchbox configuration in some cases
      auto switchConn = switchAccum.addMasterWrite(addr, value, switchParser);

      // Check if this is a shim mux register write
      auto shimMuxConns = shimMuxParser.parseShimMux(addr, value, mask);

      // If lifted mode and this is a lock write, mark for suppression
      bool shouldEmitRaw = true;
      if (emitLifted && lockParser.isLockAddress(addr)) {
        shouldEmitRaw = false;  // Don't emit raw write for lock registers
        if (liftedEmitter) {
          liftedEmitter->markLifted(addr);
        }
      }

      // If a switchbox connection was configured, record it for lifted emission
      if (switchConn.has_value()) {
        if (emitLifted && liftedEmitter) {
          liftedEmitter->recordSwitchboxConnection(*switchConn);
          shouldEmitRaw = false;  // Don't emit raw write for switchbox registers
          liftedEmitter->markLifted(addr);
        }
      }

      // If shim mux connections were configured, record them for lifted emission
      if (!shimMuxConns.empty()) {
        if (emitLifted && liftedEmitter) {
          int col = ShimMuxAddressParser::getColumn(addr);
          liftedEmitter->recordShimMuxConnections(col, shimMuxConns);
          shouldEmitRaw = false;  // Don't emit raw write for shim mux registers
          liftedEmitter->markLifted(addr);
        }
      }

      // If lifted mode and this is a core control operation, suppress it
      if (emitLifted && isCoreControlOperation(addr, mask)) {
        shouldEmitRaw = false;  // Don't emit core control boilerplate
        if (liftedEmitter) {
          liftedEmitter->markLifted(addr);
        }
      }

      // If lifted mode and this is a DMA control operation, suppress it
      if (emitLifted && isDMAControlOperation(addr, mask, value)) {
        shouldEmitRaw = false;  // Don't emit DMA control boilerplate
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

      // Check if this blockwrite is to program memory and record it for ELF extraction
      uint32_t *dataPtr = reinterpret_cast<uint32_t *>(cmd->buf);
      for (uint32_t j = 0; j < cmd->count; j++) {
        uint32_t writeAddr = addr + (j * 4);  // Each write is 4 bytes apart
        coreProgExtractor.recordWrite(writeAddr, dataPtr[j]);
      }

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

  // If transaction module provided, copy its runtime operations to the output
  if (txnModule.has_value()) {
    // The transaction module has operations directly in the DeviceOp body
    // (not wrapped in a RuntimeSequenceOp), so walk the DeviceOp
    txnModule->walk([&](AIE::DeviceOp txnDeviceOp) {
      // Use IRMapping to track SSA value mappings when cloning
      IRMapping mapper;

      // First, copy any global memrefs from the transaction module to the output device
      // These are needed for operations like NpuBlockWriteOp that reference them
      for (Operation &op : txnDeviceOp.getBody()->getOperations()) {
        if (isa<memref::GlobalOp>(&op)) {
          builder.setInsertionPointToStart(deviceBlock);
          builder.clone(op, mapper);
        }
      }

      // Then, walk through all operations in the transaction's device body
      // In lifted mode, we'll try to lift operations to high-level constructs
      // For now, we clone them as-is, but mark where lifting should happen
      for (Operation &op : txnDeviceOp.getBody()->getOperations()) {
        // Skip the terminator and globals (already copied above)
        if (isa<AIE::EndOp>(&op) || isa<memref::GlobalOp>(&op)) {
          continue;
        }

        // Clone the operation into the output runtime_sequence, using the mapper
        // to ensure SSA values are properly remapped
        // TODO: Add lifting logic here to convert raw operations to high-level ops
        builder.setInsertionPointToEnd(seqBlock);
        builder.clone(op, mapper);
      }
      return WalkResult::interrupt();  // Only process the first DeviceOp
    });
  }

  // Add terminator to runtime_sequence block
  builder.setInsertionPointToEnd(seqBlock);
  AIE::EndOp::create(builder, builder.getUnknownLoc());

  // If in lifted mode and we have a transaction module, try to lift NPU instructions
  if (emitLifted && txnModule.has_value() && liftedEmitter) {
    liftNPUInstructions(seqOp, deviceOp, *liftedEmitter);
  }

  // Emit all lifted BDs, switchboxes, and locks
  if (emitLifted && liftedEmitter) {
    builder.setInsertionPointToStart(deviceBlock);

    // Get all accumulated locks and record them for emission
    auto allLocks = lockAccum.getAllLocks();
    for (const auto &[key, lock] : allLocks) {
      liftedEmitter->recordLock(lock);
    }

    // If transaction module provided, extract BDs from it before emitting
    if (txnModule.has_value()) {
      extractBDsFromTransaction(*txnModule, deviceOp, bdParser, bdAccum,
                                dmaChannelTracker, *liftedEmitter);
    }

    // Emit standalone tiles first (tiles referenced but without BDs/switchboxes/locks)
    // This ensures all tiles (including shim tiles) are declared
    liftedEmitter->emitStandaloneTiles();

    liftedEmitter->emitAllLocks();
    liftedEmitter->emitAllBDs();
    // Don't emit aie.flow operations - switchbox configs provide complete routing
    // Emitting both flows and switchboxes causes compilation conflicts
    // liftedEmitter->emitAllFlows();
    liftedEmitter->emitAllSwitchboxes();
    liftedEmitter->emitAllShimMuxes();
  }

  // Extract and save ELF files from core program memory writes
  // This must be done BEFORE adding the device terminator
  auto coresWithPrograms = coreProgExtractor.getCoresWithPrograms();
  if (!coresWithPrograms.empty()) {
    // Save ELF files to the current directory
    // In a production system, this should use the same directory as the input xclbin
    std::string elfDir = ".";
    coreProgExtractor.saveELFFiles(elfDir);

    // Generate aie.core operations for each core with program memory
    builder.setInsertionPointToEnd(deviceBlock);

    for (const auto &tileId : coresWithPrograms) {
      // Get or create tile operation (if using lifted emitter)
      AIE::TileOp tile;
      if (emitLifted && liftedEmitter) {
        tile = liftedEmitter->getOrCreateTile(tileId.col, tileId.row);
      } else {
        // Create tile if not in lifted mode
        tile = AIE::TileOp::getOrCreate(builder, deviceOp, tileId.col, tileId.row);
      }

      // Create aie.core with elf_file attribute
      std::string elfFile = llvm::formatv("core_{0}_{1}.elf", tileId.col, tileId.row);

      auto coreOp = builder.create<AIE::CoreOp>(
          builder.getUnknownLoc(),
          builder.getIndexType(),
          tile.getResult(),
          /*stack_size=*/builder.getI32IntegerAttr(0x400),  // Default stack size
          /*link_with=*/nullptr,
          /*elf_file=*/builder.getStringAttr(elfFile),
          /*dynamic_objfifo_lowering=*/nullptr);

      // Add empty region with terminator since elf_file is present
      Block *coreBlock = &coreOp.getBody().emplaceBlock();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(coreBlock);
      AIE::EndOp::create(builder, builder.getUnknownLoc());
    }
  }

  // Add terminator to device block
  builder.setInsertionPointToEnd(deviceBlock);
  AIE::EndOp::create(builder, builder.getUnknownLoc());

  return success();
}

/// Extract BD configurations from parsed transaction module.
/// This walks the transaction module looking for aiex.npu.blockwrite operations that configure
/// DMA buffer descriptors, parses the BD fields, and adds them to the LiftedBDEmitter.
void extractBDsFromTransaction(ModuleOp txnModule, AIE::DeviceOp deviceOp,
                                BDAddressParser &bdParser, BDAccumulator &bdAccum,
                                DMAChannelTracker &dmaChannelTracker,
                                LiftedBDEmitter &emitter) {
  // Walk the transaction module to find write32 operations that configure BDs or Start_Queue
  int write32ProcessedCount = 0;
  int bdWriteCount = 0;
  int completedBDCount = 0;

  llvm::errs() << "[DEBUG extractBDsFromTransaction] Starting to walk transaction module for NPU write32 ops...\n";

  txnModule.walk([&](AIEX::NpuWrite32Op writeOp) {
    write32ProcessedCount++;
    uint32_t address = writeOp.getAddress();
    uint32_t value = writeOp.getValue();

    // Check if this is a DMA Start_Queue write and track channel assignment
    auto channelAssignment = dmaChannelTracker.parseStartQueue(address, value);
    if (channelAssignment.has_value()) {
      int col = (address >> 20) & 0xFF;
      int row = (address >> 28) & 0xFF;
      int bdIndex = channelAssignment->first;
      DMAChannelTracker::Channel channel = channelAssignment->second;
      dmaChannelTracker.recordAssignment(col, row, bdIndex, channel);
      llvm::errs() << "[DEBUG extractBDsFromTransaction] Found Start_Queue: tile(" << col << "," << row
                   << ") BD" << bdIndex << " -> channel " << static_cast<int>(channel) << "\n";
      return WalkResult::advance();
    }

    // Check if this write is configuring a BD register
    if (!bdParser.isBDAddress(address)) {
      return WalkResult::advance();
    }

    bdWriteCount++;
    llvm::errs() << "[DEBUG extractBDsFromTransaction] BD write #" << bdWriteCount
                 << ": addr=0x" << llvm::format("%08X", address)
                 << " value=0x" << llvm::format("%08X", value) << "\n";

    // Feed this write to the BD accumulator
    auto completedBD = bdAccum.addWrite(address, value, bdParser);

    // If this write completed a BD, record it
    if (completedBD.has_value()) {
      completedBDCount++;
      // Assign DMA channel to BD before recording
      auto &bd = *completedBD;
      auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
      bd.dmaChannel = static_cast<int>(channel);

      llvm::errs() << "[DEBUG extractBDsFromTransaction] *** COMPLETED BD #" << completedBDCount
                   << ": tile(" << bd.column << "," << bd.row << ") BD" << bd.bdIndex
                   << " bufferLength=" << bd.bufferLength
                   << " baseAddress=0x" << llvm::format("%08X", bd.baseAddress)
                   << " channel=" << bd.dmaChannel
                   << " lockAcqId=" << (int)bd.lockAcquire.lockId
                   << " lockRelId=" << (int)bd.lockRelId
                   << " nextBd=" << (int)bd.nextBd << " useNextBd=" << bd.useNextBd << "\n";

      emitter.recordBD(bd);

      // Record buffer length if available
      if (bd.bufferLength > 0) {
        emitter.recordBufferLength(bd.column, bd.row, bd.bdIndex, bd.bufferLength);
      }
    }

    return WalkResult::advance();
  });

  llvm::errs() << "[DEBUG extractBDsFromTransaction] Summary: processed " << write32ProcessedCount
               << " write32 ops, " << bdWriteCount << " BD writes, " << completedBDCount << " completed BDs\n";

  // Also walk the transaction module to find blockwrite operations (if any)
  int blockwriteProcessedCount = 0;
  txnModule.walk([&](AIEX::NpuBlockWriteOp blockwriteOp) {
    blockwriteProcessedCount++;
    uint32_t address = blockwriteOp.getAddress();

    // Check if this blockwrite is configuring a BD
    if (!bdParser.isBDAddress(address)) {
      return WalkResult::advance();
    }

    // Get the BD address info
    BDAddressInfo addrInfo = bdParser.parse(address);

    // Get the data being written from the blockwrite
    auto dataMemref = blockwriteOp.getData();

    // Trace back to the memref.global that contains the actual data
    if (auto getGlobalOp = dataMemref.getDefiningOp<memref::GetGlobalOp>()) {
      // Look up the global in the transaction module's device op
      AIE::DeviceOp txnDeviceOp;
      txnModule.walk([&](AIE::DeviceOp dev) {
        txnDeviceOp = dev;
        return WalkResult::interrupt();
      });

      if (!txnDeviceOp) {
        return WalkResult::advance();
      }

      auto globalOp = txnDeviceOp.lookupSymbol<memref::GlobalOp>(getGlobalOp.getName());
      if (!globalOp) {
        return WalkResult::advance();
      }

      // Extract the data values from the global
      auto dataAttr = globalOp.getInitialValue();
      if (!dataAttr) {
        return WalkResult::advance();
      }

      if (!dataAttr.has_value()) {
        return WalkResult::advance();
      }

      auto denseAttr = llvm::dyn_cast<DenseIntElementsAttr>(*dataAttr);
      if (!denseAttr) {
        return WalkResult::advance();
      }

      // Convert the dense attribute to a vector of uint32_t values
      llvm::SmallVector<uint32_t> words;
      for (auto val : denseAttr.getValues<llvm::APInt>()) {
        words.push_back(val.getZExtValue());
      }

      // The blockwrite data contains the BD registers. For compute tiles, there are 6 registers.
      // For shim/memtile, there are 8 registers. We need to feed these into the BD accumulator.

      // Determine which BD register this blockwrite is targeting
      int column = addrInfo.column;
      int row = addrInfo.row;

      // The blockwrite address points to the start of the BD (DMA_BDx_0)
      // We need to simulate writing each register in sequence
      const auto &targetModel = getTargetModel(deviceOp);
      int numRegs = 6;  // Default for compute tiles
      if (targetModel.isShimNOCorPLTile(column, row) || targetModel.isMemTile(column, row)) {
        numRegs = 8;
      }

      // Feed each word to the BD accumulator as a separate register write
      uint32_t baseAddr = address;
      for (size_t i = 0; i < words.size() && i < static_cast<size_t>(numRegs); i++) {
        uint32_t regAddr = baseAddr + (i * 4);  // Each register is 4 bytes apart
        uint32_t regValue = words[i];

        auto completedBD = bdAccum.addWrite(regAddr, regValue, bdParser);

        // If this write completed a BD, record it
        if (completedBD.has_value()) {
          // Assign DMA channel to BD before recording
          auto &bd = *completedBD;
          auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
          bd.dmaChannel = static_cast<int>(channel);
          emitter.recordBD(bd);

          // Record buffer length if available
          if (bd.bufferLength > 0) {
            emitter.recordBufferLength(bd.column, bd.row, bd.bdIndex, bd.bufferLength);
          }
        }
      }
    }

    return WalkResult::advance();
  });

  // Also look for NpuPushQueueOp operations to determine channel assignments
  txnModule.walk([&](AIEX::NpuPushQueueOp pushOp) {
    int col = pushOp.getColumn();
    int row = pushOp.getRow();
    int bdId = pushOp.getBdId();

    // Determine the channel from direction and channel index
    auto dir = pushOp.getDirection();
    int chanIdx = pushOp.getChannel();

    DMAChannelTracker::Channel channel;
    if (dir == AIE::DMAChannelDir::S2MM) {
      channel = (chanIdx == 0) ? DMAChannelTracker::Channel::S2MM_0
                                : DMAChannelTracker::Channel::S2MM_1;
    } else {  // MM2S
      channel = (chanIdx == 0) ? DMAChannelTracker::Channel::MM2S_0
                                : DMAChannelTracker::Channel::MM2S_1;
    }

    dmaChannelTracker.recordAssignment(col, row, bdId, channel);
    return WalkResult::advance();
  });

  // Flush any pending BDs from the accumulator
  auto pendingBDs = bdAccum.flush();
  for (auto &bd : pendingBDs) {
    // Assign channel if we found it in push_queue operations
    auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
    bd.dmaChannel = static_cast<int>(channel);
    emitter.recordBD(bd);
  }
}

/// Helper structure to track a sequence of operations that form a DMA transfer
struct DMATransferPattern {
  AIEX::NpuBlockWriteOp blockwrite = nullptr;
  AIEX::NpuAddressPatchOp addressPatch = nullptr;
  AIEX::NpuWrite32Op queuePush = nullptr;
  AIEX::NpuMaskWrite32Op controlWrite = nullptr;

  // Extracted parameters
  uint32_t bdAddress = 0;
  uint32_t column = 0;
  uint32_t row = 0;
  uint32_t bdId = 0;
  int32_t argIdx = -1;
  SmallVector<uint32_t, 8> bdData;

  bool isComplete() const {
    return blockwrite && addressPatch;
  }
};

/// Extract column and row from a shim tile address
static std::pair<uint32_t, uint32_t> extractTileFromAddress(uint32_t address) {
  // NPU address format: bits [27:20] = column, bits [31:28] = row
  // For shim tiles (row 0): 0x001D_XXXX format where column is encoded
  uint32_t column = (address >> 20) & 0xFF;
  uint32_t row = (address >> 28) & 0xF;
  return {column, row};
}

/// Check if an address is a BD address for shim DMA
static bool isShimBDAddress(uint32_t address) {
  // Shim DMA BD addresses start at 0x1D000 for column 0
  // Each tile is at base (col << 20) | (row << 28) | 0x1D000
  // BDs are at offsets 0x000, 0x020, 0x040, etc. (8 words = 32 bytes each)
  uint32_t tileOffset = address & 0xFFFFF;  // Mask to get offset within tile

  // BD addresses are in the range 0x1D000 - 0x1D1FF (16 BDs * 32 bytes = 512 = 0x200)
  return (tileOffset >= 0x1D000 && tileOffset < 0x1D200);
}

/// Extract BD index from a BD address
static uint32_t extractBDIndex(uint32_t address) {
  uint32_t tileOffset = address & 0xFFFFF;
  // BD addresses start at 0x1D000, each BD is 32 bytes
  return (tileOffset - 0x1D000) / 0x20;
}

/// Lift NPU instructions from raw register writes to high-level AIEX operations.
/// This function analyzes the runtime_sequence and identifies patterns that can be
/// converted from low-level operations (write32, blockwrite, etc.) to semantic
/// operations (dma_memcpy_nd, sync, etc.).
void liftNPUInstructions(AIE::RuntimeSequenceOp seqOp, AIE::DeviceOp deviceOp,
                         LiftedBDEmitter &emitter) {
  llvm::errs() << "\n=== NPU INSTRUCTION LIFTING STARTING ===\n";

  if (!seqOp) {
    llvm::errs() << "ERROR: No RuntimeSequenceOp provided\n";
    return;
  }

  Block &seqBlock = seqOp.getBody().front();
  OpBuilder builder(seqOp.getContext());

  // Collect operations to process
  SmallVector<Operation *> opsToProcess;
  for (Operation &op : seqBlock.getOperations()) {
    if (!isa<AIE::EndOp>(&op)) {
      opsToProcess.push_back(&op);
    }
  }

  llvm::errs() << "Found " << opsToProcess.size() << " operations in runtime_sequence\n";

  // Count operation types for debugging
  int numBlockwrite = 0, numAddressPatch = 0, numWrite32 = 0, numMaskWrite32 = 0, numSync = 0;
  for (Operation *op : opsToProcess) {
    if (isa<AIEX::NpuBlockWriteOp>(op)) numBlockwrite++;
    else if (isa<AIEX::NpuAddressPatchOp>(op)) numAddressPatch++;
    else if (isa<AIEX::NpuWrite32Op>(op)) numWrite32++;
    else if (isa<AIEX::NpuMaskWrite32Op>(op)) numMaskWrite32++;
    else if (isa<AIEX::NpuSyncOp>(op)) numSync++;
  }
  llvm::errs() << "Operation types: blockwrite=" << numBlockwrite
               << " address_patch=" << numAddressPatch
               << " write32=" << numWrite32
               << " maskwrite32=" << numMaskWrite32
               << " sync=" << numSync << "\n";

  // Pattern recognition: Group operations that belong to a DMA transfer
  SmallVector<DMATransferPattern> dmaPatterns;
  DMATransferPattern currentPattern;

  // Strategy: If no blockwrite operations exist (transaction binary case),
  // we need to recognize sequences of write32 operations to BD registers
  if (numBlockwrite == 0 && numAddressPatch > 0) {
    llvm::errs() << "Using write32 sequence pattern matching (transaction binary mode)\n";

    // Accumulate write32 operations by BD address
    llvm::DenseMap<uint32_t, SmallVector<std::pair<uint32_t, uint32_t>>> bdWrites; // bdBaseAddr -> [(offset, value)]

    // First pass: log first few write32 addresses to understand the pattern
    int logged = 0;
    for (size_t i = 0; i < opsToProcess.size() && logged < 10; ++i) {
      if (auto writeOp = dyn_cast<AIEX::NpuWrite32Op>(opsToProcess[i])) {
        uint32_t address = writeOp.getAddress();
        llvm::errs() << "  Write32 #" << i << " -> 0x" << llvm::format_hex(address, 8)
                     << " isShimBD=" << isShimBDAddress(address) << "\n";
        logged++;
      }
    }

    for (size_t i = 0; i < opsToProcess.size(); ++i) {
      if (auto writeOp = dyn_cast<AIEX::NpuWrite32Op>(opsToProcess[i])) {
        uint32_t address = writeOp.getAddress();

        // Check if this is a BD register write
        if (isShimBDAddress(address)) {
          // Calculate BD base address (aligned to 32 bytes = 0x20)
          uint32_t tileOffset = address & 0xFFFFF;
          uint32_t bdOffset = tileOffset - 0x1D000;
          uint32_t bdId = bdOffset / 0x20;
          uint32_t bdBaseAddr = (address & 0xFFF00000) | 0x1D000 | (bdId * 0x20);
          uint32_t wordOffset = address - bdBaseAddr;

          bdWrites[bdBaseAddr].push_back({wordOffset, writeOp.getValue()});
        }
      }
    }

    llvm::errs() << "Found " << bdWrites.size() << " BDs with write32 sequences\n";

    // Process each BD that has enough writes
    for (const auto &[bdBaseAddr, writes] : bdWrites) {
      if (writes.size() < 6) {
        llvm::errs() << "  Skipping BD at 0x" << llvm::format_hex(bdBaseAddr, 8)
                     << " - only " << writes.size() << " writes\n";
        continue;
      }

      // Reconstruct BD data array from individual writes
      DMATransferPattern pattern;
      pattern.bdAddress = bdBaseAddr;
      auto [col, row] = extractTileFromAddress(bdBaseAddr);
      pattern.column = col;
      pattern.row = row;
      pattern.bdId = extractBDIndex(bdBaseAddr);

      // Initialize BD data array with zeros
      pattern.bdData.resize(8, 0);

      // Fill in the written values
      for (const auto &[offset, value] : writes) {
        uint32_t wordIndex = offset / 4;
        if (wordIndex < 8) {
          pattern.bdData[wordIndex] = value;
        }
      }

      llvm::errs() << "  Reconstructed BD " << pattern.bdId << " at tile("
                   << col << "," << row << ") with " << writes.size() << " writes\n";

      // Find associated address_patch for this BD
      for (size_t i = 0; i < opsToProcess.size(); ++i) {
        if (auto patchOp = dyn_cast<AIEX::NpuAddressPatchOp>(opsToProcess[i])) {
          // Check if patch is for this BD (typically BD_base + 4 for buffer address field)
          uint32_t patchAddr = patchOp.getAddr();
          if (patchAddr >= bdBaseAddr && patchAddr < bdBaseAddr + 0x20) {
            pattern.addressPatch = patchOp;
            pattern.argIdx = patchOp.getArgIdx();
            llvm::errs() << "    Found address_patch at offset " << (patchAddr - bdBaseAddr)
                         << " for arg_idx=" << pattern.argIdx << "\n";
            break;
          }
        }
      }

      // Add pattern if it has the necessary components
      if (pattern.addressPatch) {
        dmaPatterns.push_back(pattern);
      }
    }
  }

  // Fallback: Use blockwrite-based pattern matching (mid-level MLIR case)
  for (size_t i = 0; i < opsToProcess.size(); ++i) {
    Operation *op = opsToProcess[i];

    // Look for blockwrite to a BD address
    if (auto blockwriteOp = dyn_cast<AIEX::NpuBlockWriteOp>(op)) {
      uint32_t address = blockwriteOp.getAddress();
      llvm::errs() << "  Blockwrite #" << i << " to address 0x" << llvm::format_hex(address, 8)
                   << " - isShimBD=" << isShimBDAddress(address) << "\n";

      if (isShimBDAddress(address)) {
        // Start a new pattern
        if (currentPattern.isComplete()) {
          dmaPatterns.push_back(currentPattern);
        }
        currentPattern = DMATransferPattern();
        currentPattern.blockwrite = blockwriteOp;
        currentPattern.bdAddress = address;

        auto [col, row] = extractTileFromAddress(address);
        currentPattern.column = col;
        currentPattern.row = row;
        currentPattern.bdId = extractBDIndex(address);

        // Extract BD data from the memref
        if (auto getGlobalOp = blockwriteOp.getData().getDefiningOp<memref::GetGlobalOp>()) {
          if (auto globalOp = deviceOp.lookupSymbol<memref::GlobalOp>(
                  getGlobalOp.getName())) {
            auto dataAttr = globalOp.getInitialValue();
            if (dataAttr.has_value()) {
              if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(*dataAttr)) {
                for (APInt val : denseAttr.getValues<APInt>()) {
                  currentPattern.bdData.push_back(val.getZExtValue());
                }
              }
            }
          }
        }
      }
    }

    // Look for address_patch that follows a blockwrite
    else if (auto patchOp = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
      if (currentPattern.blockwrite && !currentPattern.addressPatch) {
        // Check if this patch corresponds to the BD we just wrote
        // Address patch should be to BD_address + 4 (the buffer address field)
        uint32_t patchAddr = patchOp.getAddr();
        if (patchAddr == currentPattern.bdAddress + 4) {
          currentPattern.addressPatch = patchOp;
          currentPattern.argIdx = patchOp.getArgIdx();
        }
      }
    }

    // Look for maskwrite32 (control register write)
    else if (auto maskwriteOp = dyn_cast<AIEX::NpuMaskWrite32Op>(op)) {
      if (currentPattern.blockwrite && !currentPattern.controlWrite) {
        currentPattern.controlWrite = maskwriteOp;
      }
    }

    // Look for write32 (queue push)
    else if (auto writeOp = dyn_cast<AIEX::NpuWrite32Op>(op)) {
      if (currentPattern.blockwrite && !currentPattern.queuePush) {
        currentPattern.queuePush = writeOp;

        // This completes the pattern - save it
        if (currentPattern.isComplete()) {
          dmaPatterns.push_back(currentPattern);
          currentPattern = DMATransferPattern();
        }
      }
    }
  }

  // Save any remaining pattern
  if (currentPattern.isComplete()) {
    dmaPatterns.push_back(currentPattern);
  }

  // Log what we found
  if (!dmaPatterns.empty()) {
    llvm::errs() << "DEBUG: Found " << dmaPatterns.size() << " DMA transfer patterns\n";
    for (const auto &pattern : dmaPatterns) {
      llvm::errs() << "  - BD " << pattern.bdId << " at tile("
                   << pattern.column << "," << pattern.row
                   << ") with " << pattern.bdData.size() << " data words"
                   << " arg_idx=" << pattern.argIdx << "\n";
      if (!pattern.bdData.empty()) {
        llvm::errs() << "    BD data: ";
        for (size_t i = 0; i < std::min(pattern.bdData.size(), size_t(4)); ++i) {
          llvm::errs() << "0x" << llvm::format_hex(pattern.bdData[i], 8) << " ";
        }
        llvm::errs() << "...\n";
      }
    }
  }

  // Now lift the patterns to high-level operations
  SmallVector<Operation *> opsToErase;

  for (const auto &pattern : dmaPatterns) {
    if (pattern.bdData.size() < 8) {
      llvm::errs() << "Warning: BD " << pattern.bdId << " has insufficient data ("
                   << pattern.bdData.size() << " words), skipping\n";
      continue;
    }

    // Parse BD data according to shim BD format (AIEDmaToNpu.cpp lines 542-687)
    uint32_t buffer_length = pattern.bdData[0];
    // uint32_t buffer_offset = pattern.bdData[1];  // Not needed for reconstruction

    // Record buffer length for this BD so it can be used for buffer size inference
    emitter.recordBufferLength(pattern.column, pattern.row, pattern.bdId, buffer_length);

    // Word 2: enable_packet, out_of_order_id, packet_id, packet_type
    // uint32_t word2 = pattern.bdData[2];
    // uint32_t enable_packet = (word2 >> 30) & 0x1;
    // uint32_t packet_id = (word2 >> 19) & 0x1F;
    // uint32_t packet_type = (word2 >> 16) & 0x7;

    // Word 3: d0_size, d0_stride
    uint32_t word3 = pattern.bdData[3];
    uint32_t d0_size = (word3 >> 20) & 0x3FF;
    uint32_t d0_stride = word3 & 0xFFFFF;

    // Word 4: burst_length, d1_size, d1_stride
    uint32_t word4 = pattern.bdData[4];
    // uint32_t burst_length = (word4 >> 30) & 0x3;
    uint32_t d1_size = (word4 >> 20) & 0x3FF;
    uint32_t d1_stride = word4 & 0xFFFFF;

    // Word 5: d2_stride
    uint32_t word5 = pattern.bdData[5];
    uint32_t d2_stride = word5 & 0xFFFFF;

    // Word 6: iteration_current, iteration_size, iteration_stride
    uint32_t word6 = pattern.bdData[6];
    uint32_t iteration_size = (word6 >> 20) & 0x3F;
    // uint32_t iteration_stride = word6 & 0xFFFFF;

    // Word 7: next_bd, use_next_bd, valid_bd, locks
    // uint32_t word7 = pattern.bdData[7];
    // uint32_t next_bd = (word7 >> 27) & 0xF;
    // uint32_t use_next_bd = (word7 >> 26) & 0x1;
    // uint32_t valid_bd = (word7 >> 25) & 0x1;

    llvm::errs() << "  Parsed BD " << pattern.bdId << ": len=" << buffer_length
                 << " d0=" << d0_size << "x" << d0_stride
                 << " d1=" << d1_size << "x" << d1_stride
                 << " d2_stride=" << d2_stride << "\n";

    // Find the shim_dma_allocation that matches this tile and determine direction
    AIE::ShimDMAAllocationOp allocOp;
    AIE::DMAChannelDir direction = AIE::DMAChannelDir::MM2S;

    // Walk device to find shim_dma_allocation ops
    deviceOp.walk([&](AIE::ShimDMAAllocationOp op) {
      if (auto tile = op.getTileOp()) {
        if ((uint32_t)tile.getCol() == pattern.column && (uint32_t)tile.getRow() == pattern.row) {
          // Use the first matching allocation for now
          // TODO: Better matching based on channel and BD ID
          if (!allocOp) {
            allocOp = op;
            direction = op.getChannelDir();
          }
        }
      }
    });

    if (!allocOp) {
      llvm::errs() << "Warning: No shim_dma_allocation found for tile("
                   << pattern.column << "," << pattern.row << "), skipping\n";
      continue;
    }

    // Get the runtime sequence to find the memref argument
    Block &entryBlock = seqBlock;
    if (pattern.argIdx < 0 || (size_t)pattern.argIdx >= entryBlock.getNumArguments()) {
      llvm::errs() << "Warning: Invalid arg_idx " << pattern.argIdx << ", skipping\n";
      continue;
    }

    // Build the operation at the location of the blockwrite
    // builder.setInsertionPoint(pattern.blockwrite);
    // Location loc = pattern.blockwrite->getLoc();

    // Compute dimensions based on parsed BD data
    // The BD format uses hardware dimensions, need to convert to logical MLIR dimensions

    // Sizes: Use buffer_length for innermost dimension
    // iteration_size for outermost dimension if present
    int64_t size3 = (iteration_size > 0) ? iteration_size : 1;
    int64_t size2 = 1;  // d2_size not stored in shim BDs
    int64_t size1 = (d1_size > 0) ? d1_size : 1;
    int64_t size0 = (d0_size > 0) ? d0_size : 1;

    // If d0_size is 0, use buffer_length directly
    if (d0_size == 0 && d1_size == 0) {
      size0 = buffer_length;
      size1 = 1;
      size2 = 1;
      size3 = 1;
    }

    const std::vector<int64_t> staticOffsets = {0, 0, 0, 0};
    const std::vector<int64_t> staticSizes = {size3, size2, size1, size0};
    const std::vector<int64_t> staticStrides = {0, 0, 0, 1};

    // Get the memref argument
    Value memref = entryBlock.getArgument(pattern.argIdx);

    // Set insertion point before the blockwrite
    builder.setInsertionPoint(pattern.blockwrite);

    // Create the NpuDmaMemcpyNdOp using the EXACT pattern from AIECtrlPacketToDma.cpp line 195
    bool issueToken = (direction == AIE::DMAChannelDir::S2MM);
    SymbolRefAttr metadata = SymbolRefAttr::get(builder.getContext(), allocOp.getSymName());
    AIEX::NpuDmaMemcpyNdOp::create(builder, builder.getUnknownLoc(), memref,
                                   SmallVector<Value>{}, SmallVector<Value>{},
                                   SmallVector<Value>{}, ArrayRef(staticOffsets),
                                   ArrayRef(staticSizes), ArrayRef(staticStrides),
                                   nullptr, metadata, pattern.bdId,
                                   issueToken, 0, 0, 0, 0, 0, 0);

    llvm::errs() << "  Created NpuDmaMemcpyNdOp for BD " << pattern.bdId
                 << " with sizes=[" << size3 << "," << size2 << "," << size1 << "," << size0 << "]"
                 << " arg_idx=" << pattern.argIdx
                 << " metadata=" << allocOp.getSymName().str() << "\n";

    // Mark operations for erasure
    opsToErase.push_back(pattern.blockwrite);
    if (pattern.addressPatch)
      opsToErase.push_back(pattern.addressPatch);
    if (pattern.controlWrite)
      opsToErase.push_back(pattern.controlWrite);
    if (pattern.queuePush)
      opsToErase.push_back(pattern.queuePush);

  }

  // Erase old low-level operations that were replaced with high-level ops
  for (Operation *op : opsToErase) {
    op->erase();
  }

  if (!dmaPatterns.empty()) {
    llvm::errs() << "DEBUG: Successfully lifted " << dmaPatterns.size()
                 << " DMA transfer patterns to NpuDmaMemcpyNdOp\n";
    llvm::errs() << "      Erased " << opsToErase.size()
                 << " low-level operations (blockwrite, address_patch, write32)\n";
  }
}

#endif // HAVE_BOOTGEN

} // namespace

namespace xilinx {
namespace AIE {

/// Main entry point: translate xclbin binary to MLIR module.
LogicalResult AIETranslateFromXclbin(ModuleOp module, StringRef filename,
                                     bool emitLifted, StringRef npuInstsPath) {
#ifdef HAVE_BOOTGEN
  // Step 1: Extract PDI from xclbin
  std::vector<uint8_t> pdiData;
  if (failed(extractPDIFromXclbin(filename, pdiData))) {
    return module.emitError("Failed to extract PDI from xclbin");
  }

  // Step 1.5: Try to extract AIE_METADATA section (contains routing info)
  std::string metadataJson;
  (void)extractAIEMetadata(filename, metadataJson);

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

  // Step 4: If transaction binary provided, parse it first
  std::optional<ModuleOp> txnModule = std::nullopt;
  if (!npuInstsPath.empty()) {
    // Load NPU instructions file (could be ELF or raw binary)
    auto fileOrErr = llvm::MemoryBuffer::getFile(npuInstsPath);
    if (!fileOrErr) {
      return module.emitError("Failed to open NPU instructions file: ")
             << npuInstsPath;
    }

    llvm::MemoryBuffer *buffer = fileOrErr.get().get();
    std::vector<uint8_t> txnData;

    // Check if this is an ELF file (magic: 0x7F 'E' 'L' 'F')
    const uint8_t *data = reinterpret_cast<const uint8_t*>(buffer->getBufferStart());
    size_t size = buffer->getBufferSize();

    if (size >= 4 && data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
      // ELF file - extract .ctrltext section
      // The .ctrltext section is at offset 0xA0 (160) with size 0x12C (300) for typical NPU insts
      // However, we should parse the ELF header properly
      // For now, use a simple extraction based on section header at offset 0x34

      // Simple ELF32 parser: section header table offset is at bytes 32-35
      if (size < 52) {  // Minimum ELF32 header size
        return module.emitError("ELF file too small");
      }

      uint32_t sh_off = *reinterpret_cast<const uint32_t*>(data + 32);  // Section header table offset
      uint16_t sh_entsize = *reinterpret_cast<const uint16_t*>(data + 46);  // Section header entry size
      uint16_t sh_num = *reinterpret_cast<const uint16_t*>(data + 48);  // Number of section headers
      uint16_t sh_strndx = *reinterpret_cast<const uint16_t*>(data + 50);  // Section name string table index

      if (sh_off + (sh_num * sh_entsize) > size) {
        return module.emitError("Invalid ELF section header table");
      }

      // Find the .shstrtab (section name string table)
      const uint8_t *shstrtab_hdr = data + sh_off + (sh_strndx * sh_entsize);
      uint32_t shstrtab_offset = *reinterpret_cast<const uint32_t*>(shstrtab_hdr + 16);  // sh_offset

      // Search for .ctrltext section
      bool found = false;
      for (uint16_t i = 0; i < sh_num && !found; i++) {
        const uint8_t *sh = data + sh_off + (i * sh_entsize);
        uint32_t sh_name_idx = *reinterpret_cast<const uint32_t*>(sh);  // Index into shstrtab
        uint32_t sh_offset_val = *reinterpret_cast<const uint32_t*>(sh + 16);  // sh_offset
        uint32_t sh_size = *reinterpret_cast<const uint32_t*>(sh + 20);  // sh_size

        // Check section name
        if (shstrtab_offset + sh_name_idx < size) {
          const char *section_name = reinterpret_cast<const char*>(data + shstrtab_offset + sh_name_idx);
          if (std::strcmp(section_name, ".ctrltext") == 0) {
            // Found .ctrltext section - extract it
            if (sh_offset_val + sh_size <= size) {
              txnData.assign(data + sh_offset_val, data + sh_offset_val + sh_size);
              found = true;
            } else {
              return module.emitError("Invalid .ctrltext section size in ELF");
            }
          }
        }
      }

      if (!found) {
        return module.emitError("Could not find .ctrltext section in ELF file");
      }
    } else {
      // Raw binary file - use as-is
      txnData.assign(data, data + size);
    }

    // Parse transaction binary to MLIR using existing converter
    txnModule = convertTransactionBinaryToMLIR(module.getContext(), txnData);
    if (!txnModule) {
      return module.emitError("Failed to parse transaction binary");
    }
  }

  // Step 5: Emit MLIR operations (lifted or raw mode)
  // Pass the transaction module so BDs can be extracted from it
  // Also pass raw CDO data for program memory extraction
  if (failed(emitMLIRFromCDO(module, commands, emitLifted, txnModule,
                             cdoData.data(), cdoData.size()))) {
    return module.emitError("Failed to emit MLIR from CDO commands");
  }

  return success();
#else
  return module.emitError("CDO decoding not available - bootgen library was not built (OpenSSL required)");
#endif
}

} // namespace AIE
} // namespace xilinx
