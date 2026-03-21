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

// Forward declaration
void extractBDsFromTransaction(ModuleOp txnModule, AIE::DeviceOp deviceOp,
                                BDAddressParser &bdParser, BDAccumulator &bdAccum,
                                DMAChannelTracker &dmaChannelTracker,
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
  uint32_t row = (addr >> 20) & 0x7;
  uint32_t tileBase = row << 20;
  uint32_t offset = addr - tileBase;

  // Core_Control register at offset 0x32000
  // Only applies to compute tiles (row >= 2)
  if (offset == 0x32000 && row >= 2 && (mask == 1 || mask == 2)) {
    return true;
  }

  return false;
}

/// Check if a maskwrite is a DMA control operation (channel enable).
/// MemTile DMA control registers are at:
///   0xA0600: DMA_S2MM_0_Ctrl
///   0xA0608: DMA_S2MM_1_Ctrl
///   0xA0630: DMA_MM2S_0_Ctrl
///   0xA0638: DMA_MM2S_1_Ctrl
///   0xA0640: DMA_MM2S_2_Ctrl
///   0xA0648: DMA_MM2S_3_Ctrl
/// These operations enable DMA channels and are derivable from BD configuration.
static bool isDMAControlOperation(uint32_t addr, uint32_t mask, uint32_t value) {
  uint32_t row = (addr >> 20) & 0x7;
  uint32_t tileBase = row << 20;
  uint32_t offset = addr - tileBase;

  // Check if this is a memtile (row == 1)
  if (row != 1) {
    return false;
  }

  // DMA control registers for memtile
  if ((offset == 0xA0600 || offset == 0xA0608 ||  // S2MM channels
       offset == 0xA0630 || offset == 0xA0638 ||  // MM2S channels
       offset == 0xA0640 || offset == 0xA0648) &&  // More MM2S channels
      mask == 0 && value == 1) {  // Enable operation
    return true;
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

  // Extract tile offset from address
  uint32_t row = (addr >> 20) & 0x7;
  uint32_t tileBase = row << 20;
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

  // Other common initialization addresses
  // Shim tile region (row=0, low addresses)
  if (row == 0 && addr < 0x100000) {
    return true;  // Shim initialization
  }

  return false;
}

/// Emit MLIR operations from decoded CDO commands.
/// Creates aie.device, runtime_sequence, and MLIR operations for register writes.
LogicalResult emitMLIRFromCDO(ModuleOp module,
                              llvm::ArrayRef<CdoCommand *> commands,
                              bool emitLifted = false,
                              std::optional<ModuleOp> txnModule = std::nullopt) {
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
      for (Operation &op : txnDeviceOp.getBody()->getOperations()) {
        // Skip the terminator and globals (already copied above)
        if (isa<AIE::EndOp>(&op) || isa<memref::GlobalOp>(&op)) {
          continue;
        }

        // Clone the operation into the output runtime_sequence, using the mapper
        // to ensure SSA values are properly remapped
        builder.setInsertionPointToEnd(seqBlock);
        builder.clone(op, mapper);
      }
      return WalkResult::interrupt();  // Only process the first DeviceOp
    });
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
    liftedEmitter->emitAllFlows();
    liftedEmitter->emitAllSwitchboxes();
    liftedEmitter->emitAllShimMuxes();
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
      return WalkResult::advance();
    }

    // Check if this write is configuring a BD register
    if (!bdParser.isBDAddress(address)) {
      return WalkResult::advance();
    }

    // Feed this write to the BD accumulator
    auto completedBD = bdAccum.addWrite(address, value, bdParser);

    // If this write completed a BD, record it
    if (completedBD.has_value()) {
      // Assign DMA channel to BD before recording
      auto &bd = *completedBD;
      auto channel = dmaChannelTracker.getChannel(bd.column, bd.row, bd.bdIndex);
      bd.dmaChannel = static_cast<int>(channel);
      emitter.recordBD(bd);
    }

    return WalkResult::advance();
  });

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
    // Load transaction binary from file
    auto fileOrErr = llvm::MemoryBuffer::getFile(npuInstsPath);
    if (!fileOrErr) {
      return module.emitError("Failed to open NPU instructions file: ")
             << npuInstsPath;
    }

    llvm::MemoryBuffer *buffer = fileOrErr.get().get();
    std::vector<uint8_t> txnData(
        reinterpret_cast<const uint8_t*>(buffer->getBufferStart()),
        reinterpret_cast<const uint8_t*>(buffer->getBufferEnd()));

    // Parse transaction binary to MLIR using existing converter
    txnModule = convertTransactionBinaryToMLIR(module.getContext(), txnData);
    if (!txnModule) {
      return module.emitError("Failed to parse transaction binary");
    }
  }

  // Step 5: Emit MLIR operations (lifted or raw mode)
  // Pass the transaction module so BDs can be extracted from it
  if (failed(emitMLIRFromCDO(module, commands, emitLifted, txnModule))) {
    return module.emitError("Failed to emit MLIR from CDO commands");
  }

  return success();
#else
  return module.emitError("CDO decoding not available - bootgen library was not built (OpenSSL required)");
#endif
}

} // namespace AIE
} // namespace xilinx
