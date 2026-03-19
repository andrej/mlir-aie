//===- bd_lifting.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for BD (Buffer Descriptor) lifting infrastructure:
// - BDFieldExtractor: bit field parsing from BD registers
// - BDAddressParser: address decoding to identify BD registers
// - BDAccumulator: accumulating register writes into complete BD configs
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIEDMABDLifting.h"

#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace xilinx::AIE;
using namespace mlir;

//===----------------------------------------------------------------------===//
// BDFieldExtractor Tests
//===----------------------------------------------------------------------===//

void test_extract_bits() {
  std::cout << "Test: BDFieldExtractor::extractBits\n";

  // Test extracting middle bits
  uint32_t value = 0b00001111000011110000111100001111;
  uint32_t result = BDFieldExtractor::extractBits(value, 15, 8);
  uint32_t expected = 0b11110000;
  if (result != expected) {
    throw std::runtime_error("extractBits(15, 8) failed: got " +
                            std::to_string(result) + ", expected " +
                            std::to_string(expected));
  }

  // Test extracting lowest bits
  result = BDFieldExtractor::extractBits(value, 3, 0);
  expected = 0b1111;
  if (result != expected) {
    throw std::runtime_error("extractBits(3, 0) failed");
  }

  // Test extracting highest bit
  value = 0x80000000;
  result = BDFieldExtractor::extractBits(value, 31, 31);
  expected = 1;
  if (result != expected) {
    throw std::runtime_error("extractBits(31, 31) failed");
  }

  std::cout << "  ✓ extractBits works correctly\n";
}

void test_extract_signed_bits() {
  std::cout << "Test: BDFieldExtractor::extractSignedBits\n";

  // Test positive value (sign bit = 0)
  uint32_t value = 0b00111111 << 5; // 6-bit positive value
  int32_t result = BDFieldExtractor::extractSignedBits(value, 10, 5);
  int32_t expected = 0b00111111; // Still positive
  if (result != expected) {
    throw std::runtime_error("extractSignedBits positive failed");
  }

  // Test negative value (sign bit = 1)
  value = 0b11000000 << 5; // 6-bit negative value (-64 in 6-bit two's complement)
  result = BDFieldExtractor::extractSignedBits(value, 10, 5);
  // Top bit is 1, so should sign-extend: 0b11000000 -> 0xFFFFFFC0
  if (result >= 0) {
    throw std::runtime_error("extractSignedBits should return negative value");
  }

  std::cout << "  ✓ extractSignedBits works correctly\n";
}

void test_bd_reg0_fields() {
  std::cout << "Test: BD Register 0 Field Extraction\n";

  // DMA_BDx_0 format:
  // [31:28] Reserved
  // [27:14] Base Address (14 bits)
  // [13:0]  Buffer Length (14 bits)

  uint32_t reg0 = 0x00010000; // Base addr = 0x1, length = 0
  uint32_t baseAddr = BDFieldExtractor::getBaseAddress(reg0);
  uint32_t bufLen = BDFieldExtractor::getBufferLength(reg0);

  if (baseAddr != 0x1) {
    throw std::runtime_error("getBaseAddress failed: got " +
                            std::to_string(baseAddr) + ", expected 1");
  }
  if (bufLen != 0) {
    throw std::runtime_error("getBufferLength failed");
  }

  // Test with actual values
  reg0 = (0x1234 << 14) | 0x567;  // base=0x1234, len=0x567
  baseAddr = BDFieldExtractor::getBaseAddress(reg0);
  bufLen = BDFieldExtractor::getBufferLength(reg0);

  if (baseAddr != 0x1234) {
    throw std::runtime_error("getBaseAddress failed with real value");
  }
  if (bufLen != 0x567) {
    throw std::runtime_error("getBufferLength failed with real value");
  }

  std::cout << "  ✓ Register 0 field extraction works\n";
}

void test_bd_reg1_fields() {
  std::cout << "Test: BD Register 1 Field Extraction\n";

  // DMA_BDx_1 format:
  // [31] Enable Compression
  // [30] Enable Packet
  // [29:24] Out-of-order BD ID
  // [23:19] Packet ID
  // [18:16] Packet Type

  uint32_t reg1 = (1u << 31) | (1u << 30) | (0x3F << 24) | (0x15 << 19) | (0x7 << 16);

  bool compression = BDFieldExtractor::getEnableCompression(reg1);
  bool packet = BDFieldExtractor::getEnablePacket(reg1);
  uint8_t oooId = BDFieldExtractor::getOutOfOrderBdId(reg1);
  uint8_t pktId = BDFieldExtractor::getPacketId(reg1);
  uint8_t pktType = BDFieldExtractor::getPacketType(reg1);

  if (!compression) throw std::runtime_error("getEnableCompression failed");
  if (!packet) throw std::runtime_error("getEnablePacket failed");
  if (oooId != 0x3F) throw std::runtime_error("getOutOfOrderBdId failed");
  if (pktId != 0x15) throw std::runtime_error("getPacketId failed");
  if (pktType != 0x7) throw std::runtime_error("getPacketType failed");

  std::cout << "  ✓ Register 1 field extraction works\n";
}

void test_bd_reg2_fields() {
  std::cout << "Test: BD Register 2 Field Extraction (D0/D1 Stepsize)\n";

  // DMA_BDx_2 format:
  // [25:13] D1_Stepsize (encoded as actual-1)
  // [12:0]  D0_Stepsize (encoded as actual-1)

  uint32_t reg2 = (99 << 13) | 49; // D1=100, D0=50 (encoded as 99, 49)

  uint16_t d0Step = BDFieldExtractor::getD0Stepsize(reg2);
  uint16_t d1Step = BDFieldExtractor::getD1Stepsize(reg2);

  if (d0Step != 50) {
    throw std::runtime_error("getD0Stepsize failed: got " +
                            std::to_string(d0Step) + ", expected 50");
  }
  if (d1Step != 100) {
    throw std::runtime_error("getD1Stepsize failed: got " +
                            std::to_string(d1Step) + ", expected 100");
  }

  std::cout << "  ✓ Register 2 field extraction works\n";
}

void test_bd_reg3_fields() {
  std::cout << "Test: BD Register 3 Field Extraction (D0/D1 Wrap, D2 Stepsize)\n";

  // DMA_BDx_3 format:
  // [28:21] D1_Wrap
  // [20:13] D0_Wrap
  // [12:0]  D2_Stepsize (encoded as actual-1)

  uint32_t reg3 = (32 << 21) | (16 << 13) | 199; // D1_wrap=32, D0_wrap=16, D2=200

  uint8_t d0Wrap = BDFieldExtractor::getD0Wrap(reg3);
  uint8_t d1Wrap = BDFieldExtractor::getD1Wrap(reg3);
  uint16_t d2Step = BDFieldExtractor::getD2Stepsize(reg3);

  if (d0Wrap != 16) throw std::runtime_error("getD0Wrap failed");
  if (d1Wrap != 32) throw std::runtime_error("getD1Wrap failed");
  if (d2Step != 200) throw std::runtime_error("getD2Stepsize failed");

  std::cout << "  ✓ Register 3 field extraction works\n";
}

void test_bd_reg4_fields() {
  std::cout << "Test: BD Register 4 Field Extraction (Iteration)\n";

  // DMA_BDx_4 format:
  // [24:19] Iteration Current
  // [18:13] Iteration Wrap (encoded as actual-1)
  // [12:0]  Iteration Stepsize (encoded as actual-1)

  uint32_t reg4 = (5 << 19) | (9 << 13) | 99; // Current=5, Wrap=10, Step=100

  uint8_t iterCurr = BDFieldExtractor::getIterationCurrent(reg4);
  uint8_t iterWrap = BDFieldExtractor::getIterationWrap(reg4);
  uint16_t iterStep = BDFieldExtractor::getIterationStepsize(reg4);

  if (iterCurr != 5) throw std::runtime_error("getIterationCurrent failed");
  if (iterWrap != 10) throw std::runtime_error("getIterationWrap failed");
  if (iterStep != 100) throw std::runtime_error("getIterationStepsize failed");

  std::cout << "  ✓ Register 4 field extraction works\n";
}

void test_bd_reg5_fields() {
  std::cout << "Test: BD Register 5 Field Extraction (Control & Locks)\n";

  // DMA_BDx_5 format:
  // [31]    TLAST Suppress
  // [30:27] Next BD
  // [26]    Use Next BD
  // [25]    Valid BD
  // [24:18] Lock Release Value (signed)
  // [16:13] Lock Release ID
  // [12]    Lock Acquire Enable
  // [11:5]  Lock Acquire Value (signed)
  // [3:0]   Lock Acquire ID

  uint32_t reg5 = (1u << 31) |  // TLAST suppress
                  (5u << 27) |   // Next BD = 5
                  (1u << 26) |   // Use next BD
                  (1u << 25) |   // Valid BD
                  (3 << 18) |    // Lock rel value = 3
                  (7 << 13) |    // Lock rel ID = 7
                  (1u << 12) |   // Lock acq enable
                  (2 << 5) |     // Lock acq value = 2
                  (4);           // Lock acq ID = 4

  bool tlast = BDFieldExtractor::getTlastSuppress(reg5);
  uint8_t nextBd = BDFieldExtractor::getNextBd(reg5);
  bool useNext = BDFieldExtractor::getUseNextBd(reg5);
  bool valid = BDFieldExtractor::getValidBd(reg5);
  int8_t lockRelVal = BDFieldExtractor::getLockRelValue(reg5);
  uint8_t lockRelId = BDFieldExtractor::getLockRelId(reg5);
  bool lockAcqEn = BDFieldExtractor::getLockAcqEnable(reg5);
  int8_t lockAcqVal = BDFieldExtractor::getLockAcqValue(reg5);
  uint8_t lockAcqId = BDFieldExtractor::getLockAcqId(reg5);

  if (!tlast) throw std::runtime_error("getTlastSuppress failed");
  if (nextBd != 5) throw std::runtime_error("getNextBd failed");
  if (!useNext) throw std::runtime_error("getUseNextBd failed");
  if (!valid) throw std::runtime_error("getValidBd failed");
  if (lockRelVal != 3) throw std::runtime_error("getLockRelValue failed");
  if (lockRelId != 7) throw std::runtime_error("getLockRelId failed");
  if (!lockAcqEn) throw std::runtime_error("getLockAcqEnable failed");
  if (lockAcqVal != 2) throw std::runtime_error("getLockAcqValue failed");
  if (lockAcqId != 4) throw std::runtime_error("getLockAcqId failed");

  std::cout << "  ✓ Register 5 field extraction works\n";
}

//===----------------------------------------------------------------------===//
// BDAddressParser Tests
//===----------------------------------------------------------------------===//

void test_bd_address_parser_compute() {
  std::cout << "Test: BDAddressParser - Compute Tile\n";

  BDAddressParser parser(1); // 1 memtile row

  // Compute tile BD registers: base 0x1D000 within tile memory
  // Tile (2, 3) base address: 0x300000 (col=2, row=3-1=2 compute)
  // BD 5, Register 2: 0x300000 + 0x1D000 + (5 * 0x20) + (2 * 4) = 0x31D0A8
  uint32_t addr = 0x300000 + 0x1D000 + (5 * 0x20) + (2 * 4);

  BDAddressInfo info = parser.parse(addr);

  if (!info.isBDRegister) {
    throw std::runtime_error("Failed to identify compute BD register");
  }
  if (info.column != 3) {
    throw std::runtime_error("Wrong column: got " + std::to_string(info.column));
  }
  if (info.row != 2) {
    throw std::runtime_error("Wrong row: got " + std::to_string(info.row));
  }
  if (info.tileType != TileType::Compute) {
    throw std::runtime_error("Wrong tile type");
  }
  if (info.bdIndex != 5) {
    throw std::runtime_error("Wrong BD index: got " + std::to_string(info.bdIndex));
  }
  if (info.regIndex != 2) {
    throw std::runtime_error("Wrong register index: got " + std::to_string(info.regIndex));
  }

  std::cout << "  ✓ Compute tile BD address parsing works\n";
}

void test_bd_address_parser_memtile() {
  std::cout << "Test: BDAddressParser - Memory Tile\n";

  BDAddressParser parser(1); // 1 memtile row

  // Memory tile BD registers: base 0xA0000 within tile memory
  // Tile (1, 1) = memtile at row 1: 0x100000 + 0xA0000 = 0x1A0000
  // BD 10, Register 3: 0x1A0000 + (10 * 0x20) + (3 * 4) = 0x1A014C
  uint32_t addr = 0x100000 + 0xA0000 + (10 * 0x20) + (3 * 4);

  BDAddressInfo info = parser.parse(addr);

  if (!info.isBDRegister) {
    throw std::runtime_error("Failed to identify memtile BD register");
  }
  if (info.column != 1) {
    throw std::runtime_error("Wrong column");
  }
  if (info.row != 1) {
    throw std::runtime_error("Wrong row");
  }
  if (info.tileType != TileType::MemoryTile) {
    throw std::runtime_error("Wrong tile type - expected MemoryTile");
  }
  if (info.bdIndex != 10) {
    throw std::runtime_error("Wrong BD index");
  }
  if (info.regIndex != 3) {
    throw std::runtime_error("Wrong register index");
  }

  std::cout << "  ✓ Memory tile BD address parsing works\n";
}

void test_bd_address_parser_non_bd() {
  std::cout << "Test: BDAddressParser - Non-BD Address\n";

  BDAddressParser parser(1);

  // Test a random address that's not a BD register
  uint32_t addr = 0x12345678;
  BDAddressInfo info = parser.parse(addr);

  if (info.isBDRegister) {
    throw std::runtime_error("Incorrectly identified non-BD address as BD");
  }

  std::cout << "  ✓ Non-BD address correctly rejected\n";
}

//===----------------------------------------------------------------------===//
// BDAccumulator Tests
//===----------------------------------------------------------------------===//

void test_bd_accumulator_single_bd() {
  std::cout << "Test: BDAccumulator - Single Complete BD\n";

  BDAccumulator accum;
  BDAddressParser parser(1);

  // Write all 6 registers for BD 0 in tile (1, 2)
  uint32_t baseAddr = 0x200000 + 0x1D000; // Tile (1,2) compute, BD region

  uint32_t regs[6] = {
    (0x100 << 14) | 256,    // Reg 0: base=0x100, length=256
    0,                       // Reg 1: no packet
    0,                       // Reg 2: D0=1, D1=1
    0,                       // Reg 3: no wrapping
    0,                       // Reg 4: no iteration
    (1u << 25)              // Reg 5: valid bit set
  };

  std::optional<ParsedBDConfig> result;
  for (int i = 0; i < 6; i++) {
    result = accum.addWrite(baseAddr + i * 4, regs[i], parser);
  }

  // Last write should complete the BD
  if (!result.has_value()) {
    throw std::runtime_error("BD not completed after all 6 writes");
  }

  ParsedBDConfig bd = *result;
  if (bd.column != 1) throw std::runtime_error("Wrong column");
  if (bd.row != 2) throw std::runtime_error("Wrong row");
  if (bd.bdIndex != 0) throw std::runtime_error("Wrong BD index");
  if (bd.baseAddress != 0x100) throw std::runtime_error("Wrong base address");
  if (bd.bufferLength != 256) throw std::runtime_error("Wrong buffer length");
  if (!bd.validBd) throw std::runtime_error("Valid bit not set");

  std::cout << "  ✓ Single BD accumulation works\n";
}

void test_bd_accumulator_multiple_bds() {
  std::cout << "Test: BDAccumulator - Multiple BDs\n";

  BDAccumulator accum;
  BDAddressParser parser(1);

  uint32_t tileBase = 0x100000 + 0x1D000; // Tile (1,2) compute

  // Write to BD 0 and BD 1
  uint32_t bd0Addr = tileBase;
  uint32_t bd1Addr = tileBase + 0x20;

  // Write partial data to both BDs (interleaved)
  accum.addWrite(bd0Addr + 0, 0x1000, parser);  // BD0 reg0
  accum.addWrite(bd1Addr + 0, 0x2000, parser);  // BD1 reg0
  accum.addWrite(bd0Addr + 4, 0, parser);       // BD0 reg1

  // Should have 2 pending BDs
  if (accum.pendingCount() != 2) {
    throw std::runtime_error("Expected 2 pending BDs, got " +
                            std::to_string(accum.pendingCount()));
  }

  std::cout << "  ✓ Multiple BD tracking works\n";
}

void test_bd_accumulator_flush() {
  std::cout << "Test: BDAccumulator - Flush Pending BDs\n";

  BDAccumulator accum;
  BDAddressParser parser(1);

  uint32_t baseAddr = 0x100000 + 0x1D000;

  // Write incomplete BD (only 3 registers)
  accum.addWrite(baseAddr + 0, 0x1000, parser);
  accum.addWrite(baseAddr + 4, 0, parser);
  accum.addWrite(baseAddr + 8, 0, parser);

  if (!accum.hasPending()) {
    throw std::runtime_error("Should have pending BD");
  }

  // Flush should return all pending BDs
  auto flushed = accum.flush();
  if (flushed.empty()) {
    throw std::runtime_error("Flush should return pending BDs");
  }

  if (accum.hasPending()) {
    throw std::runtime_error("Accumulator should be empty after flush");
  }

  std::cout << "  ✓ Flush pending BDs works\n";
}

//===----------------------------------------------------------------------===//
// Main Test Runner
//===----------------------------------------------------------------------===//

int main() {
  try {
    std::cout << "\n=== BD Lifting Unit Tests ===\n\n";

    // BDFieldExtractor tests
    test_extract_bits();
    test_extract_signed_bits();
    test_bd_reg0_fields();
    test_bd_reg1_fields();
    test_bd_reg2_fields();
    test_bd_reg3_fields();
    test_bd_reg4_fields();
    test_bd_reg5_fields();

    // BDAddressParser tests
    test_bd_address_parser_compute();
    test_bd_address_parser_memtile();
    test_bd_address_parser_non_bd();

    // BDAccumulator tests
    test_bd_accumulator_single_bd();
    test_bd_accumulator_multiple_bds();
    test_bd_accumulator_flush();

    std::cout << "\n✓ All BD lifting tests passed!\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n✗ Test failed: " << e.what() << "\n";
    return 1;
  }
}
