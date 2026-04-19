//===- test_load_pdi_patch.cpp - Unit test for LOAD_PDI patching -*- C++ -*-===//
//
// Tests parse_instr_offsets_json() and patch_load_pdi() functions.
//
//===----------------------------------------------------------------------===//

#include "test_utils.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

static void write_file(const std::string &path, const std::string &content) {
  std::ofstream f(path);
  f << content;
}

static void write_binary_file(const std::string &path, const uint8_t *data,
                              size_t size) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), size);
}

int main() {
  // --- Setup ---

  // LOAD_PDI binary layout (4 words = 16 bytes):
  //   word[0] at offset 0:  opcode | (pdi_id << 16)
  //   word[1] at offset 4:  PDI size in bytes (size field)
  //   word[2] at offset 8:  PDI address lower 32 bits (address field)
  //   word[3] at offset 12: PDI address upper 32 bits

  // Create a fake instruction buffer with a LOAD_PDI at word offset 4
  // (byte offset 16). Pre-fill with some recognizable pattern.
  std::vector<uint32_t> instr_v = {
      0xAAAAAAAA, // word 0: some other instruction
      0xBBBBBBBB, // word 1
      0xCCCCCCCC, // word 2
      0xDDDDDDDD, // word 3
      0x00010000, // word 4 (byte 16): LOAD_PDI opcode, pdi_id=1
      0x00000000, // word 5 (byte 20): size field
      0x00000000, // word 6 (byte 24): address field lower
      0x00000000, // word 7 (byte 28): address field upper
  };

  // The LOAD_PDI is at byte offset 16.
  // size_field is at byte offset 20 (word 5)
  // address_field is at byte offset 24 (word 6)

  // Create fake JSON offsets file
  std::string json_path = "/tmp/test_offsets.json";
  write_file(json_path, R"({
  "instructions": [
    {
      "type": "load_pdi",
      "offset_bytes": 16,
      "pdi_id": 1,
      "address_field_offset_bytes": 24,
      "size_field_offset_bytes": 20
    }
  ]
})");

  // Create a fake PDI binary file (13 bytes — not 4-aligned, to test padding)
  std::string pdi_path = "/tmp/test_pdi.bin";
  uint8_t pdi_data[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D};
  write_binary_file(pdi_path, pdi_data, sizeof(pdi_data));

  // --- Test parse_instr_offsets_json ---
  auto patch_infos = test_utils::parse_instr_offsets_json(json_path);
  assert(patch_infos.size() == 1);
  assert(patch_infos[0].load_pdi_offset_bytes == 16);
  assert(patch_infos[0].pdi_id == 1);
  assert(patch_infos[0].address_field_offset_bytes == 24);
  assert(patch_infos[0].size_field_offset_bytes == 20);
  std::cout << "PASS: parse_instr_offsets_json\n";

  // --- Test patch_load_pdi ---
  size_t orig_size = instr_v.size(); // 8 words = 32 bytes
  uint32_t expected_pdi_offset = static_cast<uint32_t>(orig_size * 4); // 32

  test_utils::patch_load_pdi(instr_v, patch_infos, pdi_path);

  // PDI is 13 bytes, padded to 16 bytes = 4 words
  size_t expected_total = orig_size + 4; // 12 words
  assert(instr_v.size() == expected_total);

  // Check address field was patched (word 6 = address lower, word 7 = upper)
  assert(instr_v[6] == expected_pdi_offset);
  assert(instr_v[7] == 0);

  // Check size field was patched (word 5 = pdi size in bytes)
  assert(instr_v[5] == 13); // original unpadded size

  // Check PDI data was appended correctly (first 4 bytes as uint32_t)
  uint32_t first_pdi_word;
  std::memcpy(&first_pdi_word, pdi_data, 4);
  assert(instr_v[orig_size] == first_pdi_word);

  // Check the padding word (last word should have 0x0D in lowest byte, rest 0)
  uint8_t last_word_bytes[4];
  std::memcpy(last_word_bytes, &instr_v[orig_size + 3], 4);
  assert(last_word_bytes[0] == 0x0D);
  assert(last_word_bytes[1] == 0x00);
  assert(last_word_bytes[2] == 0x00);
  assert(last_word_bytes[3] == 0x00);

  std::cout << "PASS: patch_load_pdi\n";

  // --- Test with multiple LOAD_PDI entries ---
  std::string json_path2 = "/tmp/test_offsets2.json";
  write_file(json_path2, R"({
  "instructions": [
    {
      "type": "load_pdi",
      "offset_bytes": 0,
      "pdi_id": 0,
      "address_field_offset_bytes": 8,
      "size_field_offset_bytes": 4
    },
    {
      "type": "write32",
      "offset_bytes": 16,
      "value_field_offset_bytes": 20
    },
    {
      "type": "load_pdi",
      "offset_bytes": 24,
      "pdi_id": 1,
      "address_field_offset_bytes": 32,
      "size_field_offset_bytes": 28
    }
  ]
})");

  auto infos2 = test_utils::parse_instr_offsets_json(json_path2);
  // Should only get 2 load_pdi entries, ignoring the write32
  assert(infos2.size() == 2);
  assert(infos2[0].pdi_id == 0);
  assert(infos2[0].address_field_offset_bytes == 8);
  assert(infos2[1].pdi_id == 1);
  assert(infos2[1].address_field_offset_bytes == 32);
  std::cout << "PASS: parse with mixed types (load_pdi + write32)\n";

  // === Write32 / RTP patching tests ===

  // --- Test parse_write32_offsets_json ---
  std::string json_path3 = "/tmp/test_write32_offsets.json";
  write_file(json_path3, R"({
  "instructions": [
    {
      "type": "load_pdi",
      "offset_bytes": 0,
      "pdi_id": 0,
      "address_field_offset_bytes": 8,
      "size_field_offset_bytes": 4
    },
    {
      "type": "write32",
      "offset_bytes": 16,
      "value_field_offset_bytes": 20,
      "name": "rtp_param_A"
    },
    {
      "type": "write32",
      "offset_bytes": 24,
      "value_field_offset_bytes": 28,
      "name": "rtp_param_B"
    },
    {
      "type": "write32",
      "offset_bytes": 32,
      "value_field_offset_bytes": 36
    }
  ]
})");

  auto w32_infos = test_utils::parse_write32_offsets_json(json_path3);
  assert(w32_infos.size() == 3);
  assert(w32_infos[0].name == "rtp_param_A");
  assert(w32_infos[0].offset_bytes == 16);
  assert(w32_infos[0].value_field_offset_bytes == 20);
  assert(w32_infos[1].name == "rtp_param_B");
  assert(w32_infos[1].offset_bytes == 24);
  assert(w32_infos[1].value_field_offset_bytes == 28);
  // Third entry has no name — should parse with empty name
  assert(w32_infos[2].name.empty());
  assert(w32_infos[2].offset_bytes == 32);
  assert(w32_infos[2].value_field_offset_bytes == 36);
  std::cout << "PASS: parse_write32_offsets_json\n";

  // --- Test patch_rtp ---
  // Build a 10-word instruction vector with known values
  std::vector<uint32_t> rtp_instr(10, 0);
  rtp_instr[5] = 0xDEADBEEF; // word at byte offset 20 (value field for A)
  rtp_instr[7] = 0xCAFEBABE; // word at byte offset 28 (value field for B)

  test_utils::patch_rtp(rtp_instr, w32_infos, "rtp_param_A", 42);
  assert(rtp_instr[5] == 42);
  // B should be untouched
  assert(rtp_instr[7] == 0xCAFEBABE);

  test_utils::patch_rtp(rtp_instr, w32_infos, "rtp_param_B", 99);
  assert(rtp_instr[7] == 99);
  // A should still be 42
  assert(rtp_instr[5] == 42);
  std::cout << "PASS: patch_rtp\n";

  // --- Test patch_rtp with unknown name throws ---
  bool caught = false;
  try {
    test_utils::patch_rtp(rtp_instr, w32_infos, "nonexistent", 0);
  } catch (const std::runtime_error &) {
    caught = true;
  }
  assert(caught);
  std::cout << "PASS: patch_rtp throws on unknown name\n";

  // === Address Patch tests ===

  // --- Test parse_address_patch_offsets_json ---
  std::string json_path4 = "/tmp/test_addr_patch.json";
  uint32_t bd_mmio_base = 0x000A0000;
  uint64_t bd_addr_reg = bd_mmio_base + 0x4; // Buffer address at BD Word[1]
  write_file(json_path4, R"({
  "instructions": [
    {
      "type": "address_patch",
      "offset_bytes": 88,
      "arg_idx": 2,
      "arg_plus": 256,
      "addr": )" + std::to_string(bd_addr_reg) +
                             R"(
    }
  ]
})");

  auto addr_patches =
      test_utils::parse_address_patch_offsets_json(json_path4);
  assert(addr_patches.size() == 1);
  assert(addr_patches[0].offset_bytes == 88);
  assert(addr_patches[0].arg_idx == 2);
  assert(addr_patches[0].arg_plus == 256);
  assert(addr_patches[0].addr == bd_addr_reg);
  std::cout << "PASS: parse_address_patch_offsets_json\n";

  // --- Test patch_addresses with txn header ---
  // Build a transaction binary: txn header + WRITE + BLOCKWRITE with 8-word BD
  std::vector<uint32_t> addr_instr = {
      // Txn header (4 words)
      0x06030100, // version 0.1, device 6, num_cols 3
      0x00000000, // reserved
      0x00000002, // 2 operations
      0x00000058, // total size: 88 bytes = 22 words * 4

      // WRITE instruction (6 words)
      0x00000000, // opcode: WRITE
      0x00000000, // col/row
      0x00010000, // MMIO address
      0x00000018, // size: 24 bytes
      0x12345678, // value
      0x00000000, // unused

      // BLOCKWRITE instruction (12 words = 4 header + 8 BD payload)
      0x00000001,  // opcode: BLOCKWRITE
      0x00000000,  // col=0, row=0
      bd_mmio_base, // MMIO address of BD
      0x00000030,  // size: 48 bytes = 12 words
      // BD payload (8 words):
      0x00001000, // BD Word 0: Buffer_Length
      0x00000000, // BD Word 1: Buffer_Offset (lo addr - to be patched)
      0x00000000, // BD Word 2: (hi addr - to be patched)
      0x00000000, // BD Word 3
      0x00000000, // BD Word 4
      0x00000000, // BD Word 5
      0x00000000, // BD Word 6
      0x02000000, // BD Word 7: valid_bd
  };

  // arg_idx 2 -> buffer at 0x0001DEAD0000
  std::map<uint32_t, uint64_t> arg_addrs;
  arg_addrs[2] = 0x0001DEAD0000ULL;

  test_utils::patch_addresses(addr_instr, addr_patches, arg_addrs);

  // full_addr = 0x0001DEAD0000 + 256 = 0x0001DEAD0100
  // BD Word 1 at index 15: lo = 0xDEAD0100
  // BD Word 2 at index 16: hi = 0x00000001
  assert(addr_instr[15] == 0xDEAD0100);
  assert(addr_instr[16] == 0x00000001);
  // Verify surrounding words unchanged
  assert(addr_instr[14] == 0x00001000); // BD Word 0
  assert(addr_instr[21] == 0x02000000); // BD Word 7
  std::cout << "PASS: patch_addresses (with txn header)\n";

  // --- Test patch_addresses without txn header (raw instructions) ---
  uint32_t bd_mmio_base2 = 0x000B0000;
  uint64_t bd_addr_reg2 = bd_mmio_base2 + 0x4;
  std::vector<uint32_t> raw_instr = {
      // BLOCKWRITE instruction (12 words)
      0x00000001,   // opcode: BLOCKWRITE
      0x00000100,   // col=0, row=1
      bd_mmio_base2, // MMIO address
      0x00000030,   // size: 48 bytes
      // BD payload (8 words):
      0x00002000, // BD Word 0: Buffer_Length
      0x00000000, // BD Word 1: to be patched (lo)
      0x00000000, // BD Word 2: to be patched (hi)
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x02000000, // BD Word 7: valid_bd
  };

  test_utils::AddressPatchInfo raw_patch;
  raw_patch.offset_bytes = 48;
  raw_patch.arg_idx = 0;
  raw_patch.arg_plus = 0;
  raw_patch.addr = bd_addr_reg2;

  std::map<uint32_t, uint64_t> raw_args;
  raw_args[0] = 0xABCD00000000ULL;

  test_utils::patch_addresses(raw_instr, {raw_patch}, raw_args);

  // full_addr = 0xABCD00000000 + 0 = 0xABCD00000000
  // lo = 0x00000000, hi = 0x0000ABCD
  assert(raw_instr[5] == 0x00000000);
  assert(raw_instr[6] == 0x0000ABCD);
  assert(raw_instr[4] == 0x00002000); // BD Word 0 unchanged
  std::cout << "PASS: patch_addresses (no txn header)\n";

  // --- Test patch_addresses with multiple patches ---
  uint32_t bd_mmio_base3 = 0x000C0000;
  uint32_t bd_mmio_base4 = 0x000C0020; // Second BD, offset 32 bytes
  std::vector<uint32_t> multi_instr = {
      // BLOCKWRITE 1 (12 words)
      0x00000001, 0x00000000, bd_mmio_base3, 0x00000030,
      0x00001000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x02000000,
      // BLOCKWRITE 2 (12 words)
      0x00000001, 0x00000000, bd_mmio_base4, 0x00000030,
      0x00002000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x02000000,
  };

  std::vector<test_utils::AddressPatchInfo> multi_patches = {
      {0, 0, 0x100, bd_mmio_base3 + 0x4},
      {0, 3, 0x200, bd_mmio_base4 + 0x4},
  };
  std::map<uint32_t, uint64_t> multi_args;
  multi_args[0] = 0x10000000ULL;
  multi_args[3] = 0x20000000ULL;

  test_utils::patch_addresses(multi_instr, multi_patches, multi_args);

  // Patch 1: full_addr = 0x10000000 + 0x100 = 0x10000100
  assert(multi_instr[5] == 0x10000100);
  assert(multi_instr[6] == 0x00000000);
  // Patch 2: full_addr = 0x20000000 + 0x200 = 0x20000200
  assert(multi_instr[17] == 0x20000200);
  assert(multi_instr[18] == 0x00000000);
  std::cout << "PASS: patch_addresses (multiple patches)\n";

  // --- Test patch_addresses throws on missing arg_idx ---
  bool caught2 = false;
  try {
    std::map<uint32_t, uint64_t> empty_args;
    test_utils::patch_addresses(multi_instr, multi_patches, empty_args);
  } catch (const std::runtime_error &) {
    caught2 = true;
  }
  assert(caught2);
  std::cout << "PASS: patch_addresses throws on missing arg_idx\n";

  // --- Test parse ignores non-address_patch entries ---
  std::string json_path5 = "/tmp/test_addr_mixed.json";
  write_file(json_path5, R"({
  "instructions": [
    {
      "type": "load_pdi",
      "offset_bytes": 0,
      "pdi_id": 0,
      "address_field_offset_bytes": 8,
      "size_field_offset_bytes": 4
    },
    {
      "type": "address_patch",
      "offset_bytes": 16,
      "arg_idx": 1,
      "arg_plus": 64,
      "addr": 655364
    },
    {
      "type": "write32",
      "offset_bytes": 32,
      "value_field_offset_bytes": 36
    }
  ]
})");
  auto mixed_patches =
      test_utils::parse_address_patch_offsets_json(json_path5);
  assert(mixed_patches.size() == 1);
  assert(mixed_patches[0].arg_idx == 1);
  assert(mixed_patches[0].arg_plus == 64);
  std::cout << "PASS: parse_address_patch_offsets_json ignores non-address_patch\n";

  // Cleanup
  std::remove(json_path.c_str());
  std::remove(json_path2.c_str());
  std::remove(pdi_path.c_str());
  std::remove(json_path3.c_str());
  std::remove(json_path4.c_str());
  std::remove(json_path5.c_str());

  std::cout << "ALL TESTS PASSED\n";
  return 0;
}
