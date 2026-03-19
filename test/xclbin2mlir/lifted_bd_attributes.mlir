// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This test verifies that aie.dma_bd operations include proper attributes
// for advanced BD features like:
// - Multi-dimensional addressing (dimensions attribute)
// - Lock acquire/release (lock_acq_id, lock_acq_val, lock_rel_id, lock_rel_val)
// - BD chaining (next_bd attribute)
// - Packet mode (packet_type, packet_id)
// - Iteration (iteration attributes)

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Test that dma_bd operations can have dimension attributes
// The syntax should be: dimensions = [#aie.dma_dim<...>, ...]
// CHECK-DAG: aie.dma_bd({{.*}}) {
// CHECK-SAME: dimensions = [
// CHECK-SAME: #aie.dma_dim<

// Alternative: Check for individual dimension parameters
// stepsize and wrap are the key dimension attributes
// CHECK-DAG: stepsize = {{[0-9]+}}
// CHECK-DAG: wrap = {{[0-9]+}}

// Test that lock acquire attributes are emitted
// CHECK-DAG: lock_acq_id = {{[0-9]+}}
// CHECK-DAG: lock_acq_val = {{-?[0-9]+}}

// Test that lock release attributes are emitted
// CHECK-DAG: lock_rel_id = {{[0-9]+}}
// CHECK-DAG: lock_rel_val = {{-?[0-9]+}}

// Test that next_bd chaining is emitted
// CHECK-DAG: next_bd = {{[0-9]+}}

// Test that valid_bd flag is emitted
// CHECK-DAG: valid_bd = {{true|false}}

// Test packet mode attributes (if present)
// CHECK-DAG: enable_packet = {{true|false}}

// Test iteration attributes (if present)
// CHECK-DAG: iteration_stepsize = {{[0-9]+}}
// CHECK-DAG: iteration_wrap = {{[0-9]+}}
