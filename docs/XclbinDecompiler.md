# Xclbin Decompiler User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Use Cases](#use-cases)
3. [Command-Line Usage](#command-line-usage)
4. [Understanding Raw Mode Output](#understanding-raw-mode-output)
5. [Understanding Lifted Mode Output](#understanding-lifted-mode-output)
6. [Interpreting DMA Buffer Descriptor Configurations](#interpreting-dma-buffer-descriptor-configurations)
7. [Interpreting Switchbox Routing](#interpreting-switchbox-routing)
8. [Complete Example Walkthrough](#complete-example-walkthrough)
9. [Known Limitations](#known-limitations)
10. [Testing and Verification](#testing-and-verification)

---

## Introduction

The xclbin decompiler is a tool that converts compiled AIE binary files (`.xclbin` format) back into human-readable MLIR (Multi-Level Intermediate Representation). This reverse-engineering capability enables developers to understand, debug, and verify how their AI Engine designs were compiled and configured.

The decompiler supports two modes:
- **Raw mode** (default): Emits low-level register write operations exactly as they appear in the binary
- **Lifted mode**: Reconstructs high-level AIE operations from register writes, making the output significantly more readable and understandable

## Use Cases

The xclbin decompiler is valuable for several scenarios:

### 1. Debugging Compiled Designs
When your compiled AIE design doesn't behave as expected, the decompiler helps you:
- Verify that buffers are allocated at the correct addresses
- Check that DMA buffer descriptors have the expected dimensions and strides
- Confirm lock configurations for proper synchronization
- Validate switchbox routing paths

### 2. Reverse Engineering
For understanding existing compiled designs:
- Learn how efficient designs structure their data movement
- Understand routing patterns and resource allocation
- Study lock synchronization strategies
- Extract configuration parameters from production binaries

### 3. Understanding AIE Configuration
As an educational tool:
- See the direct mapping between high-level AIE operations and hardware registers
- Understand how MLIR AIE dialect operations translate to hardware configuration
- Learn about the AIE architecture through working examples
- Validate your mental model of AIE resource allocation

### 4. Verification and Testing
For quality assurance:
- Verify that compiler optimizations didn't introduce errors
- Compare different compilation strategies
- Ensure deterministic compilation results
- Validate that resource constraints are respected

---

## Command-Line Usage

The decompiler is invoked through the `aie-translate` tool with the `--xclbin-to-mlir` flag.

### Basic Usage (Raw Mode)

```bash
aie-translate --xclbin-to-mlir input.xclbin
```

This produces raw register write operations. Output is printed to stdout.

**Example:**
```bash
aie-translate --xclbin-to-mlir add_blockwrite.xclbin > output_raw.mlir
```

### Lifted Mode

To generate more readable, high-level operations:

```bash
aie-translate --xclbin-to-mlir --emit-lifted input.xclbin
```

**Example:**
```bash
aie-translate --xclbin-to-mlir --emit-lifted add_blockwrite.xclbin > output_lifted.mlir
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--xclbin-to-mlir` | Enable xclbin to MLIR translation | Required |
| `--emit-lifted` | Emit high-level operations instead of raw register writes | false (off) |

### Typical Workflow

```bash
# Step 1: Decompile in raw mode to see low-level details
aie-translate --xclbin-to-mlir design.xclbin > design_raw.mlir

# Step 2: Decompile in lifted mode for easier understanding
aie-translate --xclbin-to-mlir --emit-lifted design.xclbin > design_lifted.mlir

# Step 3: Compare or analyze the outputs
diff design_raw.mlir design_lifted.mlir
# Or use your preferred editor/viewer
less design_lifted.mlir
```

---

## Understanding Raw Mode Output

Raw mode emits low-level hardware register operations directly from the compiled binary. This mode preserves exact configuration details but requires understanding of AIE hardware registers.

### Output Structure

Raw mode output contains:

1. **Module and Device Declaration**: The top-level structure
2. **Block Write Data**: Constant arrays containing initialization data
3. **Runtime Sequence**: Low-level write operations

### Operation Types

#### 1. `aiex.npu.write32`
Single 32-bit register write operation.

**Format:**
```mlir
aiex.npu.write32 { address = <addr> : ui32, value = <val> : ui32 }
```

**Example:**
```mlir
aiex.npu.write32 { address = 469827584 : ui32, value = 42 : ui32 }
```

- `address`: The hardware register address (32-bit unsigned integer)
- `value`: The 32-bit value to write to the register

**Use case:** Single register configuration, flag settings, simple values

#### 2. `aiex.npu.maskwrite32`
Masked register write - only modifies specific bits within a register.

**Format:**
```mlir
aiex.npu.maskwrite32 { address = <addr> : ui32, value = <val> : ui32, mask = <mask> : ui32 }
```

**Example:**
```mlir
aiex.npu.maskwrite32 { address = 469827588 : ui32, value = 256 : ui32, mask = 65535 : ui32 }
```

- `address`: The target register address
- `value`: The value to write
- `mask`: Bit mask indicating which bits to modify (1 = modify, 0 = preserve)

**Use case:** Updating specific fields within a control register without affecting other fields

#### 3. `aiex.npu.blockwrite`
Block write operation - efficiently writes multiple consecutive values from a data buffer.

**Format:**
```mlir
aiex.npu.blockwrite(%buffer) { address = <addr> : ui32 }
```

**Example:**
```mlir
memref.global "private" constant @cdo_blockwrite_0 : memref<32xi32> = dense<[...]>
%buffer = memref.get_global @cdo_blockwrite_0 : memref<32xi32>
aiex.npu.blockwrite(%buffer) { address = 469827600 : ui32 }
```

**Use case:** Initializing arrays, buffer descriptor registers, configuration tables

### Complete Raw Mode Example

```mlir
module {
  aie.device(npu1_1col) {
    // Device structure (empty in raw mode - operations are in runtime sequence)
  }

  // Data for block writes
  memref.global "private" constant @cdo_blockwrite_0 : memref<32xi32> =
    dense<[0x00001000, 0x00000020, 0x00000001, ...]>

  aiex.runtime_sequence {
    // Configure a single register
    aiex.npu.write32 { address = 469827584 : ui32, value = 1 : ui32 }

    // Masked update to a control register
    aiex.npu.maskwrite32 {
      address = 469827588 : ui32,
      value = 256 : ui32,
      mask = 65535 : ui32
    }

    // Block write for initializing multiple registers
    %0 = memref.get_global @cdo_blockwrite_0 : memref<32xi32>
    aiex.npu.blockwrite(%0) { address = 469827600 : ui32 }
  }
}
```

### Interpreting Register Addresses

Register addresses in raw mode are absolute hardware addresses. Understanding them requires:
- Knowledge of the AIE memory map
- Device-specific register documentation
- Understanding of tile indexing and offsets

**Tip:** Use lifted mode to avoid manual address decoding. The lifted mode automatically interprets these addresses and generates corresponding high-level operations.

---

## Understanding Lifted Mode Output

Lifted mode reconstructs high-level AIE operations from raw register writes, making the output much more readable and semantically meaningful. This mode is recommended for most debugging and understanding tasks.

### Output Structure

Lifted mode output contains:

1. **Module and Device Declaration**: Same as raw mode
2. **AIE Resource Declarations**: Tiles, buffers, locks
3. **AIE Configuration Operations**: DMA BDs, switchbox routing
4. **Runtime Sequence**: Remaining operations that couldn't be lifted

### High-Level Operation Types

#### 1. `aie.tile`
Declares a tile in the AIE array.

**Format:**
```mlir
%tile = aie.tile(%col, %row)
```

**Example:**
```mlir
%tile_0_2 = aie.tile(0, 2)
```

- `%col`: Column index (0-based)
- `%row`: Row index (0-based)
- Returns a tile reference used by other operations

**Interpretation:** This identifies a specific compute or memory tile in the AIE spatial array.

#### 2. `aie.buffer`
Declares a buffer (memory allocation) within a tile.

**Format:**
```mlir
%buffer = aie.buffer(%tile) : memref<SIZE x TYPE>
```

**Example:**
```mlir
%buffer_0 = aie.buffer(%tile_0_2) : memref<1024xi32>
```

- `%tile`: The tile where this buffer is allocated
- `memref<SIZE x TYPE>`: Memory reference type describing size and element type

**Interpretation:** Allocates 1024 32-bit integers in the local memory of tile (0,2).

#### 3. `aie.lock` and `aie.use_lock`
Declares a hardware lock for synchronization and emits lock acquire/release operations.

**Format:**
```mlir
%lock = aie.lock(%tile, LOCK_ID)

// Inside aie.mem block:
aie.use_lock(%lock, Acquire, VALUE)
aie.dma_bd(...)
aie.use_lock(%lock, Release, VALUE)
```

**Example:**
```mlir
%lock_0_2_0 = aie.lock(%tile_0_2, 0)
%mem_0_2 = aie.mem(%tile_0_2) {
  aie.use_lock(%lock_0_2_0, Acquire, 1)
  aie.dma_bd(%buffer : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
  aie.use_lock(%lock_0_2_0, Release, 1)
  aie.end
}
```

- `%tile`: The tile containing this lock
- `LOCK_ID`: Hardware lock identifier (typically 0-15)
- `Acquire`/`AcquireGreaterEqual`: Lock acquire action
- `Release`: Lock release action
- `VALUE`: Lock value to acquire/release

**Note:** Lock operations are only emitted when buffer descriptors in the xclbin have lock acquire or release configured. Many simple designs may not use locks, so you may not see `aie.lock` or `aie.use_lock` operations in the output.

#### 4. `aie.mem` and `aie.dma_bd`
Defines DMA memory operations containing Buffer Descriptors.

**Format:**
```mlir
%mem = aie.mem(%tile) {
  // Basic block containing BD
  aie.dma_bd(%buffer : memref<SIZE x TYPE>, OFFSET, LENGTH) {bd_id = ID : i32}
  aie.end  // or aie.next_bd ^next_block
^next_block:
  aie.dma_bd(%buffer2 : memref<SIZE x TYPE>, OFFSET, LENGTH) {bd_id = ID : i32}
  aie.end
}
```

**Example:**
```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  // BD 0
  aie.dma_bd(%buffer_0 : memref<1024xi32>, 0, 256) {bd_id = 0 : i32}
  aie.end
^bb1:
  // BD 1 with next_bd chaining
  aie.dma_bd(%buffer_1 : memref<1024xi32>, 0, 256) {bd_id = 1 : i32}
  aie.next_bd ^bb2  // Chain to next BD
^bb2:
  // BD 2 (end of chain)
  aie.dma_bd(%buffer_2 : memref<1024xi32>, 0, 256) {bd_id = 2 : i32}
  aie.end
}
```

**Structure:**
- `aie.mem` contains the DMA memory region for a tile
- Each `aie.dma_bd` operation is in its own basic block
- Blocks terminate with either `aie.end` (no chaining) or `aie.next_bd ^block` (chain to next)
- `bd_id` attribute identifies the hardware buffer descriptor number

**Note:** In current implementation, incomplete BD configurations may show `memref<0xi32>` due to missing register values in the xclbin.

#### 5. `aie.switchbox`
Defines routing configuration for a tile's switchbox.

**Format:**
```mlir
aie.switchbox(%tile) {
  aie.connect<SOURCE_BUNDLE : CHANNEL, DEST_BUNDLE : CHANNEL>
  ...
}
```

**Example:**
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<DMA : 0, South : 0>
  aie.connect<North : 1, DMA : 1>
}
```

**Interpretation:** Configures the switchbox routing matrix. Each `aie.connect` establishes a connection from a source port to a destination port.

#### 6. `aie.connect`
Defines a single routing connection within a switchbox.

**Format:**
```mlir
aie.connect<SOURCE_BUNDLE : CHANNEL, DEST_BUNDLE : CHANNEL>
```

**Common Bundle Types:**
- `DMA`: DMA engine ports
- `North`, `South`, `East`, `West`: Inter-tile stream connections
- `Core`: AI Engine core connections
- `FIFO`: FIFO buffer connections
- `NOC`: Network-on-Chip connections

### Complete Lifted Mode Example

```mlir
module {
  aie.device(npu1_1col) {
    // Declare tiles
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // Declare buffers
    %buffer_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1024xi32>
    %buffer_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1024xi32>

    // DMA memory block with buffer descriptors
    %mem_0_2 = aie.mem(%tile_0_2) {
      // BD 0 (chains to BD 1)
      aie.dma_bd(%buffer_0 : memref<1024xi32>, 0, 256) {bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb1:
      // BD 1 (end of chain)
      aie.dma_bd(%buffer_1 : memref<1024xi32>, 0, 256) {bd_id = 1 : i32}
      aie.end
    }

    // Configure switchbox routing (if present in xclbin)
    aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<North : 0, DMA : 1>
    }

    aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, North : 0>
    }
  }

  aiex.runtime_sequence @configure() {
    // Operations that couldn't be lifted remain here as raw NPU writes
    // This includes shim tile configuration, control registers, etc.
    aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
    aie.end
  }
}
```

---

## Interpreting DMA Buffer Descriptor Configurations

DMA Buffer Descriptors (BDs) are the heart of AIE data movement. They describe how data should be transferred, including addressing patterns, synchronization, and chaining.

### Basic BD Structure

A minimal buffer descriptor specifies:
```mlir
aie.dma_bd(%buffer : memref<SIZE x TYPE>, OFFSET, LENGTH)
```

**Example:**
```mlir
aie.dma_bd(%my_buffer : memref<2048xi32>, 0, 512)
```
**Interpretation:** Transfer 512 elements (32-bit integers) starting from offset 0 in `my_buffer`.

### Buffer Addressing

#### Linear Addressing
The simplest case - sequential access through memory:
```mlir
aie.dma_bd(%buffer : memref<1024xi32>, 64, 256)
```
- Starts at element 64
- Reads/writes 256 consecutive elements
- Addresses: buffer[64], buffer[65], ..., buffer[319]

#### Multi-Dimensional Addressing
For accessing non-contiguous patterns (e.g., sub-matrices, strided access):

```mlir
aie.dma_bd(%buffer : memref<1024xi32>, 0, 256) {
  dimensions = [
    #aie.dma_dim<stepsize = 4, wrap = 16>,
    #aie.dma_dim<stepsize = 64, wrap = 4>
  ]
}
```

**How to interpret dimensions:**
- Each `#aie.dma_dim` describes one dimension of access
- `stepsize`: The stride between elements in this dimension
- `wrap`: How many steps before wrapping to next dimension

**Example interpretation (2D access):**
1. **Inner dimension** `stepsize=4, wrap=16`:
   - Take 16 steps of size 4
   - Accesses: 0, 4, 8, 12, ..., 60
2. **Outer dimension** `stepsize=64, wrap=4`:
   - After 16 inner steps, jump by 64
   - Repeat 4 times
   - Accesses rows at offsets: 0, 64, 128, 192

**Complete access pattern:**
```
Row 0: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60
Row 1: 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124
Row 2: 128, 132, 136, ...
Row 3: 192, 196, 200, ...
```

**Use case:** Accessing a 4x16 sub-matrix with element size of 4 bytes

### Lock Synchronization

Locks coordinate access between producers and consumers.

#### Lock Acquire
Before accessing data:
```mlir
lock_acq_id = 0      // Which lock to acquire
lock_acq_val = 1     // Required lock value to proceed
```

**Interpretation:**
- Wait until lock #0 has value 1
- This typically means "data is ready" (producer has filled the buffer)

#### Lock Release
After accessing data:
```mlir
lock_rel_id = 0      // Which lock to release
lock_rel_val = 0     // Value to set lock to
```

**Interpretation:**
- Set lock #0 to value 0 when transfer completes
- This typically means "buffer is available" (consumer has read the data)

#### Complete Producer-Consumer Example

**Producer BD (writes data, releases as "full"):**
```mlir
aie.dma_bd(%buffer : memref<1024xi32>, 0, 256) {
  lock_acq_id = 0
  lock_acq_val = 0     // Acquire when empty
  lock_rel_id = 0
  lock_rel_val = 1     // Release as full
}
```

**Consumer BD (reads data, releases as "empty"):**
```mlir
aie.dma_bd(%buffer : memref<1024xi32>, 0, 256) {
  lock_acq_id = 0
  lock_acq_val = 1     // Acquire when full
  lock_rel_id = 0
  lock_rel_val = 0     // Release as empty
}
```

### BD Chaining

Multiple BDs can be chained for complex transfer sequences using `aie.next_bd`:

```mlir
%mem = aie.mem(%tile_0_2) {
  // BD 0 - chains to BD 1
  aie.dma_bd(%buffer_a : memref<512xi32>, 0, 128) {bd_id = 0 : i32}
  aie.next_bd ^bb1
^bb1:
  // BD 1 - chains to BD 2
  aie.dma_bd(%buffer_b : memref<512xi32>, 0, 128) {bd_id = 1 : i32}
  aie.next_bd ^bb2
^bb2:
  // BD 2 - end of chain
  aie.dma_bd(%buffer_c : memref<512xi32>, 0, 128) {bd_id = 2 : i32}
  aie.end
}
```

**Interpretation:**
- Execute BD 0, then automatically proceed to BD 1, then BD 2
- `aie.next_bd ^block` chains to the next buffer descriptor
- `aie.end` terminates the chain
- Enables complex transfer patterns without CPU intervention
- Can create circular chains by having the last BD use `aie.next_bd` to point back to an earlier block

### Advanced BD Attributes

#### Iteration Controls
For repeated transfer patterns:
```mlir
iteration_stepsize = 1024
iteration_wrap = 8
```
**Interpretation:** Repeat this transfer 8 times, advancing the base address by 1024 elements each iteration.

#### Packet Mode
For packet-switched routing:
```mlir
enable_packet = true
packet_type = 0
packet_id = 5
```
**Interpretation:** Transfer uses packet ID 5, enabling selective routing in the switchbox.

#### Validity Flag
```mlir
valid_bd = true
```
**Interpretation:** This BD is active and will be executed. BDs with `valid_bd = false` are skipped.

### Complete DMA BD Example

The following shows a conceptual BD with advanced features. Lock operations are lifted to `aie.use_lock` operations, while dimension attributes are included when the BD has multi-dimensional addressing configured:

```mlir
%mem = aie.mem(%tile_0_2) {
  // BD 0: Transfer a 16x16 tile from a larger 256x256 matrix
  // In actual decompiled output, this would show as:
  aie.dma_bd(%matrix : memref<65536xi32>, 0, 256) {bd_id = 0 : i32}
  aie.next_bd ^bb1  // Chain to next BD
^bb1:
  // BD 1: Next transfer in chain
  aie.dma_bd(%matrix2 : memref<65536xi32>, 0, 256) {bd_id = 1 : i32}
  aie.end  // End of chain
}
```

**Note:** Lock operations are lifted to `aie.use_lock` operations within the `aie.mem` block. Dimension attributes are included when the BD has multi-dimensional addressing configured. Iteration controls are present in the raw BD register writes but may not be fully represented as operation attributes.
- Look for writes to DMA_BDx_2, DMA_BDx_3 (dimensions), and DMA_BDx_5 (locks) registers

---

## Interpreting Switchbox Routing

Switchboxes are programmable crossbar switches that route streaming data between AIE tiles and external interfaces. Understanding switchbox configuration is crucial for data flow analysis.

### Switchbox Basics

Each tile has a switchbox with multiple ports organized into "bundles":
- **DMA**: Ports connected to the tile's DMA engines
- **Core**: Ports connected to the AI Engine core
- **FIFO**: Ports connected to stream FIFO buffers
- **North, South, East, West**: Ports for inter-tile streaming
- **NOC**: Ports connected to the Network-on-Chip

### Connection Syntax

```mlir
aie.connect<SOURCE_BUNDLE : SOURCE_CHANNEL, DEST_BUNDLE : DEST_CHANNEL>
```

**Example:**
```mlir
aie.connect<DMA : 0, South : 2>
```
**Interpretation:** Connect DMA port 0 (source) to South port 2 (destination). Data from the DMA will stream southward on channel 2.

### Port Types and Bundles

#### DMA Bundle
Connects to DMA channels:
```mlir
aie.connect<North : 0, DMA : 1>
```
**Interpretation:** Route incoming data from northern tile into DMA channel 1 (typically for writing to memory).

#### Cardinal Direction Bundles (North, South, East, West)
Inter-tile streaming:
```mlir
aie.connect<West : 0, East : 0>
```
**Interpretation:** Forward data from western neighbor to eastern neighbor (relay pattern).

#### Core Bundle
Connect to/from the AI Engine core:
```mlir
aie.connect<DMA : 0, Core : 0>
```
**Interpretation:** Stream data from DMA to the compute core for processing.

#### NOC Bundle
Connect to the Network-on-Chip for off-chip memory access:
```mlir
aie.connect<DMA : 0, NOC : 0>
```
**Interpretation:** Stream data from DMA to external memory via NOC.

### Data Flow Patterns

#### Simple Pass-Through
Forward data from one tile to another:
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<South : 0, North : 0>
}
```
**Flow:** Data from tile (0,1) → tile (0,2) → tile (0,3)

#### DMA to Stream
Move data from local memory to streaming network:
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<DMA : 0, East : 1>
}
```
**Flow:** Local DMA reads buffer → streams eastward on channel 1

#### Stream to DMA
Receive streaming data into local memory:
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<West : 0, DMA : 1>
}
```
**Flow:** Stream from west → DMA channel 1 writes to local buffer

#### Broadcast (Multiple Destinations)
Route one source to multiple destinations:
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<DMA : 0, North : 0>
  aie.connect<DMA : 0, East : 0>
}
```
**Flow:** DMA output → both North and East simultaneously

#### Merge (Multiple Sources)
**Note:** Merging requires different source channels to different destinations:
```mlir
aie.switchbox(%tile_0_2) {
  aie.connect<North : 0, DMA : 0>
  aie.connect<South : 1, DMA : 1>
}
```
**Flow:** Data from north → DMA channel 0, data from south → DMA channel 1

### Multi-Hop Routing

Tracing data flow across multiple tiles:

```mlir
// Tile (0, 1) - Source
aie.switchbox(%tile_0_1) {
  aie.connect<DMA : 0, North : 0>    // Send to tile above
}

// Tile (0, 2) - Relay
aie.switchbox(%tile_0_2) {
  aie.connect<South : 0, North : 0>  // Forward northward
}

// Tile (0, 3) - Destination
aie.switchbox(%tile_0_3) {
  aie.connect<South : 0, DMA : 1>    // Receive into DMA
}
```

**Complete flow:**
1. Tile (0,1) DMA reads buffer
2. Streams north on channel 0
3. Tile (0,2) relays north on channel 0
4. Tile (0,3) receives into DMA channel 1

### Channel Numbering

Different bundles have different numbers of channels:
- **DMA**: Typically 0-1 (MM2S and S2MM channels)
- **Stream (North/South/East/West)**: Often 0-3 or more
- **Core**: Typically 0-1

**Important:** Channel numbers are local to each bundle. `DMA:0` and `North:0` are different physical ports.

### Complete Switchbox Example

```mlir
// Data pipeline: External input → Processing → Output
// Tile (0, 0) - Interface tile
aie.switchbox(%tile_0_0) {
  aie.connect<NOC : 0, North : 0>    // Input from external memory
}

// Tile (0, 1) - Processing tile (producer)
aie.switchbox(%tile_0_1) {
  aie.connect<South : 0, DMA : 1>    // Receive input to DMA
  aie.connect<DMA : 0, East : 0>     // Send processed data east
}

// Tile (1, 1) - Processing tile (consumer)
aie.switchbox(%tile_1_1) {
  aie.connect<West : 0, DMA : 1>     // Receive from west
  aie.connect<DMA : 0, South : 0>    // Send output south
}

// Tile (1, 0) - Interface tile
aie.switchbox(%tile_1_0) {
  aie.connect<North : 0, NOC : 0>    // Output to external memory
}
```

**Data flow:**
1. External memory → NOC → Tile (0,0)
2. Tile (0,0) → North → Tile (0,1) → DMA (input buffer)
3. Tile (0,1) processes data
4. Tile (0,1) DMA → East → Tile (1,1) → DMA (input buffer)
5. Tile (1,1) processes data
6. Tile (1,1) DMA → South → Tile (1,0)
7. Tile (1,0) → NOC → External memory

---

## Complete Example Walkthrough

Let's walk through a complete example showing both raw and lifted output for a simple design.

### Example Design Overview

A simple vector addition design:
- Input buffer A (1024 elements) in tile (0, 2)
- Input buffer B (1024 elements) in tile (0, 2)
- Output buffer C (1024 elements) in tile (0, 2)
- DMA transfers data in and out
- Core performs vector addition

### Step 1: Compile and Generate Xclbin

```bash
# Assume you have a design compiled to add.xclbin
ls add.xclbin
```

### Step 2: Decompile in Raw Mode

```bash
aie-translate --xclbin-to-mlir add.xclbin > add_raw.mlir
```

**Sample Output (add_raw.mlir - excerpts):**

```mlir
module {
  aie.device(npu1_1col) {
  }

  // Block write data for BD configuration
  memref.global "private" constant @cdo_blockwrite_bd0 : memref<16xi32> =
    dense<[0x00000000, 0x00000400, 0x00000001, 0x00000000,
           0x00000000, 0x00010000, 0x00000000, 0x00000000,
           ...]>

  aiex.runtime_sequence {
    // Configure BD 0 registers (input A)
    %0 = memref.get_global @cdo_blockwrite_bd0 : memref<16xi32>
    aiex.npu.blockwrite(%0) { address = 469827584 : ui32 }

    // Configure lock 0
    aiex.npu.write32 { address = 469893120 : ui32, value = 0 : ui32 }

    // Configure switchbox routing
    aiex.npu.write32 { address = 486539264 : ui32, value = 16 : ui32 }
    aiex.npu.write32 { address = 486539268 : ui32, value = 32 : ui32 }

    // More register writes...
  }
}
```

**Analysis:**
- Block write at address `469827584` configures BD registers
- Individual writes configure locks and routing
- Requires knowledge of register map to interpret
- Not immediately clear what the design does

### Step 3: Decompile in Lifted Mode

```bash
aie-translate --xclbin-to-mlir --emit-lifted add.xclbin > add_lifted.mlir
```

**Sample Output (add_lifted.mlir):**

```mlir
module {
  aie.device(npu1_1col) @xclbin_device {
    // Tile declarations
    %tile_0_2 = aie.tile(0, 2)

    // Buffer allocations (note: may show memref<0xi32> if BD config incomplete)
    %buffer_a = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1024xi32>
    %buffer_b = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1024xi32>
    %buffer_c = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1024xi32>

    // DMA memory block containing buffer descriptors
    %mem_0_2 = aie.mem(%tile_0_2) {
      // BD 0 for input A
      aie.dma_bd(%buffer_a : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
      aie.next_bd ^bb1  // Chain to BD 1
    ^bb1:
      // BD 1 for input B
      aie.dma_bd(%buffer_b : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
      aie.next_bd ^bb2  // Chain to BD 2
    ^bb2:
      // BD 2 for output C
      aie.dma_bd(%buffer_c : memref<1024xi32>, 0, 1024) {bd_id = 2 : i32}
      aie.end  // End of chain
    }

    // Switchbox routing for tile (0,2) (if configured in xclbin)
    aie.switchbox(%tile_0_2) {
      aie.connect<NOC : 0, DMA : 0>     // Input from NOC to DMA
      aie.connect<DMA : 1, NOC : 0>     // Output from DMA to NOC
    }
  }

  aiex.runtime_sequence @configure() {
    // Shim tile and other operations that couldn't be lifted
    aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
    // ... more NPU operations ...
    aie.end
  }
}
```

### Step 4: Interpret the Lifted Output

From the lifted output, we can immediately understand:

1. **Resources Used:**
   - One tile: (0, 2)
   - Three buffers: A, B, C (each 1024 elements)
   - DMA memory block with 3 buffer descriptors

2. **Data Movement (BD Chain):**
   - BD 0: Load input A, chains to BD 1
   - BD 1: Load input B, chains to BD 2
   - BD 2: Store output C, ends chain
   - The chain executes automatically: BD 0 → BD 1 → BD 2

3. **BD Structure:**
   - All BDs are contained within `aie.mem(%tile_0_2)`
   - Each BD is in its own basic block
   - `aie.next_bd` creates the BD chain
   - `aie.end` terminates the chain

4. **Routing (if present):**
   - Data flows: NOC → DMA (input path)
   - Data flows: DMA → NOC (output path)
   - Tile (0,2) serves as both input and output interface

5. **Execution Flow:**
   1. DMA executes BD 0: loads buffer A from NOC
   2. Automatically proceeds to BD 1: loads buffer B from NOC
   3. Automatically proceeds to BD 2: stores buffer C to NOC
   4. Chain completes

**Note:** Lock information is decoded from BD control registers (DMA_BDx_5) and emitted as `aie.use_lock` operations. If you don't see lock operations in the output, it means the BD's lock enable bits weren't set in the xclbin. Check the raw mode output to verify register values.

### Step 5: Debugging Use Case

Suppose the output is incorrect. Using the decompiled output:

**Check BD configurations:**
```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  aie.dma_bd(%buffer_c : memref<1024xi32>, 0, 1024) {bd_id = 2 : i32}
  aie.end
}
```
✓ Output BD (BD 2) is correctly defined with proper size

**Check buffer sizes:**
```mlir
%buffer_a = aie.buffer(%tile_0_2) : memref<1024xi32>
```
✓ All buffers are 1024 elements

**Check routing:**
```mlir
aie.connect<DMA : 1, NOC : 0>
```
✓ DMA channel 1 (typically S2MM/output) routes to NOC

**Trace the lock protocol:**
- Input locks start at 0 (empty), set to 1 (full) by DMA
- Output lock starts at 0, core sets to 1 (ready), DMA sets to 0 (done)

This systematic analysis helps identify configuration issues quickly.

---

## Summary

The xclbin decompiler (`aie-translate --xclbin-to-mlir`) is a powerful tool for understanding AIE binary configurations:

- **Raw mode**: Shows exact register operations for low-level analysis
- **Lifted mode**: Reconstructs high-level operations for readability
  - Emits `aie.tile`, `aie.buffer`, `aie.mem`, and `aie.dma_bd` operations
  - DMA buffer descriptors are organized in `aie.mem` blocks with basic block structure
  - BD chaining uses `aie.next_bd` terminator operations
  - Shim tile operations remain as raw NPU writes
  - Switchbox lifting works when routing is configured in the xclbin
- **Use cases**: Debugging, reverse engineering, learning, verification

**Recommended workflow:**
1. Start with lifted mode for quick understanding of tile/buffer/DMA structure
2. Use raw mode when you need exact register details or lock configurations
3. Combine both outputs for comprehensive analysis
4. Be aware that incomplete BD configurations may show `memref<0xi32>` - check raw mode for details

**Key benefits:**
- Verify compiler output matches expectations
- Understand existing designs without source code
- Learn AIE architecture and best practices (especially DMA BD chaining)
- Debug configuration issues systematically
- Reverse engineer xclbin files to understand data movement patterns

**Current Implementation Features (as of March 2026):**
- ✅ Raw mode: Complete register-level output
- ✅ Lifted mode: Tile, buffer, and DMA BD semantic lifting
- ✅ Next_BD chaining support with proper block structure
- ✅ Lock operations: `aie.lock` and `aie.use_lock` when locks are configured
- ✅ Shim DMA BD detection (remains as raw writes)
- ✅ Switchbox routing semantic lifting (when configured)
- ⚠️ Incomplete BDs: May show zero-length buffers when registers aren't fully written

For more information on AIE architecture and MLIR dialect operations, see:
- [AIE Dialect Documentation](https://xilinx.github.io/mlir-aie/)
- [AIEX (AIE Extensions) Dialect](https://xilinx.github.io/mlir-aie/)
- [Building and Platform Setup](Building.md)

---

## Known Limitations

### 1. Shim Tiles (Row 0) Don't Produce Buffer Operations in Lifted Mode

**Issue:** Shim tiles at row 0 do not have local tile memory, so they cannot have `aie.buffer` or `aie.mem` operations in lifted mode.

**Impact:** Buffer descriptor configurations for shim tiles remain as raw `aiex.npu.write32` operations even in lifted mode. The semantic lifting only applies to memory tiles (row 2+) and compute tiles (row 1+) that have local memory.

**Example:**
```mlir
// Even in lifted mode, shim tile (0,0) BD configurations remain as raw writes
aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}  // Shim BD config
aiex.npu.write32 {address = 2229120 : ui32, value = 0 : ui32}  // Shim BD config
```

**Why:** Shim tiles use `aie.shim_dma` operations rather than `aie.mem`, and the decompiler currently focuses on lifting memory tile buffer descriptors. Shim DMA configurations require different handling and are not currently lifted to semantic operations.

**Workaround:** When analyzing shim tile configurations:
- Refer to the raw mode output to see all shim tile operations
- Manually interpret the register addresses using AIE hardware documentation
- Focus on memory tile BDs in lifted mode for data movement analysis

**Warning Output:** When decompiling xclbin files, you may see warnings like:
```
Warning: Incomplete BD configuration found (tile 0,0 BD 0)
```

This is expected for shim tiles and indicates that the decompiler detected BD writes but chose not to lift them because shim tiles lack local memory.

### 2. Incomplete BD Configurations

**Issue:** You may see warnings about incomplete buffer descriptor configurations:
```
Warning: Incomplete BD configuration found (tile X,Y BD N)
```

**Cause:** This occurs when not all registers for a buffer descriptor are written in the CDO. Some fields may:
- Use hardware default values
- Be configured through different mechanisms
- Not be relevant for the specific operation mode

**Impact:** These incomplete BDs are still recorded in lifted mode but may have partial information:
- Buffers may show `memref<0xi32>` indicating zero-length due to missing length register
- Lock operations will only appear if lock enable bits are set in the BD control register
- Dimension information may be absent if wrap/stride registers aren't configured
The decompiled output will show the fields that were actually configured.

**Example:**
```mlir
%bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<0xi32>
%mem_0_2 = aie.mem(%tile_0_2) {
  aie.dma_bd(%bd_buf_0_2_0 : memref<0xi32>, 0, 0) {bd_id = 0 : i32}
  aie.end
}
```

**Solution:** This is usually informational and doesn't indicate a problem. The BD configuration may intentionally use default values for some fields. For complete information, check the raw mode output to see all register writes.

### 3. Block Write Data Storage

**Behavior:** Block write operations (`aiex.npu.blockwrite`) in raw mode reference `memref.global` constants that contain the block data:

```mlir
memref.global "private" constant @cdo_blockwrite_0 : memref<16xi32> = dense<[0, 1, 2, ...]>

// Later in runtime sequence:
%0 = memref.get_global @cdo_blockwrite_0 : memref<16xi32>
aiex.npu.blockwrite(%0) {address = 2228224 : ui32}
```

**Why:** These globals are created at the module level to store the block data extracted from the xclbin CDO commands. This preserves the exact data that will be written to hardware registers.

**Note:** The data in these globals is often configuration tables, initialization values, or complex multi-register setups that are more efficient to write in bulk.

### 4. Register Database Availability

**Warning:** You may see:
```
Warning: Failed to load register database. Register names will not be annotated.
```

**Impact:** This doesn't affect functionality, but register addresses won't have descriptive comments or annotations.

**Cause:** The register database provides human-readable names for hardware register addresses. If it's not available, addresses are shown as numeric values only.

**Solution:** This is a warning only. The decompiler will still function correctly, producing numeric addresses instead of annotated ones.

### 5. Bootgen Library Requirement

**Error:** If bootgen library is not available:
```
Error: CDO decoding not available - bootgen library was not built (OpenSSL required)
```

**Cause:** The decompiler uses Xilinx's bootgen library to decode CDO (Configuration Data Object) binaries. This library requires OpenSSL.

**Solution:** Rebuild MLIR-AIE with bootgen support enabled:
1. Install OpenSSL development libraries: `sudo apt-get install libssl-dev`
2. Rebuild MLIR-AIE to enable bootgen support
3. Verify that CMake configuration shows `HAVE_BOOTGEN` is enabled

### 6. Lock Operations

**Current Status:** Lock information is now fully lifted from DMA BD control registers (DMA_BDx_5) to `aie.lock` and `aie.use_lock` operations.

**Behavior:** When buffer descriptors have lock acquire or release configured:
- `aie.lock` declarations are created for each unique lock ID
- `aie.use_lock` operations are emitted before and after `aie.dma_bd` operations
- Lock acquire uses either `Acquire` or `AcquireGreaterEqual` action depending on the sign of the encoded value
- Lock release uses the `Release` action with the absolute value

**Note:** Many simple designs may not configure locks in their buffer descriptors. If you don't see lock operations in the output, it means the xclbin doesn't have lock configurations enabled. To verify this, use raw mode to see the BD control registers and check if lock enable bits are set.

### 7. Switchbox Lifting Limitations

**Current Status:** Switchbox routing configuration lifting is implemented and works for most common patterns.

**Limitation:**
- Switchboxes are only emitted if routing is explicitly configured in the xclbin
- Complex packet-switched routing may not be fully lifted to semantic operations
- Some packet configuration registers may remain as raw writes

**Expected Behavior:** In lifted mode, when switchbox routing is configured:
- `aie.switchbox` operations for tiles with routing
- `aie.connect` operations for stream connections
- Some configuration registers may remain as raw NPU writes in the runtime sequence

**Note:** Many simple designs may not configure switchbox routing in the xclbin, so you may not see any `aie.switchbox` operations even though the lifting is implemented.

---

## Testing and Verification

### Round-trip Verification Tests

Round-trip tests verify that the xclbin decompiler correctly processes xclbin files and produces valid, verifiable MLIR output. These tests are integrated with the MLIR-AIE lit test framework.

**Test Location:** `/workspace/mlir-aie/test/xclbin2mlir/roundtrip/`

### Available Tests

The roundtrip directory contains tests for both raw and lifted modes:

| Test File | Mode | Description |
|-----------|------|-------------|
| `add_blockwrite_raw.mlir` | Raw | Tests raw mode decompilation of DMA block write design |
| `add_blockwrite_lifted.mlir` | Lifted | Tests lifted mode decompilation with semantic operations |
| `ctrl_packet_reconfig_raw.mlir` | Raw | Tests raw mode for control packet reconfiguration design |
| `ctrl_packet_reconfig_lifted.mlir` | Lifted | Tests lifted mode for control packet design |

Each test verifies:
- ✓ Module structure is present
- ✓ Device declaration exists (`aie.device(npu1_1col)`)
- ✓ Runtime sequence block is created
- ✓ Appropriate operations are generated (write32, maskwrite32, or semantic operations)
- ✓ Proper terminator (`aie.end`) is present
- ✓ Specific register address ranges are accessed (BDs, switchboxes, column controls)

### Setup Environment for Testing

Before running tests, set up your environment:

```bash
# Source XRT runtime
source /opt/xilinx/xrt/setup.sh

# Activate Python virtual environment
source buildenv/bin/activate

# Set PEANO installation directory
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"

# Source MLIR-AIE environment
source mlir-aie/utils/env_setup.sh mlir-aie/install ${PEANO_INSTALL_DIR}
```

### Option 1: Running Tests with lit (Recommended)

**Run all roundtrip tests:**
```bash
cd mlir-aie/build
lit -v test/xclbin2mlir/roundtrip/
```

**Expected Output:**
```
-- Testing: 4 tests, 1 workers --
PASS: MLIR-AIE :: xclbin2mlir/roundtrip/add_blockwrite_raw.mlir (1 of 4)
PASS: MLIR-AIE :: xclbin2mlir/roundtrip/add_blockwrite_lifted.mlir (2 of 4)
PASS: MLIR-AIE :: xclbin2mlir/roundtrip/ctrl_packet_reconfig_raw.mlir (3 of 4)
PASS: MLIR-AIE :: xclbin2mlir/roundtrip/ctrl_packet_reconfig_lifted.mlir (4 of 4)

Testing Time: 3.21s
  Passed: 4
```

**Run a specific test:**
```bash
lit -v test/xclbin2mlir/roundtrip/add_blockwrite_raw.mlir
```

**Run only lifted mode tests:**
```bash
lit -v test/xclbin2mlir/roundtrip/*lifted.mlir
```

### Option 2: Using the Test Script

A convenience script is provided for quick testing:

```bash
cd mlir-aie/test/xclbin2mlir/roundtrip
./run_tests.sh
```

This script runs all roundtrip tests and reports pass/fail status.

### Option 3: Manual Testing with FileCheck

For detailed manual verification:

```bash
# Test raw mode decompilation
aie-translate --xclbin-to-mlir mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin | \
  FileCheck mlir-aie/test/xclbin2mlir/roundtrip/add_blockwrite_raw.mlir

# Test lifted mode decompilation
aie-translate --xclbin-to-mlir --emit-lifted mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin | \
  FileCheck mlir-aie/test/xclbin2mlir/roundtrip/add_blockwrite_lifted.mlir
```

**What FileCheck does:**
- Reads CHECK patterns from the .mlir test file
- Compares the decompiler output against these patterns
- Reports success if all patterns match, failure otherwise

**Example CHECK patterns from tests:**
```mlir
// CHECK: module {
// CHECK:   aie.device(npu1_1col)
// CHECK:     aie.runtime_sequence @configure() {
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}
// CHECK:       aie.end
```

### Option 4: Interactive Testing

To see the actual decompiler output:

```bash
# View raw mode output
aie-translate --xclbin-to-mlir mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin

# View lifted mode output
aie-translate --xclbin-to-mlir --emit-lifted mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin

# Save to file for detailed inspection
aie-translate --xclbin-to-mlir --emit-lifted mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin > /tmp/output.mlir
less /tmp/output.mlir
```

### Understanding Test Verification

The tests use FileCheck patterns to verify specific features. Here's what each test type checks:

**Raw Mode Tests:**
- Presence of `aiex.npu.write32` operations
- Presence of `aiex.npu.maskwrite32` operations
- Specific register address ranges:
  - DMA BD registers (222xxxx range)
  - Stream switch registers (221xxxx range)
  - Column control registers (25xxxx range)
  - AXI MM registers (176xxxx range)

**Lifted Mode Tests:**
- (Currently) Similar to raw mode, verifying NPU operations
- In future: Will verify presence of `aie.mem`, `aie.dma_bd`, `aie.switchbox` operations

### Creating New Tests

To add a new roundtrip test:

1. **Choose an xclbin file** from the test suite (or create one)
2. **Run the decompiler** to see what output is generated
3. **Create a test file** with RUN and CHECK directives:

```mlir
// RUN: aie-translate --xclbin-to-mlir %S/path/to/your.xclbin | FileCheck %s

// CHECK: module {
// CHECK:   aie.device(npu1_1col)
// Add more CHECK patterns...
```

4. **Place the test** in `mlir-aie/test/xclbin2mlir/roundtrip/`
5. **Run lit** to verify it passes

### Debugging Test Failures

If a test fails:

```bash
# Run with verbose output to see what failed
lit -v test/xclbin2mlir/roundtrip/failing_test.mlir

# See the actual decompiler output
aie-translate --xclbin-to-mlir path/to/test.xclbin

# Use FileCheck with --dump-input to see match details
aie-translate --xclbin-to-mlir test.xclbin | FileCheck --dump-input=fail test.mlir
```

Common failure reasons:
- **Address mismatch**: Register addresses changed due to compiler updates
- **Missing operations**: Expected operation type wasn't generated
- **Ordering issues**: Operations appeared in different order than CHECK patterns expect
- **Value changes**: Hardware configuration values changed

### Continuous Integration

The roundtrip tests are part of the MLIR-AIE CI/CD pipeline and run automatically on:
- Pull requests
- Main branch commits
- Release builds

This ensures the decompiler remains functional across development changes.

---

**Document Version:** 1.1
**Last Updated:** March 2026
**Maintained by:** MLIR-AIE Project
