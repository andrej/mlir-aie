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

#### 3. `aie.lock`
Declares a hardware lock for synchronization.

**Format:**
```mlir
%lock = aie.lock(%tile, LOCK_ID)
```

**Example:**
```mlir
%lock_0_2_0 = aie.lock(%tile_0_2, 0)
```

- `%tile`: The tile containing this lock
- `LOCK_ID`: Hardware lock identifier (typically 0-15)

**Interpretation:** Declares lock #0 on tile (0,2) for synchronizing access to shared resources.

#### 4. `aie.dma_bd`
Defines a DMA Buffer Descriptor for data movement.

**Format:**
```mlir
aie.dma_bd(%buffer : memref<SIZE x TYPE>, OFFSET, LENGTH) {
  // Optional attributes
  dimensions = [#aie.dma_dim<...>, ...]
  lock_acq_id = ...
  lock_acq_val = ...
  lock_rel_id = ...
  lock_rel_val = ...
  next_bd = ...
  valid_bd = ...
}
```

**Example:**
```mlir
aie.dma_bd(%buffer_0 : memref<1024xi32>, 0, 256) {
  lock_acq_id = 0
  lock_acq_val = 1
  lock_rel_id = 0
  lock_rel_val = 0
  valid_bd = true
}
```

**Parameters:**
- `%buffer`: The buffer this BD operates on
- `OFFSET`: Starting offset in the buffer (in elements)
- `LENGTH`: Number of elements to transfer

**Attributes:** (See [Interpreting DMA Buffer Descriptor Configurations](#interpreting-dma-buffer-descriptor-configurations))

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
    %buffer_0 = aie.buffer(%tile_0_2) : memref<1024xi32>
    %buffer_1 = aie.buffer(%tile_0_2) : memref<1024xi32>

    // Declare locks
    %lock_0 = aie.lock(%tile_0_2, 0)
    %lock_1 = aie.lock(%tile_0_2, 1)

    // Define DMA buffer descriptors
    aie.dma_bd(%buffer_0 : memref<1024xi32>, 0, 256) {
      lock_acq_id = 0
      lock_acq_val = 1
      lock_rel_id = 0
      lock_rel_val = 0
      next_bd = 1
      valid_bd = true
    }

    aie.dma_bd(%buffer_1 : memref<1024xi32>, 0, 256) {
      lock_acq_id = 1
      lock_acq_val = 1
      lock_rel_id = 1
      lock_rel_val = 0
      valid_bd = true
    }

    // Configure switchbox routing
    aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<North : 0, DMA : 1>
    }

    aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, North : 0>
    }
  }

  aiex.runtime_sequence {
    // Any operations that couldn't be lifted remain here
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

Multiple BDs can be chained for complex transfer sequences:

```mlir
// BD 0
aie.dma_bd(%buffer_a : memref<512xi32>, 0, 128) {
  next_bd = 1
  valid_bd = true
}

// BD 1
aie.dma_bd(%buffer_b : memref<512xi32>, 0, 128) {
  next_bd = 2
  valid_bd = true
}

// BD 2 (last in chain)
aie.dma_bd(%buffer_c : memref<512xi32>, 0, 128) {
  valid_bd = true
  // No next_bd - end of chain
}
```

**Interpretation:**
- Execute BD 0, then automatically proceed to BD 1, then BD 2
- Enables complex transfer patterns without CPU intervention
- Can create circular chains by pointing back to BD 0

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

```mlir
// Transfer a 16x16 tile from a larger 256x256 matrix
// Each element is 32-bit, access stride pattern with locking
aie.dma_bd(%matrix : memref<65536xi32>, 0, 256) {
  // 2D access: 16 rows of 16 elements each
  dimensions = [
    #aie.dma_dim<stepsize = 1, wrap = 16>,    // 16 elements per row
    #aie.dma_dim<stepsize = 256, wrap = 16>   // 16 rows, stride 256 between rows
  ]

  // Synchronization
  lock_acq_id = 0
  lock_acq_val = 1      // Wait for producer
  lock_rel_id = 0
  lock_rel_val = 0      // Signal consumer done

  // Chaining
  next_bd = 1           // Continue to next BD
  valid_bd = true
}
```

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
  aie.device(npu1_1col) {
    // Tile declarations
    %tile_0_2 = aie.tile(0, 2)

    // Buffer allocations
    %buffer_a = aie.buffer(%tile_0_2) : memref<1024xi32>
    %buffer_b = aie.buffer(%tile_0_2) : memref<1024xi32>
    %buffer_c = aie.buffer(%tile_0_2) : memref<1024xi32>

    // Lock declarations
    %lock_a = aie.lock(%tile_0_2, 0)
    %lock_b = aie.lock(%tile_0_2, 1)
    %lock_c = aie.lock(%tile_0_2, 2)

    // DMA Buffer Descriptor for input A
    aie.dma_bd(%buffer_a : memref<1024xi32>, 0, 1024) {
      lock_acq_id = 0
      lock_acq_val = 0        // Acquire empty buffer
      lock_rel_id = 0
      lock_rel_val = 1        // Release as full
      next_bd = 1
      valid_bd = true
    }

    // DMA Buffer Descriptor for input B
    aie.dma_bd(%buffer_b : memref<1024xi32>, 0, 1024) {
      lock_acq_id = 1
      lock_acq_val = 0
      lock_rel_id = 1
      lock_rel_val = 1
      next_bd = 2
      valid_bd = true
    }

    // DMA Buffer Descriptor for output C
    aie.dma_bd(%buffer_c : memref<1024xi32>, 0, 1024) {
      lock_acq_id = 2
      lock_acq_val = 1        // Wait for result
      lock_rel_id = 2
      lock_rel_val = 0        // Release as empty
      valid_bd = true
    }

    // Switchbox routing for tile (0,2)
    aie.switchbox(%tile_0_2) {
      aie.connect<NOC : 0, DMA : 0>     // Input from NOC to DMA
      aie.connect<DMA : 1, NOC : 0>     // Output from DMA to NOC
    }
  }

  aiex.runtime_sequence {
    // Remaining runtime operations
  }
}
```

### Step 4: Interpret the Lifted Output

From the lifted output, we can immediately understand:

1. **Resources Used:**
   - One tile: (0, 2)
   - Three buffers: A, B, C (each 1024 elements)
   - Three locks: 0, 1, 2

2. **Data Movement:**
   - BD 0: Load input A (acquire empty, release full)
   - BD 1: Load input B (acquire empty, release full)
   - BD 2: Store output C (acquire full/ready, release empty)

3. **Synchronization Pattern:**
   - Locks ensure data is ready before processing
   - Producer-consumer protocol on each buffer

4. **Routing:**
   - Data flows: NOC → DMA (input path)
   - Data flows: DMA → NOC (output path)
   - Tile (0,2) serves as both input and output interface

5. **Execution Flow:**
   1. DMA loads buffer A from NOC (releases lock 0 = full)
   2. DMA loads buffer B from NOC (releases lock 1 = full)
   3. Core processes A + B → C (releases lock 2 = full)
   4. DMA stores buffer C to NOC (releases lock 2 = empty)

### Step 5: Debugging Use Case

Suppose the output is incorrect. Using the decompiled output:

**Check BD configurations:**
```mlir
aie.dma_bd(%buffer_c : memref<1024xi32>, 0, 1024) {
  lock_acq_id = 2
  lock_acq_val = 1
  ...
}
```
✓ Output BD correctly waits for lock 2 = 1 (result ready)

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
- **Use cases**: Debugging, reverse engineering, learning, verification

**Recommended workflow:**
1. Start with lifted mode for quick understanding
2. Use raw mode when you need exact register details
3. Combine both outputs for comprehensive analysis

**Key benefits:**
- Verify compiler output matches expectations
- Understand existing designs without source code
- Learn AIE architecture and best practices
- Debug configuration issues systematically

For more information on AIE architecture and MLIR dialect operations, see:
- [AIE Dialect Documentation](https://xilinx.github.io/mlir-aie/)
- [AIEX (AIE Extensions) Dialect](https://xilinx.github.io/mlir-aie/)
- [Building and Platform Setup](Building.md)

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Maintained by:** MLIR-AIE Project
