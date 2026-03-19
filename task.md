# Context

You are in the repository root of MLIR-AIE, a compiler that produces binaries to run on special accelerator hardware from AMD, called Neural Processing Units (NPUs). NPUs consist of an array of so-called AI Engines, which are VLIW processors interconnected by a configurable fabric. This is a "data push" architecture; each tile has a DMA that configures how and when data is moved. Although it is DMA, think of this more as a 'data movement accelerator' rather than 'direct memory access' -- these are simply the data movers that run little programs that put data from streams into local scratchpad memories and vice versa. There are also memory and shim tiles, which are used for storage and interacting with the system's main memory, respectively.

A small dedicated 'command processor' runs a firmware that interprets a predefined limited set of instructions. The command processor has full write access to all (memory-mapped) registers and memory regions of the AIE array. It can therefore configure the program memories, switchboxes, etc. of the entire device. Some firmware instructions also allow for synchronization, waiting for some data transfers to complete, etc.

The compiler produces 'xclbin' or 'ELF' binary artifacts. These artifacts are programmed onto the device using the Xilinx runtime (XRT) in userspace and xdna-driver in kernel space (XDNA is AMD's name for this architecture). The device can be programmed via 'xclbins' via the firmware and XRT, or directly by write32/blockwrite instructions from the firmware. Both xclbins and ELFs contain PDIs (programmable device images), which you can think of as essentially maps from register addresses to their values.

The `programming_examples` and `test/npu-xrt` directories contain good examples/tests to see these flows in action.

The directory `lib/Dialect/AIE/Util/aie_registers_aie2.json` contains a database of all registers in this architecture. Use a tool to query this very large file whenever needed.

All needed environment variables can be set by sourcing `~/setup_buildenv.sh`. Always source this environment before running any commands that will need to build or run the code in this repository. Sourcing this will activate the correct Python environment and make a Vitis install available.

Always run compilation as follows:
```
source ~/setup_buildenv.sh
./utils/build-mlir-aie-from-wheels.sh
```

Always run tests as follows:
```
source ~/setup_buildenv.sh
cd build
LIT_FILTER="<test_filter_goes_here>" ninja check-aie 
# Always provide a filter. The entire test suite is too large.
```

# Objective

The goal of this project is to disassemble/decompile existing binaries for this hardware platform into our MLIR dialect at the approporiate IR-level. The end-goal is to be able to 'round-trip' xclbin/ELF decompilation and recompilation, optimally arriving at an identically, or at a minimum functionally equivalent binary. The purpose of this is to allow inspection of code, retrofits of new features to old designs, importing of code from other compiler flows, and new optimizations. 

Different levels of decompilation are possible. The higher the abstraction level, the more useful for the user. At first, we will focus only on deterministic liftings -- register writes that have unambiguous equivalent meanings in the IR.

# Meta-Instructions

- Follow a test-driven development paradigm for large portions of tasks. Create self-contained end-to-end tests that show what the desired end-result is. Challenge yourself -- think of edge cases and include those in tests, but also provide minimal, simple tests that allow you to observe progress towards the goal.
- Never change existing tests. Only simplify or change your own tests if you identified a fundamental issue with them -- not 
- Stop and ask clarifying questions whenever needed.
- Stop and ask for additional context when something is unclear or you are stuck.
- Don't fall for a sunk cost fallacy; if midway through an implementation, you identify a potential for a simpler, more general approach, switch to that approach. Don't "stick it out" because you started it.
- If you are not confident in the path you are going down, stop. Ask me to help. Think of multiple implementation strategies first, before diving into the coding. If there are choices to be made and you are not confident one is the clear best choice, present them to me so I can help choose. I have additional context you might not have.
- Constantly review your own code and identify opportunities for simplification. Existing code in this repository is _not_ off-limits -- if simplification opportunities present themselves, suggest them (but do not act on them until affirmed by the user that this is a useful change).
- Never duplicate code -- if you remember the same, or a similar, functionality being needed elsewhere, factor it out into its own function.
- Keep functions very small. Each unit of a problem should be handled by its own function that clearly denotes what it does and uses a minimal set of input/output parameters.
- Don't comment self-evident code. Don't restate code as comments. Comments are only for conceptual/architectural ideas or surprising edge cases that aren't evident from the code. Comments often become outdated since they are not tested, so the bar for adding a comment should be relatively high.

