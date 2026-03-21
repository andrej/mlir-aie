PREREQUISITES AND REQUIRED ENVIRONMENT SETUP:

ALWAYS run the following script before executing any command:

```
source /workspace/env_setup.sh
```

REBUILDING MLIR-AIE:

ALWAYS use the following commands to (re-)build MLIR-AIE after changes to the compiler:

```
source /workspace/env_setup.sh
cd /workspace/mlir-aie
./mlir-aie/utils/build-mlir-aie-from-wheels.sh
```

This command must be run from the mlir-aie root directory.

Critically important:
- ALWAYS source the above environment to ensure a working environment. 
- ALWAYS use the above build commands. After compilation, the compiled binaries will be in your PATH, so that tests and programming examples should work with the new compiler.
- Remember to clear `build` directories for tests and examples after the compiler has been rebuilt.
- If during compilation of an example, you get this error: "Error: No AIE devices found in module" -- this means a previous step failed. Clear the build directory and retry.