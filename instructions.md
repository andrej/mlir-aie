Installing prerequisites to build MLIR-AIE:

Set up the environment with the following commands:

```
BUILDENV=buildenv
python3 -m venv ${BUILDENV}
source ${BUILDENV}/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install llvm-aie -I -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
python3 -m pip install -I -r /scratch/roesti/mlir-aie/python/requirements_dev.txt
python3 -m pip install ml_dtypes
```

Setting up an environment in which you can build MLIR-AIE:

Once the prequisites are installed (one time step), please always set up your environment as follows:
```
source /opt/xilinx/xrt/setup.sh
source buildenv/bin/activate
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
source mlir-aie/utils/env_setup.sh mlir-aie/install ${PEANO_INSTALL_DIR}
export LIT_OPTS="-sv --order=random --time-tests -j1 --timeout 600 --show-unsupported --show-excluded --max-failures 1"  # IMPORTANT to not overload system!
export BUILD_TYPE=Debug
export IRON_CACHE_HOME=ironcache
```

Building MLIR-AIE:

MLIR-AIE builds on LLVM, which is very large and takes a long time to build. Almost never are modifications to LLVM necessary; instead, we can use prebuilt binaries (wheels). Always use the following command to build MLIR-AIE:

```
# Source above environment setup steps
./mlir-aie/utils/build-mlir-aie-from-wheels.sh
```

This command must be run from the mlir-aie root directory.
