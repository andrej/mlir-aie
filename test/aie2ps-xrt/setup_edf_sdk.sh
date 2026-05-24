#!/bin/bash
# Set up environment for AIE2PS (VEK385) cross-compilation.
#
# Usage:
#   source test/aie2ps-hw/setup_edf_sdk.sh [sdk_path]
#
# If the SDK is not installed, run:
#   <sdk.sh from boot_images> -d <sdk_path> -y
#
# The SDK provides a matched aarch64 cross-compiler + sysroot with XRT
# headers/libs. It must match the EDF Linux version on the board.

EDF_SDK_PATH="${1:-${EDF_SDK_PATH:-$HOME/amd-edf-sdk}}"

SYSROOT="$EDF_SDK_PATH/sysroots/cortexa72-cortexa53-amd-linux"
CXX="$EDF_SDK_PATH/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++"

if [ ! -f "$CXX" ]; then
    echo "EDF SDK not found at $EDF_SDK_PATH"
    echo "Install with: <boot_images>/sdk.sh -d $EDF_SDK_PATH -y"
    return 1 2>/dev/null || exit 1
fi

export EDF_SDK_PATH
export AIE2PS_CXX="$CXX --sysroot=$SYSROOT"
export AIE2PS_SYSROOT="$SYSROOT"

echo "EDF SDK: $EDF_SDK_PATH"
echo "AIE2PS_CXX: $AIE2PS_CXX"
