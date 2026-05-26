# npukernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
from pathlib import Path
from .trace import TraceConfig


class NPUKernel:
    """
    Represents a compiled NPU kernel.
    """

    def __init__(
        self,
        xclbin_path=None,
        insts_path=None,
        device_index=0,
        kernel_name=None,
        trace_config: TraceConfig | None = None,
        elf_path=None,
    ):
        """
        Initialize the NPUKernel.

        There are two artifact modes:
        - Legacy (npu1/npu2): pass ``xclbin_path`` and ``insts_path``.
        - Full-ELF (npu3 / AIE4): pass ``elf_path`` instead.

        Args:
            xclbin_path (str | Path, optional): Path to the xclbin file (legacy mode).
            insts_path (str | Path, optional): Path to the instructions file (legacy mode).
            device_index (int, optional): Device index. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to ``"MLIR_AIE"``
                in legacy mode and ``"main:sequence"`` in full-ELF mode.
            trace_config (TraceConfig | None, optional): Trace configuration. Defaults to None.
            elf_path (str | Path, optional): Path to the full design ELF (AIE4 / npu3 mode).
        """
        if elf_path is None and (xclbin_path is None or insts_path is None):
            raise ValueError(
                "NPUKernel requires either elf_path (full-ELF mode) or both "
                "xclbin_path and insts_path (legacy mode)."
            )
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._elf_path = elf_path
        if kernel_name is None:
            kernel_name = "main:sequence" if elf_path is not None else "MLIR_AIE"
        self._kernel_name = kernel_name
        self._trace_config = trace_config
        self._device_index = device_index

    @property
    def elf_path(self):
        """
        Get the path to the full design ELF file (AIE4 / npu3 mode), or None
        if this kernel uses the legacy xclbin+insts artifact pair.

        Returns:
            str | Path | None: The ELF path.
        """
        return self._elf_path

    @property
    def is_full_elf(self) -> bool:
        """
        Whether this kernel uses the AIE4 full-ELF artifact (single design.elf
        loaded via ``pyxrt.elf`` + ``pyxrt.ext.kernel``).
        """
        return self._elf_path is not None

    @property
    def trace_config(self) -> TraceConfig | None:
        """
        Get the trace configuration.

        Returns:
            TraceConfig | None: The trace configuration.
        """
        return self._trace_config

    @property
    def xclbin_path(self):
        """
        Get the path to the xclbin file.

        Returns:
            str | Path: The xclbin path.
        """
        return self._xclbin_path

    @property
    def insts_path(self):
        """
        Get the path to the instructions file.

        Returns:
            str | Path: The instructions path.
        """
        return self._insts_path

    @property
    def kernel_name(self):
        """
        Get the kernel name.

        Returns:
            str: The kernel name.
        """
        return self._kernel_name

    # Blocking call.
    def __call__(self, *args, **kwargs):
        """
        Run the kernel with the given arguments.
        This is a blocking call.

        Args:
            *args: Arguments passed to the kernel.
            **kwargs: Additional arguments passed to the runtime load_and_run method.

        Returns:
            The result returned by the runtime ``load_and_run`` call.
        """
        from . import DefaultNPURuntime

        if DefaultNPURuntime is None:
            raise Exception("Cannot run kernel; DefaultNPURuntime not set.")
        return DefaultNPURuntime.load_and_run(
            self,
            list(args),
            **kwargs,
        )
