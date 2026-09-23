# kernels/attention.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Attention kernel factory: fused multi-head attention."""

from pathlib import Path

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import (
    _default_source_path,
    _detect_arch,
    _make_extern,
)

# aie2p/mha.cc drives aie::mmul through mm.cc's bfp16 emulation path, which
# reduces 8x8x8 blocks.
_MHA_MAC_DIMS = (8, 8, 8)


def mha(
    b_q: int = 64,
    b_kv: int = 64,
    d: int = 64,
    vectorized: bool = True,
    use_chess: bool = False,
) -> ExternalFunction:
    """Fused multi-head attention over bf16 blocks.

    mha.cc includes mm.cc and softmax.cc, so one translation unit carries every
    symbol a flash-attention design calls. The returned ExternalFunction binds
    the QK matmul; the rest of the step hangs off it as ``zero``,
    ``init_scale_buffer``, ``partial_softmax``, ``matmul_pv`` and ``rescale_o``,
    each bound from the same object so asking for one does not compile mha.cc
    again.

    K arrives column-major, which is what ``-DB_COL_MAJ`` builds. ``mac_dims``
    reports the block shape the emulation path reduces, so a design tiles from
    the kernel rather than from a table of its own.

    Args:
        b_q: Rows of Q in one block.
        b_kv: Columns of K in one block.
        d: Head dimension.
        vectorized: If ``True`` bind the vectorized QK matmul wrapper.
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Returns:
        ExternalFunction for the QK matmul, carrying the rest as attributes.

    Raises:
        FileNotFoundError: On an architecture that ships no mha.cc.
    """
    from aie.utils import config

    q_ty = np.ndarray[(b_q * d,), np.dtype[bfloat16]]
    k_ty = np.ndarray[(d * b_kv,), np.dtype[bfloat16]]
    qk_ty = np.ndarray[(b_q * b_kv,), np.dtype[bfloat16]]
    scale_ty = np.ndarray[(4 * b_q,), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    arch = _detect_arch()
    source = _default_source_path("mha.cc")
    # mha.cc includes softmax.cc and mm.cc by name, and mm.cc reaches vec_math.h
    # in the runtime library.
    include = [
        str(Path(source).parent),
        str(Path(config.root_path()) / "aie_runtime_lib" / arch.upper()),
    ]

    suffix = "" if vectorized else "_scalar"
    extern = _make_extern(
        f"matmul_bf16_bf16_wrapper{suffix}",
        source,
        [q_ty, k_ty, qk_ty, rtp_ty],
        compile_flags=[
            "-Dbf16_bf16_ONLY",
            f"-DDIM_M={b_q}",
            f"-DDIM_K={d}",
            f"-DDIM_N={b_kv}",
            "-DROUND_CONV_EVEN",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DB_COL_MAJ",
        ],
        use_chess=use_chess,
        extra_include_dirs=include,
    )
    extern.mac_dims = _MHA_MAC_DIMS
    extern.zero = extern.object_file.bind("zero_bf16", [qk_ty])
    extern.init_scale_buffer = extern.object_file.bind(
        "init_scale_buffer", [scale_ty, np.int32]
    )
    extern.partial_softmax = extern.object_file.bind(
        "partial_softmax",
        [
            qk_ty,
            qk_ty,
            scale_ty,
            rtp_ty,
            bfloat16,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    extern.matmul_pv = extern.object_file.bind(
        "matmul_PV",
        [qk_ty, k_ty, qk_ty, scale_ty, np.int32, np.int32, rtp_ty],
    )
    extern.rescale_o = extern.object_file.bind(
        "rescale_O", [qk_ty, scale_ty, np.int32, rtp_ty]
    )
    return extern
