# _hash.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Content-addressed hashing for the JIT cache.

Two halves so callers can distinguish "recipe changed" from "rebuild needed":

* `_compute_recipe_hash`   — generator identity + compile_kwargs +
  aiecc/compile flags. Target-independent design identity.
* `_compute_artifact_hash` — source / object content + tool mtimes +
  target device.  Captures things that change the *output* of compilation
  without changing the *recipe*.

Design inputs (sources, objects, a `Path` generator) are identified by their
**content**.  mtime is not a property of a file, it is a property of how the
file arrived: a fresh clone, a `pip install`, a `cp` or a `touch` all restamp it
without changing a byte, and restoring one hides a change that did happen.  Tool
identity stays on mtime, which is cheap and moves whenever the toolchain is
rebuilt or reinstalled.

`_compute_hash` composes both into the 24-hex cache-key
``CompilableDesign`` uses to address ``$NPU_CACHE_HOME``.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Callable, Iterator, Mapping

from ._introspect import _introspect_generator

logger = logging.getLogger(__name__)

# Read granularity for content digests.  Bounded so a large input is streamed
# rather than materialised, which keeps MemoryError off this path.
_DIGEST_CHUNK = 1 << 20


def _content_digest(path: Path | str) -> str:
    """Digest a file by content, streamed.

    Returns a marker instead of raising: an input we cannot read is still an
    input, and it must not silently collapse onto the same key as an input we
    can.  The marker embeds the error class so "missing" and "unreadable" stay
    distinguishable, which keeps a later fail-closed change a pure policy edit
    rather than a re-plumbing.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_DIGEST_CHUNK):
                h.update(chunk)
    except (OSError, ValueError) as exc:
        return f"<unreadable:{type(exc).__name__}>"
    return h.hexdigest()


def _device_identity_key(device) -> tuple[str, str, str, str]:
    """Return the cache-relevant identity of an IRON device."""
    if device is None:
        return ("none", "", "", "")
    return (
        f"{type(device).__module__}.{type(device).__qualname__}",
        str(getattr(device, "arch", "")),
        str(getattr(device, "cols", "")),
        str(getattr(device, "rows", "")),
    )


def _is_design(value) -> bool:
    """Return True for a value that carries both halves of a design key.

    Asked of the class, so that reading the answer does not compute a hash.
    """
    cls = type(value)
    return hasattr(cls, "recipe_hash") and hasattr(cls, "artifact_hash")


def _parts(value) -> list[tuple[str, Any]] | None:
    """Return the labelled parts of a container or a dataclass, else ``None``.

    Dict keys sort, so that one insertion order does not give a key another
    reading. A value of any other kind has no parts a caller may descend into.
    """
    if isinstance(value, (list, tuple)):
        return [(str(index), item) for index, item in enumerate(value)]
    if isinstance(value, dict):
        return [(str(key), value[key]) for key in sorted(value, key=str)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
    return None


def _designs_in(value) -> Iterator[Any]:
    """Yield every design a compile-time argument holds.

    A design that takes another design as an argument compiles what the child
    compiles, so the child's inputs are the parent's inputs. Descent stops at a
    design: its own key already covers what it holds.
    """
    if _is_design(value):
        yield value
        return
    for _, part in _parts(value) or ():
        yield from _designs_in(part)


def _without_location(const):
    """Return a constant with file and line info dropped, nested code objects included.

    Applied to every constant, not just code objects, so one cannot keep its
    location by sitting inside a tuple.
    """
    if isinstance(const, tuple):
        return tuple(_without_location(c) for c in const)
    if not isinstance(const, CodeType):
        return const
    return const.replace(
        co_consts=tuple(_without_location(c) for c in const.co_consts),
        co_filename="",
        co_firstlineno=1,
        co_linetable=b"",
    )


def _code_identity(code: CodeType) -> bytes:
    """Stable bytes for a code object: what marshal writes into a ``.pyc``.

    ``repr()`` of a code object embeds its address, so a key built from it moves
    between processes. marshal covers bytecode, names, varnames, flags and
    constants, recursing into nested code, with no addresses and a hash-seed
    independent frozenset order. ``co_names`` matters: ``matmul_bf16(a, b, c)``
    and ``matmul_i8(a, b, c)`` compile to identical bytecode.

    Location is stripped first: it is not part of the design, and keying on it
    would split the cache per checkout. Version 4 is pinned because
    ``marshal.version`` is 4 through 3.13 and 5 from 3.14.
    """
    return marshal.dumps(_without_location(code), 4)


def _compute_recipe_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
) -> str:
    """Hash of the "recipe": generator bytecode + CompileTime[T] kwargs + flags.

    Captures the target-independent generator and compile configuration. It
    omits device identity, so equal recipe hashes can produce different
    target-specialized MLIR.

    ``full_elf`` is part of the recipe: full-ELF and xclbin+insts builds emit
    different MLIR (the former injects ``npu.load_pdi``) and different
    artifacts, so they must not share a cache entry.

    ``include_paths`` likewise: they are ``-I`` directories forwarded to the
    C++ compiler, so two otherwise identical designs pointed at different
    header trees compile to different objects. Hashed in ORDER, not sorted
    like the flag lists above, because ``-I`` search order decides which
    header wins when two directories provide the same name.

    A kwarg that is itself a design contributes that design's recipe hash, and
    so does a design nested in a container or a dataclass. The artifact half of
    such a child goes to the artifact half of the parent.
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        h.update(str(generator).encode())
        h.update(_content_digest(generator).encode())
    else:
        h.update(_code_identity(generator.__code__))
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(getattr(generator, "__module__", "").encode())
        hints, sig, (_, _, dispatch_params, _) = _introspect_generator(generator)
        # Dispatch defaults are call-time values; explicitly bound defaults are
        # unused. Neither changes the compiled program.
        h.update(
            repr(
                [
                    param.replace(
                        annotation=hints.get(name, param.annotation),
                        default=(
                            param.empty
                            if name in dispatch_params or name in compile_kwargs
                            else param.default
                        ),
                    )
                    for name, param in sig.parameters.items()
                ]
            ).encode()
        )

    def _kwarg_repr(v):
        if _is_design(v):
            # repr of a design carries an address, so a parent keyed on it moves
            # between processes. The child's own recipe half keeps both halves apart.
            return ("design:", v.recipe_hash)
        parts = _parts(v)
        if parts is not None:
            # Descend, so that a design nested in a container or a dataclass
            # reaches the key through the branch above.
            return (
                type(v).__name__,
                {label: _kwarg_repr(part) for label, part in parts},
            )
        if callable(v) and hasattr(v, "__code__"):
            closure = (
                tuple(c.cell_contents for c in v.__closure__) if v.__closure__ else None
            )
            try:
                closure_repr = repr(closure)
            except Exception:
                closure_repr = "<unhashable closure>"
            return (
                "fn:",
                _code_identity(v.__code__).hex(),
                repr(getattr(v, "__defaults__", None)),
                repr(getattr(v, "__kwdefaults__", None)),
                closure_repr,
            )
        return str(v)

    try:
        kwargs_json = json.dumps(
            {k: _kwarg_repr(v) for k, v in sorted(compile_kwargs.items())}
        ).encode()
    except (TypeError, ValueError):
        kwargs_json = repr(sorted(compile_kwargs.items())).encode()
    h.update(kwargs_json)

    h.update(repr(sorted(aiecc_flags)).encode())
    h.update(repr(sorted(compile_flags)).encode())
    h.update(f"full_elf={full_elf}".encode())
    h.update(repr([str(p) for p in include_paths]).encode())

    return h.hexdigest()


def _tool_identity(name: str, resolve: Callable[[], str | Path]) -> str:
    """Identify a resolved compiler component without probing an executable."""
    try:
        path = Path(resolve()).resolve()
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        logger.warning("_compute_artifact_hash: %s absent (%s)", name, exc)
        return "absent"


def _compute_artifact_hash(
    generator: Callable | Path,
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    fold_ddr_addr_offset: bool,
    has_dispatch_params: bool = False,
    compile_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Hash of the "artifacts": source/object content + tool mtimes + device.

    Captures everything that can change the *output* of compilation without
    changing the *recipe*: edited C++ kernels, swapped object files, upgraded
    Peano / aiecc, retargeted device.

    ``fold_ddr_addr_offset`` is the active backend's DDR-patch ABI: XRT/CPU emit
    a folded ``insts.bin`` and HRX an unfolded one, so the two must never share a
    cache entry. It is resolved once by the caller and passed in explicitly (no
    silent default) so the cache key and the compilation can never disagree.

    ``has_dispatch_params`` additionally hashes the host C++ compiler used to
    build the dispatch library. Its generated source is covered by aiecc's
    identity above; Python does not run a separate translation pipeline.

    ``compile_kwargs`` contributes the artifact hash of every design the
    arguments hold, so that editing a child's kernel rebuilds the parent.
    """
    h = hashlib.sha256()

    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        h.update(_content_digest(sf).encode())

    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        h.update(_content_digest(of).encode())

    for name, value in sorted((compile_kwargs or {}).items()):
        for design in _designs_in(value):
            h.update(f"{name}={design.artifact_hash}".encode())

    h.update(f"fold_ddr_addr_offset={fold_ddr_addr_offset}".encode())
    # Static .mlir is target-agnostic; compiled kernels need a device identifier.
    # Missing components collapse to a constant + WARNING log so cross-target
    # cache collisions surface instead of silently aliasing.
    if not isinstance(generator, Path):
        try:
            from aie.utils import get_current_device
            from aie.utils.compile.utils import resolve_target_arch

            device = get_current_device(probe_runtime=False)
            target_arch = resolve_target_arch(device)
            target_device = _device_identity_key(device)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "_compute_artifact_hash: target_arch unresolved (%s); using 'unknown'",
                exc,
            )
            target_arch = "unknown"
            target_device = ("unknown", "", "", "")

        h.update(f"target_arch={target_arch}|target_device={target_device!r}".encode())
        from aie.utils import config as _config

        tools = {
            "peano": _config.peano_cxx_path,
            "aiecc": _config.aiecc_path,
        }
        if has_dispatch_params:
            tools["host_cxx"] = _config.host_cxx_path
        for name, resolve in tools.items():
            h.update(f"{name}={_tool_identity(name, resolve)}".encode())

    return h.hexdigest()


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    fold_ddr_addr_offset: bool = True,
    has_dispatch_params: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
) -> str:
    """Stable 24-hex SHA-256 cache key combining recipe + artifact hashes."""
    recipe = _compute_recipe_hash(
        generator, compile_kwargs, aiecc_flags, compile_flags, full_elf, include_paths
    )
    artifact = _compute_artifact_hash(
        generator,
        source_files,
        object_files,
        fold_ddr_addr_offset,
        has_dispatch_params,
        compile_kwargs,
    )
    return hashlib.sha256(f"{recipe}|{artifact}".encode()).hexdigest()[:24]
