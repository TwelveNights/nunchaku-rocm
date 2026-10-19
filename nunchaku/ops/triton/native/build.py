"""
Runtime compilation of the native kernels, with an on-disk code-object cache.

Kernel sources ship as ``.hip`` files next to this module and are compiled
with hipRTC on first use for the exact GPU arch present (e.g. ``gfx1101``).
The resulting code object is cached under

``$NUNCHAKU_CACHE_DIR/native/`` if set, else
``%LOCALAPPDATA%\\nunchaku\\cache\\native\\`` (Windows) or
``${XDG_CACHE_HOME:-~/.cache}/nunchaku/native/`` (elsewhere),

keyed by a hash of the source and headers, the arch, the compiler identity
(:meth:`~.hiprtc.Compiler.identity`) and the compile options, so editing a
kernel or upgrading ROCm recompiles automatically. Entries are written to a
temporary file and renamed into place, so concurrent processes never see a
partial file: they may both compile, and whichever rename lands last wins
with identical bytes. An unwritable cache directory only costs a recompile
per process.
"""

import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["cache_dir", "code_object", "SOURCE_DIR"]

logger = logging.getLogger(__name__)

SOURCE_DIR = Path(__file__).resolve().parent
# bump to invalidate every cached entry (e.g. if the cache format changes)
_CACHE_FORMAT = 1


def cache_dir() -> Path:
    root = os.environ.get("NUNCHAKU_CACHE_DIR")
    if root:
        base = Path(root)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "nunchaku" / "cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nunchaku"
    return base / "native"


def _read_cached(path: Path) -> bytes | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data or None


def _write_atomic(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    except OSError as e:
        logger.debug("native kernel cache not writable (%s): %s", path.parent, e)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        # e.g. Windows refuses to replace a file another process has open; that process wrote the same bytes
        logger.debug("could not write native kernel cache entry %s: %s", path, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def code_object(
    source_name: str,
    headers: list[str],
    arch: str,
    options: list[str],
    what: str,
    refresh: bool = False,
    source_dir: Path = SOURCE_DIR,
) -> tuple[bytes, bool]:
    """
    Code object for ``source_dir/source_name`` compiled for ``arch``, from the
    cache or freshly compiled. ``headers`` are files in ``source_dir`` the
    source may ``#include``. ``what`` names the kernels in the log message.
    ``refresh`` skips the cache lookup (the fresh result still replaces the
    entry), for a cached object that failed to load. Returns
    ``(code, from_cache)``. Raises on compile failure.
    """
    from .hiprtc import compiler

    source = (Path(source_dir) / source_name).read_text(encoding="utf-8")
    hdrs = {h: (Path(source_dir) / h).read_text(encoding="utf-8") for h in headers}
    opts = [f"--offload-arch={arch}", *options]
    rtc = compiler()

    key = hashlib.sha256()
    for part in (
        str(_CACHE_FORMAT),
        arch,
        rtc.identity(),
        "\0".join(opts),
        source_name,
        source,
        *[f"{n}\0{s}" for n, s in sorted(hdrs.items())],
    ):
        key.update(part.encode())
        key.update(b"\0\1")
    path = cache_dir() / f"{Path(source_name).stem}-{arch}-{key.hexdigest()[:32]}.co"

    code = None if refresh else _read_cached(path)
    if code is not None:
        return code, True

    logger.info("Compiling %s for %s with hipRTC (one-time, cached in %s)", what, arch, path.parent)
    t0 = time.perf_counter()
    code = rtc.compile(source, source_name, hdrs, opts)
    logger.info("Compiled %s in %.1f s", what, time.perf_counter() - t0)
    _write_atomic(path, code)
    return code, False
