"""
ctypes bindings for hipRTC, HIP's runtime compiler.

hipRTC compiles HIP device code in-process (through comgr's embedded clang)
into a code object for one GPU arch. It needs no host toolchain, which is
what makes runtime compilation viable on Windows, where the ROCm Python
wheels ship it as ``hiprtc*.dll`` in ``_rocm_sdk_core/bin`` next to the HIP
runtime.

The library is looked up, in order: one already loaded into the process,
the directory of the HIP runtime torch loaded (so the two always match),
the ROCm Python wheels, ``$ROCM_PATH``/``$HIP_PATH``/``/opt/rocm``, and
finally the dynamic loader's default search.
"""

import ctypes
import glob
import os
import re
import sys
from pathlib import Path

from .hip import hip_library_path, loaded_libraries, rocm_sdk_core_dir

__all__ = ["HipRtcError", "Compiler", "compiler"]


class HipRtcError(RuntimeError):
    pass


if sys.platform == "win32":
    _NAME = re.compile(r"hiprtc(\d+)?\.dll$", re.IGNORECASE)
    _GLOBS = ("hiprtc*.dll",)
    _SUBDIRS = ("bin",)
else:
    _NAME = re.compile(r"libhiprtc\.so(\.\d+)*$")
    _GLOBS = ("libhiprtc.so", "libhiprtc.so.*")
    _SUBDIRS = ("lib",)


def _candidates() -> list[str]:
    found = [p for p in loaded_libraries() if _NAME.match(os.path.basename(p))]
    dirs = []
    try:
        dirs.append(Path(hip_library_path()).parent)
    except Exception:
        pass
    sdk = rocm_sdk_core_dir()
    if sdk is not None:
        dirs += [sdk / s for s in _SUBDIRS]
    for var in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        if os.environ.get(var):
            dirs += [Path(os.environ[var]) / s for s in _SUBDIRS]
    if sys.platform != "win32":
        dirs.append(Path("/opt/rocm/lib"))
    for d in dirs:
        for pattern in _GLOBS:
            for p in sorted(glob.glob(str(d / pattern))):
                if _NAME.match(os.path.basename(p)):
                    found.append(p)
    return list(dict.fromkeys(found))


class Compiler:
    """One loaded hipRTC library."""

    def __init__(self):
        errors = []
        lib = None
        for path in _candidates() + ([] if sys.platform == "win32" else ["libhiprtc.so"]):
            try:
                lib = ctypes.CDLL(path)
                lib.hiprtcCreateProgram  # noqa: B018 -- make sure this really is hipRTC
                self.path = path
                break
            except (OSError, AttributeError) as e:
                errors.append(f"{path}: {e}")
                lib = None
        if lib is None:
            raise HipRtcError("hipRTC library not found" + (": " + "; ".join(errors) if errors else ""))
        vp, i = ctypes.c_void_p, ctypes.c_int
        cpp = ctypes.POINTER(ctypes.c_char_p)
        lib.hiprtcGetErrorString.argtypes = [i]
        lib.hiprtcGetErrorString.restype = ctypes.c_char_p
        lib.hiprtcVersion.argtypes = [ctypes.POINTER(i), ctypes.POINTER(i)]
        lib.hiprtcCreateProgram.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p, ctypes.c_char_p, i, cpp, cpp]
        lib.hiprtcCompileProgram.argtypes = [vp, i, cpp]
        lib.hiprtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(ctypes.c_size_t)]
        lib.hiprtcGetProgramLog.argtypes = [vp, ctypes.c_char_p]
        lib.hiprtcGetCodeSize.argtypes = [vp, ctypes.POINTER(ctypes.c_size_t)]
        lib.hiprtcGetCode.argtypes = [vp, ctypes.c_char_p]
        lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]
        self.lib = lib
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(lib.hiprtcVersion(ctypes.byref(major), ctypes.byref(minor)), "hiprtcVersion")
        self.version = (major.value, minor.value)

    def identity(self) -> str:
        """
        Identifies the compiler for cache keys: hipRTC's version plus the
        library's file name and size (the version alone is only major.minor,
        which does not change between patch releases' compilers).
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = -1
        return f"hiprtc {self.version[0]}.{self.version[1]} {os.path.basename(self.path)} {size}"

    def _check(self, err: int, what: str, log: str = "") -> None:
        if err != 0:
            msg = self.lib.hiprtcGetErrorString(err)
            text = f"{what} failed: hiprtc error {err} ({msg.decode(errors='replace') if msg else '?'})"
            raise HipRtcError(text + (f"\n{log}" if log else ""))

    def compile(self, source: str, name: str, headers: dict[str, str], options: list[str]) -> bytes:
        """Compile ``source`` (which may ``#include`` any of ``headers`` by name); returns the code object."""
        lib = self.lib
        prog = ctypes.c_void_p()
        hdr_src = (ctypes.c_char_p * max(1, len(headers)))(*[s.encode() for s in headers.values()])
        hdr_names = (ctypes.c_char_p * max(1, len(headers)))(*[n.encode() for n in headers])
        self._check(
            lib.hiprtcCreateProgram(
                ctypes.byref(prog), source.encode(), name.encode(), len(headers), hdr_src, hdr_names
            ),
            "hiprtcCreateProgram",
        )
        try:
            opts = (ctypes.c_char_p * max(1, len(options)))(*[o.encode() for o in options])
            err = lib.hiprtcCompileProgram(prog, len(options), opts)
            log = ""
            n = ctypes.c_size_t()
            if lib.hiprtcGetProgramLogSize(prog, ctypes.byref(n)) == 0 and n.value > 1:
                buf = ctypes.create_string_buffer(n.value)
                if lib.hiprtcGetProgramLog(prog, buf) == 0:
                    log = buf.value.decode(errors="replace").strip()
            self._check(err, f"hipRTC compile of {name}", log)
            self._check(lib.hiprtcGetCodeSize(prog, ctypes.byref(n)), "hiprtcGetCodeSize")
            code = ctypes.create_string_buffer(n.value)
            self._check(lib.hiprtcGetCode(prog, code), "hiprtcGetCode")
            return code.raw
        finally:
            lib.hiprtcDestroyProgram(ctypes.byref(prog))


_compiler: Compiler | None = None


def compiler() -> Compiler:
    global _compiler
    if _compiler is None:
        _compiler = Compiler()
    return _compiler
