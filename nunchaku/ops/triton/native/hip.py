"""
ctypes bindings for the few HIP runtime calls the native kernels need.

The kernels are launched through the HIP runtime that torch has already
loaded (``amdhip64_*.dll`` on Windows, ``libamdhip64.so`` on Linux), on
torch's current stream, so they order correctly with every other kernel and
need no compiler or host toolchain at launch time. Binding a *second* copy of
the runtime would give it its own device contexts and streams, so the library
is located by looking at what the process has loaded rather than by name
search.
"""

import ctypes
import os
import re
import sys
from pathlib import Path

import torch

__all__ = ["HipError", "Module", "hip_library_path", "runtime"]


class HipError(RuntimeError):
    pass


_WIN_HIP = re.compile(r"amdhip64(_\d+)?\.dll$", re.IGNORECASE)


def _windows_loaded_modules() -> list[str]:
    """Full paths of every module loaded into this process (Windows)."""
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    proc = k32.GetCurrentProcess()
    count = 1024
    while True:
        mods = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(proc, mods, ctypes.sizeof(mods), ctypes.byref(needed)):
            return []
        if needed.value <= ctypes.sizeof(mods):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    paths = []
    buf = ctypes.create_unicode_buffer(32768)
    for h in mods[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        if h and psapi.GetModuleFileNameExW(proc, h, buf, len(buf)):
            paths.append(buf.value)
    return paths


def _linux_loaded_libraries() -> list[str]:
    paths = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    paths.append(parts[5].strip())
    except OSError:
        pass
    return list(dict.fromkeys(paths))


def loaded_libraries() -> list[str]:
    """Paths of the shared libraries currently loaded into this process."""
    return _windows_loaded_modules() if sys.platform == "win32" else _linux_loaded_libraries()


def hip_library_path() -> str:
    """Path of the HIP runtime torch is using (initializes torch's GPU state)."""
    if torch.version.hip is None:
        raise HipError("torch is not a ROCm build")
    torch.cuda.init()  # makes sure torch has loaded (and initialized) the runtime
    for path in loaded_libraries():
        name = os.path.basename(path)
        if (sys.platform == "win32" and _WIN_HIP.match(name)) or (
            sys.platform != "win32" and name.startswith("libamdhip64.so")
        ):
            return path
    raise HipError("could not find the HIP runtime library loaded by torch")


class _Runtime:
    def __init__(self):
        self.path = hip_library_path()
        lib = ctypes.CDLL(self.path)  # already loaded: the same module handle torch uses
        vp, u, i = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(vp), vp]
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        # hipModuleLaunchKernel(f, grid xyz, block xyz, shared bytes, stream, kernelParams, extra)
        lib.hipModuleLaunchKernel.argtypes = [vp, u, u, u, u, u, u, u, vp, vp, vp]
        lib.hipGetErrorString.argtypes = [i]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipGetLastError.argtypes = []
        self.lib = lib
        self.launch_kernel = lib.hipModuleLaunchKernel

    def check(self, err: int, what: str) -> None:
        if err != 0:
            # a failed call also sets HIP's per-thread "last error", which torch would report from its next
            # kernel launch check; clear it -- the error is raised here instead
            self.lib.hipGetLastError()
            msg = self.lib.hipGetErrorString(err)
            raise HipError(f"{what} failed: hip error {err} ({msg.decode(errors='replace') if msg else '?'})")


_runtime: _Runtime | None = None


def runtime() -> _Runtime:
    global _runtime
    if _runtime is None:
        _runtime = _Runtime()
    return _runtime


class Module:
    """A code object loaded on one device, with its kernels looked up by (``extern "C"``) name."""

    def __init__(self, code: bytes, device: int):
        rt = runtime()
        self.device = device
        self._image = ctypes.create_string_buffer(code, len(code))  # kept alive with the module
        self.handle = ctypes.c_void_p()
        with torch.cuda.device(device):
            rt.check(rt.lib.hipModuleLoadData(ctypes.byref(self.handle), self._image), "hipModuleLoadData")
        self._functions: dict[str, ctypes.c_void_p] = {}

    def function(self, name: str) -> ctypes.c_void_p:
        f = self._functions.get(name)
        if f is None:
            rt = runtime()
            f = ctypes.c_void_p()
            rt.check(
                rt.lib.hipModuleGetFunction(ctypes.byref(f), self.handle, name.encode()),
                f"hipModuleGetFunction({name})",
            )
            self._functions[name] = f
        return f

    def launch(self, name: str, grid: int, block: int, args: tuple[ctypes.Structure, ...]) -> None:
        """
        Launch a 1-D grid on torch's current stream of this module's device.
        ``args`` holds one ctypes object per kernel parameter, laid out as the
        kernel declares it.
        """
        rt = runtime()
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if torch.cuda.current_device() != self.device:
            with torch.cuda.device(self.device):
                err = rt.launch_kernel(self.function(name), grid, 1, 1, block, 1, 1, 0, stream, params, None)
        else:
            err = rt.launch_kernel(self.function(name), grid, 1, 1, block, 1, 1, 0, stream, params, None)
        rt.check(err, f"hipModuleLaunchKernel({name})")


def rocm_sdk_core_dir() -> Path | None:
    """The ``_rocm_sdk_core`` package directory of the ROCm Python wheels (Windows torch ships HIP there)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("_rocm_sdk_core")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])
