"""
Run API-format ComfyUI workflows headless, time them and check their images.

  python run_workflows.py --comfy <ComfyUI dir> [--extra-model-paths <yaml>] [--image <file>] [--runs 3] \
      [--out <dir>] [--reference <dir>] [--comfy-arg=--highvram ...] wf.json ...

Starts ``main.py`` on a free local port, submits each workflow ``--runs`` times through the HTTP API
(the seed of run i is i, so ComfyUI's cache does not skip sampling and runs are reproducible across machines;
loaders stay cached), waits for it, and prints the wall time per run and the saved images. ``--image`` is copied
into ComfyUI's input directory and used by every ``LoadImage`` node. The server is stopped at the end.

``--out`` saves every image as ``<workflow>-run<i>.png``; ``--reference`` compares each image with the file of
the same name in another directory (e.g. the ``--out`` of a full-precision run) and prints the PSNR.

GPU memory: ``/system_stats`` is polled during each run; "peak" is the most device memory in use (total minus
ComfyUI's free estimate, which counts PyTorch's cached-but-unused blocks as free), "base" is the same before the
first workflow (other processes, display, PyTorch context).
"""

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True)
ap.add_argument("--extra-model-paths")
ap.add_argument("--image")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--timeout", type=float, default=1800)
ap.add_argument("--out", help="save each image as <out>/<workflow>-run<i>.png")
ap.add_argument("--reference", help="directory of reference images (same names as --out) to compare against")
ap.add_argument("--log", help="ComfyUI server log (default: <comfy>/run_workflows.log)")
ap.add_argument("--comfy-arg", action="append", default=[], help="extra main.py argument (repeatable)")
ap.add_argument("workflows", nargs="+")
args = ap.parse_args()

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
url = f"http://127.0.0.1:{port}"


def call(path, payload=None, raw=False):
    req = urllib.request.Request(
        url + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if raw else json.loads(data)


def vram_used_gib():
    dev = call("/system_stats")["devices"][0]
    return (dev["vram_total"] - dev["vram_free"]) / 1024**3


class VramPoller:
    """Polls the device memory in use in a background thread while a workflow runs."""

    def __enter__(self):
        self.peak, self.stop = 0.0, threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, vram_used_gib())
            except OSError:
                pass
            self.stop.wait(0.1)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def psnr(a_png: bytes, ref_path: str) -> float:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(a_png)).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"{ref_path}: shape {b.shape} != {a.shape}")
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


cmd = [
    sys.executable,
    "-u",
    os.path.join(args.comfy, "main.py"),
    "--listen",
    "127.0.0.1",
    "--port",
    str(port),
    "--disable-auto-launch",
    *args.comfy_arg,
]
if args.extra_model_paths:
    cmd += ["--extra-model-paths-config", args.extra_model_paths]
if args.out:
    os.makedirs(args.out, exist_ok=True)
log = open(args.log or os.path.join(args.comfy, "run_workflows.log"), "w")
server = subprocess.Popen(cmd, cwd=args.comfy, stdout=log, stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while True:
        try:
            call("/system_stats")
            break
        except OSError:
            if server.poll() is not None or time.time() - t0 > 300:
                raise SystemExit(f"ComfyUI did not start; see {log.name}")
            time.sleep(1)
    base = vram_used_gib()
    print(f"ComfyUI up in {time.time() - t0:.1f}s on port {port}; GPU memory in use: {base:.2f} GiB", flush=True)

    image_name = None
    if args.image:
        image_name = os.path.basename(args.image)
        shutil.copy(args.image, os.path.join(args.comfy, "input", image_name))

    for wf_path in args.workflows:
        wf = json.load(open(wf_path))
        wf_name = os.path.splitext(os.path.basename(wf_path))[0]
        for run in range(args.runs):
            for node in wf.values():
                inputs = node["inputs"]
                for key in ("seed", "noise_seed"):
                    if key in inputs:
                        inputs[key] = run
                if node["class_type"] == "LoadImage" and image_name:
                    inputs["image"] = image_name
            t = time.time()
            with VramPoller() as vram:
                try:
                    pid = call("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
                except urllib.error.HTTPError as e:  # validation errors, e.g. a missing node or model file
                    raise SystemExit(f"{os.path.basename(wf_path)} rejected: {e.read().decode()[:2000]}")
                while True:
                    hist = call(f"/history/{pid}")
                    if pid in hist:
                        break
                    if server.poll() is not None or time.time() - t > args.timeout:
                        raise SystemExit(f"ComfyUI exited or timed out; see {log.name}")
                    time.sleep(0.25)
            elapsed = time.time() - t
            status = hist[pid]["status"]
            if status.get("status_str") != "success":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise SystemExit(f"{os.path.basename(wf_path)} failed: {json.dumps(msgs)[:2000]}")
            images = [i for o in hist[pid]["outputs"].values() for i in o.get("images", [])]
            line = (
                f"{wf_name} run{run + 1}: {elapsed:.2f}s, peak GPU memory {vram.peak:.2f} GiB (base {base:.2f}) "
                f"-> {[i['filename'] for i in images]}"
            )
            for k, img in enumerate(images):
                png = call(
                    "/view?" + urllib.parse.urlencode({k2: img[k2] for k2 in ("filename", "subfolder", "type")}),
                    raw=True,
                )
                name = f"{wf_name}-run{run + 1}" + (f"-{k}" if len(images) > 1 else "") + ".png"
                if args.out:
                    with open(os.path.join(args.out, name), "wb") as f:
                        f.write(png)
                ref = os.path.join(args.reference, name) if args.reference else None
                if ref and os.path.exists(ref):
                    line += f"; PSNR vs reference {psnr(png, ref):.2f} dB"
                elif ref:
                    line += "; no reference image"
            print(line, flush=True)
finally:
    server.terminate()
    try:
        server.wait(30)
    except subprocess.TimeoutExpired:
        server.kill()
    log.close()
os._exit(0)
