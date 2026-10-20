"""
Run API-format ComfyUI workflows headless and time them.

  python run_workflows.py --comfy <ComfyUI dir> [--extra-model-paths <yaml>] [--image <file>] [--runs 3] \
      [--comfy-arg=--highvram ...] wf.json ...

Starts ``main.py`` on a free local port, submits each workflow ``--runs`` times through the HTTP API
(changing the seed every run so ComfyUI's cache does not skip sampling; loaders stay cached), waits for
it, and prints the wall time per run and the saved images. ``--image`` is copied into ComfyUI's input
directory and used by every ``LoadImage`` node. The server is stopped at the end.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True)
ap.add_argument("--extra-model-paths")
ap.add_argument("--image")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--timeout", type=float, default=1800)
ap.add_argument("--comfy-arg", action="append", default=[], help="extra main.py argument (repeatable)")
ap.add_argument("workflows", nargs="+")
args = ap.parse_args()

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
url = f"http://127.0.0.1:{port}"


def call(path, payload=None):
    req = urllib.request.Request(url + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


cmd = [sys.executable, "-u", os.path.join(args.comfy, "main.py"), "--listen", "127.0.0.1", "--port", str(port),
       "--disable-auto-launch", *args.comfy_arg]
if args.extra_model_paths:
    cmd += ["--extra-model-paths-config", args.extra_model_paths]
log = open(os.path.join(args.comfy, "run_workflows.log"), "w")
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
    print(f"ComfyUI up in {time.time() - t0:.1f}s on port {port}", flush=True)

    image_name = None
    if args.image:
        image_name = os.path.basename(args.image)
        shutil.copy(args.image, os.path.join(args.comfy, "input", image_name))

    for wf_path in args.workflows:
        wf = json.load(open(wf_path))
        for run in range(args.runs):
            for node in wf.values():
                inputs = node["inputs"]
                for key in ("seed", "noise_seed"):
                    if key in inputs:
                        inputs[key] = run
                if node["class_type"] == "LoadImage" and image_name:
                    inputs["image"] = image_name
            t = time.time()
            pid = call("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
            while True:
                hist = call(f"/history/{pid}")
                if pid in hist:
                    break
                if server.poll() is not None or time.time() - t > args.timeout:
                    raise SystemExit(f"ComfyUI exited or timed out; see {log.name}")
                time.sleep(0.25)
            status = hist[pid]["status"]
            images = [i["filename"] for o in hist[pid]["outputs"].values() for i in o.get("images", [])]
            if status.get("status_str") != "success":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise SystemExit(f"{os.path.basename(wf_path)} failed: {json.dumps(msgs)[:2000]}")
            print(f"{os.path.basename(wf_path)} run{run + 1}: {time.time() - t:.2f}s -> {images}", flush=True)
finally:
    server.terminate()
    try:
        server.wait(30)
    except subprocess.TimeoutExpired:
        server.kill()
    log.close()
os._exit(0)
