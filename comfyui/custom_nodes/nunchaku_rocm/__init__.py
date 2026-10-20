"""
ComfyUI custom nodes for nunchaku-rocm (Nunchaku SVDQuant models on AMD GPUs).

Install by adding ``comfyui/custom_nodes`` of this repo as a ComfyUI custom-nodes path (see ../../README.md).
"""

import os

# select the Triton backend before nunchaku is first imported; NUNCHAKU_BACKEND=cuda can still override
os.environ.setdefault("NUNCHAKU_BACKEND", "triton")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
