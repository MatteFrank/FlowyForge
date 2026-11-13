"""Epoch-boundary RNG snapshots and atomic checkpoints shared by both runners."""

import hashlib
import json
import random

import numpy as np
import torch


def _rng_state(device, loaders):
    numpy_state = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
             "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()}}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state, device, loaders):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    for name, loader in loaders.items():
        loader.generator.set_state(state["loaders"][name])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def state_digest(payload):
    """Fingerprint nested primitive/tensor state, independent of torch.save ZIP names."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode() + b"[")
            for item in value:
                visit(item)
            digest.update(b"]")
        else:
            digest.update(json.dumps(value, allow_nan=False).encode() + b";")

    visit(payload)
    return digest.hexdigest()
