from __future__ import annotations

import hashlib

import numpy as np

_KeyPart = int | str


def _key_to_int(key: _KeyPart) -> int:
    if isinstance(key, (int, np.integer)):
        value = int(key)
        if value < 0:
            raise ValueError(f"integer rng key parts must be non-negative, got {value}")
        return value
    if isinstance(key, str):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)
    raise TypeError(f"rng key parts must be int or str, got {type(key).__name__}")


def child(seed: int, *keys: _KeyPart) -> np.random.Generator:
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    spawn_key = tuple(_key_to_int(k) for k in keys)
    seed_sequence = np.random.SeedSequence(entropy=int(seed), spawn_key=spawn_key)
    return np.random.default_rng(seed_sequence)
