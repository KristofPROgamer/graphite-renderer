"""Deterministic, order-independent child random number generators.

The pipeline runs many independent stochastic sub-steps (per-layer, per-
stroke, per-tile...). To keep the whole run reproducible given (image,
config, seed) *without* making results depend on the order those sub-steps
happen to execute in, every consumer derives its own
:class:`numpy.random.Generator` from the single global seed plus a tuple of
"key" values that describe *what* it is (e.g. ``rng.child(seed, "L1_tone",
layer_index)``), never *when* it runs.

This works because :class:`numpy.random.SeedSequence` with the same
``entropy`` and ``spawn_key`` always produces the same generator state,
regardless of how many other ``SeedSequence`` objects have been created
before or after it.
"""

from __future__ import annotations

import hashlib

import numpy as np

_KeyPart = int | str


def _key_to_int(key: _KeyPart) -> int:
    """Map a str/int key part to a stable non-negative integer.

    Strings are hashed with SHA-256 (truncated to 64 bits) rather than
    Python's builtin ``hash()``, which is randomized per-process unless
    ``PYTHONHASHSEED`` is fixed and therefore not safe for reproducibility.
    """
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
    """Derive a reproducible child ``Generator`` from ``seed`` and ``keys``.

    Args:
        seed: the pipeline's single global seed.
        *keys: any number of int/str tags identifying this stream, e.g.
            ``rng.child(seed, "hatching", region_name, layer_index)``.

    Returns:
        A ``numpy.random.Generator`` whose stream depends only on
        ``(seed, keys)`` -- not on call order, not on how many other
        children have already been spawned.

    Example:
        >>> a = child(0, "layer", 1)
        >>> b = child(0, "layer", 1)
        >>> a.random() == b.random()
        False
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    spawn_key = tuple(_key_to_int(k) for k in keys)
    seed_sequence = np.random.SeedSequence(entropy=int(seed), spawn_key=spawn_key)
    return np.random.default_rng(seed_sequence)
