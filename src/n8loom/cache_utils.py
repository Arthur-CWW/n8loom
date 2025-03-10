from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

KVFrag = namedtuple("KVFrag", ["keys", "values"])


def mx_copy(x: mx.array) -> mx.array:
    if x is None:
        raise ValueError("Cannot copy None array")
    return mx.array(np.array(x))


def frag_cache(
    cache: List[KVCache], start_idx: int = 0, end_idx: int = -1
) -> List[KVFrag]:
    """Extracts and converts a slice of key-value pairs from model layer caches into fragments.

    Args:
            cache: List of KVCache objects, one per model layer, containing cached key-value pairs
            start_idx: Starting index for extraction (default: 0)
            end_idx: Ending index for extraction (default: -1)

    Returns:
            List of KVFrag objects, one per layer, each containing the extracted keys and values
            arrays from the specified index range

    Example:
            >>> layer_caches = [KVCache(...), KVCache(...)]  # List of caches for each layer
            >>> fragments = frag_cache(layer_caches, 0, 10)  # Get first 10 positions
    """
    if not cache:
        raise ValueError("Cache list cannot be empty")

    frags = []
    for layer_cache in cache:
        if layer_cache.keys is None or layer_cache.values is None:
            raise ValueError("Layer cache contains None keys or values")
        keys = mx_copy(layer_cache.keys[:, :, start_idx:end_idx])
        values = mx_copy(layer_cache.values[:, :, start_idx:end_idx])
        frags.append(KVFrag(keys, values))
    return frags


def clip_frag(frags: List[KVFrag], token_limit: int) -> List[KVFrag]:
    """Clips a list of key-value fragments to a specified token limit.

    Args:
            frags: List of KVFrag objects - one per layer - to clip
            token_limit: Maximum number of tokens to retain in each fragment

    Returns:
            List of KVFrag objects, each containing the clipped keys and values arrays
    """
    if not frags:
        raise ValueError("Fragment list cannot be empty")

    clipped_frags = []
    for frag in frags:
        if frag.keys is None or frag.values is None:
            raise ValueError("Fragment contains None keys or values")
        keys = mx_copy(frag.keys[:, :, :token_limit])
        values = mx_copy(frag.values[:, :, :token_limit])
        clipped_frags.append(KVFrag(keys, values))
    return clipped_frags


def frag_batch_gen(
    cache: List[KVCache], total_prompt_len: int, generated_lengths: List[int]
) -> List[List[KVFrag]]:
    if not cache or not cache[0].keys is not None:
        raise ValueError("Cache list cannot be empty and must contain valid keys")

    frags = []
    B = cache[0].keys.shape[0]
    for i in range(B):
        batch_frags = []
        for layer_cache in cache:
            if layer_cache.keys is None or layer_cache.values is None:
                raise ValueError("Layer cache contains None keys or values")
            keys = mx_copy(
                layer_cache.keys[
                    i : i + 1,
                    :,
                    total_prompt_len : total_prompt_len + generated_lengths[i],
                ]
            )
            values = mx_copy(
                layer_cache.values[
                    i : i + 1,
                    :,
                    total_prompt_len : total_prompt_len + generated_lengths[i],
                ]
            )
            batch_frags.append(KVFrag(keys, values))
        frags.append(batch_frags)
    return frags


def fuse_cache_frags(frags: List[List[KVFrag]]) -> List[KVCache]:
    """Fuses a list of key-value fragments into a list of model layer caches.

    Args:
            frags: List of lists of KVFrag objects - first dimension is the layer index, second is the list of fragments to merge

    Returns:
            List of KVCache objects, one per model layer, containing the fused key-value pairs from the fragments, concatenated along the sequence dimension.

    Example:
            >>> fragments = [[KVFrag(...), KVFrag(...)], [KVFrag(...), KVFrag(...)]]
            >>> layer_caches = fuse_cache_frags(fragments)
    """
    if not frags:
        raise ValueError("Fragment list cannot be empty")

    caches = []
    for layer_frags in frags:
        if not layer_frags:
            raise ValueError("Layer fragments cannot be empty")
        keys = mx.concat(
            [frag.keys for frag in layer_frags if frag.keys is not None], axis=2
        )
        values = mx.concat(
            [frag.values for frag in layer_frags if frag.values is not None], axis=2
        )
        cache = KVCache()
        cache.keys = keys
        cache.values = values
        cache.offset = keys.shape[2]
        caches.append(cache)
    return caches
