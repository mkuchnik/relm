"""Utilities for notebooks, benchmarking, and metrics.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import functools
import itertools
import sys
import time

import tqdm
import tqdm.notebook


def _get_tqdm_library():
    return tqdm.autonotebook


def benchmark_iterator(iterator, num_samples=None, wrap_pbar=True):
    """Benchmarks an iterator and wraps it with convenience functions."""
    total = None
    it = iterator
    if num_samples:
        if not isinstance(num_samples, int):
            raise ValueError("Expect integer, but got: {}".format(num_samples))

        total = num_samples
        it = itertools.islice(it, num_samples)
    if wrap_pbar:
        it = _get_tqdm_library().tqdm(it, total=total, file=sys.stdout,
                                      position=0, leave=True)
    return _benchmark_iterator(it)


def _benchmark_iterator(iterator):
    """Benchmarks an iterator."""
    start_time = time.perf_counter()
    for num_samples, _ in enumerate(iterator):
        pass
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return num_samples / total_time


def sample_iterator(iterator, num_samples, wrap_pbar=True):
    """Sample (at most) num_samples from the iterator."""
    it = itertools.islice(iterator, num_samples)
    if wrap_pbar:
        with _get_tqdm_library().tqdm(total=num_samples, file=sys.stdout,
                                      position=0, leave=True) as pbar:
            for x in it:
                pbar.update(1)
                yield x
    else:
        yield from it


def levenshtein_distance(str1: str, str2: str):
    """Return the levenshtein distance between two strings."""

    @functools.lru_cache(maxsize=(len(str1) + 1) * (len(str2) + 1))
    def helper(i, j):
        if i <= 0:
            return max(j, 0)
        elif j <= 0:
            return max(i, 0)
        else:
            if str1[i-1] == str2[j-1]:
                return helper(i-1, j-1)
            else:
                return min(
                    helper(i-1, j),
                    helper(i, j-1),
                    helper(i-1, j-1)
                ) + 1

    return helper(len(str1), len(str2))
