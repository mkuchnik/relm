"""Utilities for notebooks, benchmarking, and metrics."""

import itertools
import sys
import time
from difflib import ndiff

import pandas as pd
import tqdm
import tqdm.notebook


def isnotebook() -> bool:
    """Return true if Ipython Notebook.

    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def _get_tqdm_library():
    # NOTE(mkuchnik): There is also autonotebook
    if isnotebook():
        return tqdm.notebook
    else:
        return tqdm


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


def serialize_iterator(filename, iterator, num_samples,
                       column_names, wrap_pbar=True):
    """Save samples of iterator to file."""
    samples = sample_iterator(iterator, num_samples, wrap_pbar=True)
    df = samples_to_dataframe(samples, column_names)
    df.to_csv(filename, index=False)


def deserialize_iterator(filename):
    """Invert serialization of serialize_iterator from file."""
    df = pd.read_csv(filename)
    iterator = dataframe_to_samples(df)
    return iterator


def samples_to_dataframe(samples, column_names):
    """Save samples to dataframe."""
    df = pd.DataFrame(data=samples,
                      columns=column_names)
    return df


def dataframe_to_samples(df):
    """Yield samples from dataframe."""
    for row in df.itertuples(index=False):
        yield tuple(row)


def levenshtein_distance(str1: str, str2: str):
    """Return the levenshtein distance between two strings."""
    return levenshtein_distance_dp(str1, str2)


def levenshtein_distance_diff(str1: str, str2: str):
    """Return the levenshtein distance between two strings."""
    # https://codereview.stackexchange.com/questions/217065/calculate-levenshtein-distance-between-two-strings-in-python
    counter = {"+": 0, "-": 0}
    distance = 0
    for edit_code, *_ in ndiff(str1, str2):
        if edit_code == " ":
            distance += max(counter.values())
            counter = {"+": 0, "-": 0}
        else:
            counter[edit_code] += 1
    distance += max(counter.values())
    return distance


def levenshtein_distance_dp(s: str, t: str):
    """Return the levenshtein distance between two strings."""
    # https://codereview.stackexchange.com/questions/269117/levenshtein-distance-using-dynamic-programming-in-python-3
    m, n = len(s) + 1, len(t) + 1
    d = [[0] * n for _ in range(m)]

    for i in range(1, m):
        d[i][0] = i

    for j in range(1, n):
        d[0][j] = j

    for j in range(1, n):
        for i in range(1, m):
            substitution_cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + substitution_cost)

    return d[m - 1][n - 1]
