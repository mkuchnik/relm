"""Perform indexing across a test_relm instance.

Contains visualization and tensor operations.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import scipy.stats

import relm.relm_logging

logger = relm.relm_logging.get_logger()

EMPTY_TOKEN = -1
START_STRING = "<start>"
STRING_DELIMITER = " "


def isPrimitive(obj):
    """Return true if primitive type."""
    return not hasattr(obj, '__dict__')


def top_N_elements_idx(a, n):
    """Get the first n highest values."""
    return np.argpartition(a, -n)[-n:]


def path_to_top_k(test_relm, path):
    """Map a token path to a list of top-k values.

    :param test_relm: The test_relm to use to access top-k.
    :param path: The tuple of tokens for the path.
    :return A top-k rank for each token.
    """
    path_probs = test_relm.point_query_tokens(
        path, return_all_probs=True)
    prob_path_ranks = relm.indexing.get_top_k_ranks(
        path_probs)
    prob_path_ranks = np.take_along_axis(
        prob_path_ranks,
        np.array(path)[:, None],
        axis=-1).flatten()
    return prob_path_ranks


def get_top_k_mask(a, k):
    """Get the top k elements of a 2d matrix."""
    assert len(a.shape) == 2
    ind = np.argpartition(a, -k, axis=-1)[:, -k:]
    # Flip so first element is maximal, second is second max, etc.
    ind = np.fliplr(ind)
    return ind


def rank_array(a, axis=None, reverse=False):
    """Get the rank of all elements in array.

    By default, the smallest element will get rank 1 and biggest will get rank
    N. Set reverse to return the biggest to smallest rank.

    @param a The numpy vector or matrix
    @param axis The axis to rank over
    @param reverse Sort from biggest to smallest
    """
    if reverse:
        aa = a * -1
    else:
        aa = a
    ranks = scipy.stats.rankdata(aa, method="ordinal", axis=axis)
    return ranks


def get_element_rank(a, element_idx, reverse=False):
    """Get the rank of the element at element_idx.

    Note: This starts at rank 1 rather than 0.
    """
    ranks = rank_array(a, reverse=reverse)
    element_rank = ranks[element_idx]
    return element_rank


def get_top_k_ranks(a):
    """Get the ranks of each element in the matrix.

    For example, if a is n*m, where n is the number of steps/samples and m is
    the number of vocabulary items, then for each of the n steps, we will
    assign a rank where 0 is most likely.
    """
    assert len(a.shape) == 2, "Expected 2d array of probabilities"
    # These ranks are inverted where max index is assigned to most likely
    ranks = rank_array(a, axis=-1, reverse=True)
    return ranks
