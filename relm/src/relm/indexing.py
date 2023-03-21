"""Perform indexing across a test_relm instance.

Contains visualization and tensor operations.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats
import torch

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


def _safe_add_edge(G, a, b, **kwargs):
    """Add edge only if types are primitive."""
    assert isPrimitive(a)
    assert isPrimitive(b)
    for v in kwargs.values():
        assert isPrimitive(v)
    G.add_edge(a, b, **kwargs)


def relm_to_nx_depth_first_search(test_relm, max_depth=None,
                                  keep_top_k_probs=None):
    """Use DFS to build a DiGraph from a test_relm."""
    if max_depth is None:
        max_depth = 6

    def _visit_node(test_relm, state_tokens, G, max_depth, keep_top_k_probs):
        """Use DFS to visit test_relm nodes."""
        transition_probabilities = test_relm._next_token_query_tokens(
            state_tokens)
        if state_tokens is not None:
            existing_tokens = state_tokens[0].tolist()
        else:
            existing_tokens = []
        existing_tokens_str = STRING_DELIMITER.join(
            (str(x) for x in existing_tokens))
        if not existing_tokens_str:
            existing_tokens_str = START_STRING
        if keep_top_k_probs:
            top_elements = top_N_elements_idx(
                transition_probabilities, keep_top_k_probs)
            transition_probabilities = transition_probabilities[top_elements]
            transition_iterator = zip(top_elements, transition_probabilities)
        else:
            transition_iterator = enumerate(transition_probabilities)
        for node, edge in transition_iterator:
            new_state_tokens = existing_tokens + [node.item()]
            new_tokens_str = STRING_DELIMITER.join(
                (str(x) for x in new_state_tokens))
            _safe_add_edge(
                G,
                existing_tokens_str,
                new_tokens_str,
                weight=float(edge))
            new_state_tokens = torch.LongTensor([new_state_tokens])
            if len(new_state_tokens[0]) < max_depth:
                _visit_node(
                    test_relm,
                    new_state_tokens,
                    G,
                    max_depth,
                    keep_top_k_probs)
        return G

    G = nx.DiGraph()
    state = None
    _visit_node(test_relm, state, G, max_depth, keep_top_k_probs)
    return G


def generic_relm_to_nx_depth_first_search(test_relm, max_depth=None,
                                          transition_filter_fn=None):
    """Use DFS to build a DiGraph from a test_relm.

    This is supposed to be more generic than relm_to_nx_depth_first_search

    :param test_relm: The test relm used for the search.
    :param max_depth: The depth to terminate at.
    :param transition_filter_fn: A function from state and
    transition probabilities to
    an iterator of viable transition probabilities indicies and their values.
    :return A networkX graph
    """
    if max_depth is None:
        max_depth = 6

    # TODO(mkuchnik): Improve CUDA performance

    def _visit_node(test_relm, state_tokens, G, max_depth):
        """Use DFS to visit test_relm nodes."""
        logger.debug("Expanding transitions (DFS visit): {}".format(
            state_tokens))
        transition_probabilities = test_relm._next_token_query_tokens(
            state_tokens)
        if state_tokens is not None:
            existing_tokens = state_tokens[0].tolist()
        else:
            existing_tokens = []
        existing_tokens_str = STRING_DELIMITER.join(
            (str(x) for x in existing_tokens))
        if not existing_tokens_str:
            existing_tokens_str = START_STRING
        if transition_filter_fn:
            transition_iterator = transition_filter_fn(
                state_tokens,
                transition_probabilities)
        else:
            transition_iterator = enumerate(transition_probabilities)
        for node, edge in transition_iterator:
            if not isinstance(node, int):
                # Value may be still tensor
                node = node.item()
            new_state_tokens = existing_tokens + [node]
            new_tokens_str = STRING_DELIMITER.join(
                (str(x) for x in new_state_tokens))
            _safe_add_edge(
                G,
                existing_tokens_str,
                new_tokens_str,
                weight=float(edge))
            new_state_tokens = torch.tensor(
                [new_state_tokens], dtype=torch.int64,
                device=relm.device())
            if len(new_state_tokens[0]) < max_depth:
                _visit_node(
                    test_relm,
                    new_state_tokens,
                    G,
                    max_depth
                )
        return G

    G = nx.DiGraph()
    state = None
    _visit_node(test_relm, state, G, max_depth)
    return G


def _build_nx_node_name_map(test_relm, G):
    """Decode a nx DiGraph with test_relm so the names can be plotted."""
    node_map = {}
    for n, d in G.nodes(data=True):
        n_vals = n.split(STRING_DELIMITER)
        if n != START_STRING:
            val = str(test_relm._decode_gen_sequence([int(n_vals[-1])]))
        else:
            val = START_STRING
        node_map[str(n)] = {"name": val}
    return node_map


def visualize_nx_graph(test_relm, G, add_attributes=True):
    """Plot a DiGraph using test_relm for labels.

    If add_attributes is true, add labels to the DiGraph using test_relm.
    """
    if add_attributes:
        node_map = _build_nx_node_name_map(test_relm, G)
        nx.set_node_attributes(G, node_map)
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato", root=START_STRING)
    labels = nx.get_node_attributes(G, 'name')
    # Color edges by weight
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    plt.figure(3, figsize=(12, 12))
    nx.draw(
        G,
        pos,
        labels=labels,
        edge_color=weights,
        node_size=500,
        width=2.0,
        edge_cmap=plt.cm.Reds)
