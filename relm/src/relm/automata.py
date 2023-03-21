"""Build automata from patterns to match patterns."""

import collections
import enum
import functools
import heapq
import inspect
import math
import multiprocessing
import pprint
import random
import typing

import networkx as nx
import numpy as np
import pywrapfst as fst
import torch
from scipy.sparse import coo_matrix
from torch.profiler import record_function

import relm.relm_logging

START_STRING = "<start>"
STRING_DELIMITER = " "

logger = relm.relm_logging.get_logger()


def _check_type_compatible(value, expected_types):
    if not isinstance(value, expected_types):
        raise ValueError("Unsupported type: {} for '{}'. Expected {}".format(
            type(value), value, expected_types))


def _check_automata_valid(automata):
    if not automata.verify():
        raise RuntimeError("Automata {} failed verification.".format(automata))


def _check_valid_token_values(list_of_tokens):
    if any(x == 0 for x in list_of_tokens):
        raise ValueError("Zero found in tokens: {}".format(list_of_tokens))


def automata_from_token_list(list_of_tokens):
    """Create an automata from a list of tokens.

    The automaton should accept the list.

    For example, [1, 2, 3] will create an automata that accepts "1 2 3".

    :param list list_of_tokens: A python list or tuple containing the sequence
    of tokens to accept
    :return: An openfst automata
    """
    _check_type_compatible(list_of_tokens, (list, tuple))
    _check_valid_token_values(list_of_tokens)
    # NOTE(mkuchnik): Alternatively, we can use pynini with accep method using
    # the special bracket syntax e.g., "[1][2][3]" which should convert the
    # bracket content to integer.
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    zero = fst.Weight.zero(f.weight_type())
    f.reserve_states(len(list_of_tokens))
    states = []

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)
    states.append(start_state)

    # Transition states
    for i, t in enumerate(list_of_tokens):
        last_state = states[i]
        s = f.add_state()
        f.reserve_arcs(last_state, 1)  # Optional.
        f.add_arc(last_state, fst.Arc(t, t, one, s))
        states.append(s)

    # Let's set all states to have non-accepting final weight
    for s in states:
        f.set_final(s, zero)

    # End states
    end_state = states[-1]
    # By default, state gets infinite weight or zero weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def union_automata_from_token_list(list_of_tokens):
    """Create a union automata from a list of tokens.

    The automaton should accept any of the tokens in the list.

    For example, [1, 2, 3] will create an automata that accepts "1" or "2" or
    "3".

    :param list list_of_tokens: A python list or tuple containing the sequence
    of tokens to accept
    :return: An openfst automata
    """
    _check_type_compatible(list_of_tokens, (list, tuple, set))
    _check_valid_token_values(list_of_tokens)
    list_of_tokens = set(list_of_tokens)
    # NOTE(mkuchnik): Alternatively, we can use pynini with accep method using
    # the special bracket syntax e.g., "[1][2][3]" which should convert the
    # bracket content to integer.
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    zero = fst.Weight.zero(f.weight_type())
    f.reserve_states(2)

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)

    # Transition state
    end_state = f.add_state()
    f.reserve_arcs(start_state, len(list_of_tokens))  # Optional.
    for t in list_of_tokens:
        f.add_arc(start_state, fst.Arc(t, t, one, end_state))

    # Let's set all states to have non-accepting final weight
    f.set_final(start_state, zero)

    # By default, state gets infinite weight or zero weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def union_automatas(automatas,
                    determinize=True,
                    minimize=True,
                    rm_epsilon=True,
                    verify=True):
    """Take the union of a list of automata."""
    union_automata = None
    for automata in automatas:
        if union_automata:
            union_automata = union_automata.union(automata)
        else:
            union_automata = automata
    if union_automata:
        union_automata = finalize_automata(
            union_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
        return union_automata
    else:
        raise ValueError("Automatas are empty")


def concat_automatas(automatas,
                     determinize=True,
                     minimize=True,
                     rm_epsilon=True,
                     verify=True):
    """Take the concatenation of a list of automata."""
    concat_automata = None
    for automata in automatas:
        if concat_automata:
            concat_automata = concat_automata.concat(automata)
        else:
            concat_automata = automata
    if concat_automata:
        concat_automata = finalize_automata(
            concat_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
    else:
        raise ValueError("Automatas are empty")
    return concat_automata


def null_automata():
    """Return an automata that is null."""
    return automata_from_token_list([])


def optional_automata(automata,
                      determinize=True,
                      minimize=True,
                      rm_epsilon=True,
                      verify=True):
    """Return an automata that matches on the given automata or nothing."""
    null_auto = null_automata()
    automatas = [automata, null_auto]
    union_automata = union_automatas(automatas,
                                     determinize,
                                     minimize,
                                     rm_epsilon,
                                     verify)
    return union_automata


def repeat_automata(automata,
                    min_length,
                    max_length,
                    determinize=True,
                    minimize=True,
                    rm_epsilon=True,
                    verify=True):
    """Return an automata that matches on the given automata repeated.

    For basic automata (e.g., strings), this functions as expected.
    For example, 'abc' can be repeated to '', 'abc', 'abcabc'.
    For complicated automata (e.g., a union), the repeats can traverse any of
    the paths. Care must be taken to ensure that the subexpression that is
    supposed to be repeated rather than the top-level expression.

    :param min_length The minimum length to accept on. If 0, accepts empty
    strings.
    :param max_length The maximum length to accept on. If None, accepts
    infinite length strings.
    :return An openfst automata.
    """
    if min_length is None:
        raise ValueError("min_length can't be None")
    if min_length < 0:
        raise ValueError("min_length can't be negative. Got {}".format(
            min_length))
    if max_length is not None and max_length < 0:
        raise ValueError("max_length can't be negative. Got {}".format(
            max_length))
    if max_length is not None and max_length < min_length:
        raise ValueError("Max length can't be less than min_length, but found"
                         " max_length={} and min_length={}".format(
                             max_length, min_length))

    def _repeat_automata_fn(automata_to_repeat, repeat_times):
        if repeat_times < 0:
            raise ValueError("repeat_times must be positive."
                             " Got {}".format(repeat_times))
        if repeat_times == 0:
            return null_automata()
        elif repeat_times == 1:
            return automata_to_repeat.copy()
        else:
            repeat_automata = automata_to_repeat.copy()
            for _ in range(repeat_times - 1):
                repeat_automata = repeat_automata.concat(
                    automata_to_repeat)
            return repeat_automata

    def _repeat_up_to_fn(automata_to_repeat, repeat_times):
        if repeat_times < 0:
            raise ValueError("repeat_times must be positive."
                             " Got {}".format(repeat_times))
        if repeat_times == 0:
            return null_automata()
        elif repeat_times == 1:
            return automata_to_repeat.copy().closure(closure_type="plus")
        else:
            # 1 copy
            repeat_automata = automata_to_repeat.copy()
            for _ in range(repeat_times - 1):
                # Plus N - 1 optional copies
                repeat_automata = repeat_automata.concat(
                    optional_automata(automata_to_repeat.copy()))
            return repeat_automata

    if min_length == 0 and max_length is None:
        repeat_automata = automata.copy().closure(closure_type="star")
        repeat_automata = finalize_automata(
            repeat_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
        return repeat_automata
    elif min_length == 1 and max_length is None:
        repeat_automata = automata.copy().closure(closure_type="plus")
        repeat_automata = finalize_automata(
            repeat_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
        return repeat_automata
    elif max_length is not None:
        # Finite repeat
        diff_length = max_length - min_length
        if min_length:
            prefix_auto = _repeat_automata_fn(automata, min_length)
            # Next symbols have final states
            repeat_automata = prefix_auto
            # Allow Diff copies
            for _ in range(diff_length):
                # Plus N - 1 optional copies
                repeat_automata = repeat_automata.concat(
                    optional_automata(automata,
                                      determinize=False,
                                      minimize=False)
                )
        else:
            # Allow 0 copies
            repeat_automata = optional_automata(automata)
            # Allow Diff copies
            for _ in range(diff_length - 1):
                # Plus N - 1 optional copies
                repeat_automata = repeat_automata.concat(
                    optional_automata(automata,
                                      determinize=False,
                                      minimize=False)
                )
        repeat_automata = finalize_automata(
            repeat_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
        return repeat_automata
    else:
        assert max_length is None
        # Infinite repeat with prefix
        prefix_auto = _repeat_automata_fn(automata, min_length)
        repeat_automata = automata.closure(closure_type="star")
        repeat_automata = prefix_auto.concat(repeat_automata)
        repeat_automata = finalize_automata(
            repeat_automata,
            determinize=determinize,
            minimize=minimize,
            rm_epsilon=rm_epsilon,
            verify=verify)
        return repeat_automata


def automata_union_from_list_of_token_list(list_of_list_of_tokens,
                                           determinize=True,
                                           minimize=True,
                                           rm_epsilon=True,
                                           verify=True):
    """Create an automata accepting any of the list of tokens.

    For example, {(1, 2, 3), (4)} can accept either "1 2 3" or "4".

    :param list list_of_list_of_tokens: A set of tuples. Each tuple is
    equivalent in terms of decoding.
    :param bool determinize: Whether to return a (near) DFA.
    :param bool minimize: Whether to return a minimal automata.
    :param bool rm_epsilon: Whether to remove all epsilon transitions.
    :param bool verify: Whether to verify the automata after construction.
    :return: The automata that accepts all sets of tuples.
    """
    _check_type_compatible(list_of_list_of_tokens, (list, set))
    automatas = []
    for list_of_tokens in list_of_list_of_tokens:
        automata = automata_from_token_list(list_of_tokens)
        automatas.append(automata)

    # Union them all together
    union_automata = automatas[0]
    for automata in automatas[1:]:
        union_automata = union_automata.union(automata)

    union_automata = finalize_automata(
        union_automata,
        determinize=determinize,
        minimize=minimize,
        rm_epsilon=rm_epsilon,
        verify=verify)

    return union_automata


def finalize_automata(automata,
                      determinize=True,
                      minimize=True,
                      rm_epsilon=True,
                      verify=True):
    """Run determinization, minimzation, and other verification."""
    for i in range(2):
        if determinize:
            automata = fst.determinize(automata)

        if minimize:
            automata = automata.minimize()

        if rm_epsilon:
            automata = automata.rmepsilon()

    if verify:
        _check_automata_valid(automata)

    return automata


def automata_concatenated_union_from_list_of_list_of_token_list(
        list_of_list_of_list_of_tokens,
        determinize=True,
        minimize=True,
        rm_epsilon=True,
        verify=True):
    """Concatenate unions together to match a sentence.

    Input is a list of sets of tuples representing the possible tokenizations.
    """
    union_automatas = []
    for list_of_list_of_tokens in list_of_list_of_list_of_tokens:
        automata = automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
        )
        union_automatas.append(automata)

    concatenated_automata = union_automatas[0]
    for automata in union_automatas[1:]:
        concatenated_automata = concatenated_automata.concat(automata)

    concatenated_automata = finalize_automata(
        concatenated_automata,
        determinize=determinize,
        minimize=minimize,
        rm_epsilon=rm_epsilon,
        verify=verify)

    return concatenated_automata


def _iterate_networkx_DAG_paths(G):
    """Iterate all paths from source to sinks.

    https://stackoverflow.com/questions/55711945/networkx-getting-all-possible-paths-in-dag
    """
    roots = []
    leaves = []
    for node in G.nodes:
        if G.in_degree(node) == 0:  # it's a root
            roots.append(node)
        elif G.out_degree(node) == 0:  # it's a leaf
            leaves.append(node)

    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(G, root, leaf):
                yield path


def networkx_to_openfst_automata(G, determinize=True, minimize=True,
                                 rm_epsilon=True,
                                 verify=True):
    """Convert a networkx digraph to an openfst automata."""
    automatas = []
    for path in _iterate_networkx_DAG_paths(G):
        # TODO(mkuchnik): We take advantage of the node label encoding, though
        # we should refine this algorithm to be agnostic of encoding
        p = path[-1]
        list_of_tokens = p.split(" ")
        list_of_tokens = list(map(int, list_of_tokens))
        automata = automata_from_token_list(list_of_tokens)
        automatas.append(automata)

    # Union them all together
    union_automata = automatas[0]
    for automata in automatas[1:]:
        union_automata = union_automata.union(automata)

    union_automata = finalize_automata(
        union_automata,
        determinize=determinize,
        minimize=minimize,
        rm_epsilon=rm_epsilon,
        verify=verify)

    return union_automata


def openfst_automata_to_networkx(automata):
    """Convert an openfst automata to a networkX digraph.

    See `_draw_automata_with_networkx` for a drawing example.
    """
    dod = automata_to_dict_of_dicts(automata)
    G = nx.from_dict_of_dicts(dod, create_using=nx.DiGraph)
    return G


def _draw_automata_with_networkx(automata):
    """Draw an automata as if it were a networkX plot."""
    G = relm.automata.openfst_automata_to_networkx(automata)
    nx.draw(G, with_labels=True)
    edge_labels = {}
    for u, v, a in G.edges(data=True):
        edge_labels[(u, v)] = a["edge_label"]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red'
    )


def _find_parents_of_state(automata, state, return_arcs=False):
    """Find the nodes that point to the state.

    Note: the time complexity is O(V*E), so this is expensive.
    """
    parents = []
    for s in automata.states():
        for a in automata.arcs(s):
            if a.nextstate == state:
                if return_arcs:
                    parents.append((s, a))
                else:
                    parents.append(s)
    return parents


def _find_parents_of_all_states(automata, return_arcs=False):
    """Find the nodes that point to all states e.g., backward pointers.

    The time complexity is O(V*E), so this is expensive, but is much better
    than calling _find_parents_of_state in a loop.
    """
    parents = {s: [] for s in automata.states()}
    for s in automata.states():
        for a in automata.arcs(s):
            if return_arcs:
                parents[a.nextstate].append((s, a))
            else:
                parents[a.nextstate].append(s)
    return parents


def contains_epsilon(automata):
    """Return true if epsilon transition in automata."""
    for s in automata.states():
        for a in automata.arcs(s):
            if a.ilabel == 0 or a.olabel == 0:
                return True
    return False


def remap_edge_values(automata, edges_to_new_values: dict):
    """Remaps the edges of automata using the provided dict.

    :param edges_to_new_values A dict from edge label (int) to int or
    tuple[int]. If tuple, the values are expanded to new nodes.

    For example, if edges_to_new_values maps N to N+1, the
    edges will be N+1 for the returned automata.
    """
    automata = automata.copy()
    one = fst.Weight.one(automata.weight_type())
    automata_old_states = list(automata.states())
    # NOTE(mkuchnik): Arcs will be stale/deleted
    for s in automata_old_states:
        old_arcs = map(lambda x: x.copy(), automata.arcs(s))
        automata.delete_arcs(s)
        for a in old_arcs:
            assert a.ilabel == a.olabel, "Expected labels to be the same"
            try:
                new_value = edges_to_new_values[a.ilabel]
            except KeyError as ex:
                logger.error("'{}' not found in dict:\n{}".format(
                    a.ilabel, pprint.pformat(edges_to_new_values)))
                raise ex
            if isinstance(new_value, tuple):
                # Vector
                # Here, we assume the original node is has properties of the
                # last node we add. For example, if the original node was a
                # final node, we have to propogate that to the end.
                # For example, we may have:
                # o -> * -> ...
                # which should be changed to:
                # o -> o -> * -> ...
                # Therefore, we should add new "dummy" nodes into the prefix

                # Here, we leverage epsilon transitions
                # We will find the relevant arc
                # o -> *
                # We can then espilon expand it (e->)
                # o e-> * -> * e-> *
                # Notice, that we have ``dummy transitions'' (shown using |)
                # o e-> | * -> * | e-> *
                # Now, we can expand the dummy portion with what we want
                # o e-> | * -> * -> * -> * | e-> *
                # We can do that by leaving a gap in the middle
                # And pasting the edges in
                # o e-> | *              * | e-> *
                # o e-> | *    * -> *    * | e-> *
                # o e-> | * -> * -> * -> * | e-> *

                # NOTE(mkuchnik): in our implementation, we just directly add
                # the arcs without epsilons

                src_node = s
                tgt_node = a.nextstate

                curr_node = src_node
                for i, x in enumerate(new_value):
                    ilabel = x
                    olabel = x
                    if i < len(new_value) - 1:
                        new_node = automata.add_state()
                    else:
                        new_node = tgt_node
                    new_arc = fst.Arc(ilabel, olabel, one, new_node)
                    automata.add_arc(curr_node, new_arc)
                    curr_node = new_node

            elif isinstance(new_value, int):
                # Scalar
                a.ilabel = new_value
                a.olabel = new_value
                automata.add_arc(s, a)
            else:
                raise ValueError("Unknown type: {}".format(type(new_value)))
    return automata


def attach_symbol_table(automata, symbol_table, attach_inputs=True,
                        attach_outputs=True):
    """Attaches a symbol table to the automata.

    Symbol table can be a dict or list. If it's a dict, we'll assume the dict
    maps from symbol key to symbol value e.g., 1="the". Otherwise, we'll assume
    that the list is in order e.g., symbol_table[1]="the".
    """
    if not attach_inputs and not attach_outputs:
        raise ValueError("Either inputs or outputs should be attached")
    if isinstance(symbol_table, dict):
        _symbol_table = fst.SymbolTable()
        for k, v in symbol_table.items():
            try:
                _symbol_table.add_symbol(symbol=v, key=k)
            except TypeError as ex:
                logger.error("Error adding k={}, v={}".format(k, v))
                logger.error(ex)
                raise ex
        symbol_table = _symbol_table
    elif isinstance(symbol_table, list):
        _symbol_table = fst.SymbolTable()
        for k, v in enumerate(symbol_table):
            try:
                _symbol_table.add_symbol(symbol=v, key=k)
            except TypeError as ex:
                logger.error("Error adding k={}, v={}".format(k, v))
                logger.error(ex)
                raise ex
        symbol_table = _symbol_table
    elif not isinstance(symbol_table, (fst.SymbolTable, fst.SymbolTableView)):
        raise ValueError("Expected symbol table, got {}".format(
            type(symbol_table)))
    if attach_inputs:
        automata.set_input_symbols(symbol_table)
    if attach_outputs:
        automata.set_output_symbols(symbol_table)
    return automata


def automata_to_dict_of_dicts(automata):
    """Return a dict of dicts of the neighbors reachable by each state.

    dod = {0: {1: {"weight": 1}}}
    means there is an edge from 0 to 1 with weight 1.

    From:
    https://networkx.org/documentation/stable/reference/generated/networkx.convert.from_dict_of_dicts.html
    """
    connectivity_dict = dict()
    for state in automata.states():
        neighbors_dict = dict()
        for arc in automata.arcs(state):
            metadata_dict = {
                "edge_label": arc.ilabel,
                "weight": arc.weight,
            }
            if arc.nextstate in neighbors_dict:
                raise RuntimeError("Found parallel edge. Use multigraph.")
            neighbors_dict[arc.nextstate] = metadata_dict
        connectivity_dict[state] = neighbors_dict
    return connectivity_dict


def automata_to_connectivity_map(automata):
    """Return a dict of dicts of the neighbors reachable by each state.

    Notice that this differs from automata_to_dict_of_dicts because each state
    is keyed on ilabel.
    We also remove the metadata dict to just link to nextstate.
    """
    connectivity_dict = dict()
    for state in automata.states():
        neighbors_dict = dict()
        for arc in automata.arcs(state):
            neighbors_dict[arc.ilabel] = arc.nextstate
        connectivity_dict[state] = neighbors_dict
    return connectivity_dict


def automata_to_neighbor_connectivity_dict(automata):
    """Return a dict of the neighbors reachable by each state."""
    connectivity_dict = dict()
    for state in automata.states():
        neighbors = tuple([arc.nextstate for arc in automata.arcs(state)])
        connectivity_dict[state] = neighbors
    return connectivity_dict


def _is_cyclic_graph(g):
    """Return True if the directed graph g has a cycle.

    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    """
    # https://codereview.stackexchange.com/questions/86021/check-if-a-directed-graph-contains-a-cycle
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)


def is_cyclic_automata(automata):
    """Return True if the automata contains a cycle."""
    connectivity_dict = automata_to_neighbor_connectivity_dict(automata)
    return _is_cyclic_graph(connectivity_dict)


def _num_strings_counter_naive_fn(max_length, s, M, f) -> int:
    """Count automata strings with combinatorics.

    Takes starting state s, transition matrix M, and final states, f, and
    returns the number of max_length strings.

    This implementation uses the direct matrix exponentiation form.
    Recall that matrix-multiplying square matrices of size NxN takes O(N^3)
    time. If we are to do this procedure k times, we get O(k*N^3).

    Algorithm from:
    https://math.stackexchange.com/questions/2245373/find-the-number-of-accepted-matches-for-a-dfa-defined-by-a-regular-expression
    """
    return int(s.T.dot(M**max_length).dot(f).toarray()[0][0])


def _num_strings_counter_JNF_fn(max_length, s, M, f) -> int:
    """Count automata strings with combinatorics.

    Takes starting state s, transition matrix M, and final states, f, and
    returns the number of max_length strings.

    This implementation uses Jordan normal form, which reduces exponentiation
    over matrices to exponentiation of a list of numbers.


    Algorithm from:
    https://math.stackexchange.com/questions/2245373/find-the-number-of-accepted-matches-for-a-dfa-defined-by-a-regular-expression
    """
    raise NotImplementedError("Implement JNF")


def string_cardinality_generator_for_automata(automata, dtype=None):
    """Count the number of strings of length m in an automata.

    Algorithm from:
    https://math.stackexchange.com/questions/2245373/find-the-number-of-accepted-matches-for-a-dfa-defined-by-a-regular-expression
    """
    if dtype is None:
        dtype = np.longlong
    start_state = automata_start_state(automata, raise_exception=False)
    if start_state < 0:
        logger.warning("Empty automata encountered.")
        return lambda x: 0
    final_states = automata_final_states(automata)

    entries = []
    i = None
    for i, state in enumerate(automata.states()):
        arcs = list(automata.arcs(state))
        for arc in arcs:
            entries.append((state, arc.nextstate, 1))
    if i is None:
        # Automata is empty
        return 0
    max_states = i + 1
    row, col, data = zip(*entries)
    M = coo_matrix((data, (row, col)), shape=(max_states, max_states),
                   dtype=dtype)
    M = M.tocsr()
    entries = [(start_state, 0, 1)]
    row, col, data = zip(*entries)
    s = coo_matrix((data, (row, col)), shape=(max_states, 1),
                   dtype=dtype)
    s = s.tocsr()
    entries = [(ff, 0, 1) for ff in final_states]
    row, col, data = zip(*entries)
    f = coo_matrix((data, (row, col)), (max_states, 1),
                   dtype=dtype)
    f = f.tocsr()

    num_strings = functools.partial(_num_strings_counter_naive_fn,
                                    s=s,
                                    M=M,
                                    f=f)

    return num_strings


def string_cardinality_sum_for_automata(automata, max_length=None,
                                        parallelism=None, return_counts=False,
                                        fail_fast=None):
    """Count the total number of strings in an automata.

    If a max_length is not given, assumes all strings are valid. This is a
    problem is the automata has cycles, as then the count is infinite.

    :param automata: A openfst automata
    :param max length: The max length of strings to consider
    :param parallelism: Set to True to enable parallelism.
    :param return_counts: Set to True to return the array of counts.
    :return The number of strings generated by this automata
    """
    is_cyclic = is_cyclic_automata(automata)
    if not max_length and is_cyclic:
        raise ValueError("Automata is cyclic and max_length is not given.")
    num_strings_fn = string_cardinality_generator_for_automata(automata)

    # If automata is not cyclic, we can calculate the maximum path
    if max_length:
        if is_cyclic:
            # With cycle, we can travel to states infinite times
            max_length = max_length
        else:
            # TODO(mkuchnik): Test this code path
            max_path = max_path_distance(automata)
            max_length = min(max_length, max_path)
    else:
        max_length = max_path_distance(automata)

    def runner(num_strings_fn):
        total_count = 0
        string_length_iter = range(1, max_length + 1)
        if return_counts:
            all_counts_array = []

        if parallelism:
            logger.debug("Summing string length function in parallel")
            with multiprocessing.Pool() as pool:
                it = pool.map(num_strings_fn, string_length_iter)
                for i, x in enumerate(it):
                    if x < 0:
                        raise OverflowError("num_strings is negative ({})."
                                            " Overflow at i={}?".format(x, i))
                    total_count += x
                    if return_counts:
                        all_counts_array.append(x)
        else:
            for i in string_length_iter:
                logger.debug("Summing string length function: i={}".format(i))
                x = num_strings_fn(i)
                if x < 0:
                    raise OverflowError("num_strings is negative ({})."
                                        " Overflow at i={}?".format(x, i))
                total_count += x
                if return_counts:
                    all_counts_array.append(x)
        if return_counts:
            return total_count, all_counts_array
        else:
            return total_count

    try:
        return runner(num_strings_fn)
    except OverflowError as ex:
        logger.error(ex)
        logger.info(
            "Overflow maybe occured. Falling back to higher precision.")
        if not fail_fast:
            num_strings_fn = string_cardinality_generator_for_automata(
                automata, dtype=np.longdouble)
            return runner(num_strings_fn)
        else:
            return -1, None


def _empty_generator():
    """Return a generator that ends immediately."""
    logger.debug("Empty generator emitted.")
    return
    yield None


def topological_sort(automata, max_depth: typing.Optional[int] = None,
                     return_depth: bool = False):
    """Return a topological sort of the nodes (a list).

    Topological sort is a total order on the vertices such that:
    for every edge u -> v, u < v.

    Max depth is required for cyclic graphs and gives an unrolled view of the
    automata.

    Closely follows:
    https://en.wikipedia.org/wiki/Topological_sorting

    NOTE: we do not care about final states, only connectivity
    """
    if not automata.num_states():
        return []
    if max_depth is None and is_cyclic_automata(automata):
        raise RuntimeError(
            "Topological sort not possible with cycles without max_depth.")
    elif max_depth is not None:
        raise NotImplementedError(
            "max_depth is not implemented but is {}".format(max_depth))

    L = []
    to_visit_nodes = set(automata.states())
    open_nodes = set()
    visited_nodes = set()

    def visit(n, curr_depth=None):
        if curr_depth is None:
            curr_depth = 0
        if max_depth and curr_depth > max_depth:
            return
        if n in visited_nodes:
            return
        if n in open_nodes:
            raise RuntimeError("Not in DAG")
        open_nodes.add(n)
        for arc in automata.arcs(n):
            next_state = arc.nextstate
            visit(next_state, curr_depth + 1)
        open_nodes.remove(n)
        visited_nodes.add(n)
        to_visit_nodes.remove(n)
        L.append((n, curr_depth))

    # A path is valid if it ends with a final state
    start_state = automata_start_state(automata)
    visit(start_state)

    assert not to_visit_nodes

    if return_depth:
        return list(reversed(L))
    else:
        return list(map(lambda x: x[0], reversed(L)))


def BFS_from_automata(automata, return_edges_visited=False):
    """Breadth first search from an automata."""
    if not automata.num_states():
        return _empty_generator()

    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    def breadth_first_search():
        # Here "nodes" are paths in the automata
        # Tuples of (state, edges_visited)
        Q = collections.deque([((start_state,), tuple())])
        while Q:
            curr_path, edges_visited = Q.popleft()
            last_state = curr_path[-1]
            for arc in automata.arcs(last_state):
                next_edge = arc.ilabel
                next_state = arc.nextstate
                new_path = curr_path + (next_state,)
                new_edges_visited = edges_visited + (next_edge,)
                if new_path[-1] in final_states:
                    if return_edges_visited:
                        yield new_path, new_edges_visited
                    else:
                        yield new_path
                Q.append((new_path, new_edges_visited))

    return breadth_first_search()


def DFS_from_automata(automata, return_edges_visited=False, max_length=None):
    """Depth first search from an automata."""
    if not automata.num_states():
        return _empty_generator()
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    def depth_first_search():
        # Here "nodes" are paths in the automata
        # Tuples of (state, edges_visited)
        Q = collections.deque([((start_state,), tuple())])
        while Q:
            # NOTE(mkuchnik): Notice we pop() instead of popleft()
            curr_path, edges_visited = Q.pop()
            if not max_length or len(edges_visited) < max_length:
                last_state = curr_path[-1]
                for arc in automata.arcs(last_state):
                    next_edge = arc.ilabel
                    next_state = arc.nextstate
                    new_path = curr_path + (next_state,)
                    new_edges_visited = edges_visited + (next_edge,)
                    if new_path[-1] in final_states:
                        if return_edges_visited:
                            yield new_path, new_edges_visited
                        else:
                            yield new_path
                    Q.append((new_path, new_edges_visited))

    return depth_first_search()


class IDDFSMaxDepthException(Exception):
    """Throw when hitting max depth for an iterative deepening DFS round."""

    pass


def iterative_deepening_DFS_from_automata(
        automata, return_edges_visited=False, start_length=None):
    """Search with Iterative Deepending Depth first search from an automata."""
    if not automata.num_states():
        return _empty_generator()
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    def depth_first_search(max_depth):
        # Here "nodes" are paths in the automata
        # Tuples of (state, edges_visited)
        Q = collections.deque([((start_state,), tuple())])
        while Q:
            # NOTE(mkuchnik): Notice we pop() instead of popleft()
            curr_path, edges_visited = Q.pop()
            last_state = curr_path[-1]
            if len(curr_path) >= max_depth:
                raise IDDFSMaxDepthException(
                    "Max depth of {} >= {} hit".format(
                        len(curr_path), max_depth))
            for arc in automata.arcs(last_state):
                next_edge = arc.ilabel
                next_state = arc.nextstate
                new_path = curr_path + (next_state,)
                new_edges_visited = edges_visited + (next_edge,)
                if new_path[-1] in final_states:
                    if return_edges_visited:
                        yield new_path, new_edges_visited
                    else:
                        yield new_path
                Q.append((new_path, new_edges_visited))

    if start_length is None:
        curr_depth = 1
    else:
        curr_depth = start_length

    # Run until we exhaust DFS round
    still_running = True
    while still_running:
        ret = depth_first_search(curr_depth)
        try:
            for x in ret:
                yield x
            # We exhausted a DFS run without depth exception
            still_running = False
        except IDDFSMaxDepthException:
            curr_depth += 1


def sample_from_automata(automata, max_length, return_edges_visited=False):
    """Random sampling from an automata.

    Currently, this sampling is not necessarily uniform over all paths.
    https://math.stackexchange.com/questions/2673132/a-procedure-for-sampling-paths-in-a-directed-acyclic-graph
    There are some connections between this and sampling from a markov chain.

    :param automata: The automata to sample from
    :param max_length: The maximum length of paths to return
    :param return_edges_visited: Return edge paths in addition to nodes
    :return An iterator over paths (and optionally edges).
    """
    if not automata.num_states():
        return _empty_generator()
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    def random_depth_first_search(max_depth):
        # Here "nodes" are paths in the automata
        # Tuples of (state, edges_visited)
        Q = [((start_state,), tuple())]
        while Q:
            # NOTE(mkuchnik): Notice we choice() instead of popleft()
            curr_path, edges_visited = random.choice(Q)
            last_state = curr_path[-1]
            if len(curr_path) >= max_depth:
                continue
            for arc in automata.arcs(last_state):
                next_edge = arc.ilabel
                next_state = arc.nextstate
                new_path = curr_path + (next_state,)
                new_edges_visited = edges_visited + (next_edge,)
                if new_path[-1] in final_states:
                    if return_edges_visited:
                        yield new_path, new_edges_visited
                    else:
                        yield new_path
                Q.append((new_path, new_edges_visited))

    return random_depth_first_search(max_length)


class DijkstraCostAggregationType(enum.Enum):
    """Determine the aggregation function for costs."""

    SUM = enum.auto()
    MAX = enum.auto()


class PriorityQueue:
    """A priority queue wrapper around heapq.

    From:
    https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """

    def __init__(self, data=None):
        """Initialize the PriorityQueue."""
        if data is None:
            elements = []
        else:
            elements = list(data)
            heapq.heapify(elements)
        self.elements: typing.List[typing.Tuple[float, typing.T]] = elements

    def empty(self) -> bool:
        """Check if empty."""
        return not self.elements

    def put(self, priority_item: typing.Tuple[float, typing.T]):
        """Put an element into the priority queue."""
        priority, item = priority_item
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> typing.Tuple[float, typing.T]:
        """Pop an element off the priority queue."""
        return heapq.heappop(self.elements)

    def peak(self) -> typing.Tuple[float, typing.T]:
        """Peak the element at the head of the priority queue."""
        return self.elements[0]

    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return "{{PQ:{}}}".format(self.elements)

    def __repr__(self) -> str:
        """Return a string representation of the queue."""
        return "{{PQ:{}}}".format(self.elements)

    def __len__(self) -> int:
        """Return the length of the queue."""
        return len(self.elements)


def dijkstra_from_automata(automata, expand_neighbor_costs_fn,
                           return_edges_visited=False,
                           return_costs=False,
                           max_sequence_length=None,
                           automata_decode_function=None,
                           cost_aggregation_function=None,
                           prefetch_factor=None,
                           batch_size=None,
                           beam_size=None,
                           batch_expand_neighbor_costs_fn=None
                           ):
    """Dijkstra search from an automata.

    automata: A openfst automata
    expand_neighbor_costs_fn: A function that takes the current path and
    returns costs for all neighbors. The function must return a list-like
    datastructure of comparable items. For example, a numpy array of int or a
    list of comparable tuples.

    Relation to A*:
    Since A* has a function that is identical to the shortest path function
    except for the added heuristic:
    cost(u) = cost(parent(u)) + w(u) + h(u, goal)

    where h(u, goal) is the heuristic, we can simply add the heuristic to the
    cost function. However, since we don't have a specific goal, we have to
    find the minimum to all goals.
    """
    # TODO(mkuchnik): Add dataclass for heapq
    # https://ivergara.github.io/deeper-dataclasses.html
    # https://docs.python.org/3/library/heapq.html#basic-examples
    if not automata.num_states():
        return _empty_generator()
    if cost_aggregation_function is None:
        cost_aggregation_function = DijkstraCostAggregationType.SUM
    if cost_aggregation_function is DijkstraCostAggregationType.SUM:
        def binary_sum(x, y):
            return x + y
        cost_agg_fn = binary_sum
    elif cost_aggregation_function is DijkstraCostAggregationType.MAX:
        # Assume broadcast
        cost_agg_fn = torch.maximum
    else:
        raise ValueError("Unknown cost aggregation function: {}".format(
            cost_aggregation_function))
    if prefetch_factor is None:
        prefetch_factor = 0
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    if not callable(expand_neighbor_costs_fn):
        raise ValueError("Expected callable for expand_neighbor_costs_fn."
                         " Found {}".format(type(expand_neighbor_costs_fn)))
    sig = inspect.signature(expand_neighbor_costs_fn)
    if len(sig.parameters) != 1:
        raise ValueError("Expected callable for expand_neighbor_costs_fn"
                         " with 1 arg. Found {} args.".format(
                             len(sig.parameters)))

    logger.debug("Starting dijkstra")

    # Variant following:
    # https://www.redblobgames.com/pathfinding/a-star/introduction.html
    def dijkstra_search(track_all_costs=False):
        logger.debug("Enter dijkstra search")
        # Tuples of (priority, (state, edges_visited))
        Q = PriorityQueue()
        Q.put((0, ((start_state,), tuple())))

        # NOTE(mkuchnik): A visited state in this case is the whole path. This
        # is because loops e.g., wildcards will otherwise never expand the loop
        # past the initial loop. Additionally, only the shortest edge will be
        # used, which may result in missing solutions.

        if track_all_costs:
            costs_so_far = dict()
        else:
            automata_dod_cache = automata_to_connectivity_map(automata)
            logger.debug("automata cache: {}".format(automata_dod_cache))
            if automata_decode_function:

                def remap_dod_arcs(arcs):
                    """Map a dict of edge->state transitions."""
                    return {automata_decode_function(k): v
                            for k, v in arcs.items()}

                automata_dod_cache = {k: remap_dod_arcs(v)
                                      for k, v in automata_dod_cache.items()}
                logger.debug("remapped automata cache: {}".format(
                    automata_dod_cache))

        logger.debug("Enter dijkstra Q")

        # For both single and batch prefetching
        prefetch_buffers = {}

        if prefetch_factor:
            _prefetch_stats = {
                "hits": 0,
                "accesses": 0,
            }

            def prefetch_costs(edges_visited):
                if (len(prefetch_buffers) < prefetch_factor
                        and edges_visited not in prefetch_buffers):
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                    prefetch_buffers[edges_visited] = new_costs

            def batch_prefetch_costs_batch(batch_edges_visited):
                with record_function("batch_expand_neighbor_costs_fn"):
                    new_costs = batch_expand_neighbor_costs_fn(
                        batch_edges_visited)
                if callable(new_costs):
                    new_costs = new_costs()
                new_costs = new_costs.cpu()
                for i, edges_visited in enumerate(batch_edges_visited):
                    prefetch_buffers[edges_visited] = new_costs[i]

            def get_costs(edges_visited):
                _prefetch_stats["accesses"] += 1
                try:
                    new_costs = prefetch_buffers[edges_visited]
                    del prefetch_buffers[edges_visited]
                    _prefetch_stats["hits"] += 1
                except KeyError:
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                if callable(new_costs):
                    # Assume lambda
                    return new_costs()
                else:
                    return new_costs

            def prefetch_stats():
                return _prefetch_stats

        else:
            _prefetch_stats = {
                "hits": 0,
                "accesses": 0,
            }

            def prefetch_costs(edges_visited):
                pass

            def batch_prefetch_costs_batch(batch_edges_visited):
                with record_function("batch_expand_neighbor_costs_fn"):
                    new_costs = batch_expand_neighbor_costs_fn(
                        batch_edges_visited)
                if callable(new_costs):
                    new_costs = new_costs()
                # TODO(mkuchnik): We can lazily unpack by putting this logic in
                # a lamda
                new_costs = new_costs.cpu()
                for i, edges_visited in enumerate(batch_edges_visited):
                    prefetch_buffers[edges_visited] = new_costs[i]

            def get_costs(edges_visited):
                _prefetch_stats["accesses"] += 1
                try:
                    new_costs = prefetch_buffers[edges_visited]
                    del prefetch_buffers[edges_visited]
                    _prefetch_stats["hits"] += 1
                except KeyError:
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                if callable(new_costs):
                    # Assume lambda
                    return new_costs()
                else:
                    return new_costs

            def prefetch_stats():
                return _prefetch_stats

        def prefetch_next_element(Q):
            _, (_, prefetch_edges_visited) = Q.peak()
            prefetch_costs(prefetch_edges_visited)

        beam_Q = None
        if batch_size is None and beam_size is not None:
            raise ValueError("Beam_size must be set for batching")
        local_batch_size = batch_size
        while not Q.empty() or (beam_Q and not beam_Q.empty()):
            if beam_size is not None and (not beam_Q or beam_Q.empty()):
                beam_Q = []
                for i in range(beam_size):
                    if Q.empty():
                        break
                    beam_Q.append(Q.get())
                # TODO(mkuchnik): We have a sorted list so heap is redundant
                logger.debug("Filled beamQ with {} elements".format(
                    len(beam_Q)))
                beam_Q = PriorityQueue(beam_Q)
            if beam_size is not None and beam_Q:
                # Prefetch all beam_Q elements
                batch_edges_visited = []
                for beam_Q_list_idx in range(len(beam_Q.elements)):
                    # TODO(mkuchnik): Don't sample same element twice
                    _, (_, edges_visited) = \
                        beam_Q.elements[beam_Q_list_idx]
                    batch_edges_visited.append(edges_visited)
                    if len(batch_edges_visited) >= local_batch_size:
                        batch_prefetch_costs_batch(batch_edges_visited)
                        batch_edges_visited = []
                if len(batch_edges_visited):
                    batch_prefetch_costs_batch(batch_edges_visited)
                curr_cost, (curr_path, edges_visited) = beam_Q.get()
            else:
                curr_cost, (curr_path, edges_visited) = Q.get()
                prefetch_costs(edges_visited)
                if prefetch_factor >= 2 and not Q.empty():
                    if prefetch_factor == 2:
                        prefetch_next_element(Q)
                    elif prefetch_factor > 2:
                        raise NotImplementedError(
                            "Prefetch {} > 2 not implemented.".format(
                                prefetch_factor))

            logger.debug("Dijkstra pop: {}. Cost: {}".format(edges_visited,
                                                             curr_cost))
            curr_state = curr_path[-1]
            if curr_state in final_states:
                # ret a tuple of potentially <path, edges, cost>
                ret = [curr_path]
                if return_edges_visited:
                    ret.append(edges_visited)
                if return_costs:
                    ret.append(curr_cost)
                ret = tuple(ret)
                if len(ret) == 1:
                    # If only path, unpack
                    ret = ret[0]
                yield ret
            new_costs = get_costs(edges_visited)
            logger.debug("New costs: {}".format(new_costs.shape))
            # TODO(mkuchnik): Remove
            new_costs = torch.tensor(new_costs)
            if torch.any(new_costs < 0):
                raise ValueError("Costs can only be positive: {}".format(
                    new_costs))
            logger.debug("Prefetch stats: {}".format(prefetch_stats()))
            with record_function("dijkstra_cost_prep"):
                new_costs = cost_agg_fn(curr_cost, new_costs)
                are_infinite_costs = torch.isinf(new_costs).cpu().numpy()
                if torch.is_tensor(new_costs):
                    new_costs = new_costs.cpu().numpy()

            def dijkstra_Q_loop_slow():
                # TODO(mkuchnik): This linear cost loop is very costly.
                for arc in automata.arcs(curr_state):
                    next_edge = arc.ilabel
                    if automata_decode_function:
                        next_edge = automata_decode_function(next_edge)
                    next_state = arc.nextstate
                    new_path = curr_path + (next_state,)
                    new_edges_visited = edges_visited + (next_edge,)
                    new_cost = new_costs[next_edge]
                    new_cost_is_infinite = are_infinite_costs[next_edge]
                    if not track_all_costs:
                        if not new_cost_is_infinite:
                            Q.put((new_cost,
                                   (new_path, new_edges_visited)))
                    else:
                        try:
                            old_cost = costs_so_far[new_path]
                            # Found old state
                            if (new_cost < old_cost and not
                                    new_cost_is_infinite):
                                costs_so_far[new_path] = new_cost
                                Q.put((new_cost,
                                       (new_path, new_edges_visited)))
                        except KeyError:
                            if not new_cost_is_infinite:
                                # Didn't find old state
                                costs_so_far[new_path] = new_cost
                                Q.put((new_cost,
                                       (new_path, new_edges_visited)))

            def dijkstra_Q_loop_fast():
                # TODO(mkuchnik): This linear cost loop is very costly.
                assert not track_all_costs
                arcs = automata_dod_cache[curr_path[-1]]
                valid_edges = np.where(~are_infinite_costs)[0]
                for next_edge in valid_edges:
                    try:
                        next_state = arcs[next_edge]
                    except KeyError:
                        # There is no edge
                        continue
                    new_path = curr_path + (next_state,)
                    new_edges_visited = edges_visited + (next_edge,)
                    new_cost = new_costs[next_edge]
                    Q.put((new_cost,
                           (new_path, new_edges_visited)))

            if not track_all_costs:
                dijkstra_Q_loop = dijkstra_Q_loop_fast
            else:
                dijkstra_Q_loop = dijkstra_Q_loop_slow

            with record_function("dijkstra_Q_loop"):
                if ((max_sequence_length is None) or
                        (len(edges_visited) < max_sequence_length)):
                    dijkstra_Q_loop()
                else:
                    logger.debug("Edges {} longer than max length {}".format(
                        edges_visited, max_sequence_length))
                logger.debug("Queue length: {}".format(len(Q)))
                if beam_Q:
                    logger.debug("Beam queue length: {}".format(len(beam_Q)))

    return dijkstra_search()


def random_sampling_from_automata(
    automata, expand_neighbor_costs_fn,
    return_edges_visited=False,
    return_costs=False,
    max_sequence_length=None,
    replacement=True,
    automata_decode_function=None,
):
    """Random sampling search from an automata.

    automata: A openfst automata
    expand_neighbor_costs_fn: A function that takes the current path and
    returns costs for all neighbors. The function must return a list-like
    datastructure of comparable items. For example, a numpy array of int or a
    list of comparable tuples.
    """
    # TODO(mkuchnik): Add dataclass for heapq
    # https://ivergara.github.io/deeper-dataclasses.html
    # https://docs.python.org/3/library/heapq.html#basic-examples
    if not automata.num_states():
        return _empty_generator()
    prefetch_factor = None
    if prefetch_factor is None:
        prefetch_factor = 0
    beam_size = None
    batch_expand_neighbor_costs_fn = None
    batch_size = 1
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    if not callable(expand_neighbor_costs_fn):
        raise ValueError("Expected callable for expand_neighbor_costs_fn."
                         " Found {}".format(type(expand_neighbor_costs_fn)))
    sig = inspect.signature(expand_neighbor_costs_fn)
    if len(sig.parameters) != 1:
        raise ValueError("Expected callable for expand_neighbor_costs_fn"
                         " with 1 arg. Found {} args.".format(
                             len(sig.parameters)))

    logger.debug("Starting random sampling")

    # Variant following:
    # https://www.redblobgames.com/pathfinding/a-star/introduction.html
    def random_search(track_all_costs=False):
        logger.debug("Enter random search")
        # Tuples of (priority, (state, edges_visited))
        Q = PriorityQueue()
        Q.put((1., ((start_state,), tuple(), 1.)))

        # NOTE(mkuchnik): A visited state in this case is the whole path. This
        # is because loops e.g., wildcards will otherwise never expand the loop
        # past the initial loop. Additionally, only the shortest edge will be
        # used, which may result in missing solutions.

        if track_all_costs:
            raise NotImplementedError("Tracking not implemented")
        else:
            automata_dod_cache = automata_to_connectivity_map(automata)
            logger.debug("automata cache: {}".format(automata_dod_cache))
            if automata_decode_function:

                def remap_dod_arcs(arcs):
                    """Map a dict of edge->state transitions."""
                    return {automata_decode_function(k): v
                            for k, v in arcs.items()}

                automata_dod_cache = {k: remap_dod_arcs(v)
                                      for k, v in automata_dod_cache.items()}
                logger.debug("remapped automata cache: {}".format(
                    automata_dod_cache))

        logger.debug("Enter random Q")

        # For both single and batch prefetching
        prefetch_buffers = {}

        if prefetch_factor:
            _prefetch_stats = {
                "hits": 0,
                "accesses": 0,
            }

            def prefetch_costs(edges_visited):
                if (len(prefetch_buffers) < prefetch_factor
                        and edges_visited not in prefetch_buffers):
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                    prefetch_buffers[edges_visited] = new_costs

            def batch_prefetch_costs_batch(batch_edges_visited):
                with record_function("batch_expand_neighbor_costs_fn"):
                    new_costs = batch_expand_neighbor_costs_fn(
                        batch_edges_visited)
                if callable(new_costs):
                    new_costs = new_costs()
                new_costs = new_costs.cpu()
                for i, edges_visited in enumerate(batch_edges_visited):
                    prefetch_buffers[edges_visited] = new_costs[i]

            def get_costs(edges_visited):
                _prefetch_stats["accesses"] += 1
                try:
                    new_costs = prefetch_buffers[edges_visited]
                    del prefetch_buffers[edges_visited]
                    _prefetch_stats["hits"] += 1
                except KeyError:
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                if callable(new_costs):
                    # Assume lambda
                    return new_costs()
                else:
                    return new_costs

            def prefetch_stats():
                return _prefetch_stats

        else:
            _prefetch_stats = {
                "hits": 0,
                "accesses": 0,
            }

            def prefetch_costs(edges_visited):
                pass

            def batch_prefetch_costs_batch(batch_edges_visited):
                with record_function("batch_expand_neighbor_costs_fn"):
                    new_costs = batch_expand_neighbor_costs_fn(
                        batch_edges_visited)
                if callable(new_costs):
                    new_costs = new_costs()
                # TODO(mkuchnik): We can lazily unpack by putting this logic in
                # a lamda
                new_costs = new_costs.cpu()
                for i, edges_visited in enumerate(batch_edges_visited):
                    prefetch_buffers[edges_visited] = new_costs[i]

            def get_costs(edges_visited):
                _prefetch_stats["accesses"] += 1
                try:
                    new_costs = prefetch_buffers[edges_visited]
                    del prefetch_buffers[edges_visited]
                    _prefetch_stats["hits"] += 1
                except KeyError:
                    with record_function("expand_neighbor_costs_fn"):
                        new_costs = expand_neighbor_costs_fn(edges_visited)
                if callable(new_costs):
                    # Assume lambda
                    return new_costs()
                else:
                    return new_costs

            def prefetch_stats():
                return _prefetch_stats

        def prefetch_next_element(Q):
            _, (_, prefetch_edges_visited) = Q.peak()
            prefetch_costs(prefetch_edges_visited)

        # A-Res algorithm
        # https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res

        beam_Q = None
        local_batch_size = batch_size if batch_size is not None else 16
        while not Q.empty() or (beam_Q and not beam_Q.empty()):
            if beam_size is not None and (not beam_Q or beam_Q.empty()):
                beam_Q = []
                for i in range(beam_size):
                    if Q.empty():
                        break
                    beam_Q.append(Q.get())
                # TODO(mkuchnik): We have a sorted list so heap is redundant
                logger.debug("Filled beamQ with {} elements".format(
                    len(beam_Q)))
                beam_Q = PriorityQueue(beam_Q)
            if beam_size is not None and beam_Q:
                # Prefetch all beam_Q elements
                batch_edges_visited = []
                for beam_Q_list_idx in range(len(beam_Q.elements)):
                    # TODO(mkuchnik): Don't sample same element twice
                    _, (_, edges_visited) = \
                        beam_Q.elements[beam_Q_list_idx]
                    batch_edges_visited.append(edges_visited)
                    if len(batch_edges_visited) >= local_batch_size:
                        batch_prefetch_costs_batch(batch_edges_visited)
                        batch_edges_visited = []
                if len(batch_edges_visited):
                    batch_prefetch_costs_batch(batch_edges_visited)
                curr_cost, (curr_path, edges_visited, curr_prob) = beam_Q.get()
            else:
                curr_cost, (curr_path, edges_visited, curr_prob) = Q.get()
                prefetch_costs(edges_visited)
                if prefetch_factor >= 2 and not Q.empty():
                    if prefetch_factor == 2:
                        prefetch_next_element(Q)
                    elif prefetch_factor > 2:
                        raise NotImplementedError(
                            "Prefetch {} > 2 not implemented.".format(
                                prefetch_factor))

            logger.debug("Random pop: {}. Cost: {}".format(edges_visited,
                                                           curr_cost))
            curr_state = curr_path[-1]
            if curr_state in final_states:
                # ret a tuple of potentially <path, edges, cost>
                ret = [curr_path]
                if return_edges_visited:
                    ret.append(edges_visited)
                if return_costs:
                    ret.append(curr_cost)
                ret = tuple(ret)
                if len(ret) == 1:
                    # If only path, unpack
                    ret = ret[0]
                yield ret
            new_costs = get_costs(edges_visited)
            logger.debug("New costs: {}".format(new_costs.shape))
            # TODO(mkuchnik): Remove
            new_costs = torch.tensor(new_costs)
            if torch.any(new_costs < 0):
                raise ValueError("Costs can only be positive: {}".format(
                    new_costs))
            logger.debug("Prefetch stats: {}".format(prefetch_stats()))
            with record_function("random_cost_prep"):
                # NOTE(mkuchnik): Use probability (without prior weight)
                new_probabilities = new_costs
                if not replacement:
                    new_costs = torch.log(new_costs) + curr_cost
                    new_probabilities = new_costs
                    random_costs = torch.rand(new_costs.shape)
                    new_costs = -torch.log(random_costs) * new_costs
                are_infinite_costs = (new_costs <= 0.).cpu().numpy()
                if torch.is_tensor(new_costs):
                    new_costs = new_costs.cpu().numpy()
                if replacement:
                    new_probabilities = new_costs
                else:
                    new_probabilities = new_probabilities.cpu().numpy()

            def random_Q_loop_fast():
                # TODO(mkuchnik): This linear cost loop is very costly.
                # TODO(mkuchnik): Refactor using torch generators/multinomial
                assert not track_all_costs
                arcs = automata_dod_cache[curr_path[-1]]
                valid_edges = np.where(~are_infinite_costs)[0]
                if replacement:
                    if len(valid_edges):
                        # Remove edges that aren't in transition
                        p = np.zeros(new_costs.shape)
                        for next_edge in valid_edges:
                            try:
                                next_state = arcs[next_edge]
                            except KeyError:
                                continue
                            p[next_edge] = new_costs[next_edge]
                        p_CDF = np.cumsum(p, dtype=np.double)
                        # If nothing is viable, CDF will be zeros
                        if p_CDF[-1] > 0:
                            rand = np.random.uniform(low=0.,
                                                     high=p_CDF[-1],
                                                     size=1)
                            next_edge = np.argmax((p_CDF > rand) &
                                                  ~are_infinite_costs)
                            logger.debug(
                                "Sampling {} with {} probability.".format(
                                    next_edge, p[next_edge]
                                )
                            )
                            next_state = arcs[next_edge]
                            new_path = curr_path + (next_state,)
                            new_edges_visited = edges_visited + (next_edge,)
                            new_cost = new_costs[next_edge]
                            new_prob = new_probabilities[next_edge]
                            Q.put((new_cost,
                                   (new_path, new_edges_visited, new_prob))
                                  )
                else:
                    for next_edge in valid_edges:
                        try:
                            next_state = arcs[next_edge]
                        except KeyError:
                            # There is no edge
                            continue
                        new_path = curr_path + (next_state,)
                        new_edges_visited = edges_visited + (next_edge,)
                        new_cost = new_costs[next_edge]
                        new_prob = new_probabilities[next_edge]
                        Q.put((new_cost,
                               (new_path, new_edges_visited, new_prob))
                              )

            if not track_all_costs:
                random_Q_loop = random_Q_loop_fast
            else:
                raise NotImplementedError(
                    "tracking not implemented for random sampling.")

            with record_function("random_Q_loop"):
                if ((max_sequence_length is None) or
                        (len(edges_visited) < max_sequence_length)):
                    random_Q_loop()
                else:
                    logger.debug("Edges {} longer than max length {}".format(
                        edges_visited, max_sequence_length))
                logger.debug("Queue length: {}".format(len(Q)))
                if beam_Q:
                    logger.debug("Beam queue length: {}".format(len(beam_Q)))
            if replacement and Q.empty():
                # Add initial state
                Q.put((1., ((start_state,), tuple(), 1.)))

    return random_search()


def greedy_search_from_automata(automata, expand_neighbor_costs_fn,
                                return_edges_visited=False,
                                return_costs=False,
                                max_sequence_length=None,
                                automata_decode_function=None):
    """Greedy BFS search from an automata.

    automata: A openfst automata
    expand_neighbor_costs_fn: A function that takes the current path and
    returns costs for all neighbors. The function must return a list-like
    datastructure of comparable items. For example, a numpy array of int or a
    list of comparable tuples.
    """
    # TODO(mkuchnik): Add dataclass for heapq
    # https://ivergara.github.io/deeper-dataclasses.html
    # https://docs.python.org/3/library/heapq.html#basic-examples
    if not automata.num_states():
        return _empty_generator()
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    if not callable(expand_neighbor_costs_fn):
        raise ValueError("Expected callable for expand_neighbor_costs_fn."
                         " Found {}".format(type(expand_neighbor_costs_fn)))
    sig = inspect.signature(expand_neighbor_costs_fn)
    if len(sig.parameters) != 1:
        raise ValueError("Expected callable for expand_neighbor_costs_fn"
                         " with 1 arg. Found {} args.".format(
                             len(sig.parameters)))

    logger.debug("Starting greedy search.")

    def greedy_search(track_all_costs=False):
        logger.debug("Enter greedy search")
        # Tuples of (priority, (state, edges_visited))
        Q = PriorityQueue()
        Q.put((0, ((start_state,), tuple())))

        # NOTE(mkuchnik): A visited state in this case is the whole path. This
        # is because loops e.g., wildcards will otherwise never expand the loop
        # past the initial loop. Additionally, only the shortest edge will be
        # used, which may result in missing solutions.

        if track_all_costs:
            costs_so_far = dict()

        logger.debug("Enter greedy Q")
        while not Q.empty():
            curr_cost, (curr_path, edges_visited) = Q.get()
            logger.debug("Greedy pop: {}. Cost: {}".format(edges_visited,
                                                           curr_cost))
            if math.isinf(curr_cost):
                # Don't bother expanding if already impossible
                continue
            curr_state = curr_path[-1]
            if curr_state in final_states:
                # ret a tuple of potentially <path, edges, cost>
                ret = [curr_path]
                if return_edges_visited:
                    ret.append(edges_visited)
                if return_costs:
                    ret.append(curr_cost)
                ret = tuple(ret)
                if len(ret) == 1:
                    # If only path, unpack
                    ret = ret[0]
                yield ret
            new_costs = expand_neighbor_costs_fn(edges_visited)
            if any(new_costs < 0):
                raise ValueError("Costs can only be positive: {}".format(
                    new_costs))
            if ((max_sequence_length is None) or
                    (len(edges_visited) < max_sequence_length)):
                for arc in automata.arcs(curr_state):
                    next_edge = arc.ilabel
                    if automata_decode_function:
                        next_edge = automata_decode_function(next_edge)
                    next_state = arc.nextstate
                    new_path = curr_path + (next_state,)
                    new_edges_visited = edges_visited + (next_edge,)
                    new_cost = new_costs[next_edge]  # Greedy
                    if track_all_costs:
                        try:
                            old_cost = costs_so_far[new_path]
                            # Found old state
                            if (new_cost < old_cost and not
                                    math.isinf(new_cost)):
                                costs_so_far[new_path] = new_cost
                        except KeyError:
                            if not math.isinf(new_cost):
                                # Didn't find old state
                                costs_so_far[new_path] = new_cost
                    if not math.isinf(new_cost):
                        Q.put((new_cost,
                               (new_path, new_edges_visited)))

    return greedy_search()


def union_fst(all_possible_elements):
    """Build an automata that accepts all elements once."""
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    all_possible_elements = set(all_possible_elements)

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)

    # Transition states
    end_state = f.add_state()
    f.reserve_arcs(start_state, len(all_possible_elements))  # Optional.
    # to final
    for e in all_possible_elements:
        f.add_arc(start_state, fst.Arc(e, e, one, end_state))

    # By default, state gets infinite weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def wildcard_fst(all_possible_elements):
    """Build an automata that accepts all elements specified forever.

    Note: it does not accept no elements.
    """
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    all_possible_elements = set(all_possible_elements)

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)

    # Transition states
    end_state = f.add_state()
    f.reserve_arcs(start_state, len(all_possible_elements))  # Optional.
    # to final
    for e in all_possible_elements:
        f.add_arc(start_state, fst.Arc(e, e, one, end_state))
    f.reserve_arcs(end_state, len(all_possible_elements))  # Optional.
    # final to itself
    for e in all_possible_elements:
        f.add_arc(end_state, fst.Arc(e, e, one, end_state))

    # By default, state gets infinite weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def any_fst(all_possible_elements):
    """Build an automata that accepts all elements once."""
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    all_possible_elements = set(all_possible_elements)

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)

    # Transition states
    end_state = f.add_state()
    f.reserve_arcs(start_state, len(all_possible_elements))  # Optional.
    # to final
    for e in all_possible_elements:
        f.add_arc(start_state, fst.Arc(e, e, one, end_state))

    # By default, state gets infinite weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def automata_next_states(automata, transition_list,
                         return_edges_visited=False,
                         return_arcs=False,
                         filter_nonfinal_transition_states=False,
                         filter_nonfinal_terminal_states=False):
    """Return states that are reachable given the transition_list."""
    if not automata.num_states():
        return _empty_generator()
    # A path is valid if it ends with a final state
    final_states = automata_final_states(automata)
    start_state = automata_start_state(automata)

    def depth_first_search():
        # Here "nodes" are paths in the automata
        visited = set()
        # Tuples of (state, edges_visited)
        Q = collections.deque([((start_state,), tuple())])
        i = 0
        while Q:
            # NOTE(mkuchnik): Notice we pop() instead of popleft()
            curr_path, edges_visited = Q.pop()
            last_state = curr_path[-1]
            if i >= len(transition_list):
                break
            num_edges_found = 0
            for arc in automata.arcs(last_state):
                next_edge = arc.ilabel
                next_state = arc.nextstate
                if (next_edge == transition_list[i] and
                        (not filter_nonfinal_transition_states
                         or next_state in final_states)):
                    num_edges_found += 1
                    new_path = curr_path + (next_state,)
                    new_edges_visited = edges_visited + (next_edge,)
                    if new_path not in visited:
                        visited.add(new_path)
                        Q.append((new_path, new_edges_visited))
                    i += 1
                    break
            if num_edges_found != 1:
                # Fell off path
                return _empty_generator()
        if len(edges_visited) == len(transition_list):
            for arc in automata.arcs(last_state):
                next_state = arc.nextstate
                if (not filter_nonfinal_terminal_states
                        or next_state in final_states):
                    if return_edges_visited:
                        if return_arcs:
                            yield next_state, arc
                        else:
                            next_edge = arc.ilabel
                            yield next_state, next_edge
                    else:
                        yield next_state
        else:
            raise RuntimeError(
                "Failed to trace path of {}. visited: {}.".format(
                    transition_list, edges_visited))
    return depth_first_search()


def automata_start_state(automata, raise_exception=True):
    """Return the start states in the automata."""
    start_state = automata.start()
    if start_state < 0 and raise_exception:
        raise RuntimeError("Automata contains no start state.")
    return start_state


def automata_final_states(automata):
    """Return a list of final states in the automata."""
    # https://github.com/kylebgorman/pynini/issues/45
    zero = fst.Weight.zero(automata.weight_type())
    return [s for s in automata.states() if automata.final(s) != zero]


def convert_automata_to_prefix_acceptor(automata, return_copy=True):
    """Return a automata with all prefixes to acceptor accepted.

    Any state that is on the path to the acceptor is an acceptor.
    """
    if return_copy:
        f = automata.copy()
    else:
        f = automata
    one = fst.Weight.one(f.weight_type())
    for state in f.states():
        f.set_final(state, one)
    return f


def convert_automata_to_prefix_no_acceptor(automata, return_copy=True):
    """Return a automata with all prefixes to acceptor NOT accepted."""
    if return_copy:
        f = automata.copy()
    else:
        f = automata
    zero = fst.Weight.zero(f.weight_type())
    for state in f.states():
        f.set_final(state, zero)
    return f


def convert_automata_to_sink_acceptor(automata, return_copy=True):
    """Return a automata with all sinks accepted."""
    if is_cyclic_automata(automata):
        raise ValueError("Cycles not supported.")
    if return_copy:
        f = automata.copy()
    else:
        f = automata

    one = fst.Weight.one(f.weight_type())
    for state in f.states():
        has_arc = False
        for arc in f.arcs(state):
            has_arc = True
            break
        if not has_arc:
            f.set_final(state, one)
    return f


def apply_fst(elements, automata_op, is_project=False):
    """Compose a linear automata generated from `elements` with `automata_op`.

    From
    https://stackoverflow.com/questions/9390536/how-do-you-even-give-an-openfst-made-fst-input-where-does-the-output-go

    Args:
        elements (list): ordered list of edge symbols for a linear automata.
        automata_op (Fst): automata that will be applied.
        is_project (str, optional): whether to keep only the "input" or
        "output" labels.
        kwargs: Additional arguments to the compiler of the linear automata .
    """
    linear_automata = automata_from_token_list(elements)
    symbol_table = automata_op.input_symbols()
    if symbol_table is not None:
        # NOTE(mkuchnik): Avoid expensive copies here by re-using the table.
        attach_symbol_table(linear_automata, symbol_table)
    out = fst.compose(linear_automata, automata_op)
    if is_project:
        out.project("output")
    return out


def accepted(output_apply):
    """Return True if output of `apply_fst` for acceptor accepts the string.

    From:
    https://stackoverflow.com/questions/45213112/why-openfst-does-not-seem-to-have-run-or-accept-or-transduce-command
    """
    return output_apply.num_states() != 0


def apply_fst_accepted(elements, automata_op, is_project=False) -> bool:
    """Return True if elements is accepted by automata."""
    ret = apply_fst(elements, automata_op, is_project)
    ret = accepted(ret)
    return ret


def used_keys_set(automata) -> typing.Set[int]:
    """
    Retrieve the set of symbols (keys) used in the automata.

    This is useful if the symbol table is bigger than necessary.
    """
    used_keys = set()
    for s in automata.states():
        for a in automata.arcs(s):
            used_keys.add(a.ilabel)
            used_keys.add(a.olabel)
    return used_keys


def truncate_automata(automata, max_depth):
    """Truncate an automata to a length of max_depth."""
    symbol_table = dict(automata.input_symbols())
    all_possible_elements = used_keys_set(automata)
    # NOTE(mkuchnik): The memory complexity is dependent on the size of the
    # joining FST. As an optimization, we only consider keys that are used,
    # because the intersection will only be able to match on those. This
    # sparsity reduces memory complexity substantially.
    all_element_fst = any_fst(all_possible_elements)
    all_element_fst_depth = repeat_automata(
        all_element_fst,
        0,
        max_depth,
        determinize=False,
        minimize=False,
    )
    attach_symbol_table(all_element_fst_depth, symbol_table)
    all_element_fst_depth = convert_automata_to_prefix_acceptor(
        all_element_fst_depth, return_copy=False)
    automata.arcsort()
    all_element_fst_depth.arcsort()
    truncated_automata = fst.intersect(automata, all_element_fst_depth)
    truncated_automata = finalize_automata(truncated_automata)
    attach_symbol_table(truncated_automata, symbol_table)
    return truncated_automata


def max_path_distance(automata):
    """
    Find the maximum path distance e.g., opposite of shortest path.

    This is NP-Hard, but we can solve DAGs in linear time.

    We use the number of hops (edges) as the distance.
    """
    if is_cyclic_automata(automata):
        raise RuntimeError("Cyclic automata are not supported.")
    start_state = automata_start_state(automata, raise_exception=False)
    if start_state < 0:
        logger.warning("Empty automata encountered.")
        return 0

    distances = {s: -np.inf for s in automata.states()}
    distances[start_state] = 0
    for vertex in topological_sort(automata,
                                   max_depth=None,
                                   return_depth=False):
        for arc in automata.arcs(vertex):
            next_vertex = arc.nextstate
            next_distance = distances[vertex] + 1
            # Relaxation
            if distances[next_vertex] < next_distance:
                distances[next_vertex] = next_distance
    return max(distances.values())


def normalize_automata(automata, max_length: typing.Optional[int] = None):
    """Return an automata with all paths weighted to have uniform sampling.

    This algorithm operates in-place to adjust weights.

    Time complexity: same as DFS O(V+E)
    Space complexity: O(V+E)
    """
    return _normalize_automata_combinatorial(
        automata, max_length, dtype=np.longdouble)


def _normalize_automata_combinatorial(
        automata, max_length: typing.Optional[int] = None, dtype=None):
    """Return an automata with all paths weighted to have uniform sampling.

    This algorithm operates in-place to adjust weights.
    """
    if is_cyclic_automata(automata):
        # TODO(mkuchnik): Add max-depth support
        raise RuntimeError("Cyclic automata can't be normalized.")

    if dtype is None:
        dtype = np.longlong

    if max_length is None:
        max_length = max_path_distance(automata)

    start_state = automata_start_state(automata, raise_exception=False)
    if start_state < 0:
        logger.warning("Empty automata encountered.")
        return
    final_states = automata_final_states(automata)

    entries = []
    i = None
    for i, state in enumerate(automata.states()):
        arcs = list(automata.arcs(state))
        for arc in arcs:
            entries.append((state, arc.nextstate, 1))
    if i is None:
        # Automata is empty
        return
    max_states = i + 1

    # Build transition matrix
    row, col, data = zip(*entries)
    M = coo_matrix((data, (row, col)), shape=(max_states, max_states),
                   dtype=dtype)
    M = M.tocsr()

    # Build final vector
    entries = [(ff, 0, 1) for ff in final_states]
    row, col, data = zip(*entries)
    f = coo_matrix((data, (row, col)), (max_states, 1),
                   dtype=dtype)
    f = f.tocsr()

    # Precompute transition matrices for length up to max_length
    M_to_Ns = []
    # First entry is raised to 0 i.e., identity matrix
    M_to_Ns.append(M**0)
    M_to_N = 1
    for length in range(1, max_length+1):
        M_to_N = M_to_N * M  # M^length
        M_to_Ns.append(M_to_N)

    def string_count_from_vertex(start_vertex, this_max_length):
        """Run the count algorithm from start_vertex up to max_length."""
        # Build start vector
        entries = [(start_vertex, 0, 1)]
        row, col, data = zip(*entries)
        s = coo_matrix((data, (row, col)), shape=(max_states, 1),
                       dtype=dtype)
        s = s.tocsr()
        num_strings = 0
        for i, M_to_N in enumerate(M_to_Ns):
            if i > this_max_length:
                # Get first length elements
                break
            string_count = int(s.T.dot(M_to_N).dot(f).toarray()[0][0])
            if string_count < 0:
                raise OverflowError("num_strings is negative ({})."
                                    " Overflow at i={}?".format(
                                        string_count, i))
            num_strings += string_count
        return num_strings

    # NOTE(mkuchnik): We don't use max_depth as we are in a DAG
    vertex_count_map = dict()
    for vertex, depth in topological_sort(automata, max_depth=None,
                                          return_depth=True):
        string_count = string_count_from_vertex(vertex, max_length - depth)
        vertex_count_map[vertex] = string_count

    logger.debug("Vertex string count map: {}".format(
        pprint.pformat(vertex_count_map)))

    # Make a copy
    new_automata = fst.VectorFst()
    new_automata.reserve_states(sum(1 for _ in automata.states()))
    for s in automata.states():
        ss = new_automata.add_state()
        # Set final weight
        final_weight = automata.final(s)
        new_automata.set_final(ss, final_weight)

    # Set start
    start_state = automata_start_state(automata, raise_exception=True)
    new_automata.set_start(start_state)

    for vertex, depth in topological_sort(automata, max_depth=None,
                                          return_depth=True):
        curr_sum = vertex_count_map[vertex]
        self_sum = string_count_from_vertex(vertex, 0)
        leaving_sum = curr_sum - self_sum
        seen_sum = 0
        seen_ratios_sum = 0
        any_arcs_seen = False
        for arc in automata.arcs(vertex):
            any_arcs_seen = True
            arc_sum = vertex_count_map[arc.nextstate]
            if curr_sum == 0 and arc_sum == 0:
                # Division by 0
                ratio = 1.0
            else:
                # We want to weight by the ratio of success to parent
                ratio = float(arc_sum) / float(leaving_sum)
            seen_sum += arc_sum
            assert ratio >= 0.0, "Ratio = {}".format(ratio)
            assert ratio <= 1.0, "Ratio = {}".format(ratio)
            logger.debug("Weight for state->arc {}->{} is {}".format(
                vertex, arc.ilabel, ratio))
            ti = arc.ilabel
            to = arc.olabel
            weight = ratio
            seen_ratios_sum += ratio
            s = arc.nextstate
            # Copy new arc over
            new_automata.add_arc(vertex, fst.Arc(ti, to, weight, s))
        if curr_sum < seen_sum:
            # We expect the current sum to be an upper bound, because it will
            # include all arcs as well as direct edges (which are discounted
            # due to max length).
            # For example, "the" will start with 4 accept strings, but one of
            # the 4 will be the token representing "the" and thus the remaining
            # arcs only sum to 3.
            logger.warning("String count for {} is {}, but arcs sum to {}"
                           .format(vertex, curr_sum, seen_sum))
        if (any_arcs_seen and
                (seen_ratios_sum < 0.95 or seen_ratios_sum > 1.05)):
            logger.warning("Expected ratios to sum to 1.0, but got {} for {}."
                           .format(seen_ratios_sum, vertex))

    attach_symbol_table(new_automata,
                        dict(automata.input_symbols()),
                        attach_inputs=True,
                        attach_outputs=False)
    attach_symbol_table(new_automata,
                        dict(automata.output_symbols()),
                        attach_inputs=False,
                        attach_outputs=True)

    return new_automata


def _normalize_automata_DFS(automata, max_length: typing.Optional[int] = None):
    """Return an automata with all paths weighted to have uniform sampling.

    This algorithm operates in-place to adjust weights.

    Time complexity: same as DFS O(V+E)
    Space complexity: O(V+E)
    """
    if is_cyclic_automata(automata):
        # TODO(mkuchnik): Add max-depth support
        raise RuntimeError("Cyclic automata can't be normalized.")

    # An edge is uniquely identified by a tuple of (state, edge_id)
    edge_counts = {}
    # 1) Do DFS to get all paths (linear time)
    dfs_iter = DFS_from_automata(
            automata, return_edges_visited=True, max_length=max_length)
    # TODO(mkuchnik): Remove once we have faster implementation
    for path_states, path_edges in dfs_iter:
        # 2) Accumulate for all edges and increment when edge used by 1
        for s, e in zip(path_states, path_edges):
            try:
                edge_counts[(s, e)] += 1
            except KeyError:
                edge_counts[(s, e)] = 1

    # Make a copy
    f = fst.VectorFst()
    f.reserve_states(sum(1 for _ in automata.states()))
    for s in automata.states():
        ss = f.add_state()
        # Set final weight
        final_weight = automata.final(s)
        f.set_final(ss, final_weight)

    # Set start
    start_state = automata_start_state(automata, raise_exception=True)
    f.set_start(start_state)

    # 3) Normalize
    for s in automata.states():
        state_total_count = 0
        # Compute total
        for arc in automata.arcs(s):
            e = arc.ilabel
            try:
                state_edge_count = edge_counts[(s, e)]
            except KeyError:
                state_edge_count = 0
            state_total_count += state_edge_count
        # Set arcs
        for arc in automata.arcs(s):
            e = arc.ilabel
            try:
                state_edge_count = edge_counts[(s, e)]
            except KeyError:
                state_edge_count = 0
            # Apply to weight
            ratio = (float(state_edge_count) /
                     float(state_total_count))
            vertex = s
            ti = arc.ilabel
            to = arc.olabel
            weight = ratio
            sn = arc.nextstate
            # Copy new arc over
            f.add_arc(vertex, fst.Arc(ti, to, weight, sn))

    attach_symbol_table(f,
                        dict(automata.input_symbols()),
                        attach_inputs=True,
                        attach_outputs=False)
    attach_symbol_table(f,
                        dict(automata.output_symbols()),
                        attach_inputs=False,
                        attach_outputs=True)
    return f


def summarize_automata(automata, expand_str=False) -> str:
    """Print an automata into a string."""
    if expand_str:
        if automata.input_symbols():
            input_symbols = dict(automata.input_symbols())
        else:
            input_symbols = None
        if automata.output_symbols():
            output_symbols = dict(automata.output_symbols())
        else:
            output_symbols = None

    def arc_to_str(arc):
        if expand_str:
            if input_symbols:
                istr = input_symbols[arc.ilabel]
            else:
                istr = None
            if output_symbols:
                ostr = output_symbols[arc.olabel]
            else:
                ostr = None
            return "{} {} ({}) {} ({}) {}".format(
                arc.nextstate, arc.ilabel, istr, arc.olabel, ostr, arc.weight)
        else:
            return "{} {} {} {}".format(arc.nextstate, arc.ilabel, arc.olabel,
                                        arc.weight)

    def visit(state, arc, curr_buffer):
        curr_buffer.append("{} -> {}".format(state, arc_to_str(arc)))

    curr_buffer = []
    for s in automata.states():
        for a in automata.arcs(s):
            visit(s, a, curr_buffer)

    return "\n".join(curr_buffer)


def find_suffix_from_prefix(prefix_automata, full_automata):
    """Find a suffix automata such that suffix + prefix = full_automata."""
    # First we need to find states that are only in suffix. We do this with the
    # difference operation over all acceptings states. Any accepting state is
    # in the suffix.
    prefix_automata_acceptor = convert_automata_to_prefix_acceptor(
        prefix_automata)
    prefix_automata_acceptor = prefix_automata_acceptor.project("output")
    full_automata_acceptor = convert_automata_to_prefix_acceptor(
        full_automata)
    prefix_automata_acceptor.arcsort()
    full_automata_acceptor.arcsort()
    if prefix_automata is full_automata:
        # NOTE(mkuchnik): Difference can fail if prefix == full
        raise ValueError("Both automata must be unique")
    difference_automata = fst.difference(full_automata_acceptor,
                                         prefix_automata_acceptor)
    suffix_states = automata_final_states(difference_automata)
    suffix_states = set(suffix_states)

    # Make a new FST, copying the acceptor states over
    # First, determine a map between states
    f = fst.VectorFst()
    state_map = dict()
    for old_state in suffix_states:
        new_state = f.add_state()
        state_map[old_state] = new_state

    # Now, copy the arcs over
    for old_state in suffix_states:
        new_state = state_map[old_state]
        for arc in difference_automata.arcs(old_state):
            ti = arc.ilabel
            to = arc.olabel
            w = arc.weight
            nextstate = state_map[arc.nextstate]
            new_arc = fst.Arc(ti, to, w, nextstate)
            f.add_arc(new_state, new_arc)

    # Now, create the start state
    start_state = f.add_state()
    f.set_start(start_state)

    # Now we partition the nodes into two connected components, and use the
    # edges connecting them as start edges
    for state in difference_automata.states():
        for arc in difference_automata.arcs(state):
            if arc.nextstate in suffix_states:
                ti = arc.ilabel
                to = arc.olabel
                w = arc.weight
                nextstate = state_map[arc.nextstate]
                new_arc = fst.Arc(ti, to, w, nextstate)
                f.add_arc(start_state, new_arc)
    input_symbols = difference_automata.input_symbols()
    if input_symbols:
        attach_symbol_table(f,
                            input_symbols,
                            attach_inputs=True,
                            attach_outputs=False)
    output_symbols = difference_automata.output_symbols()
    if output_symbols:
        attach_symbol_table(f,
                            output_symbols,
                            attach_inputs=False,
                            attach_outputs=True)

    old_final_states = automata_final_states(full_automata)
    one = fst.Weight.one(f.weight_type())
    # Finally, copy the end states
    for old_state in old_final_states:
        try:
            new_state = state_map[old_state]
        except KeyError:
            continue
        f.set_final(new_state, one)

    f = finalize_automata(f)

    return f
