"""Tests for relm automata."""

import unittest

import numpy as np
import pywrapfst as fst

import relm.automata
import relm.regex_token_preprocessor


class TestAutomata(unittest.TestCase):
    """Test that automata accepts desired strings."""

    def test_acceptance(self):
        """Test that token lists are accepted."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([3, 2, 1], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([3], dfa))

        dfa = relm.automata.automata_from_token_list([3])
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([3, 2, 1], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([], dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([3], dfa))

        # We also accept tuples
        dfa = relm.automata.automata_from_token_list((1, 2, 3))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([3, 2, 1], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([], dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([3], dfa))

    def test_symbol_table(self):
        """Test that symbols tables can be created from dict."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        symbol_table = {
            1: "One",
            2: "Two",
            3: "Three",
        }
        relm.automata.attach_symbol_table(dfa, symbol_table)
        input_symbol_table = dfa.input_symbols()
        output_symbol_table = dfa.output_symbols()
        self.assertEqual(input_symbol_table.num_symbols(), len(symbol_table))
        self.assertEqual(output_symbol_table.num_symbols(), len(symbol_table))
        for k, v in symbol_table.items():
            self.assertEqual(v, input_symbol_table.find(k))
            self.assertEqual(v, output_symbol_table.find(k))

    def _test_wildcard_helper(self, start_n, stop_n):
        # Shift by 1 to avoid 0
        start_n += 1
        stop_n += 1
        accepted_states = list(range(1, start_n))
        dfa = relm.automata.wildcard_fst(accepted_states)
        for s in accepted_states:
            self.assertTrue(relm.automata.apply_fst_accepted([s], dfa))
        not_accepted_states = list(range(start_n, stop_n))
        for s in not_accepted_states:
            self.assertFalse(relm.automata.apply_fst_accepted([s], dfa))

    def test_wildcard(self):
        """Test that wildcards (of limited cardinality) work."""
        self._test_wildcard_helper(30, 1000)
        self._test_wildcard_helper(10, 100)
        self._test_wildcard_helper(500, 10000)

    def test_automata_union(self):
        """Test building a union automata."""
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 3],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa))

    def test_automata_union_fast(self):
        """Test building a union automata from a list."""
        list_of_tokens = (1, 2, 3)
        dfa = relm.automata.union_automata_from_token_list(list_of_tokens)
        self.assertTrue(relm.automata.apply_fst_accepted([1],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([2],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([3],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 3],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([], dfa))

    def test_prefix_conversion(self):
        """Test that any prefix is accepted."""
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2],
                                                          dfa))
        original_start_state = relm.automata.automata_start_state(dfa)
        original_final_states = relm.automata.automata_final_states(dfa)
        dfa2 = relm.automata.convert_automata_to_prefix_acceptor(
            dfa, return_copy=True)
        end_start_state = relm.automata.automata_start_state(dfa2)
        end_final_states = relm.automata.automata_final_states(dfa2)
        self.assertEqual(original_start_state, end_start_state)
        self.assertNotEqual(set(original_final_states), set(end_final_states))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([3],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([2],
                                                          dfa2))
        for i in range(5, 100):
            self.assertFalse(relm.automata.apply_fst_accepted([i],
                                                              dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 1],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([2, 1],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4, 1],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 1, 1],
                                                          dfa2))

        dfa3 = relm.automata.convert_automata_to_prefix_no_acceptor(
            dfa2, return_copy=True)
        num_states = sum(1 for states in dfa.states())
        num_states3 = sum(1 for states in dfa3.states())
        self.assertEqual(num_states, num_states3)
        final_states = relm.automata.automata_final_states(dfa3)
        self.assertEqual(final_states, [])
        # TODO(mkuchnik): Should empty set [] be accepted?

    def test_BFS(self):
        """Test breadth first search."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        all_expected_paths = {(0, 1, 2, 3,)}
        all_BFS_paths = set(relm.automata.BFS_from_automata(dfa))
        self.assertEqual(all_expected_paths, all_BFS_paths)
        all_expected_edges = {(1, 2, 3,)}
        all_BFS_paths2, all_edges_visited = zip(
            *relm.automata.BFS_from_automata(
                dfa, return_edges_visited=True))
        all_BFS_paths2 = set(all_BFS_paths2)
        all_edges_visited = set(all_edges_visited)
        self.assertEqual(all_BFS_paths, all_BFS_paths2)
        self.assertEqual(all_expected_edges, all_edges_visited)

    def test_DFS(self):
        """Test breadth first search."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        all_expected_paths = {(0, 1, 2, 3,)}
        all_DFS_paths = set(relm.automata.DFS_from_automata(dfa))
        self.assertEqual(all_expected_paths, all_DFS_paths)
        all_expected_edges = {(1, 2, 3,)}
        all_DFS_paths2, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                dfa, return_edges_visited=True))
        all_DFS_paths2 = set(all_DFS_paths2)
        all_edges_visited = set(all_edges_visited)
        self.assertEqual(all_DFS_paths, all_DFS_paths2)
        self.assertEqual(all_expected_edges, all_edges_visited)

    def test_IDDFS(self):
        """Test iterative deepening depth first search."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        all_expected_paths = {(0, 1, 2, 3,)}
        all_DFS_paths = set(
            relm.automata.iterative_deepening_DFS_from_automata(dfa))
        self.assertEqual(all_expected_paths, all_DFS_paths)
        all_expected_edges = {(1, 2, 3,)}
        all_DFS_paths2, all_edges_visited = zip(
            *relm.automata.iterative_deepening_DFS_from_automata(
                dfa, return_edges_visited=True))
        all_DFS_paths2 = set(all_DFS_paths2)
        all_edges_visited = set(all_edges_visited)
        self.assertEqual(all_DFS_paths, all_DFS_paths2)
        self.assertEqual(all_expected_edges, all_edges_visited)

    def test_dijkstra(self):
        """Test dijkstra search."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        all_expected_paths = {(0, 1, 2, 3,)}

        def expand_neighbor_costs_fn(state_tokens):
            # TODO(mkuchnik): Replace with neighbor-based index
            return np.ones(10)

        all_shortest_paths = set(
            relm.automata.dijkstra_from_automata(
                dfa, expand_neighbor_costs_fn,
                return_edges_visited=False))
        self.assertEqual(all_expected_paths, all_shortest_paths)
        all_expected_edges = {(1, 2, 3,)}
        all_shortest_paths2, all_edges_visited, all_costs = zip(
            *relm.automata.dijkstra_from_automata(
                dfa, expand_neighbor_costs_fn,
                return_edges_visited=True, return_costs=True))
        all_shortest_paths2 = set(all_shortest_paths2)
        all_edges_visited = set(all_edges_visited)
        all_costs = set(all_costs)
        self.assertEqual(all_shortest_paths, all_shortest_paths2)
        self.assertEqual(all_expected_edges, all_edges_visited)
        all_expected_costs = set([3.0])
        self.assertEqual(all_expected_costs, all_costs)

    def test_dijkstra_blocked(self):
        """Test dijkstra search when costs is infinite."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        all_expected_paths = set()

        def expand_neighbor_costs_fn(state_tokens):
            # TODO(mkuchnik): Replace with neighbor-based index
            return np.ones(10) * np.inf

        all_shortest_paths = set(
            relm.automata.dijkstra_from_automata(
                dfa, expand_neighbor_costs_fn,
                return_edges_visited=False))
        self.assertEqual(all_expected_paths, all_shortest_paths)

    def test_cardinality(self):
        """Test that cardinality returns same result as BFS."""
        list_of_list_of_tokens = {(1, 2, 3), (4,), (5, 6), (7, 8, 9), (1, 10)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        BFS_len = sum(1 for _ in relm.automata.BFS_from_automata(dfa))
        combo_len = relm.automata.string_cardinality_sum_for_automata(dfa)
        self.assertEqual(BFS_len, combo_len)

    def test_optional_automata(self):
        """Test that optional (? in regex) automata work."""
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa))
        dfa = relm.automata.optional_automata(dfa)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa))

    def test_repeat_automata(self):
        """Test that repeat (* in regex) automata work."""
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(
            dfa, min_length=0, max_length=None)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4, 4],
                                                         dfa2))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(
            dfa, min_length=0, max_length=1)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4, 4],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4, 4, 4],
                                                          dfa2))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(
            dfa, min_length=0, max_length=2)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4, 4, 4],
                                                          dfa2))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(
            dfa, min_length=1, max_length=None)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4, 4],
                                                         dfa2))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(
            dfa, min_length=2, max_length=None)
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4, 4],
                                                         dfa2))
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        dfa2 = relm.automata.repeat_automata(dfa, min_length=1, max_length=3)
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 1, 2, 3],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2],
                                                          dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([5],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([4, 4, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted(
            [4 for _ in range(100)],
            dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted([4, 4, 4, 4],
                                                          dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3],
                                                         dfa2))
        self.assertTrue(relm.automata.apply_fst_accepted([1, 2, 3, 1, 2, 3, 4],
                                                         dfa2))
        self.assertFalse(relm.automata.apply_fst_accepted(
            [1, 2, 3, 1, 2, 3, 4, 4],
            dfa2))

    def test_next_states(self):
        """Test finding of next states at transition."""
        list_of_list_of_tokens = {(1, 2, 3), (4,), (5, 6), (7, 8, 9), (1, 10)}

        def fast_path(accept_automata, _state_tokens):
            transition_idx_path = (
                relm.automata.automata_next_states(
                    accept_automata,
                    _state_tokens,
                    return_edges_visited=True,
                    filter_nonfinal_terminal_states=True)
            )
            transition_idx_path = map(lambda x: x[1],
                                      transition_idx_path)
            return transition_idx_path

        def slow_path(accept_automata, _state_tokens):
            transition_idxs = {xx for x in list_of_list_of_tokens for xx in x}
            next_transition_automata = (
                relm.automata.union_fst(transition_idxs))
            if _state_tokens:
                transition_automata = (
                    relm.automata.automata_from_token_list(
                        _state_tokens))
                transition_automata = transition_automata.concat(
                    next_transition_automata)
            else:
                transition_automata = next_transition_automata

            intersected_automata = fst.intersect(
                accept_automata,
                transition_automata)
            intersected_automata = relm.automata.finalize_automata(
                intersected_automata,
                verify=False,
            )
            all_intersected_paths = (
                relm.automata.DFS_from_automata(
                    intersected_automata,
                    return_edges_visited=True))

            transition_idx_path = map(lambda x: (x[1][-1]),
                                      all_intersected_paths)
            return transition_idx_path

        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        transition_list = [1, 2]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([3]))
        path1 = set(list(fast_path(dfa, transition_list)))
        path2 = set(list(slow_path(dfa, transition_list)))
        self.assertEqual(path1, path2)
        transition_list = [1]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([2, 10]))
        path1 = set(list(fast_path(dfa, transition_list)))
        path2 = set(list(slow_path(dfa, transition_list)))
        self.assertEqual(path1, path2)
        transition_list = [4]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([]))
        path1 = set(list(fast_path(dfa, transition_list)))
        path2 = set(list(slow_path(dfa, transition_list)))
        self.assertEqual(path1, path2)
        transition_list = [7, 8]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([9]))
        path1 = set(list(fast_path(dfa, transition_list)))
        path2 = set(list(slow_path(dfa, transition_list)))
        self.assertEqual(path1, path2)

        # The with simplify
        list_of_list_of_tokens = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        symbol_table = {
            1169: "the",
            83: "t",
            400: "th",
            68: "e",
        }
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        transition_list = [83]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([258, 71]))
        transition_list = [400]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([68]))
        transition_list = [68]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([]))
        # With prefix
        dfa = relm.automata.convert_automata_to_prefix_acceptor(
            dfa, return_copy=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        transition_list = [83]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([258, 71]))
        transition_list = [400]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([68]))
        transition_list = [68]
        next_states = relm.automata.automata_next_states(
            dfa, transition_list,
            return_edges_visited=True)
        next_edges = set([x[1] for x in next_states])
        self.assertEqual(next_edges, set([]))

    def test_topological_sort(self):
        """Test the topological sort."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        self.assertEqual(relm.automata.topological_sort(dfa), [0, 1, 2, 3])
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        topo_sort = list(relm.automata.topological_sort(dfa))
        start_state = relm.automata.automata_start_state(dfa)
        self.assertEqual(topo_sort[0], start_state)
        final_states = relm.automata.automata_final_states(dfa)
        self.assertEqual(len(final_states), 1)
        final_state = final_states[0]
        self.assertEqual(topo_sort[-1], final_state)

    def test_truncate_automata(self):
        """Test the truncation of automata at depth."""
        symbol_table = {
            1: "one",
        }
        # With null
        dfa1 = relm.automata.wildcard_fst([1])
        dfa1 = relm.automata.union_automatas([dfa1,
                                              relm.automata.null_automata()])
        relm.automata.attach_symbol_table(dfa1, symbol_table)
        dfa2_0 = relm.automata.automata_from_token_list([])
        dfa2_1 = relm.automata.automata_from_token_list([1])
        dfa2_2 = relm.automata.automata_from_token_list([1, 1])
        dfa2_3 = relm.automata.automata_from_token_list([1, 1, 1])
        dfa2 = relm.automata.union_automatas([dfa2_0, dfa2_1, dfa2_2, dfa2_3])
        relm.automata.attach_symbol_table(dfa2, symbol_table)
        self.assertFalse(fst.equivalent(dfa1, dfa2))
        max_depth = 3
        dfa1_trunc = relm.automata.truncate_automata(dfa1, max_depth)
        relm.automata.attach_symbol_table(dfa1_trunc, symbol_table)
        self.assertTrue(fst.equivalent(dfa1_trunc, dfa2))

        symbol_table = {
            1: "one",
        }
        # Without null
        dfa1 = relm.automata.wildcard_fst([1])
        relm.automata.attach_symbol_table(dfa1, symbol_table)
        dfa2_1 = relm.automata.automata_from_token_list([1])
        dfa2_2 = relm.automata.automata_from_token_list([1, 1])
        dfa2_3 = relm.automata.automata_from_token_list([1, 1, 1])
        dfa2 = relm.automata.union_automatas([dfa2_1, dfa2_2, dfa2_3])
        relm.automata.attach_symbol_table(dfa2, symbol_table)
        self.assertFalse(fst.equivalent(dfa1, dfa2))
        max_depth = 3
        dfa1_trunc = relm.automata.truncate_automata(dfa1, max_depth)
        relm.automata.attach_symbol_table(dfa1_trunc, symbol_table)
        self.assertTrue(fst.equivalent(dfa1_trunc, dfa2))

        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
        }
        dfa1 = relm.automata.automata_from_token_list([1, 2, 3])
        relm.automata.attach_symbol_table(dfa1, symbol_table)
        dfa2 = relm.automata.automata_from_token_list([1, 2, 3])
        relm.automata.attach_symbol_table(dfa2, symbol_table)
        self.assertTrue(fst.equivalent(dfa1, dfa2))
        max_depth = 3
        dfa1_trunc = relm.automata.truncate_automata(dfa1, max_depth)
        relm.automata.attach_symbol_table(dfa1_trunc, symbol_table)
        self.assertTrue(fst.equivalent(dfa1_trunc, dfa2))
        max_depth = 2
        dfa1_trunc = relm.automata.truncate_automata(dfa1, max_depth)
        relm.automata.attach_symbol_table(dfa1_trunc, symbol_table)
        self.assertFalse(fst.equivalent(dfa1_trunc, dfa2))
        max_depth = 100
        dfa1_trunc = relm.automata.truncate_automata(dfa1, max_depth)
        relm.automata.attach_symbol_table(dfa1_trunc, symbol_table)
        self.assertTrue(fst.equivalent(dfa1_trunc, dfa2))

    def test_max_distance(self):
        """Test max distance works."""
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        distance = relm.automata.max_path_distance(dfa)
        self.assertEqual(distance, 3)
        list_of_list_of_tokens = {(1, 2, 3), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        distance = relm.automata.max_path_distance(dfa)
        self.assertEqual(distance, 3)
        list_of_list_of_tokens = {(1, 2, 3), (5, 6)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        distance = relm.automata.max_path_distance(dfa)
        self.assertEqual(distance, 3)
        list_of_list_of_tokens = {(1, 2, 3), (4,), (5, 6)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        distance = relm.automata.max_path_distance(dfa)
        self.assertEqual(distance, 3)
        list_of_list_of_tokens = {(1, 2, 3), (4,), (4, 5), (7, 8, 9, 10)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        distance = relm.automata.max_path_distance(dfa)
        self.assertEqual(distance, 4)

    def test_normalize(self):
        """Test that automata normalization works."""
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
        }
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata.normalize_automata(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1, 2, 3), (1, 2, 4)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata.normalize_automata(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                if a.ilabel == 1:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 2:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 3:
                    self.assertEqual(float(a.weight), 0.5)
                elif a.ilabel == 4:
                    self.assertEqual(float(a.weight), 0.5)
                else:
                    raise ValueError("Unknown arc {}".format(a))
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (2,), (3,), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata.normalize_automata(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 0.25)
        # Prefix
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (1, 2,), (1, 2, 3,), (1, 2, 3, 4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata.normalize_automata(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)

    def test_normalize_implementations_DFS(self):
        """Test automata normalization implementations."""
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
        }
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_DFS(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1, 2, 3), (1, 2, 4)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_DFS(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                if a.ilabel == 1:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 2:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 3:
                    self.assertEqual(float(a.weight), 0.5)
                elif a.ilabel == 4:
                    self.assertEqual(float(a.weight), 0.5)
                else:
                    raise ValueError("Unknown arc {}".format(a))
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (2,), (3,), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_DFS(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 0.25)
        # Prefix
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (1, 2,), (1, 2, 3,), (1, 2, 3, 4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_DFS(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)

    def test_normalize_implementations_combinatorial(self):
        """Test automata normalization implementations."""
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
        }
        dfa = relm.automata.automata_from_token_list([1, 2, 3])
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_combinatorial(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1, 2, 3), (1, 2, 4)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_combinatorial(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                if a.ilabel == 1:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 2:
                    self.assertEqual(float(a.weight), 1.0)
                elif a.ilabel == 3:
                    self.assertEqual(float(a.weight), 0.5)
                elif a.ilabel == 4:
                    self.assertEqual(float(a.weight), 0.5)
                else:
                    raise ValueError("Unknown arc {}".format(a))
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (2,), (3,), (4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_combinatorial(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 0.25)
        # Prefix
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (1, 2,), (1, 2, 3,), (1, 2, 3, 4,)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa = relm.automata._normalize_automata_combinatorial(dfa)
        for s in normalized_dfa.states():
            for a in normalized_dfa.arcs(s):
                self.assertEqual(float(a.weight), 1.0)

    def test_normalize_implementations_equivalent(self):
        """Test automata normalization implementations are equivalent."""
        regex = "the"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        dfa = relm.automata.automata_from_token_list(char_tokens)
        dfa = relm.automata.attach_symbol_table(
            dfa, inverted_full_vocab)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        dfa = preprocessor.transform(dfa)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))
        regex = "The man was walking."
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        dfa = relm.automata.automata_from_token_list(char_tokens)
        dfa = relm.automata.attach_symbol_table(
            dfa, inverted_full_vocab)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        dfa = preprocessor.transform(dfa)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))

        regex = "aaaaaaaaaaaaaa"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        dfa = relm.automata.automata_from_token_list(char_tokens)
        dfa = relm.automata.attach_symbol_table(
            dfa, inverted_full_vocab)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        dfa = preprocessor.transform(dfa)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))

        # a, b, bb, bbb
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (2,), (2, 2,), (2, 2, 2)}
        dfa = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(dfa, symbol_table)
        normalized_dfa_dfs = relm.automata._normalize_automata_DFS(dfa)
        normalized_dfa_combo = relm.automata._normalize_automata_combinatorial(
            dfa)
        self.assertTrue(fst.equivalent(normalized_dfa_dfs,
                                       normalized_dfa_combo))

    def test_find_suffix(self):
        """Test finding suffix."""
        prefix_automata = relm.automata.automata_from_token_list([1, 2, 3])
        suffix_automata = relm.automata.automata_from_token_list([4])
        full_automata = prefix_automata.copy().concat(suffix_automata)
        found_suffix_automata = relm.automata.find_suffix_from_prefix(
            prefix_automata, full_automata)
        self.assertTrue(fst.equivalent(suffix_automata, found_suffix_automata))

        prefix_automata = relm.automata.automata_from_token_list([1, 2, 3])
        suffix_automata = relm.automata.automata_from_token_list([])
        full_automata = prefix_automata.copy().concat(suffix_automata)
        found_suffix_automata = relm.automata.find_suffix_from_prefix(
            prefix_automata, full_automata)
        num_states = sum(1 for states in found_suffix_automata.states())
        self.assertEqual(num_states, 0)

        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        list_of_list_of_tokens = {(1,), (2,), (2, 2,), (2, 2, 2)}
        prefix_automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        relm.automata.attach_symbol_table(prefix_automata, symbol_table)
        suffix_automata = prefix_automata.copy()
        full_automata = prefix_automata.copy().concat(suffix_automata)
        found_suffix_automata = relm.automata.find_suffix_from_prefix(
            prefix_automata, full_automata)
        self.assertTrue(fst.equivalent(suffix_automata, found_suffix_automata))

    def test_add_sink_acceptor(self):
        """Test adding a sink acceptor."""
        automata = relm.automata.automata_from_token_list([1, 2, 3])
        acceptless_automata = \
            relm.automata.convert_automata_to_prefix_no_acceptor(
                automata)
        automata2 = \
            relm.automata.convert_automata_to_sink_acceptor(
                acceptless_automata)
        automata2 = relm.automata.finalize_automata(automata2)
        self.assertTrue(fst.equivalent(automata, automata2))

    def test_contains_epsilon(self):
        """Test that epsilon transitions are detected."""
        automata = relm.automata.automata_from_token_list([1, 2, 3])
        self.assertFalse(relm.automata.contains_epsilon(automata))
        automata = automata.concat(
            automata)
        self.assertTrue(relm.automata.contains_epsilon(automata))

    def test_remap_edge_values(self):
        """Test that remapping edges works."""
        automata = relm.automata.automata_from_token_list([1])
        edges_to_new_values = {
            1: 1,
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        self.assertTrue(fst.equivalent(automata, automata2))
        edges_to_new_values = {
            1: 2,
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        ref_automata = relm.automata.automata_from_token_list([2])
        self.assertFalse(fst.equivalent(automata, automata2))
        self.assertTrue(fst.equivalent(ref_automata, automata2))
        edges_to_new_values = {
            1: (2, 3),
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        ref_automata = relm.automata.automata_from_token_list([2, 3])
        self.assertFalse(fst.equivalent(automata, automata2))
        self.assertTrue(fst.equivalent(ref_automata, automata2))

        list_of_list_of_tokens = {(1,), (2,)}
        automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        edges_to_new_values = {
            1: (2, 3),
            2: 2,
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        list_of_list_of_tokens = {(2, 3), (2,)}
        ref_automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(fst.equivalent(ref_automata, automata2))

        list_of_list_of_tokens = {(1,), (2,)}
        automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        edges_to_new_values = {
            1: (2, 3),
            2: (2, 3),
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        list_of_list_of_tokens = {(2, 3)}
        ref_automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(fst.equivalent(ref_automata, automata2))

        list_of_list_of_tokens = {(1, 2, 3, 10), (1, 2, 3, 11)}
        automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        edges_to_new_values = {
            1: (2,),
            2: (2, 3),
            3: 3,
            10: 10,
            11: 11,
        }
        automata2 = relm.automata.remap_edge_values(
            automata, edges_to_new_values)
        automata2 = relm.automata.finalize_automata(automata2)
        list_of_list_of_tokens = {(2, 2, 3, 3, 10), (2, 2, 3, 3, 11)}
        ref_automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        self.assertTrue(fst.equivalent(ref_automata, automata2))
