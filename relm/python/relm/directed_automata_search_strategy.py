"""Automata-based searches which do score paths.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""
import itertools
import pprint
import time

import numpy as np
import pywrapfst as fst
import torch
import torch.profiler

import relm.indexing
import relm.model_wrapper
from relm.automata_search_strategy import AutomataSearchStrategy
from relm.relm_search_query import SearchQuery

logger = relm.relm_logging.get_logger()


class DirectedAutomataSearchStrategy(AutomataSearchStrategy):
    """Represents automata-based search strategies that do score paths."""

    def __init__(self, test_relm: relm.model_wrapper.TestableModel):
        """Take a test_relm object."""
        super().__init__(test_relm)

    def _search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        logger.debug("Searching query: {}".format(query))
        # TODO(mkuchnik): Investigate building both of these in parallel
        automata_start_time = time.perf_counter()
        query_automata = self._build_query_automata(query)
        automata_end_time = time.perf_counter()
        logger.info("Compiled query automata in {:.2} seconds".format(
            automata_end_time - automata_start_time))
        automata_start_time = time.perf_counter()
        accept_automata = self._build_accept_automata(query)
        automata_end_time = time.perf_counter()
        logger.info("Compiled accept automata in {:.2} seconds".format(
            automata_end_time - automata_start_time))
        if query.experimental_fast_start is not None:
            fast_start = query.experimental_fast_start
        else:
            fast_start = False
        if not query.experimental_very_fast_start:
            query_cardinality, counts_by_length = \
                self._inspect_query_automata(query_automata,
                                             fast_start=fast_start,
                                             log_tag="[Query]")
            if query_cardinality:
                first_accepting_length = next(
                    (i for i, x in enumerate(counts_by_length) if x),
                    None) + 1
                logger.info("First accepting length: {}".format(
                    first_accepting_length))
            else:
                logger.info("No matching sequences in query!")
                return []
        else:
            query_cardinality = np.inf
            first_accepting_length = 0
            logger.info("Skipping query analysis for very fast start.")

        if accept_automata and not fast_start:
            self._inspect_query_automata(accept_automata, fast_start=True,
                                         log_tag="[Accept]")

        if not query_cardinality:
            logger.warning("Query automata is empty. Returning.")
            return []

        logger.info("Entering search iterator.")
        expected_num_samples = query.num_samples

        if query.num_samples is None:
            expected_num_samples = query_cardinality
        else:
            expected_num_samples = min(query.num_samples,
                                       query_cardinality)

        logger.info("Expecting to return {} samples.".format(
            expected_num_samples))
        bfs_memory_complexity = self._bfs_memory_space(query_cardinality)
        available_memory = self._available_memory()
        try:
            bfs_memory_fraction = bfs_memory_complexity / available_memory
            logger.info("BFS memory complexity: {:.2%}=({}/{}GB)".format(
                bfs_memory_fraction,
                bfs_memory_complexity // 1e9,  # Bytes to GB
                available_memory // 1e9))  # Bytes to GB
        except OverflowError:
            logger.warning("BFS memory complexity too large to compute!")

        assert (query.experimental_dijkstra or query.experimental_greedy_search
                or query.experimental_random_sampling)

        if not (query.experimental_dijkstra
                or query.experimental_greedy_search
                or query.experimental_random_sampling):
            raise ValueError("Directed query not selected.")

        if query.experimental_dijkstra:
            logger.info("Forcing dijkstra")
            aggregation_function = \
                query.experimental_dijkstra_aggregation_function

            search_iter = (
                self._dijkstra_matching_query_strings(
                    query,
                    query_automata,
                    accept_automata,
                    aggregation_function=aggregation_function,
                ))
        elif query.experimental_greedy_search:
            logger.info("Forcing greedy")
            search_iter = (
                self._greedy_matching_query_strings(
                    query,
                    query_automata,
                    accept_automata,
                ))
        elif query.experimental_random_sampling:
            logger.info("Forcing random")
            aggregation_function = \
                query.experimental_dijkstra_aggregation_function
            unbiased_sampling = \
                query.experimental_random_sampling_normalization
            if unbiased_sampling is None:
                unbiased_sampling = True
            search_iter = (
                self._random_sampling_matching_query_strings(
                    query,
                    query_automata,
                    accept_automata,
                    aggregation_function=aggregation_function,
                    unbiased_sampling=unbiased_sampling,
                ))
        else:
            raise RuntimeError("Unchecked path.")

        # However, we still can sanity check if flag is passed
        if query.experimental_safe:
            model_cache = dict()

            def check_only_accepting(path_cost):
                path, cost = path_cost
                if self._model_accepts(
                    query, path, model_cache,
                        accept_automata=accept_automata):
                    return path, cost
                else:
                    prob_path_ranks = relm.indexing.path_to_top_k(
                        self.test_relm, path)

                    assert len(prob_path_ranks) == len(path)
                    valid_top_k = (query.top_k_sampling is None
                                   or np.all(prob_path_ranks <=
                                             query.top_k_sampling))
                    misc_dict = {"top_k_ranks": prob_path_ranks,
                                 "valid_top_k": valid_top_k,
                                 "cost": cost,
                                 }
                    misc_msg = pprint.pformat(misc_dict)
                    raise RuntimeError(
                        "Path: {} ('{}') proposed but not"
                        " accepted.{}".format(
                            path,
                            self.test_relm.tokens_to_words(path),
                            misc_msg,
                        )
                    )

            search_iter = map(check_only_accepting, search_iter)

        return search_iter

    def _dijkstra_matching_query_strings(self,
                                         query,
                                         query_automata,
                                         accept_automata,
                                         return_str=False,
                                         aggregation_function=None):
        """Dijkstra visit strings that can match on the automata."""
        top_k_sampling = query.top_k_sampling
        max_sequence_length = query.sequence_length

        if accept_automata:
            logger.info("Accept automata:\n{}".format(
                relm.automata.summarize_automata(accept_automata, True)))
        logger.info("Query automata:\n{}".format(
            relm.automata.summarize_automata(query_automata, True)))

        if max_sequence_length is not None:
            logger.info("Sequence length limited to:"
                        " {}".format(max_sequence_length))

        def cost_fn(state_toks, verbose=False):
            # TODO(mkuchnik): when top-k disabled, don't run
            transition_probabilities = \
                self._model_predict_transition_probabilities(
                    state_toks, return_numpy=False,
                    temperature=query.temperature)
            if verbose:
                top_ret = transition_probabilities.topk(20)
                logger.debug(
                    "State toks '{}' -> '{}'\nval {}\nidx {}\nstr {}"
                    .format(
                        state_toks,
                        transition_probabilities.shape,
                        top_ret[0],
                        top_ret[1],
                        self.test_relm.tokenizer.convert_ids_to_tokens(
                            top_ret[1])))
            return transition_probabilities

        def expand_neighbor_costs_fn(state_tokens):
            logger.debug("Expanding states: {}".format(state_tokens))
            if not state_tokens:
                state_tokens = None
            if query.experimental_inner_query_callback is not None:
                # TODO(mkuchnik): Hacky
                query.experimental_inner_query_callback()

            # Schedule GPU work
            transition_probabilities = cost_fn(state_tokens)
            if top_k_sampling is not None:
                top_k_val = torch.topk(transition_probabilities,
                                       axis=0,
                                       k=top_k_sampling)
                original_transition_probabilites = transition_probabilities
                transition_probabilities = torch.zeros_like(
                    transition_probabilities)
                transition_probabilities = \
                    transition_probabilities.scatter(
                        0,
                        top_k_val.indices,
                        top_k_val.values)
            else:
                original_transition_probabilites = transition_probabilities

            if state_tokens is not None:
                _state_tokens = tuple(state_tokens)
            else:
                _state_tokens = tuple()

            # We map from this vanilla state to encoded state
            _state_tokens = self._encode_automata_tokens(
                _state_tokens)

            accepted_probs = np.zeros(len(transition_probabilities),
                                      dtype=bool)
            if accept_automata is not None:
                logger.debug("Calculating accepts for state: {}".format(
                    _state_tokens))
                _, transition_idx_path = get_next_states(accept_automata,
                                                         _state_tokens)
                transition_idx_path = self._decode_automata_tokens(
                    transition_idx_path)

                set_accepted_probs = False
                with torch.profiler.record_function("transition_path_idx"):
                    for i in transition_idx_path:
                        logger.debug("Setting transition accept path: {}"
                                     .format(i))
                        accepted_probs[i] = True
                        set_accepted_probs = True
                if not set_accepted_probs:
                    automata_str = relm.automata.summarize_automata(
                        accept_automata)
                    logger.info(
                        "Accept automata given but no accepts found at state"
                        " {} of automata:\n{}".format(
                            _state_tokens, automata_str)
                    )

            if query.experimental_advanced_parsing_dynamic_minimize:
                # We have to remove non-minimal paths
                proposed_sentence_base = \
                    self.test_relm._decode_gen_sequence(
                        _state_tokens)
                proposed_paths = [(i,) for i in
                                  range(len(accepted_probs))]
                proposed_sentences_suffix = \
                    self.test_relm._batch_decode_gen_sequence(
                        proposed_paths)
                proposed_sentences = [proposed_sentence_base + pss
                                      for pss
                                      in proposed_sentences_suffix]
                proposed_paths = [tuple(_state_tokens) + pp for pp in
                                  proposed_paths]
                double_encoded_paths = \
                    self.test_relm.tokenizer(proposed_sentences)
                double_encoded_paths = double_encoded_paths["input_ids"]
                for i, (pp, dp) in enumerate(zip(proposed_paths,
                                                 double_encoded_paths)):
                    logger.debug("pp != dp. {}\n{}".format(pp, dp))
                    if pp != tuple(dp):
                        accepted_probs[i] = False
                        transition_probabilities[i] = False

            if not query.experimental_penalized_accepted_probability:
                transition_probabilities[accepted_probs] = 1.0
            else:
                # NOTE(mkuchnik): Because we are likely not in top-k,
                # the current log_probability is nan. Rather, just copy.
                transition_probabilities[accepted_probs] = \
                    original_transition_probabilites[accepted_probs]

            if query.experimental_avoid_not_accepted_probability:
                if np.any(accepted_probs):
                    transition_probabilities[~accepted_probs] = 0.0

            transition_probabilities /= torch.sum(
                transition_probabilities)

            log_probs = -torch.log(transition_probabilities)

            log_probs = log_probs.cpu().numpy()
            if top_k_sampling is not None:
                filtered_idxs = torch.ones(log_probs.shape,
                                           dtype=bool)
                top_k_indices = top_k_val.indices.cpu()
                filtered_idxs.scatter_(0, top_k_indices, 0)
                filtered_idxs = filtered_idxs.numpy()
                assert np.sum(~filtered_idxs) == top_k_sampling, \
                    "Expected to find k={} samples, found {}".format(
                        top_k_sampling, np.sum(~filtered_idxs))
                log_probs[filtered_idxs & ~accepted_probs] = np.inf
                # zero penalty for accepted
                if not (query
                        .experimental_penalized_accepted_probability):
                    log_probs[accepted_probs] = 0.0
                if query.experimental_avoid_not_accepted_probability:
                    if np.any(accepted_probs):
                        log_probs[~accepted_probs] = np.inf

            return log_probs

        def automata_decode_function(token):
            return self._decode_automata_token(token)

        all_viable_paths = relm.automata.dijkstra_from_automata(
            query_automata, expand_neighbor_costs_fn,
            return_edges_visited=True,
            max_sequence_length=max_sequence_length,
            cost_aggregation_function=aggregation_function,
            automata_decode_function=automata_decode_function,
            return_costs=True,
            beam_size=query.experimental_dijkstra_beam_size,
            batch_size=None,
        )
        # TODO(mkuchnik): Investigate prefetching
        all_edges_visited = map(lambda x: (x[1], x[2]), all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _batched_dijkstra_matching_query_strings(self,
                                                 query,
                                                 query_automata,
                                                 accept_automata,
                                                 batch_size,
                                                 return_str=False,
                                                 aggregation_function=None):
        """Dijkstra visit strings that can match on the automata with batch."""
        top_k_sampling = query.top_k_sampling
        max_sequence_length = query.sequence_length
        verbose = False

        def batch_cost_fn(state_toks_batch):
            transition_probabilities = \
                self._batch_model_predict_transition_probabilities(
                    state_toks_batch, return_numpy=False,
                    temperature=query.temperature)
            return transition_probabilities

        def batch_expand_neighbor_costs_fn(batch_state_tokens):
            logger.debug("Expanding states: {}".format(batch_state_tokens))

            def null_to_none(x):
                return None if not x else x

            batch_state_tokens = list(map(null_to_none, batch_state_tokens))
            if query.experimental_inner_query_callback is not None:
                # TODO(mkuchnik): Hacky
                query.experimental_inner_query_callback()

            # Schedule GPU work
            transition_probabilities = batch_cost_fn(batch_state_tokens)
            assert len(transition_probabilities.shape) == 2
            logger.info("Transition_probabilities: {}".format(
                transition_probabilities.shape))
            if top_k_sampling is not None:
                top_k_val = torch.topk(transition_probabilities,
                                       axis=1,
                                       k=top_k_sampling)
                original_transition_probabilites = transition_probabilities
                transition_probabilities = torch.zeros_like(
                    transition_probabilities)
                transition_probabilities = \
                    transition_probabilities.scatter(
                        1,
                        top_k_val.indices,
                        top_k_val.values)

            accepted_probs = torch.zeros(transition_probabilities.shape,
                                         dtype=bool,
                                         device=self.test_relm.device())
            if accept_automata is not None:

                def state_tokens_builder_fn(x):
                    if x is not None:
                        return tuple(x)
                    else:
                        return tuple()

                _batch_state_tokens = list(map(state_tokens_builder_fn,
                                               batch_state_tokens))
                for i in range(len(accepted_probs)):
                    _state_tokens = _batch_state_tokens[i]

                    _, transition_idx_path = get_next_states(accept_automata,
                                                             _state_tokens)
                    transition_idx_path = self._decode_automata_tokens(
                        transition_idx_path)
                    with torch.profiler.record_function("transition_path_idx"):
                        for j in transition_idx_path:
                            accepted_probs[i, j] = True

            if query.experimental_advanced_parsing_dynamic_minimize:
                raise NotImplementedError(
                    "Dynamic parsing not implemented for batching")

            if not query.experimental_penalized_accepted_probability:
                transition_probabilities[accepted_probs] = 1.0
            else:
                # NOTE(mkuchnik): Because we are likely not in top-k,
                # the current log_probability is nan. Rather, just copy.
                transition_probabilities[accepted_probs] = \
                    original_transition_probabilites[accepted_probs]
            if query.experimental_avoid_not_accepted_probability:
                if np.any(accepted_probs):
                    transition_probabilities[~accepted_probs] = 0.0
            transition_probabilities /= torch.sum(
                transition_probabilities, axis=1).unsqueeze(-1)

            log_probs = -torch.log(transition_probabilities)

            if top_k_sampling is not None:
                filtered_idxs = torch.ones(
                    log_probs.shape,
                    dtype=bool,
                    device=self.test_relm.device())
                top_k_indices = top_k_val.indices
                filtered_idxs.scatter_(1, top_k_indices, 0)
                if verbose:
                    _num_nonfiltered_idxs = torch.sum(~filtered_idxs,
                                                      axis=1)
                    assert torch.all(
                        _num_nonfiltered_idxs == top_k_sampling), \
                        "Expected to find k={} samples," \
                        " found {}".format(
                            top_k_sampling,
                            _num_nonfiltered_idxs,
                        )
                    logger.debug("Filtered idxs: {}".format(
                        filtered_idxs))
                    logger.debug("Filtered idxs: {}".format(
                        np.where(filtered_idxs.cpu().numpy())))
                    # TODO(mkuchnik): Check and cleanup
                    logger.debug("Accepted probs: {}".format(
                        accepted_probs))
                    logger.debug("Accepted probs: {}".format(
                        np.where(accepted_probs.cpu().numpy())))
                filtered_mask = filtered_idxs & ~accepted_probs
                log_probs[filtered_mask] = np.inf
                # zero penalty for accepted
                if not (query
                        .experimental_penalized_accepted_probability):
                    log_probs[accepted_probs] = 0.0
                if query.experimental_avoid_not_accepted_probability:
                    if np.any(accepted_probs):
                        log_probs[~accepted_probs] = np.inf
            logger.debug("Returning log probs: {}".format(log_probs.shape))
            return log_probs

        def automata_decode_function(token):
            return self._decode_automata_token(token)

        expand_neighbor_costs_fn = None

        all_viable_paths = relm.automata.dijkstra_from_automata(
            query_automata, expand_neighbor_costs_fn,
            return_edges_visited=True,
            max_sequence_length=max_sequence_length,
            cost_aggregation_function=aggregation_function,
            automata_decode_function=automata_decode_function,
            return_costs=True,
            beam_size=query.experimental_dijkstra_beam_size,
            batch_size=batch_size,
            batch_expand_neighbor_costs_fn=batch_expand_neighbor_costs_fn,
        )
        # TODO(mkuchnik): Investigate prefetching
        all_edges_visited = map(lambda x: (x[1], x[2]), all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _greedy_matching_query_strings(self,
                                       query,
                                       query_automata,
                                       accept_automata,
                                       return_str=False,
                                       aggregation_function=None):
        """Greedy visit strings that can match on the automata."""
        top_k_sampling = query.top_k_sampling
        max_sequence_length = query.sequence_length
        if max_sequence_length is not None:
            logger.info("Sequence length limited to:"
                        " {}".format(max_sequence_length))

        def cost_fn(state_toks, verbose=False):
            # TODO(mkuchnik): when top-k disabled, don't run
            if not isinstance(state_toks, dict):
                transition_probabilities = \
                    self._model_predict_transition_probabilities(
                        state_toks, return_numpy=False,
                        temperature=query.temperature)
            else:
                # Assume dict of input_ids and attention_mask
                transition_probabilities = \
                    self._model_predict_transition_probabilities(
                        **state_toks, return_numpy=False,
                        temperature=query.temperature)
            if verbose:
                top_ret = transition_probabilities.topk(20)
                logger.debug(
                    "State toks '{}' -> '{}'\nval {}\nidx {}\nstr {}"
                    .format(
                        state_toks,
                        transition_probabilities.shape,
                        top_ret[0],
                        top_ret[1],
                        self.test_relm.tokenizer.convert_ids_to_tokens(
                            top_ret[1])))
            return transition_probabilities.cpu().numpy()

        def expand_neighbor_costs_fn(state_tokens):
            logger.debug("Expanding states: {}".format(state_tokens))
            if not state_tokens:
                state_tokens = None
            if query.experimental_inner_query_callback is not None:
                # TODO(mkuchnik): Hacky
                query.experimental_inner_query_callback()
            transition_probabilities = cost_fn(state_tokens)
            if accept_automata is not None:
                accepted_probs = np.zeros(len(transition_probabilities),
                                          dtype=bool)
                transition_idxs = list(
                    range(1, len(transition_probabilities) - 1))
                if state_tokens is not None:
                    _state_tokens = tuple(state_tokens)
                else:
                    _state_tokens = tuple()
                if _state_tokens:
                    transition_automata = (
                        relm.automata.automata_from_token_list(_state_tokens))
                    next_transition_automata = (
                        relm.automata.union_fst(transition_idxs))
                    # TODO(mkuchnik): Optimize
                    transition_automata = transition_automata.concat(
                        next_transition_automata)
                else:
                    transition_automata = (
                        relm.automata.union_fst(transition_idxs))

                # TODO(mkuchnik): Optimize
                transition_automata.arcsort()
                intersected_automata = fst.intersect(transition_automata,
                                                     accept_automata)
                intersected_automata = relm.automata.finalize_automata(
                    intersected_automata)
                all_intersected_paths = relm.automata.BFS_from_automata(
                    intersected_automata, return_edges_visited=True)
                transition_idx_path = map(lambda x: (x[1][-1]),
                                          all_intersected_paths)
                transition_idx_path = self._decode_automata_tokens(
                    transition_idx_path)
                accepted_a_path = False
                for i in transition_idx_path:
                    accepted_a_path = True
                    path = _state_tokens + (i,)
                    logger.debug("Accepting via automata: {}".format(path))
                    accepted_probs[i] = True
                if not accepted_a_path:
                    path = _state_tokens
                    logger.debug("Didn't accept prefix via automata: {}"
                                 ".".format(path))
            else:
                accepted_probs = np.zeros(len(transition_probabilities),
                                          dtype=bool)
            token_ranks = relm.indexing.rank_array(transition_probabilities,
                                                   reverse=True)
            # Filter and renormalize
            if top_k_sampling is not None:
                filtered_idxs = token_ranks > top_k_sampling
                assert np.sum(~filtered_idxs) == top_k_sampling, \
                    "Expected to find k={} samples, found {}".format(
                        top_k_sampling, np.sum(~filtered_idxs))
            else:
                filtered_idxs = np.zeros(len(transition_probabilities),
                                         dtype=bool)
            transition_probabilities[filtered_idxs] = 0.0
            transition_probabilities[accepted_probs] = 1.0
            transition_probabilities /= sum(transition_probabilities)
            transition_probabilities[filtered_idxs] = 1e-30
            log_probs = -np.log(transition_probabilities)
            log_probs[filtered_idxs & ~accepted_probs] = np.inf
            return log_probs

        def automata_decode_function(token):
            return self._decode_automata_token(token)

        all_viable_paths = relm.automata.greedy_search_from_automata(
            query_automata, expand_neighbor_costs_fn,
            return_edges_visited=True,
            max_sequence_length=max_sequence_length,
            return_costs=True,
            automata_decode_function=automata_decode_function,
        )
        # TODO(mkuchnik): Investigate prefetching
        all_edges_visited = map(lambda x: (x[1], x[2]), all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _random_sampling_matching_query_strings(
        self,
        query,
        query_automata,
        accept_automata,
        return_str=False,
        aggregation_function=None,
        unbiased_sampling=True
    ):
        """Randomly visit strings that can match on the automata."""
        top_k_sampling = query.top_k_sampling
        max_sequence_length = query.sequence_length
        if max_sequence_length is not None:
            logger.info("Sequence length limited to:"
                        " {}".format(max_sequence_length))

        if unbiased_sampling:
            # NOTE(mkuchnik): Only query automata has all path counts to
            # normalize
            logger.info("Normalizing query automata.")
            max_query_length = query.sequence_length
            if max_query_length is None:
                max_query_length = self._max_n()
            else:
                max_query_length = min(max_query_length, self._max_n())
            if accept_automata:
                # NOTE(mkuchnik): Normalization only necessary if a
                # a prefix exists
                suffix_automata = relm.automata.find_suffix_from_prefix(
                    accept_automata, query_automata)
                prefix_automata = accept_automata.copy()
                patched_prefix_automata = \
                    relm.automata.convert_automata_to_prefix_no_acceptor(
                        prefix_automata)
                # TODO(mkuchnik): Avoid copies
                patched_prefix_automata = \
                    relm.automata.convert_automata_to_sink_acceptor(
                        patched_prefix_automata)

                new_full_automata = patched_prefix_automata.concat(
                    suffix_automata)
                # We have accepts from prefix, remove them
                prefix_automata.arcsort()
                new_full_automata = fst.difference(new_full_automata,
                                                   prefix_automata)
                new_full_automata = relm.automata.finalize_automata(
                    new_full_automata)
                new_full_automata.arcsort()
                difference_automata = fst.difference(
                    query_automata, new_full_automata)

                dropped_edges = []
                dfs_iter = relm.automata.DFS_from_automata(
                    difference_automata, return_edges_visited=True)
                for _, edges in itertools.islice(dfs_iter, 100):
                    dropped_edges.append(edges)
                if dropped_edges:
                    logger.debug("Dropping cross-paths: {}".format(
                        dropped_edges))

            reassign = False
            if reassign:
                accept_automata = \
                    relm.automata.convert_automata_to_prefix_acceptor(
                        prefix_automata)
                query_automata = new_full_automata
            if accept_automata:
                query_automata = relm.automata.normalize_automata(
                    query_automata, max_length=max_query_length)
            else:
                logger.info(
                    "Normalization not necessary without accept automata.")

        def cost_fn(state_toks):
            # TODO(mkuchnik): when top-k disabled, don't run
            transition_probabilities = \
                self._model_predict_transition_probabilities(
                    state_toks, return_numpy=False,
                    temperature=query.temperature)
            return transition_probabilities

        def expand_neighbor_costs_fn(state_tokens):
            logger.debug("Expanding states: {}".format(state_tokens))
            if not state_tokens:
                state_tokens = None
            if query.experimental_inner_query_callback is not None:
                # TODO(mkuchnik): Hacky
                query.experimental_inner_query_callback()

            # Schedule GPU work
            transition_probabilities = cost_fn(state_tokens)
            if top_k_sampling is not None:
                top_k_val = torch.topk(transition_probabilities,
                                       axis=0,
                                       k=top_k_sampling)
                original_transition_probabilites = transition_probabilities
                transition_probabilities = torch.zeros_like(
                    transition_probabilities)
                transition_probabilities = \
                    transition_probabilities.scatter(
                        0,
                        top_k_val.indices,
                        top_k_val.values)
            else:
                original_transition_probabilites = transition_probabilities

            if state_tokens is not None:
                _state_tokens = tuple(state_tokens)
            else:
                _state_tokens = tuple()

            # We map from this vanilla state to encoded state
            _state_tokens = self._encode_automata_tokens(
                _state_tokens)

            if accept_automata is not None:
                accepted_probs = np.zeros(len(transition_probabilities),
                                          dtype=bool)
                logger.debug("Calculating accepts for state: {}".format(
                    _state_tokens))
                _, transition_idx_path = get_next_states(accept_automata,
                                                         _state_tokens)
                transition_idx_path = self._decode_automata_tokens(
                    transition_idx_path)

                set_accepted_probs = False
                with torch.profiler.record_function("transition_path_idx"):
                    for i in transition_idx_path:
                        logger.debug("Setting transition accept path: {}"
                                     .format(i))
                        accepted_probs[i] = True
                        set_accepted_probs = True
                if not set_accepted_probs:
                    automata_str = relm.automata.summarize_automata(
                        accept_automata)
                    logger.info(
                        "Accept automata given but no accepts found at state"
                        " {} of automata:\n{}".format(
                            _state_tokens, automata_str)
                    )
            else:
                accepted_probs = np.zeros(len(transition_probabilities),
                                          dtype=bool)
                set_accepted_probs = False
                logger.debug("No accept automata given")

            if query.experimental_advanced_parsing_dynamic_minimize:
                # We have to remove non-minimal paths
                proposed_sentence_base = \
                    self.test_relm._decode_gen_sequence(
                        _state_tokens)
                proposed_paths = [(i,) for i in
                                  range(len(accepted_probs))]
                proposed_sentences_suffix = \
                    self.test_relm._batch_decode_gen_sequence(
                        proposed_paths)
                proposed_sentences = [proposed_sentence_base + pss
                                      for pss
                                      in proposed_sentences_suffix]
                proposed_paths = [_state_tokens + pp for pp in
                                  proposed_paths]
                double_encoded_paths = \
                    self.test_relm.tokenizer(proposed_sentences)
                double_encoded_paths = double_encoded_paths["input_ids"]
                for i, (pp, dp) in enumerate(zip(proposed_paths,
                                                 double_encoded_paths)):
                    logger.debug("pp != dp. {}\n{}".format(pp, dp))
                    if pp != tuple(dp):
                        accepted_probs[i] = False
                        transition_probabilities[i] = False

            if not query.experimental_penalized_accepted_probability:
                if unbiased_sampling:
                    # NOTE(mkuchnik): Rather than set accepted probs to 1,
                    # we weight them by the path count
                    # The semantics we have are as follows:
                    # Suppose we have the automata represented by a -> b, where
                    # "a" is a prefix. If we have deletes over "a,b" as well,
                    # the total string set is "a", "b", "ab".
                    # ***Prefix***
                    # We have already said that "a" is a prefix, so if "a" is
                    # deleted, then "" is a prefix, but "b" will never be a
                    # prefix, which naturally falls out of having all prefixes
                    # of a prefix be part of the prefix.
                    # ***Suffix***
                    # Since there are 3 strings in the set, but only "a" and
                    # "a_0" follow from a prefix, we have "a" prefix with 66%
                    # chance and "" prefix with 33% chance.

                    # Step 1: Compute Accept set
                    if accept_automata:
                        _, query_transition_idx_path = get_next_states(
                            accept_automata, _state_tokens, return_arcs=True,
                            filter_nonfinal_transition_states=False,
                            filter_nonfinal_terminal_states=False,
                        )
                        accept_tokens = set()
                        for arc in query_transition_idx_path:
                            token = arc.ilabel
                            accept_tokens.add(token)
                    else:
                        accept_tokens = set()

                    # Step 2: Compute weights
                    _, query_transition_idx_path = get_next_states(
                        query_automata, _state_tokens, return_arcs=True,
                        filter_nonfinal_transition_states=False,
                        filter_nonfinal_terminal_states=False,
                    )
                    logger.debug("Unbiased reweighing for state {}".format(
                        _state_tokens))
                    logger.debug("Accepted probs: {}".format(
                        np.where(accepted_probs)))

                    _failed_accept_tokens = []
                    _success_accept_tokens = {}
                    for arc in query_transition_idx_path:
                        token = arc.ilabel
                        weight = float(arc.weight)
                        if token in accept_tokens:
                            token = self._decode_automata_token(token)
                            # NOTE(mkuchnik): Empty state is always acceptor,
                            # so we always take the weight
                            # Copy weight over from query automata
                            _success_accept_tokens[token] = weight
                        else:
                            token = self._decode_automata_token(token)
                            if set_accepted_probs:
                                _failed_accept_tokens.append(token)

                    if _success_accept_tokens and not _failed_accept_tokens:
                        # Clear and set
                        transition_probabilities *= 0.0
                        for token, weight in _success_accept_tokens.items():
                            transition_probabilities[token] = weight
                        # NOTE(mkuchnik):
                        # experimental_avoid_not_accepted_probability does what
                        # we do here anyway

                    if _failed_accept_tokens:
                        logger.debug(
                            "Next tokens {} not in accepted set from"
                            " state {}"
                            .format(_failed_accept_tokens, _state_tokens)
                        )
                        if _success_accept_tokens:
                            logger.debug(
                                "Tried to set tokens to accept weight {}"
                                .format(_success_accept_tokens)
                            )
                        logger.debug("Accept tokens: {}".format(
                            accept_tokens))
                    elif _success_accept_tokens:
                        logger.debug(
                            "Set tokens to accept weight {}".format(
                                _success_accept_tokens)
                        )

                else:
                    transition_probabilities[accepted_probs] = 1.0
                    if query.experimental_avoid_not_accepted_probability:
                        if np.any(accepted_probs):
                            transition_probabilities[~accepted_probs] = 0.0
            else:
                # NOTE(mkuchnik): Because we are likely not in top-k,
                # the current log_probability is nan. Rather, just copy.
                transition_probabilities[accepted_probs] = \
                    original_transition_probabilites[accepted_probs]
                if query.experimental_avoid_not_accepted_probability:
                    if np.any(accepted_probs):
                        transition_probabilities[~accepted_probs] = 0.0

            transition_probabilities /= torch.sum(
                transition_probabilities)

            # NOTE(mkuchnik): We use standard probabilities
            transition_probabilities = transition_probabilities.cpu().numpy()
            return transition_probabilities

        def automata_decode_function(token):
            return self._decode_automata_token(token)

        all_viable_paths = relm.automata.random_sampling_from_automata(
            query_automata, expand_neighbor_costs_fn,
            return_edges_visited=True,
            max_sequence_length=max_sequence_length,
            return_costs=True,
            automata_decode_function=automata_decode_function,
        )
        # TODO(mkuchnik): Investigate prefetching
        all_edges_visited = map(lambda x: (x[1], x[2]), all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited


def get_next_states(accept_automata, state_tokens, return_arcs=False,
                    filter_nonfinal_terminal_states=False,
                    filter_nonfinal_transition_states=False,
                    ):
    """Find the next states given the automata."""
    transition_idx_path = relm.automata.automata_next_states(
        accept_automata,
        state_tokens,
        return_edges_visited=True,
        return_arcs=return_arcs,
        filter_nonfinal_terminal_states=filter_nonfinal_terminal_states,
        filter_nonfinal_transition_states=filter_nonfinal_transition_states,
    )
    try:
        states, transition_idx_path = zip(
            *transition_idx_path)
    except ValueError:
        states = []
        transition_idx_path = []
    return states, transition_idx_path
