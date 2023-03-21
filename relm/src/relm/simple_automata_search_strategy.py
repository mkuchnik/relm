"""Automata-based searches which do not score paths."""
import itertools
import sys
import time

import numpy as np

import relm.model_wrapper
from relm.automata_search_strategy import AutomataSearchStrategy
from relm.relm_search_query import SearchQuery

BIG_QUERY_THRESHOLD = 10000000000  # 10^10


logger = relm.relm_logging.get_logger()


class SimpleAutomataSearchStrategy(AutomataSearchStrategy):
    """Represents automata-based search strategies that don't score paths."""

    def __init__(self, test_relm: relm.model_wrapper.TestableModel):
        """Take a test_relm object."""
        super().__init__(test_relm)

    def _model_accepts(self, query: SearchQuery, path, model_cache=None,
                       accept_automata=None):
        """Return true if the model accepts the path under query parameters.

        For top-k queries, a model does not accept a path if any of the
        prefixes
        are not accepted. Therefore, we can cache all prefix computations to
        re-use them when testing many paths with common prefixes.

        :param query: The query used for search.
        :param path: The list of tokens to test.
        :param model_cache: An optional cache of past answers (a dict).
        :param accept_automata: An optional automata that automatically accepts
        a string if it matches.
        """
        if accept_automata is not None and relm.automata.apply_fst_accepted(
                path, accept_automata):
            logger.debug("Accepting via automata ({}):"
                         " {}".format(id(accept_automata), path))
            return True
        if query.top_k_sampling is not None:
            top_k_sampling = query.top_k_sampling
            if top_k_sampling > self._max_k():
                logger.info("Top-k sampling set to {}, which is greater than"
                            " the maximum value of {}. Thresholding.".format(
                                top_k_sampling, self._max_k(),
                            ))
                top_k_sampling = self._max_k()
                # TODO(mkuchnik): Consider setting to None, which would bypass
                # queries altogether.
            state_tokens = None
            # TODO(mkuchnik): Test caching logic
            for token in path:
                state_tokens_tup = (tuple(state_tokens) if state_tokens
                                    else tuple())
                next_state_tokens_tup = state_tokens_tup + tuple([token])
                if (model_cache is not None
                        and next_state_tokens_tup in model_cache):
                    # Cache hit
                    if next_state_tokens_tup:
                        # Don't log empty tuple as it's very common.
                        logger.debug("Cache hit: {}".format(
                            next_state_tokens_tup))
                    model_contains_path = model_cache[next_state_tokens_tup]
                else:
                    transition_probabilities = \
                        self._model_predict_transition_probabilities(
                            state_tokens, temperature=query.temperature)
                    # TODO(mkuchnik): Test threshold
                    token_rank = relm.indexing.get_element_rank(
                        transition_probabilities, token, reverse=True)
                    model_contains_path = token_rank <= top_k_sampling
                    if query.experimental_log_k:
                        logger.debug("Token {} has rank k={}".format(
                            next_state_tokens_tup,
                            token_rank))
                    if query.experimental_safe:
                        prob_path_ranks = relm.indexing.path_to_top_k(
                            self.test_relm, next_state_tokens_tup)
                        if prob_path_ranks[-1] != token_rank:
                            raise RuntimeError(
                                "Token rank calculated as {}"
                                ", but found indexing rank {} for path {}"
                                .format(token_rank,
                                        prob_path_ranks[-1],
                                        next_state_tokens_tup)
                            )
                        if (model_contains_path
                                and np.any(prob_path_ranks > top_k_sampling)):
                            raise RuntimeError(
                                "Found top-k violation (k={}) for tokens {}"
                                " with ranks {}".format(top_k_sampling,
                                                        next_state_tokens_tup,
                                                        prob_path_ranks))
                    if model_cache is not None:
                        logger.debug("Cache insert: {}".format(
                            next_state_tokens_tup))
                        model_cache[next_state_tokens_tup] = \
                            model_contains_path
                if not model_contains_path:
                    return False
                state_tokens = [] if state_tokens is None else state_tokens
                state_tokens.append(token)
        return True

    def _path_search_iter(self, query: SearchQuery, paths,
                          accept_automata=None):
        """Search the model using the finite set of paths."""
        if (query.experimental_cache_optimization is None
                or query.experimental_cache_optimization):
            logger.info("Initializing search function and cache.")
            # TODO(mkuchnik): Reduce memory complexity
            model_cache = dict()
        else:
            logger.info("No cache is being used in search function.")
            model_cache = None
        for i, path in enumerate(paths):
            if not i:
                logger.info("First path was yielded.")
            # NOTE(mkuchnik): We convert from automata to model tokens
            path = self._decode_automata_tokens(path)
            if self._model_accepts(query, path, model_cache,
                                   accept_automata=accept_automata):
                logger.debug("Model accepts path: {}".format(path))
                yield path

    def _search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        logger.debug("Searching query: {}".format(query))
        # TODO(mkuchnik): Investigate building both of these in parallel
        automata_start_time = time.perf_counter()
        query_automata = self._build_query_automata(query)
        accept_automata = self._build_accept_automata(query)
        automata_end_time = time.perf_counter()
        logger.info("Compiled automata in {:.2} seconds".format(
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
            first_accepting_length = next(
                (i for i, x in enumerate(counts_by_length) if x),
                None) + 1
            logger.info("First accepting length: {}".format(
                first_accepting_length))
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

        if query.num_samples is not None:
            if ((bfs_memory_complexity > available_memory) or
                    (query_cardinality > BIG_QUERY_THRESHOLD)):
                if query.top_k_sampling is not None:
                    logger.info("Big query. Using IDDFS sampling.")
                    if (query.experimental_IDDFS_optimization
                            is None or
                            query.experimental_IDDFS_optimization):
                        # TODO(mkuchnik): Should this be increased by 1?
                        IDDFS_start_length = first_accepting_length + 1
                        logger.info("IDDFS start length optimization enabled")
                    else:
                        IDDFS_start_length = None
                    all_edges_visited = (
                        self._iterative_deepening_dfs_matching_query_strings(
                            query_automata,
                            start_length=IDDFS_start_length,
                        ))
                    # NOTE(mkuchnik): We truncate the IDDFS after all
                    # possibilities are seen.
                    # However, islice only works on native ints.
                    if query_cardinality <= sys.maxsize:
                        all_edges_visited = itertools.islice(all_edges_visited,
                                                             query_cardinality)
                else:
                    logger.info("Big query without top-k. Using DFS sampling.")
                    all_edges_visited = (
                        self._dfs_matching_query_strings(
                            query_automata))
            else:
                logger.info("Small query. Using BFS sampling.")
                all_edges_visited = self._bfs_matching_query_strings(
                    query_automata)
        else:
            # NOTE(mkuchnik): If query.num_samples is off, we need to enumerate
            # all possible matches. This is probably bad to do randomly.
            logger.info("Result subsampling off. Using BFS sampling.")
            all_edges_visited = self._bfs_matching_query_strings(
                query_automata)

        logger.info("Paths planned.")
        assert not (query.experimental_dijkstra and not
                    query.experimental_greedy_search)
        if query.sequence_length is not None:
            raise NotImplementedError(
                "Sequence length only implemented for dijkstra")
        search_iter = self._path_search_iter(
            query, all_edges_visited, accept_automata=accept_automata)
        return search_iter

    def _bfs_matching_query_strings(self, query_automata, return_str=False):
        """BFS visit strings that can match on the automata."""
        all_viable_paths = relm.automata.BFS_from_automata(
            query_automata, return_edges_visited=True)
        all_edges_visited = map(lambda x: x[1], all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _dfs_matching_query_strings(self, query_automata, return_str=False):
        """DFS visit strings that can match on the automata."""
        all_viable_paths = relm.automata.DFS_from_automata(
            query_automata, return_edges_visited=True)
        all_edges_visited = map(lambda x: x[1], all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _iterative_deepening_dfs_matching_query_strings(
            self, query_automata, return_str=False, start_length=None):
        """IDDFS visit strings that can match on the automata.

        Note: this iterator does not terminate! You have to manually truncate
        it after all possibilities exhausted.
        """
        all_viable_paths = relm.automata.iterative_deepening_DFS_from_automata(
            query_automata,
            return_edges_visited=True,
            start_length=start_length)
        all_edges_visited = map(lambda x: x[1], all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited
