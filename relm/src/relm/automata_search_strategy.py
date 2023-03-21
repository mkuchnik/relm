"""Search strategies for automata searching."""
import itertools
import pprint
from abc import abstractmethod
from typing import List

import torch
import torch.profiler

import relm.regex_parser_strategy
import relm.regex_token_remapper
import relm.relm_logging
import relm.system_util
from relm.abstract_search_strategy import QueryResult, SearchStrategy
from relm.relm_search_query import RegexBackendType, SearchQuery

logger = relm.relm_logging.get_logger()

# NOTE(mkuchnik): Just to force boundary errors on bugs, we map to over 56k
DEFAULT_REMAPPER = relm.regex_token_remapper.OffsetTokenRemapper(
    offset=60000)


class AutomataSearchStrategy(SearchStrategy):
    """Search a test_relm using automata methods."""

    def __init__(self, test_relm: relm.model_wrapper.TestableModel):
        """Take a test_relm object."""
        assert isinstance(test_relm, relm.model_wrapper.TestableModel)
        self.test_relm = test_relm
        self.token_remapper = DEFAULT_REMAPPER

    def _build_query_automata_simple_parsing(self, query: SearchQuery,
                                             use_accept_string: bool):
        if use_accept_string:
            logger.info("Parsing accept string: '{}'".format(query.accept_str))
        else:
            logger.info("Parsing query string: '{}'".format(query.query_str))
        if query.experimental_advanced_parsing_static_minimize:
            raise NotImplementedError(
                "Static Minimize not implemented for simple parser")
        query_str = (query.query_str if not use_accept_string else
                     query.accept_str)
        parser = relm.regex_parser_strategy.SimpleRegexAutomataParser(
            self.test_relm,
            preprocessors=query.experimental_automata_preprocessors,
            token_remapper=self.token_remapper,
        )
        query_automata = parser.parse(query_str)
        return query_automata

    def _build_query_automata_advanced_parsing(self, query: SearchQuery,
                                               use_accept_string: bool):
        if use_accept_string:
            logger.info("Parsing accept string: '{}'".format(query.accept_str))
        else:
            logger.info("Parsing query string: '{}'".format(query.query_str))
        query_str = (query.query_str if not use_accept_string else
                     query.accept_str)

        static_minimize = query.experimental_advanced_parsing_static_minimize
        if query.experimental_advanced_parsing_static_minimize_prefix_only:
            if not use_accept_string:
                logger.info("Not minimizing query string.")
                static_minimize = False

        if query.experimental_regex_backend == RegexBackendType.PYTHON:
            parser = relm.regex_parser_strategy.PythonRegexAutomataParser(
                test_relm=self.test_relm,
                simplify=query.experimental_advanced_parsing_simplify,
                static_minimize=static_minimize,
                max_n=self._max_n(),
                preprocessors=query.experimental_automata_preprocessors,
                token_remapper=self.token_remapper,
            )
        elif query.experimental_regex_backend == RegexBackendType.RUST:
            parser = relm.regex_parser_strategy.RustRegexAutomataParser(
                test_relm=self.test_relm,
                simplify=query.experimental_advanced_parsing_simplify,
                static_minimize=static_minimize,
                max_n=self._max_n(),
                preprocessors=query.experimental_automata_preprocessors,
                token_remapper=self.token_remapper,
            )
        else:
            raise ValueError("Unknown regex backend: {}".format(
                query.experimental_regex_backend))

        query_automata = parser.parse(query_str)

        return query_automata

    def _build_query_automata(self, query: SearchQuery):
        """Create the query automata.

        This automata represents all the ways the query string can be matched.
        """
        if query.experimental_query_automata is not None:
            logger.warning("Using experimental query automata.")
            return query.experimental_query_automata
        if query.experimental_advanced_parsing:
            auto = self._build_query_automata_advanced_parsing(
                query, use_accept_string=False)
        else:
            auto = self._build_query_automata_simple_parsing(
                query, use_accept_string=False)
        if query.experimental_add_eos_token:
            encoding_generator = relm.text_encodings.TextEncodingsGenerator(
                self.test_relm)
            eos_token = encoding_generator.end_of_text_token()
            eos_token = self._encode_automata_token(eos_token)
            eos_automata = relm.automata.automata_from_token_list(
                [eos_token])
            auto = relm.automata.concat_automatas([auto, eos_automata])

        if query.experimental_truncate_automata:
            max_query_length = query.sequence_length
            if max_query_length is None:
                max_query_length = self._max_n()
            else:
                max_query_length = min(max_query_length, self._max_n())
            truncation_necessary = relm.automata.is_cyclic_automata(auto)
            if not truncation_necessary:
                pattern_length = relm.automata.max_path_distance(auto)
                logger.debug("Detected max query path length of {}".format(
                    pattern_length))
                truncation_necessary = pattern_length > max_query_length
                if not truncation_necessary:
                    logger.info("Automata truncation is not necessary")
            if truncation_necessary:
                logger.warning(
                    "Truncating automata to length {}. This is memory heavy."
                    .format(max_query_length)
                )
                auto = relm.automata.truncate_automata(
                    auto, max_query_length)
                logger.warning("Truncated automata.")

        return auto

    def _build_accept_automata(self, query: SearchQuery):
        """Create the query automata.

        This automata represents all the ways the query string can be matched.
        """
        if query.experimental_accept_automata is not None:
            logger.warning("Using experimental accept automata.")
            return query.experimental_accept_automata
        elif query.accept_str:  # Can't be None or ""
            if query.experimental_advanced_parsing:
                logger.info("Advanced parsing accept automata.")
                auto = self._build_query_automata_advanced_parsing(
                    query, use_accept_string=True)
            else:
                logger.info("Simple parsing accept automata.")
                auto = self._build_query_automata_simple_parsing(
                    query, use_accept_string=True)
            # Add prefixes
            auto = relm.automata.convert_automata_to_prefix_acceptor(
                auto, return_copy=True)
            # TODO(mkuchnik): Consider EOS token
            return auto
        else:
            return None

    def plan(self, query: SearchQuery):
        """Return the query automata used for search."""
        query_automata = self._build_query_automata(query)
        return query_automata

    def plan_accept(self, query: SearchQuery):
        """Return the accept automata used for search."""
        query_automata = self._build_accept_automata(query)
        return query_automata

    def _max_k(self):
        """Return the maximum value of number of top tokens to select.

        This is the breadth of the search.
        """
        return self.test_relm.tokenizer.vocab_size

    def _max_n(self):
        """Return the maximum value of depth of tokens to select."""
        # TODO(mkuchnik): Add logic for query max sequence length
        return self.test_relm.model.config.n_positions

    def _encode_automata_token(self, token: int):
        """Return the equivalent automata token for model token."""
        if self.token_remapper:
            return self.token_remapper.encode(token)
        else:
            return token

    def _encode_automata_tokens(self, tokens: List[int]):
        """Return the equivalent automata tokens for model tokens."""
        if self.token_remapper:
            return list(map(lambda token: self.token_remapper.encode(token),
                            tokens))
        else:
            return tokens

    def _decode_automata_token(self, token: int):
        """Return the equivalent model token for automata token."""
        if self.token_remapper:
            return self.token_remapper.decode(token)
        else:
            return token

    def _decode_automata_tokens(self, tokens: List[int]):
        """Return the equivalent model tokens for automata tokens."""
        if self.token_remapper:
            return list(map(lambda token: self.token_remapper.decode(token),
                            tokens))
        else:
            return tokens

    def _model_predict_transition_probabilities(self, state_tokens,
                                                return_numpy=True,
                                                temperature=None):
        with torch.profiler.record_function("sequential_model_inference"):
            if state_tokens is None:
                state_tokens = [self.test_relm.tokenizer.bos_token_id]
            else:
                # TODO(mkuchnik): Add flag to automatically add bos.
                state_tokens = [self.test_relm.tokenizer.bos_token_id,
                                *state_tokens]
            logger.info("Standard scheduling {}".format(state_tokens))
            # TODO(mkuchnik):
            # Use masking on device to avoid transfer.
            torch_state_tokens = torch.tensor(
                state_tokens, dtype=torch.int64, requires_grad=False).to(
                    self.test_relm.device(),
                    non_blocking=True)
            transition_probabilities = (
                self.test_relm._simple_next_token_query_tokens(
                    torch_state_tokens, return_numpy=return_numpy,
                    temperature=temperature))
            return transition_probabilities[-1]

    def _batch_model_predict_transition_probabilities(
            self, state_tokens_batch,
            return_numpy=True,
            temperature=None):
        """Batch runner.

        Takes a potentially ragged array of tokens and pads them to run on the
        model.
        """
        logger.info("Batch scheduling {}".format(state_tokens_batch))
        # Pre-processing
        with torch.profiler.record_function("batch_model_inference"):
            max_batch_len = 0
            for state_tokens in state_tokens_batch:
                if state_tokens:
                    max_batch_len = max(max_batch_len, len(state_tokens))
            # Add one for BOS
            max_batch_len += 1

            _state_tokens_batch = list(state_tokens_batch)
            attention_masks = []
            for i in range(len(state_tokens_batch)):
                state_tokens = state_tokens_batch[i]
                if state_tokens:
                    state_tokens_size = len(state_tokens) + 1
                else:
                    state_tokens_size = 1
                padding_amount = max_batch_len - state_tokens_size
                # Left padding
                # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
                padding = [self.test_relm.tokenizer.bos_token_id
                           for _ in range(padding_amount)]
                if state_tokens is None:
                    state_tokens = [*padding,
                                    self.test_relm.tokenizer.bos_token_id,
                                    ]
                else:
                    # TODO(mkuchnik): Add flag to automatically add bos.
                    state_tokens = [*padding,
                                    self.test_relm.tokenizer.bos_token_id,
                                    *state_tokens,
                                    ]
                _state_tokens_batch[i] = state_tokens
                attention_mask = torch.ones(len(state_tokens), dtype=int)
                attention_mask[:padding_amount] = 0
                assert attention_mask.shape == (max_batch_len,), \
                    "Expected size {} got {}".format((max_batch_len,),
                                                     attention_mask.shape)
                attention_masks.append(attention_mask)

            input_ids = torch.tensor(
                _state_tokens_batch,
                dtype=torch.int64,
                requires_grad=False)
            input_ids = input_ids.to(
                    self.test_relm.device(),
                    non_blocking=True)

            attention_mask = torch.stack(attention_masks)
            attention_mask = attention_mask.to(
                    self.test_relm.device(),
                    non_blocking=True)

            logger.debug(
                "Batch scheduling with {} using inputs {}"
                " and attention {}".format(
                    state_tokens_batch, input_ids.shape, attention_mask.shape))

            # TODO(mkuchnik): Pass attention mask for non-gpt models

            transition_probabilities = (
                self.test_relm._simple_next_token_query_tokens(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_numpy=return_numpy,
                    temperature=temperature))

            # Assume batch * num_inputs * vocab_size
            # TODO(mkuchnik): Add some caching
            return transition_probabilities[:, -1, :]

    def _inspect_query_automata(self, query_automata, fast_start=False,
                                log_tag=None):
        """Debug information for query automata."""
        is_cyclic_query = relm.automata.is_cyclic_automata(query_automata)
        if not log_tag:
            log_tag = ""
        logger.info("{}Query is cyclic: {}".format(log_tag, is_cyclic_query))
        # NOTE(mkuchnik): For cyclic queries, just generate strings that
        # the model has slots for. This actually holds for all queries.
        # counts_by_length is the array from 1 to max_n that represents final
        # states in the automata when traversing it
        unique_query_strings, counts_by_length = (
            relm.automata.string_cardinality_sum_for_automata(
                query_automata,
                max_length=self._max_n(),
                return_counts=True
            )
        )
        logger.info("{}Query cardinality: {} (~10^{})".format(
            log_tag,
            unique_query_strings,
            len(str(unique_query_strings)) - 1))
        if not unique_query_strings:
            # Nothing to do - empty
            return unique_query_strings, counts_by_length
        if not fast_start:
            sample_iter = self._sample_matching_query_strings(query_automata,
                                                              return_str=True)
            sample_iter = itertools.islice(sample_iter, 10)
            samples = list(sample_iter)
            logger.info("{}Example matching strings: {}".format(
                log_tag, pprint.pformat(samples)))
        return unique_query_strings, counts_by_length

    def _sample_matching_query_strings(self, query_automata, return_str=False):
        """Sample strings that can match on the automata."""
        all_viable_paths = relm.automata.sample_from_automata(
            query_automata,
            max_length=self._max_n(),
            return_edges_visited=True)
        all_edges_visited = map(lambda x: self._decode_automata_tokens(x[1]),
                                all_viable_paths)
        if return_str:
            all_edges_visited = map(
                lambda x: (x, self.test_relm.tokens_to_words(x)),
                all_edges_visited
            )
        return all_edges_visited

    def _bfs_memory_space(self, query_cardinality) -> int:
        """Return the max memory needed by BFS.

        BFS uses O(|V|) space, where |V| is the vertex set size. Alternatively,
        BFS uses O(b^(d+1)) space, where b is braching factor and d is depth.
        In other words, every node of the graph may be expanded.

        As each vertex
        is a path, we know the vertex can only contain about sizeof(int) * n
        strings inside of itself, where n is the max path length.
        """
        # Assume int for state size
        size_of_vertex = self._max_n() * 4
        return size_of_vertex * query_cardinality

    def _available_memory(self) -> int:
        """Return the amount of free system memory."""
        return relm.system_util.available_memory()

    def _build_filter(self, filter_str: str, regex_backend: RegexBackendType,
                      add_eos_token: bool, add_bos_token: bool):
        """Return a automata representing the filter string."""
        # NOTE(mkuchnik): For simplicity, we don't minimize.
        # TODO(mkuchnik): Add preprocessors
        if regex_backend == RegexBackendType.PYTHON:
            parser = relm.regex_parser_strategy.PythonRegexAutomataParser(
                test_relm=self.test_relm,
                simplify=True,
                static_minimize=False,
                max_n=self._max_n(),
                token_remapper=self.token_remapper,
            )
        elif regex_backend == RegexBackendType.RUST:
            parser = relm.regex_parser_strategy.RustRegexAutomataParser(
                test_relm=self.test_relm,
                simplify=True,
                static_minimize=False,
                max_n=self._max_n(),
                token_remapper=self.token_remapper,
            )
        else:
            raise ValueError("Unknown regex backend: {}".format(
                regex_backend))

        logger.debug("Starting filter parse")
        filter_automata = parser.parse(filter_str)
        logger.debug("Finished filter parse")

        if add_bos_token:
            encoding_generator = relm.text_encodings.TextEncodingsGenerator(
                self.test_relm)
            bos_token = encoding_generator.beginning_of_text_token()
            bos_token = self._encode_automata_token(bos_token)
            bos_automata = relm.automata.automata_from_token_list(
                [bos_token])
            filter_automata = relm.automata.concat_automatas(
                [bos_automata, filter_automata])

        if add_eos_token:
            encoding_generator = relm.text_encodings.TextEncodingsGenerator(
                self.test_relm)
            eos_token = encoding_generator.end_of_text_token()
            eos_token = self._encode_automata_token(eos_token)
            eos_automata = relm.automata.automata_from_token_list(
                [eos_token])
            filter_automata = relm.automata.concat_automatas(
                [filter_automata, eos_automata])

        debug = False
        if debug:
            sample_iter = self._sample_matching_query_strings(filter_automata,
                                                              return_str=True)
            sample_iter = itertools.islice(sample_iter, 10)
            samples = list(sample_iter)
            log_tag = "[Filter]"
            logger.info("{}Example matching strings: {}".format(
                log_tag, pprint.pformat(samples)))

        def automata_filter_fn(toks) -> bool:
            """Return True if allowed."""
            # Match encoding
            toks = self._encode_automata_tokens(toks)

            linear_automata = relm.automata.automata_from_token_list(toks)
            symbol_table = dict(filter_automata.output_symbols())
            linear_automata = relm.automata.attach_symbol_table(
                linear_automata, symbol_table)
            import pywrapfst as fst
            out = fst.compose(linear_automata, filter_automata)
            does_match = out.num_states() > 0
            return not does_match

        return automata_filter_fn

    def search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        search_iter = self._search(query)

        if query.experimental_filter_str:
            # TODO(mkuchnik): We assume bos is not visible
            logger.info("Building filter {}".format(
                query.experimental_filter_str))
            add_bos_token = False
            # TODO(mkuchik): Predicate pushdown would be more efficient
            query_filter = self._build_filter(query.experimental_filter_str,
                                              query.experimental_regex_backend,
                                              query.experimental_add_eos_token,
                                              add_bos_token,
                                              )
            search_iter = filter(lambda x: query_filter(x[0]), search_iter)
            logger.info("Finished filter {}".format(
                query.experimental_filter_str))

        if relm.facade._is_directed_query(query):
            # TODO(mkuchnik): We probably want this logic in directed query
            simple_results = (query.experimental_advanced_results is None or
                              not query.experimental_advanced_results)
            if simple_results:
                # Remove cost
                search_iter = map(lambda x: tuple(x[0]), search_iter)
            else:
                def package_result(x):
                    return QueryResult(tokens=x[0],
                                       score=x[1])
                search_iter = map(package_result, search_iter)

        logger.info("Starting sampling iterator.")
        ret = relm.util.sample_iterator(search_iter,
                                        query.num_samples,
                                        wrap_pbar=query.progress_bar)

        def log_entry(x):
            logger.info("Yielded sample: {}".format(x))
            return x

        ret = map(log_entry, ret)
        if query.num_samples:
            ret = list(ret)
            logger.debug("Returned {}".format(ret))
        else:
            # TODO(mkuchnik): Similarly wrap with pbar
            logger.info("Yielding iterator")
        return ret

    @abstractmethod
    def _search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        raise NotImplementedError("Search is not implemented for {}"
                                  .format(self))
