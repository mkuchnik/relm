"""Classes for parsing regexes into automata."""

import itertools
import pprint
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
import pywrapfst as fst

import relm.regex_backend.python_regex_backend as python_regex_backend
import relm.regex_backend.rust_regex_backend as rust_regex_backend
import relm.regex_graph_optimizations
import relm.relm_logging
import relm.text_encodings
from relm.regex_token_preprocessor import AutomataPreprocessor
from relm.regex_token_remapper import RegexTokenRemapper

logger = relm.relm_logging.get_logger()

automata_type = fst.VectorFst


class RegexAutomataParser(ABC):
    """Define a class to turn regex strings into automata."""

    @abstractmethod
    def parse(self, regex: str) -> automata_type:
        """Convert a regex string into a automata."""
        raise NotImplementedError("Parse is not implemented.")


class SimpleRegexAutomataParser(RegexAutomataParser):
    """A parser that heuristically parses regex."""

    def __init__(self,
                 test_relm,
                 preprocessors:
                 Optional[Iterable[AutomataPreprocessor]] = None,
                 token_remapper: Optional[RegexTokenRemapper] = None):
        """Initialize the simple regex parser with a test relm."""
        self.test_relm = test_relm
        if not preprocessors:
            preprocessors = []
        self.preprocessors = preprocessors
        self.token_remapper = token_remapper

    def parse(self, regex: str) -> automata_type:
        """Convert a regex string into a automata."""
        logger.info("Starting Simple regex parse.")
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        query_str = regex
        if "*" in query_str or "\\" in query_str:
            logger.warning("Found potential special characters in query.")
        disjunction_query_str = query_str.split("|")
        automatas = []
        for s in disjunction_query_str:
            query_tokens_rep = (
                encoding_generator
                .generate_all_equivalent_substrings_for_sentence(
                    s, fast=True))
            if self.token_remapper:
                # List of sets of tuples
                def remap_tokens(tokens_rep):
                    remapped_tokens_rep = []
                    for s in query_tokens_rep:
                        word_set = set()
                        for t in s:
                            tup = []
                            for e in t:
                                ee = self.token_remapper.encode(e)
                                tup.append(ee)
                            tup = tuple(tup)
                            word_set.add(tup)
                        remapped_tokens_rep.append(word_set)
                    return remapped_tokens_rep

                query_tokens_rep = remap_tokens(query_tokens_rep)

            logger.info("Query '{}' transformed to tokens '{}'".format(
                s, query_tokens_rep))
            automata = \
                (relm
                 .automata
                 .automata_concatenated_union_from_list_of_list_of_token_list(
                     query_tokens_rep)
                 )
            automatas.append(automata)
        query_automata = relm.automata.union_automatas(automatas)
        union_automatas_kwargs = {
            "minimize": False,
            "determinize": False,
        }
        for p in self.preprocessors:
            logger.info("Running preprocessor: {}".format(p))
            query_automata = p.transform(
                query_automata,
                self.token_remapper,
                union_automatas_kwargs=union_automatas_kwargs)
        if self.preprocessors:
            logger.info("Re-finalizing automata.")
            query_automata = relm.automata.finalize_automata(query_automata)
        symbol_table = self._symbol_table()
        logger.info("Attaching symbol table with {} entries".format(
            len(symbol_table)))
        relm.automata.attach_symbol_table(query_automata, symbol_table)
        query_automata = relm.automata.finalize_automata(query_automata)
        return query_automata

    def _symbol_table(self):
        # TODO(mkuchnik): Remove expensive copy
        if self.token_remapper:
            symbol_table = {self.token_remapper.encode(v): k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        else:
            symbol_table = {v: k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        return symbol_table


def _simplify_automata(test_relm,
                       automata: automata_type,
                       simplify: bool,
                       static_minimize: bool,
                       max_n: int,
                       token_remapper: Optional[RegexTokenRemapper] = None):
    """Convert an automata such that full vocabulary is used."""
    encoding_generator = relm.text_encodings.TextEncodingsGenerator(
        test_relm)

    # Int to Char
    input_map = dict(automata.input_symbols())
    if 0 in input_map:
        # Remove epsilon
        del input_map[0]
    # Query only unicode characters that appear in automata symbols
    # Unicode map is a map from characters to tokens
    used_characters = list(input_map.values())
    logger.debug("Used characters: {}".format(used_characters))

    unicode_map = {char: tuple(encoding_generator.bpe_encoding(char, False))
                   for char in used_characters}
    logger.debug("Unicode map num elements: {}".format(len(unicode_map)))

    def simplify_scalar_tup(x):
        """Unpack trivial tuples."""
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x

    # Map Char -> tup
    inverted_basic_char_map = {}
    # Now we need to iterate over the symbols in the input map
    for symbol in input_map.values():
        # First, look up what the tokens are in unicode.
        # The tokens are a tuple because unicode is 1 or more chars
        tokens = unicode_map[symbol]
        assert len(tokens) >= 1, "{}".format(tokens)
        if token_remapper:
            # The tokens are in standard space. Add encoding.
            tokens = tuple(map(lambda tok: token_remapper.encode(tok), tokens))
            for tok in tokens:
                assert tok < (60000 + 56257), "{}".format(tok)
        else:
            for tok in tokens:
                assert tok < 56257, "{}".format(tok)
        # Use a scalar if possible, else use the tuple
        inverted_basic_char_map[symbol] = simplify_scalar_tup(tokens)

    logger.debug("Inverted char map: {}".format(
        pprint.pformat(inverted_basic_char_map)))

    char_toks_mapping = {
        " ": encoding_generator.space_token(),
        "\n": encoding_generator.newline_token(),
        "\t": encoding_generator.tab_token(),
    }

    if token_remapper:
        def remap_f(x):
            return token_remapper.encode(x)
    else:
        def remap_f(x):
            return x

    for src_char, dst_toks in char_toks_mapping.items():
        inverted_basic_char_map[src_char] = remap_f(dst_toks)

    new_input_map = dict()  # Map from int -> str for symbols
    remap_input_map = dict()  # Map from int -> tup[int] for expanding graph
    inverted_basic_char_lookup_errors = list()
    # Loop over original input map's string set
    for old_token, c in input_map.items():
        if token_remapper:
            # NOTE(mkuchnik): Epsilon (0) may be added
            assert old_token == 0 or old_token >= 60000, "{}".format(old_token)
            assert old_token <= (60000 + 56257), "{}".format(old_token)
        else:
            assert old_token >= 0, "{}".format(old_token)
            assert old_token <= 56257, "{}".format(old_token)
        assert isinstance(c, str), \
            "Expected str got {}".format(type(c))
        try:
            char_tup = inverted_basic_char_map[c]
        except KeyError:
            inverted_basic_char_lookup_errors.append(c)
            continue
        if isinstance(char_tup, tuple):
            # We can't encode tuples, so split them into their tokens
            char_tup = tuple(list(map(int, char_tup)))
            for x in char_tup:
                new_input_map[int(x)] = f"<{x}>"
        else:
            assert isinstance(char_tup, (int, np.int64)), \
                "Expected int got {}".format(type(char_tup))
            char_tup = int(char_tup)
            # Should be a scalar int. Int -> str
            new_input_map[char_tup] = c
        # Preserve new tuple for remapper
        remap_input_map[old_token] = char_tup

    if inverted_basic_char_lookup_errors:
        logger.debug("Inverted Basic Char Map Key Errors: {}".format(
            pprint.pformat(inverted_basic_char_lookup_errors)))
    del inverted_basic_char_lookup_errors
    # Sample dict
    logger.debug("Remapping with input map:\n{}\nto new map:\n{}".format(
        pprint.pformat(dict(itertools.islice(input_map.items(), 100))),
        pprint.pformat(dict(itertools.islice(new_input_map.items(), 100)))
    ))
    try:
        automata = relm.automata.remap_edge_values(automata, remap_input_map)
    except Exception as ex:
        logger.error(ex)
        logger.error("Failed to remap automata: {}\nwith remap input map:\n{}"
                     .format(automata, pprint.pformat(remap_input_map)))
        raise ex

    def verify_symbol_table(symbol_table: dict):
        if token_remapper:
            for k in symbol_table:
                assert k >= 60000, "{}".format(k)
                assert k <= (60000 + 56257), "{}".format(k)
        else:
            for k in symbol_table:
                assert k >= 0, "{}".format(k)
                assert k <= 56257, "{}".format(k)

    verify_symbol_table(new_input_map)

    relm.automata.attach_symbol_table(automata, new_input_map)

    if simplify:
        full_vocab = encoding_generator.vocab()
        # TODO(mkuchnik): Merge with prior map
        special_chars_map = {
            " ": encoding_generator.space_char(),
            "\n": encoding_generator.newline_char(),
            "\t": encoding_generator.tab_char()
        }
        for natural_char, encoded_char in special_chars_map.items():
            full_vocab = {k.replace(encoded_char, natural_char): v
                          for k, v in full_vocab.items()}

        if static_minimize:
            logger.info("Statically minimizing automata.")
            automata = (relm.regex_graph_optimizations
                        .minimize_canonical_automata(
                            automata,
                            test_relm,
                            full_vocab=full_vocab,
                            max_depth=max_n,
                            token_remapper=token_remapper)
                        )
        else:
            logger.info("Statically simplifying automata.")
            automata = (relm.regex_graph_optimizations
                        .simplify_automata_symbols(
                            automata, full_vocab,
                            token_remapper=token_remapper)
                        )
        verify_symbol_table(dict(automata.input_symbols()))
    else:
        logger.info("Not simplifying automata.")
    return automata


class RustRegexAutomataParser(RegexAutomataParser):
    """A parser implemented in Rust."""

    def __init__(self,
                 test_relm,
                 simplify: bool,
                 static_minimize: bool,
                 max_n: int,
                 preprocessors:
                 Optional[Iterable[AutomataPreprocessor]] = None,
                 token_remapper: Optional[RegexTokenRemapper] = None):
        """Initialize the regex parser with a test relm."""
        self.test_relm = test_relm
        self.simplify = simplify
        self.static_minimize = static_minimize
        self.max_n = max_n
        if not preprocessors:
            preprocessors = []
        self.preprocessors = preprocessors
        self.token_remapper = token_remapper

    def _regex_parse_to_automata(self, regex: str) -> automata_type:
        automata = rust_regex_backend.regex_to_automata(
            regex, token_remapper=self.token_remapper)
        return automata

    def parse(self, regex: str) -> automata_type:
        """Convert a regex string into a automata."""
        # TODO(mkuchnik): Add validation pass to check for backslash delimiter
        # errors and other common syntax problems.
        logger.info("Starting Rust regex parse.")
        automata = self._regex_parse_to_automata(regex)
        logger.info("Starting Automata processing.")
        union_automatas_kwargs = {
            "minimize": False,
            "determinize": False,
        }
        for p in self.preprocessors:
            logger.info("Running preprocessor: {}".format(p))
            automata = p.transform(
                automata,
                self.token_remapper,
                union_automatas_kwargs=union_automatas_kwargs)
        if self.preprocessors:
            logger.info("Re-finalizing automata.")
            automata = relm.automata.finalize_automata(automata)
        automata = _simplify_automata(
            test_relm=self.test_relm,
            automata=automata,
            simplify=self.simplify,
            static_minimize=self.static_minimize,
            max_n=self.max_n,
            token_remapper=self.token_remapper,
        )
        # TODO(mkuchnik): Investigate if new symbol table is necessary
        symbol_table = self._symbol_table()
        logger.info("Attaching symbol table with {} entries".format(
            len(symbol_table)))
        relm.automata.attach_symbol_table(automata, symbol_table)
        automata = relm.automata.finalize_automata(automata)
        return automata

    def _symbol_table(self):
        # TODO(mkuchnik): Remove expensive copy
        if self.token_remapper:
            symbol_table = {self.token_remapper.encode(v): k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        else:
            symbol_table = {v: k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        return symbol_table


class PythonRegexAutomataParser(RegexAutomataParser):
    """A parser implemented in Python."""

    def __init__(self,
                 test_relm,
                 simplify: bool,
                 static_minimize: bool,
                 max_n: int,
                 preprocessors:
                 Optional[Iterable[AutomataPreprocessor]] = None,
                 token_remapper: Optional[RegexTokenRemapper] = None):
        """Initialize the regex parser with a test relm."""
        self.test_relm = test_relm
        self.simplify = simplify
        self.static_minimize = static_minimize
        self.max_n = max_n
        if not preprocessors:
            preprocessors = []
        self.preprocessors = preprocessors
        self.token_remapper = token_remapper

    def _regex_parse_to_automata(self, regex: str) -> automata_type:
        it = python_regex_backend.regex_string_emitter(regex)
        logger.debug("Compiling {} to openfst".format(repr(it)))
        automata = it.to_openfst(self.token_remapper)
        return automata

    def parse(self, regex: str) -> automata_type:
        """Convert a regex string into a automata."""
        # TODO(mkuchnik): Add validation pass to check for backslash delimiter
        # errors and other common syntax problems.
        logger.info("Starting Python regex parse.")
        automata = self._regex_parse_to_automata(regex)
        logger.info("Starting Automata processing.")
        union_automatas_kwargs = {
            "minimize": False,
            "determinize": False,
        }
        for p in self.preprocessors:
            logger.info("Running preprocessor: {}".format(p))
            automata = p.transform(
                automata,
                self.token_remapper,
                union_automatas_kwargs=union_automatas_kwargs)
        if self.preprocessors:
            logger.info("Re-finalizing automata.")
            automata = relm.automata.finalize_automata(automata)
        automata = _simplify_automata(
            test_relm=self.test_relm,
            automata=automata,
            simplify=self.simplify,
            static_minimize=self.static_minimize,
            max_n=self.max_n,
            token_remapper=self.token_remapper,
        )
        symbol_table = self._symbol_table()
        logger.info("Attaching symbol table with {} entries".format(
            len(symbol_table)))
        relm.automata.attach_symbol_table(automata, symbol_table)
        automata = relm.automata.finalize_automata(automata)
        return automata

    def _symbol_table(self):
        # TODO(mkuchnik): Remove expensive copy
        if self.token_remapper:
            symbol_table = {self.token_remapper.encode(v): k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        else:
            symbol_table = {v: k for (k, v) in
                            self.test_relm.tokenizer.vocab.items()}
        return symbol_table
