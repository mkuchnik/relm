"""Tests for parser strategies.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import unittest

import pywrapfst as fst
from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.automata
import relm.model_wrapper
import relm.regex_parser_strategy
from relm.regex_token_remapper import OffsetTokenRemapper


class TestRegexParsers(unittest.TestCase):
    """Test the parsers for ReLM."""

    def parser_helper(self, test_relm, parser_fn, token_remapper,
                      parser_kwargs, escape_chars):
        """Run tests on permutations of "the"."""
        # query is "the"
        # 1) non-simplified
        regex = "the"
        parser = parser_fn(
            test_relm,
            simplify=False,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        expected_tokens = [83, 71, 68]
        if token_remapper:
            expected_tokens = list(map(lambda x: token_remapper.encode(x),
                                       expected_tokens))
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = relm.automata.automata_from_token_list(
            expected_tokens)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 2) simplified
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 3) minimized
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=True,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[1169]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)

        # query is "The"
        # 1) non-simplified
        regex = "The"
        parser = parser_fn(
            test_relm,
            simplify=False,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        expected_tokens = [51, 71, 68]
        if token_remapper:
            expected_tokens = list(map(lambda x: token_remapper.encode(x),
                                       expected_tokens))
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = relm.automata.automata_from_token_list(
            expected_tokens)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 2) simplified
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[464], [51, 258], [817, 68], [51, 71, 68]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 3) minimized
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=True,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[464]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)

        # query is "THE"
        # 1) non-simplified
        regex = "THE"
        parser = parser_fn(
            test_relm,
            simplify=False,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        expected_tokens = [51, 39, 36]
        if token_remapper:
            expected_tokens = list(map(lambda x: token_remapper.encode(x),
                                       expected_tokens))
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = relm.automata.automata_from_token_list(
            expected_tokens)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 2) simplified
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=False,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[10970], [4221, 36], [51, 13909], [51, 39, 36]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)
        # 3) minimized
        parser = parser_fn(
            test_relm,
            simplify=True,
            static_minimize=True,
            max_n=1024,
            **parser_kwargs,
        )
        automata = parser.parse(regex)
        symbol_table = dict(automata.input_symbols())
        if token_remapper:
            self.assertTrue(all(token_remapper.encode(ord(x))
                                in symbol_table for x in regex))
        else:
            self.assertTrue(all(ord(x) in symbol_table for x in regex))
        used_symbols = relm.automata.used_keys_set(automata)
        representations = [[10970]]
        if token_remapper:
            representations = list(map(lambda x: [token_remapper.encode(xx) for
                                                  xx in x],
                                   representations))
        expected_tokens = [xx for x in representations for xx in x]
        self.assertEqual(used_symbols, set(expected_tokens))
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                representations)
        relm.automata.attach_symbol_table(expected_automata, symbol_table)
        are_equivalent = fst.equivalent(
            automata, expected_automata)
        self.assertTrue(are_equivalent)

        # Simple chars
        for char in ["\n", "\t", "\n", " ", "t", "h", "E", "   "]:
            if escape_chars:
                regex = char.encode("unicode_escape").decode("utf-8")
            else:
                regex = char
            parser = parser_fn(
                test_relm,
                simplify=False,
                static_minimize=False,
                max_n=1024,
                **parser_kwargs,
            )
            automata = parser.parse(regex)
            symbol_table = dict(automata.input_symbols())
            if token_remapper:
                self.assertTrue(all(token_remapper.encode(ord(x))
                                    in symbol_table for x in regex))
            else:
                self.assertTrue(all(ord(x) in symbol_table for x in regex))
            used_symbols = relm.automata.used_keys_set(automata)
            expected_tokens = [int(c) for c in
                               test_relm.words_to_tokens(char)[0]]
            if token_remapper:
                expected_tokens = list(map(lambda x: token_remapper.encode(x),
                                           expected_tokens))
            self.assertEqual(used_symbols, set(expected_tokens))
            expected_automata = relm.automata.automata_from_token_list(
                expected_tokens)
            relm.automata.attach_symbol_table(expected_automata, symbol_table)
            are_equivalent = fst.equivalent(
                automata, expected_automata)
            self.assertTrue(are_equivalent)

        # Minimized words
        for char in [" the", " the ", " the A apple a", "\n\n"]:
            if escape_chars:
                regex = char.encode("unicode_escape").decode("utf-8")
            else:
                regex = char
            parser = parser_fn(
                test_relm,
                simplify=True,
                static_minimize=True,
                max_n=1024,
                **parser_kwargs,
            )
            automata = parser.parse(regex)
            symbol_table = dict(automata.input_symbols())
            if token_remapper:
                self.assertTrue(all(token_remapper.encode(ord(x))
                                    in symbol_table for x in regex))
            else:
                self.assertTrue(all(ord(x) in symbol_table for x in regex))
            used_symbols = relm.automata.used_keys_set(automata)
            expected_tokens = [int(c) for c in
                               test_relm.words_to_tokens(char)[0]]
            if token_remapper:
                expected_tokens = list(map(lambda x: token_remapper.encode(x),
                                           expected_tokens))
            self.assertEqual(used_symbols, set(expected_tokens))
            expected_automata = relm.automata.automata_from_token_list(
                expected_tokens)
            relm.automata.attach_symbol_table(expected_automata, symbol_table)
            are_equivalent = fst.equivalent(
                automata, expected_automata)
            self.assertTrue(are_equivalent)

        unicode_char_map = {
            "π": {(139, 222), (46582,), },
            "θ": {(138, 116), },
            "э": {(141, 235), },
            "’": {(158, 222, 247), (447, 247), },
            "–": {(158, 222, 241), (447, 241), (1906,), },
            "®": {(126, 106), (7461,), },
            "©": {(126, 102), (16224,), },
            "¶": {(126, 114), (26604,), },
            "·": {(126, 115), (9129,), },
            "´": {(126, 112), (18265,), },
            "‘": {(158, 222, 246), (447, 246), },
            "”": {(158, 222, 251), (447, 251), },
            "я": {(141, 237), (40623,), },
            "東": {(162, 251, 109), (30266, 109), },
        }

        # Unicode
        for char, token_repr in unicode_char_map.items():
            if token_remapper:
                token_repr = {tuple(token_remapper.encode(xx) for xx in x)
                              for x in token_repr}
            regex = char
            parser = parser_fn(
                test_relm,
                simplify=True,
                static_minimize=False,
                max_n=1024,
                **parser_kwargs,
            )
            automata = parser.parse(regex)
            symbol_table = dict(automata.input_symbols())
            if token_remapper:
                self.assertTrue(all(token_remapper.encode(ord(x))
                                    in symbol_table for x in regex))
            else:
                self.assertTrue(all(ord(x) in symbol_table for x in regex))
            used_symbols = relm.automata.used_keys_set(automata)
            expected_symbols = {t for tr in token_repr for t in tr}
            self.assertEqual(used_symbols,
                             expected_symbols)
            expected_automata = relm.automata.automata_from_token_list(
                expected_tokens)
            list_of_list_of_tokens = token_repr
            expected_automata = \
                relm.automata.automata_union_from_list_of_token_list(
                    list_of_list_of_tokens,
                    determinize=True,
                    minimize=True,
                    rm_epsilon=True,
                    verify=True)
            relm.automata.attach_symbol_table(expected_automata, symbol_table)
            are_equivalent = fst.equivalent(
                automata, expected_automata)
            self.assertTrue(are_equivalent)

    def test_python_backend(self):
        """Test that the Python backend works."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

        parser_fn = relm.regex_parser_strategy.PythonRegexAutomataParser
        token_remapper = None
        parser_kwargs = {}
        escape_chars = True
        self.parser_helper(test_relm, parser_fn, token_remapper, parser_kwargs,
                           escape_chars)

        parser_fn = relm.regex_parser_strategy.PythonRegexAutomataParser
        token_remapper = OffsetTokenRemapper(offset=60000)
        parser_kwargs = {"token_remapper": token_remapper}
        self.parser_helper(test_relm, parser_fn, token_remapper, parser_kwargs,
                           escape_chars)

    def test_rust_backend(self):
        """Test that the Rust backend works."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

        parser_fn = relm.regex_parser_strategy.RustRegexAutomataParser
        token_remapper = None
        parser_kwargs = {}
        escape_chars = True
        self.parser_helper(test_relm, parser_fn, token_remapper, parser_kwargs,
                           escape_chars)

        parser_fn = relm.regex_parser_strategy.RustRegexAutomataParser
        token_remapper = OffsetTokenRemapper(offset=60000)
        parser_kwargs = {"token_remapper": token_remapper}
        self.parser_helper(test_relm, parser_fn, token_remapper, parser_kwargs,
                           escape_chars)
