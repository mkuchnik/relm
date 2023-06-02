"""Tests for relm graph optimizations.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import unittest

import pywrapfst as fst
from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.automata
import relm.regex_graph_optimizations


class TestRegexGraphOptimizationIntegration(unittest.TestCase):
    """Test that regex graph optimizations work."""

    def test_static_minimize(self):
        """Test that static minimize works."""
        regex = "the"
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        test_relm = relm.model_wrapper.TestableModel(model, tokenizer)
        full_vocab = tokenizer.vocab
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        max_depth = 1024
        simplify_automata = (
            relm.regex_graph_optimizations.minimize_canonical_automata(
                automata, test_relm, max_depth, full_vocab)
        )
        expected_automatas = []
        for char_tokens in [[full_vocab["the"]]]:
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        inverted_full_vocab = {t: c for (c, t) in full_vocab.items()}
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)
