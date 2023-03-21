"""Tests for relm graph optimizations."""

import unittest

import pywrapfst as fst
from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.automata
import relm.regex_graph_optimizations
from relm.regex_token_remapper import OffsetTokenRemapper


class TestRegexGraphOptimization(unittest.TestCase):
    """Test that regex graph optimizations work."""

    def test_simplify(self):
        """Test that simplify works."""
        regex = "the"
        full_vocab = {"the": 1169, "t": 1, "h": 2, "e": 3, "th": 4}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata, full_vocab)
        )
        expected_automatas = []
        for char_tokens in [[1169], [1, 2, 3], [4, 3]]:
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)

        regex = "the"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata, full_vocab)
        )
        expected_automatas = []
        representations = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        for char_tokens in representations:
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)

        regex = "he"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata, full_vocab)
        )
        expected_automatas = []
        representations = [[258], [71, 68]]
        for char_tokens in representations:
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)

        token_remapper = OffsetTokenRemapper(offset=60000)
        regex = "the"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        # NOTE(mkuchnik): We assume automata encoded
        char_tokens = list(map(lambda x: token_remapper.encode(full_vocab[x]),
                               regex))
        inverted_full_vocab = {token_remapper.encode(v): k
                               for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata, full_vocab, token_remapper=token_remapper)
        )
        expected_automatas = []
        representations = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        for char_tokens in representations:
            # NOTE(mkuchnik): We assume all representations encoded
            char_tokens = [token_remapper.encode(c) for c in char_tokens]
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)

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

    def test_simplify_composition_equivalence(self):
        """Test that simplfy is equivalent to composition."""
        regex = "the"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata.copy(), full_vocab)
        )
        compose_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols_openfst(
                automata.copy(), full_vocab)
        )
        # NOTE(mkuchnik): Compose adds epsilon to vocab
        inverted_full_vocab = {t: c for (c, t) in full_vocab.items()}
        inverted_full_vocab[0] = "ε"
        simplify_automata = relm.automata.attach_symbol_table(
            simplify_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, compose_automata)
        self.assertTrue(are_equivalent)

        regex = "the"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols_openfst(
                automata, full_vocab)
        )
        expected_automatas = []
        representations = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        for char_tokens in representations:
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        inverted_full_vocab = {t: c for (c, t) in full_vocab.items()}
        # Add epsilon
        inverted_full_vocab[0] = "ε"
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)

        regex = "he"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols(
                automata.copy(), full_vocab)
        )
        compose_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols_openfst(
                automata.copy(), full_vocab)
        )
        # NOTE(mkuchnik): Compose adds epsilon to vocab
        inverted_full_vocab = {t: c for (c, t) in full_vocab.items()}
        inverted_full_vocab[0] = "ε"
        simplify_automata = relm.automata.attach_symbol_table(
            simplify_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, compose_automata)
        self.assertTrue(are_equivalent)

        token_remapper = OffsetTokenRemapper(offset=60000)
        regex = "the"
        full_vocab = {"the": 1169, "t": 83, "h": 71, "e": 68, "th": 400,
                      "he": 258}
        # NOTE(mkuchnik): We assume automata encoded
        char_tokens = list(map(lambda x: token_remapper.encode(full_vocab[x]),
                               regex))
        inverted_full_vocab = {token_remapper.encode(v): k
                               for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        simplify_automata = (
            relm.regex_graph_optimizations.simplify_automata_symbols_openfst(
                automata, full_vocab, token_remapper=token_remapper)
        )
        expected_automatas = []
        representations = [[1169], [83, 258], [400, 68], [83, 71, 68]]
        for char_tokens in representations:
            # NOTE(mkuchnik): We assume all representations encoded
            char_tokens = [token_remapper.encode(c) for c in char_tokens]
            _automata = relm.automata.automata_from_token_list(char_tokens)
            expected_automatas.append(_automata)
        expected_automata = relm.automata.union_automatas(expected_automatas)
        # Add epsilon
        inverted_full_vocab[0] = "ε"
        expected_automata = relm.automata.attach_symbol_table(
            expected_automata, inverted_full_vocab)
        are_equivalent = fst.equivalent(
            simplify_automata, expected_automata)
        self.assertTrue(are_equivalent)
