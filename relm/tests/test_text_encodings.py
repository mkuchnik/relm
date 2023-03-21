"""Tests for relm test encodings."""

import itertools
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.model_wrapper
import relm.text_encodings


class TestTextEncodings(unittest.TestCase):
    """Test text encoding generator."""

    def setUp(self):
        """Create a test relm."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        self.test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

    def _test_invertible_helper(self, sentence):
        """Go from string to tokens and back."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        sentence_toks = \
            encoding_generator.string_to_basic_chars_tokens_flat(sentence)
        recovered_sentence = self.test_relm.tokens_to_words(sentence_toks)
        self.assertEqual(sentence, recovered_sentence)

    def test_invertible(self):
        """Test that encodings are invertible."""
        self._test_invertible_helper("The first thing")
        self._test_invertible_helper("A big man ")
        self._test_invertible_helper(" the")

    def test_basic_characters(self):
        """Test vocabulary is a superset of basic characters."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        basic_chars = set(encoding_generator.basic_characters())
        all_vocabulary_words = encoding_generator.all_vocabulary_words()
        for w in all_vocabulary_words:
            self.assertTrue(set(w).issubset(basic_chars))

    def _test_equivalent_word_substrings_helper(self, word: str):
        """Test that all equivalent token lists can be inverted to word."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        list_of_reps = (encoding_generator
                        .generate_all_equivalent_substrings_for_word(word))
        list_of_reps = list(list_of_reps)  # Remove generator
        self.assertGreater(len(list_of_reps), 0)
        for token_list in list_of_reps:
            recovered_word = self.test_relm.tokens_to_words(token_list)
            self.assertEqual(word, recovered_word)

    def _test_equivalent_word_substrings_helper_fast(self, word: str):
        """Test that all equivalent token lists can be inverted to word."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        list_of_reps = (encoding_generator
                        .generate_all_equivalent_substrings_for_word_fast(
                            word))
        list_of_reps = list(list_of_reps)  # Remove generator
        self.assertGreater(len(list_of_reps), 0)
        for token_list in list_of_reps:
            recovered_word = self.test_relm.tokens_to_words(token_list)
            self.assertEqual(word, recovered_word)

    def test_equivalent_word_substrings(self):
        """Test that equivalent word substrings can be generated."""
        self._test_equivalent_word_substrings_helper("The")
        self._test_equivalent_word_substrings_helper("first")
        self._test_equivalent_word_substrings_helper("thing")
        self._test_equivalent_word_substrings_helper("A")
        self._test_equivalent_word_substrings_helper("big")
        self._test_equivalent_word_substrings_helper("man")
        self._test_equivalent_word_substrings_helper("the")
        self._test_equivalent_word_substrings_helper("an")
        # Adding spaces
        self._test_equivalent_word_substrings_helper(" The")
        self._test_equivalent_word_substrings_helper(" The ")
        self._test_equivalent_word_substrings_helper(" first")
        self._test_equivalent_word_substrings_helper(" first ")
        self._test_equivalent_word_substrings_helper(" thing")
        self._test_equivalent_word_substrings_helper(" thing ")
        self._test_equivalent_word_substrings_helper(" an  ")

    def test_equivalent_word_substrings_fast(self):
        """Test that equivalent word substrings can be generated."""
        self._test_equivalent_word_substrings_helper_fast("The")
        self._test_equivalent_word_substrings_helper_fast("first")
        self._test_equivalent_word_substrings_helper_fast("thing")
        self._test_equivalent_word_substrings_helper_fast("A")
        self._test_equivalent_word_substrings_helper_fast("big")
        self._test_equivalent_word_substrings_helper_fast("man")
        self._test_equivalent_word_substrings_helper_fast("the")
        self._test_equivalent_word_substrings_helper_fast("an")
        # Adding spaces
        self._test_equivalent_word_substrings_helper_fast(" The")
        self._test_equivalent_word_substrings_helper_fast(" The ")
        self._test_equivalent_word_substrings_helper_fast(" first")
        self._test_equivalent_word_substrings_helper_fast(" first ")
        self._test_equivalent_word_substrings_helper_fast(" thing")
        self._test_equivalent_word_substrings_helper_fast(" thing ")
        self._test_equivalent_word_substrings_helper_fast(" an  ")

    def _test_partitions_helper(self, word: str):
        """Test that 2^(n-1) partitions of string."""
        all_partitions = list(relm.text_encodings._partitions(word))
        self.assertEqual(len(all_partitions),
                         2**(len(word) - 1) if len(word) else 1)
        for p in all_partitions:
            self.assertEqual("".join(p), word)

    def test_partitions(self):
        """Test that string partitioning function is correct."""
        self._test_partitions_helper("apple")
        self._test_partitions_helper("a")
        self._test_partitions_helper("the")
        self._test_partitions_helper("")

    def _test_equivalent_sentence_substrings_helper(self, sentence: str):
        """Test that all equivalent token lists can be inverted to sentence."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        list_of_reps = (encoding_generator
                        .generate_all_equivalent_substrings_for_sentence(
                            sentence))
        list_of_reps = list(list_of_reps)  # Remove generator
        self.assertGreater(len(list_of_reps), 0)
        words = sentence.split(" ")
        self.assertEqual(len(list_of_reps), len(words))
        sample_sentence_tokens = itertools.product(*list_of_reps)
        for s in sample_sentence_tokens:
            flat_tokens = [x for tup in s for x in tup]
            recovered_sentence = self.test_relm.tokens_to_words(flat_tokens)
            self.assertEqual(recovered_sentence, sentence)

    def _test_equivalent_sentence_substrings_helper_fast(self, sentence: str):
        """Test that all equivalent token lists can be inverted to sentence."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        list_of_reps = (encoding_generator
                        .generate_all_equivalent_substrings_for_sentence(
                            sentence, fast=True))
        list_of_reps = list(list_of_reps)  # Remove generator
        self.assertGreater(len(list_of_reps), 0)
        words = sentence.split(" ")
        self.assertEqual(len(list_of_reps), len(words))
        sample_sentence_tokens = itertools.product(*list_of_reps)
        for s in sample_sentence_tokens:
            flat_tokens = [x for tup in s for x in tup]
            recovered_sentence = self.test_relm.tokens_to_words(flat_tokens)
            self.assertEqual(recovered_sentence, sentence)

    def test_equivalent_sentence_substrings(self):
        """Test that equivalent sentence substrings can be generated."""
        self._test_equivalent_sentence_substrings_helper("The first thing")
        self._test_equivalent_sentence_substrings_helper("The")
        self._test_equivalent_sentence_substrings_helper("first")
        self._test_equivalent_sentence_substrings_helper("thing")
        self._test_equivalent_sentence_substrings_helper("\n")
        self._test_equivalent_sentence_substrings_helper("\t")

    def test_equivalent_sentence_substrings_fast(self):
        """Test that equivalent sentence substrings can be generated."""
        self._test_equivalent_sentence_substrings_helper_fast(
            "The first thing")
        self._test_equivalent_sentence_substrings_helper_fast("The")
        self._test_equivalent_sentence_substrings_helper_fast("first")
        self._test_equivalent_sentence_substrings_helper_fast("thing")
        self._test_equivalent_sentence_substrings_helper_fast("\n")
        self._test_equivalent_sentence_substrings_helper_fast("\t")

    def test_bpe_encodings(self):
        """Test that BPE encoding."""
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            self.test_relm)
        text = "я"
        toks = encoding_generator.bpe_encoding(text, True)
        expected_toks = [40623]
        self.assertEqual(toks, expected_toks)
        toks = encoding_generator.bpe_encoding(text, False)
        expected_toks = [141, 237]
        self.assertEqual(toks, expected_toks)
        text = "The first thing"
        toks = encoding_generator.bpe_encoding(text, True)
        expected_toks = self.test_relm.words_to_tokens(text)[0].tolist()
        self.assertEqual(toks, expected_toks)
