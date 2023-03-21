"""Tests for relm token preprocessors."""

import string
import unittest

import pywrapfst as fst

import relm.automata
import relm.regex_token_preprocessor
import relm.regex_token_remapper
import relm.util


class TestRegexTokenPreprocessor(unittest.TestCase):
    """Test that regex token preprocessors work."""

    def test_levenshtein(self):
        """Test that Levenshtein transformer works."""
        regex = "the"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 3)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        new_automata = preprocessor.transform(automata)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 95)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        full_vocab = {x: ord(x) for x in string.printable}
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz")), new_automata)
        )
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == "the":
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

    def test_levenshtein_big(self):
        """Test that Levenshtein transformer works for bigger edits."""
        regex = "the"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 3)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            num_edits=2
        )
        new_automata = preprocessor.transform(automata)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 95)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        full_vocab = {x: ord(x) for x in string.printable}
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz")), new_automata)
        )
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == "the":
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 3,
                                "'{}' has distance {}".format(word, distance))

    def test_levenshtein_limited_charset(self):
        """Test that Levenshtein transformer works."""
        regex = "the"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual({chr(x) for x in old_keys}, {"t", "h", "e"})
        charset = "thea"
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            charset
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(automata)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual({chr(x) for x in new_keys}, {"a", "t", "h", "e"})
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        self.assertEqual(len(new_keys), 4)
        full_vocab = {x: ord(x) for x in string.printable}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tae")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thea")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == "the":
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

        regex = "t"
        full_vocab = {x: ord(x) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        automata = relm.automata.finalize_automata(automata)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual({chr(x) for x in old_keys}, {"t"})
        # Mismatch charset
        charset = "ABC"
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            charset
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata)
        new_automata = relm.automata.finalize_automata(new_automata)
        # Swap table to add ABC
        relm.automata.attach_symbol_table(automata,
                                          new_automata.input_symbols())
        full_vocab = {x: ord(x) for x in regex + charset}
        # NOTE(mkuchnik): t and null
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "t")), new_automata)
        )
        # NOTE(mkuchnik): t and insert A
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tA")), new_automata)
        )
        # NOTE(mkuchnik): remove t (not allowed)
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "")), new_automata)
        )
        self.assertFalse(fst.equivalent(new_automata, automata))
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            charset,
            allow_passthrough_deletes=True,
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata)
        new_automata = relm.automata.finalize_automata(new_automata)
        full_vocab = {x: ord(x) for x in regex + charset}
        # NOTE(mkuchnik): t and null
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "t")), new_automata)
        )
        # NOTE(mkuchnik): t and insert A
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tA")), new_automata)
        )
        # NOTE(mkuchnik): remove t (now allowed) and insert A
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "")), new_automata)
        )
        self.assertFalse(fst.equivalent(new_automata, automata))

    def test_levenshtein_remapper(self):
        """Test that Levenshtein transformer works with remappers."""
        regex = "the"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 3)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        new_automata = preprocessor.transform(automata, token_remapper)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 95)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        full_vocab = {x: token_remapper.encode(ord(x))
                      for x in string.printable}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == regex:
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

        regex = "the®"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 4)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer()
        new_automata = preprocessor.transform(automata, token_remapper)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 96)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        vocab_set = set(string.printable).union(set(["®"]))
        full_vocab = {x: token_remapper.encode(ord(x))
                      for x in vocab_set}
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz®")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz®")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == regex:
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

        regex = "the®"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 4)
        preprocessor = \
            relm.regex_token_preprocessor.LevenshteinTransformer(
                allow_passthrough_deletes=True)
        new_automata = preprocessor.transform(automata, token_remapper)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 96)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        vocab_set = set(string.printable).union(set(["®"]))
        full_vocab = {x: token_remapper.encode(ord(x))
                      for x in vocab_set}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz®")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thee")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == regex:
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

        regex = "the®"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 4)
        preprocessor = \
            relm.regex_token_preprocessor.LevenshteinTransformer(
                allow_passthrough_deletes=True,
                allow_passthrough_substitutions=True)
        new_automata = preprocessor.transform(automata, token_remapper)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 96)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        vocab_set = set(string.printable).union(set(["®"]))
        full_vocab = {x: token_remapper.encode(ord(x))
                      for x in vocab_set}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz®")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz®")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thee")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == regex:
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

        regex = "the"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        old_keys = relm.automata.used_keys_set(automata)
        self.assertEqual(len(old_keys), 3)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            allow_deletes=False,
            allow_inserts=False,
            allow_substitutions=False,
        )
        new_automata = preprocessor.transform(automata, token_remapper)
        new_keys = relm.automata.used_keys_set(new_automata)
        self.assertEqual(len(new_keys), 3)
        self.assertFalse(relm.automata.apply_fst_accepted([], new_automata))
        full_vocab = {x: token_remapper.encode(ord(x))
                      for x in string.printable}
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        self.assertTrue(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "the")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "th")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "he")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tce")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "thz")), new_automata)
        )
        self.assertFalse(relm.automata.apply_fst_accepted(
            list(map(lambda x: full_vocab[x], "tcz")), new_automata)
        )
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        _, all_edges_visited = zip(
            *relm.automata.DFS_from_automata(
                new_automata, return_edges_visited=True))
        all_edges_visited = map(
            lambda edges: "".join(inverted_full_vocab[c] for c in edges),
            all_edges_visited
        )
        all_edges_visited = map(
            lambda x: (x, relm.util.levenshtein_distance(x, regex)),
            all_edges_visited,
        )
        all_edges_visited = list(all_edges_visited)
        for word, distance in all_edges_visited:
            if word == regex:
                self.assertEqual(distance, 0,
                                 "'{}' has distance {}".format(word, distance))
            else:
                self.assertLess(distance, 2,
                                "'{}' has distance {}".format(word, distance))

    def test_levenshtein_equivalence(self):
        """Test that the generated automata are equivalent."""
        # Expect identity
        regex = "the"
        token_remapper = relm.regex_token_remapper.OffsetTokenRemapper(60000)
        full_vocab = {x: token_remapper.encode(ord(x)) for x in regex}
        char_tokens = list(map(lambda x: full_vocab[x], regex))
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        automata = relm.automata.automata_from_token_list(char_tokens)
        automata = relm.automata.attach_symbol_table(
            automata, inverted_full_vocab)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            allow_deletes=False,
            allow_inserts=False,
            allow_substitutions=False,
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata, token_remapper)
        # Swap table to add vocab
        relm.automata.attach_symbol_table(automata,
                                          new_automata.input_symbols())
        self.assertTrue(fst.equivalent(new_automata, automata))

        list_of_list_of_tokens = {(1,), (2,), (2, 2,), (2, 2, 2)}
        token_remapper = None
        automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        symbol_table = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
        }
        relm.automata.attach_symbol_table(automata, symbol_table)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            allow_deletes=False,
            allow_inserts=False,
            allow_substitutions=False,
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata, token_remapper)
        relm.automata.attach_symbol_table(automata,
                                          new_automata.input_symbols())
        self.assertTrue(fst.equivalent(new_automata, automata))
        charset = ''.join([chr(x) for x in symbol_table.keys()])
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            charset,
            allow_deletes=True,
            allow_inserts=False,
            allow_substitutions=False,
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata, token_remapper)
        relm.automata.attach_symbol_table(automata,
                                          new_automata.input_symbols())
        self.assertFalse(fst.equivalent(new_automata, automata))

        list_of_list_of_tokens = {(ord('a'),), }
        token_remapper = None
        automata = relm.automata.automata_union_from_list_of_token_list(
            list_of_list_of_tokens,
            determinize=True,
            minimize=True,
            rm_epsilon=True,
            verify=True)
        symbol_table = {
            ord('a'): 'a',
        }
        relm.automata.attach_symbol_table(automata, symbol_table)
        preprocessor = relm.regex_token_preprocessor.LevenshteinTransformer(
            allow_deletes=True,
            allow_inserts=False,
            allow_substitutions=False,
        )
        new_automata = automata.copy()
        new_automata = preprocessor.transform(new_automata, token_remapper)
        list_of_list_of_tokens = {(ord('a'),), tuple()}
        expected_automata = \
            relm.automata.automata_union_from_list_of_token_list(
                list_of_list_of_tokens,
                determinize=True,
                minimize=True,
                rm_epsilon=True,
                verify=True)
        relm.automata.attach_symbol_table(expected_automata,
                                          new_automata.input_symbols())
        self.assertTrue(fst.equivalent(new_automata, expected_automata))
