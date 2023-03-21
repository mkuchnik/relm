"""Tests for end-to-end relm."""

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.facade
import relm.regex_token_preprocessor
import relm.util


class TestInterface(unittest.TestCase):
    """Test the public relm interface."""

    def test_search(self):
        """Test that token lists are accepted."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query = relm.facade.SearchQuery("The .*")
        query.num_samples = 5
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.RANDOMIZED
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 5)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertTrue(s.startswith("The "))

    def test_search_iterator(self):
        """Test that token lists are accepted."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query = relm.facade.SearchQuery("The .*")
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.RANDOMIZED
        ret = relm.facade.search(model, tokenizer, query)
        i = 0
        for i, x in enumerate(ret):
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertTrue(s.startswith("The "))
            if i > 10:
                break
        self.assertGreater(i, 10)

    def test_search_tail(self):
        """Test that token lists are accepted."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query = relm.facade.SearchQuery("The .* man")
        query.num_samples = 3
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.RANDOMIZED
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 3)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertTrue(s.startswith("The "))
            self.assertTrue(s.endswith(" man"))

    def test_search_automata_python(self):
        """Test that token lists are accepted."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query = relm.facade.SearchQuery("The man")
        query.num_samples = 5
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.PYTHON
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 5)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertEqual(s, "The man")
        query = relm.facade.SearchQuery("(The)|(man)")
        query.num_samples = 5
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.PYTHON
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 5)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertTrue(s == "The" or s == "man")

        # Test accept
        query_str = "a( b)?"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = None
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.top_k_sampling = 1
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.PYTHON
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 0)

        # Test accept
        query_str = "a( b)?"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = "a"
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.top_k_sampling = 1
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.PYTHON
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 1)

        query_str = "a b"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = "a"
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.PYTHON
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer()]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 1489)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))

    def test_search_automata_rust(self):
        """Test that token lists are accepted."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query = relm.facade.SearchQuery("The man")
        query.num_samples = 5
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 5)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertEqual(s, "The man")
        query = relm.facade.SearchQuery("(The)|(man)")
        query.num_samples = 5
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        ret = relm.facade.search(model, tokenizer, query)
        self.assertEqual(len(ret), 5)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            self.assertTrue(s == "The" or s == "man")

        # Test accept
        query_str = "a( b)?"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = None
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.top_k_sampling = 1
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 0)

        # Test accept
        query_str = "a( b)?"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = "a"
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.top_k_sampling = 1
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 1)

        query_str = "a b"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = "a"
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_dijkstra = True
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer()]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        self.assertEqual(len(ret), 1489)
        for x in ret:
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))

    def test_search_levenshtein(self):
        """Test tricky Levenshteins."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)
        query_str = " w"
        query = relm.facade.SearchQuery(query_str)
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        charset = "".join(set(query_str))
        charset += "?"
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer(
                charset,
                allow_inserts=True,
                allow_deletes=True,
                allow_substitutions=True,
            )]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        for x in ret:
            # NOTE(mkuchnik): Tokenization skips spaces unless we explicitly
            # add it.
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))

        query_str = "Hello"
        query = relm.facade.SearchQuery(query_str)
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        charset = None
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer(
                charset,
                allow_inserts=True,
                allow_deletes=True,
                allow_substitutions=True,
            )]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        for x in ret:
            # NOTE(mkuchnik): Tokenization skips spaces unless we explicitly
            # add it.
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))

        # Random sampling
        query_str = "Hello"
        query = relm.facade.SearchQuery(query_str)
        query.num_samples = 30
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        query.experimental_random_sampling = True
        query.experimental_penalized_accepted_probability = False
        query.experimental_penalized_accepted_probability = False
        query.experimental_add_eos_token = True
        query.experimental_truncate_automata = True
        charset = None
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer(
                charset,
                allow_inserts=True,
                allow_deletes=True,
                allow_substitutions=True,
            )]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        for x in ret:
            # NOTE(mkuchnik): Tokenization skips spaces unless we explicitly
            # add it.
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))

        query_str = "codE"
        query = relm.facade.SearchQuery(query_str)
        query.num_samples = None
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        charset = None
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer(
                charset,
                num_edits=2,
                allow_inserts=True,
                allow_deletes=True,
                allow_substitutions=True,
            )]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        max_distance = 0
        for x in ret:
            # NOTE(mkuchnik): Tokenization skips spaces unless we explicitly
            # add it.
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            max_distance = max(distance, max_distance)
            self.assertLess(distance, 3,
                            "'{}' ({}) has distance {}".format(s, x, distance))
        self.assertEqual(max_distance, 2)

        # Random sampling
        query_str = "a b"
        query = relm.facade.SearchQuery(query_str)
        query.accept_str = "a"
        query.num_samples = 30
        query.progress_bar = False
        query.backend = relm.facade.SearchBackendType.AUTOMATA
        query.experimental_advanced_parsing = True
        query.experimental_regex_backend = relm.facade.RegexBackendType.RUST
        query.experimental_random_sampling = True
        query.experimental_penalized_accepted_probability = False
        query.experimental_penalized_accepted_probability = False
        query.experimental_add_eos_token = True
        query.experimental_truncate_automata = True
        charset = None
        query.experimental_automata_preprocessors = \
            [relm.regex_token_preprocessor.LevenshteinTransformer(
                charset,
                allow_inserts=True,
                allow_deletes=True,
                allow_substitutions=True,
            )]
        ret = relm.facade.search(model, tokenizer, query)
        ret = list(ret)
        for x in ret:
            # NOTE(mkuchnik): Tokenization skips spaces unless we explicitly
            # add it.
            s = tokenizer.decode(x, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            distance = relm.util.levenshtein_distance(
                s, query_str)
            self.assertLess(distance, 2,
                            "'{}' ({}) has distance {}".format(s, x, distance))
