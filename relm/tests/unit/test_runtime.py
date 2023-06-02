"""Tests for relm runtime.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import unittest

import relm.automata
import relm.regex_backend.rust_regex_backend as rust_regex_backend


class TestRuntime(unittest.TestCase):
    """Test that runtime mirrors expected Python behavior (end-to-end)."""

    def test_regex_to_fst(self):
        """Test that regex compilation works."""
        regex = "a|b|c|(defg)*"
        dfa = rust_regex_backend.regex_to_automata(regex)
        chars = [['a'], ['b'], ['c'], [], ['d', 'e', 'f', 'g'],
                 ['d', 'e', 'f', 'g', 'd', 'e', 'f', 'g']]
        token_patterns = map(lambda x: list(map(ord, x)), chars)
        for pattern in token_patterns:
            self.assertTrue(relm.automata.apply_fst_accepted(pattern, dfa))
        chars = [['d'], ['x'], ['A'], ['Z']]
        token_patterns = map(lambda x: list(map(ord, x)), chars)
        for pattern in token_patterns:
            self.assertFalse(relm.automata.apply_fst_accepted(pattern, dfa))
