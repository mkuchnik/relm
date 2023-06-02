"""Tests for relm regex matcher.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import unittest

import relm.regex_matcher


class TestRegexMatcher(unittest.TestCase):
    """Test that regex matcher works."""

    def test_symbol(self):
        """Test that symbols are accepted."""
        matcher = relm.regex_matcher.build_regex("a")
        self.assertTrue(matcher.match("a"))
        self.assertFalse(matcher.match("b"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
        self.assertFalse(matcher.match("aa"))
        matcher = relm.regex_matcher.build_regex("b")
        self.assertTrue(matcher.match("b"))
        self.assertFalse(matcher.match("a"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
        self.assertFalse(matcher.match("aa"))
        self.assertFalse(matcher.match("bb"))

    def test_concatenated_symbols(self):
        """Test that sequences of symbols are accepted."""
        matcher = relm.regex_matcher.build_regex("aa")
        self.assertTrue(matcher.match("aa"))
        self.assertFalse(matcher.match("b"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
        self.assertFalse(matcher.match("a"))
        matcher = relm.regex_matcher.build_regex("ab")
        self.assertTrue(matcher.match("ab"))
        self.assertFalse(matcher.match("b"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
        self.assertFalse(matcher.match("ba"))
        matcher = relm.regex_matcher.build_regex("abc")
        self.assertTrue(matcher.match("abc"))
        self.assertFalse(matcher.match("b"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
        self.assertFalse(matcher.match("ba"))
        self.assertFalse(matcher.match("ab"))
        self.assertFalse(matcher.match("abd"))
        self.assertFalse(matcher.match("cba"))
        matcher = relm.regex_matcher.build_regex("abcdefghifk")
        self.assertTrue(matcher.match("abcdefghifk"))
        self.assertFalse(matcher.match("abcdefghif"))
        self.assertFalse(matcher.match("bcdefghifk"))
        self.assertFalse(matcher.match("abcdeghifk"))
        self.assertFalse(matcher.match("a"))
        self.assertFalse(matcher.match(""))

    def test_or_symbols(self):
        """Test that sequences of OR symbols are accepted."""
        matcher = relm.regex_matcher.build_regex("a|b")
        self.assertTrue(matcher.match("a"))
        self.assertTrue(matcher.match("b"))
        self.assertFalse(matcher.match("ab"))
        self.assertFalse(matcher.match("ba"))
        self.assertFalse(matcher.match("C"))
        self.assertFalse(matcher.match(""))
