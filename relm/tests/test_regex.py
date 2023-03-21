"""Tests for relm regex."""

import unittest

import relm.automata
import relm.regex


class TestRegexGenerator(unittest.TestCase):
    """Test that regex generator works."""

    def test_acceptance(self):
        """Test that token lists are accepted."""
        generator = relm.regex.RegexGenerator({1, 2, 3})
        regex = generator.sample()
        self.assertTrue(isinstance(regex, list))
