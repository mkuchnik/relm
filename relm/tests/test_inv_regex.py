"""Test that regexes can be generated in inverted fashion."""

import unittest

import relm.regex_backend.python_regex_backend as python_regex_backend


class TestInvertedRegex(unittest.TestCase):
    """Test that inverted regex generates desired strings."""

    def test_literal(self):
        """Test that literals are emitted."""
        regex = "apple"
        expected_strings = {regex}
        it = python_regex_backend.regex_string_iterator(regex)
        self.assertEqual(set(it), expected_strings)

    def test_repetition(self):
        """Test that optional strings are emitted."""
        regex = "(apple)(_jacks)?"
        expected_strings = {"apple", "apple_jacks"}
        it = python_regex_backend.regex_string_iterator(regex)
        self.assertEqual(set(it), expected_strings)

    def test_or(self):
        """Test that or'd strings are emitted."""
        regex = "apples|oranges"
        expected_strings = {"apples", "oranges"}
        it = python_regex_backend.regex_string_iterator(regex)
        self.assertEqual(set(it), expected_strings)

    def test_big_test(self):
        """Test a bunch of regex queries."""
        tests = r"""
        [A-EA]
        [A-D]*
        [A-D]{3}
        X[A-C]{3}Y
        X[A-C]{3}\(
        X\d
        foobar\d\d
        foobar{2}
        foobar{2,9}
        fooba[rz]{2}
        (foobar){2}
        ([01]\d)|(2[0-5])
        (?:[01]\d)|(2[0-5])
        ([01]\d\d)|(2[0-4]\d)|(25[0-5])
        [A-C]{1,2}
        [A-C]{0,3}
        [A-C]\s[A-C]\s[A-C]
        [A-C]\s?[A-C][A-C]
        [A-C]\s([A-C][A-C])
        [A-C]\s([A-C][A-C])?
        [A-C]{2}\d{2}
        @|TH[12]
        @(@|TH[12])?
        @(@|TH[12]|AL[12]|SP[123]|TB(1[0-9]?|20?|[3-9]))?
        @(@|TH[12]|AL[12]|SP[123]|TB(1[0-9]?|20?|[3-9])|OH(1[0-9]?|2[0-9]?|30?|[4-9]))?
        (([ECMP]|HA|AK)[SD]|HS)T
        [A-CV]{2}
        A[cglmrstu]|B[aehikr]?|C[adeflmorsu]?|D[bsy]|E[rsu]|F[emr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airu]|M[dgnot]|N[abdeiop]?|Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|Uu[bhopqst]|U|V|W|Xe|Yb?|Z[nr]
        (a|b)|(x|y)
        (a|b) (x|y)
        [ABCDEFG](?:#|##|b|bb)?(?:maj|min|m|sus|aug|dim)?[0-9]?(?:/[ABCDEFG](?:#|##|b|bb)?)?
        (Fri|Mon|S(atur|un)|T(hur|ue)s|Wednes)day
        A(pril|ugust)|((Dec|Nov|Sept)em|Octo)ber|(Febr|Jan)uary|Ju(ly|ne)|Ma(rch|y)
        """.splitlines()
        tests = list(filter(lambda x: x, tests))  # Remove blanks

        expected_counts = [5, None, 64, 27, 27, 10, 100, 1, 8, 4, 1, 26, 26,
                           256, 12, 40, 27, 54, 27, 30, 900, 3, 4, 29, 59, 13,
                           16, 118, 4, 4, 97020, 7, 12]

        for t, expected_num in zip(tests, expected_counts):
            t = t.strip()
            try:
                num = python_regex_backend.count(
                    python_regex_backend.invert(t))
                cardinality = python_regex_backend.invert_cardinality(t)
                self.assertEqual(num, expected_num)
                self.assertEqual(cardinality, expected_num)
            except python_regex_backend.ParseFatalException:
                self.assertEqual(expected_num, None)
                continue

    def test_small_test(self):
        """Test microbenchmark-style tests."""
        tests = [
            r"(a){1,3}",
            r"\.",
            r"\n",
            r"\t",
        ]
        expected_counts = [3, 1, 1, 1]
        for t, expected_num in zip(tests, expected_counts):
            try:
                num = python_regex_backend.count(
                    python_regex_backend.invert(t))
                cardinality = python_regex_backend.invert_cardinality(t)
                self.assertEqual(num, expected_num)
                self.assertEqual(cardinality, expected_num)
            except python_regex_backend.ParseFatalException:
                self.assertEqual(expected_num, None)
                continue
