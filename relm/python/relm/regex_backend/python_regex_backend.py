"""Parse Regex to Generative AST and FST using pyparsing.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0

This software is based off of the PyParsing invRegex.py example,
which is licensed under the MIT License:

    Copyright 2008, Paul McGuire

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import functools
import itertools
from abc import ABC
from typing import Callable, Generator, Optional

import relm.automata
from relm.regex_token_remapper import RegexTokenRemapper

__all__ = ["count", "invert"]

import pyparsing
from pyparsing import (Combine, Literal, ParseFatalException, ParserElement,
                       ParseResults, SkipTo, Suppress, Word, infixNotation,
                       nums, oneOf, opAssoc, srange)

printables = pyparsing.pyparsing_unicode.printables

ParserElement.enablePackrat()


def _add_cardinality(a, b):
    if a is None:
        return None
    if b is None:
        return None
    return a + b


def sanitize_regex_literal_str(regex: str):
    """Remove special characters in regex that should be literal."""
    return (regex.replace("\n", "\\n")
                 .replace(".", r"\.")
                 .replace("*", "\\*")
                 .replace("+", "\\+")
                 .replace("?", "\\?")
                 .replace(":", "\\:")
                 .replace("[", "\\[")
                 .replace("]", "\\]")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("(", "\\(")
                 .replace(")", "\\)")
                 .replace("|", "\\|")
                 .replace("\t", "\\t"))


def regex_string_iterator(regex: str, allow_infinite: bool = True):
    """Process a regex into an iterator over strings."""
    if allow_infinite:
        sample_infinite = False
        throw_on_infinite = False
    else:
        sample_infinite = False
        throw_on_infinite = True
    p = parser(sample_infinite=sample_infinite,
               throw_on_infinite=throw_on_infinite)
    ret = p.parseString(regex)
    ret = GroupEmitter(ret)
    make_gen = ret.makeGenerator()
    return make_gen()


def regex_string_emitter(regex: str, allow_infinite: bool = True):
    """Process a regex into an emitter object."""
    if allow_infinite:
        sample_infinite = False
        throw_on_infinite = False
    else:
        sample_infinite = False
        throw_on_infinite = True
    # TODO(mkuchnik): Add empty emitter for empty string
    p = parser(sample_infinite=sample_infinite,
               throw_on_infinite=throw_on_infinite)
    ret = p.parseString(regex)
    ret = GroupEmitter(ret)
    return ret


class RegexEmitter(ABC):
    """Abstract class for regex emitter.

    Used to generate from the regex it represents.
    """

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Generate from the regex."""
        raise NotImplementedError("Generator not implemented for {}.".format(
            type(self)
        ))

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        raise NotImplementedError("Generator not implemented for {}."
                                  .format(type(self))
                                  )

    def to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None,
                   verify: bool = True):
        """Return an openfst representation of the emitter."""
        fst = self._to_openfst(token_remapper)
        symbol_table = {}
        for c in self.symbol_set():
            try:
                ec = c.encode('utf8')
            except UnicodeEncodeError:
                continue
            if token_remapper:
                encoded_c = token_remapper.encode(ord(c))
            else:
                encoded_c = ord(c)
            symbol_table[encoded_c] = ec.decode("utf8")
        relm.automata.attach_symbol_table(fst, symbol_table)
        if verify and not fst.verify():
            raise ValueError("Automata failed verification")
        return fst

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        raise NotImplementedError("to_openfst not implemented for {}."
                                  .format(type(self))
                                  )

    def symbol_set(self):
        """Return the set of symbols used for the generator.

        Note that these are not remapped.
        """
        raise NotImplementedError("Symbol set not implemented for {}."
                                  .format(type(self))
                                  )

    def name(self) -> str:
        """Return the name of the class or canonical name."""
        return type(self).__name__


class EmptyEmitter(RegexEmitter):
    """Represents the empty set."""

    def __str__(self):
        return "EmptyEmitter"

    def __repr__(self):
        return "EmptyEmitter"

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Generate from the regex."""
        def genChars():
            return
            yield None

        return genChars

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        return 0

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        return relm.automata.null_automata()

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return set()


class CharacterRangeEmitter(RegexEmitter):
    """Represents a range of literal characters e.g., 'abcde'."""

    def __init__(self, chars):
        # remove duplicate chars in character range, but preserve original
        # order
        seen = set()
        self.charset = "".join(
            seen.add(c) or c for c in chars if c not in seen)

    def __str__(self):
        return "[{}]".format(self.charset)

    def __repr__(self):
        return "{}:[{}]".format(self.name(), self.charset)

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Generate from the regex."""
        def genChars():
            yield from self.charset

        return genChars

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        return len(self.charset)

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        query_tokens_rep = self.charset
        query_tokens_rep = map(lambda x: ord(x), query_tokens_rep)
        if token_remapper:
            query_tokens_rep = map(lambda x: token_remapper.encode(x),
                                   query_tokens_rep)
        query_tokens_rep = set(query_tokens_rep)
        if query_tokens_rep:
            automata = relm.automata.union_automata_from_token_list(
                query_tokens_rep)
        else:
            automata = relm.automata.null_automata()
        return automata

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return set(x for x in self.charset)


class OptionalEmitter(RegexEmitter):
    """Represents a '?' optional expression.

    Same as {0, 1}
    """

    def __init__(self, expr):
        if not isinstance(expr, RegexEmitter):
            raise ValueError(
                "expr must be a RegexEmitter, but found {}".format(type(expr)))
        self.expr = expr

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Yield nothing and then yield the optional element."""
        def optionalGen():
            yield ""
            yield from self.expr.makeGenerator()()

        return optionalGen

    def __str__(self):
        return "{}?".format(self.expr)

    def __repr__(self):
        return "{}:{}".format(self.name(), repr(self.expr))

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        return 1 + self.expr.cardinality()

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        automata = self.expr._to_openfst(token_remapper)
        optional_automata = relm.automata.optional_automata(automata)
        return optional_automata

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return self.expr.symbol_set()


class RepetitionEmitter(RegexEmitter):
    """Represents a '*' repeat 0 or more times expression (or variants).

    Can be used to implement *, +, and ?, since these are special repeat cases.
    """

    def __init__(self, expr, start_length=None, end_length=None):
        if not isinstance(expr, RegexEmitter):
            raise ValueError(
                "expr must be a RegexEmitter, but found {}".format(type(expr)))
        if start_length is None:
            start_length = 0
        self.expr = expr
        self.start_length = start_length
        self.end_length = end_length

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Yield nothing and then yield the element 1+ times.

        [A-D]* should expand to
        ""
        "A" "B" "C" "D"
        "AA" "AB" "AC" "AD" "BA" "BB" "BC" "BD" "CA" "CB" "CC" ...

        Thus for Inner Regex R, and length N, we have |R|^N, where |R| is the
        cardinality of the regex (e.g., 4 in this case).
        """
        def repetitionGen():
            length = self.start_length
            while self.end_length is None or length <= self.end_length:
                # Try all combinations of length
                it = itertools.combinations_with_replacement(
                    self.expr.makeGenerator()(), length)
                it = map(lambda x: "".join(x), it)
                yield from it
                length += 1

        return repetitionGen

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        if self.end_length is None:
            return None
        expr_card = self.expr.cardinality()
        length = self.end_length - self.start_length + 1
        return expr_card ** length

    def __str__(self):
        return "{}:({}, {},{})".format(
            self.name(),
            self.expr,
            self.start_length,
            self.end_length)

    def __repr__(self):
        return "{}:({}, {},{})".format(
            self.name(),
            self.expr,
            self.start_length,
            self.end_length)

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        automata = self.expr._to_openfst(token_remapper)
        repeat_automata = relm.automata.repeat_automata(
            automata, min_length=self.start_length, max_length=self.end_length)
        return repeat_automata

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return self.expr.symbol_set()


class KleeneRepetitionEmitter(RepetitionEmitter):
    """Represents a '*' repeat 0 or more times expression."""

    def __init__(self, expr):
        super().__init__(expr)


class PlusRepetitionEmitter(RepetitionEmitter):
    """Represents a '+' repeat 1 or more times expression."""

    def __init__(self, expr):
        super().__init__(expr, start_length=1)


class DotEmitter(RegexEmitter):
    """Represents a '.' catch-all expression."""

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        """Yield all printables."""
        def dotGen():
            yield from printables

        return dotGen

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        return len(printables)

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return set(printables)


class GroupEmitter(RegexEmitter):
    """Contains a group of concatenated results.

    Commonly surrounded by parenthesis e.g., ((ab)cd).
    """

    def __init__(self, exprs):
        for expr in exprs:
            if not isinstance(expr, RegexEmitter):
                raise ValueError(
                    "expr must be a RegexEmitter, but found {}".format(
                        type(expr)))
        self._exprs = exprs
        self.exprs = ParseResults(exprs)

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        def groupGen():
            # TODO(mkuchnik): Investigate replacing with combinations
            def recurseList(elist):
                if len(elist) == 1:
                    yield from elist[0].makeGenerator()()
                else:
                    for s in elist[0].makeGenerator()():
                        for s2 in recurseList(elist[1:]):
                            yield s + s2

            if self.exprs:
                yield from recurseList(self.exprs)

        return groupGen

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        if self.exprs:
            def recurseList(elist):
                if len(elist) == 1:
                    s_card = elist[0].cardinality()
                    return s_card
                else:
                    s_card = elist[0].cardinality()
                    if s_card is None:
                        return None
                    s2_card = recurseList(elist[1:])
                    if s2_card is None:
                        return None
                    card = s_card * s2_card
                    return card
            card = recurseList(self.exprs)
        else:
            card = 0
        return card

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        if self.exprs:
            autos = [x._to_openfst(token_remapper) for x in self.exprs]
            auto = relm.automata.concat_automatas(autos)
        else:
            auto = relm.automata.null_automata()
        return auto

    def __repr__(self):
        return "{}:{}".format(self.name(), repr(self._exprs))

    def __str__(self):
        return "{}:{}".format(self.name(), str(self._exprs))

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        ss = set()
        for expr in self.exprs:
            ss = ss.union(expr.symbol_set())
        return ss


class AlternativeEmitter(RegexEmitter):
    """Represent an alternative (OR) of results e.g., <stuff>|<stuff>."""

    def __init__(self, exprs):
        for expr in exprs:
            if not isinstance(expr, RegexEmitter):
                raise ValueError(
                    "expr must be a RegexEmitter, but found {}".format(
                        type(expr)))
        self.exprs = exprs

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        def altGen():
            for e in self.exprs:
                yield from e.makeGenerator()()

        return altGen

    def __repr__(self):
        return "{}:{}".format(self.name(), repr(self.exprs))

    def __str__(self):
        return "{}:{}".format(self.name(), str(self.exprs))

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        card = 0
        for e in self.exprs:
            e_card = e.cardinality()
            if e_card is None:
                card = None
                break
            else:
                card += e_card

        return card

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        if self.exprs:
            autos = list(map(lambda x: x._to_openfst(token_remapper),
                             self.exprs))
            auto = relm.automata.union_automatas(autos)
        else:
            auto = relm.automata.null_automata()
        return auto

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        ss = set()
        for expr in self.exprs:
            ss = ss.union(expr.symbol_set())
        return ss


class LiteralEmitter(RegexEmitter):
    """A literal such as 'a' or '1'."""

    def __init__(self, lit):
        self.lit = lit

    def __str__(self):
        return "'{}'".format(self.lit)

    def __repr__(self):
        return "{}:'{}'".format(self.name(), self.lit)

    def makeGenerator(self) -> Callable[[], Generator[str, None, None]]:
        def litGen():
            yield self.lit

        return litGen

    def cardinality(self) -> Optional[int]:
        """Calculate the size of the set (if finite).

        :returns The size of the set or None if infinite.
        """
        return 1

    def _to_openfst(self, token_remapper: Optional[RegexTokenRemapper] = None):
        """Return an openfst representation of the emitter."""
        if token_remapper:
            list_of_tokens = [token_remapper.encode(ord(self.lit))]
        else:
            list_of_tokens = [ord(self.lit)]
        automata = \
            (relm
             .automata
             .automata_from_token_list(
                 list_of_tokens)
             )
        return automata

    def symbol_set(self):
        """Return the set of symbols used for the generator."""
        return set([self.lit])


def handleRange(toks):
    return CharacterRangeEmitter(srange(toks[0]))


def handleRepetition(toks, sample_infinite=False, throw_on_infinite=True):
    toks = toks[0]
    if toks[1] in "*":
        if sample_infinite:
            return RepetitionEmitter(toks[0], start_length=0, end_length=50)
        else:
            if throw_on_infinite:
                raise ParseFatalException("* is infinite")
            else:
                return KleeneRepetitionEmitter(toks[0])
    elif toks[1] in "+":
        if sample_infinite:
            return RepetitionEmitter(toks[0], start_length=1, end_length=50)
        else:
            if throw_on_infinite:
                raise ParseFatalException("+ is infinite")
            else:
                return PlusRepetitionEmitter(toks[0])
    elif toks[1] == "?":
        return OptionalEmitter(toks[0])
    elif "count" in toks:
        return GroupEmitter([toks[0]] * int(toks.count))
    elif "minCount" in toks:
        # NOTE(mkuchnik): The interpretation of this pattern is that any of the
        # characters in the set can be chosen to come up with a string of
        # length minlength and length maxlength.
        mincount = int(toks.minCount)
        maxcount = int(toks.maxCount)
        optcount = maxcount - mincount
        if optcount:
            opt = OptionalEmitter(toks[0])
            for i in range(1, optcount):
                opt = OptionalEmitter(GroupEmitter([toks[0], opt]))
            return GroupEmitter([toks[0]] * mincount + [opt])
        else:
            return [toks[0]] * mincount
    else:
        raise NotImplementedError("Don't know how to handle toks: {}".format(
            toks))


def handleLiteral(toks):
    lit = ""
    for t in toks:
        if t[0] == "\\":
            if t[1] == "t":
                lit += "\t"
            elif t[1] == "n":
                # NOTE(mkuchnik): Added
                lit += "\n"
            else:
                lit += t[1]
        else:
            lit += t
    return LiteralEmitter(lit)


def handleMacro(toks):
    macroChar = toks[0][1]
    if macroChar == "d":
        return CharacterRangeEmitter("0123456789")
    elif macroChar == "w":
        return CharacterRangeEmitter(srange("[A-Za-z0-9_]"))
    elif macroChar == "s":
        return LiteralEmitter(" ")
    else:
        raise ParseFatalException(
            "", 0, "unsupported macro character (" + macroChar + ")"
        )


def handleSequence(toks):
    return GroupEmitter(toks[0])


def handleDot():
    return CharacterRangeEmitter(printables)


def handleAlternative(toks):
    return AlternativeEmitter(toks[0])


@functools.lru_cache(maxsize=None)
def parser(sample_infinite=False, throw_on_infinite=True):
    """Build the parser regex matching.

    Does implicit caching to speed up repeated calls.
    """
    ParserElement.setDefaultWhitespaceChars("")
    lbrack, rbrack, lbrace, rbrace, lparen, rparen, colon, qmark = map(
        Literal, "[]{}():?"
    )

    reMacro = Combine("\\" + oneOf(list("dws")))
    escapedChar = ~reMacro + Combine("\\" + oneOf(list(printables)))
    reLiteralChar = (
        "".join(c for c in printables if c not in r"\[]{}().*?+|") + " \t"
    )

    reRange = Combine(lbrack + SkipTo(rbrack, ignore=escapedChar) + rbrack)
    reLiteral = escapedChar | oneOf(list(reLiteralChar))
    reNonCaptureGroup = Suppress("?:")
    reDot = Literal(".")
    repetition = (
        (lbrace + Word(nums)("count") + rbrace)
        | (lbrace + Word(nums)("minCount") + ","
           + Word(nums)("maxCount") + rbrace)
        | oneOf(list("*+?"))
    )

    reRange.setParseAction(handleRange)
    reLiteral.setParseAction(handleLiteral)
    reMacro.setParseAction(handleMacro)
    reDot.setParseAction(handleDot)

    reTerm = reLiteral | reRange | reMacro | reDot | reNonCaptureGroup
    _handleRepetition = functools.partial(handleRepetition,
                                          sample_infinite=sample_infinite,
                                          throw_on_infinite=throw_on_infinite)
    reExpr = infixNotation(
        reTerm,
        [
            (repetition, 1, opAssoc.LEFT, _handleRepetition),
            (None, 2, opAssoc.LEFT, handleSequence),
            (Suppress("|"), 2, opAssoc.LEFT, handleAlternative),
        ],
    )
    _parser = reExpr

    return _parser


def count(gen):
    """Count the number of elements returned by a generator."""
    return sum(1 for _ in gen)


def invert(regex):
    r"""Return the strings that match the regex.

    Call this routine as a generator to return all the strings that
    match the input regular expression.
        for s in invert(r"[A-Z]{3}\d{3}"):
            print s
    """
    ast = parser().parseString(regex)
    invReGenerator = GroupEmitter(ast).makeGenerator()
    return invReGenerator()


def invert_cardinality(regex):
    """Return the cardinality of the set of the regex.

    :return The calculated cardinality of the regex.
    """
    ast = parser().parseString(regex)
    return GroupEmitter(ast).cardinality()


def big_test():
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

    expected_counts = [5, None, 64, 27, 27, 10, 100, 1, 8, 4, 1, 26, 26, 256,
                       12, 40, 27, 54, 27, 30, 900, 3, 4, 29, 59, 13, 16, 118,
                       4, 4, 97020, 7, 12]

    for t, expected_num in zip(tests, expected_counts):
        t = t.strip()
        print("-" * 50)
        print(t)
        try:
            num = count(invert(t))
            print(num)
            cardinality = invert_cardinality(t)
            print(cardinality)
            maxprint = 30
            for s in invert(t):
                print(s)
                maxprint -= 1
                if not maxprint:
                    break
            assert num == expected_num, \
                "Expected {} but got {} results for {}.".format(
                    expected_num, num, t)
        except ParseFatalException as pfe:
            assert expected_num is None, \
                "Expected to see {} results, but did not parse {}.".format(
                    expected_num, t
                )
            print(pfe.msg)
            print("")
            continue
        print("")


def small_test():
    tests = [
        r"(a){1,3}",
        r"i\\nlike\\ncheese",
        r"the •",  # TODO(mkuchnik): The dot is omitted (in general non-ASCII)
    ]
    expected_counts = [3, 1, 1]
    for t, expected_num in zip(tests, expected_counts):
        try:
            num = count(invert(t))
            print(num)
            cardinality = invert_cardinality(t)
            print(cardinality)
            maxprint = 30
            for s in invert(t):
                print(s)
                maxprint -= 1
                if not maxprint:
                    break
            assert num == expected_num, \
                "Expected {} but got {} results for {}.".format(
                    expected_num, num, t)
            assert cardinality == expected_num, \
                "Expected {} but got {} cardinality for {}.".format(
                    expected_num, cardinality, t)
        except ParseFatalException as pfe:
            assert expected_num is None, \
                "Expected to see {} results, but did not parse {}.".format(
                    expected_num, t
                )
            print(pfe.msg)
            print("")
            continue
        print("")


def main():
    """Run some tests."""
    big_test()
    small_test()


if __name__ == "__main__":
    main()
