"""Module re-implementing regexes to get NFA and DFA data structures."""

from abc import ABC

import relm.regex_backend.python_regex_backend as python_regex_backend

from . import relm_logging

logger = relm_logging.get_logger()


class Regex(ABC):
    """Abstract class for regex matching."""

    def match(self, test_string: str) -> bool:
        """Return true if Regex matches string."""
        pass

    def match_or_throw(self, test_string: str) -> bool:
        """Return true if Regex matches string."""
        ret = self.match(test_string)
        if not ret:
            raise ValueError("{} does not match".format(test_string))
        return ret

    def greedy_match(self, test_string: str) -> (bool, str):
        """Return true if Regex matches part of string and return remaining."""
        return (self.match(test_string), "")


class WrappedEmitterRegex(Regex):
    """Wraps a python_regex_backend emitter object."""

    def __init__(self, emitter):
        """Wrap an emitter."""
        self.emitter = emitter
        self._gen = None

    def _make_gen(self):
        """Cache a generator from emitter."""
        if self._gen is None:
            self._gen = self.emitter.makeGenerator()

    def match(self, test_string: str) -> bool:
        """Return true if Regex matches string."""
        self._make_gen()
        # TODO(mkuchnik): This is brute force.
        return test_string in self._gen()


def build_regex(regex_str: str) -> Regex:
    """Take a regex like 'abc|d' and return a Regex object to match it."""
    if not isinstance(regex_str, str):
        raise ValueError("Expected str, got: {}".format(type(regex_str)))
    emitter = python_regex_backend.regex_string_emitter(regex_str)
    regex = WrappedEmitterRegex(emitter)
    return regex
