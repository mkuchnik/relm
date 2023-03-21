"""Old (deprecated) functions to randomly sample matching patterns."""

import random
import re
from dataclasses import dataclass

from . import relm_logging

logger = relm_logging.get_logger()


@dataclass
class OptimizationFlags:
    """Flags to enable regex search features."""

    truncate_max_tokens: bool = True
    indexing: bool = False
    replace_badmatch_with_none: bool = False
    verbose: bool = False
    top_k_sampling: int = None


@dataclass
class SearchStats:
    """Statistics for hits and misses during regex search."""

    n_hits: int = 0
    n_tries: int = 0
    n_false_positives: int = 0


def simple_prefix_in_regex(query):
    """Return a string (not-regex) prefix in regex."""
    special_chars = set([
        "(",
        ")",
        "[",
        "]",
        "*",
        "+",
        ".",
        "|",
    ])
    special_char_idxs = [i for i, char in enumerate(query) if char in
                         special_chars]
    # TODO(mkuchnik): Need to implement full stack-based parsing
    if not special_char_idxs:
        return query
    first_special_idx = min(special_char_idxs)
    return query[:first_special_idx]


def infer_max_length_in_regex(query):
    """Find the maximum number of words in the query."""
    # TODO(mkuchnik): Hacky and will not work in general
    # for example, [\w ]+ is infinite length, but we will not catch it.
    # Need basic parsing to catch brackets
    words = [x.strip() for x in query.split(" ")]

    def is_basic_word(word):
        if "+" not in word and "*" not in word:
            return True
        else:
            if word == r"[\w]+" or word == r"[\w]*":
                return True
        return False

    def count_length_word(word):
        if word == ".*":
            return None
        elif word == ".+":
            return None
        elif is_basic_word(word):
            return 1
        else:
            return None

    word_lengths = [count_length_word(word) for word in words]
    if None in word_lengths:
        return None
    else:
        sum_len = sum(word_lengths)
        # Make it one bigger to be safe
        # TODO(mkuchnik): Investigate relation between token length and words
        return sum_len + 1


def sampled_regex_search_iterator(test_relm, query, return_probabilities=False,
                                  return_tokens=False,
                                  optimization_flags=None, proxy_model=None,
                                  batch_size=None,
                                  test_relm_confirm_fn=None):
    """Run the query on the test_relm and yields all matching samples."""
    prefix = simple_prefix_in_regex(query)
    logger.info(
        "Starting query search with pattern: '{}'. Prefix: '{}'".format(
            query, prefix))
    re_pattern = re.compile(query)

    if optimization_flags is None:
        optimization_flags = OptimizationFlags()

    if optimization_flags.truncate_max_tokens:
        max_query_length = infer_max_length_in_regex(query)
    else:
        max_query_length = None
    if max_query_length:
        logger.warning("Max query length set to: {}".format(max_query_length))

    def unpack_sample_from_ret(ret):
        # If probabilities are used, we will see tuples
        if return_probabilities or return_tokens:
            return ret[0]
        else:
            return ret

    if proxy_model:
        assert test_relm_confirm_fn, "Must provide confirm fn if using proxy"

    top_k = optimization_flags.top_k_sampling
    stats = SearchStats()
    if proxy_model:
        for i, x in enumerate(
            # Sample from proxy
            proxy_model.sample_iterator(
                prefix,
                batch_size=batch_size,
                return_probabilities=return_probabilities,
                return_tokens=return_tokens,
                max_length=max_query_length,
                top_k=top_k)):
            sample = unpack_sample_from_ret(x)
            assert isinstance(sample, str), \
                "Sample must be a string. Got {} of type {}.".format(
                    sample, type(sample))
            stats.n_tries += 1
            if re_pattern.fullmatch(sample):
                stats.n_hits += 1
                if not test_relm_confirm_fn(test_relm, sample):
                    stats.n_false_positives += 1
                else:
                    yield x
            elif optimization_flags.replace_badmatch_with_none:
                yield None
            if i % 100 == 0 and optimization_flags.verbose:
                p_hit = float(stats.n_hits) / float(stats.n_tries)
                p_false_positive = (float(stats.n_false_positives)
                                    / float(stats.n_tries))
                # TODO(mkuchnik): Convert to timer
                msg = "P_hit: {}. P_false_positive: {}".format(
                    p_hit, p_false_positive)
                logger.info(msg)
    else:
        for i, x in enumerate(
            test_relm.sample_iterator(
                prefix,
                batch_size=batch_size,
                return_probabilities=return_probabilities,
                return_tokens=return_tokens,
                max_length=max_query_length,
                top_k=top_k)):
            sample = unpack_sample_from_ret(x)
            stats.n_tries += 1
            assert isinstance(sample, str), \
                "Sample must be a string. Got {} of type {}.".format(
                    sample, type(sample))
            if re_pattern.fullmatch(sample):
                stats.n_hits += 1
                yield x
            elif optimization_flags.replace_badmatch_with_none:
                yield None
            if i % 100 == 0 and optimization_flags.verbose:
                p_hit = float(stats.n_hits) / float(stats.n_tries)
                # TODO(mkuchnik): Convert to timer
                msg = "P_hit: {}.".format(p_hit)
                logger.info(msg)


class RegexGenerator:
    """A generator over random regexes.

    For example, if the language set is {1, 2, 3}
    the generator may generate:
    [1]
    [1, 2, 3]
    [1, 2, 3, 3, 3]
    [1, 2, 3, 3, '|', 3]
    [1, 2, 3, '*']
    ['(', 1, 2, 3, ')', '|', '.', '*']

    Notice that strings are special characters.
    """

    def __init__(self, language_set, max_samples=None):
        """Initialize the generate to use a language."""
        self.language = set(language_set)
        if max_samples is None:
            max_samples = 10
        self.max_samples = max_samples
        self.ops = [lambda x: self._sample_token(x),
                    lambda x: self._sample_parenthesis(x),
                    lambda x: self._sample_or(x),
                    lambda x: self._sample_wildcard(x),
                    lambda x: self._sample_any_token(x)]

    def _sample_token(self, curr_sample):
        return [*curr_sample, random.sample(self.language, 1)]

    def _sample_parenthesis(self, curr_sample):
        return ["(", *curr_sample, ")"]

    def _sample_or(self, curr_sample):
        return [*curr_sample, "|"]

    def _sample_wildcard(self, curr_sample):
        return [*curr_sample, "*"]

    def _sample_any_token(self, curr_sample):
        return [*curr_sample, "."]

    def sample(self):
        """Return a random regex."""
        curr_sample = []
        n_samples = random.randint(1, self.max_samples)
        for _ in range(n_samples):
            op = random.choice(self.ops)
            curr_sample = op(curr_sample)
        return curr_sample
