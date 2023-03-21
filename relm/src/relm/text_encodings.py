r"""Various utilities for reasoning about tokenizer textual encodings of words.

For example, the tokenizer has a set of tokens, which can be composed to words.
This module is meant to determine the set of tokens and answer questions on how
they are composed to particular words.
"""

import collections.abc
import functools

import regex as re

from . import relm_logging

logger = relm_logging.get_logger()

_unicode_character_cache = dict()


def _partitions(s: str):
    """Generate all partitions of a string.

    Note, this generates 2^{n-1} output strings, where n is the length of
    the string.

    From
    https://stackoverflow.com/questions/52167339/get-all-possible-str-partitions-of-any-length
    """
    if len(s) > 0:
        for i in range(1, len(s) + 1):
            first, rest = s[:i], s[i:]
            for p in _partitions(rest):
                yield [first] + p
    else:
        yield []


class LazyDict(collections.abc.Mapping):
    """
    A dictionary that stores a function and args to lazily evaluate the map.

    For example, when initialized with a dict of keys and values, this dict
    will use the values as a function that it invokes for that key.

    https://stackoverflow.com/questions/16669367/setup-dictionary-lazily
    """

    def __init__(self, *args, **kw):
        """Store a dict of keys and function values."""
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        """Execute the stored function with the key."""
        func, args = self._raw_dict.__getitem__(key)
        return func(*args)

    def __iter__(self):
        """Return an iterator over the raw dict."""
        return iter(self._raw_dict)

    def __len__(self):
        """Return the size of the raw dict."""
        return len(self._raw_dict)


class TextEncodingsGenerator:
    """Generate tokens from a string or regex."""

    def __init__(self, test_relm):
        """Initialize generator with the test_relm and setup caches."""
        self.test_relm = test_relm
        # Cache
        # NOTE(mkuchnik): self.tokenizer.vocab is a function that implements a
        # dictionary creation. Just cache it here.
        # https://github.com/huggingface/tokenizers/issues/483
        self._vocab = self.fast_tokenizer.vocab
        self._space_token = self.space_token()
        self._basic_chars_set = set(self.basic_characters())

    @property
    def tokenizer(self):
        """Return the test_relm's tokenizer."""
        return self.test_relm.tokenizer

    @property
    @functools.lru_cache(maxsize=1)
    def slow_tokenizer(self):
        """Return the test_relm's tokenizer."""
        if not self.tokenizer.is_fast:
            return self.tokenizer
        tokenizer_name = self.tokenizer.name_or_path
        logger.info(
            "Attemping to retrieave HuggingFace slow tokenizer: {}.".format(
                tokenizer_name)
        )
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    @property
    @functools.lru_cache(maxsize=1)
    def fast_tokenizer(self):
        """Return the test_relm's tokenizer."""
        if self.tokenizer.is_fast:
            return self.tokenizer
        tokenizer_name = self.tokenizer.name_or_path
        logger.info(
            "Attemping to retrieave HuggingFace slow tokenizer: {}.".format(
                tokenizer_name)
        )
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def vocab(self):
        """Return a copy of the vocabulary."""
        return self._vocab

    def vocab_size(self):
        """Return the size of the tokenizer vocabulary."""
        return self.tokenizer.vocab_size

    def basic_characters(self):
        """Return a sorted list of non-paired characters in the tokenizer."""
        return sorted([x for x in self._vocab.keys() if len(x) == 1])

    def unicode_characters(self, characters=None, return_ids=True):
        """Return a list of unicode characters.

        :param characters The set of particular characters to return.
        :param return_ids Return both keys and values
        """
        if characters is None:
            characters = [chr(c) for c in range(0, 0x10FFFF + 1)]
        new_characters = list(set(characters) -
                              set(_unicode_character_cache.keys()))
        if new_characters:
            toks = self.tokenizer.batch_encode_plus(
                new_characters,
                return_attention_mask=False,
                return_tensors=None,
                )["input_ids"]
            toks = list(map(lambda x: tuple(int(xx) for xx in x), toks))
        else:
            toks = []
        for c, t in zip(new_characters, toks):
            _unicode_character_cache[c] = t
        toks = map(lambda x: _unicode_character_cache[x], characters)
        if return_ids:
            char_toks = zip(characters, toks)
        else:
            char_toks = toks
        return char_toks

    def unicode_characters_map(self):
        """Return a lazy map of unicode characters."""
        characters = map(lambda c: chr(c), range(0, 0x10FFFF + 1))
        d = {}
        encoder_fn = functools.partial(self.tokenizer.encode_plus,
                                       return_attention_mask=False,
                                       return_tensors=None)

        def unpacked_encoder_fn(c):
            ret = encoder_fn(c)
            ret = ret["input_ids"][0]
            return tuple(int(x) for x in ret)

        for c in characters:
            d[c] = (unpacked_encoder_fn, (c,))
        char_toks = LazyDict(d)
        return char_toks

    def all_vocabulary_words(self):
        """Return a list of all words possible in the vocabulary."""
        return [k for k, v in sorted(
            self._vocab.items(), key=lambda item: item[1])]

    def space_char(self):
        """Return the special character representing whitespace."""
        return "Ġ"

    def newline_char(self):
        """Return the special character representing newline."""
        return "Ċ"

    def tab_char(self):
        """Return the special character representing tab."""
        return "ĉ"

    def space_token(self):
        """Return the numeric token ID of the space character."""
        return self.test_relm.words_to_tokens(" ")[0].item()

    def newline_token(self):
        """Return the numeric token ID of the newline character."""
        return self.test_relm.words_to_tokens("\n")[0].item()

    def tab_token(self):
        """Return the numeric token ID of the tab character."""
        return self.test_relm.words_to_tokens("\t")[0].item()

    def beginning_of_text_token(self):
        """Return the EOT token."""
        return self.tokenizer.bos_token_id

    def end_of_text_token(self):
        """Return the EOT token."""
        return self._vocab['<|endoftext|>']

    def with_space_vocab(self):
        """Return the keys,tokens that start with a space."""
        def starts_with_space(key):
            return key[0] == "Ġ"
        for k, v in self._vocab.items():
            if starts_with_space(k):
                yield k, v

    def without_space_vocab(self):
        """Return the keys,tokens that don't start with a space."""
        def not_starts_with_space(key):
            return key[0] != "Ġ"
        for k, v in self._vocab.items():
            if not_starts_with_space(k):
                yield k, v

    def map_complex_char_to_basic_char(self, char_str: str):
        """Convert a string into a list of characters."""
        _basic_chars = [c for c in char_str]
        assert all(c in self._basic_chars_set for c in _basic_chars)
        return _basic_chars

    def string_to_basic_chars_tokens(self, my_str: str):
        """Convert a string into a list of list of tokens."""
        str_tokens = self.test_relm.words_to_tokens(my_str)[0]
        basic_chars_toks_list = []
        for tok in str_tokens:
            tok_str = self.test_relm.tokens_to_words([tok]).strip()
            basic_chars = self.map_complex_char_to_basic_char(tok_str)
            basic_chars_toks = list(
                map(lambda x: self.test_relm.words_to_tokens(x).item(),
                    basic_chars))
            basic_chars_toks_list.append(basic_chars_toks)
        return basic_chars_toks_list

    def string_to_basic_chars_tokens_flat(self, my_str: str):
        """Convert a string into a flattened list of tokens.

        Same as above, but flattens the list of words into a list of
        characters, inserting spaces.
        """
        buffer = []
        basic_chars_toks_list = self.string_to_basic_chars_tokens(my_str)
        if len(my_str) and my_str[0] == " ":
            # TODO(mkuchnik): Fix extra spaces getting stripped
            buffer.append(self._space_token)
        for toks in basic_chars_toks_list:
            buffer.extend(toks)
            buffer.append(self._space_token)
        buffer.pop(-1)
        return buffer

    def bpe_encoding(self, text: str, apply_full_encoding: bool = False):
        """Map a word to BPE.

        Adapted from GPT-2 encoder.
        NOTE: This does not simplify the BPE using the full encoder unless
        apply_full_encoding is on.
        """
        bpe_tokens = []
        pat = self.slow_tokenizer.pat
        for token in re.findall(pat, text):
            encoded_toks = "".join(self.slow_tokenizer.byte_encoder[b] for b in
                                   token.encode("utf-8"))
            if apply_full_encoding:
                # Fuse and remap
                encoded_toks = self.slow_tokenizer.bpe(encoded_toks).split(" ")
            encoded_toks = [self.slow_tokenizer.encoder[t]
                            for t in encoded_toks]
            bpe_tokens.extend(encoded_toks)
        return bpe_tokens

    def generate_all_equivalent_substrings_for_word(self, word: str):
        """Map a word to a list of equivalent list of tokens.

        This is done by iterating over all partitions of the string, and
        checking if all partitions are mappable to the vocabulary. While this
        function can handle spaces, it is not recommended to feed in whole
        sentences because the complexity is O(2^n), where n is the length of
        the sentence. Rather, prefer to use
        generate_all_equivalent_substrings_for_sentence.

        @param word: A string representing a word
        @return: A set of list of integers
        """
        equivalent_substrings = []
        if len(word) > 15:
            logger.warning("Word '{}' is very long. Expect slowdown.".format(
                word))

        for p in _partitions(word):
            tokens = []
            for chars in p:
                tokenized_chars = self.tokenizer.tokenize(chars)
                if len(tokenized_chars) == 1:
                    # NOTE(mkuchnik): self.tokenizer.vocab[chars] depends on
                    # spaces not being present. We use the space-familiar
                    # wrapper. We are only interested in 1-to-1 relationships
                    # between strings and tokens. Therefore, if tokenize emits
                    # many tokens, we know that substring is not representable
                    # in the vocabulary without splitting it (which effectively
                    # makes it a different partition), so we drop it.
                    # Otherwise, we can use test_relm's words_to_tokens, and
                    # just use a set to remove duplicate tuples.
                    try:
                        tok = self._vocab[tokenized_chars[0]]
                    except KeyError:
                        tokens = None
                        break
                    tokens.append(tok)
                else:
                    tokens = None
                    break
            if tokens is not None:
                equivalent_substrings.append(tuple(tokens))

        return set(equivalent_substrings)

    def generate_all_equivalent_substrings_for_word_fast(self, word: str):
        """Map a word to a list of equivalent list of tokens.

        This is done by iterating over the dict with the word in DFS manner,
        resulting in O(m * n) complexity, where n is the number of chars and m
        is the number of tokens.

        @param word: A string representing a word
        @return: A set of list of integers
        """
        equivalent_substrings = []

        def clean_k(k):
            k = k.replace(self.space_char(), " ")  # usually G
            k = k.replace(self.newline_char(), "\n")  # usually C
            k = k.replace(self.tab_char(), "\t")  # usually c
            return k

        def clean_v(v):
            return v

        clean_vocab = {clean_k(k): clean_v(v) for k, v in self._vocab.items()}
        # TODO(mkuchnik): Replace with Trie to get O(n) complexity
        starts_with = {}
        for k in clean_vocab:
            try:
                starts_with[k[0]].append(k)
            except KeyError:
                starts_with[k[0]] = [k]

        def dfs(word, idx, tokens):
            if idx == len(word):
                equivalent_substrings.append(tuple(tokens))
                return
            elif idx > len(word):
                raise RuntimeError("Fell off word.")
            else:
                first_char = word[idx]
                relevant_keys = starts_with[first_char]
                remaining_chars = len(word) - idx
                for k in relevant_keys:
                    if len(k) <= remaining_chars:
                        is_match = True
                        for i, kk in enumerate(k):
                            if kk != word[idx + i]:
                                is_match = False
                                break
                        if is_match:
                            tokens.append(clean_vocab[k])
                            dfs(word, idx + len(k), tokens)
                            tokens.pop(-1)

        tokens = []
        idx = 0

        dfs(word, idx, tokens)

        return set(equivalent_substrings)

    def generate_all_equivalent_substrings_for_sentence(self, sentence: str,
                                                        fast: bool = False):
        """Map a sentence to a list of list of equivalent list of tokens.

        Wrapper around generate_all_equivalent_substrings_for_word for
        space-separated strings. Because sentences are split by space, this
        function can return all the sets for each word that are equivalent
        rather than enumering the much larger list of possible token strings.
        If the word length is bounded, we should expect linear complexity with
        respect to sentence, as each word will take up to some max time to
        process.

        @param word: A string representing a space-separated sentence
        @param fast: A bool to enable a faster transform
        @return: A list of list of list of integers
        """
        words = sentence.split(" ")
        list_of_tokens = []
        for i, word in enumerate(words):
            # TODO(mkuchnik): Properly fix all extra spaces getting stripped
            if i:
                word = " {}".format(word)
            if fast:
                word_reps = \
                    self.generate_all_equivalent_substrings_for_word_fast(
                        word)
            else:
                word_reps = self.generate_all_equivalent_substrings_for_word(
                    word)
            assert word_reps, \
                "{} returned empty reps: {}".format(word, word_reps)
            list_of_tokens.append(word_reps)
        return list_of_tokens
