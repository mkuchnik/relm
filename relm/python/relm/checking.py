"""Check if a model can generate a string.

This module is meant to check that a model can generate the corresponding
string. Arbitrary queries are not supported; the string must be fixed.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

import relm.indexing
import relm.relm_logging
import relm.text_encodings

logger = relm.relm_logging.get_logger()


@dataclass
class HFGenerateOptions:
    """Flags from HuggingFace generate."""

    num_beams: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None
    bad_words_ids: Optional[Iterable[int]] = None
    force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] \
        = None
    length_penalty: Optional[float] = None
    decoder_start_token_id: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    prefix_allowed_tokens_fn: \
        Optional[Callable[[int, torch.Tensor], List[int]]] = None
    exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None


def can_generate_tokens(test_relm, test_tokens, prefix=None,
                        options: Optional[HFGenerateOptions] = None):
    """Return True if the model can generate the tokens.

    Return False if the model can't generate the tokens.
    """
    return _can_generate_tokens(test_relm, test_tokens, prefix, options)


def _is_array(maybe_array) -> bool:
    return isinstance(maybe_array, (list, tuple, np.ndarray, torch.Tensor))


def _is_1d_array(maybe_array) -> bool:
    if _is_array(maybe_array):
        if isinstance(maybe_array, (torch.Tensor, np.ndarray)):
            return len(maybe_array.shape) == 1
        else:
            return all(not _is_array(x) for x in maybe_array)
    else:
        return False


def _can_generate_tokens(test_relm, test_tokens, prefix=None,
                         options: Optional[HFGenerateOptions] = None,
                         model_cache=None):
    if options.num_beams is not None:
        raise NotImplementedError("Num beams is not supported")
    if options.temperature is not None:
        raise NotImplementedError("Temperature is not supported")
    if options.top_k is not None:
        top_k_sampling = options.top_k
    else:
        top_k_sampling = None
    if options.top_p is not None:
        raise NotImplementedError("Top_p is not supported")
    if options.bad_words_ids is not None:
        raise NotImplementedError("bad_words_ids is not supported")
    if options.force_words_ids is not None:
        raise NotImplementedError("force_words_ids is not supported")
    if options.length_penalty is not None:
        raise NotImplementedError("length_penalty is not supported")
    if options.decoder_start_token_id is not None:
        raise NotImplementedError("decoder_start_token_id is not supported")
    if options.num_beam_groups is not None:
        raise NotImplementedError("num_beam_groups is not supported")
    if options.diversity_penalty is not None:
        raise NotImplementedError("diversity_penalty is not supported")
    if options.prefix_allowed_tokens_fn is not None:
        raise NotImplementedError("prefix_allowed_tokens_fn is not supported")
    if options.exponential_decay_length_penalty is not None:
        raise NotImplementedError(
            "exponential_decay_length_penalty is not supported")

    if not _is_1d_array(test_tokens):
        raise ValueError(
            "Expected test_tokens to be 1d array. Got {} of type {}.".format(
                test_tokens, type(test_tokens))
        )

    if prefix is None:
        if test_tokens[0] == test_relm.tokenizer.bos_token_id:
            prefix = tuple()
        else:
            logger.info("Adding bos token to {}".format(test_tokens))
            prefix = (test_relm.tokenizer.bos_token_id,)
    else:
        prefix = tuple(prefix)
        if not len(prefix) or prefix[0] != test_relm.tokenizer.bos_token_id:
            # Add BOS if Null or missing
            prefix = (test_relm.tokenizer.bos_token_id,) + prefix
    num_bos = sum(1 for x in prefix
                  if x == test_relm.tokenizer.bos_token_id)
    if num_bos != 1:
        logger.warning("Found {} bos in prefix.".format(num_bos))

    test_tokens = prefix + tuple(test_tokens)
    test_tokens = torch.tensor(test_tokens,
                               dtype=torch.long,
                               device=test_relm.device())
    num_bos = sum(1 for x in test_tokens
                  if x == test_relm.tokenizer.bos_token_id)
    if num_bos != 1:
        logger.warning("Found {}!=1 bos in test_tokens.".format(num_bos))
    prefix_length = len(prefix)
    past = None
    use_cache = True
    logger.info("Test tokens: {} (prefix='{}' (length: {}))".format(
        test_tokens, prefix, prefix_length))

    for i in range(len(test_tokens) - 1):
        state_tokens = test_tokens[:i + 1]  # up to n
        next_token = test_tokens[i + 1]  # n + 1
        state_tokens_tup = tuple(state_tokens)
        eval_state_tokens_tup = state_tokens_tup + (next_token,)
        # NOTE(mkuchnik): First prefix length tokens are prefix
        in_suffix = i >= (prefix_length - 1)
        logger.info("State tokens tup i={}/{}: {} predicting: {}."
                    " In suffix: {}".format(
                        i, len(test_tokens),
                        state_tokens_tup, next_token, in_suffix,
                    ))
        if model_cache is not None and eval_state_tokens_tup in model_cache:
            # Cache hit
            token_rank = model_cache[eval_state_tokens_tup]
            if all(state_tokens == test_relm.tokenizer.bos_token_id):
                # NOTE(mkuchnik): We shouldn't reject BOS initial tokens
                # TODO(mkuchnik): We should have BOS as an explicit prefix
                model_contains_path = True
            elif top_k_sampling is not None and in_suffix:
                model_contains_path = token_rank <= top_k_sampling
            else:
                model_contains_path = True
            if not model_contains_path:
                logger.info("Rejecting {} by cache (k={})".format(
                    eval_state_tokens_tup, token_rank))
            else:
                logger.info("Accepting {} by cache (k={})".format(
                    eval_state_tokens_tup, token_rank))
            # NOTE(mkuchnik): We lost past state by loading from cache.
            # However, we can early exit if the cache hit is negative.
            if not model_contains_path:
                return False
        torch_state_tokens = state_tokens[-1:].clone().detach().to(
            test_relm.device())
        transition_probabilities, past = (
            test_relm._simple_next_token_query_tokens(
                torch_state_tokens, use_cache=use_cache,
                past_key_values=past))
        # TODO(mkuchnik): Replace with torch primitive
        token_rank = relm.indexing.get_element_rank(
            transition_probabilities, next_token, reverse=True)
        if top_k_sampling is not None and in_suffix:
            token_ranks = relm.indexing.rank_array(
                transition_probabilities, reverse=True)
            # NOTE(mkuchnik): We can't reject bos
            filtered_idxs = token_ranks > top_k_sampling
            assert token_ranks[next_token] == token_rank, \
                "Token rank inconsistency. Got {} vs. {}".format(
                    token_ranks[next_token],
                    token_rank)
            model_contains_path = token_rank <= top_k_sampling
            assert np.sum(~filtered_idxs) == top_k_sampling, \
                "Expected to find k={} samples, found {}".format(
                    top_k_sampling, np.sum(~filtered_idxs))
            if all(state_tokens == test_relm.tokenizer.bos_token_id):
                # NOTE(mkuchnik): We shouldn't reject BOS initial token
                if next_token == test_relm.tokenizer.bos_token_id:
                    logger.info("Accepting {} because starting BOS".format(
                        state_tokens))
                    model_contains_path = True
            logger.debug("Token {} rank: {}".format(
                next_token, token_rank))
            if not model_contains_path:
                logger.info("Rejecting {} because k={}/{}".format(
                    state_tokens, token_rank, top_k_sampling))
                closest_accepted_rank_mask = token_ranks == top_k_sampling
                closest_accepted_rank_idx = np.where(
                    closest_accepted_rank_mask)
                assert transition_probabilities.shape[0] == 1
                closest_prob = \
                    transition_probabilities[0, closest_accepted_rank_idx]
                this_prob = \
                    transition_probabilities[0, next_token]
                closest_diff = closest_prob - this_prob
                logger.info("Diff with rank {} element".format(closest_diff))

        else:
            logger.debug("Prefix (i={}) hit.".format(i))
            model_contains_path = True
        if model_cache is not None:
            logger.info("Setting cache {} to {}".format(
                eval_state_tokens_tup, model_contains_path))
            model_cache[eval_state_tokens_tup] = \
                token_rank
        if not model_contains_path:
            return False
    return True


def can_generate_string(test_relm, test_string, prefix=None,
                        options: Optional[HFGenerateOptions] = None,
                        check_standard_encoding_only=True):
    """Return True if the model can generate the string.

    Return False if the
    model can't generate the string. Raise exception if check can't complete
    for some reason (e.g., timeout).

    In other words, if you generate a string from a model, that string
    will return 'True', but a totally random string will likely be 'False'.
    Ideally, we'd return a certificate of generation.

    We want to support all common generation methods:
    https://huggingface.co/docs/transformers/internal/generation_utils

    Set check_standard_encoding_only to False to check all possible encodings.
    """
    if check_standard_encoding_only:
        sample_sentence_tokens = \
            test_relm.words_to_tokens(test_string).cpu().numpy().tolist()[0]
        sample_sentence_tokens = tuple(sample_sentence_tokens)
        sample_sentence_tokens = [sample_sentence_tokens]
        model_cache = None  # NOTE(mkuchnik): Cache not useful for 1
    else:
        encoding_generator = relm.text_encodings.TextEncodingsGenerator(
            test_relm)
        list_of_token_reps = (encoding_generator
                              .generate_all_equivalent_substrings_for_sentence(
                                  test_string, fast=True))
        list_of_token_reps = list(list_of_token_reps)  # Remove generator
        logger.debug("Generator tokens '{}'".format(
            list_of_token_reps))
        sample_sentence_tokens = itertools.product(*list_of_token_reps)
        sample_sentence_tokens = map(lambda s: [x for tup in s for x in tup],
                                     sample_sentence_tokens)
        model_cache = dict()

    batch_decode = False
    # NOTE(mkuchnik): We disable batch decode because it may create too many
    # sentences in memory and cause OOM

    if batch_decode:
        sample_sentence_tokens = list(sample_sentence_tokens)
        decoded_sentences = test_relm._batch_decode_gen_sequence(
            sample_sentence_tokens)
    else:
        decoded_sentences = map(lambda x: test_relm.tokens_to_words(x),
                                sample_sentence_tokens)

    prefix_tokens = None
    if prefix:
        prefix_tokens = test_relm.words_to_tokens(prefix)

    for s, decoded_s in zip(sample_sentence_tokens, decoded_sentences):
        assert decoded_s == test_string, \
            "Expected {} got {}".format(test_string, decoded_s)
        if _can_generate_tokens(test_relm, s, prefix_tokens, options,
                                model_cache=model_cache):
            return s
    return None
