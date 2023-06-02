"""Tests for relm model wrapper.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import relm.model_wrapper


class TestModelWrapper(unittest.TestCase):
    """Test that the wrapper is equivalent to hand-written functions."""

    def setUp(self):
        """Initialize model."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (AutoModelForCausalLM.from_pretrained("gpt2")
                      .to(self.device))

    def test_sampling(self):
        """Test that the next word is predicted correctly."""
        test_relm = relm.model_wrapper.TestableModel(self.model,
                                                     self.tokenizer)
        max_length = 10
        samples, tokens = test_relm.sample(
            num_samples=1,
            return_tokens=True,
            do_sample=False,
            max_length=max_length)
        tokens = tokens.cpu().numpy().tolist()
        self.assertEqual(len(samples), 1)
        self.assertTrue(isinstance(samples[0], str))
        greedy1 = self.greedy_samples(max_length)
        greedy2 = self.greedy_samples_cached(max_length)
        self.assertEqual(greedy1, greedy2)
        self.assertEqual(greedy1, tokens)
        relm_greedy = self.relm_greedy_samples(test_relm, max_length)
        self.assertEqual(greedy1, relm_greedy)
        relm_greedy_cached = self.relm_greedy_samples_cached(
            test_relm, max_length)
        self.assertEqual(relm_greedy, relm_greedy_cached)
        relm_greedy2 = self.relm_greedy_samples2(test_relm, max_length)
        self.assertEqual(greedy1, relm_greedy2)
        relm_greedy2_none = self.relm_greedy_samples2_none(
            test_relm, max_length)
        self.assertEqual(greedy2, relm_greedy2_none)
        relm_greedy_topk = self.relm_greedy_samples(test_relm, max_length,
                                                    top_k=1)
        self.assertEqual(greedy1, relm_greedy_topk)
        relm_greedy_cached = self.relm_greedy_samples_cached(
            test_relm, max_length, top_k=1)
        self.assertEqual(relm_greedy, relm_greedy_cached)
        relm_greedy2_topk = self.relm_greedy_samples2(test_relm, max_length,
                                                      top_k=1)
        self.assertEqual(greedy1, relm_greedy2_topk)
        relm_greedy2_none_topk = self.relm_greedy_samples2_none(
            test_relm, max_length, top_k=1)
        self.assertEqual(greedy1, relm_greedy2_none_topk)
        relm_greedy_topk = self.relm_greedy_samples(test_relm, max_length,
                                                    top_k=1, temperature=0.7)
        self.assertEqual(greedy1, relm_greedy_topk)
        relm_greedy_cached = self.relm_greedy_samples_cached(
            test_relm, max_length, top_k=1, temperature=0.7)
        self.assertEqual(relm_greedy, relm_greedy_cached)
        relm_greedy2_topk = self.relm_greedy_samples2(test_relm, max_length,
                                                      top_k=1, temperature=0.7)
        self.assertEqual(greedy1, relm_greedy2_topk)
        relm_greedy2_none_topk = self.relm_greedy_samples2_none(
            test_relm, max_length, top_k=1, temperature=0.7)
        self.assertEqual(greedy1, relm_greedy2_none_topk)

    def greedy_samples(self, max_len):
        """Sample slowly with greedy sampling.

        Return 1x1 list.
        """
        tokens = [self.tokenizer.bos_token_id]
        for i in range(max_len - 1):
            _inp = torch.tensor(tokens, device=self.device)
            with torch.no_grad():
                out = self.model.forward(input_ids=_inp)
            pred = out.logits
            tok = torch.argmax(pred[-1, :], axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]

    def greedy_samples_cached(self, max_len):
        """Sample faster with cached sampling.

        Return 1x1 list.
        """
        tokens = [self.tokenizer.bos_token_id]
        past = None
        tok = torch.tensor(tokens[-1], device=self.device)
        for i in range(max_len - 1):
            _inp = tok.unsqueeze(0)
            with torch.no_grad():
                out = self.model.forward(input_ids=_inp, use_cache=True,
                                         past_key_values=past)
            past = out.past_key_values
            pred = out.logits
            tok = torch.argmax(pred[-1, :], axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]

    def relm_greedy_samples(self, test_relm, max_len, top_k=None,
                            temperature=None):
        """Sample via test_relm next word.

        Return 1x1 list.
        """
        tokens = [self.tokenizer.bos_token_id]
        for i in range(max_len - 1):
            _inp = torch.tensor(tokens, device=self.device)
            pred = test_relm._simple_next_token_query_tokens(
                _inp, return_numpy=False, top_k=top_k, temperature=temperature)
            tok = torch.argmax(pred[-1, :], axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]

    def relm_greedy_samples_cached(self, test_relm, max_len, top_k=None,
                                   temperature=None):
        """Sample via test_relm next word with caching.

        Return 1x1 list.
        """
        tokens = [self.tokenizer.bos_token_id]
        past = None
        for i in range(max_len - 1):
            _inp = torch.tensor(tokens[-1:], device=self.device)
            pred, past = test_relm._simple_next_token_query_tokens(
                _inp, return_numpy=False, top_k=top_k, temperature=temperature,
                use_cache=True, past_key_values=past,
            )
            tok = torch.argmax(pred[-1, :], axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]

    def relm_greedy_samples2_none(self, test_relm, max_len, top_k=None,
                                  temperature=None):
        """Sample via test_relm next word with non-simple implementation.

        Uses None rather than bos.
        Return 1x1 list.
        """
        tokens = []
        for i in range(max_len - 1):
            if i:
                _inp = torch.tensor(tokens, device=self.device)
                _inp = _inp.unsqueeze(0)
            else:
                _inp = None
                tokens.append(self.tokenizer.bos_token_id)
            pred = test_relm._next_token_query_tokens(
                _inp, return_numpy=False, top_k=top_k, temperature=temperature)
            tok = torch.argmax(pred, axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]

    def relm_greedy_samples2(self, test_relm, max_len, top_k=None,
                             temperature=None):
        """Sample via test_relm next word with non-simple implementation.

        Return 1x1 list.
        """
        tokens = [self.tokenizer.bos_token_id]
        for i in range(max_len - 1):
            _inp = torch.tensor(tokens, device=self.device)
            _inp = _inp.unsqueeze(0)
            pred = test_relm._next_token_query_tokens(
                _inp, return_numpy=False, top_k=top_k, temperature=temperature)
            tok = torch.argmax(pred, axis=0)
            tokens.append(tok)

        def map_to_int(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().item()
            return x
        tokens = list(map(map_to_int, tokens))
        return [tokens]
