"""Wrapper for huggingface models and tokenizers.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import torch

from . import relm_logging

logging = relm_logging.get_logger()

DEFAULT_BATCH_SIZE = 128


class TestableModel(object):
    """A model that can be probed for testing."""

    def __init__(self, model, tokenizer):
        """Wrap a model and tokenizer."""
        # TODO(mkuchnik): Check that model is of type AutoModelForCasualLM
        self.model = model
        self.tokenizer = tokenizer

    def _perplexity_query(self, words):
        """Run a standard inference step and returns perplexities.

        Perplexity is:
        exp(sum(log(probs))), where the length of the sequence may also be used
        to normalized.

        Params:
        @words:  A string to run through the model

        Returns: A perplexity
        """
        # https://stackoverflow.com/questions/63543006/how-can-i-find-the-probability-of-a-sentence-using-gpt-2
        # TODO(mkuchnik): optimize
        tokens_tensor = self.tokenizer.encode(words, add_special_tokens=False,
                                              return_tensors="pt")
        return self._perplexity_query_tokens(tokens_tensor)

    def _perplexity_query_tokens(self, tokens_tensor, return_numpy=True):
        if tokens_tensor is None:
            raise ValueError("Expected tokens_tensor to be a tensor. Got "
                             "{}".format(type(tokens_tensor)))
        tokens_tensor = torch.tensor(tokens_tensor)
        tokens_tensor = tokens_tensor.to(self.device(), non_blocking=True)
        with torch.no_grad():
            # Get logits
            loss = self.model(tokens_tensor, labels=tokens_tensor)[0]
            perp = torch.exp(loss)
        if return_numpy:
            perp = perp.cpu().numpy()
        return perp

    def _perplexity_next_query_tokens(self, tokens_tensor, batch_size=None):
        if tokens_tensor is not None:
            tokens_tensor = tokens_tensor.to(self.device())
            if len(tokens_tensor.shape) != 2 or tokens_tensor.shape[0] != 1:
                raise ValueError("Expected tokens_tensor with shape (1, N)."
                                 " Found {}.".format(tokens_tensor.shape))
        else:
            tokens_tensor = None
        max_i = len(self.tokenizer.vocab)
        with torch.no_grad():
            # Get logits
            output = self.model(tokens_tensor,
                                labels=tokens_tensor,
                                use_cache=True)
            past = output.past_key_values
            del output
            losses = []
            tensors = torch.arange(0, max_i, device=self.device())[:, None]
            if batch_size is None:
                for i in range(len(tensors)):
                    batch_tensors = tensors[i, :]
                    output = self.model(batch_tensors,
                                        labels=batch_tensors,
                                        use_cache=False,
                                        past_key_values=past)
                    loss = output.loss
                    losses.append(loss)
            else:
                for batch_tensors in torch.split(tensors, batch_size):
                    output = self.model(batch_tensors,
                                        labels=batch_tensors,
                                        use_cache=False,
                                        past_key_values=past.expand(
                                            *batch_tensors.shape))
                    loss = output.loss
                    losses.append(loss)
            losses = map(lambda loss:
                         loss.cpu().detatch().numpy().flatten().tolist(),
                         losses)
            losses = [ll for loss in losses for ll in loss]
            loss = torch.tensor(losses)
            return np.exp(loss.cpu().detach().numpy())

    def words_to_tokens(self, words):
        """Map from words to tokens.

        :param words A string of words.
        :return A Tensor of the ids for each word.
        """
        input_ids = self.tokenizer(words, return_tensors="pt").input_ids
        return input_ids

    def tokens_to_words(self, tokens):
        """Map from tokens to words.

        :param tokens A tensor of tokens
        :return A string representing the tokens
        """
        return self._decode_gen_sequence(tokens)

    def point_query(self, words, return_all_probs: bool = False):
        """Map words to a list of probabilities.

        Runs a standard inference step and returns probabilities of each word.

        Params:
        @words:  A string to run through the model.
        @return_all_probs: Return n*m matrix of all probabilities.

        Returns: A probability for each token
        """
        if words is None:
            raise ValueError("Point query expects a string to calculate"
                             " next-word probabilities")
        input_ids = self.tokenizer(words, return_tensors="pt").input_ids
        return self.point_query_tokens(input_ids,
                                       return_all_probs=return_all_probs)

    def _convert_to_pytorch_tensor(self, a):
        a_t = torch.tensor(a, device=self.device())
        logging.debug("Implicitly converting {} to pytorch.".format(
            a))
        return a_t

    def point_query_tokens(self, input_ids, return_all_probs=False,
                           top_k=None):
        """Map tokens to a list of probabilities."""
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        if isinstance(input_ids, (tuple, list, np.ndarray)):
            input_ids = self._convert_to_pytorch_tensor(input_ids)
        if len(input_ids.shape) != 2:
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]
            else:
                raise ValueError("Expected input_ids shape of 1xN or N but got"
                                 " {}".format(input_ids.shape))
        elif input_ids.shape[0] != 1:
            raise ValueError("Expected input_ids shape of 1xN or N but got"
                             " {}".format(input_ids.shape))
        all_probs = []
        past = None
        # It's recommended to add BOS
        # https://github.com/huggingface/transformers/issues/3311
        # https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/generate_unconditional_samples.py#L60
        # However, HF only adds it when there isn't a prefix, which makes
        # generation inconsistent if the prefix is iteratively computed.
        # TODO(mkuchnik): Probably want to push this logic up
        missing_bos = (input_ids[0, 0] != self.tokenizer.bos_token_id)
        if missing_bos:
            bos_tensor = \
                torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device())
            input_ids = torch.cat([bos_tensor, input_ids], dim=1)
            logging.info(
                "Missing BOS in query. Pushing to front of {}.".format(
                    input_ids.shape))
        for i in range(input_ids.shape[1]-1):
            prefix = input_ids[:, i].view(-1, 1)
            # Probabilities over all words
            probs, past = self._simple_next_token_query_tokens(
                input_ids=prefix, use_cache=True, past_key_values=past,
                top_k=top_k)
            assert probs.shape[0] == 1, \
                "Expected probs shape[0]==1, got {}".format(probs.shape)
            assert probs.shape[1] == 1, \
                "Expected probs shape[1]==1, got {}".format(probs.shape)
            if not return_all_probs:
                # Probability over the desired word
                next_word_idx = input_ids[0, i+1]
                gen_probs = probs[0, 0, next_word_idx]
            else:
                gen_probs = probs
            all_probs.append(gen_probs)
        return np.array(all_probs)

    def _simple_next_token_query(self, words):
        """Map words to a list of probabilities.

        Runs a standard inference step and returns probabilities of next
        token

        This should return identical output to next_token_query.

        Params:
        @words:  A string to run through the model

        Returns: A probability for next token
        """
        input_ids = self.tokenizer(words, return_tensors="pt").input_ids
        return self._simple_next_token_query_tokens(input_ids)

    def _simple_next_token_query_tokens(self, input_ids, return_numpy=True,
                                        top_k=None, temperature=None,
                                        validate=False, use_cache=None,
                                        past_key_values=None,
                                        attention_mask=None):
        """Map tokens to a list of probabilities.

        Runs a standard inference step and returns probabilities of next
        token

        This should return identical output to next_token_query.

        Params:
        @input_ids: The ids of tokens to predict

        Returns: A probability for next token
        """
        if attention_mask is not None:
            assert input_ids.shape == attention_mask.shape, \
                ("Input ids have shape {}"
                 " but attention_mask has shape {}".format(
                     input_ids.shape, attention_mask.shape))
        # NOTE(mkuchnik): attention mask may not used for GPT-2 like models
        # because the model is causal. Simply ignore unattended outputs.
        # https://github.com/huggingface/transformers/issues/808
        # Otherwise, there is a patch to left-pad
        # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517
        with torch.no_grad():
            output = self.model(input_ids, use_cache=use_cache,
                                past_key_values=past_key_values,
                                attention_mask=attention_mask)
            # Get logits
            logits = output[0]
            if use_cache:
                # Get past
                past = output[1]
            if temperature:
                logits = logits / temperature
            probs = logits.softmax(-1)
            if validate:
                if not torch.any(probs >= 0):
                    logging.warning("Probs negative: {}".format(
                        probs[probs < 0]))
                if not torch.sum(probs) <= 1.01:
                    logging.warning("Probs don't sum to 1: {}".format(
                        torch.sum(probs)))
            if top_k:
                _axis = len(probs.shape) - 1
                top_k_val = torch.topk(probs, axis=_axis, k=top_k)
                probs[:] = 0.
                probs = probs.scatter(_axis,
                                      top_k_val.indices,
                                      top_k_val.values)
                # Normalize
                probs /= torch.sum(probs)
            if return_numpy:
                probs = probs.cpu().detach().numpy()
        if use_cache:
            return probs, past
        else:
            return probs

    def _next_token_query_tokens(self, input_ids, return_numpy=True,
                                 top_k=None, temperature=None, validate=False):
        """Map a token input to a probability over the next token.

        Runs a standard inference step and returns probabilities of next
        token

        Params:
        @input_ids: The ids of tokens to predict

        Returns: A probability for next token
        """
        # TODO(mkuchnik): Optimize
        # https://stackoverflow.com/questions/62703391/estimate-token-probability-logits-given-a-sentence-without-computing-the-entire
        # https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py
        if input_ids is not None and len(input_ids.shape) != 2:
            raise ValueError("Expected input_ids to have size 2. Got "
                             "{}".format(input_ids.shape))
        with torch.no_grad():
            generated_outputs = self.model.generate(
                input_ids, do_sample=False, num_return_sequences=1,
                output_scores=True,
                max_new_tokens=1,
                num_beams=1,
                early_stopping=False,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            # TODO(mkuchnik): Assumes return_dict_in_generate=True
            probs = torch.stack(generated_outputs["scores"], dim=1)
            if temperature is not None:
                probs = probs / temperature
            probs = probs.softmax(-1).squeeze()
            if validate:
                all_valid_probs = torch.all((probs >= 0).bool()).item()
                sum_probs = torch.sum(probs).item()
                assert all_valid_probs and sum_probs <= 1.01
            if return_numpy:
                probs = probs.cpu().detach().numpy()
            # TODO(mkuchnik): Return tokens, too
            return probs

    def next_token_query(self, words):
        """Return probabilities of next token.

        Params:
        @words:  A string to run through the model

        Returns: A probability for next token
        """
        input_ids = self.tokenizer(words, return_tensors="pt").input_ids
        return self._next_token_query_tokens(input_ids)

    def _decode_gen_sequence(self, gen_sequence):
        return self.tokenizer.decode(gen_sequence, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def _batch_decode_gen_sequence(self, gen_sequence):
        return self.tokenizer.batch_decode(gen_sequence,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)

    def sample(self, prefix=None, num_samples=None, return_probabilities=False,
               return_tokens=False,
               do_sample=None, max_length=None, num_beams=None, top_k=None):
        """Return list of string samples of size num_samples.

        If prefix is
        provided, all returned samples will have common prefix.
        """
        if not prefix:
            input_ids = None
        else:
            input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids
        if do_sample is None:
            do_sample = True

        num_samples = 1 if not num_samples else num_samples
        num_beams = 1 if num_beams is None else num_beams

        if input_ids is not None:
            logging.debug("Transfering input to device: {}".format(
                self.device()))
            # Transfer inputs to same device as model
            input_ids = input_ids.to(self.device())

        # TODO(mkuchnik): Investigate early_stopping
        # https://huggingface.co/blog/how-to-generate
        generated_outputs = self.model.generate(
            input_ids,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            output_scores=bool(return_probabilities),
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        output_tokens_list = generated_outputs["sequences"]
        text_outputs = self._batch_decode_gen_sequence(output_tokens_list)

        ret = [text_outputs]

        if return_probabilities:
            # https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
            gen_sequences = generated_outputs["sequences"]
            gen_sequences = gen_sequences[:, input_ids.shape[-1]:]
            probs = torch.stack(generated_outputs["scores"], dim=1).double()
            # NOTE(mkuchnik): watch out for numerical problems
            probs = probs.softmax(-1)
            gen_probs = torch.gather(
                probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            # Now to get the actual probs, take their product
            # unique_prob_per_sequence = gen_probs.prod(-1) #noqa
            # We return unreduced prods because they are usually near 0
            ret.append(gen_probs)

        if return_tokens:
            ret.append(output_tokens_list)

        if len(ret) == 1:
            ret = ret[0]

        return ret

    def sample_iterator(self, prefix, batch_size=None,
                        return_probabilities=False,
                        return_tokens=False,
                        max_length=None,
                        num_beams=None,
                        top_k=None):
        """Return an iterator over samples from the model.

        Hides inference work behind an iterator. Automatically batches work
        and unbatches when returning iterator.
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
            # TODO(mkuchnik): Automatic batch-size tuning.
        while True:
            # TODO(mkuchnik): prefetch outputs
            outputs = self.sample(prefix, num_samples=batch_size,
                                  return_probabilities=return_probabilities,
                                  return_tokens=return_tokens,
                                  max_length=max_length,
                                  num_beams=num_beams,
                                  top_k=top_k)
            logging.debug("Generated outputs")
            if not return_probabilities and not return_tokens:
                yield from iter(outputs)
            else:
                yield from zip(*outputs)

    def device(self):
        """Return the pytorch device the model is located on."""
        return next(self.model.parameters()).device

    def __str__(self):
        """Return a string representation of the model wrapper."""
        return "TestableModel with {}".format(str(self.model))
