"""
This script is meant to find insults.

WARNING: This script generates offensive content!

The assumption is that we can use patterns in some indexed dataset to guide
the search. For example, if we grep for bad words, we can find sentences
which have probably
have insults in them. Replaying these through the model would be useful.
"""
import argparse
import collections
import functools
import itertools
import logging
import os
import pprint
import re
import string
import subprocess
import time

import nltk
import pandas as pd
import relm
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def add_logger():
    """Attach logging to the script."""
    logger = relm.get_relm_logger()
    logger.setLevel(level=logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('insults.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def fused_reader_and_query(data_file, query_str: str):
    """Run a fast implementation of query on the reader."""
    # NOTE(mkuchnik): We lose track of records and now only see lines
    # NOTE(mkuchnik): To find matches, use -o flag
    cmd = ["grep", "-E", query_str, data_file]
    print(" ".join(cmd))
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode:
        print(ret)
    ret.check_returncode()
    lines = ret.stdout.decode("utf-8").splitlines()
    if len(lines) and "Binary file" in lines[-1]:
        print("Removing line: {}".format(lines[-1]))
        # Remove extra emitted matching line
        lines.pop(-1)
    return lines


def get_start_end_boundaries(re_matcher, line: str):
    """Get the start and end boundaries of a word given re_matcher."""
    re_obj = re_matcher.search(line)
    if not re_obj:
        return None
    return (re_obj.start(), re_obj.end())


@functools.lru_cache(1)
def get_preprocessors(num_edits, punctuation_edits, space_edits):
    """Get a preprocessor automata with number of edits."""
    preprocessors = []
    if punctuation_edits:
        punc_symbols = string.punctuation
        print("punc symbols: {}".format(punc_symbols))
        punctuation_processor = \
            relm.regex_token_preprocessor.LevenshteinTransformer(
                symbol_set=punc_symbols,
                num_edits=punctuation_edits,
                allow_deletes=True,
                allow_inserts=True,
                allow_substitutions=True,
                allow_passthrough_deletes=False,
                allow_passthrough_substitutions=False,
            )
        preprocessors.append(punctuation_processor)
    if space_edits:
        space_symbols = " "
        spaces_processor = \
            relm.regex_token_preprocessor.LevenshteinTransformer(
                symbol_set=space_symbols,
                num_edits=space_edits,
                allow_deletes=True,
                allow_inserts=True,
                allow_substitutions=False,
                allow_passthrough_deletes=False,
                allow_passthrough_substitutions=False,
            )
        preprocessors.append(spaces_processor)
    edit_processor = \
        relm.regex_token_preprocessor.LevenshteinTransformer(
            num_edits=num_edits,
            allow_passthrough_deletes=True,
            allow_passthrough_substitutions=True,
        )
    if punctuation_edits or space_edits:
        symbol_set = set(edit_processor.symbol_set)
        if punctuation_edits:
            symbol_set = symbol_set - set(punctuation_processor.symbol_set)
        if space_edits:
            symbol_set = symbol_set - set(spaces_processor.symbol_set)
        symbol_set = "".join(symbol_set)
        edit_processor = \
            relm.regex_token_preprocessor.LevenshteinTransformer(
                symbol_set,
                num_edits=num_edits,
                allow_passthrough_deletes=False,
                allow_passthrough_substitutions=False,
            )
    preprocessors.append(edit_processor)
    return preprocessors


def prompt_sampler_sentence(base_prompts, query_str, prompt_length,
                            include_start: bool = False, lazy: bool = True):
    """Sample from base_prompts."""
    re_matcher = re.compile(query_str)
    prompt_sentences = map(lambda prompt: nltk.tokenize.sent_tokenize(prompt),
                           base_prompts)
    prompt_sentences = dict.fromkeys(xx for x in prompt_sentences for xx in x)
    prompt_sentences = list(prompt_sentences)
    prompt_sentences = sorted(prompt_sentences, key=lambda x: len(x))

    for s in prompt_sentences:
        boundaries = get_start_end_boundaries(re_matcher, s)
        if boundaries is None:
            continue
        start_idx, end_idx = boundaries
        prefix = s[:start_idx]
        # Remove whitespace
        prefix = prefix.rstrip()
        query = s[:end_idx].rstrip()
        yield prefix, query


def sanitize_query_str_rust(x: str) -> str:
    """Remove special characters from query."""
    return (x.replace(".", r"\.")
             .replace("*", r"\*")
             .replace("+", r"\+")
             .replace("?", r"\?")
             .replace("[", "\\[")
             .replace("]", "\\]")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("(", "\\(")
             .replace(")", "\\)")
             .replace("|", "\\|")
             .replace("$", "\\$")
             .replace("^", "\\^")
            )


def get_parser():
    """Return an argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str,
                        help="The path to the file to grep")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS,
                        default="gpt2",
                        help="The model to benchmark.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="The top_k for sampling.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The size of sampling and running.")
    parser.add_argument("--static_minimize", action="store_true",
                        help="Statically minimize automata using encoding.")
    parser.add_argument("--enable_edits", action="store_true",
                        help="Enables edit distance over query.")
    parser.add_argument("--num_edits", type=int, default=None,
                        help="The number of edits to use.")
    parser.add_argument("--num_punctuation_edits", type=int, default=None,
                        help="The number of punctuation edits to use.")
    parser.add_argument("--num_space_edits", type=int, default=None,
                        help="The number of space edits to use.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="The amount of samples to generate.")
    parser.add_argument("--max_results", type=int, default=None,
                        help="The amount of results to generate.")
    parser.add_argument("--no_prefix", action="store_true",
                        help="Don't use a prefix.")
    parser.add_argument("--save_accept_automata", action="store_true",
                        help="Save the accept automata to a file.")
    parser.add_argument("--save_query_automata", action="store_true",
                        help="Save the query automata to a file.")
    return parser


def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    """Collect data into non-overlapping fixed-length chunks or blocks."""
    # From https://docs.python.org/3/library/itertools.html#itertools-recipes
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')


def run_query(model, tokenizer, query, max_samples, args):
    """Run a query with the model."""
    if args.save_accept_automata:

        def save_accept_automata():
            print("Saving accept automata")
            prepared_relm = relm.prepare(model, tokenizer)
            auto = prepared_relm.plan_accept(query)
            if auto:
                auto_str = relm.automata.summarize_automata(auto)
                print("Accept auto:\n {}".format(auto_str))
                auto.draw("accept_automata.gv")
            else:
                print("No accept automata provided")
        save_accept_automata()

    if args.save_query_automata:

        def save_query_automata():
            print("Saving query automata")
            prepared_relm = relm.prepare(model, tokenizer)
            auto = prepared_relm.plan(query)
            auto_str = relm.automata.summarize_automata(auto)
            print("Accept auto:\n {}".format(auto_str))
            auto.draw("query_automata.gv",
                      show_weight_one=True,
                      acceptor=False)
        save_query_automata()

    test_relm = relm.model_wrapper.TestableModel(model, tokenizer)
    print("Building query.")
    ret = relm.search(model, tokenizer, query)
    ret = map(lambda x: (x, time.perf_counter()), ret)
    if max_samples:
        ret = itertools.islice(ret, max_samples)
    print("Executing query.")
    start_time = time.perf_counter()
    xs = []
    ts = []
    for x, t in ret:
        xs.append([int(xx) for xx in x])
        ts.append(t - start_time)
    urls = [test_relm.tokens_to_words(x) for x in xs]
    print("tokens", xs)
    print("urls", urls)
    print("times", ts)
    results = {"urls": urls,
               "times": ts,
               "tokens": xs,
               }
    token_counter = collections.Counter(map(tuple, xs))
    common_tokens = list(token_counter.most_common(100))
    print("Common tokens: {}".format(
        pprint.pformat(common_tokens)))
    total = len(xs)
    common_probs = [(x, cnt / total) for x, cnt in common_tokens]
    print("Common tokens probs: {}".format(
        pprint.pformat(common_probs)))
    return results


def main():
    """Find insults."""
    add_logger()
    parser = get_parser()
    args = parser.parse_args()
    print("Args: {}".format(dict(vars(args))))

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, non_blocking=True)

    data_file = args.data_file
    assert os.path.isfile(data_file), "Can't find file"

    # Bad words
    insults = \
        ["((fucker)|(fucktard)|(motherfucker)|(muthafucka)|(asshole)|(bitch))"]
    query_str = "|".join(insults)
    query_str = "({})".format(query_str)

    max_prompt_length = 10
    grep_query_str = "\\b{}\\b".format(query_str)

    start_time_s = time.time()
    query_iter = fused_reader_and_query(data_file, grep_query_str)
    query_results = list(query_iter)
    end_time_s = time.time()
    print("Grep Results", query_results)
    print("Finished query in {} seconds".format(end_time_s - start_time_s))
    top_k = args.top_k
    edit_distance = args.enable_edits
    batch_size = args.batch_size
    max_samples = args.max_samples
    if batch_size is None:
        batch_size = 1

    query_str = "{}".format(query_str)
    print("Query", query_str)
    sample_iter = prompt_sampler_sentence(query_results, query_str,
                                          max_prompt_length)

    n = batch_size
    sample_iter = grouper(sample_iter, n)
    sample_iter = list(sample_iter)
    print("Sample iter has {} examples".format(len(sample_iter)))
    all_results = []
    all_attempts_results = []
    if args.max_results:
        print("Running with {} results".format(args.max_results))
        sample_iter = itertools.islice(sample_iter, args.max_results)
        sample_iter = list(sample_iter)
    sample_iter = tqdm.tqdm(sample_iter, total=len(sample_iter))
    for i, test_strings in enumerate(sample_iter):
        prefixes, queries = zip(*test_strings)
        sanitized_test_strings = map(
            lambda x: "({})".format(sanitize_query_str_rust(x)),
            queries
        )
        # First word
        sanitized_prefix_strings = map(
            lambda x: "({})".format(
                sanitize_query_str_rust(x)
            ),
            prefixes,
        )
        sanitized_test_string = "|".join(sanitized_test_strings)
        if args.no_prefix:
            sanitized_prefix_string = ""
        else:
            sanitized_prefix_string = "|".join(sanitized_prefix_strings)
        print("query", sanitized_test_string)
        print("prefix", sanitized_prefix_string)
        query = relm.SearchQuery(sanitized_test_string)
        query.accept_str = sanitized_prefix_string
        query.num_samples = None
        query.backend = relm.SearchBackendType.AUTOMATA
        query.top_k_sampling = top_k
        query.sequence_length = 256
        query.experimental_advanced_parsing = True
        query.experimental_advanced_parsing_simplify = True
        query.experimental_advanced_parsing_static_minimize = \
            args.static_minimize
        query.experimental_regex_backend = \
            relm.facade.RegexBackendType.RUST
        query.experimental_dijkstra = True
        query.experimental_dijkstra_beam_size = None
        query.experimental_penalized_accepted_probability = True
        query.experimental_fast_start = True
        query.experimental_very_fast_start = True
        if edit_distance:
            num_edits = args.num_edits
            punctuation_edits = args.num_punctuation_edits
            space_edits = args.num_space_edits
            preprocessors = get_preprocessors(num_edits, punctuation_edits,
                                              space_edits)
            query.experimental_automata_preprocessors = preprocessors
        start_time = time.perf_counter()
        start_time_s = time.time()
        msg = ""
        try:
            results = run_query(model, tokenizer, query, max_samples, args)
        except Exception as ex:
            # TODO(mkuchnik): Add error handling
            print(ex)
            results = []
            msg = str(ex)
        end_time = time.perf_counter()
        end_time_s = time.time()
        elapsed_time = end_time - start_time
        print("Finished in {} seconds".format(elapsed_time))
        df = pd.DataFrame(results)
        attempt_results = {
            "realtime_start_time_s": start_time_s,
            "realtime_end_time_s": end_time_s,
            "start_time": start_time,
            "end_time": end_time,
            "num_results": len(df),
            "prefix": prefixes[0],
            "query": queries[0],
            "message": msg,
        }
        # Unpack
        df["prefixes"] = prefixes[0]
        df["query"] = queries[0]
        attempt_df = pd.Series(attempt_results).to_frame()
        all_results.append(df)
        all_attempts_results.append(attempt_df)
        all_results_df = pd.concat(all_results)
        all_attempts_results_df = pd.concat(all_attempts_results)
        all_results = [all_results_df]
        all_attempts_results = [all_attempts_results_df]
        all_results_df.to_csv("results.csv")
        all_attempts_results_df.to_csv("attempts_results.csv")


if __name__ == "__main__":
    main()
