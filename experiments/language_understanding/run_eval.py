"""Run experiments mirroring Section 3.3 of GPT2 paper."""

import argparse
import pandas as pd
import logging
import time
import functools
import collections
import pprint
import string
import re
import itertools
import tqdm
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import relm

SUPPORTED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def add_logger():
    """Attach logging to the script."""
    logger = relm.get_relm_logger()
    logger.setLevel(level=logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('lambada.log')
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
    return logger


def preprocess(text):
    """GPT-2 dataset processing."""
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("–", "-")
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace("é", "e")
    text = text.replace("—", "-")
    text = text.replace("\xa0", " ")
    text = text.replace("\u200a", "")
    text = text.replace("…", "...")
    text = text.replace("ñ", "n")
    return '\n'+text.strip()


def split_text(text):
    """Split text by newline."""
    xs = []
    ys = []
    for series in text:
        line_str = series.strip()
        words = line_str.split(" ")
        x = " ".join(words[:-1])
        y = words[-1]
        xs.append(x)
        ys.append(y)
    return xs, ys


def get_words():
    """Return the list of words on Unix systems."""
    with open("/usr/share/dict/words") as f:
        words = f.read().splitlines()
    return words


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


def read_data():
    """Read and process lambada."""
    f = "../../lambada_test.jsonl"
    if not pathlib.Path(f).exists():
        raise RuntimeError("Lambada dataset '{}' does not exist.".format(f))
    df = pd.read_json(f, lines=True)
    df["processed"] = df["text"].map(preprocess)
    text = df['processed']
    x, y = split_text(text)
    df["x"] = x
    df["y"] = y
    return df


def get_parser():
    """Return an argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS,
                        default="gpt2",
                        help="The model to benchmark.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="The top_k for sampling.")
    parser.add_argument("--static_minimize", action="store_true",
                        help="Statically minimize automata using encoding.")
    parser.add_argument("--add_eos_token", action="store_true",
                        help="Force EOS.")
    parser.add_argument("--remove_stop_words", action="store_true",
                        help="Remove stop words.")
    parser.add_argument("--force_context_words", action="store_true",
                        help="Only use words in context.")
    parser.add_argument("--max_results", type=int, default=None,
                        help="The number of results to run.")
    return parser


@functools.lru_cache(1)
def get_stop_words():
    """Return a set of stop words."""
    # Download stopwords data, just in case it's not downloaded
    import nltk
    nltk.download("stopwords")

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return stop_words


def run_query(model, tokenizer, query, max_samples, remove_stop_words):
    """Run a query with the model."""
    test_relm = relm.model_wrapper.TestableModel(model, tokenizer)
    print("Building query.")
    if remove_stop_words:
        stop_words = get_stop_words()
        print("stop_words", stop_words)
        print("him" in stop_words)
        stop_words_str = "|".join("({})".format(s) for s in stop_words)
        prefix = query.accept_str
        puncs = [".", "!", "?"]
        punctuation_str = ["({})".format(sanitize_query_str_rust(x)) for x in
                           puncs]
        punctuation_str = "({})".format("|".join(punctuation_str))
        suffix = punctuation_str
        filter_str = "{} ({})({})?(\")?".format(
            prefix, stop_words_str, suffix)
        print("Filter str: {}".format(filter_str))
        query.experimental_filter_str = filter_str
    ret = relm.search(model, tokenizer, query)
    ret = map(lambda x: (x, time.perf_counter()), ret)
    print("Executing query.")
    start_time = time.perf_counter()
    xs = []
    ts = []
    urls = []
    for x, t in ret:
        url = test_relm.tokens_to_words(x)
        if remove_stop_words:
            stop_words = get_stop_words()
            last_word = sentence_to_pred(url)
            print("url: {}".format(url))
            print("Last word: {}".format(last_word))
            if last_word in stop_words:
                raise RuntimeError("Found stop word: {}".format(last_word))
        xs.append([int(xx) for xx in x])
        ts.append(t - start_time)
        urls.append(url)
        if max_samples and len(xs) >= max_samples:
            break
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


def sentence_to_pred(sentence):
    """Convert a sentence into a predicted word string without punctuation."""
    pred = sentence.split(" ")[-1].translate(
        str.maketrans('', '', string.punctuation))
    return pred


def main():
    """Run the main function."""
    logger = add_logger()
    parser = get_parser()
    args = parser.parse_args()
    print("Args: {}".format(dict(vars(args))))
    df = read_data()

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, non_blocking=True)

    top_k = args.top_k
    max_samples = 1

    sample_iter = zip(df["x"], df["y"])
    all_results = []
    all_attempts_results = []

    ignore_set = {182, }

    if args.max_results:
        sample_iter = itertools.islice(sample_iter, args.max_results)
    sample_iter = list(sample_iter)
    sample_iter = tqdm.tqdm(sample_iter)

    for i, (x, y) in enumerate(sample_iter):
        logger.info("Starting processing sample {}: {} {}".format(i, x, y))
        if i in ignore_set:
            continue
        if args.max_results and i >= args.max_results:
            break
        print(i)
        prefixes = [x]
        queries = [" ".join([x, y])]
        puncs = [".", "!", "?"]
        punctuation_str = ["({})".format(sanitize_query_str_rust(x)) for x in
                           puncs]
        punctuation_str = "({})".format("|".join(punctuation_str))
        if args.force_context_words:
            words_used = set(re.findall(r'[\w]+', x))
            words_used_str = "({})".format(
                "|".join("({})".format(w) for w in words_used))
            word_str = " ({})({})?(\")?".format(words_used_str,
                                                punctuation_str)
        else:
            word_str = " ([a-zA-Z]+)({})?(\")?".format(punctuation_str)
        sanitized_test_strings = sanitize_query_str_rust(x) + word_str
        sanitized_test_strings = [sanitized_test_strings]
        # All but last word
        sanitized_prefix_strings = map(
            lambda x: "({})".format(
                sanitize_query_str_rust(x)
            ),
            prefixes,
        )
        sanitized_test_string = "|".join(sanitized_test_strings)
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
        query.experimental_advanced_parsing_static_minimize_prefix_only = True
        query.experimental_regex_backend = \
            relm.facade.RegexBackendType.RUST
        query.experimental_dijkstra = True
        query.experimental_dijkstra_beam_size = None
        query.experimental_penalized_accepted_probability = False
        query.experimental_avoid_not_accepted_probability = True
        query.experimental_fast_start = True
        query.experimental_very_fast_start = True
        query.experimental_add_eos_token = args.add_eos_token
        remove_stop_words = args.remove_stop_words
        start_time = time.perf_counter()
        start_time_s = time.time()
        msg = ""
        try:
            results = run_query(model, tokenizer, query, max_samples,
                                remove_stop_words)
        except Exception as ex:
            # TODO(mkuchnik): Add handling
            print(ex)
            results = {"urls": [],
                       "times": [],
                       "tokens": [],
                       }
            msg = str(ex)
        print(results)
        pred = sentence_to_pred(results["urls"][0])
        acc = pred == y
        print("Predicted {} vs {} ({})".format(pred, y, acc))
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
            "prediction": pred,
            "x": x,
            "y": y,
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
