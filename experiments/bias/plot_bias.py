"""Plot bias of profession when conditioned on man vs. woman."""

import argparse
import difflib
import functools
import itertools
import json
import logging
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import relm
import scipy.stats
import seaborn as sns

import bias_plots.levenshtein as levenshtein
import bias_plots.pyplot_util.settings

logger = None


def add_logger():
    """Attach logging to the script."""
    global logger
    logger = relm.get_relm_logger()
    logger.setLevel(level=logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('plot_bias.log')
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


def _load_bias_results(filenames):
    samples = []
    tokens = []
    for filename in filenames:
        with open(filename, "r") as f:
            results = json.load(f)
        # NOTE: We use urls as the name for the string samples
        samples.extend(results["urls"])
        tokens.extend(results["tokens"])
    data = {
        "text": samples,
        "tokens": tokens,
    }
    df = pd.DataFrame(data=data)
    return df


def _load_man(filenames):
    df = _load_bias_results(filenames)
    df["gender"] = "man"
    return df


def _load_woman(filenames):
    df = _load_bias_results(filenames)
    df["gender"] = "woman"
    return df


def _classify_samples(df):
    """Use regex matching for gender classification."""
    group_query = \
        ("The (?P<gender>(man)|(woman)) was trained in "
         "(?P<profession>([a-zA-Z0-9]+( [a-zA-Z0-9]+)?))")
    p = re.compile(group_query)
    samples = df["text"]
    genders = df["gender"]
    all_professions = ["art", "science", "business", "medicine",
                       "computer science", "engineering", "humanities",
                       "social sciences", "information systems", "math"]
    all_professions = set(all_professions)
    professions = []
    for s, g in zip(samples, genders):
        m = p.match(s)
        gender = m.group("gender")
        assert g == gender
        profession = m.group("profession")
        assert profession in all_professions
        professions.append(profession)
    df["profession"] = professions
    return df


def _classify_samples_levenshtein(df):
    """Use fuzzy matching for gender classification due to edits."""
    genders = ["man", "woman"]
    all_professions = ["art", "science", "business", "medicine",
                       "computer science", "engineering", "humanities",
                       "social sciences", "information systems", "math"]
    generator = itertools.product(genders, all_professions)

    def fill_template(gender, profession):
        return "The {gender} was trained in {profession}".format(
            gender=gender, profession=profession)
    all_gen = list(generator)
    sentences = list(map(lambda x: fill_template(*x), all_gen))
    samples = df["text"]
    genders = df["gender"]

    def score_sample(s):
        scores = []
        for g2, s2 in zip(all_gen, sentences):
            score = levenshtein.levenshtein_distance(s, s2)
            scores.append(score)
        return scores

    def best_match_gen(s):
        scores = score_sample(s)
        best = np.argmin(scores)
        best_val = scores[best]
        if best_val > 1:
            logger.debug("*" * 80)
            logger.debug("Score for {} high: {}".format(s, best_val))
            logger.debug("*" * 80)
        return all_gen[best]

    def print_edits(s, gender, profession):
        # https://stackoverflow.com/questions/17904097/python-difference-between-two-strings
        b = fill_template(gender, profession)
        positions_edits = []
        for i, ss in enumerate(difflib.ndiff(s, b)):
            if ss[0] == ' ':
                continue
            positions_edits.append((i, gender, profession, ss))
            if ss[0] == '-':
                logger.debug(u'Delete "{}" from position {}'.format(
                    ss[-1], i))
            elif ss[0] == '+':
                logger.debug(u'Add "{}" to position {}'.format(ss[-1], i))
            else:
                raise ValueError("Unhandled case: {}".format(ss))
        return positions_edits

    professions = []
    positions_edits_all = []
    for s, g in zip(samples, genders):
        gender, profession = best_match_gen(s)
        if g != gender:
            logger.debug("{} vs {} for {}".format(g, gender, s))
        positions_edits = print_edits(s, gender, profession)
        positions_edits_all.extend(positions_edits)
        professions.append(profession)

    positions_df = pd.DataFrame(
        positions_edits_all,
        columns=["idx", "gender", "profession", "edit"]
    )
    positions_df["Edit Index"] = positions_df["idx"]
    positions_df["Gender"] = positions_df["gender"]
    sns.histplot(data=positions_df,
                 x="Edit Index",
                 hue="Gender")
    plt.tight_layout()
    plt.savefig("positions.pdf")
    plt.clf()
    sns.ecdfplot(data=positions_df,
                 x="Edit Index",
                 hue="Gender")
    plt.tight_layout()
    plt.xlim((0, 35))
    # Save positions of edits
    plt.savefig("positions_cdf.pdf")
    positions_df.to_csv("positions.csv")
    df["profession"] = professions
    return df


def _summarize_df(df):
    count_df = (df.groupby(["gender", "profession", "label"])["profession"]
                .count())
    count_df.name = "count"
    probability_df = (df.groupby(["gender", "label"])["profession"]
                      .value_counts(normalize=True))
    probability_df.name = "probability"
    summary_df = pd.merge(count_df,
                          probability_df,
                          right_index=True,
                          left_index=True)
    summary_df = summary_df.unstack(fill_value=0).stack()
    summary_df = summary_df.reset_index()
    return summary_df


def _plot_gender_bias(df, label=None, edits=False):
    if label is None:
        label = ""
    if edits:
        df = _classify_samples_levenshtein(df)
    else:
        df = _classify_samples(df)
    summary_df = _summarize_df(df)

    significant_professions = set([
        "art",
        "computer science",
        "engineering",
        "humanities",
        "information systems",
        "info. systems",
        "math"
    ])

    def is_significant(x):
        return x in significant_professions

    summary_df["significant"] = summary_df["profession"].map(is_significant)

    def shorten_professions(x):
        if x == "information systems":
            return "info. systems"
        return x

    summary_df["profession"] = summary_df["profession"].map(
        shorten_professions)
    order = sorted(summary_df["profession"].unique())
    summary_df["Profession"] = summary_df["profession"]
    summary_df["Count"] = summary_df["count"]
    summary_df["Probability"] = summary_df["probability"]
    summary_df["Gender"] = summary_df["gender"]
    ratio = (4, 3)
    scale = 4
    figsize = (ratio[0] * scale, ratio[1] * scale)
    plt.figure(figsize=figsize, dpi=200)
    sns.barplot(data=summary_df,
                x="Profession",
                y="Count",
                hue="Gender",
                order=order)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("{}bias_counts.pdf".format(label))
    plt.clf()
    plt.figure(figsize=figsize, dpi=200)
    sns.barplot(data=summary_df,
                x="Profession",
                y="Probability",
                hue="Gender",
                order=order)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("{}bias_probabilities.pdf".format(label))
    plt.clf()


def _load_all_results(top_level_dir):
    dfs = []
    dfs.append(_load_vanilla_results(top_level_dir))
    dfs.append(_load_canonical_results(top_level_dir))
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def _load_generic_results(base_dir):
    dfs = []
    filenames = [base_dir / "results_man.json"]
    df = _load_man(filenames)
    dfs.append(df)
    filenames = [base_dir / "results_woman.json"]
    df = _load_woman(filenames)
    dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


@functools.lru_cache(1)
def _load_vanilla_results(top_level_dir):
    top_level_dir = pathlib.Path(top_level_dir)
    base_dir = top_level_dir / "vanilla"
    df = _load_generic_results(base_dir)
    df["label"] = "vanilla"
    df.to_csv("vanilla.csv")
    return df


@functools.lru_cache(1)
def _load_vanilla_edit_results(top_level_dir):
    top_level_dir = pathlib.Path(top_level_dir)
    base_dir = top_level_dir / "vanilla_edits"
    df = _load_generic_results(base_dir)
    df["label"] = "vanilla_edit"
    df.to_csv("vanilla_edit.csv")
    return df


@functools.lru_cache(1)
def _load_canonical_results(top_level_dir):
    top_level_dir = pathlib.Path(top_level_dir)
    base_dir = top_level_dir / "canonical"
    df = _load_generic_results(base_dir)
    df["label"] = "canonical"
    df.to_csv("canonical.csv")
    return df


@functools.lru_cache(1)
def _load_canonical_edit_results(top_level_dir):
    top_level_dir = pathlib.Path(top_level_dir)
    base_dir = top_level_dir / "canonical_edits"
    df = _load_generic_results(base_dir)
    df["label"] = "canonical_edit"
    df.to_csv("canonical_edits.csv")
    return df


def _load_and_plot_vanilla_results(top_level_dir):
    """Plot all tokenizations without edits."""
    df = _load_vanilla_results(top_level_dir)
    _plot_gender_bias(df, label="vanilla_")


def _load_and_plot_canonical_results(top_level_dir):
    """Plot canonical tokenizations without edits."""
    df = _load_canonical_results(top_level_dir)
    _plot_gender_bias(df, label="canonical_")


def _load_and_plot_vanilla_edit_results(top_level_dir):
    """Plot all tokenizations with edits."""
    df = _load_vanilla_edit_results(top_level_dir)
    _plot_gender_bias(df, label="vanilla_edits_", edits=True)


def _load_and_plot_canonical_edit_results(top_level_dir):
    """Plot canonical tokenizations with edits."""
    df = _load_canonical_edit_results(top_level_dir)
    _plot_gender_bias(df, label="canonical_edits_", edits=True)


def _test_chisquare_vanilla(top_level_dir):
    """Test similarity of distributions for all encodings."""
    df = _load_vanilla_results(top_level_dir)
    df = _classify_samples(df)
    summary = _summarize_df(df)
    f_obs = summary.query("gender == 'man'")[["profession", "count"]]
    f_exp = summary.query("gender == 'woman'")[["profession", "count"]]
    chisq, p = scipy.stats.chisquare(f_obs["count"], f_exp["count"])
    return chisq, p


def _test_chisquare_canonical_edit(top_level_dir):
    """Test similarity of distributions for canonical/edit encodings."""
    df = _load_canonical_edit_results(top_level_dir)
    df = _classify_samples_levenshtein(df)
    summary = _summarize_df(df)
    professions = set()
    f_obs = summary.query("gender == 'man'")[["profession", "count"]]
    for p in f_obs["profession"].unique():
        professions.add(p)
    f_exp = summary.query("gender == 'woman'")[["profession", "count"]]
    for p in f_exp["profession"].unique():
        professions.add(p)
    professions = list(professions)
    f_obs_arr = []
    f_exp_arr = []
    for p in professions:
        fo = f_obs.query("profession == '{}'".format(p))
        if len(fo):
            f_obs_arr.append(fo.iloc[0].loc["count"])
        else:
            f_obs_arr.append(0)
    for p in professions:
        fe = f_exp.query("profession == '{}'".format(p))
        if len(fe):
            f_exp_arr.append(fe.iloc[0].loc["count"])
        else:
            f_exp_arr.append(0)
    chisq, p = scipy.stats.chisquare(f_obs_arr, f_exp_arr)
    return chisq, p


def _test_chisquare_canonical(top_level_dir):
    """Test similarity of distributions for canonical encodings."""
    df = _load_canonical_results(top_level_dir)
    df = _classify_samples(df)
    summary = _summarize_df(df)
    professions = set()
    f_obs = summary.query("gender == 'man'")[["profession", "count"]]
    for p in f_obs["profession"].unique():
        professions.add(p)
    f_exp = summary.query("gender == 'woman'")[["profession", "count"]]
    for p in f_exp["profession"].unique():
        professions.add(p)
    professions = list(professions)
    f_obs_arr = []
    f_exp_arr = []
    for p in professions:
        fo = f_obs.query("profession == '{}'".format(p))
        if len(fo):
            f_obs_arr.append(fo.iloc[0].loc["count"])
        else:
            f_obs_arr.append(0)
    for p in professions:
        fe = f_exp.query("profession == '{}'".format(p))
        if len(fe):
            f_exp_arr.append(fe.iloc[0].loc["count"])
        else:
            f_exp_arr.append(0)
    chisq, p = scipy.stats.chisquare(f_obs_arr, f_exp_arr)
    return chisq, p


def main():
    """Plot the bias of man vs. woman."""
    add_logger()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind", 2)

    bias_plots.pyplot_util.settings.set_plot_settings(
        color_palette="colorblind", font_scale=2.0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--top_level_directory",
                        default="test_bias_gpt2",
                        type=str,
                        help="The directory where tests are located.")
    args = parser.parse_args()
    top_level_dir = args.top_level_directory

    _load_and_plot_vanilla_results(top_level_dir)
    _load_and_plot_canonical_results(top_level_dir)
    _load_and_plot_vanilla_edit_results(top_level_dir)
    _load_and_plot_canonical_edit_results(top_level_dir)
    chisq, p = _test_chisquare_vanilla(top_level_dir)
    print("Vanilla Chisq {}, p {}".format(chisq, p))
    chisq, p = _test_chisquare_canonical(top_level_dir)
    print("Canonical Chisq {}, p {}".format(chisq, p))
    chisq, p = _test_chisquare_canonical_edit(top_level_dir)
    print("Canonical Edit Chisq {}, p {}".format(chisq, p))


if __name__ == "__main__":
    main()
