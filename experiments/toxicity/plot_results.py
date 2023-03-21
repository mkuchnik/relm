"""Plotting code."""

import matplotlib.pyplot as plt
import pandas as pd
import relm
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

import convert_attempts
import pyplot_util.settings

pyplot_util.settings.set_plot_settings(color_palette="colorblind",
                                       n_colors=2,
                                       font_scale=2)
ratio = (13, 7)
scale = 2
figsize = (ratio[0] * scale, ratio[1] * scale)
plt.figure(figsize=figsize, dpi=200)
kwargs = {
    "linewidth": 6.0,
}

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, return_dict_in_generate=True,
    pad_token_id=tokenizer.eos_token_id)
model.eval()
test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


top_level_dir = "test_insults_gpt2-xl"
max_num_attempts = 185


def process_attempts_df(attempts_df):
    """Preprocess results."""
    print(attempts_df)
    print(attempts_df.dtypes)
    attempts_df["normalized_start_time"] = \
        attempts_df["start_time"] - attempts_df["start_time"].min()
    attempts_df["cumulative_results"] = \
        attempts_df["num_results"].cumsum()
    attempts_df["query_len"] = \
        attempts_df["query"].map(lambda x: len(x))
    attempts_df["extract_success"] = \
        attempts_df["num_results"] > 0
    attempts_df["Normalized Start Time (min)"] = \
        attempts_df["normalized_start_time"] / 60
    attempts_df["Cumulative Number of Results"] = \
        attempts_df["cumulative_results"]
    attempts_df["cumulative_attempts"] = \
        attempts_df["num_results"].map(lambda x: 1).cumsum()
    attempts_df["Cumulative Number of Attempts"] = \
        attempts_df["cumulative_attempts"]
    return attempts_df


def load_prompted_results():
    """Load results that do use prefix."""
    baseline_attempts_df = convert_attempts.convert_attempts_results_to_df(
        "{}/baseline/attempts_results.csv".format(
            top_level_dir,
        )
    )
    baseline_attempts_df = process_attempts_df(baseline_attempts_df)
    baseline_attempts_df["Type"] = "Baseline"
    standard_attempts_df = convert_attempts.convert_attempts_results_to_df(
        "{}/standard/attempts_results.csv".format(
            top_level_dir,
        )
    )
    standard_attempts_df = process_attempts_df(standard_attempts_df)
    standard_attempts_df["Type"] = "ReLM"
    joined_attempts_df = pd.concat([standard_attempts_df,
                                    baseline_attempts_df])
    return joined_attempts_df


def load_unprompted_results():
    """Load results that don't use prefix."""
    df = pd.read_csv(
        "{}/standard_noprefix/results.csv".format(
            top_level_dir,
        )
    )
    attempts_df = convert_attempts.convert_attempts_results_to_df(
        "{}/standard_noprefix/attempts_results.csv".format(
            top_level_dir,
        )
    )
    attempts_df = process_attempts_df(attempts_df)

    # Extract tokens from string representation
    df["tokens"] = df["tokens"].map(
        lambda x: list(map(int, x.strip("][").split(", "))))
    # The length of the query
    df["query_length"] = df["query"].map(lambda x: len(x))
    # Get canonical representation using tokenizer
    df["canonical_representation"] = df["urls"].map(
        lambda x: list(test_relm.words_to_tokens(x)[0].numpy().tolist()))
    # Check if token representation is equal
    df["is_canonical"] = df["canonical_representation"] == df["tokens"]
    # Check if string representation has edits
    df["no_edit"] = df["urls"] == df["query"]
    # The set of characters in the query
    df["query_set"] = df["query"].map(set)
    # The set of characters in the returned response
    df["urls_set"] = df["urls"].map(set)
    # Add: In url but not in query
    df["add_diff_set"] = (df["urls_set"] - df["query_set"]).map(tuple)
    # Remove: In query but not in url
    df["remove_diff_set"] = (df["query_set"] - df["urls_set"]).map(tuple)

    # Breakdown of common extractions
    print("Common extractions",
          attempts_df.groupby("query")["num_results"].sum().sort_values())
    # Breakdown of extraction success rate
    attempts_df["extract_success"] = attempts_df["num_results"] > 0
    attempts_df["cumulative_attempts"] = (attempts_df["num_results"]
                                          .map(lambda x: 1)
                                          .cumsum())
    print("Num results: {} ({} attempts)".format(
            attempts_df["num_results"].sum(),
            attempts_df["cumulative_attempts"].max(),
          ))

    print("Success rate",
          attempts_df["extract_success"].mean())
    print("Success rate (for first {} examples)".format(max_num_attempts),
          (attempts_df
           .query("cumulative_attempts < {}".format(max_num_attempts))
           ["extract_success"]
           .mean())
          )

    edits_breakdown_p = df.groupby(["no_edit"])["urls"].count() / len(df)
    print("Edits breakdown\n{}".format(edits_breakdown_p))
    canonical_breakdown_p = (df.groupby(["is_canonical"])["urls"].count()
                             / len(df))
    print("Canonical breakdown\n{}".format(canonical_breakdown_p))

    # Breakdown of extractions by canonical/edits
    edits_canonical_breakdown = (df
                                 .groupby(["is_canonical", "no_edit"])["urls"]
                                 .count())
    edits_canonical_breakdown_p = edits_canonical_breakdown / len(df)
    print("Edits/Canonical breakdown\n{}".format(edits_canonical_breakdown))
    print("Edits/Canonical breakdown\n{}".format(edits_canonical_breakdown_p))

    # Breakdown of added characters
    add_diff_set_breakdown = (df
                              .groupby("add_diff_set")["urls"]
                              .count()
                              .sort_values())
    add_diff_set_breakdown_p = add_diff_set_breakdown / len(df)
    print("Added characters: {}".format(add_diff_set_breakdown_p.nlargest(10)))
    # Breakdown of removed characters
    remove_diff_set_breakdown = (df
                                 .groupby("remove_diff_set")["urls"]
                                 .count()
                                 .sort_values())
    remove_diff_set_breakdown_p = remove_diff_set_breakdown / len(df)
    print("Removed characters: {}".format(
        remove_diff_set_breakdown_p.nlargest(10)))

    num_cum_canonical_df = (df.groupby(["is_canonical",
                                        "no_edit",
                                        "query_length"])[["urls"]]
                            .count()
                            .unstack(["query_length"], fill_value=0)
                            .stack())
    num_cum_canonical_df = num_cum_canonical_df.sort_values(["query_length"])
    num_cum_canonical_df["cumsum"] = (num_cum_canonical_df
                                      .groupby(
                                          ["is_canonical", "no_edit"])["urls"]
                                      .apply(lambda x: x.cumsum()))
    num_cum_canonical_df["num_results"] = num_cum_canonical_df["cumsum"]
    num_cum_canonical_df = num_cum_canonical_df.reset_index()
    num_cum_canonical_df["Query Length"] = num_cum_canonical_df["query_length"]
    num_cum_canonical_df["Number of Results"] = \
        num_cum_canonical_df["num_results"]
    num_cum_canonical_df["cumulative_results"] = \
        num_cum_canonical_df["num_results"].cumsum()
    num_cum_canonical_df["Canonical"] = num_cum_canonical_df["is_canonical"]
    num_cum_canonical_df["extract_success"] = \
        num_cum_canonical_df["num_results"] > 0
    num_cum_canonical_df["Edits"] = \
        num_cum_canonical_df["no_edit"] == False  # noqa
    return num_cum_canonical_df


def plot_prompted_results():
    """Plot results that use prefix."""
    prompted_results = load_prompted_results()
    print("PROMPTED", prompted_results)
    sns.lineplot(data=prompted_results.query("cumulative_results <= 160"),
                 x="Cumulative Number of Attempts",
                 y="Cumulative Number of Results",
                 hue="Type",
                 **kwargs,
                 )
    plt.ylabel("Cumulative Number of Extractions")
    plt.tight_layout()
    plt.savefig("cumulative_by_idx_results_line_joined.pdf")
    plt.clf()
    max_attempt = (prompted_results
                   .query("cumulative_results <= 160")
                   .query("Type == 'ReLM'")['cumulative_attempts'].max())
    print('prompted success rate (first {} attempts)'.format(max_attempt),
          prompted_results.query(
              "cumulative_attempts <= {}".format(max_attempt))
          .groupby("Type")["extract_success"].mean())
    print('prompted success rate (all attempts)',
          prompted_results.groupby("Type")["extract_success"].mean())


def plot_unprompted_results():
    """Plot results that don't use prefix."""
    num_cum_canonical_df = load_unprompted_results()
    sns.lineplot(
        data=num_cum_canonical_df,
        x="Query Length",
        y="Number of Results",
        hue="Canonical",
        style="Edits",
        **kwargs,
    )
    plt.ylabel("Cumulative Number of Results")
    plt.tight_layout()
    plt.savefig("cumulative_breakdown_results_line.pdf")
    plt.clf()


plot_prompted_results()
plot_unprompted_results()
