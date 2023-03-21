"""Joins and plots results_joined files."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.1)
sns.set_palette("colorblind", 8)

parser = argparse.ArgumentParser()
parser.add_argument("top_level_directory",
                    type=str,
                    help="The directory where experiments are located.")
args = parser.parse_args()
top_level_dir = args.top_level_directory

dfs = []
orders = []
f1 = "{}/relm/results_joined_gpt2-xl.csv".format(top_level_dir)
df1 = pd.read_csv(f1)
df1["name"] = "gpt2xl_relm"
df1["length"] = "ReLM"
df1["eventID"] = df1.groupby("temperature").cumcount()
df1["with_duplicates"] = False
dfs.append(df1)
df1 = df1.copy().drop_duplicates("clean_urls")
df1["cumulative_validated_urls"] = (
    df1[["temperature", "is_validated_valid"]]
    .groupby("temperature")["is_validated_valid"].apply(lambda x:
                                                        x.cumsum())
)

orders.append("ReLM")
df1["with_duplicates"] = True
dfs.append(df1)

for n in [1, 2, 4, 8, 16, 32, 64]:
    f =  \
        "{}/baseline_{}/results_joined_gpt2-xl.csv".format(top_level_dir, n)
    df = pd.read_csv(f)
    df["name"] = "gpt2xl_baseline_{}".format(n)
    df["length"] = n
    df["eventID"] = np.arange(len(df))
    df["with_duplicates"] = True
    dfs.append(df)
    orders.append(n)
    # Remove duplicates
    df = df.copy().drop_duplicates("clean_urls")
    df["cumulative_validated_urls"] = (
        df[["temperature", "is_validated_valid"]]
        .groupby("temperature")["is_validated_valid"].apply(lambda x:
                                                            x.cumsum())
    )
    df["with_duplicates"] = False
    print(df)
    dfs.append(df)

mega_df = pd.concat(dfs)


def bool_to_int(x):
    """Convert boolean to int or None."""
    if x == "True" or x == "False":
        return int(bool(x))
    elif np.isnan(float(x)):
        return None
    else:
        try:
            return int(x)
        except ValueError:
            return int(float(x))


mega_df["cumulative_validated_urls"] = \
    mega_df["cumulative_validated_urls"].map(bool_to_int)
cumulative_attempted_urls = mega_df.groupby(["name", "temperature",
                                             "with_duplicates"]).cumcount()
cumulative_attempted_urls.name = "cumulative_attempted_urls"
mega_df = pd.concat([mega_df, cumulative_attempted_urls], axis=1)
print("mega_df", mega_df)
throughput = mega_df.groupby(["name",
                              "temperature",
                              "length",
                              "with_duplicates"])[["cumulative_validated_urls",
                                                   "times"]].apply(
    lambda x: x.cumulative_validated_urls.max() / x.times.max())
throughput.name = "throughput"
throughput = throughput.reset_index()
print(throughput)


mega_df["With Duplicates"] = mega_df["with_duplicates"]
mega_df["Time (min)"] = mega_df["minutes"]
mega_df["Cumulative Validated URLs"] = mega_df["cumulative_validated_urls"]


def _is_int(n):
    """Return True if n is integer."""
    try:
        int(n)
        return True
    except:  # noqa
        return False


def n_to_name(n):
    """Convert baseline N to string."""
    if _is_int(n):
        nn = int(n)
        return "Baseline (n={})".format(nn)
    else:
        return n


mega_df["Method"] = mega_df["length"].map(n_to_name)

throughput["Throughput (Val. URL/Sec)"] = throughput["throughput"]
throughput["Method"] = throughput["length"].map(n_to_name)

_orders = list(map(n_to_name, orders))
print("orders", orders)

with sns.plotting_context("paper", font_scale=1.6):
    g = sns.barplot(data=throughput.query("with_duplicates == False"),
                    x="Method",
                    y="Throughput (Val. URL/Sec)",
                    order=_orders,
                    )
    plt.xticks(rotation=30)
    plt.tight_layout()
    g.set_yscale("log")
    plt.savefig(
        "gpt2xl_relm_vs_baselines_throughput_noduplicates_notemp_log.pdf")
    plt.clf()

with sns.axes_style("white"):
    plt.figure(figsize=(13, 7), dpi=100)
    kwargs = {
        "linewidth": 2.0,
    }
    g = sns.lineplot(data=mega_df,
                     x="Time (min)", y="Cumulative Validated URLs",
                     hue="Method",
                     style="With Duplicates",
                     **kwargs)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig("gpt2xl_relm_vs_baselines_duplicates.pdf")
    plt.clf()

with sns.axes_style("white"):
    plt.figure(figsize=(13, 7), dpi=100)
    kwargs = {
        "linewidth": 2.0,
    }
    g = sns.lineplot(data=mega_df.query(
        "with_duplicates == False & times < 301"),
                     x="Time (min)", y="Cumulative Validated URLs",
                     hue="Method",
                     alpha=0.8,
                     **kwargs)
    g.set_yscale('log')
    plt.ylim(1, None)
    plt.tight_layout()
    plt.savefig("gpt2xl_relm_vs_baselines_noduplicates_zoom_log_alpha.pdf")
    plt.clf()
