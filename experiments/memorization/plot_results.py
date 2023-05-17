"""Check if URLs are valid."""

import argparse
import concurrent.futures
import json
import socket

import httplib2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
import validators
import re

model = "gpt2-xl"

plot_prefix = model
parser = argparse.ArgumentParser()
parser.add_argument("top_level_directory",
                    type=str,
                    help="The directory where results are located.")
args = parser.parse_args()
top_level_dir = args.top_level_directory

filename = "{}/results.json".format(top_level_dir)
with open(filename, "r") as f:
    results = json.load(f)
df = pd.DataFrame(results)
df["temperature"] = 1.0
df["model"] = model

print(df)
df["count"] = df["urls"].map(lambda x: 1).cumsum()
df["url_length"] = df["urls"].map(lambda x: len(x))

print(df.sort_values(["url_length"]))
urls = df["urls"].tolist()
prefixes = []
clean_urls = []
for u in urls:
    prefix = "None"
    clean_url = u
    clean_url = re.sub("<|endoftext|>$", "", clean_url.split("\n")[0])
    prefixes.append(prefix)
    clean_urls.append(clean_url)
df["prefixes"] = prefixes
df["clean_urls"] = clean_urls
df["minutes"] = df["times"] / 60.
df["hours"] = df["minutes"] / 60.

print(df)

sns.lineplot(data=df, x="minutes", y="url_length", hue="prefixes")
plt.savefig("{}_minutes_vs_url_length_vs_prefixes.png".format(plot_prefix))
plt.clf()

num_queries = df.groupby(["temperature", "prefixes"])["times"].count()
num_seconds = df.groupby(["temperature", "prefixes"])["times"].max()
queries_per_second = num_queries / num_seconds

num_queries = df.groupby(["temperature"])["times"].count()
num_seconds = df.groupby(["temperature"])["times"].max()
queries_per_second = (num_queries / num_seconds).reset_index()
queries_per_second["queries_per_second"] = queries_per_second["times"]
g = sns.barplot(x="temperature", y="queries_per_second",
                data=queries_per_second)
plt.savefig("{}_minutes_vs_qps_barplot.png".format(plot_prefix))
plt.clf()

# Validator part of URLs

maybe_valid = [validators.url(x) for x in df["clean_urls"]]
df["is_maybe_valid"] = maybe_valid


def is_valid_url_light(x, strict=True, return_code=False):
    """Check if url is valid."""
    if not validators.url(x):
        return False
    h = httplib2.Http(timeout=2)
    h.follow_redirects = False
    try:
        resp = h.request(x, 'HEAD')
    except httplib2.ServerNotFoundError:
        return False
    except (httplib2.HttpLib2Error, socket.error):
        return None
    except:  # noqa
        return None
    if return_code:
        return int(resp[0]['status'])
    if strict:
        ret = int(resp[0]['status']) < 300
    else:
        ret = int(resp[0]['status']) < 400
    return ret


is_valid_url = lambda x: is_valid_url_light(x, return_code=True)  # noqa

sorted_urls = list(df["clean_urls"])
print("sorted urls", sorted_urls)
with tqdm.tqdm(total=len(sorted_urls)) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(is_valid_url, url): url
                   for url in sorted_urls}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            results[url] = future.result()
            pbar.update(1)


def extract_results(x):
    """Return results from threadpool's work."""
    if x in results:
        return results[x]
    return None


df["is_validated_valid_code"] = df["clean_urls"].map(extract_results)


def code_to_valid(x):
    """Convert HTTP code to boolean."""
    if x is None or isinstance(x, bool):
        return x
    elif isinstance(x, int):
        return x < 300
    else:
        return x


df["is_validated_valid"] = df["is_validated_valid_code"].map(code_to_valid)
print(df)

df["cumulative_validated_urls"] = (df[["temperature",
                                       "is_validated_valid"]]
                                   .groupby("temperature")
                                   ["is_validated_valid"]
                                   .apply(lambda x: x.cumsum()))
sns.lineplot(data=df, x="minutes", y="cumulative_validated_urls",
             hue="temperature")
plt.savefig("{}_minutes_vs_validated_urls_vs_temperature.png".format(
    plot_prefix))
plt.clf()

df.to_csv("{}/results_joined_{}.csv".format(top_level_dir, model))
