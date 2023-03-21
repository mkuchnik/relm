"""Convert attempts.csv to a rewritten, more structured schema."""

import csv
import io

import pandas as pd


def convert_attempts_results_to_df(results_file):
    """Convert attempts_results.csv to new_schema."""
    with open(results_file, "r") as f:
        data = f.read()
    items = []
    lines = data.splitlines()
    lines = lines[1:]
    it = csv.reader(lines, delimiter=",")
    for k, v in it:
        if k == "realtime_start_time_s":
            # Init
            item = dict()
        item[k] = v
        if k == "y":
            # Flush
            items.append(item)
            del item
    buff = io.StringIO()
    new_df = pd.DataFrame(items)
    new_df.to_csv(buff)
    buff.seek(0)
    new_df = pd.read_csv(buff)
    return new_df
