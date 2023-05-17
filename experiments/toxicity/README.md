# Toxicity
This directory corresponds to finding toxic content in models using an auxilary
dataset and a list of bad words.
For the experiments, we see if a sequence containing the bad word can be
extracted, and we focus on GPT2-XL.

## Data
The auxilary dataset we use is the first file (i.e., "00") from The Pile.
We decompress this file and save it as a text file.
To do so, run the command (note: this requires roughly 50GiB space):

```bash
get_dataset.sh
```

## Experiments
The experiments are split between prompted and not.
Prompted results have a larger search space than unprompted, so they take longer
per item to complete.
To run all configurations fully, run the following.
Results are saved after each extraction attempt, so it is safe to
cancel the run after sufficient extractions have been attempted.

```bash
bash run_insults_gpt2xl.sh
```

To make this experiment run faster, consider changing `MAX_SAMPLES` in
the scripts as well as passing `--max_results`.

Once results are run, you can plot them with:
```bash
python3 plot_results.py
```
