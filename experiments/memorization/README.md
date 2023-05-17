# Memorization
This experiment tests ReLM's ability to find URLs in a model.
First, attemps at URL extractions are performed.
Then, we validate whether the URL is valid.
We focus on GPT2-XL.

Attempt mining with ReLM and the baseline is performed with.

```bash
bash run_url_gpt2xl.sh
```

This will generate a results directory, which has a `results.json` file
describing the extraction attempts for each of the experiments.
To make this experiment run faster, consider changing `MAX_SAMPLES` in
`run_url_gpt2xl.sh`.

Now, before moving to the next step, first install the required requirements.

```bash
pip3 install -r requirements.txt
```

Then, we post-process these file to determine if they are valid URLs with:

```bash
pip3 install -r requirements.txt
for f in test_memorization_gpt2-xl/*; do python3 plot_results.py $f; done
```

NOTE: Please be mindful of how many URLs you are validating and how quickly you
are doing so---repeatedly querying a website may get your IP address flagged as
potentially performing a Denial of Service (DoS) attack!
If this happens, you will be throttled!

The resulting dataframes can be plotted in aggregate with:

```bash
python3 mega_plot_results.py test_memorization_gpt2-xl
```

