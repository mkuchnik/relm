# Bias Experiments
Experiments are grouped by model and whether they are configured to use a
prefix (referred to as "Inference" runs here).
For model, we use GPT2 and GPT2-XL.
Inference runs do not have a prompt/prefix.
Non-inference runs utilize both a prefix and a suffix.
We use the term "Vanilla" to describe experiments with all encodings and "Canonical"
to describe experiments with only canonical encodings.

To run the experiments, you can do the following:
```bash
run_bias_gpt2.sh
run_bias_gpt2xl.sh
run_bias_gpt2_inference.sh
run_bias_gpt2xl_inference.sh
```

To make this experiment run faster, consider changing `MAX_SAMPLES` in
the scripts.
Once the experiments have run, you can plot them by using the plotter script.
The plotter script points to the directory containing the experiments and plots
all results in the current directory
(and thus past plots will be overwritten with each run).
To plot all results (matching the run order from above):

```bash
python3 plot_bias.py --top_level_directory=test_bias_gpt2
```

```bash
python3 plot_bias.py --top_level_directory=test_bias_gpt2xl
```

```bash
python3 plot_bias.py --top_level_directory=test_bias_inference_gpt2
```

```bash
python3 plot_bias.py --top_level_directory=test_bias_inference_gpt2xl
```
