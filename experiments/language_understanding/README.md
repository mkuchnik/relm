# Language Understanding
In these experiments, we run word completion using the LAMBADA dataset.
For models, we use both GPT2 and GPT2-XL.

We run this task with various ReLM parameters to induce different accuracy behavior.
The configurations are:
baseline,
words (referred to here as baseline\_words),
adding EOS (referred to as standard),
and
removing stop words (referred to as standard\_stop).

## Data
To download the dataset, run:

```bash
bash get_dataset.sh
```

## Experiments
To run the experiments, run:
```bash
bash run_lambada_gpt2.sh 
bash run_lambada_gpt2xl.sh 
```

To print the accuracy results, run:

```bash
python3 plot_results.py test_knowledge_gpt2
python3 plot_results.py test_knowledge_gpt2-xl
```
