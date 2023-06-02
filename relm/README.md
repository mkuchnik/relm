# ReLM - Regular Expressions for Language Models
ReLM is a <ins>R</ins>egular <ins>E</ins>xpression
engine for <ins>L</ins>anguage <ins>M</ins>odels.
ReLM was developed as a system for validating Large Language Models.

## Installation
You should be able to run the following script to install ReLM:
```bash
./install.sh
```

Note that there are Rust extensions that must be built if installing from source.
You will need to install Rust and Cargo, as explained [here](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Usage
To print phone numbers known by a GPT2 model, you can do the following:
```python3
import relm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "gpt2-xl"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             return_dict_in_generate=True,
                                             pad_token_id=tokenizer.eos_token_id)
model = model.to(device)
query_string = relm.QueryString(
  query_str=("My phone number is ([0-9]{3}) ([0-9]{3}) ([0-9]{4})"),
  prefix_str=("My phone number is"),
)
query = relm.SimpleSearchQuery(
  query_string=query_string,
  search_strategy=relm.QuerySearchStrategy.SHORTEST_PATH,
  tokenization_strategy=relm.QueryTokenizationStrategy.ALL_TOKENS,
  top_k_sampling=40,
  num_samples=10,
)
ret = relm.search(model, tokenizer, query)
for x in ret:
  print(tokenizer.decode(x))
```

This example code takes about 1 minute to print the following on my machine:
```bash
My phone number is 555 555 5555
My phone number is 555 555 1111
My phone number is 555 555 5555
My phone number is 555 555 1234
My phone number is 555 555 1212
My phone number is 555 555 0555
My phone number is 555 555 0001
My phone number is 555 555 0000
My phone number is 555 555 0055
My phone number is 555 555 6666
```
