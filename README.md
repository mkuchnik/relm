# ReLM
A repository for replicating the experiments found in
"Validating Large Language Models with ReLM" (MLSys '23).
ReLM is a <ins>R</ins>egular <ins>E</ins>xpression
engine for <ins>L</ins>anguage <ins>M</ins>odels.
The goal of ReLM is to make it easier for users to
test aspects of a language model, such as memorization, bias, toxicity finding,
and language understanding.
For example, to find the most likely (i.e., potentially memorized)
phone numbers in the largest GPT2 model under top-k=40
decoding, you can run the following code snippet:

```python3
import relm
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             return_dict_in_generate=True,
                                             pad_token_id=tokenizer.eos_token_id)
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

As can be seen, the top number is `555 555 5555`, which is a widely used
fake phone number.

## Hardware Requirements
We use the following hardware setup in our experiments.

CPU: AMD Ryzen 5800X,
GPU: Nvidia RTX 3080 10GiB,
RAM: 32GiB

## Software Requirements
ReLM builds strongly on PyTorch and HuggingFace, with a focus on models similar
to GPT-2.
As such, the current architecture assumes the model fits on a single-node.
The OS we use is Ubuntu 20.04.

We recommend using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a standardized environment.
We use a Python3.7 environment (py37) for both building and installing the
following software.

To install:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
You will have to scroll though and type "yes" at the end. You should leave the
options as the defaults.


After install, you will want to create the environment. To create it:

```bash
conda create -n py37 python=3.7
```

To activate the environment:
```bash
conda activate py37
```

You can then install dependencies inside this environment.
We additionally use Rust as a backend for parts of the ReLM runtime.
Therefore, you will need to install a Rust compiler and build the corresponding
extensions.

###### PyTorch
Install PyTorch (more instructions
[here](https://pytorch.org/get-started/locally/).
We used PyTorch 1.10, which requires [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive).

```bash
# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

If you are not using a GPU, install the CPU version:
```bash
# CPU Only
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
```

###### Rust
You will need to install Rust and Cargo, as explained [here](https://doc.rust-lang.org/cargo/getting-started/installation.html).
The easiest way is to run:
```bash
curl https://sh.rustup.rs -sSf | sh
```

You should also have a C linker installed:
```bash
apt install build-essential
```

###### ReLM Install
Build and install the Rust bindings for ReLM.
```bash
pushd rust_regex_compiler
cargo build --release
popd
pushd rust_regex_compiler_bindings
bash install_deps.sh
bash install.sh
popd
```

Build and install ReLM.
```bash
pushd relm
bash install.sh
popd
```

###### Additional Dependencies
You can install additional dependencies from the provided requirements file.
```bash
pip install -r requirements.txt
```

## Getting Started
We recommend checking out the Jupyter Notebook
[Introduction_to_ReLM](notebook/Introduction_to_ReLM.ipynb) to get started.

To run it, you will need to install additional dependencies in the conda
environment.
```bash
conda install nb_conda
conda install -c conda-forge ipywidgets
```

Then you can do:
```bash
cd notebook
jupyter-notebook Introduction_to_ReLM.ipynb
```

## Experiments
Experiments in the paper can be found under the [Experiments](experiments)
directory.
Each experiment has its own README with instructions.
Namely, we use the datasets The Pile and LAMBADA with GPT2 and GPT2-XL models.
These experiments can take a while to run and can be made smaller by limiting
the number of samples generated.
