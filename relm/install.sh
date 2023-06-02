#!/bin/bash

set -e

python3 -m pip install maturin==1.0.0
python3 -m pip uninstall -y relm || echo "No existing build"
maturin develop --release