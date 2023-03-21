#!/bin/bash

set -e

python3 -m pip install --upgrade build
python3 -m build
python3 -m pip uninstall -y relm-mkuchnik || echo "No existing build"
python3 -m pip install dist/*.whl