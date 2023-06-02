#!/bin/bash

docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release  # or other maturin arguments
