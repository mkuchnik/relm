#!/bin/bash

CUDA_VISIBLE_DEVICES="" python3 -m unittest discover tests/unit
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover tests/integration
