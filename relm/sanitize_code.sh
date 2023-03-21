#!/bin/bash

autopep8 --in-place --aggressive --aggressive --aggressive src/relm/**/*.py
autopep8 --in-place --aggressive --aggressive --aggressive tests/*.py
isort src/relm/.
isort tests