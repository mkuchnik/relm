#!/bin/bash

autopep8 --in-place --aggressive --aggressive --aggressive python/relm/**/*.py
autopep8 --in-place --aggressive --aggressive --aggressive tests/**/*.py
isort python/relm/.
isort tests/.