#!/bin/bash
echo "Downloading The Pile file"
wget https://the-eye.eu/public/AI/pile/train/00.jsonl.zst
echo "Download dependencies"
pip install -r requirements.txt
echo "Decompressing to 00.txt"
python3 convert_data.py > 00.txt
