#!/bin/bash
MAX_SAMPLES=10000
MODEL="gpt2-xl"
TOP_DIR="test_memorization_${MODEL}"

relm_test() {
  SCRIPT_DIR="$(pwd)/relm_runner.py"
  TEST_DIR="${TOP_DIR}/relm"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --query=any_url \
    --max_samples=${MAX_SAMPLES} \
    --top_k=40 \
    --model=${MODEL} \
    --results_file="results.json"
  popd || exit
}

baseline_test() {
  SCRIPT_DIR="$(pwd)/baseline/run_generation.py"
  TEST_DIR="${TOP_DIR}/baseline_${LENGTH}"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --model_type="${MODEL}" \
    --model_name_or_path="${MODEL}" \
    --length="${LENGTH}" \
    --k=40 \
    --p=1.0 \
    --prompt=https://www. \
    --num_return_sequences=1 \
    --add_special_tokens \
    --num_total_samples="${MAX_SAMPLES}"
  popd || exit
}

relm_test
LENGTH=1  # Power of 2
baseline_test
LENGTH=2  # Power of 2
baseline_test
LENGTH=4  # Power of 2
baseline_test
LENGTH=8  # Power of 2
baseline_test
LENGTH=16  # Power of 2
baseline_test
LENGTH=32  # Power of 2
baseline_test
LENGTH=64  # Power of 2
baseline_test