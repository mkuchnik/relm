#!/bin/bash
MODEL="gpt2-xl"
TOP_DIR="test_insults_${MODEL}"
SCRIPT_DIR="$(pwd)/find_insults.py"
DATA_FILE="$(pwd)/00.txt"

standard_test() {
  SCRIPT_DIR="$(pwd)/find_insults.py"
  TEST_DIR="${TOP_DIR}/standard"
  MAX_SAMPLES=1
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    "${DATA_FILE}" \
    --model="${MODEL}" \
    --top_k=40 \
    --enable_edits \
    --num_edits=1 \
    --num_punctuation_edits=0 \
    --num_space_edits=0 \
    --batch_size=1 \
    --max_samples="${MAX_SAMPLES}" \
    --max_results=250
  popd || exit
}

baseline_test() {
  SCRIPT_DIR="$(pwd)/find_insults.py"
  TEST_DIR="${TOP_DIR}/baseline"
  MAX_SAMPLES=1
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    "${DATA_FILE}" \
    --model="${MODEL}" \
    --top_k=40 \
    --batch_size=1 \
    --static_minimize \
    --max_samples="${MAX_SAMPLES}" \
    --max_results=500
  popd || exit
}

standard_test_no_prefix() {
  SCRIPT_DIR="$(pwd)/find_insults.py"
  TEST_DIR="${TOP_DIR}/standard_noprefix"
  MAX_SAMPLES=1000
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    "${DATA_FILE}" \
    --model="${MODEL}" \
    --top_k=40 \
    --enable_edits \
    --num_edits=1 \
    --num_punctuation_edits=0 \
    --num_space_edits=0 \
    --batch_size=1 \
    --no_prefix \
    --max_samples="${MAX_SAMPLES}"
  popd || exit
}

baseline_test_no_prefix() {
  SCRIPT_DIR="$(pwd)/find_insults.py"
  TEST_DIR="${TOP_DIR}/baseline_noprefix"
  MAX_SAMPLES=1000
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    "${DATA_FILE}" \
    --model="${MODEL}" \
    --top_k=40 \
    --batch_size=1 \
    --static_minimize \
    --no_prefix \
    --max_samples="${MAX_SAMPLES}"
  popd || exit
}

baseline_test
standard_test
standard_test_no_prefix
baseline_test_no_prefix
