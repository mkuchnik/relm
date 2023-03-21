#!/bin/bash
MAX_SAMPLES=5000
MODEL="gpt2-xl"
TOP_DIR="test_bias_inference_${MODEL}"
SCRIPT_DIR="$(pwd)/relm_runner.py"

vanilla_test() {
  TEST_DIR="${TOP_DIR}/vanilla"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  # Vanilla random man
  python3 "${SCRIPT_DIR}" \
    --query=professions_man_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --results_file="results_man.json"
  # Vanilla random woman
  python3 "${SCRIPT_DIR}" \
    --query=professions_woman_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --results_file="results_woman.json"
  popd || exit
}

vanilla_edits_test() {
  TEST_DIR="${TOP_DIR}/vanilla_edits"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  # Vanilla random man
  python3 "${SCRIPT_DIR}" \
    --query=professions_man_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --enable_edits \
    --results_file="results_man.json"
  # Vanilla random woman
  python3 "${SCRIPT_DIR}" \
    --query=professions_woman_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --enable_edits \
    --results_file="results_woman.json"
  popd || exit
}

canonical_test() {
  TEST_DIR="${TOP_DIR}/canonical"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  # Vanilla random man
  python3 "${SCRIPT_DIR}" \
    --query=professions_man_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --static_minimize \
    --results_file="results_man.json"
  # Vanilla random woman
  python3 "${SCRIPT_DIR}" \
    --query=professions_woman_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --static_minimize \
    --results_file="results_woman.json"
  popd || exit
}

canonical_edits_test() {
  TEST_DIR="${TOP_DIR}/canonical_edits"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  # Vanilla random man
  python3 "${SCRIPT_DIR}" \
    --query=professions_man_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --static_minimize \
    --enable_edits \
    --results_file="results_man.json"
  # Vanilla random woman
  python3 "${SCRIPT_DIR}" \
    --query=professions_woman_inference \
    --max_samples=${MAX_SAMPLES} \
    --temperature=1.0 \
    --model=${MODEL} \
    --query_mode="random" \
    --static_minimize \
    --enable_edits \
    --results_file="results_woman.json"
  popd || exit
}

vanilla_edits_test
canonical_edits_test
canonical_test
vanilla_test