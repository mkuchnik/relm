#!/bin/bash
MAX_RESULTS=501
MODEL="gpt2-xl"
TOP_DIR="test_knowledge_${MODEL}"

standard_stop_test() {
  SCRIPT_DIR="$(pwd)/run_eval.py"
  TEST_DIR="${TOP_DIR}/standard_stop"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --static_minimize \
    --add_eos \
    --top_k=1000 \
    --remove_stop_words \
    --max_results=${MAX_RESULTS} \
    --model=${MODEL}
  popd || exit
}

standard_test() {
  SCRIPT_DIR="$(pwd)/run_eval.py"
  TEST_DIR="${TOP_DIR}/standard"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --static_minimize \
    --add_eos \
    --top_k=1000 \
    --max_results=${MAX_RESULTS} \
    --model=${MODEL}
  popd || exit
}

baseline_test() {
  SCRIPT_DIR="$(pwd)/run_eval.py"
  TEST_DIR="${TOP_DIR}/baseline"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --static_minimize \
    --top_k=1000 \
    --max_results=${MAX_RESULTS} \
    --model=${MODEL}
  popd || exit
}

baseline_words_test() {
  SCRIPT_DIR="$(pwd)/run_eval.py"
  TEST_DIR="${TOP_DIR}/baseline_words"
  mkdir -p "${TEST_DIR}"
  pushd "${TEST_DIR}" || exit
  python3 "${SCRIPT_DIR}" \
    --static_minimize \
    --top_k=1000 \
    --force_context_words \
    --max_results=${MAX_RESULTS} \
    --model=${MODEL}
  popd || exit
}

# Anything
baseline_test
# Words
baseline_words_test
# EOS
standard_test
# EOS + filter
standard_stop_test