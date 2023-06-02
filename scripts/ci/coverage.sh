#!/bin/bash
COVERAGE_DIR="${GITHUB_WORKSPACE}/coverage"
mkdir -p "${COVERAGE_DIR}"
cd "${GITHUB_WORKSPACE}/build"
lcov -c -d . -o "${COVERAGE_DIR}/tmp.info"
lcov --remove "${COVERAGE_DIR}/tmp.info" \
              "/usr/include/*" \
              "${HOME}/*" \
              "../" \
              -o "${COVERAGE_DIR}/lcov.info"