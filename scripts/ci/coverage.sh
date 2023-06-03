#!/bin/bash
COVERAGE_DIR="${GITHUB_WORKSPACE}/coverage"
mkdir -p "${COVERAGE_DIR}"
cd "${GITHUB_WORKSPACE}/build"
lcov -c -d . -o "${COVERAGE_DIR}/tmp.info"
lcov --remove "${COVERAGE_DIR}/tmp.info" \
              "/usr/*" \
              "*/spack/*" \
              -o "${COVERAGE_DIR}/lcov.info"
genhtml "${COVERAGE_DIR}/tmp.info" --output-directory coverage_report

#lcov -c -d . -o "tmp.info"
#lcov --remove "tmp.info" \
#              "/usr/*" \
#              "*/spack/*" \
#              -o "lcov.info"
#genhtml "lcov.info" --output-directory coverage_report
