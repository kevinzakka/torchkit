#!/usr/bin/env bash
#
# Modified from https://raw.githubusercontent.com/HumanCompatibleAI/seals/master/ci/code_checks.sh
set -x
set -e

SRC_FILES=(torchkit/ tests/ docs/conf.py setup.py)

if [ "$(uname)" == "Darwin" ]; then
  N_CPU=$(sysctl -n hw.ncpu)
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  N_CPU=$(grep -c ^processor /proc/cpuinfo)
fi

echo "Source format checking"
flake8 ${SRC_FILES[@]}
black --check ${SRC_FILES}

if [ "$skipexpensive" != "true" ]; then
  echo "Building docs (validates docstrings)"
  pushd docs/
  make clean
  make html
  popd

  echo "Type checking"
  pytype -n "${N_CPU}" ${SRC_FILES[@]}
fi
