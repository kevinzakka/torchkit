# Modified from https://raw.githubusercontent.com/HumanCompatibleAI/seals/master/ci/code_checks.sh
#!/usr/bin/env bash

SRC_FILES=(torchkit/ tests/ docs/conf.py setup.py)

set -x  # echo commands
set -e  # quit immediately on error

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
  pytype ${SRC_FILES[@]}
fi
