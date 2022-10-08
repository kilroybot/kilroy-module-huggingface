#!/bin/bash --login

set +euo pipefail
conda activate kilroy-module-huggingface
set -euo pipefail

exec "$@"
