#!/bin/zsh
set -eu
experiment_root=${0:A:h}
workspace_root=${experiment_root:h:h:h}
engine_runtime='/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/engine-runtime'
engine_source=${ENGINE_SOURCE_ROOT:-$engine_runtime/baseline-source}
export PYTHONPATH="$engine_source/src:$engine_source/tests:$engine_source:$engine_runtime/site:/private/tmp/sndl-class-packages-749rpbye/site${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX='/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/shd-recovery/pycache'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUTF8=1
export PYTHONHASHSEED=0
exec "$workspace_root/.venv-04/bin/python" "$@"
