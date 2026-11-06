#!/bin/zsh
# Run from any directory; dependencies remain isolated from the upstream checkout.
set -eu
scale_root=${0:A:h}
workspace_root=${scale_root:h:h:h}
export PYTHONPATH="/private/tmp/sndl-class-packages-749rpbye/site:/private/tmp/sndl-class-source-tlefts_z/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="/Users/arthurkhvan/.codex/visualizations/2026/11/03/01a19f80-851f-7960-9c7a-18ef1711b019/shd-recovery/pycache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
exec "$workspace_root/.venv-04/bin/python" "$@"
