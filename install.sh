#!/usr/bin/env bash
# ===================================================================
#  Spiking Neural Data Lake -- POSIX (Linux/macOS) installer launcher.
#  Finds Python 3.11+ and runs packaging/install.py, which builds a
#  virtual environment (.venv) and installs the package + every extra
#  it can on this machine (gpu/torch, genn are best-effort).
#
#    ./install.sh                 # install everything possible
#    ./install.sh --skip gpu,genn # skip the ~2 GB torch stack + genn
# ===================================================================
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

py=""
for cand in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    py="$cand"
    break
  fi
done

if [ -z "$py" ]; then
  echo "Python 3.11+ was not found. Install it (https://www.python.org/downloads/) and re-run ./install.sh" >&2
  exit 1
fi

echo "Using: $py"
exec "$py" "$here/packaging/install.py" "$@"
