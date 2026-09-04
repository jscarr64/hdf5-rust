#!/usr/bin/env bash
# MATLAB h5read gold. Requires `matlab` on PATH or MATLAB=/path/to/matlab.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [ -n "${MATLAB:-}" ]; then
  :
elif command -v matlab >/dev/null 2>&1; then
  export MATLAB="$(command -v matlab)"
else
  echo "MATLAB binary not found. Export MATLAB=/path/to/matlab and re-run."
  exit 1
fi
cargo test --test interop gold_matlab_h5read -- --ignored --nocapture
