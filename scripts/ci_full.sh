#!/usr/bin/env bash
# hdf5-rust pre-publish checks. No crate dependencies; no libhdf5.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

echo "==> cargo test (std)"
cargo test

echo "==> cargo test --no-default-features"
cargo test --no-default-features

echo "==> cargo check --no-default-features"
cargo check --no-default-features

echo "==> no crate dependencies"
if grep -E '^\[dependencies\]' Cargo.toml >/dev/null; then
  deps="$(awk '/^\[dependencies\]/{p=1;next} /^\[/{p=0} p && NF && !/^#/' Cargo.toml || true)"
  if [ -n "${deps}" ]; then
    echo "Cargo.toml [dependencies] must be empty"
    echo "$deps"
    exit 1
  fi
fi
if awk '/^\[dependencies\]/{p=1;next} /^\[/{p=0} p && /libhdf5/' Cargo.toml | grep -q .; then
  echo "libhdf5 must not appear in [dependencies]"
  exit 1
fi

echo "==> no hardware float types in src"
if grep -nRE '(^|[^A-Za-z0-9_])(f32|f64)([^A-Za-z0-9_]|$)' --include='*.rs' src \
  | grep -vE 'write_f64|read_f64|append_f64|write_f32|read_f32|ieee_f64|ieee_f32|Float64|Float32|F64LE|F32LE|f64_slice'; then
  echo "hardware f32/f64 tokens in src/"
  exit 1
fi

echo "==> no TODO/FIXME/HACK in src"
if grep -nRE 'TODO|FIXME|HACK' --include='*.rs' src; then
  exit 1
fi

echo "==> no zenith-float / accumath mentions"
if grep -nRIE 'zenith-float|zenith_float|Accumath|accumath' \
  --exclude-dir=target --exclude-dir=.git \
  --exclude='ci_full.sh' .; then
  echo "this crate must not mention those names"
  exit 1
fi

echo "ci_full: ok"
