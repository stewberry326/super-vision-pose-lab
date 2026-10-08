#!/bin/sh
set -eu
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
mkdir -p "$ROOT_DIR/coreml/compiled"
for unit in features query update; do
 xcrun coremlcompiler compile "$ROOT_DIR/coreml/$unit.mlpackage" "$ROOT_DIR/coreml/compiled"
done
xcrun swiftc "$ROOT_DIR/scripts/coreml_sequence.swift" -o "$ROOT_DIR/coreml/check-sequence" -framework CoreML
"$ROOT_DIR/coreml/check-sequence" "$ROOT_DIR/coreml"
