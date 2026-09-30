#!/bin/bash
# Build the shared operator library.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -O3 -march=native -shared -fPIC -std=c++17 \
    -o build/libnpuops.so cpp/operators.cpp
echo "built: build/libnpuops.so"
