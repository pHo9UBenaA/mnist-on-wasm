#!/bin/sh

set -ex

# This example requires to *not* create ES modules, therefore we pass the flag
# `--target no-modules`
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target no-modules
