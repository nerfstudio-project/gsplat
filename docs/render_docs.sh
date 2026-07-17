#!/usr/bin/env bash
# Render the gsplat Sphinx documentation.
#
# Usage:
#   docs/render_docs.sh [OUTPUT_DIR]
#
# OUTPUT_DIR defaults to build/html (relative to the repo root).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-build/html}"

# -W fails on warnings (e.g. "document isn't included in any toctree" if
#   docs/source/index.rst is ever truncated or a new page is added without
#   wiring it in).
# --keep-going collects every warning before exiting so a single failure
#   doesn't hide the rest.
# The sphinx-build console entry point does not add the working directory to
# Python's import path, so expose the source package explicitly for autodoc.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    sphinx-build -W --keep-going docs/source "$OUTPUT_DIR"
