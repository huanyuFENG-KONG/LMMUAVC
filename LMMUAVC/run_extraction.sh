#!/bin/bash
# Run feature extraction from repository root
# Usage: ./run_extraction.sh [data_path] [extra args...]
# Example: ./run_extraction.sh /path/to/MMAUD/data

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$1" ]; then DATA_ROOT="$1"; shift; else DATA_ROOT="${REPO_ROOT}/data"; fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

cd "${REPO_ROOT}/visual_processing/preprocessing"
python extract_and_align_features.py --data-root "${DATA_ROOT}" "$@"
