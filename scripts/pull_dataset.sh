#!/bin/bash
# usage: scripts/pull_dataset.sh [name]   (default: llg_field_switching)
set -euo pipefail
DATASET="${1:-llg_field_switching}"
uv run hf download "jack-smithy/$DATASET" --repo-type=dataset --local-dir "datasets/$DATASET"
