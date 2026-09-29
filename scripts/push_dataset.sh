#!/bin/bash
# usage: scripts/push_dataset.sh [name]   (default: llg_field_switching)
set -euo pipefail
DATASET="${1:-llg_field_switching}"
uv run hf upload "jack-smithy/$DATASET" "datasets/$DATASET" --repo-type=dataset
