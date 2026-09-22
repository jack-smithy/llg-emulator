#!/bin/bash

CONFIGURATION="normalized-fno-4-step-lr-1e-3-long"

pids=()
for SEED in {0..3}; do
    uv run eval.py --configuration "$CONFIGURATION" --seed "$SEED" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid" || echo "pid $pid failed with $?"
done