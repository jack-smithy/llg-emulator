#!/bin/bash
#
# A long recipe as LINKS jobs of <= 3 h, each resuming from the last checkpoint and
# starting only after the previous one succeeded (a failed link cancels the rest):
#   scripts/chain.sh configs/energy-wide-long.yaml 0 4 --gres=gpu:full:1 -c 8
# Extra arguments go to sbatch. Links after the last training step only evaluate.
set -euo pipefail
config=$1 seed=$2 links=$3
shift 3
dep=()
for _ in $(seq "$links"); do
    id=$(sbatch --parsable --array="$seed" -t 3:00:00 --export=ALL,MAX_MINUTES=160 \
        "${dep[@]}" "$@" scripts/train.slrm "$config")
    echo "$id"
    dep=(--dependency=afterok:"$id")
done
