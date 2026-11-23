#!/usr/bin/env bash
# Fine-tune and evaluate every backbone with all ten seeds (Section 4.3).
# About 52 GPU-hours of training on one RTX 3090, plus evaluation.
# Resumable: skips runs whose checkpoint or results already exist.
#
#   bash scripts/run_all_finetuning.sh
#   BACKBONES="tiny 100M" SEEDS="1 2" bash scripts/run_all_finetuning.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
BACKBONES=${BACKBONES:-"tiny 100M 300M 600M"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}

for bb in $BACKBONES; do
  for s in $SEEDS; do
    if [ ! -f "runs/training/seed$s/${bb}_finetune_cost.csv" ]; then
      echo "[$(date +%F_%T)] train $bb seed $s"
      python scripts/03_finetune.py --backbone "$bb" --seed "$s"
    fi
    res="runs/finetuned/seed$s/results_${bb}.csv"
    if [ ! -f "$res" ] || [ "$(wc -l < "$res")" -lt 10001 ]; then
      echo "[$(date +%F_%T)] evaluate $bb seed $s"
      python scripts/02_evaluate.py --backbone "$bb" --finetuned --seed "$s"
    fi
  done
done
echo "done"
