#!/usr/bin/env bash
# Fase de experimentación. Requiere haber corrido: python -m src.build_cache
# Uso: bash run_experiments.sh screening | cv <modelo1> <lr1> <modelo2> <lr2>
set -e
PY=${PY:-/c/Users/Luis/hms_data/.venv/Scripts/python}
mkdir -p results/logs

run () { tag=$1; shift; if [ -f "results/oof/$tag.csv" ]; then echo "=== $tag ya existe, se omite"; return; fi; echo ">>> $tag"; $PY -u -m src.train --tag "$tag" "$@" > "results/logs/$tag.log" 2>&1 || echo "FALLÓ $tag"; tail -2 "results/logs/$tag.log"; }

if [ "$1" = "screening" ]; then
  [ -f results/oof/hgb_bandas.csv ] || { $PY -u -m src.baselines > results/logs/baselines.log 2>&1; tail -5 results/logs/baselines.log; }
  # Arquitecturas preentrenadas (fold 0)
  run f0_effb0         --model efficientnet_b0.ra_in1k   --folds 0 --lr 1e-3
  run f0_resnet34      --model resnet34.a1_in1k          --folds 0 --lr 1e-3
  run f0_effb2         --model efficientnet_b2.ra_in1k   --folds 0 --lr 1e-3 --accum 2
  run f0_convnext_t    --model convnext_tiny.fb_in22k_ft_in1k --folds 0 --lr 2e-4 --accum 2
  run f0_regnety016    --model regnety_016.tv2_in1k      --folds 0 --lr 1e-3 --accum 2
  # Ablaciones con EfficientNet-B0
  run f0_effb0_scratch --model efficientnet_b0.ra_in1k   --folds 0 --lr 1e-3 --no-pretrained
  run f0_effb0_kaggle  --model efficientnet_b0.ra_in1k   --folds 0 --lr 1e-3 --input kaggle
  run f0_effb0_eeg     --model efficientnet_b0.ra_in1k   --folds 0 --lr 1e-3 --input eeg
elif [ "$1" = "cv" ]; then
  run cv_$2 --model "$3" --folds all --lr "$4" --save ${5:+--accum $5}
fi
