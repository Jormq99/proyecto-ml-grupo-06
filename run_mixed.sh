#!/usr/bin/env bash
# Iteración 3: alta + baja calidad. Enfoque A (modelo único / dos cabezas) y B (especialistas + enrutamiento).
PY=${PY:-/c/Users/Luis/hms_data/.venv/Scripts/python}
L=results/logs; mkdir -p $L
step () { echo ">>> $1 $(date +%H:%M)"; shift; "$@" || echo "FALLÓ"; }
M="$PY -u -m src.mixed"
step tune_single_effb0 $M tune --arch effb0 --mode single --trials 16 > $L/tune_mix_single_effb0.log 2>&1
grep MEJOR $L/tune_mix_single_effb0.log
step cv_single_effb0   $M cv --arch effb0 --mode single > $L/mix_single_effb0.log 2>&1
grep -E "^fold" $L/mix_single_effb0.log
step cv_dual_effb0     $M cv --arch effb0 --mode dual --params mix_single_effb0 --override '{"gate_weight":0.3}' > $L/mix_dual_effb0.log 2>&1
grep -E "^fold" $L/mix_dual_effb0.log
step tune_lq_effb0     $M tune --arch effb0 --mode lq --trials 10 > $L/tune_mix_lq_effb0.log 2>&1
grep MEJOR $L/tune_mix_lq_effb0.log
step cv_lq_effb0       $M cv --arch effb0 --mode lq > $L/mix_lq_effb0.log 2>&1
grep -E "^fold" $L/mix_lq_effb0.log
step tune_single_convnext $M tune --arch convnext_t --mode single --trials 7 > $L/tune_mix_single_convnext_t.log 2>&1
grep MEJOR $L/tune_mix_single_convnext_t.log
step cv_single_convnext   $M cv --arch convnext_t --mode single > $L/mix_single_convnext_t.log 2>&1
grep -E "^fold" $L/mix_single_convnext_t.log
step cv_lq_convnext       $M cv --arch convnext_t --mode lq --params base --override '{"epochs":10}' > $L/mix_lq_convnext_t.log 2>&1
grep -E "^fold" $L/mix_lq_convnext_t.log
[ -f src/routing.py ] && step routing $PY -u -m src.routing > $L/routing.log 2>&1
step cv_dual_convnext     $M cv --arch convnext_t --mode dual --params mix_single_convnext_t --override '{"gate_weight":0.3}' > $L/mix_dual_convnext_t.log 2>&1
grep -E "^fold" $L/mix_dual_convnext_t.log
[ -f src/routing.py ] && step routing $PY -u -m src.routing > $L/routing.log 2>&1
echo "FIN $(date +%H:%M)"
