#!/usr/bin/env bash
# Fase 2: entrenamiento solo con alta calidad (>=10 votos), optimización bayesiana y test en <10 votos.
PY=${PY:-/c/Users/Luis/hms_data/.venv/Scripts/python}
L=results/logs; mkdir -p $L
step () { echo ">>> $1 $(date +%H:%M)"; shift; "$@" || echo "FALLÓ"; }
[ -f results/oof/hq_effb0_default.csv ] || step control_effb0 $PY -u -m src.hq cv --arch effb0 --params default --tag hq_effb0_default > $L/hq_effb0_default.log 2>&1
tail -1 $L/hq_effb0_default.log
step tune_effb0      $PY -u -m src.hq tune --arch effb0 --trials 30      > $L/tune_effb0.log 2>&1
grep MEJOR $L/tune_effb0.log
step tune_convnext   $PY -u -m src.hq tune --arch convnext_t --trials 20 > $L/tune_convnext_t.log 2>&1
grep MEJOR $L/tune_convnext_t.log
step cv_effb0        $PY -u -m src.hq cv --arch effb0 --params best      > $L/hq_effb0.log 2>&1
tail -1 $L/hq_effb0.log
step cv_convnext     $PY -u -m src.hq cv --arch convnext_t --params best > $L/hq_convnext_t.log 2>&1
tail -1 $L/hq_convnext_t.log
step ensamble $PY -u -m src.ensemble_opt > $L/ensemble_opt.log 2>&1
tail -25 $L/ensemble_opt.log
step auditoria $PY -u -m src.imbalance_audit > $L/imbalance_audit.log 2>&1
echo "FIN $(date +%H:%M)"
