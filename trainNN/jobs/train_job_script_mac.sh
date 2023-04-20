#!/bin/bash
umask 007

ulimit -n 2048
ulimit -n

SECONDS=0

nohup time -h python -u ../run_bichrom.py -training_schema_yaml ../../../data/Ascl1_12hr_real/trainig_data/bichrom.yaml -len 500 -outdir  train_out_mac_epochs_15_check -nbins 10 -net bimodal -epochs 15 > job_epochs_15_check.out&

#python -u ../run_bichrom.py -training_schema_yaml ../../../data/Ascl1_12hr_real/trainig_data/bichrom.yaml -len 500 -outdir  train_out_mac_epochs_15_check -nbins 10 -net bimodal -epochs 15

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED
