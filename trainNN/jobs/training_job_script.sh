#!/bin/bash
#SBATCH --account=sam77_h
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40GB
#SBATCH --partition=sla-prio
#SBATCH --job-name=train_bichrom
#SBATCH --output=slurm-%x-%j.out
umask 007

python ../run_bichrom.py -training_schema_yaml ../../../data/Ascl1_12hr_real/trainig_data/bichrom.yaml -len 500 -outdir train_out -nbins 10
