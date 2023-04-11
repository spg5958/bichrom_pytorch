#!/bin/bash
#SBATCH --account=open
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=80GB
#SBATCH --partition=open
#SBATCH --job-name=train_bichrom
#SBATCH --output=slurm-%x-%j.out
umask 007

SECONDS=0

python ../run_bichrom.py -training_schema_yaml ../../../data/Ascl1_12hr_real/trainig_data/bichrom.yaml -len 500 -outdir train_out_roar_collab_epochs_15_r2 -nbins 10 -net bimodal -epochs 15

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED
