#!/bin/bash
#SBATCH --account=sam77_h
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --partition=sla-prio
#SBATCH --job-name=test
#SBATCH --output=slurm-%x-%j.out
umask 007

cd ..
pwd
ls

