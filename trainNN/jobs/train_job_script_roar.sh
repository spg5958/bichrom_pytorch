#!/bin/bash
#PBS -W umask=0007  
#PBS -W group_list=sam77_collab
#PBS -j oe
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=4:gpus=1:rhel7
#PBS -l pmem=40gb
#PBS -A sam77_i_g_gc_default

SECONDS=0

conda activate pytorch

# Go to the submission directory
cd $PBS_O_WORKDIR
pwd

module load cuda/10.2.89

# On RHEL7 nodes, if you got the error: Could not load dynamic library 'libcudart.so.10.1' when using tensorflow, you can try the following steps to let GPU be used.
## mkdir -p ~/usr/lib64/
## ln -s /opt/aci/sw/cuda/10.2.89_gcc-4.8.5-pjb/lib64/libcudart.so.10.2.89 ~/usr/lib64/libcudart.so.10.1
## export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:~/usr/lib64"

# Run the job itself
#=======
# To test your CUDA and NVIDIA GPU driver version: 
nvidia-smi

python ../run_bichrom.py -training_schema_yaml ../../../data/Ascl1_12hr_real/trainig_data/bichrom.yaml -len 500 -outdir train_out -nbins 10

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED
