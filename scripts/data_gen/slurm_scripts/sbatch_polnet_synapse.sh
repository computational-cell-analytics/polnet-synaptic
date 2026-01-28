#!/bin/bash
 
#SBATCH -p standard96s:shared
#SBATCH --job-name=test_synapse_parallel
#SBATCH -o out/synapse/run1/slurm-%j.out
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
 
source ~/.bashrc
micromamba activate -p /mnt/lustre-grete/projects/nim00020/sage/envs/polnet-synaptic

OUT_DIR=out/synapse/run1
mkdir -p "$OUT_DIR"

python all_features_short_sn_parallel.py \
  --out_dir $OUT_DIR \
  --ntomos 1 \
  --voi_vsize 7.56 \
  --voi_shape 1024 1024 375

  # test 6 1 tomo, 7.56A - took more than 24h
  # test 7 3 tomos, 7.56A