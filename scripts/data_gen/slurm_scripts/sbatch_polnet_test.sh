#!/bin/bash
 
#SBATCH -p standard96s:shared
#SBATCH --job-name=debug_voi_vsize
#SBATCH -o out/test9/slurm-%j.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
 
source ~/.bashrc
micromamba activate -p /mnt/lustre-grete/projects/nim00020/sage/envs/polnet-synaptic

OUT_DIR=out/test9
mkdir -p "$OUT_DIR"

python all_features_argument.py \
  --out_dir $OUT_DIR \
  --ntomos 1 \
  --detector_snr 0.10 0.15 \
  --voi_vsize 13.8 \
  --voi_shape 400 400 200


  # test 8 - deactivate mt generation - works
  # test 9 - normal network generation