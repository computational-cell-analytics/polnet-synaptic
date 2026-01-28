#!/bin/bash
#SBATCH -p standard96s:shared
#SBATCH --job-name=snr_array
#SBATCH --array=0-9%5
#SBATCH --output=out/run4/logs/slurm-%A_%a.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

PARENT_DIR="out/run4"
OUT_DIR="$PARENT_DIR/task$SLURM_ARRAY_TASK_ID"
LOG_DIR="$PARENT_DIR/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

source ~/.bashrc
micromamba activate polnet

# voxel size deepict
VOI_VSIZE=13.8

# helix input files
MT_IN="in_helix/mt.hns"

# select for current task
ACTIN_IN=""

srun python all_features_argument.py \
  --out_dir $OUT_DIR \
  --ntomos 2 \
  --voi_vsize $VOI_VSIZE \
  --detector_snr 0.10 0.15 \
  --voi_shape 628 628 184 \
  --helix_list $ACTIN_IN $MT_IN
