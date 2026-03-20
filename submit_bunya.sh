#!/bin/bash
#SBATCH --job-name=sleap_train
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=sleap_%j.log
#SBATCH --error=sleap_%j.log

###############################################################################
# SLEAP Training on Bunya (UQ HPC)
#
# --- Connection ---
#   ssh s4948012@bunya.rcc.uq.edu.au
#   cd ~/multi-object-computer-vision-tracking
#
# --- Scheduler: SLURM (NOT PBS) ---
#   Partition:  gpu_cuda
#   QoS:        gpu
#   GPU alloc:  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=06:00:00
#   Python:     module load python/3.10.4-gcccore-11.3.0
#   Env:        source .venv/bin/activate  (venv, NOT conda)
#
# --- Submit jobs ---
#   sbatch submit_bunya.sh 20        # 20 epochs
#   sbatch submit_bunya.sh 50        # 50 epochs
#   sbatch submit_bunya.sh           # default: 50 epochs
#   bash run_both.sh                 # both 20 and 50 at once
#
# --- Monitor ---
#   squeue -u $USER                  # check job status
#   scancel <jobid>                  # cancel a job
#   tail -f sleap_<jobid>.log        # watch output
#
# --- Results ---
#   results_20ep/
#   results_50ep/
###############################################################################

EPOCHS=${1:-50}

cd ~/multi-object-computer-vision-tracking || exit 1

module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

echo "============================================"
echo "SLEAP Training Job"
echo "  Epochs:    $EPOCHS"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  GPU:"
nvidia-smi
echo "============================================"

python train_sleap.py \
    --epochs "$EPOCHS" \
    --batch_size 4 \
    --lr 0.0001 \
    --output_dir "results_${EPOCHS}ep"

echo "Job finished at $(date)"
