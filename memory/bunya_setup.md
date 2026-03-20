---
name: bunya_hpc_setup
description: UQ Bunya HPC uses SLURM (not PBS), user s4948012, module python/3.10.4-gcccore-11.3.0, venv-based envs, gpu_cuda partition with qos=gpu
type: reference
---

Bunya HPC (UQ) connection and job setup:
- SSH: `ssh s4948012@bunya.rcc.uq.edu.au`
- Scheduler: **SLURM** (sbatch/salloc/squeue), NOT PBS
- GPU partition: `--partition=gpu_cuda --qos=gpu`
- GPU allocation example: `--gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00`
- Python module: `module load python/3.11.3-gcccore-12.3.0` (sleap-nn requires >=3.11)
- Environment: venv-based (`source .venv/bin/activate`), not conda
- Existing project path: `~/multi-object-computer-vision-tracking`
- Interactive: `salloc ...` then `srun --pty bash`
