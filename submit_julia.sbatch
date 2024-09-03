#!/bin/bash
#
#SBATCH --job-name=traj_run
#
#SBATCH -n 9
#SBATCH --ntasks-per-node 1
#SBATCH --time 09:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error logs/%x-%j.err

ml julia/1.10.0
julia --machine-file <(srun hostname -s) $HOME/LabNotebooks/DickeModel/DickeTrajDistributed.jl
