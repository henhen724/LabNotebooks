#!/bin/bash
#
#SBATCH --job-name=modulate_dicke_production_run
#
#SBATCH --time=5:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2G

ml julia/1.10.0
julia --machine-file <(srun hostname -s) ./DickeModel/DickeTrajDistributed.jl
