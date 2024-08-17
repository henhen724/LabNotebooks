#!/bin/bash
#
#SBATCH --job-name=test_julia
#
#SBATCH --time=10:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

ml julia
julia --machine-file <(srun hostname -s) ./DickeModel/test_script.jl
