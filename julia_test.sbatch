#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --time 5:0

ml julia
julia --machine-file <(srun hostname -s)  ./julia_parallel_test.jl
