using QuantumOptics
using DiffEqNoiseProcess
using PyPlot
using LaTeXStrings
using Random
using DelimitedFiles
using NPZ, Printf
using LinearAlgebra
using DifferentialEquations
using SpecialFunctions
using SparseArrays
using StatsBase
using Optim
using JLD2
include("HenryLib.jl")

# below = 1.0 .- exp10.(range(-2.0, 0.0, 15))
# above = 1.0 .+ exp10.(range(-2.0, 0.5, 15))

κ = 200.0
Nspin = 100
tmax = 500.0

for seed in [42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240]#cat([42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240], mod.(7727 * range(1, 90), 1087), dims=1)
    for λrel in range(0.0, 3.0, 30)
        #print("Starting seed=$(seed)lambda=$(round(λrel,digits=3))")
        tout, psi_t, W, fb, sb, bases, a, Sx, Sy, Sz = single_run_dicke_hetrodyne(seed, λrel, κ=κ, tmax=tmax, Nspin=Nspin)
        jldsave("DickeModelRslts5/seed=$(seed)lambda=$(round(λrel,digits=3)).jld2"; tout, psi_t, W)
    end
end