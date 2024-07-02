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
using BenchmarkTools
include("HenryLib.jl")

function dicke_model_SS_rho_and_g1(λrel; κ=2*π*0.15, Δc=2*π*20, ωz=2*π*0.01, fockmax=4, Nspin=20, tmax=500.0, recordtimes=1000, rho_init=nothing)
    fb, sb, bases, a, Sx, Sy, Sz = make_operators(fockmax, Nspin)
    λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)
    λ = λrel * λc

    H0 = Δc * dagger(a) * a + ωz * Sz + 2 * λ * (dagger(a) + a) * Sx / sqrt(Nspin)
    C = sqrt(2 * κ) * a
    println("Starting Steady State Solve")
    if isnothing(rho_init)
        ρ_it = steadystate.iterative(H0, [C])
    else
        ρ_it = steadystate.iterative(H0, [C], rho0=rho_init)
    end
    tspan = range(0.0, tmax, recordtimes)
    print("S.S. Solve done. Now finding g1.")
    g1 = timecorrelations.correlation(tspan, ρ_it, H0, [C], dagger(a), a)

    return ρ_it, g1, fb, sb, bases, a, Sx, Sy, Sz
end

λrels = LinRange(0.0, 3.0, 30)
κ = 200.0
Nspin = 100

rslt = load("DickeModel/SSRslts/lambda=0.207Nspin=100.jld2")
rho_init = rslt["ρ_it"]
short_list = λrels[4:length(λrels)]

for λrel in short_list
    ρ_it, g1, fb, sb, bases, a, Sx, Sy, Sz = dicke_model_SS_rho_and_g1(λrel, κ=κ, Nspin=Nspin, rho_init=rho_init)
    global rho_init = ρ_it
    jldsave("DickeModel/SSRslts/lambda=$(round(λrel,digits=3))Nspin=$(Nspin).jld2"; ρ_it, g1)
end