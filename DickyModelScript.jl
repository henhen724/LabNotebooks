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

function mb(op, bases, idx)

    numHilberts = size(bases, 1)

    if idx == 1
        mbop = op
    else
        mbop = identityoperator(bases[1])
    end

    for i = 2:numHilberts

        if i == idx
            mbop = tensor(mbop, op)
        else
            mbop = tensor(mbop, identityoperator(bases[i]))
        end

    end

    return mbop
end

fockmax = 10
Nspin = 20
fb = FockBasis(fockmax)
sb = SpinBasis(Nspin // 2)
bases = [sb, fb]
a = mb(destroy(fb), bases, 2)
Sx = mb(sigmax(sb), bases, 1) / 2
Sy = mb(sigmay(sb), bases, 1) / 2
Sz = mb(sigmaz(sb), bases, 1) / 2

function single_run(seed, λrel, κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01)# ALL IN MHz
    # κ = 2π*0.15 # MHz
    # Δc = 2π*20 # MHz
    # ωz = 2π*0.01 # MHz
    λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)
    λ = λrel * λc
    free_energy(α) = Δc * (α[1]^2) - Nspin * sqrt((ωz^2) / 4 + (4 * α[1]^2) * (λ^2) / Nspin)
    result = optimize(free_energy, ones(1), BFGS())
    αinit = -Optim.minimizer(result)[1]
    φinit = atan(2 * λ * abs(αinit) / sqrt(Nspin), ωz / 2)

    dt = 0.0001
    recordtimes = 5000
    tspan = range(0.0, 1000.0 / (2 * κ), recordtimes)

    C = sqrt(2 * κ) * a
    H0 = Δc * dagger(a) * a + ωz * Sz + 2 * λ * (dagger(a) + a) * Sx / sqrt(Nspin)

    ψ0 = tensor(spindown(sb), coherentstate(fb, αinit))# + coherentstate(b,-0.7))
    ψ0 = exp(im * φinit * Sy) * ψ0
    ψ0 = normalize!(ψ0)
    @assert abs(expect(mb(projector(basisstate(fb, 11)), bases, 2), ψ0)) < 0.01
    println("Running with seed: ", seed)
    Hs = C
    Y = dagger(C)
    CdagC = -0.5im * dagger(C) * C
    H_nl(ψ) = im * expect(Y, normalize(ψ)) * Hs + CdagC

    fdet_homodyne(t, ψ) = H0 + H_nl(ψ)
    fst_homodyne(t, ψ) = [Hs]

    W = WienerProcess(0.0, im * 0.0, im * 0.0)

    tout, psi_t = stochastic.schroedinger_dynamic(tspan, ψ0, fdet_homodyne, fst_homodyne; dt=dt, normalize_state=true, noise=W, seed=seed, alg=SOSRI2(), reltol=10^-4, abstol=10^-4, maxiters=10^8)
    return tout, psi_t, W
end

# below = 1.0 .- exp10.(range(-2.0, 0.0, 15))
# above = 1.0 .+ exp10.(range(-2.0, 0.5, 15))

for seed in [42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240]#cat([42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240], mod.(7727 * range(1, 90), 1087), dims=1)
    for λrel in range(0.0, 5.0, 30)
        #print("Starting seed=$(seed)lambda=$(round(λrel,digits=3))")
        tout, psi_t, W = single_run(seed, λrel)
        jldsave("DickeModelRslts2/seed=$(seed)lambda=$(round(λrel,digits=3)).jld2"; tout, psi_t, W)
    end
end

