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

function make_operators(fockmax, Nspin)
    fb = FockBasis(fockmax)
    sb = SpinBasis(Nspin // 2)
    bases = [sb, fb]
    a = mb(destroy(fb), bases, 2)
    Sx = mb(sigmax(sb), bases, 1) / 2
    Sy = mb(sigmay(sb), bases, 1) / 2
    Sz = mb(sigmaz(sb), bases, 1) / 2
    return fb, sb, bases, a, Sx, Sy, Sz
end

function single_run_dicke_hetrodyne(seed, λrel; κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, fockmax=4, Nspin=20, tmax=500.0)# ALL IN MHz
    fb, sb, bases, a, Sx, Sy, Sz = make_operators(fockmax, Nspin)

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
    tspan = range(0.0, tmax, recordtimes)

    C = sqrt(2 * κ) * a
    H0 = Δc * dagger(a) * a + ωz * Sz + 2 * λ * (dagger(a) + a) * Sx / sqrt(Nspin)

    ψ0 = tensor(spindown(sb), coherentstate(fb, αinit))# + coherentstate(b,-0.7))
    ψ0 = exp(im * φinit * Sy) * ψ0
    ψ0 = normalize!(ψ0)
    @assert abs(expect(mb(projector(basisstate(fb, fockmax + 1)), bases, 2), ψ0)) < 0.01
    print("Running with seed: ", seed)
    Hs = C
    Y = dagger(C)
    CdagC = -0.5im * dagger(C) * C
    H_nl(ψ) = im * expect(Y, normalize(ψ)) * Hs + CdagC

    fdet_homodyne(t, ψ) = H0 + H_nl(ψ)
    fst_homodyne(t, ψ) = [Hs]

    W = WienerProcess(0.0, im * 0.0, im * 0.0)

    tout, psi_t = stochastic.schroedinger_dynamic(tspan, ψ0, fdet_homodyne, fst_homodyne; dt=dt, normalize_state=true, noise=W, seed=seed, alg=SOSRI2(), reltol=10^-4, abstol=10^-4, maxiters=10^8)
    return tout, psi_t, W, fb, sb, bases, a, Sx, Sy, Sz
end

function window(signal, time, dt)
    Ntime = length(time)
    Ttotal = time[Ntime] - time[1]
    Nfilt = Int64(ceil(Ttotal / dt))
    filtered = im * zeros(ComplexF64, Nfilt)
    time_indx = 1
    for i = 1:Nfilt
        window_size = 0
        while time_indx <= Ntime && time[time_indx] < dt * (i + 1)
            # print(time_indx, " ", Ntime,"\n")
            window_size += 1
            # print(length(filtered), " ", Nfilt, "\n")
            filtered[i] += signal[time_indx]
            time_indx += 1
        end
        filtered[i] /= window_size
    end
    return filtered
end

function exp_filter(signal, time_scale, dt)
    N = length(signal)
    filtered = zeros(typeof(signal[1]), N)
    for i = 2:N
        filtered[i] = filtered[i-1] * exp(-dt / time_scale) + signal[i] * (1 - exp(-dt / time_scale))
        # filtered[i] = filtered[i-1]*0.9 + 0.2*homodyne_[i]
    end
    return filtered
end

function two_point_correlator(signal, time, dt, time_steps::Int; prefilter::Union{Nothing,Float64}=nothing, delay_start::Union{Nothing,Integer}=nothing)
    if delay_start != nothing
        signal = signal[delay_start:length(signal)]
        time = time[delay_start:length(time)]
    end
    # signal = window(signal, time, dt)
    N = length(signal)
    # if prefilter != nothing
    #     signal = filter(signal, prefilter, dt)
    #     N = N-1
    # end
    rslt = zeros(ComplexF64, time_steps)
    for i = 0:(time_steps-1)
        rslt[i+1] = sum(signal[1:N-i] .* conj(signal[i+1:N])) / (N - i)
    end
    return rslt
end

function make_white_noise(time, W)
    dt = time[2] - time[1]
    W.dt = dt
    u = nothing
    p = nothing
    white_noise = zeros(ComplexF64, length(time))
    calculate_step!(W, dt, u, p)
    for i in 1:(length(time))
        accept_step!(W, dt, u, p)
        white_noise[i] = (W.u[i+1] - W.u[i]) / dt
    end
    return white_noise
end