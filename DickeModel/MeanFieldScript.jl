using DifferentialEquations
using LaTeXStrings
using JLD2
using Printf
using PyPlot
using LsqFit
using RandomNumbers.Xorshifts
include("HenryLib.jl")

# observable indeces
a = 1
Sx = 2
Sy = 3
Sz = 4
function single_run_dicke_hetrodyne_meanfield(seed, λrel::Function; κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, fockmax=4, Nspin=10000, tmax=500.0) # all in MHz or uS
    println("Running simulation")
    λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)
    λ(t) = λc * λrel(t)

    inital = [0im, 0.0, 0.0, -Nspin / 2.0]
    if λrel(0) > 1.0
        θ_α = atan(-κ / Δc)
        Szinit = -ωz * Nspin / (8 * λ(0)^2) * sqrt(Δc^2 + κ^2) / cos(θ_α)
        Sxinit = sqrt(Nspin^2 / 4 - Szinit^2)
        αinit = 2im * λ(0) / (sqrt(Nspin) * (-1im * Δc - κ)) * Sxinit

        inital = [αinit, Sxinit, 0.0, Szinit]
    end


    function dicke(du, u, p, t)
        du[a] = (-1im * Δc - κ) * u[a] - 2im * λ(t) / sqrt(Nspin) * u[Sx]
        du[Sx] = -ωz * u[Sy]
        du[Sy] = ωz * u[Sx] - 2 / sqrt(Nspin) * (conj(λ(t)) * u[a] + λ(t) * conj(u[a])) * u[Sz]
        du[Sz] = 2 / sqrt(Nspin) * (conj(λ(t)) * u[a] + λ(t) * conj(u[a])) * u[Sy]
    end

    function σ_dicke(du, u, p, t)
        du[a] = sqrt(κ / 2)
        du[Sx] = 0.0
        du[Sy] = 0.0
        du[Sz] = 0.0
    end

    rng = Xoroshiro128Plus(seed)
    W = WienerProcess(0.0, im * 0.0, im * 0.0, rng=rng)

    # can_dual(::Type{ComplexF64}) = true

    prob_dicke = SDEProblem{true}(dicke, σ_dicke, inital, (0.0, tmax), noise=W, dt=1 / (5 * Δc))
    sol = solve(prob_dicke, SOSRI2(), maxiters=10^8, saveat=0.01)
    return sol, W
end

for seed in [42, 1337, 1729]#[42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240]#cat([42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240], mod.(7727 * range(1, 90), 1087), dims=1)
    for λrel in range(0.0, 3.0, 50)
        #print("Starting seed=$(seed)lambda=$(round(λrel,digits=3))")
        κ=2π * 0.15
        λrelfnc(t) = λrel + 0.1 * sin(κ * t / 20.0)
        sol, W = single_run_dicke_hetrodyne_meanfield(seed, λrelfnc; tmax=5000.0)
        println("Simulation Complete")
        WArr = [sol.W(t)[1] for t in sol.t]
        jldsave("MeanFieldRsltsModulated/seed=$(seed)lambda=$(round(λrel,digits=3)).jld2"; sol.t, sol.u, WArr)
        println("Finshed svaing")
    end
end