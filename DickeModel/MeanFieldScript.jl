using DifferentialEquations, LaTeXStrings, JLD2, Printf, PyPlot, LsqFit, RandomNumbers.Xorshifts, DiffEqGPU, CUDA, ProgressLogging, Profile, StaticArrays
include("HenryLib.jl")

a = 1
Sx = 2
Sy = 3
Sz = 4
function single_run_dicke_hetrodyne_meanfield(seed, λrel::Function; κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, Nspin=10000, tmax=500.0, num_points=50000) # all in MHz or uS

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


    function dicke!(du, u, p, t)
        du[a] = (-1im * Δc - κ) * u[a] - 2im * λ(t) / sqrt(Nspin) * u[Sx]
        du[Sx] = -ωz * u[Sy]
        du[Sy] = ωz * u[Sx] - 2 / sqrt(Nspin) * (conj(λ(t)) * u[a] + λ(t) * conj(u[a])) * u[Sz]
        du[Sz] = 2 / sqrt(Nspin) * (conj(λ(t)) * u[a] + λ(t) * conj(u[a])) * u[Sy]
    end

    function σ_dicke!(du, u, p, t)
        du[a] = sqrt(κ / 2)
        du[Sx] = 0.0
        du[Sy] = 0.0
        du[Sz] = 0.0
    end

    tspan = LinRange(0.0, tmax, num_points)

    saved_values = SavedValues(eltype(tspan), Tuple{typeof(inital),typeof(inital)})
    function saving_fnc(u, t, integrator)
        return (u, integrator.sol.W.u[end])
    end
    cb = SavingCallback(saving_fnc, saved_values; saveat=tspan)

    prob_dicke = SDEProblem{true}(dicke!, σ_dicke!, inital, (0.0, tmax), dt=1 / (5 * Δc))
    sol = solve(prob_dicke, SOSRI2(), maxiters=10^8, save_noise=true, seed=seed, progress=true, callback=cb)
    return saved_values, sol
end

seed = 24
λrelfnc(t) = 1.1
@btime sol = single_run_dicke_hetrodyne_meanfield(seed, λrelfnc, tmax=5.0)

# for seed in [42, 1337, 1729]#[42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240]#cat([42, 1337, 1729, 724, 333, 137, 31459, 271828, 24, 240], mod.(7727 * range(1, 90), 1087), dims=1)
#     for λrel in range(0.0, 3.0, 50)
#         println("Starting seed=$(seed)lambda=$(round(λrel,digits=3))")
#         κ = 2π * 0.15
#         λrelfnc(t) = λrel + 0.1 * sin(κ * t / 20.0)
#         sol = single_run_dicke_hetrodyne_meanfield(seed, λrelfnc; tmax=5000.0)
#         println("Simulation Complete")
#         jldsave("MeanFieldRsltsModulated2/seed=$(seed)lambda=$(round(λrel,digits=3)).jld2"; sol)
#         println("Finshed saving")
#     end
# end