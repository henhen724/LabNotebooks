using DifferentialEquations, DiffEqGPU, CUDA, ProgressLogging, Plots, SparseArrays, DifferentialEquations.EnsembleAnalysis, StaticArrays

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    du[4] = 0
end

function multiplicative_noise(du, u, p, t)
    # println(du)
    du[1, 1] = 0.1 #* u[1]
    du[2, 2] = 0.4
    du[4, 1] = 1.0
end

NRate = spzeros(4, 2)
NRate[1, 1] = 1
NRate[4, 1] = 1
NRate[2, 2] = 1

# CUDA.allowscalar(false)
u0 = ComplexF32[1.0; 0.0; 0.0; 0.0]
tspan = (0.0f0, 10.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = SDEProblem(lorenz, multiplicative_noise, u0, tspan, p, noise_rate_prototype=NRate)
solve(prob, SRA2(), saveat=1.0f0)