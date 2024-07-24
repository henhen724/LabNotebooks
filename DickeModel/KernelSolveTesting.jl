using DifferentialEquations, DiffEqGPU, Plots, SparseArrays, DifferentialEquations.EnsembleAnalysis, StaticArrays

function lorenz(u, p, t)
    du1 = p[1] * (u[2] - u[1])
    du2 = u[1] * (p[2] - u[3]) - u[2]
    du3 = u[1] * u[2] - p[3] * u[3]
    du4 = 0
    return SVector{4}(du1, du2, du3, du4)
end

function multiplicative_noise(u, p, t)
    # println(du)
    du11 = 0.1 #* u[1]
    du22 = 0.4
    du41 = 1.0
    return @SMatrix[du11 0.0;
        0.0 du22;
        0.0 0.0;
        0.0 0.0;
        du41 0.0]
end

NRate = spzeros(4, 2)
NRate[1, 1] = 1
NRate[4, 1] = 1
NRate[2, 2] = 1

# CUDA.allowscalar(false)
u0 = @SVector ComplexF32[1.0; 0.0; 0.0; 0.0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = SDEProblem(lorenz, multiplicative_noise, u0, tspan, p, noise_rate_prototype=NRate)

prob_func = (prob, i, repeat) -> remake(prob, p=(@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
# EnsembleGPUArray(CUDA.CUDABackend())
# EnsembleCPUArray()
sol = solve(monteprob, SRA2(), EnsembleCPUArray(), trajectories=10_000, saveat=1.0f0)