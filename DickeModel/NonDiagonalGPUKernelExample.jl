using DifferentialEquations, DiffEqGPU, CUDA, StaticArrays, SafeTestsets, Test

@info "Non-Diagonal Noise"

function f(u, p, t)
    return 0.00 .* u
end

function g(u, p, t)
    du1_1 = 1.0u[1]
    du2_1 = 1.0u[1]
    du3_1 = 0.0u[1]
    du4_1 = 0.0u[1]
    du1_2 = 0.0u[2]
    du2_2 = 0.0u[2]
    du3_2 = 1.0u[2]
    du4_2 = 1.0u[2]
    return SMatrix{4,2}(du1_1, du2_1, du3_1, du4_1, du1_2, du2_2, du3_2, du4_2)
end

u0 = @SVector ones(Float32, 4)
dt = Float32(1 // 2^(8))
noise_rate_prototype = @SMatrix zeros(Float32, 4, 2)
prob = SDEProblem(f, g, u0, (0.0f0, 1.0f0), noise_rate_prototype=noise_rate_prototype)
monteprob = EnsembleProblem(prob)

sol = solve(
    monteprob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()), dt=dt, trajectories=10,
    adaptive=false)

@test sol.converged == true
@test sol[1][end][1] == sol[1][end][2]
@test sol[1][end][3] == sol[1][end][4]