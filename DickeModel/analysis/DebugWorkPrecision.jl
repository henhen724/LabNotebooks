using StochasticDiffEq, Plots, DiffEqDevTools
# Linear SDE system
f_lin = function (du, u, p, t)
    du[1] = -0.5 * u[1]
end

g_lin = function (du, u, p, t)
    du[1] = im * u[1]
end

lin_analytic = function (u₀, p, t, Wt)
    u₀ .* exp.(im .* Wt)
end

tspan = (0.0, 10.0)
noise = StochasticDiffEq.RealWienerProcess!(0.0, [0.0])
prob = SDEProblem(SDEFunction(f_lin, g_lin; analytic=lin_analytic), ComplexF64[1.0], tspan, noise=noise)

reltols = 1.0 ./ 10.0 .^ (1:3)
abstols = reltols#[0.0 for i in eachindex(reltols)]

setups = [Dict(:alg => EM(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false),
    Dict(:alg => LambaEM(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false),
    Dict(:alg => EulerHeun(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false),
    Dict(:alg => LambaEulerHeun(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false),
    Dict(:alg => RKMilCommute(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false),
    Dict(:alg => RKMilGeneral(), :dts => 1.0 ./ 2.0 .^ ((1:length(reltols)) .+ 3), :adaptive => false)]

names = ["EM", "LambaEM", "EulerHeun", "LambaEulerHeun", "RKMilCommute", "RKMilGeneral"]

# sol = solve(prob, setups[1][:alg];abstol = 1.0/ (10.0^5),
#             reltol = 1.0/ (10.0^7), dt = 1.0/2.0^6, adaptive = false,
#             timeseries_errors = true,
#             dense_errors = false)

wp = WorkPrecisionSet(prob, abstols, reltols, setups, 1 // 2^(10); numruns=1, numruns_error=10, trajectories=3, names=names, maxiters=1e7, error_estimate=:l2)
plot(wp)