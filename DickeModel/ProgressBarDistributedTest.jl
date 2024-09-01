using Distributed

# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
end

# load dependencies in a *separate* @everywhere block
@everywhere begin
    # load dependencies
    using DifferentialEquations, ProgressLogging, ProgressMeter, TerminalLoggers
    using Logging: global_logger

    global_logger(TerminalLogger(right_justify=150))
    function lorenz!(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end
    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 100.0)
    prob = ODEProblem(lorenz!, u0, tspan)
    ensembleprob = EnsembleProblem(prob)
end

using Dates

outdir = joinpath(@__DIR__, "results/$(Dates.today())")
mkpath(outdir)
cp(@__FILE__, joinpath(outdir, "gen_script.jl"), force=true)

# files to process
sol = solve(ensembleprob, Tsit5(), EnsembleDistributed(), trajectories=1000, progress=true, progress_name="Ensemble Solve", progress_id=1)