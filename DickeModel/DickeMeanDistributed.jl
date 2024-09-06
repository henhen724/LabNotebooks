using Distributed, Dates #, ClusterManagers

@everywhere const save_dir = ENV["SCRATCH"] #@__DIR__

outdir = joinpath(save_dir, "results/$(Dates.today())/$(basename(@__FILE__)[begin:end-3])")
mkpath(outdir)
cp(@__FILE__, joinpath(outdir, "gen_script.jl"), force=true)

# addprocs(SlurmManager(4), partition="normal", t="00:3:00")

@everywhere begin 
    using Pkg
    function setup_worker()
        println("Hello from process $(myid()) on host $(gethostname()).")
        Pkg.activate(@__DIR__)
        Pkg.instantiate()
        Pkg.precompile()
        cd(save_dir)
    end
end
for i in workers()
    remotecall_wait(setup_worker, i)
end


@everywhere begin
    using QuantumOptics, OrdinaryDiffEq, StochasticDiffEq, DiffEqCallbacks, ProgressLogging, ProgressMeter, JLD2, Dates
    using TerminalLoggers: TerminalLogger
    using Logging: global_logger

    global_logger(TerminalLogger(right_justify=150))


    function smoothstep!(x)
        if x < 0
            return 0
        elseif x > 1
            return 1
        else
            return 3 * x^2 - 2 * x^3
        end
    end

    Nspin = 10
    seed = 1337
    λ0s = LinRange(0.5, 1.5, 15)
    λmods = [0.05, 0.2, 0.5]#LinRange(0.0, 0.5, 5)
    ωmods = 2π * 1e-6 * [500.0]#,1000.0]# 500.0, 1000.0, 2000.0, 10000.0, 50000.0] #Hz

    function prob_func(old_prob, i, repeat)
        κ = 2π * 0.15 # MHz
        Δc = 2π * 20 # MHz
        ωz = 2π * 0.01 # MHz
        idx1 = div(i - 1, length(λmods) * length(ωmods)) + 1
        idx2 = div((i - 1) % (length(λmods) * length(ωmods)), length(ωmods)) + 1
        idx3 = ((i - 1) % length(ωmods)) + 1
        λ0 = λ0s[idx1]
        λmod = λmods[idx2]
        ωmod = ωmods[idx3] # MHz

        # u0 = zeros(ComplexF64, Ntot)
        # semiclassical.recast!(u0, ψ_sc0)
        gc = sqrt(ωz * (Δc^2 + κ^2) / abs(Nspin * Δc))
        grel!(t) = (λ0 + λmod * sin(ωmod * t)) * smoothstep!(t / 200.0)

        αplus = Δc / (-Δc + ωz - im * κ) + Δc / (-Δc - ωz - im * κ)
        αminus = Δc / (-Δc + ωz - im * κ) - Δc / (-Δc - ωz - im * κ)

        #      Sx   Sy   Sz   Q
        u0 = [0.0, 0.0, 1.0, 0.0]

        function f!(du, u, p, t)
            du[1] = -(ωz + real(αminus)*(grel!(t)*gc)^2/(4*Δc))*u[2] + (grel!(t)*gc)^2 * u[3]/(2*Δc)*(imag(αminus) - κ/Δc * real(conj(αplus)*αminus))*u[1] - (grel!(t)*gc)^2 * κ/(4*Δc) * (imag(conj(αplus)*αminus)*u[2]+conj(αminus)*αminus*u[1])
            du[2] = (ωz + real(αminus)*(grel!(t)*gc)^2/(4*Δc))*u[1] + (grel!(t)*gc)^2 * u[3]/(2*Δc)*(2*real(αminus)*u[1] - (imag(αminus) + κ/Δc * real(conj(αplus)*αminus))*u[2]) - (grel!(t)*gc)^2 * κ/(4*Δc)*(imag(conj(αplus)*αminus)*u[1]+conj(αplus)*αplus*u[2])
            du[3] = (grel!(t)*gc)^2 /(2*Δc)*(imag(αminus)*(u[2]^2 - u[1]^2) + κ/Δc * real(conj(αplus)*αminus)*(u[2]^2 + u[1]^2) - 2*real(αplus)*u[2]*u[1]) - (grel!(t)*gc)^2 * κ/(4*Δc)*(conj(αplus)*αplus + conj(αminus)*αminus)*u[3]
            du[4] = grel!(t)*gc * sqrt(κ) / (2 * Δc) * (αplus * u[1] + im * αminus * u[2])
        end

        num_noise = length(fst_heterodyne!(0.0, ψ_sc0.quantum))
        noise_prototype = zeros(ComplexF64, (Ntot, num_noise))

        function g!(du, u, p, t)
            du[,1]
            du[,2]
        end

        CurrW = StochasticDiffEq.RealWienerProcess!(0.0, zeros(num_noise), save_everystep=false)

        prob = SDEProblem(f!, g!, u0, (tspan[begin], tspan[end]); noise_rate_prototype=noise_prototype, noise=CurrW)
        return prob
    end


    outdir = joinpath(save_dir, "results/$(Dates.today())/$(basename(@__FILE__)[begin:end-3])")
    function output_func(sol, i)
        new_sol = Dict(:t => sol.t, :u => sol.u)
        try
            jldsave(joinpath(outdir, "sol_$(i).jld2"); sol=new_sol)
        catch e
            println("Ensemble $(i) failed to save to a solution file.")
            println(e)
        end
        return (new_sol, false)
    end

    init_prob = prob_func(nothing, 1, 1)
    ensembleprob = EnsembleProblem(init_prob, prob_func=prob_func, output_func=output_func)
end

sol = solve(ensembleprob, RKMilGeneral(; ii_approx=IICommutative()), EnsembleSplitThreads(), trajectories=length(λ0s) * length(λmods) * length(ωmods), adaptive=false,
    dt=(2 // 1)^(-11),
    save_everystep=false,
    save_start=true,
    save_end=false,
    saveat=tspan,
    callback=full_cb,
    progress=true)

@save joinpath(outdir, "ensemblesol.jld2") sol
