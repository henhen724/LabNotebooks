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
    sb = SpinBasis(Nspin // 2)
    Sx = sigmax(sb) / 2
    Sy = sigmay(sb) / 2
    Sz = sigmaz(sb) / 2
    idOp = identityoperator(sb)
    ψ0 = spindown(sb)
    ψ0 = normalize!(ψ0)
    Q0 = 0 # charge on the photodiode at time 0
    cl0 = ComplexF64[Q0]
    ψ_sc0 = semiclassical.State(ψ0, cl0)
    tmax = 2000.0 # μs
    recordtimes = 4000
    tspan = range(0.0, tmax, recordtimes)
    stateG = copy(ψ_sc0)
    dstateG = copy(ψ_sc0)
    Nq = length(ψ_sc0.quantum)
    Nc = length(ψ_sc0.classical)
    Ntot = Nq + Nc
    u0 = zeros(ComplexF64, Ntot)
    semiclassical.recast!(u0, ψ_sc0)

    λ0s = [1.0, 3.0, 10.0]#LinRange(0.9, 1.5, 5)
    t_ramps = 10.0 .^ LinRange(2.0, 6.0, 10)
    # λmods = [0.05, 0.2, 0.5]#LinRange(0.0, 0.5, 5)
    # ωmods = 2π * 1e-6 * [500.0]#,1000.0]# 500.0, 1000.0, 2000.0, 10000.0, 50000.0] #Hz



    function norm_func(u, t, integrator)
        semiclassical.recast!(stateG, u)
        normalize!(stateG)
        semiclassical.recast!(u, stateG)
    end
    ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
        func_everystep=true,
        func_start=false)
    full_cb = OrdinaryDiffEq.CallbackSet(nothing, ncb, nothing)

    function prob_func(old_prob, i, repeat)
        κ = 2π * 0.15 # MHz
        Δc = -2π * 80 # MHz
        ωz = 2π * 0.01 # MHz
        idx1 = div(i - 1, length(t_ramps)) + 1
        idx2 = ((i - 1) % length(t_ramps)) + 1
        λ0 = λ0s[idx1]
        t_ramp = t_ramps[idx2]

        # u0 = zeros(ComplexF64, Ntot)
        # semiclassical.recast!(u0, ψ_sc0)
        gc = sqrt(ωz * (Δc^2 + κ^2) / abs(Nspin * Δc))
        grel!(t) = λ0 * smoothstep!(t / t_ramp)

        αplus = Δc / (-Δc + ωz - im * κ) + Δc / (-Δc - ωz - im * κ)
        αminus = Δc / (-Δc + ωz - im * κ) - Δc / (-Δc - ωz - im * κ)

        C0 = gc * sqrt(κ) / (2 * Δc) * (αplus * Sx + im * αminus * Sy)
        C!(t) = grel!(t) * C0

        H0T1 = ωz * Sz
        H0T2 = (gc)^2 / (4 * Δc) * Sx * (2 * real(αplus) * Sx - 2 * imag(αminus) * Sy)
        function H0!(t)
            return H0T1 - H0T2 * (grel!(t))^2
        end
        function H_nl!(ψ, t)
            Ct = C!(t)
            return im * expect(dagger(Ct), normalize(ψ)) * Ct - 0.5im * dagger(Ct) * Ct - 0.5im * expect(dagger(Ct), normalize(ψ)) * expect(Ct, normalize(ψ)) * idOp
        end
        fdet_heterodyne!(t, ψ) = H0!(t) + H_nl!(ψ, t)
        function fst_heterodyne!(t, ψ)
            Ct = C!(t)
            return [(Ct - expect(Ct, normalize(ψ)) * idOp) / sqrt(2), im * (Ct - expect(Ct, normalize(ψ)) * idOp) / sqrt(2)]
        end

        function f!(du, u, p, t)
            semiclassical.recast!(dstateG, du)
            semiclassical.recast!(stateG, u)
            timeevolution.dschroedinger_dynamic!(dstateG.quantum, fdet_heterodyne!, stateG.quantum, t)
            dstateG.classical[1] = expect(C!(t), normalize!(stateG.quantum))
            semiclassical.recast!(du, dstateG)
        end

        num_noise = length(fst_heterodyne!(0.0, ψ_sc0.quantum))
        noise_prototype = zeros(ComplexF64, (Ntot, num_noise))

        function g!(du, u, p, t)
            semiclassical.recast!(stateG, u)
            dx = @view du[1:Nq, :]
            stochastic.dschroedinger_stochastic(dx, t, stateG.quantum, fst_heterodyne!, dstateG.quantum, num_noise)
            du[Nq+1, 1] = 1.0 / sqrt(2)
            du[Nq+1, 2] = 1.0im / sqrt(2)
            du
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

sol = solve(ensembleprob, RKMilGeneral(; ii_approx=IICommutative()), EnsembleSplitThreads(), trajectories=length(λ0s) * length(t_ramps), adaptive=false,
    dt=(2 // 1)^(-11),
    save_everystep=false,
    save_start=true,
    save_end=false,
    saveat=tspan,
    callback=full_cb,
    progress=true)

@save joinpath(outdir, "ensemblesol.jld2") sol
