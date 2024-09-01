using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
end


@everywhere begin
    using QuantumOptics, OrdinaryDiffEq, StochasticDiffEq, DiffEqCallbacks, ProgressLogging, ProgressMeter, JLD2
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
    tmax = 50.0#20000.0 # μs
    recordtimes = 50#20000
    tspan = range(0.0, tmax, recordtimes)
    stateG = copy(ψ_sc0)
    dstateG = copy(ψ_sc0)
    Nq = length(ψ_sc0.quantum)
    Nc = length(ψ_sc0.classical)
    Ntot = Nq + Nc
    u0 = zeros(ComplexF64, Ntot)
    semiclassical.recast!(u0, ψ_sc0)

    λ0s = LinRange(0.8, 1.2, 5)
    λmods = LinRange(0.0, 0.5, 5)
    ωmods = 2π * 1e-6 * [100.0]#, 500.0, 1000.0, 2000.0]#, 10000.0, 50000.0] #Hz


    function norm_func(u, t, integrator)
        semiclassical.recast!(stateG, u)
        normalize!(stateG)
        semiclassical.recast!(u, stateG)
    end
    # function fout(t, state)
    #     copy(state)
    # end
    # function fout_(x, t, integrator)
    #     semiclassical.recast!(stateG, x)
    #     copy(stateG)
    # end
    # Base.@pure pure_inference(fout, T) = Core.Compiler.return_type(fout, T)
    # out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(ψ_sc0)})
    ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
        func_everystep=true,
        func_start=false)
    # out = DiffEqCallbacks.SavedValues(eltype(tspan), out_type)
    # scb = DiffEqCallbacks.SavingCallback(fout_, out, saveat=tspan,
    #     save_everystep=false,
    #     save_start=false,
    #     tdir=first(tspan) < last(tspan) ? one(eltype(tspan)) : -one(eltype(tspan)))
    full_cb = OrdinaryDiffEq.CallbackSet(nothing, ncb, nothing)

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

        CurrW = StochasticDiffEq.RealWienerProcess!(0.0, zeros(num_noise))

        prob = SDEProblem(f!, g!, u0, (tspan[begin], tspan[end]); noise_rate_prototype=noise_prototype, noise=CurrW)
        return prob
    end

    function output_func(sol, i)
        return (Dict(:t => sol.t, :u => sol.u), false)
    end

    init_prob = prob_func(nothing, 1, 1)
    ensembleprob = EnsembleProblem(init_prob, prob_func=prob_func, output_func=output_func)
end

using Dates

outdir = joinpath(@__DIR__, "results/$(Dates.today())/$(basename(@__FILE__)[begin:end-3])")
mkpath(outdir)
cp(@__FILE__, joinpath(outdir, "gen_script.jl"), force=true)

sol = solve(ensembleprob, RKMilGeneral(; ii_approx=IICommutative()), EnsembleDistributed(), trajectories=length(λ0s) * length(λmods) * length(ωmods), adaptive=false,
    dt=(2 // 1)^(-11),
    save_everystep=false,
    save_start=true,
    save_end=false,
    saveat=tspan,
    callback=full_cb,
    progress=true)

@save joinpath(outdir, "ensemblesol.jld2") sol