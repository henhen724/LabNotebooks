using QuantumOptics, OrdinaryDiffEq, StochasticDiffEq, DiffEqCallbacks, ProgressLogging, ProgressMeter, Dates, JLD2, BenchmarkTools

function mb(op, bases, idx)

    numHilberts = size(bases, 1)

    if idx == 1
        mbop = op
    else
        mbop = identityoperator(bases[1])
    end

    for i = 2:numHilberts

        if i == idx
            mbop = tensor(mbop, op)
        else
            mbop = tensor(mbop, identityoperator(bases[i]))
        end

    end

    return mbop
end

function smoothstep!(x)
    if x < 0
        return 0
    elseif x > 1
        return 1
    else
        return 3 * x^2 - 2 * x^3
    end
end

function rescaled_autodensity!(values)
    median_v = median(values)
    val_hat = abs.(values .- median_v)
    sigma_dx = std(val_hat[begin+1:end] - val_hat[begin:end-1])
    median_hat = median(val_hat)
    Up = val_hat[val_hat.>median_hat]
    Down = val_hat[val_hat.<=median_hat]
    return sigma_dx * (1.0 / std(Up) - 1.0 / std(Down))
end
function binder!(values)
    val_hat = values .- mean(values)
    return 1 - mean(val_hat .^ 4) / (3 * mean(val_hat .^ 2)^2)
end

function binder_no_mean_sub!(values)
    return 1 - mean(values .^ 4) / (3 * mean(values .^ 2)^2)
end

function make_operators(fockmax, Nspin)
    fb = FockBasis(fockmax)
    sb = SpinBasis(Nspin // 2)
    bases = [sb, fb]
    a = mb(destroy(fb), bases, 2)
    Sx = mb(sigmax(sb), bases, 1) / 2
    Sy = mb(sigmay(sb), bases, 1) / 2
    Sz = mb(sigmaz(sb), bases, 1) / 2
    idOp = mb(identityoperator(sb), bases, 1)
    return fb, sb, bases, a, Sx, Sy, Sz, idOp
end

function make_operators(Nspin)
    sb = SpinBasis(Nspin // 2)
    Sx = sigmax(sb) / 2
    Sy = sigmay(sb) / 2
    Sz = sigmaz(sb) / 2
    idOp = identityoperator(sb)
    return sb, Sx, Sy, Sz, idOp
end

function dicke_hetrodyne_atom_only_prob(; Nspin=10, κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, λ0=1.0, t_ramp=500.0, λmod=0.0, ωmod=2π * 1e-6 * 500.0, tmax=500.0, recordtimes=500, noise=nothing, save_noise=false)
    sb, Sx, Sy, Sz, idOp = make_operators(Nspin)

    ψ0 = spindown(sb)
    ψ0 = normalize!(ψ0)
    Q0 = 0 # charge on the photodiode at time 0
    cl0 = ComplexF64[Q0]
    ψ_sc0 = semiclassical.State(ψ0, cl0)
    tspan = range(0.0, tmax, recordtimes)
    stateG = copy(ψ_sc0)
    dstateG = copy(ψ_sc0)
    Nq = length(ψ_sc0.quantum)
    Nc = length(ψ_sc0.classical)
    Ntot = Nq + Nc
    u0 = zeros(ComplexF64, Ntot)
    semiclassical.recast!(u0, ψ_sc0)
    function norm_func(u, t, integrator)
        semiclassical.recast!(stateG, u)
        normalize!(stateG)
        semiclassical.recast!(u, stateG)
    end
    ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
        func_everystep=true,
        func_start=false)
    gc = sqrt(ωz * (Δc^2 + κ^2) / abs(Nspin * Δc))
    grel!(t) = (λ0 + λmod * sin(ωmod * t)) * smoothstep!(t / t_ramp)

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

    Base.@pure pure_inference(fout, T) = Core.Compiler.return_type(fout, T)
    function fout(t, state)
        copy(state)
    end
    function fout_(x, t, integrator)
        semiclassical.recast!(stateG, x)
        copy(stateG)
    end
    out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(ψ_sc0)})
    out = DiffEqCallbacks.SavedValues(eltype(tspan), out_type)
    scb = DiffEqCallbacks.SavingCallback(fout_, out, saveat=tspan,
        save_everystep=false,
        save_start=false,
        tdir=first(tspan) < last(tspan) ? one(eltype(tspan)) : -one(eltype(tspan)))
    full_cb = OrdinaryDiffEq.CallbackSet(nothing, ncb, scb)

    if noise isa Nothing
        noise = StochasticDiffEq.RealWienerProcess!(0.0, zeros(num_noise), save_everystep=save_noise)
    end

    prob = SDEProblem(f!, g!, u0, (tspan[begin], tspan[end]); noise_rate_prototype=noise_prototype, noise=noise)
    prob, full_cb, tspan, out, noise
end

function dicke_hetrodyne_atom_only_meanfield_prob(; Nspin=10, κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, λ0=1.0, t_ramp=500.0, t_hold=0.0, λmod=0.0, ωmod=2π * 1e-6 * 500.0, tmax=500.0, recordtimes=500, noise=nothing, save_noise=false)
    # function norm_func(u, t, integrator)
    #     semiclassical.recast!(stateG, u)
    #     normalize!(stateG)
    #     semiclassical.recast!(u, stateG)
    # end
    # ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
    #     func_everystep=true,
    #     func_start=false)
    tspan = range(0.0, tmax, recordtimes)
    gc = sqrt(ωz * (Δc^2 + κ^2) / abs(Nspin * Δc))
    grel!(t) = (λ0 + λmod * sin(ωmod * t)) * smoothstep!((t - t_hold) / t_ramp)

    αplus = Δc / (-Δc + ωz - im * κ) + Δc / (-Δc - ωz - im * κ)
    αminus = Δc / (-Δc + ωz - im * κ) - Δc / (-Δc - ωz - im * κ)

    spin_len = Nspin / 2.0

    #                       Sx             Sy          Sz              Q
    u0 = ComplexF64[0.0, 0.0, -spin_len, 0.0]

    function f!(du, u, p, t)
        du[1] = -(ωz + real(αminus) * (grel!(t) * gc)^2 / (4 * Δc)) * u[2] + (grel!(t) * gc)^2 * u[3] / (2 * Δc) * (imag(αminus) - κ / Δc * real(conj(αplus) * αminus)) * u[1] - (grel!(t) * gc)^2 * κ / (4 * Δc^2) * (imag(conj(αplus) * αminus) * u[2] + conj(αminus) * αminus * u[1])
        du[2] = (ωz + real(αminus) * (grel!(t) * gc)^2 / (4 * Δc)) * u[1] + (grel!(t) * gc)^2 * u[3] / (2 * Δc) * (2 * real(αplus) * u[1] - (imag(αminus) + (κ / Δc) * real(conj(αplus) * αminus)) * u[2]) - (grel!(t) * gc)^2 * κ / (4 * Δc^2) * (imag(conj(αplus) * αminus) * u[1] + conj(αplus) * αplus * u[2])
        du[3] = (grel!(t) * gc)^2 / (2 * Δc) * (imag(αminus) * (u[2]^2 - u[1]^2) + κ / Δc * real(conj(αplus) * αminus) * (u[2]^2 + u[1]^2) - 2 * real(αplus) * u[2] * u[1]) - (grel!(t) * gc)^2 * κ / (4 * Δc^2) * (conj(αplus) * αplus + conj(αminus) * αminus) * u[3]
        du[4] = grel!(t) * gc * sqrt(κ) / (2 * Δc) * (αplus * u[1] + im * αminus * u[2])
    end

    num_noise = 2
    noise_prototype = zeros(ComplexF64, (4, num_noise))

    function g!(du, u, p, t)
        du[1, 1] = grel!(t) * gc * sqrt(κ) / (2 * Δc) * u[3] * real(αminus)
        du[2, 1] = -grel!(t) * gc * sqrt(κ) / (2 * Δc) * u[3] * imag(αplus)
        du[3, 1] = grel!(t) * gc * sqrt(κ) / (2 * Δc) * (u[2] * imag(αplus) - u[1] * real(αminus))
        du[1, 2] = -grel!(t) * gc * sqrt(κ) / (2 * Δc) * u[3] * imag(αminus)
        du[2, 2] = -grel!(t) * gc * sqrt(κ) / (2 * Δc) * u[3] * real(αplus)
        du[3, 2] = grel!(t) * gc * sqrt(κ) / (2 * Δc) * (u[2] * real(αplus) + u[1] * imag(αminus))

        du[4, 1] = 1.0 / sqrt(2)
        du[4, 2] = 1.0im / sqrt(2)
    end

    function fout(x, t, integrator)
        return copy(x)
    end
    out_type = typeof(u0)
    out = DiffEqCallbacks.SavedValues(eltype(tspan), out_type)
    scb = DiffEqCallbacks.SavingCallback(fout, out, saveat=tspan,
        save_everystep=false,
        save_start=false,
        tdir=first(tspan) < last(tspan) ? one(eltype(tspan)) : -one(eltype(tspan)))
    full_cb = OrdinaryDiffEq.CallbackSet(nothing, nothing, scb)

    if noise isa Nothing
        noise = StochasticDiffEq.RealWienerProcess!(0.0, zeros(num_noise), save_everystep=save_noise)
    end

    prob = SDEProblem(f!, g!, u0, (tspan[begin], tspan[end]); noise_rate_prototype=noise_prototype, noise=noise)
    prob, full_cb, tspan, out, noise
end

function single_run_dicke_hetrodyne(seed, λrel::Number; κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, fockmax=4, Nspin=20, tmax=500.0, dt=0.0001, recordtimes=5000)# ALL IN MHz
    fb, sb, bases, a, Sx, Sy, Sz, idOp = make_operators(fockmax, Nspin)

    # κ = 2π*0.15 # MHz
    # Δc = 2π*20 # MHz
    # ωz = 2π*0.01 # MHz
    λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)
    λ = λrel * λc

    αinit = 0.0
    φinit = 0.0

    if λrel > 1.0
        θ_α = atan(-κ / Δc)
        Szinit = -ωz * Nspin / (8 * λ^2) * sqrt(Δc^2 + κ^2) / cos(θ_α)
        Sxinit = sqrt(Nspin^2 / 4 - Szinit^2)
        φinit = atan(Sxinit / Szinit)
        αinit = 2im * λ / (sqrt(Nspin) * (-1im * Δc - κ)) * Sxinit
    end

    tspan = range(0.0, tmax, recordtimes)

    ψ0 = tensor(spindown(sb), coherentstate(fb, αinit))# + coherentstate(b,-0.7))
    ψ0 = exp(im * φinit * Sy) * ψ0
    ψ0 = normalize!(ψ0)
    @assert abs(expect(mb(projector(basisstate(fb, fockmax + 1)), bases, 2), ψ0)) < 0.01
    print("Running with seed: ", seed)

    C = sqrt(2 * κ) * a
    H0 = Δc * dagger(a) * a + ωz * Sz + 2 * λ * (dagger(a) + a) * Sx / sqrt(Nspin)
    H_nl(ψ) = im * expect(dagger(C), normalize(ψ)) * C - 0.5im * dagger(C) * C - 0.5im * expect(dagger(C), normalize(ψ)) * expect(C, normalize(ψ)) * idOp
    fdet_homodyne(t, ψ) = H0 + H_nl(ψ)
    fst_homodyne(t, ψ) = [C - expect(C, normalize(ψ)) * idOp]

    W = StochasticDiffEq.WienerProcess(0.0, im * 0.0, im * 0.0)

    tout, psi_t = stochastic.schroedinger_dynamic(tspan, ψ0, fdet_homodyne, fst_homodyne; dt=dt, normalize_state=true, noise=W, seed=seed, alg=SOSRI2(), reltol=10^-4, abstol=10^-4, maxiters=10^8)
    return tout, psi_t, W, fb, sb, bases, a, Sx, Sy, Sz
end

function single_run_dicke_hetrodyne(seed, λrel::Function; κ=2π * 0.15, Δc=2π * 20, ωz=2π * 0.01, fockmax=4, Nspin=20, tmax=500.0, dt=0.0001, recordtimes=5000)# ALL IN MHz
    fb, sb, bases, a, Sx, Sy, Sz, idOp = make_operators(fockmax, Nspin)

    # κ = 2π*0.15 # MHz
    # Δc = 2π*20 # MHz
    # ωz = 2π*0.01 # MHz
    λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)

    αinit = 0.0
    φinit = 0.0

    tspan = range(0.0, tmax, recordtimes)

    ψ0 = tensor(spindown(sb), coherentstate(fb, αinit))# + coherentstate(b,-0.7))
    ψ0 = exp(im * φinit * Sy) * ψ0
    ψ0 = normalize!(ψ0)
    @assert abs(expect(mb(projector(basisstate(fb, fockmax + 1)), bases, 2), ψ0)) < 0.01
    print("Running with seed: ", seed)

    C = sqrt(2 * κ) * a
    H0(t) = Δc * dagger(a) * a + ωz * Sz + 2 * λc * (λrel(t) * dagger(a) + conj(λrel(t)) * a) * Sx / sqrt(Nspin)
    H_nl(ψ) = im * expect(dagger(C), normalize(ψ)) * C - 0.5im * dagger(C) * C - 0.5im * expect(dagger(C), normalize(ψ)) * expect(C, normalize(ψ)) * idOp
    fdet_homodyne(t, ψ) = H0(t) + H_nl(ψ)
    fst_homodyne(t, ψ) = [C - expect(C, normalize(ψ)) * idOp]

    W = StochasticDiffEq.WienerProcess(0.0, im * 0.0, im * 0.0)

    tout, psi_t = stochastic.schroedinger_dynamic(tspan, ψ0, fdet_homodyne, fst_homodyne; dt=dt, normalize_state=true, noise=W, seed=seed, alg=SOSRI2(), reltol=10^-4, abstol=10^-4, maxiters=10^8)
    return tout, psi_t, W, fb, sb, bases, a, Sx, Sy, Sz
end

function window(signal, time, dt)
    Ntime = length(time)
    Ttotal = time[Ntime] - time[1]
    Nfilt = Int64(ceil(Ttotal / dt))
    filtered = im * zeros(ComplexF64, Nfilt)
    time_indx = 1
    for i = 1:Nfilt
        window_size = 0
        while time_indx <= Ntime && time[time_indx] < dt * (i + 1)
            # print(time_indx, " ", Ntime,"\n")
            window_size += 1
            # print(length(filtered), " ", Nfilt, "\n")
            filtered[i] += signal[time_indx]
            time_indx += 1
        end
        filtered[i] /= window_size
    end
    return filtered
end

function exp_filter(signal, time_scale, dt)
    N = length(signal)
    filtered = zeros(typeof(signal[1]), N)
    for i = 2:N
        filtered[i] = filtered[i-1] * exp(-dt / time_scale) + signal[i] * (1 - exp(-dt / time_scale))
        # filtered[i] = filtered[i-1]*0.9 + 0.2*homodyne_[i]
    end
    return filtered
end

function two_point_correlator(signal, time, dt, time_steps::Int; prefilter::Union{Nothing,Float64}=nothing, delay_start::Union{Nothing,Integer}=nothing)
    # signal = window(signal, time, dt)
    N = length(signal)
    # if prefilter != nothing
    #     signal = filter(signal, prefilter, dt)
    #     N = N-1
    # end
    rslt = zeros(ComplexF64, time_steps)
    for i = 0:(time_steps-1)
        rslt[i+1] = sum(signal[1:N-i] .* conj(signal[i+1:N])) / (N - i)
    end
    return rslt
end

function two_point_correlator(signal, time_steps::Int)
    N = length(signal)
    rslt = zeros(ComplexF64, time_steps)
    for i = 0:(time_steps-1)
        rslt[i+1] = sum(signal[1:N-i] .* conj(signal[i+1:N])) / (N - i)
    end
    return rslt
end

function make_white_noise(time, W)
    dt = time[2] - time[1]
    W.dt = dt
    u = nothing
    p = nothing
    white_noise = zeros(ComplexF64, length(time))
    calculate_step!(W, dt, u, p)
    for i in 1:(length(time))
        accept_step!(W, dt, u, p)
        white_noise[i] = (W.u[i+1] - W.u[i]) / dt
    end
    return white_noise
end