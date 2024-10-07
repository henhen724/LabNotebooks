using Distributed, Dates #, ClusterManagers

@everywhere const save_dir = @__DIR__# ENV["SCRATCH"] #

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
    setup_worker()
end
# for i in workers()
#     remotecall_wait(setup_worker, i)
# end


@everywhere begin
    using JLD2
    include("HenryLib.jl")
    outdir = joinpath(save_dir, "results/$(Dates.today())/$(basename(@__FILE__)[begin:end-3])")

    Tfilt = 100.0
    Nspin = 100_000
    dt = (2.0)^(-12)
    tmax = 1.0#100_000.0
    κ = 2π * 0.15
    Δc = -2π * 80
    ωz = 2π * 0.01
    t_ramp = 600.0
    t_hold = 100.0
    λmod = 0.0
    ωmod = 0.0
    recordtimes = 50000
    save_noise = false

    λ0s = LinRange(0.5, 1.5, 20)
    seeds = 1:10
    println("Finish preping worker.")
end

@sync @distributed for (λ0Indx, seedIndx) in collect(Iterators.product(eachindex(λ0s), eachindex(seeds)))
    λ0 = λ0s[λ0Indx]
    seed = λ0Indx * seeds[seedIndx]
    println("Begin: λ0 = $λ0, seed = $seed")

    prob, full_cb, tspan, out, CurrW = dicke_hetrodyne_atom_only_meanfield_prob(; Nspin=Nspin, κ=κ, Δc=Δc, ωz=ωz, λ0=λ0, t_ramp=t_ramp, t_hold=t_hold, λmod=λmod, ωmod=ωmod, tmax=tmax, recordtimes=recordtimes, save_noise=save_noise) # ,Sxinit=Nspin/2*sin(pi/12),Szinit=-Nspin/2*cos(pi/12)
    sol1 = solve(prob, RKMilGeneral(; ii_approx=IICommutative());
        adaptive=false,
        dt=dt,
        save_everystep=false,
        save_start=false,
        save_end=false,
        saveat=tspan,
        callback=full_cb,
        seed=seed)
    vec_t = copy(out.saveval)
    tout = copy(out.t)
    println("End: λ0 = $λ0, seed = $seed")
    jldsave(joinpath(outdir, "vec_t_$(seedIndx)_$(λ0Indx).jld2"); tout=tout, vec_t=vec_t)#joinpath(outdir, "vec_t_$(seedIndx)_$(λ0Indx).npz")
    println("Saved: λ0 = $λ0, seed = $seed")
end