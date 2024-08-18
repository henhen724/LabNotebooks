const a = 1
const Sx = 2
const Sy = 3
const Sz = 4
const Qhet = 5
const seed = 24, const κ = 2π * 0.15, const Δc = 2π * 20, const ωz = 2π * 0.01, const Nspin = 10000, const tmax = 5.0

λc = 1 / 2 * sqrt((Δc^2 + κ^2) / Δc * ωz)

λ0 = 0.9 * λc
λmod_st = 0.0#0.1*λc
λmod_ω = 0.0

u0 = ComplexF32[0.0, 0.0, 0.0, -Nspin/2.0, 0.0]
if λ0 > λc
    θ_α = atan(-κ / Δc)
    Szinit = -ωz * Nspin / (8 * λ0^2) * sqrt(Δc^2 + κ^2) / cos(θ_α)
    Sxinit = sqrt(Nspin^2 / 4 - Szinit^2)
    αinit = 2im * λ0 / (sqrt(Nspin) * (-1im * Δc - κ)) * Sxinit

    u0 = ComplexF32[αinit, Sxinit, 0.0, Szinit, 0.0]
end
pinit = Float32[Δc, ωz, κ, λ0, λmod_st, λmod_ω]

function dicke!(du, u, p, t)
    Δc, ωz, κ, λ0, λmod_st, λmod_ω = p
    λ = λ0 + λmod_st * sin(λmod_ω * t)
    @inbounds du[a] = (-1im * Δc - κ) * u[a] - 2im * λ / sqrt(Nspin) * u[Sx]
    @inbounds du[Sx] = -ωz * u[Sy]
    @inbounds du[Sy] = ωz * u[Sx] - 2 / sqrt(Nspin) * (conj(λ) * u[a] + λ * conj(u[a])) * u[Sz]
    @inbounds du[Sz] = 2 / sqrt(Nspin) * (conj(λ) * u[a] + λ * conj(u[a])) * u[Sy]
    @inbounds du[Qhet] = sqrt(2 * κ) * conj(u[a])
end

function σ_dicke!(du, u, p, t)
    Δc, ωz, κ, λ0, λmod_st, λmod_ω = p
    @inbounds du[a, 1] = sqrt(κ / 2)
    @inbounds du[Qhet, 1] = 1.0
end

NRate = spzeros(ComplexF32, 5, 1)
NRate[a, 1] = 1
NRate[Qhet, 1] = 1

prob_dicke = SDEProblem{true}(dicke!, σ_dicke!, u0, (0.0, tmax), pinit, noise_rate_prototype=NRate)
ens_prob_dicke = EnsembleProblem(prob_dicke)
sol = solve(ens_prob_dicke, SRA2(), EnsembleThreads(); trajectories=100)
