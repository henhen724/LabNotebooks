function to_1d_index(i::Int, j::Int, ncols::Int)::Int
    return (i - 1) * ncols + j
end

function to_2d_indices(index::Int, ncols::Int)::Tuple{Int,Int}
    i = div(index - 1, ncols) + 1
    j = mod(index - 1, ncols) + 1
    return (i, j)
end

function safe_index(vec::Vector{T}, idx::Int) where {T}
    return (1 <= idx <= length(vec)) ? vec[idx] : zero(T)
end

function safe_index_2D(vec::Vector{T}, i::Int, j::Int, nrows::Int, ncols::Int)
    if 1 <= i <= nrows && 1 <= j <= ncols
        return vec[1+to_1d_index(i, j)]
    else
        return 0
    end
end

using DifferentialEquations

momentum_cutoff_long = 21
momentum_cutoff_trans = 21
@assert momentum_cutoff_long % 2 == 1 & momentum_cutoff_trans % 2 == 1
vec_dim = 1 + momentum_cutoff_long * momentum_cutoff_trans
u0 = zeros(ComplexF64, vec_dim)

noise_prototype = zeros(Float64, (vec_dim, 2)) #hetrodyne
# noise_prototype = zeros(Float64, (vec_dim, 1)) #homodyne

κ = 0.15 #MHz
ω_c = 80.0 #MHz
ω_r = 0.01 #MHz
E_0 = 1.0 #MHz
P = 1.0 #MHz



function sde_drift(du, u, p, t)
    du[1] = -(κ + im * ω_c) * u[1]
    for i in 1:momentum_cutoff_long
        for j in 1:momentum_cutoff_trans
            mom_indx = to_1d_index(i, j, momentum_cutoff_trans)
            n = i - (momentum_cutoff_long + 1) // 2
            m = j - (momentum_cutoff_trans + 1) // 2
            const trans_sum = 0.0
            du[1+mom_indx] = -im * ω_r * ((n^2 + m^2) * u[1+mom_indx] - conj(u[1]) * u[1] * trans_sum - P * (u[1] + conj(u[1])))
        end
    end
end

function sde_diffusion(du, u, p, t)
    du[1, 1] = sqrt(κ / 4)
    du[1, 2] = im * sqrt(κ / 4) # comment out for homodyne
end


tspan = (0.0, 1.0)

prob = SDEProblem(sde_drift, sde_diffusion, u0, tspan)
sol = solve(prob, SOSRI2())

println(sol)