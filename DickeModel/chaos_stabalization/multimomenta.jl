function to_1d_index(i::Int, j::Int, ncols::Int)::Int
    return (i - 1) * ncols + j
end

function to_2d_indices(index::Int, ncols::Int)::Tuple{Int,Int}
    i = div(index - 1, ncols) + 1
    j = mod(index - 1, ncols) + 1
    return (i, j)
end

using DifferentialEquations

momentum_cutoff_long = 20
momentum_cutoff_trans = 20
κ = 0.15 #MHz
ω_c = 80.0 #MHz
ω_r = 0.01 #MHz

function sde_drift(du, u, p, t)
    du[1] = (-κ + im * ω_c) * u[1]
    for n in 1:momentum_cutoff_long
        for m in 1:momentum_cutoff_trans
            mom_indx = to_1d_index(n, m, momentum_cutoff_trans)
            du[1] += 0.0
            du[1+mom_indx] = im * ω_r * (n^2 + m^2) * u[1+mom_indx]
        end
    end
end

function sde_diffusion(du, u, p, t)
    du[1, 1] = sqrt(κ / 2)
end

vec_dim = 1 + momentum_cutoff_long * momentum_cutoff_trans
u0 = zeros(ComplexF64, vec_dim)
tspan = (0.0, 1.0)

prob = SDEProblem(sde_drift, sde_diffusion, u0, tspan)
sol = solve(prob, SOSRI2())

println(sol)