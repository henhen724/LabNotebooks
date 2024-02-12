using DifferentialEquations
using Plots
using LaTeXStrings
using JLD2
using Printf


function save_correlators(Δ, ω, κ, g)
    # 1: (a+a^\dag)/2 
    # 2: i(a-a^\dag)/2 
    # 3: S_x 
    # 4: S_y 
    # 5: S_z
    #Δ = 1.0 # MHz
    #ω = 1.0 # MHz
    #κ = 0.01

    g_c = 1/2*sqrt(ω*(Δ^2 + κ^2)/Δ)

    # if g > g_c
    #     inital = [g*Δ/(Δ^2 + κ^2), 0, 1.0, 0, (Δ^2 + κ^2)/Δ*ω/g^2]
    # else
    #     inital = [0., 0., 1.0, 0., 0.]
    # end
    angle = 2*pi*rand(Float64, 2)
    angle[1] = angle[1] % pi
    x = rand(Float64)
    inital = [x,0., sin(angle[1]), cos(angle[1])*cos(angle[2]), cos(angle[1])*sin(angle[2])]

    t_last_bang = 0.0
    cool_down_t = 0.1
    max_x = 0.9

    function dicke(du, u, p, t)
        if u[1] < max_x && t - t_last_bang > cool_down_t
            t_last_bang = t
            g = -g
        end

        du[1] = Δ*u[2]-κ*u[1]
        du[2] = -Δ*u[1]-κ*u[2]+g*u[3]
        du[3] = -ω*u[4]
        du[4] = ω*u[3]+g*u[1]*u[5]
        du[5] = -g*u[1]*u[4]
    end

    function σ_dicke(du, u, p, t)
        du[1] = -sqrt(κ/2)
        du[2] = -sqrt(κ/2)
        du[3] = 0.
        du[4] = 0.
        du[5] = 0.
    end

    prob_sde_lorenz = SDEProblem(dicke, σ_dicke, inital, (0.0, 4000.0))
    sol = solve(prob_sde_lorenz, SKenCarp())

    twopointxx = [sum([sol[j][1]sol[j+i][1] for j=1:length(sol)-i])/(length(sol)-i) for i=1:100]
    twopointxy = [sum([sol[j][1]sol[j+i][2] for j=1:length(sol)-i])/(length(sol)-i) for i=1:100]
    twopointyx = [sum([sol[j][2]sol[j+i][1] for j=1:length(sol)-i])/(length(sol)-i) for i=1:100]
    twopointyy = [sum([sol[j][2]sol[j+i][2] for j=1:length(sol)-i])/(length(sol)-i) for i=1:100]
    
    gstr = @sprintf("%.3f", g)

    JLD2.save_object("BangBangRuns/d=$Δ w=$ω k=$κ g=$gstr xx.jld2", twopointxx)
    JLD2.save_object("BangBangRuns/d=$Δ w=$ω k=$κ g=$gstr xy.jld2", twopointxy)
    JLD2.save_object("BangBangRuns/d=$Δ w=$ω k=$κ g=$gstr yx.jld2", twopointyx)
    JLD2.save_object("BangBangRuns/d=$Δ w=$ω k=$κ g=$gstr yy.jld2", twopointyy)
end

for g in LinRange(0.3,0.5, 21)
    save_correlators(1.0, 1.0, 0.01, g)
end