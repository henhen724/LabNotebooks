using QuantumOptics
using PyPlot
using LaTeXStrings
using Random
using DelimitedFiles
using NPZ, Printf
using LinearAlgebra
using DifferentialEquations
using SpecialFunctions
using StatsBase
using SparseArrays
include("BrendanLib.jl")

# NOTE: Here we are approximating f^±(t) = g(t). This is only true
# for sufficiently slow ramps, but should be a good approximation for our regimes.

# NOTE: We are cycling the Pauli matrices to work in the X basis,
# x,y,z -> z,x,y. Now the projectors onto sx states are just the unit vectors.

# This version is balanced homodyne.

function main()

    println("")

    # Constants - not allowed to change!
    M = 1      # Number of atoms per ensemble
    S = M / 2.0

    # Main Parameters
    Ne = 10                    # Number of spin ensembles
    ωc = 2 * π * 80 # 80               # Mode detuning MHz
    κ = 2 * π * 0.13#13 #0.13              # Cavity loss MHz
    ωz0 = 2 * π * 0.015  #0.015          # (Initial) atomic frequency MHz
    # β = 0.5 * im / sqrt(Ne)           # Complex LO amplitude × √(flux rate MHz)
    β = 0.05 * im / sqrt(Ne) #0.5

    # Ramp functions
    rampTime = 4000 #10000.0 #2000.0           # us
    rampFactor = 5 #5        # Power, g^2/gc^2
    offset = 0.0
    stepTime = 600
    f(t) = smoothstep((t - 100) / stepTime) # This function needs to go to 1.
    # f(t) =  max(0,min(1, (1+1.3*tanh((t-100-0.5*stepTime)/(0.5*stepTime)))/2 ))
    gNorm(t) = sqrt(rampFactor * f(t)) #sqrt(max(0,rampFactor * f(t) )) # Dimensionless
    ωz(t) = ωz0 * max(0, 1 - (1 - offset) * smoothstep((t - 100) / stepTime))
    # ωz(t) = ωz0 * (1-f(t))

    # Time step for noise
    doNoise = κ != 0 # Use this if κ=0 to save some time.
    dt = 0.02#0.1   # us

    # Measurement related
    doSecondOrder = true
    numTvalsObs = 2000
    numTvalsState = 100

    # Set the seed for random number generation
    randSeed = 2688 #abs(rand(Int16)) #853 #23523  # For N=12, 8628, 7211
    Random.seed!(randSeed)
    println("Seed: " * string(randSeed))

    #   Construct a J matrix
    ##

    # J = randn(Ne,Ne)
    # J = (J+J')./2

    # # # # pos, J = confocalJrealHalfPlane(Ne,s=0.0,ϕ=0.03,w=2)
    # while true

    pos, J = confocalJrealHalfPlane(Ne, s=0.0, ϕ=0.02, w=2)
    # pos, J = confocalJrealHalfPlane(Ne,s=0.0,ϕ=0.02,w=0.5)
    # J = ones(Ne,Ne) + 0.1*I
    # J = (J + J')/2 # Ensure symmetric
    # w=5
    # β=10
    # pos, J = confocalJ(Ne,w,β,precision=2)
    localMins(J - diagm(diag(J)))
    # println("Accept J? (y/n)")
    # a = readline()
    # if a=="y"
    #     display(J)
    #     break
    # end

    # end

    # randIdx = 3 #rand(1:100)
    # println(randIdx)
    # J = allJs[:,:,randIdx]
    # localMins(J-diagm(diag(J)))

    # J = npzread("JsN15_w2.npz")[:,:,86]

    #   Analyze J matrix and make pump strength function
    ##

    # Compute eigendecomposition and make sure J is PSD
    Jeig = eigen(J)
    Jevals, Jevecs = Jeig.values, Jeig.vectors #eigenTrunc(K,tol=tol)
    if minimum(Jevals) < 0
        println("Warning: J is not PSD, with eigenvalue " * string(minimum(Jevals)))
        minVal = minimum(Jevals)
        for i = 1:Ne
            J[i, i] = J[i, i] - minVal
        end
        Jeig = eigen(J)
        Jevals, Jevecs = Jeig.values, Jeig.vectors
        Jevals = abs.(Jevals)
    end

    # Compute mode couplings
    α = zeros(Ne, Ne)
    for i = 1:Ne
        for m = 1:Ne
            α[i, m] = sqrt(Jevals[m]) * Jevecs[i, m]
        end
    end

    # Compute critical pump strength (given initial ωz)
    λmax = maximum(eigen(J).values)
    gcJ = sqrt((ωc^2 + κ^2) / ωc * ωz0 / (M * λmax)) # This isnt quite right because ωz is now a function of time, but its ok.
    gc = gcJ #2.7 # Manual override for constant ramp between J matrices.
    println("Critical g: " * string(gc))
    println("Max eigval: " * string(λmax))

    g(t) = gc * gNorm(t)

    # Find tc for semiclassical dynamics
    # tz where ωz becomes static
    # ts is where the Hamiltonian becomes time independent
    tc = -1
    tz = -1
    ts = -1
    for i = 1:10000
        tval = i / 10000 * rampTime

        if tc == -1 && g(tval)^2 / gcJ^2 >= ωz(tval) / ωz0
            tc = tval
        end

        if tz == -1 && abs(ωz(tval) / ωz0) <= offset + 1e-10
            tz = tval
        end

        if f(tval) >= 1 - 1e-10 && tz > 0
            ts = tval
            break
        end
    end
    if tc == -1 || tz == -1 || ts == -1
        println("Could not find static time!")
        return -1
    end
    print("tc: ")
    println(tc)
    print("tz: ")
    println(tz)
    print("ts: ")
    println(ts)

    tspan = (0, rampTime)

    # Compute scaled J matrix. Needs a factor of g^2.
    J = ωc / (ωc^2 + κ^2) * α * α'
    J = (J + J') / 2
    Jnon = copy(J)
    for i = 1:Ne
        Jnon[i, i] = 0
    end

    # Info about J matrix
    GS, E0 = localMins(Jnon, quiet=true)

    # Compute unscaled J
    αOut = α * α'
    αOut = (αOut + αOut') / 2

    # Compute spontaneous emission rate
    Γ = 2π * 6 # MHz
    Mclump = 1e5
    g0 = 2π * 1.5 # MHz
    Γeff = 2 * Γ * ωc * ωz0 / (Mclump * λmax * g0^2) * rampFactor
    τeff = 1 / Γeff
    print("Spon. decay lifetime [ms]: ")
    display(τeff / 1000)

    #   Find ground state ground energies
    ##
    # Compute energy of binarized lowest energy eigenvector
    binEvec = sign.(Jevecs[:, length(Jevals)])
    EminBin = -binEvec' * Jnon * binEvec

    # Do steepest descent on binarized eigenvector
    binEvecSD = copy(binEvec)
    SD!(binEvecSD, Jnon)
    E_SD = -binEvecSD' * Jnon * binEvecSD

    #   Build the Hamiltonian
    ##

    HilbertSize = (M + 1)^Ne

    bases = []
    bs = SpinBasis(M // 2)
    for i = 1:Ne
        push!(bases, bs)
    end

    # Make identity matrix
    id = mb(identityoperator(bs), bases, 1)
    idData = id.data
    idS = identityoperator(bs).data

    # Sz (Sy) operators. Needs factor of ωz
    Hz = 0 * id.data
    for i = 1:Ne
        Hz = Hz + 0.5 * mb(sigmay(bs), bases, i).data
    end

    # Interaction terms
    HXX = 0 * id
    HYY = 0 * id
    HXY = 0 * id
    for i = 1:Ne
        for j = 1:Ne

            # Can diagonal elements be ignored here since theyre the identity? Seems to speed up the calculation
            HXX = HXX + αOut[i, j] * mb(sigmaz(bs), bases, i) * mb(sigmaz(bs), bases, j)
            HYY = HYY + αOut[i, j] * mb(sigmax(bs), bases, i) * mb(sigmax(bs), bases, j)
            HXY = HXY + αOut[i, j] * mb(sigmaz(bs), bases, i) * mb(sigmax(bs), bases, j)
        end
    end
    HXX = HXX.data
    HXY = HXY.data
    HYY = HYY.data

    # Collapse operators. Only includes factor of -√(κ/2)
    CX = SparseMatrixCSC{Complex{Float64},Int64}[]
    CY = SparseMatrixCSC{Complex{Float64},Int64}[]
    CP = SparseMatrixCSC{Complex{Float64},Int64}[]
    for m = 1:Ne
        opX = 0 * id.data
        opY = 0 * id.data
        for i = 1:Ne
            opX = opX + α[i, m] * mb(0.5 * sigmaz(bs), bases, i).data
            opY = opY + α[i, m] * mb(0.5 * sigmax(bs), bases, i).data
        end
        opX .*= -sqrt(κ / 2)
        opY .*= -sqrt(κ / 2)
        opP = opX + im * opY

        push!(CX, opX)
        push!(CY, opY)
        push!(CP, opP)
    end

    # # Check if the collapse oeprators are orthogonal
    # innerProds = zeros(Ne,Ne)
    # for m=1:Ne
    #     for n=1:Ne
    #         innerProds[m,n] = tr(CX[m]*CX[n])
    #     end
    # end
    # display(innerProds)

    #   Initial conditions
    ##
    # Normal state
    # Find the eigenstate
    σy = Matrix(sigmay(bs).data)
    y_evals, y_evecs = eigen(σy)
    normalState = y_evecs[:, 1]
    ψdown = spindown(bs)
    ψdown.data .= normalState
    ψ0 = ψdown
    for i = 2:Ne
        ψ0 = tensor(ψ0, ψdown)  # Normal state
    end

    # # Random Sx state
    # ψ0 = spindown(bs) + sign(randn())*spinup(bs)
    # for i=2:Ne
    #     ψ0 = tensor(ψ0,spindown(bs)+sign(randn())*spinup(bs))
    # end

    # # Ground sx state
    # ψ0 = spindown(bs) + sign(GS[1])*spinup(bs)
    # for i=2:Ne
    #     ψ0 = tensor(ψ0,spindown(bs)+sign(GS[i])*spinup(bs))
    # end

    ψ0 = normalize!(ψ0)
    ψtemplate = copy(ψ0)
    ψ0 = complex.(ψ0.data)

    HilbertSize = length(ψ0)

    println("Hilbert space size: " * string(HilbertSize))

    #   Make derivative function and H update function
    ##

    rampsDone = false
    function QSD_Deterministic(du, u, p, t)

        # Compute coefficients
        gt = g(t)
        gt2 = gt^2
        ωzt = ωz(t)
        αx = 1 / (ωc + ωzt - im * κ) + 1 / (ωc - ωzt - im * κ)
        αy = 1 / (ωc + ωzt - im * κ) - 1 / (ωc - ωzt - im * κ)
        aXX = -0.5 * gt2 * real(αx) - 0.25 * im * κ * gt2 * abs(αx)^2
        aXY = 0.5 * gt2 * imag(αy) + 0.5 * im * κ * gt2 * imag(conj(αx) * αy)
        aYY = -0.25 * im * κ * gt2 * abs(αy)^2

        if rampsDone
            mul!(du, Hs, u)
        else

            # Ising
            mul!(du, HXX, u, -im * aXX, 0)

            # ωz parts
            if ωzt > 0
                mul!(du, Hz, u, -im * ωzt, 1)
                mul!(du, HXY, u, -im * aXY, 1)
                mul!(du, HYY, u, -im * aYY, 1)
            end

        end

    end

    function jac!(J, u, p, t)

        # Compute coefficients
        gt = g(t)
        gt2 = gt^2
        ωzt = ωz(t)
        αx = 1 / (ωc + ωzt - im * κ) + 1 / (ωc - ωzt - im * κ)
        αy = 1 / (ωc + ωzt - im * κ) - 1 / (ωc - ωzt - im * κ)
        aXX = -0.5 * gt2 * real(αx) - 0.25 * im * κ * gt2 * abs(αx)^2
        aXY = 0.5 * gt2 * imag(αy) + 0.5 * im * κ * gt2 * imag(conj(αx) * αy)
        aYY = -0.25 * im * κ * gt2 * abs(αy)^2

        if rampsDone
            J .= Hs
        else

            J .= -im * aXX .* HXX

            # ωz parts
            if ωzt > 0
                J .+= -im .* (ωzt .* Hz .+ aXY .* HXY .+ aYY .* HYY)
            end
        end
    end

    # Compute sparsity pattern
    jac0 = 0 * id.data
    jac!(jac0, randn(HilbertSize), 0, 1)

    # Initialize ODE functions with jacobian built in
    fun = ODEFunction(QSD_Deterministic; jac=jac!, jac_prototype=jac0)

    #  Observables related quantities
    ##
    x_op, y_op, z_op, xx_op, yy_op, zz_op = makeSpinObservables(Ne, bases, bs)

    # Derived quantities
    sqrtdt = sqrt(dt)
    numSteps = Int(ceil(tspan[2] / dt))
    numTvalsObs = min(numSteps, numTvalsObs)
    numTvalsState = min(numSteps, numTvalsState)
    stepsPerSave = Int(floor(numSteps / (numTvalsObs - 1.0)))
    stepsPerStateSave = Int(floor(numSteps / (numTvalsState - 1.0)))

    # Observables
    t_, E_ = zeros(numTvalsObs), zeros(numTvalsObs)
    x_, y_, z_, ξ_ = zeros(numTvalsObs, Ne), zeros(numTvalsObs, Ne), zeros(numTvalsObs, Ne), complex.(zeros(numTvalsObs, Ne))
    xx_, yy_, zz_ = zeros(numTvalsObs, Ne, Ne), zeros(numTvalsObs, Ne, Ne), zeros(numTvalsObs, Ne, Ne)
    homodyne = zeros(Ne, numTvalsObs)
    ψt = complex.(zeros(HilbertSize, numTvalsState))
    ψtvals = zeros(numTvalsState)

    #   Solve the ODE
    ##
    # Choose your solver
    solver = TsitPap8() #CFRLDDRK64()  ## #TSLDDRK74() # TsitPap8() #CFRLDDRK64()
    abstol = 1e-10 #1e-15 # Default is 1e-6
    reltol = 1e-10 # Default is 1e-3
    maxiters = 1e6

    # Make the problem instance
    prob = ODEProblem(fun, ψ0, tspan)

    # Reseed the noise instance
    Random.seed!(Int(round(mod(time() * 300, 1000))))

    # Solve the ODE
    println("Solving trajectory... ")
    start = time()
    integrator = init(prob, solver, dt=dt, save_on=false, abstol=abstol, reltol=reltol, save_everystep=false, maxiters=maxiters)
    saveIdx = 1
    stateIdx = 1
    ψrand = copy(ψ0)
    jumpProbs = zeros(2 * Ne)
    jumpCnts = zeros(numSteps)
    Hs = 0 * id.data
    for step = 1:numSteps

        # Deterministic step
        step!(integrator, dt, true)

        # Jumps and nonlinear term
        if doNoise

            # Compute variables
            gt = g(integrator.t)
            gt2 = gt^2
            ωzt = ωz(integrator.t)
            αx = 1 / (ωc + ωzt - im * κ) + 1 / (ωc - ωzt - im * κ)
            αy = 1 / (ωc + ωzt - im * κ) - 1 / (ωc - ωzt - im * κ)
            dplus = 1 / (ωc + ωzt - im * κ)
            dminus = 1 / (ωc - ωzt - im * κ)

            normVal = sqrt(abs(dot(integrator.u, integrator.u)))
            integrator.u ./= normVal

            # Check for a jump
            exval = dot(integrator.u, HXX, integrator.u) * 0.5 * κ * gt2 * abs(αx)^2
            if ωzt > 0
                exval += dot(integrator.u, HYY, integrator.u) * 0.5 * κ * gt2 * abs(αy)^2
                exval -= dot(integrator.u, HXY, integrator.u) * κ * gt2 * imag(conj(αx) * αy)
            end
            pJump = (1 - exp(-dt * (Ne * abs(β)^2 + real(exval))))
            # pJump = dt*( Ne*abs(β)^2 + real(exval) )
            if rand() < pJump

                # Compute jump probabilities
                @inbounds for m = 1:Ne

                    mul!(ψrand, CX[m], integrator.u, gt * αx, 0)
                    if ωzt > 0
                        mul!(ψrand, CY[m], integrator.u, im * gt * αy, 1)
                    end

                    ψrand .+= im * β .* integrator.u
                    jumpProbs[2*m-1] = real(dot(ψrand, ψrand))

                    ψrand .-= 2 * im * β .* integrator.u
                    jumpProbs[2*m] = real(dot(ψrand, ψrand))

                end

                # Pick a jump to do
                idx = sample(Weights(jumpProbs))
                modeIdx = Int(ceil(idx / 2.0))
                plusOrMinus = mod(idx, 2)

                # Do the jump and renormalize
                mul!(ψrand, CX[modeIdx], integrator.u, gt * αx, 0)
                if ωzt > 0
                    mul!(ψrand, CY[modeIdx], integrator.u, -im * gt * αy, 1)
                end
                if plusOrMinus == 1
                    ψrand .+= im * β .* integrator.u
                    homodyne[modeIdx, min(numTvalsObs, saveIdx)] += 1
                else
                    ψrand .-= im * β .* integrator.u
                    homodyne[modeIdx, min(numTvalsObs, saveIdx)] -= 1
                end
                integrator.u .= ψrand
                normVal = sqrt(real(dot(integrator.u, integrator.u)))
                integrator.u ./= normVal
                # set_u!(integrator, )

                jumpCnts[step] = 1
            end
        end

        # Saving / measuring
        if mod(step - 1, stepsPerSave) == 0 && saveIdx <= numTvalsObs

            # Renormalize
            normVal = sqrt(real(dot(integrator.u, integrator.u)))
            integrator.u ./= normVal

            @inbounds for i = 1:Ne
                x_[saveIdx, i] = real(dot(integrator.u, z_op[i], integrator.u))
                y_[saveIdx, i] = real(dot(integrator.u, x_op[i], integrator.u))
                z_[saveIdx, i] = real(dot(integrator.u, y_op[i], integrator.u))

                if doSecondOrder
                    @inbounds for j = i:Ne
                        xx_[saveIdx, i, j] = real(dot(integrator.u, zz_op[i, j], integrator.u))
                        xx_[saveIdx, j, i] = xx_[saveIdx, i, j]
                    end
                end
            end
            t_[saveIdx] = integrator.t

            # Compute expectation of the Hamiltonian
            gt = g(integrator.t)
            gt2 = gt^2
            ωzt = ωz(integrator.t)
            αx = 1 / (ωc + ωzt - im * κ) + 1 / (ωc - ωzt - im * κ)
            αy = 1 / (ωc + ωzt - im * κ) - 1 / (ωc - ωzt - im * κ)
            aXX = -0.5 * gt2 * real(αx)
            aXY = 0.5 * gt2 * imag(αy)

            E_[saveIdx] = ωzt * real(dot(integrator.u, Hz, integrator.u))
            E_[saveIdx] += aXX * real(dot(integrator.u, HXX, integrator.u))
            E_[saveIdx] += aXY * real(dot(integrator.u, HXY, integrator.u))

            saveIdx += 1
        end

        # Saving the state vector
        if mod(step - 1, stepsPerStateSave) == 0 && stateIdx <= numTvalsState
            ψt[:, stateIdx] .= integrator.u
            ψtvals[stateIdx] = integrator.t
            stateIdx += 1
        end

        if rampsDone == false && integrator.t > tz
            rampsDone = true
            print("Ramps done: ")
            print(round((time() - start) / 60, digits=1))
            println(" (m)")

            # Compute final Hamiltonian
            gt = g(integrator.t)
            gt2 = gt^2
            ωzt = ωz(integrator.t)
            αx = 1 / (ωc + ωzt - im * κ) + 1 / (ωc - ωzt - im * κ)
            αy = 1 / (ωc + ωzt - im * κ) - 1 / (ωc - ωzt - im * κ)
            aXX = -0.5 * gt2 * real(αx) - 0.25 * im * κ * gt2 * abs(αx)^2
            aXY = 0.5 * gt2 * imag(αy) + 0.5 * im * κ * gt2 * imag(conj(αx) * αy)
            aYY = -0.25 * im * κ * gt2 * abs(αy)^2
            # aJZ =  0.5 * gt2 * real(αy) + 0.5  * im * κ * gt2 * real( conj(αx) * αy)

            Hs .= -im * (aXX * HXX .+ aXY * HXY .+ aYY * HYY) # .+ aJZ*HJZ)
            dropzeros!(Hs)
        end

    end
    print(round((time() - start) / 60, digits=1))
    println(" (m)")

    expVals = (t_, x_, y_, z_, xx_, homodyne, E_)

    return integrator, ψt, ψtvals, expVals, Ne, S, g, ωz, tc, J, E_SD, E0, GS, ωc, κ, α, bs, bases, jumpCnts

end

# Run main
integrator, ψt, ψtvals, expVals, Ne, S, g, ωz, tc, J, E_SD, E0, GS, ωc, κ, α, bs, bases, jumpCnts = main()
t_, x_, y_, z_, xx_, homodyne, E_ = expVals
M = Int(2 * S)
HilbertSize = (M + 1)^Ne

numTvalsObs = length(t_)

# Compute things
##

# Normalize the state vector
for i = 1:length(ψtvals)
    norm = sqrt(sum(abs.(ψt[:, i]) .^ 2))
    ψt[:, i] ./= norm
end

# Normalize J and energies to make it dimensionless
J *= ωc

# # Compute overlap distribution of final state
# println("Computing overlaps...")
# olaps = zeros(Ne+1,length(ψtvals))
# for t=1:length(ψtvals)
#     olaps[:,t] .= olapGenMC_V2(ψt[:,t],Ne,M,steps=1000)
# end
# histvals = range(-1,1,length=Ne+1)

# # Compute magnetization
# mags = zeros(Ne+1,length(ψtvals))
# for t=1:length(ψtvals)
#     mags[:,t] .= magGen(ψt[:,t],Ne,M)
# end

# Compute finite integration time homodyne
intSteps = 500
homodyneFinite = zeros(numTvalsObs, Ne)
for i = 1:Ne
    for t = 1:numTvalsObs
        homodyneFinite[t, i] = sum(homodyne[i, max(1, t - intSteps):t])
    end
end

# Compute integrated homodyne
homodyneInt = cumsum(homodyne, dims=2)
homodyneLocal = zeros(numTvalsObs, Ne)
homodyneLocalFinite = zeros(numTvalsObs, Ne)
for i = 1:Ne
    for m = 1:Ne
        homodyneLocal[:, i] .+= α[i, m] .* homodyneInt[m, :]
        homodyneLocalFinite[:, i] .+= α[i, m] .* homodyneFinite[:, m]
    end
end

# Compute Ising energy from phase of light
E_Ising = zeros(numTvalsObs)
E_IsingBin = zeros(numTvalsObs)
E_Measured = zeros(numTvalsObs)
for i = 1:Ne
    for j = 1:Ne
        if i == j
            continue
        end
        E_Ising .+= -J[i, j] * xx_[:, i, j] / S^2 / ωc
        E_IsingBin .+= -J[i, j] * sign.(x_[:, i]) .* sign.(x_[:, j]) / ωc
        E_Measured .+= -J[i, j] * sign.(homodyneLocal[:, i]) .* sign.(homodyneLocal[:, j]) / ωc
    end
end

# Compute entanglement entropy
entent = zeros(numTvalsObs, Ne)
for i = 1:Ne
    entent[:, i] .= entanglementEnt.(x_[:, i], y_[:, i], z_[:, i])
end

# Save stuff
##

# saveIdx = ARGS[1]
# saveIdxInt = parse(Int,saveIdx)

# # ψf = integrator.u
# # npzwrite("psif_"*string(saveIdxInt)*".npz",ψf)

# npzwrite("x_"*saveIdx*".npz",x_)
# npzwrite("t_"*saveIdx*".npz",t_)
# npzwrite("E_"*saveIdx*".npz", abs.((E_Ising.-E0)/E0) )

# Plots
##

# colors_ = colors5()
colors_ = colors15()

# X
fig = figure()
xlabel("Time (us)", fontsize=12)
for i = 1:Ne
    plot(t_, x_[:, i] ./ S, "-", label=string(i), linewidth=1.15, color=colors_[i], alpha=1, zorder=100 - i)
end
plot(t_, (g.(t_) / g(t_[end])) .^ 2, "--", color="black", alpha=1, label="Pump", zorder=100)
plot(t_, ωz.(t_) / ωz(0), "--", color="red", alpha=1, label=L"\omega_z", zorder=100)
axvline(tc, color="tab:blue", linestyle="--", alpha=1, label=L"t_c")
ylim(-1.05, 1.05)
# xlim(0,4000)
grid()
ylabel(L"\langle S_i^x \rangle /S", fontsize=12)
legend(loc=(0.6, 0.05), ncol=2)
PyPlot.display_figs()

# Energy record
fig = figure()
# plot(t_[1:lastSave],abs.((E_Ising.-E0)/E0)[1:lastSave])
plot(t_, E_Ising)
xlabel("Time (us)", fontsize=12)
ylabel(L"\langle H_{\mathrm{Ising}} \rangle", fontsize=12)
axvline(tc, color="black", linestyle="--", alpha=0.3)
# axhline(abs(EminBin/E0-1),color="tab:red",linestyle=":",alpha=0.5,label="Eλ")
axhline(E_SD, color="tab:green", linestyle="-", alpha=0.85, label="Eλ SD")
axhline(E0, color="tab:red", linestyle="-", alpha=0.85, label="E0")
legend()
# ylim(-0.02,1.02)
grid()
PyPlot.display_figs()

# # Full energy record
# fig = figure()
# # plot(t_[1:lastSave],abs.((E_Ising.-E0)/E0)[1:lastSave])
# plot(t_,E_)
# xlabel("Time (us)",fontsize=12)
# ylabel(L"\langle H \rangle",fontsize=12)
# axvline(tc,color="black",linestyle="--",alpha=0.3)
# # ylim(-0.02,1.02)
# grid()
# PyPlot.display_figs()
#
# Entanglement entropy
fig = figure(dpi=300)
for i = 1:Ne
    plot(t_, entent[:, i] ./ log(2), label=string(i))
end
# xlim(0,4000)
xlabel("Time (us)", fontsize=12)
ylabel("Entanglement entropy / log(2)", fontsize=12)
axvline(tc, color="black", linestyle="--", alpha=0.3)
legend()
grid()
PyPlot.display_figs()

# # Binarized Energy record
# fig = figure()
# plot(t_,abs.((E_IsingBin.-E0)/E0))
# plot(t_,abs.((E_Measured.-E0)/E0))
# xlabel("Time (us)",fontsize=12)
# ylabel("Ising energy record",fontsize=12)
# axvline(tc,color="black",linestyle="--",alpha=0.3)
# # axhline(abs(EminBin/E0-1),color="tab:red",linestyle=":",alpha=0.5,label="Eλ")
# axhline(abs(E_SD/E0-1),color="tab:green",linestyle="--",alpha=0.5,label="Eλ SD")
# legend()
# PyPlot.display_figs()
#
# Sx overlaps
fig = figure()
cutoff = 1e-8
for i = 1:HilbertSize
    if (abs.(ψt[i, end]) .^ 2) > cutoff #pops[i]>cutoff
        semilogy(ψtvals, (abs.(ψt[i, :]) .^ 2)')
        # println(i)
    end
end
ylim(max(1e-5, cutoff), 1.25)
grid()
xlabel("Time [us]", fontsize=12)
ylabel("Sx Overlap", fontsize=12)
PyPlot.display_figs()
#
# # fig = figure()
# # # plot(jumpCnts,"o",markersize=2)
# # hist(jumpCnts)
# # xlabel("Jumps per time step")
# # ylabel("Counts")
# # PyPlot.display_figs()
#
# # # Sx overlaps v2
# # fig = figure()
# # cutoff = 1e-3
# # for i=1:HilbertSize
# #     if (abs.(ψt[i,lastState]).^2)>cutoff #pops[i]>cutoff
# #         plot(ψtvals,(abs.(ψt[i,:]).^2)')
# #         # println(i)
# #     end
# # end
# # ylim(0.0,0.5)
# # grid()
# # xlabel("Time [us]",fontsize=12)
# # ylabel("Sx Overlap",fontsize=12)
# # PyPlot.display_figs()
#
# Measurement records (including a minus sign to match atoms)
fig = figure(dpi=300)
maxval = maximum(abs.(homodyneLocal)) * 1.05
for i = 1:Ne
    # plot(t_[1:lastSave],-ErealInt[1:lastSave,i]./maxval,"-",markersize=1,color=colors[i])
    plot(t_ / 1e3, homodyneLocal[:, i] ./ maxval, "-", color=colors_[i], linewidth=2)
    # plot(tSubset,EimagCum[:,i],"-",markersize=1,color=colors_[1+mod(i-1,5)],alpha=0.25)
    # plot(tSubset,dIrecFilt[:,m],"-")
end
# plot(t_/1e3,(g.(t_)/g(t_[end])).^2,"--",color="black",alpha=1,label="Pump",zorder=100,linewidth=2)
xlabel("Time (ms)", fontsize=12)
# ylabel("Balanced homodyne record (integrated) [clicks]",fontsize=12)
ylabel("Cumulative measurement record", fontsize=12)
# axvline(tc/1e3,color="black",linestyle="--",alpha=0.3)
# grid()
xticks(fontsize=14)
yticks(fontsize=14)
# xlim(0,4)
# fill_between([0,t_[end]/1e3],[0,0],[1.1,1.1],alpha=0.3,color="#2F6BBD")
# fill_between([0,t_[end]/1e3],[0,0],[-1.1,-1.1],alpha=0.3,color="#FF8000")
ylim(-1.1, 1.1)
# xlim(0,t_[end])
# savefig("Fig1Record.svg")
# savefig("Fig1Record.png",transparent=true)
PyPlot.display_figs()
#
# println("Done.")







# # Overlap distribution (single)
# # idx = 1
# fig=figure(dpi=300)
# tidx = 100
# bar(histvals,olaps[:,tidx]/sum(olaps[:,tidx]),width=2/(Ne),edgecolor="black",zorder=1,color="tab:red",alpha=0.85)
# grid(zorder=2,alpha=0.4)
# xlabel("Overlap",fontsize=12)
# ylabel("Probability distribution",fontsize=12)
# PyPlot.display_figs()

# # All overlaps
# fig,ax = subplots(2,5,figsize=(15,8))
# for i=1:10
#     ax[Int(ceil(i/5)),mod(i-1,5)+1].bar(histvals,olaps[:,i],width=2/(Ne),edgecolor="black",zorder=1)
#     ax[i].grid(zorder=2,alpha=0.4)
# end
# suptitle("Overlaps",fontsize=12)
# PyPlot.display_figs()
#
# # All mags
# fig,ax = subplots(2,5,figsize=(15,8))
# for i=1:10
#     ax[Int(ceil(i/5)),mod(i-1,5)+1].bar(histvals,mags[:,i],width=2/(Ne),edgecolor="black",zorder=1)
#     ax[i].grid(zorder=2,alpha=0.4)
# end
# suptitle("Magnetization",fontsize=12)
# PyPlot.display_figs()

# # Bloch sphere
# using3D()
# fig = figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection="3d")
# nBloch = 100
# u = range(0,2*π,length=nBloch);
# v = range(0,π,length=nBloch);
# xs = cos.(u) * sin.(v)';
# ys = sin.(u) * sin.(v)';
# zs = ones(nBloch) * cos.(v)';
# surf(xs, ys, zs, rstride=1, cstride=2, alpha=0.07, cmap=get_cmap("Greys_r"))#color="black")
# ax.plot(zeros(nBloch), sin.(u), cos.(u), color="black", alpha=0.6)
# ax.plot(sin.(u), zeros(nBloch), cos.(u), color="black", alpha=0.6)
# ax.plot(sin.(u), cos.(u), zeros(nBloch), color="black", alpha=0.6)
# ax._axis3don = false
# ax.plot([-1,1], [0,0], [0,0], color="black", alpha=1)
# ax.plot([0,0], [-1,1], [0,0], color="black", alpha=1)
# ax.plot([0,0], [0,0], [-1,1], color="black", alpha=1)
# for i=1:Ne
#     ax.plot(x_[:,i]./S, y_[:,i]./S, z_[:,i]./S)
# end
# ax.view_init(10, 110)
# xlim(-1,1)
# ylim(-1,1)
# zlim(-1,1)
# PyPlot.display_figs()
#
# # Bloch spheres individual
# cmap = get_cmap("tab10")
# using3D()
# fig = figure(figsize=(15,15))
# nBloch = 100
# u = range(0,2*π,length=nBloch);
# v = range(0,π,length=nBloch);
# xs = cos.(u) * sin.(v)';
# ys = sin.(u) * sin.(v)';
# zs = ones(nBloch) * cos.(v)';
# for i=1:4
#     ax = fig.add_subplot(2,2,i,projection="3d")
#     ax.plot_surface(xs, ys, zs, rstride=1, cstride=2, alpha=0.07, cmap=get_cmap("Greys_r"))#color="black")
#     ax.plot(zeros(nBloch), sin.(u), cos.(u), color="black", alpha=0.6)
#     ax.plot(sin.(u), zeros(nBloch), cos.(u), color="black", alpha=0.6)
#     ax.plot(sin.(u), cos.(u), zeros(nBloch), color="black", alpha=0.6)
#     ax._axis3don = false
#     ax.plot([-1,1], [0,0], [0,0], color="black", alpha=0.6)
#     ax.plot([0,0], [-1,1], [0,0], color="black", alpha=0.6)
#     ax.plot([0,0], [0,0], [-1,1], color="black", alpha=0.6)
#     #
#     ax.plot(x_[:,i]./S, y_[:,i]./S, z_[:,i]./S,color=cmap(i-1))
#     ax.view_init(10, 110)
# end
# # xlim(-1,1)
# # ylim(-1,1)
# # zlim(-1,1)
# PyPlot.display_figs()
#
# # 2D Plot
# fig = figure()
# for i=1:Ne
#     plot(x_[:,i]./S,z_[:,i]./S)
# end
# PyPlot.display_figs()

###########################
# META PLOTS
###########################

# tqTraj = copy(t_)
#
# # EqTraj = []
# push!(EqTraj,copy(E_))
