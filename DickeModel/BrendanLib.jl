using QuantumOptics

##########################################
# Quantum stuff
##########################################

# Computes the entanglement entropy for spin-1/2 systems given
# x = 0.5*⟨σ^x⟩ and so on.
function entanglementEnt(x, y, z)

    ρ = complex.(zeros(2, 2)) + 0.5 * I
    ρ .+= x * [0 1; 1 0]
    ρ .+= y * [0 -im; im 0]
    ρ .+= z * [1 0; 0 -1]

    evals, evecs = eigen(ρ)
    evals = real.(evals)

    ent = 0
    if evals[1] > 0
        ent -= evals[1] * log(evals[1])
    end
    if evals[2] > 0
        ent -= evals[2] * log(evals[2])
    end

    return ent
end

##########################################
# Tensor operations
##########################################

# Make many-body operators function
# bases should be a list of basis objects
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

function makeSpinObservables(Ne, bases, bs)

    # Make operators
    optype = typeof((mb(sigmax(bs), bases, 1)).data)
    x_op, y_op, z_op = Array{optype}(undef, Ne), Array{optype}(undef, Ne), Array{optype}(undef, Ne), Array{optype}(undef, Ne)
    xx_op, yy_op, zz_op = Array{optype}(undef, Ne, Ne), Array{optype}(undef, Ne, Ne), Array{optype}(undef, Ne, Ne)
    for i = 1:Ne
        x_op[i] = (0.5 * mb(sigmax(bs), bases, i)).data
        y_op[i] = (0.5 * mb(sigmay(bs), bases, i)).data
        z_op[i] = (0.5 * mb(sigmaz(bs), bases, i)).data

        for j = i:Ne
            xx_op[i, j] = (0.25 * mb(sigmax(bs), bases, i) * mb(sigmax(bs), bases, j)).data
            yy_op[i, j] = (0.25 * mb(sigmay(bs), bases, i) * mb(sigmay(bs), bases, j)).data
            zz_op[i, j] = (0.25 * mb(sigmaz(bs), bases, i) * mb(sigmaz(bs), bases, j)).data
            xx_op[j, i] = xx_op[i, j]
            yy_op[j, i] = yy_op[i, j]
            zz_op[j, i] = zz_op[i, j]
        end
    end

    return x_op, y_op, z_op, xx_op, yy_op, zz_op
end

##########################################
# J functions
##########################################

function Dloc(x1, x2, y1, y2, ϕ, sx, sy)

    wx2 = 1 + 2 * sx^2
    wy2 = 1 + 2 * sy^2

    ϕx = log((1 + 2 * sx^2) / (1 - 2 * sx^2))
    ϕy = log((1 + 2 * sy^2) / (1 - 2 * sy^2))

    prefactor = 2 / (π * wx2 * wy2 * sqrt((1 - exp(-2 * (ϕ + ϕx))) * (1 - exp(-2 * (ϕ + ϕy)))))

    arg = -(1 - (exp(ϕx) - exp(-ϕ)) / (wx2 * sinh(ϕ + ϕx))) * (x1^2 + x2^2) / wx2
    arg += -(1 - (exp(ϕy) - exp(-ϕ)) / (wy2 * sinh(ϕ + ϕy))) * (y1^2 + y2^2) / wy2

    argLoc = -exp(ϕx) * (x1 - x2)^2 / (wx2^2 * sinh(ϕ + ϕx)) - exp(ϕy) * (y1 - y2)^2 / (wy2^2 * sinh(ϕ + ϕy))

    argMir = -exp(ϕx) * (x1 + x2)^2 / (wx2^2 * sinh(ϕ + ϕx)) - exp(ϕy) * (y1 + y2)^2 / (wy2^2 * sinh(ϕ + ϕy))

    return prefactor * (exp(arg + argLoc) + exp(arg + argMir))
end

function Dnon(x1, x2, y1, y2, ϕ, sx, sy)

    wx2 = 1 + 2 * sx^2
    wy2 = 1 + 2 * sy^2

    ϕx = log((1 + 2 * sx^2) / (1 - 2 * sx^2))
    ϕy = log((1 + 2 * sy^2) / (1 - 2 * sy^2))

    prefactor = 4 / (π * wx2 * wy2 * sqrt((1 + exp(-2 * (ϕ + ϕx))) * (1 + exp(-2 * (ϕ + ϕy)))))

    arg = -(1 - exp(-ϕ) / (wx2 * cosh(ϕ + ϕx))) * (x1^2 + x2^2) / wx2
    arg += -(1 - exp(-ϕ) / (wy2 * cosh(ϕ + ϕy))) * (y1^2 + y2^2) / wy2

    cosArg = 2 * exp(ϕx) * x1 * x2 / (wx2^2 * cosh(ϕ + ϕx)) + 2 * exp(ϕy) * y1 * y2 / (wy2^2 * cosh(ϕ + ϕy))

    return prefactor * (exp(arg + im * cosArg) + exp(arg - im * cosArg)) / 2
end

# s is the Gaussian waist size in units of w0. Should be like 3 um * 0.395 / 35 um = 0.03?
function confocalJreal(N; ϕ=0, s=0.1, w=2, pos=-1, precision=3)

    if pos == -1
        pos = randn(N, 2) * w
    end
    x = pos[:, 1]
    y = pos[:, 2]

    J = zeros(N, N)
    for i = 1:N
        for j = i:N
            J[i, j] = Dloc(x[i], x[j], y[i], y[j], ϕ, s, s) + Dnon(x[i], x[j], y[i], y[j], ϕ, s, s)
            J[j, i] = J[i, j]
        end
    end

    return pos, J
end

function confocalJrealHalfPlane(N; ϕ=0, s=0.1, w=2, pos=-1, precision=3)

    if pos == -1
        pos = randn(N, 2) * w
    end
    x = abs.(pos[:, 1])
    y = pos[:, 2]

    J = zeros(N, N)
    for i = 1:N
        for j = i:N
            J[i, j] = Dloc(x[i], x[j], y[i], y[j], ϕ, s, s) + Dnon(x[i], x[j], y[i], y[j], ϕ, s, s)
            J[j, i] = J[i, j]
        end
    end

    return pos, J
end

function confocalPosToJ(x, y; ϕ=0.02, s=0.0, neg=false)

    N = size(x)[1]

    J = zeros(N, N)
    for i = 1:N
        for j = i:N
            if neg
                J[i, j] = Dloc(x[i], x[j], y[i], y[j], ϕ, s, s) - Dnon(x[i], x[j], y[i], y[j], ϕ, s, s)
            else
                J[i, j] = Dloc(x[i], x[j], y[i], y[j], ϕ, s, s) + Dnon(x[i], x[j], y[i], y[j], ϕ, s, s)
            end
            J[j, i] = J[i, j]
        end
    end

    return J
end

function localMins(J; doPlot=false, quiet=false)

    # Compute eigenvectors
    Jeig = eigen(J)
    Jevals, Jevecs = Jeig.values, Jeig.vectors #eigenTrunc(K,tol=tol)
    binEvec = Int.(sign.(Jevecs[:, length(Jevals)]))
    Ne = size(J)[1]

    # Compute energy of binarized lowest energy eigenvector
    if !quiet
        println(" ")
        # print("Evec:");println(Jevecs[:,length(Jevals)])
        print("Max eval: ")
        display(maximum(Jeig.values))
    end

    # Check for zeros
    for i in eachindex(binEvec, 1)
        if abs(binEvec[i]) < 0.5
            println("Bin evec has zeros.")
            return -1
        end
    end

    EminBin = -binEvec' * J * binEvec

    # Do steepest descent on binarized eigenvector
    binEvecSD = copy(binEvec)
    # if !quiet
    #     print("BinEvecSD:");println(binEvecSD)
    # end
    SD!(binEvecSD, J, minStep=1e-5)
    E_SD = -binEvecSD' * J * binEvecSD

    # Find the local minima
    state = zeros(Ne)
    mins = []
    Es = []
    GS = []
    Emin = 10000
    for j = 1:2^(Ne-1)

        spins = Int.(spinify(state))
        E = -spins' * J * spins

        if E < Emin
            Emin = E
            GS = copy(Int.(spins))
        end

        # Check if local min
        isLocalMin = true
        for k = 1:Ne
            modSpins = copy(spins)
            modSpins[k] *= -1
            Emod = -modSpins' * J * modSpins
            if Emod < E
                isLocalMin = false
                break
            end
        end
        if isLocalMin
            if spins[1] < 0
                push!(mins, -1 .* copy(spins))
            else
                push!(mins, copy(spins))
            end
            push!(Es, E)
        end

        binaryInc(state)
    end

    numMins = length(Es)
    isEigEasy = abs(EminBin - Emin) < 1e-12
    isSDEasy = abs(E_SD - Emin) < 1e-12

    if length(Es) > 1
        sorted = sort(Es)
        # if !quiet
        #     print("Gap:");println(sorted[2]-sorted[1])
        # end
    end

    E0 = -1
    if length(Es) > 1
        E0 = sorted[1]
    else
        E0 = Es[1]
    end

    # Sort by energy
    order = sortperm(Es)
    Es = Es[order]
    mins = mins[order]

    # See what the typical spin flip energy around the global min is
    Eflip = 0
    for i = 1:Ne
        Eflip += 2 * GS[i] * dot(J[i, :], GS)
    end
    Eflip /= Ne
    if !quiet
        print("Avg spin flip: ")
        println(Eflip)
        print("E_flip / lambda: ")
        println(Eflip / Jevals[end])
    end

    if !quiet
        # print("J:");println(J)
        # print("Mins:");println(mins)
        print("E0: ")
        println(minimum(Es))
        print("Es-E0:")
        println((Es .- Emin))
        # print("Eig sol:");println(binEvec)
        # print("Eig sol SD:");println(binEvecSD)
        # print("GS:");println(GS)
        print("Eig hard:")
        println(!isEigEasy)
        print("SD hard:")
        println(!isSDEasy)
    end

    # Compute overlaps
    overlaps = zeros(numMins, numMins)
    for i = 1:numMins
        for j = i:numMins
            overlaps[i, j] = abs(dot(mins[i], mins[j]) / Ne)
            overlaps[j, i] = overlaps[i, j]
        end
    end
    display(round.(overlaps, digits=2))

    if doPlot
        fig = figure()
        plot(Es ./ abs(E0), "o")
        xlabel("Local minima")
        ylabel("Normalized Ising energy")
        PyPlot.display_figs()
    end

    return GS, E0
end

# This version does not normalize by anything
# and assumes spin-1/2 variables
function JgapV2(J)

    Ne = size(J)[1]

    # Find the gap
    state = zeros(Ne)
    E0 = 10000
    E1 = 10000
    s0 = zeros(Ne)
    s1 = zeros(Ne)
    for j = 1:2^(Ne-1)

        spins = Int.(spinify(state))
        E = -spins' * J * spins

        if E < E0
            E1 = E0
            s1 .= s0
            E0 = E
            s0 .= spins
        elseif E < E1
            E1 = E
            s1 .= spins
        end

        binaryInc(state)
    end

    return (E1 - E0), s0, s1
end

##########################################
# Metropolis / spin functions
##########################################

function binaryInc(state)
    for i in eachindex(state, 1)
        if state[i] == 1
            state[i] = 0
        else
            state[i] = 1
            break
        end
    end
    return state
end

function spinify(state)
    return state .* 2 .- 1
end

# A function to get Ising states
function idx2state(idx, M, Ne)
    return Int.(ones(Ne) .- 2 * digits(idx - 1, base=M + 1, pad=Ne))
end

# Steepest descent. Assumes binary +/- 1 valuees for state.
# Assumes E = -∑ J_ij s_i s_j
function SD!(state, J; minStep=1e-10)

    Jdiag = diag(J)
    while true

        dE = state .* (J * state) .- Jdiag
        i = argmin(dE)

        if dE[i] < -abs(minStep)
            state[i] *= -1
        else
            return state
        end
    end
end

##########################################
# Colors
##########################################

function colors15()
    colors = ["#568f3b",
        "#d644b3",
        "#6e5ddc",
        "#e2a540",
        "#6cce49",
        "#9c7a2d",
        "#55cb98",
        "#d6496d",
        "#b545df",
        "#cb669f",
        "#b073cc",
        "#dd412e",
        "#ce6846",
        "#5d83d6",
        "#b5c048"]
    return colors
end

function colors5()
    colors = ["#DC3E2F", "#2F6BBD", "#25944E", "#FF8000", "#B848B5"]
    return colors
end

function colors5d()
    colors = ["#9C0001", "#003D87", "#00601F", "#7C4E00", "#800080"]
    return colors
end

##########################################
# Misc
##########################################

# This is called "smoothstep" on wikipedia
function smoothstep(x)
    if x < 0
        return 0
    elseif x > 1
        return 1
    else
        return 3 * x^2 - 2 * x^3
    end
end

function getFileIndex(basename; maxIdx=1000, ftype="npz")

    for i = 1:maxIdx
        fname = basename * "_" * string(i) * "." * ftype

        if isfile(fname)
            continue
        else
            return i
        end
    end
    return -1
end
