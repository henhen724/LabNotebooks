using QuantumOptics

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

# Define parameters
ω = 0.0#MHz Cavity frequency
κ = 2π * 30.0#MHz Decay rate
g = 2π * 45.0#MHz Coupling strength
ε = 44.3#MHz Drive strength
γperp = 2π * 2.5#MHz Spotaneous emission rate

# Define basis and operators
N = 10  # Truncation of Fock space
fb = FockBasis(N)
sb = SpinBasis(1 // 2)
bases = [fb, sb]

a = mb(destroy(fb), bases, 1)
σm = mb(sigmam(sb), bases, 2)
idOp = mb(identityoperator(sb), bases, 2)

# Define Hamiltonian
Heff = ω * dagger(a) * a + g * (dagger(a) * σm + a * dagger(σm)) + ε * (a - dagger(a)) * σm

# Define collapse operators
c_ops = [sqrt(2 * γperp) * σm, sqrt(2 * κ) * a]

# Initial state (coherent state)
α = 0.5#im * ε / (κ / 2)
ψ0 = tensor(coherentstate(fb, α), spindown(sb))

# Time evolution
tlist = 0:0.001:1
tout, psi_t = timeevolution.mcwf_h(tlist, ψ0, Heff, c_ops)


# Plot results
using Plots
plot(tlist, real(expect(dagger(a) * a, psi_t)), xlabel="Time", ylabel="Photon number", label="⟨n⟩", ylim=(0, 3))
plot(tlist, real(expect(dagger(σm) * σm, psi_t)), xlabel="Time", ylabel="Excited State Population", label="⟨σ₊σ₋⟩")
plot(tlist, real(expect(idOp, psi_t)), xlabel="Time", ylabel="Normalization", label="1")


# Function to calculate probabilities of each Fock state
function fock_probabilities(result, N)
    probs = zeros(length(result.t), N + 1)
    for i in 1:length(result.t)
        for n in 0:N
            fock_state = tensor(fockstate(fb, n), spindown(sb))
            probs[i, n+1] = real(expect(fock_state * dagger(fock_state), result.states[i]))
        end
    end
    return probs
end

# Plot probabilities
plot(tlist, probs, xlabel="Time", ylabel="Probability", label=["|0⟩" "|1⟩" "|2⟩" "|3⟩" "|4⟩" "|5⟩" "|6⟩" "|7⟩" "|8⟩" "|9⟩" "|10⟩"])