using DifferentialEquations, ProgressLogging, TerminalLoggers, Logging

global_logger(TerminalLogger(right_justify=150))

function simple_ode!(du, u, p, t)
    du[1] = -u[1]
end

# Initial conditions and time span
u0 = [1.0]
tspan = (0.0, 1000.0)

function prob_func(prob, i, repeat)
    ODEProblem(simple_ode!, u0, tspan)
end

# Define the output function
function output_func(sol, i)
    println("Solution $i completed.")
    return (sol, false)
end

# Initialize the ensemble problem
init_prob = prob_func(nothing, 1, 1)
ensembleprob = EnsembleProblem(init_prob, prob_func=prob_func, output_func=output_func)

# Solve the ensemble problem with progress bar
sol = solve(ensembleprob, Tsit5(), EnsembleThreads(), trajectories=10, progress=true)