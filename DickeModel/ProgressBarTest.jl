using Distributed, Dates, ProgressMeter

@everywhere begin
    const save_dir = @__DIR__#ENV["SCRATCH"]
    using Pkg
end

outdir = joinpath(save_dir, "results/$(Dates.today())/$(basename(@__FILE__)[begin:end-3])")
mkpath(outdir)
cp(@__FILE__, joinpath(outdir, "gen_script.jl"), force=true)

function setup_worker()
    println("Hello from process $(myid()) on host $(gethostname()).")
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
    cd(save_dir)
end

for i in workers()
    remotecall_wait(setup_worker, i)
end

@everywhere begin
    using JLD2, Dates
    using TerminalLoggers: TerminalLogger
    using Logging: global_logger
    using ProgressMeter: progress, Progress, next!

    global_logger(TerminalLogger(right_justify=150))

    # Create a progress bar to track the progress of all workers combined
    progress = Progress(length(workers()), desc="Overall Progress")

    # Function to update the progress bar
    function update_progress()
        next!(progress)
    end
end

# Function to update the progress bar on each worker
@everywhere function update_worker_progress()
    update_progress()
end

# Call the update_worker_progress function on each worker
for i in workers()
    remotecall_wait(update_worker_progress, i)
end