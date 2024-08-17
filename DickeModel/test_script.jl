using Distributed

# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
end

# load dependencies in a *separate* @everywhere block
@everywhere begin
    # load dependencies
    using ProgressMeter
    using CSV

    # helper functions
    function process(infile, outfile)
        # read file from disk
        csv = CSV.File(infile)

        # perform calculations
        sleep(60)

        # save new file to disk
        CSV.write(outfile, csv)
    end
end

using Dates

indir = joinpath(@__DIR__, "data")
outdir = joinpath(@__DIR__, "results/$(Dates.today())")
mkpath(outdir)
cp(@__FILE__, joinpath(outdir, "gen_script.jl"), force=true)

# files to process
infiles = readdir(indir, join=true)
outfiles = joinpath.(outdir, basename.(infiles))
nfiles = length(infiles)

status = @showprogress pmap(1:nfiles) do i
    try
        process(infiles[i], outfiles[i])
        true # success
    catch e
        false # failure
    end
end