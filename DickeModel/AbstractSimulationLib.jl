#=
This file defines the Abstract simulation type. This type creates a generic interface for running saving and plotting simulations. It is design to make simluations as reproducable as possible and easy to run in batches.

LATER: Hope to extend the make running on slurm easy
=#

struct AbstractSimulation{paramT,rsltT}
    params::paramT
    rslt::Union{rsltT,nothing}
    _run::Function = paramT -> rsltT
    run::Function = AbstractSimulation -> nothing
    complete::Bool
end

AbstractSimulation.run = (a:AbstractSimulation) -> {
    a.rslt = a._run(a.params);
    a.complete = true;
}