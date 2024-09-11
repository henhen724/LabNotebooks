using PyPlot, LinearAlgebra

"""
    blochplot3D(x, y, z; Nspin=nothing, ax=nothing, show_start_end=true, kwargs...)

Plot a 3D trajectory using the Bloch sphere representation.

## Arguments
- `x::Vector`: \$\\langle S_x \\rangle\$ of the trajectory points.
- `y::Vector`: \$\\langle S_y \\rangle\$ of the trajectory points.
- `z::Vector`: \$\\langle S_z \\rangle\$ of the trajectory points.
- `Nspin::Union{Nothing, Real} = nothing`: Maximum spin value. If not provided, it is calculated as twice the maximum absolute value among `x`, `y`, and `z`.
- `ax::Union{Nothing, Any} = nothing`: Axes object to plot on. If not provided, a new figure and axes will be created.
- `show_start_end::Bool = true`: Whether to show markers for the start and end points of the trajectory.
- `kwargs...`: Additional keyword arguments to be passed to the `plot_surface` and `plot3D` functions.

## Example
"""
function blochplot3D(x, y, z; Nspin=nothing, ax=nothing, show_start_end=true, kwargs...)
    if Nspin isa Nothing
        Nspin = max(abs.([x..., y..., z...])...) * 2.0
    end
    if ax isa Nothing
        fig = figure()
        ax = fig.add_subplot(111, projection="3d")
    end
    u = LinRange(0.0, 2 * pi, 100)
    v = LinRange(0.0, pi, 100)
    xS = (Nspin / 2) * cos.(u) * sin.(v)'
    yS = (Nspin / 2) * sin.(u) * sin.(v)'
    zS = (Nspin / 2) * ones(size(u)) * cos.(v)'
    ax.plot_surface(xS, yS, zS, color="gray", alpha=0.2)
    ax.plot3D(x, y, z, label="Trajectory")
    if show_start_end
        ax.plot3D([x[begin]], [y[begin]], [z[begin]], marker="o", color="red", label="Start")
        ax.plot3D([x[end]], [y[end]], [z[end]], marker="o", color="green", label="End")
    end
    ax.set_xlim(-Nspin / 2, Nspin / 2)
    ax.set_ylim(-Nspin / 2, Nspin / 2)
    ax.set_zlim(-Nspin / 2, Nspin / 2)
end


