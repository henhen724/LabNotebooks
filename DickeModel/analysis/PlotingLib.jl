using PyPlot, LinearAlgebra

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
        ax.plot3D([x[1]], [y[1]], [z[1]], marker="o", color="red", label="Start")
        ax.plot3D([x[end]], [y[end]], [z[end]], marker="o", color="green", label="End")
    end
    ax.set_xlim(-Nspin / 2, Nspin / 2)
    ax.set_ylim(-Nspin / 2, Nspin / 2)
    ax.set_zlim(-Nspin / 2, Nspin / 2)
end


