# ==============================================================================
# src2d/grid2d.jl — Grid2D: transverse Cartesian (x,y) with ghost cells on all
# four sides, flat cell indexing.
#
# There is no axis and no origin: unlike the 1-D radial grid (src/grid.jl), which
# puts r=0 on a face and needs odd-parity ghost reflection, the transverse plane
# is a plain Cartesian box. Boundaries are outflow on all four sides.
#
# FLAT INDEXING is the load-bearing choice of this port. State is stored as
# U[a, i] with a single cell index
#
#       i = (ix - 1) * Nytot + iy,        ix, iy ∈ 1:Nxtot, 1:Nytot
#
# exactly as the 1-D solver stores U[a, i]. That is what lets `src/timestepper.jl`,
# `src/mood.jl` and the Work arrays carry over unchanged: they never interpret i.
# y is the fast index, so a y-sweep is contiguous and an x-sweep strides by Nytot.
# ==============================================================================

struct Grid2D
    Nx::Int
    Ny::Int
    nghost::Int

    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64

    dx::Float64
    dy::Float64

    Nxtot::Int          # Nx + 2*nghost
    Nytot::Int          # Ny + 2*nghost
    Ntot::Int           # Nxtot * Nytot

    xC::Vector{Float64} # cell centers, length Nxtot (includes ghosts)
    yC::Vector{Float64} # cell centers, length Nytot (includes ghosts)
end

"""
    make_grid2d(Nx, Ny; xmax, ymax, nghost=3)

Cell-centered Cartesian grid on `[-xmax, xmax] × [-ymax, ymax]` with `nghost`
ghost layers on every side. Physical cell centers are at
`xmin + (i - nghost - 0.5) dx`, so the box is symmetric about the origin and no
cell center sits exactly on `x=0` or `y=0` for even `Nx`, `Ny`.

Symmetry about the origin is deliberate: gate G4 seeds an azimuthally symmetric
IC and measures the x↔y asymmetry, which is only meaningful on a grid that does
not itself distinguish the axes.
"""
function make_grid2d(Nx::Int, Ny::Int; xmax::Float64 = 20.0, ymax::Float64 = 20.0,
                     nghost::Int = 3)
    @assert Nx > 8 && Ny > 8
    @assert xmax > 0 && ymax > 0

    xmin = -xmax
    ymin = -ymax

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

    Nxtot = Nx + 2*nghost
    Nytot = Ny + 2*nghost

    xC = [ xmin + (i - nghost - 0.5)*dx for i in 1:Nxtot ]
    yC = [ ymin + (j - nghost - 0.5)*dy for j in 1:Nytot ]

    return Grid2D(Nx, Ny, nghost, xmin, xmax, ymin, ymax, dx, dy,
                  Nxtot, Nytot, Nxtot*Nytot, xC, yC)
end

"""
    make_grid_2d(Nx, Ny; kwargs...)

Alias of [`make_grid2d`](@ref), spelled like the 1+1D `make_grid_1d`. The two
solvers' library interfaces are otherwise named alike (`build_model_{1,2}d`,
`run_sim_{1,2}d!`, `fields_{1,2}d`); this was the one name that broke the pattern,
and `make_grid2d` is kept because ~60 call sites use it.
"""
const make_grid_2d = make_grid2d

# ---- flat index helpers -------------------------------------------------------

@inline lin(g::Grid2D, ix::Int, iy::Int) = (ix - 1)*g.Nytot + iy

@inline function unlin(g::Grid2D, i::Int)
    ix = div(i - 1, g.Nytot) + 1
    iy = i - (ix - 1)*g.Nytot
    return ix, iy
end

"""Interior (non-ghost) index ranges."""
@inline xrange_int(g::Grid2D) = (g.nghost+1):(g.nghost+g.Nx)
@inline yrange_int(g::Grid2D) = (g.nghost+1):(g.nghost+g.Ny)

@inline is_interior(g::Grid2D, ix::Int, iy::Int) =
    (ix > g.nghost) & (ix <= g.nghost + g.Nx) &
    (iy > g.nghost) & (iy <= g.nghost + g.Ny)

"""Transverse cell area — uniform, but named so call sites read as volume weights."""
@inline cell_area(g::Grid2D) = g.dx * g.dy
