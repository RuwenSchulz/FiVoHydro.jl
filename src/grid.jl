# =========================
# src/grid.jl   (STEP 3)
# =========================

struct Grid1D
    Nr::Int
    nghost::Int
    rmin::Float64
    rmax::Float64
    dr::Float64
    rC::Vector{Float64}  # cell centers
    rF::Vector{Float64}  # faces
end

"""
Axis at a face (r=0 is a face). Cell centers are at (i-0.5)dr in physical domain.
Includes nghost ghost cells on both sides.
"""
function make_grid(Nr::Int; rmax::Float64 = 15.0, nghost::Int = 3)
    @assert Nr > 32
    dr   = rmax / Nr
    rmin = 0.0
    Ntot = Nr + 2*nghost

    rC = [ (rmin - nghost*dr) + (i - 0.5)*dr for i in 1:Ntot ]
    rF = [ (rmin - nghost*dr) + (i - 1)*dr   for i in 1:(Ntot + 1) ]

    return Grid1D(Nr, nghost, rmin, rmax, dr, rC, rF)
end
