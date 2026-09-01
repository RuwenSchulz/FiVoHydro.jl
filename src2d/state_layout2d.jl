# ==============================================================================
# src2d/state_layout2d.jl
#
# State layout for the 2+1D bulk+charge solver. Counterpart of
# src/state_layout.jl.
#
# CONSERVED SET (see TWOD_PROGRAM.md §1):
#     Dtau = τ J^τ        E = T^{ττ}        Sx = T^{τx}       Sy = T^{τy}
# Only Dtau carries the explicit τ weight; the τ factor for E and S^i is absorbed
# into the geometric source. This mirrors the 1-D convention exactly
# (src/primrec.jl:778-780), so the two solvers can be compared field by field.
#
# ADVECTED DISSIPATIVE DOFS (relaxed by operator splitting, not conserved):
#     nux, nuy                 charge diffusion current ν^x, ν^y
#     Pi                       bulk pressure Π
#     pixx, pixy, piyy, pieta  shear (pieta = π^η_η = τ²π^{ηη}; one redundant dof,
#                              see shear2d.jl)
#
# No parity metadata: unlike the 1-D radial grid there is no axis, so ghost cells
# are plain outflow copies. The `odd` machinery of StateLayout has no 2-D analogue.
# ==============================================================================

struct StateLayout2D
    names::Vector{Symbol}
    idx::Dict{Symbol,Int}

    iDtau::Int
    iSx::Int
    iSy::Int
    iE::Int

    hasNu::Bool
    iNux::Int
    iNuy::Int

    hasPi::Bool
    iPi::Int

    hasShear::Bool
    iPixx::Int
    iPixy::Int
    iPiyy::Int
    iPieta::Int
end

"""
    make_layout2d(; with_charge=true, with_bulk=true, with_shear=true)

Build the state layout. The conserved four are always present; each dissipative
sector can be switched off, which removes its fields from the state vector
entirely (rather than carrying zeros) so that ideal-fluid gates run the bare
scheme. Field order is fixed and dense.
"""
function make_layout2d(; with_charge::Bool = true, with_bulk::Bool = true,
                         with_shear::Bool = true)
    names = Symbol[:Dtau, :Sx, :Sy, :E]
    with_charge && append!(names, (:nux, :nuy))
    with_bulk   && push!(names, :Pi)
    with_shear  && append!(names, (:pixx, :pixy, :piyy, :pieta))

    idx = Dict(s => i for (i, s) in pairs(names))

    return StateLayout2D(
        names, idx,
        idx[:Dtau], idx[:Sx], idx[:Sy], idx[:E],
        with_charge, get(idx, :nux, 0), get(idx, :nuy, 0),
        with_bulk,   get(idx, :Pi, 0),
        with_shear,  get(idx, :pixx, 0), get(idx, :pixy, 0),
                     get(idx, :piyy, 0), get(idx, :pieta, 0),
    )
end

@inline nvars(L::StateLayout2D) = length(L.names)

"""Pull the physical dissipative dofs of cell `i` out of `U`, zero where absent."""
@inline function dissipatives_at(U::AbstractMatrix, i::Int, L::StateLayout2D)
    @inbounds begin
        nux   = L.hasNu    ? phys_from_stored(U[L.iNux,   i]) : 0.0
        nuy   = L.hasNu    ? phys_from_stored(U[L.iNuy,   i]) : 0.0
        Pi    = L.hasPi    ? phys_from_stored(U[L.iPi,    i]) : 0.0
        pixx  = L.hasShear ? phys_from_stored(U[L.iPixx,  i]) : 0.0
        pixy  = L.hasShear ? phys_from_stored(U[L.iPixy,  i]) : 0.0
        piyy  = L.hasShear ? phys_from_stored(U[L.iPiyy,  i]) : 0.0
        pieta = L.hasShear ? phys_from_stored(U[L.iPieta, i]) : 0.0
    end
    return nux, nuy, Pi, pixx, pixy, piyy, pieta
end
