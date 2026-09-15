# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
function apply_bc!(U, grid, τ, model::IdealDiffViscModel)
    L    = model.layout
    ng   = grid.nghost
    Ntot = size(U,2)

    @inbounds for g in 1:ng
        iG = ng + 1 - g
        iI = ng + g
        for a in 1:length(L.names)
            U[a,iG] = L.odd[a] ? -U[a,iI] : U[a,iI]
        end
    end

    i0 = ng + 1
    # ⚠ S_r is the CONSERVED radial momentum, and cell i0 is at r = dr/2, not r = 0. Zeroing it
    # imposes u^r = 0 half a cell off the axis and throws away an O(dr) quantity. See
    # `AXIS_CELL_EXACT` in src/rhs.jl for the measurement and for how to restore the old behaviour.
    @inbounds if !AXIS_CELL_EXACT
        U[L.iSr, i0] = 0.0
    end

    # Axis regularity for shear: at r=0 symmetry implies transverse isotropy,
    # i.e. π^r_r = π^φ_φ. With tracelessness this gives 2π^r_r + π^η_η = 0.
    # We enforce this only at the first physical cell (r=dr/2) to avoid a
    # spurious left-edge bend in shear profiles.
    if L.hasPiR && L.hasPiEta
        @inbounds begin
            piEta_phys = phys_from_stored(U[L.iPiEta, i0])
            piR_phys   = -0.5 * piEta_phys
            U[L.iPiR, i0] = stored_from_phys(piR_phys)
        end
    end

    i_last = Ntot - ng
    @inbounds for g in 1:ng
        iG = Ntot - ng + g
        for a in 1:length(L.names)
            U[a, iG] = U[a, i_last]
        end
    end
    return nothing
end
