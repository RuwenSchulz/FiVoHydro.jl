# ==============================================================================
# src2d/bc2d.jl — boundary conditions.
#
# Counterpart of src/boundary_conditions.jl, and much shorter: the transverse
# plane has no axis. The 1-D solver must reflect odd-parity variables across
# r = 0, pin S_r = 0 on the first face, and impose the axis-regularity condition
# π^r_r = π^φ_φ; none of that has a 2-D analogue. All four sides are plain
# outflow (zero-gradient) copies.
#
# `:periodic` is provided for the sound-wave benchmark (gate G3), where outflow
# would pollute a travelling wave.
# ==============================================================================

function apply_bc_2d!(U::AbstractMatrix, g::Grid2D, L::StateLayout2D; bc::Symbol = :outflow)
    ng = g.nghost
    nv = nvars(L)
    ix0 = ng + 1; ix1 = ng + g.Nx
    iy0 = ng + 1; iy1 = ng + g.Ny

    if bc === :periodic
        @inbounds for k in 1:ng
            for iy in 1:g.Nytot
                iL = lin(g, ng + 1 - k, iy);  sL = lin(g, ix1 + 1 - k, iy)
                iR = lin(g, ix1 + k, iy);     sR = lin(g, ix0 - 1 + k, iy)
                for a in 1:nv
                    U[a,iL] = U[a,sL]; U[a,iR] = U[a,sR]
                end
            end
        end
        @inbounds for k in 1:ng
            for ix in 1:g.Nxtot
                iB = lin(g, ix, ng + 1 - k);  sB = lin(g, ix, iy1 + 1 - k)
                iT = lin(g, ix, iy1 + k);     sT = lin(g, ix, iy0 - 1 + k)
                for a in 1:nv
                    U[a,iB] = U[a,sB]; U[a,iT] = U[a,sT]
                end
            end
        end
        return nothing
    end

    # ---- outflow: copy the nearest interior cell outward ----
    # x faces first, over the full y extent including y-ghosts, then y faces over
    # the full x extent. Doing x first and then y (rather than the reverse) fills
    # the corner blocks consistently from the x-edge values.
    @inbounds for k in 1:ng
        for iy in 1:g.Nytot
            iL = lin(g, ng + 1 - k, iy); sL = lin(g, ix0, iy)
            iR = lin(g, ix1 + k, iy);    sR = lin(g, ix1, iy)
            for a in 1:nv
                U[a,iL] = U[a,sL]; U[a,iR] = U[a,sR]
            end
        end
    end
    @inbounds for k in 1:ng
        for ix in 1:g.Nxtot
            iB = lin(g, ix, ng + 1 - k); sB = lin(g, ix, iy0)
            iT = lin(g, ix, iy1 + k);    sT = lin(g, ix, iy1)
            for a in 1:nv
                U[a,iB] = U[a,sB]; U[a,iT] = U[a,sT]
            end
        end
    end
    return nothing
end
