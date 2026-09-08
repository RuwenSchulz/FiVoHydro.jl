# ==============================================================================
# src2d/reconstruction2d.jl — MUSCL reconstruction in (yT, φ, u^x, u^y).
#
# Counterpart of src/reconstruction.jl, which reconstructs (yT, φ, y=asinh u^r).
# Reconstructing the VELOCITY COMPONENTS rather than a rapidity is the same
# choice made in primrec2d.jl and for the same reason: a 2-D rapidity would need
# a magnitude and an angle, and the angle is undefined at u = 0.
#
# The scheme is UNSPLIT: x and y slopes are built independently on the same cell
# data and both flux divergences are summed into one dU per RK stage. That is the
# structure `Julia/FiVo2DIdeal.jl` already validates on the 2-D
# Riemann and cylindrical-explosion benchmarks.
#
# Directions are handled by STRIDE over the flat index: +1 steps in y (contiguous),
# +Nytot steps in x. See grid2d.jl.
# ==============================================================================

"""
    reconstruct_muscl_2d!(UL, UR, σ, yT, φ, ux, uy, g, dir)

Fill face states for every face normal to `dir`. `UL[:,i]` / `UR[:,i]` are the
left and right states of the face between cell `i` and cell `i + stride`.

Cells without a full 5-point stencil in `dir` fall back to piecewise-constant,
mirroring the 1-D guard `i <= ng+1 || i >= Ntot-ng-1`.
"""
function reconstruct_muscl_2d!(UL::AbstractMatrix, UR::AbstractMatrix, σ::AbstractMatrix,
                               yT::Vector{Float64}, φ::Vector{Float64},
                               ux::Vector{Float64}, uy::Vector{Float64},
                               g::Grid2D, dir::Symbol; limiter = mc_limiter)
    st = (dir === :x) ? g.Nytot : 1
    ng = g.nghost

    fill!(σ, 0.0)

    # ---- slopes: need i-st and i+st ----
    ixlo, ixhi, iylo, iyhi = if dir === :x
        (2, g.Nxtot - 1, 1, g.Nytot)
    else
        (1, g.Nxtot, 2, g.Nytot - 1)
    end

    @inbounds Threads.@threads for ix in ixlo:ixhi
        for iy in iylo:iyhi
            i = lin(g, ix, iy)
            im = i - st; ip = i + st
            σ[1,i] = limiter(yT[i]-yT[im], yT[ip]-yT[i])
            σ[2,i] = limiter(φ[i] -φ[im],  φ[ip] -φ[i])
            σ[3,i] = limiter(ux[i]-ux[im], ux[ip]-ux[i])
            σ[4,i] = limiter(uy[i]-uy[im], uy[ip]-uy[i])
        end
    end

    # ---- face states ----
    fxlo, fxhi, fylo, fyhi = if dir === :x
        (1, g.Nxtot - 1, 1, g.Nytot)
    else
        (1, g.Nxtot, 1, g.Nytot - 1)
    end

    @inbounds Threads.@threads for ix in fxlo:fxhi
        for iy in fylo:fyhi
            i  = lin(g, ix, iy)
            ip = i + st

            # index along `dir` of the left cell, for the stencil guard
            k = (dir === :x) ? ix : iy
            kmax = (dir === :x) ? g.Nxtot : g.Nytot

            if k <= ng || k >= kmax - ng
                UL[1,i] = yT[i];  UL[2,i] = φ[i];  UL[3,i] = ux[i];  UL[4,i] = uy[i]
                UR[1,i] = yT[ip]; UR[2,i] = φ[ip]; UR[3,i] = ux[ip]; UR[4,i] = uy[ip]
                continue
            end

            im = i - st
            ipp = ip + st

            UL[1,i] = yT[i] + 0.5*σ[1,i]
            UL[2,i] = φ[i]  + 0.5*σ[2,i]
            UL[3,i] = ux[i] + 0.5*σ[3,i]
            UL[4,i] = uy[i] + 0.5*σ[4,i]

            UR[1,i] = yT[ip] - 0.5*σ[1,ip]
            UR[2,i] = φ[ip]  - 0.5*σ[2,ip]
            UR[3,i] = ux[ip] - 0.5*σ[3,ip]
            UR[4,i] = uy[ip] - 0.5*σ[4,ip]

            # Monotonicity clamp to local neighbour bounds — the 1-D code's
            # guard against admissible-but-wrong MUSCL wiggles.
            UL[1,i] = clamp(UL[1,i], min(yT[im],yT[i],yT[ip]), max(yT[im],yT[i],yT[ip]))
            UL[2,i] = clamp(UL[2,i], min(φ[im], φ[i], φ[ip]),  max(φ[im], φ[i], φ[ip]))
            UL[3,i] = clamp(UL[3,i], min(ux[im],ux[i],ux[ip]), max(ux[im],ux[i],ux[ip]))
            UL[4,i] = clamp(UL[4,i], min(uy[im],uy[i],uy[ip]), max(uy[im],uy[i],uy[ip]))

            UR[1,i] = clamp(UR[1,i], min(yT[i],yT[ip],yT[ipp]), max(yT[i],yT[ip],yT[ipp]))
            UR[2,i] = clamp(UR[2,i], min(φ[i], φ[ip], φ[ipp]),  max(φ[i], φ[ip], φ[ipp]))
            UR[3,i] = clamp(UR[3,i], min(ux[i],ux[ip],ux[ipp]), max(ux[i],ux[ip],ux[ipp]))
            UR[4,i] = clamp(UR[4,i], min(uy[i],uy[ip],uy[ipp]), max(uy[i],uy[ip],uy[ipp]))
        end
    end
    return nothing
end
