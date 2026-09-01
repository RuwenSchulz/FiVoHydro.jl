# ==============================================================================
# src2d/fluxes2d.jl — physical fluxes, wavespeeds, HLLE, geometric sources.
#
# Counterpart of src/fluxes.jl. The flux table is TWOD_PROGRAM.md §1:
#
#   conserved   x-flux      y-flux      source
#   D̃ = τJ^τ    τ J^x       τ J^y       0
#   S^x = T^{τx} T^{xx}      T^{xy}      -S^x/τ
#   S^y = T^{τy} T^{xy}      T^{yy}      -S^y/τ
#   E   = T^{ττ} T^{τx}      T^{τy}      -(E + P + Π + τ²π^{ηη})/τ
#
# Two things vanish relative to the 1-D radial solver: the cylindrical
# `+(P + Π + r²π^{φφ})/r` momentum source and the `1/r ∂_r(r F)` flux weighting.
# In Cartesian Milne the only geometry left is the Bjorken `1/τ` dilution.
# ==============================================================================

"""
    flux_cell_2d!(F, prim, τ, dir, L, model)

Physical flux of every conserved and advected variable through a face normal to
`dir` (`:x` or `:y`). Dissipative dofs are advected as passive scalars with the
fluid 3-velocity, exactly as `src/fluxes.jl:flux_cell!` does in 1-D.
"""
@inline function flux_cell_2d!(F::AbstractVector, prim::PrimIdealVisc2D, τ::Float64,
                               dir::Symbol, L::StateLayout2D, model)
    fill!(F, 0.0)

    ux = prim.ux; uy = prim.uy
    uτ = sqrt(1 + ux*ux + uy*uy)
    invuτ = safe_inv(uτ)
    vx = ux*invuτ; vy = uy*invuτ

    Ptot = prim.P + prim.Pi
    weff = (prim.e + prim.P) + prim.Pi

    Π = shear_tensor_contravariant_2d(ux, uy, uτ, τ, prim.pixx, prim.pixy, prim.piyy, prim.pieta)

    Jx = prim.n*ux + prim.nux
    Jy = prim.n*uy + prim.nuy

    if dir === :x
        F[L.iDtau] = τ * Jx
        F[L.iSx]   = weff*ux*ux + Ptot + Π.xx
        F[L.iSy]   = weff*ux*uy         + Π.xy
        F[L.iE]    = weff*uτ*ux         + Π.tx
        v = vx
    else
        F[L.iDtau] = τ * Jy
        F[L.iSx]   = weff*ux*uy         + Π.xy
        F[L.iSy]   = weff*uy*uy + Ptot  + Π.yy
        F[L.iE]    = weff*uτ*uy         + Π.ty
        v = vy
    end

    if L.hasNu && model.advect_nu
        F[L.iNux] = stored_from_phys(prim.nux) * v
        F[L.iNuy] = stored_from_phys(prim.nuy) * v
    end
    if L.hasPi && model.advect_Pi
        F[L.iPi] = stored_from_phys(prim.Pi) * v
    end
    if L.hasShear && model.advect_pi
        F[L.iPixx]  = stored_from_phys(prim.pixx)  * v
        F[L.iPixy]  = stored_from_phys(prim.pixy)  * v
        F[L.iPiyy]  = stored_from_phys(prim.piyy)  * v
        F[L.iPieta] = stored_from_phys(prim.pieta) * v
    end
    return nothing
end

"""
    wavespeeds_2d(T, μ, ux, uy, eos, dir; ...) -> (λ-, λ+)

Relativistic characteristic speeds normal to `dir`. Unlike the 1-D case, the
TRANSVERSE velocity enters: the general expression is

    λ± = [ v_n(1-c_s²) ± c_s sqrt( (1-v²)(1 - v²c_s² - v_n²(1-c_s²)) ) ] / (1 - v²c_s²)

which reduces to the 1-D `(v ± c_s)/(1 ± v c_s)` of `src/fluxes.jl` when the
transverse velocity vanishes. Using the 1-D form here would UNDER-estimate the
signal speed for oblique flow and break HLLE's positivity guarantee, so the full
form is used.

The dissipative widening of `c_s²` follows the 1-D convention: a conservative
bound so HLLE speeds are never under-estimated.
"""
@inline function wavespeeds_2d(T::Float64, μ::Float64, ux::Float64, uy::Float64, eos,
                               dir::Symbol; Pi_phys::Float64 = 0.0, pinn_phys::Float64 = 0.0,
                               e::Float64 = 0.0, P::Float64 = 0.0)
    cs2 = clamp(eos_cs2(T, μ, eos), 0.0, 0.999)
    w = e + P
    if w > TINY
        cs2 = min(cs2 + (abs(Pi_phys) + abs(pinn_phys))/w, 0.999)
    end

    uτ = sqrt(1 + ux*ux + uy*uy)
    invuτ = safe_inv(uτ)
    vx = ux*invuτ; vy = uy*invuτ
    v2 = vx*vx + vy*vy
    vn = (dir === :x) ? vx : vy

    den = 1 - v2*cs2
    if den <= TINY
        return -1.0, 1.0
    end
    cs = safe_sqrt(cs2)
    rad = (1 - v2) * (den - vn*vn*(1 - cs2))
    disc = cs * safe_sqrt(max(rad, 0.0))

    λm = (vn*(1 - cs2) - disc) / den
    λp = (vn*(1 - cs2) + disc) / den
    return clamp(λm, -1.0, 1.0), clamp(λp, -1.0, 1.0)
end

"""
    hlle_flux_2d!(Fh, col, ULc, URc, primL, primR, eos, τ, dir, L, model, tmpFL, tmpFR)

HLLE numerical flux into column `col` of `Fh`. Identical in structure to
`src/fluxes.jl:hlle_flux_col_cons!`.
"""
@inline function hlle_flux_2d!(Fh::AbstractMatrix, col::Int,
                               ULc::AbstractMatrix, URc::AbstractMatrix,
                               primL::PrimIdealVisc2D, primR::PrimIdealVisc2D,
                               eos, τ::Float64, dir::Symbol,
                               L::StateLayout2D, model,
                               tmpFL::AbstractVector, tmpFR::AbstractVector)
    flux_cell_2d!(tmpFL, primL, τ, dir, L, model)
    flux_cell_2d!(tmpFR, primR, τ, dir, L, model)

    pinnL = (dir === :x) ? primL.pixx : primL.piyy
    pinnR = (dir === :x) ? primR.pixx : primR.piyy

    λmL, λpL = wavespeeds_2d(primL.T, primL.mu, primL.ux, primL.uy, eos, dir;
                             Pi_phys = primL.Pi, pinn_phys = pinnL, e = primL.e, P = primL.P)
    λmR, λpR = wavespeeds_2d(primR.T, primR.mu, primR.ux, primR.uy, eos, dir;
                             Pi_phys = primR.Pi, pinn_phys = pinnR, e = primR.e, P = primR.P)

    sL = min(λmL, λmR)
    sR = max(λpL, λpR)
    Nv = size(Fh, 1)

    if sL >= 0
        @inbounds for a in 1:Nv
            Fh[a,col] = tmpFL[a]
        end
    elseif sR <= 0
        @inbounds for a in 1:Nv
            Fh[a,col] = tmpFR[a]
        end
    else
        inv = 1/(sR - sL + TINY)
        @inbounds for a in 1:Nv
            Fh[a,col] = (sR*tmpFL[a] - sL*tmpFR[a] + sR*sL*(URc[a,col] - ULc[a,col])) * inv
        end
    end
    return sL, sR
end

"""
    source_cell_2d!(S, i, Sx, Sy, E, P, Pi, pieta, τ, L)

Geometric (Bjorken) source. `pieta` is the stored mixed component `π^η_η = τ²π^{ηη}`,
so it enters the energy source directly — no `τ²` factor, unlike the 1-D code which
converts `Π^{ηη}` back to the mixed form at the call site (`src/fluxes.jl`).
"""
@inline function source_cell_2d!(S::AbstractMatrix, i::Int,
                                 Sx::Float64, Sy::Float64, E::Float64,
                                 P::Float64, Pi_phys::Float64, pieta_phys::Float64,
                                 τ::Float64, L::StateLayout2D)
    invτ = safe_inv(τ)
    @inbounds begin
        S[L.iSx, i] = -Sx * invτ
        S[L.iSy, i] = -Sy * invτ
        S[L.iE,  i] = -(E + (P + Pi_phys) + pieta_phys) * invτ
    end
    return nothing
end
