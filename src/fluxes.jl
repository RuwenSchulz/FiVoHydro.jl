# ------------------------------------------------------------
# Fluxes + sources + wavespeeds
# ------------------------------------------------------------
@inline function flux_cell!(F::AbstractVector, prim::PrimIdealVisc, r::Float64, τ::Float64, model::IdealDiffViscModel)
    L = layout(model)
    fill!(F, 0.0)

    ur = prim.ur
    uτ = sqrt(1 + ur^2)
    v  = safe_div(ur, uτ)

    Ptot = prim.P + prim.Pi
    weff = (prim.e + prim.P) + prim.Pi

    Π = shear_tensor_contravariant(ur, uτ, r, τ, prim.piR, prim.piEta)

    F[L.iDtau] = τ * (prim.n * ur + prim.nur)

    F[L.iSr] = weff * ur^2 + Ptot + Π.rr
    F[L.iE]  = weff * ur * uτ + Π.tr

    if L.hasNur
        F[L.iNur] = model.advect_nur ? (stored_from_phys(prim.nur) * v) : 0.0
    end
    if L.hasPi
        F[L.iPi] = model.advect_Pi ? (stored_from_phys(prim.Pi) * v) : 0.0
    end
    if L.hasPiR
        F[L.iPiR] = model.advect_pi ? (stored_from_phys(prim.piR) * v) : 0.0
    end
    if L.hasPiEta
        F[L.iPiEta] = model.advect_pi ? (stored_from_phys(prim.piEta) * v) : 0.0
    end

    return nothing
end

@inline function wavespeeds_from_prim(T::Float64, μ::Float64, ur::Float64, eos;
                                       Pi_phys::Float64=0.0, piR_phys::Float64=0.0,
                                       e::Float64=0.0, P::Float64=0.0)
    cs2 = clamp(eos_cs2(T, μ, eos), 0.0, 0.999)
    # Viscous correction: frozen bulk + shear modify the flux Jacobian eigenvalues.
    # Add a conservative bound so HLLE signal speeds are never under-estimated.
    w = e + P
    if w > TINY
        cs2 = min(cs2 + (abs(Pi_phys) + abs(piR_phys)) / w, 0.999)
    end
    cs = safe_sqrt(cs2)
    uτ = sqrt(1 + ur^2)
    v  = safe_div(ur, uτ)
    ap = (v + cs) / (1 + v*cs + TINY)
    am = (v - cs) / (1 - v*cs + TINY)
    return am, ap
end

# ------------------------------------------------------------
# prim->cons into column, HLLE, source
# ------------------------------------------------------------
@inline function hlle_flux_col_cons!(Fh::AbstractMatrix, col::Int,
                                    ULc::AbstractMatrix, URc::AbstractMatrix,
                                    primL::PrimIdealVisc, primR::PrimIdealVisc,
                                    eos, rL::Float64, rR::Float64, τ::Float64, model::IdealDiffViscModel,
                                    tmpFL::AbstractVector, tmpFR::AbstractVector)
    flux_cell!(tmpFL, primL, rL, τ, model)
    flux_cell!(tmpFR, primR, rR, τ, model)

    λmL, λpL = wavespeeds_from_prim(primL.T, primL.mu, primL.ur, eos;
                                     Pi_phys=primL.Pi, piR_phys=primL.piR,
                                     e=primL.e, P=primL.P)
    λmR, λpR = wavespeeds_from_prim(primR.T, primR.mu, primR.ur, eos;
                                     Pi_phys=primR.Pi, piR_phys=primR.piR,
                                     e=primR.e, P=primR.P)
    sL = min(λmL, λmR)
    sR = max(λpL, λpR)

    Nvars = size(Fh, 1)

    if sL ≥ 0
        @inbounds for a in 1:Nvars
            Fh[a,col] = tmpFL[a]
        end
        return true
    elseif sR ≤ 0
        @inbounds for a in 1:Nvars
            Fh[a,col] = tmpFR[a]
        end
        return true
    else
        inv = 1/(sR - sL + TINY)
        @inbounds for a in 1:Nvars
            UL = ULc[a,col]
            UR = URc[a,col]
            Fh[a,col] = (sR*tmpFL[a] - sL*tmpFR[a] + sR*sL*(UR - UL)) * inv
        end
        return true
    end
end

@inline function hlle_flux_lr_U!(Fh::AbstractMatrix, col::Int,
                                U::AbstractMatrix, iL::Int, iR::Int,
                                primL::PrimIdealVisc, primR::PrimIdealVisc,
                                eos, rL::Float64, rR::Float64, τ::Float64, model::IdealDiffViscModel,
                                tmpFL::AbstractVector, tmpFR::AbstractVector)
    flux_cell!(tmpFL, primL, rL, τ, model)
    flux_cell!(tmpFR, primR, rR, τ, model)

    λmL, λpL = wavespeeds_from_prim(primL.T, primL.mu, primL.ur, eos;
                                     Pi_phys=primL.Pi, piR_phys=primL.piR,
                                     e=primL.e, P=primL.P)
    λmR, λpR = wavespeeds_from_prim(primR.T, primR.mu, primR.ur, eos;
                                     Pi_phys=primR.Pi, piR_phys=primR.piR,
                                     e=primR.e, P=primR.P)
    sL = min(λmL, λmR)
    sR = max(λpL, λpR)

    Nvars = size(Fh, 1)

    if sL ≥ 0
        @inbounds for a in 1:Nvars
            Fh[a,col] = tmpFL[a]
        end
        return true
    elseif sR ≤ 0
        @inbounds for a in 1:Nvars
            Fh[a,col] = tmpFR[a]
        end
        return true
    else
        inv = 1/(sR - sL + TINY)
        @inbounds for a in 1:Nvars
            UL = U[a,iL]
            UR = U[a,iR]
            Fh[a,col] = (sR*tmpFL[a] - sL*tmpFR[a] + sR*sL*(UR - UL)) * inv
        end
        return true
    end
end

@inline function source_cell_fast_col!(S::AbstractMatrix, i::Int,
                                      Sr::Float64, E::Float64,
                                      P::Float64, Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                      ur::Float64,
                                      rC::Float64, τ::Float64,
                                      model::IdealDiffViscModel)
    L = layout(model)
    @inbounds begin
        S[L.iSr, i] = 0.0
        S[L.iE,  i] = 0.0

        invτ = safe_inv(τ)
        S[L.iSr, i] += -Sr * invτ
        uτ = sqrt(1 + ur^2)
        Π = shear_tensor_contravariant(ur, uτ, rC, τ, piR_phys, piEta_phys)

        # Convert contravariant Π^{ηη}, Π^{φφ} back into the effective mixed-diagonal
        # pieces used in the geometric source terms.
        piEta_eff = (τ*τ) * Π.etaeta
        piPhi_eff = (rC*rC) * Π.phph

        S[L.iE,  i] += -(E + ((P + Pi_phys) + piEta_eff)) * invτ
        S[L.iSr, i] += safe_div(((P + Pi_phys) + piPhi_eff), rC)
    end
    return nothing
end
