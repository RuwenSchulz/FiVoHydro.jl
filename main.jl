#!/usr/bin/env julia
# ==============================================================================
# hydro_ideal_axisface_trustworthy_stable_phi_diff_visc_clean.jl
#
# Cleaned + fixes applied:
#   - Introduced TINY (=1e-300) and removed all Float64-underflow "1e-500" guards
#   - HLLE / wavespeeds / fits now use TINY-guarded denominators
#   - diff_tauN: K2 protected against near-zero to avoid Inf ratios
#   - compute_dt_from_work now honors CFL, CFLτ, diff_dt_coeff and is called with run-time values
#   - Added prime_work_from_U!() to initialize work caches before the first dt-from-work call
#   - (Recommended physics) ν_NS uses the conventional sign: ν_NS = -κ uτ^2 ∂r α
#
# NOTE ON SIGN CONVENTIONS (IMPORTANT):
#   stored = DISS_SIGN * physical   (DISS_SIGN = ±1)
# ==============================================================================
module hydro

using LinearAlgebra
using Printf
using CSV
using Logging
using Base.Threads
using Interpolations
using SpecialFunctions
using Tables

# ------------------------------------------------------------
# Load split files
# ------------------------------------------------------------
const _SRC = joinpath(@__DIR__, "src")
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "grid.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "io.jl"))
include(joinpath(_SRC, "primitives.jl"))
include(joinpath(_SRC, "state_layout.jl"))
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "diagnostics.jl"))
include(joinpath(_SRC, "floors.jl"))
include(joinpath(_SRC, "primrec.jl"))
include(joinpath(_SRC, "work.jl"))
include(joinpath(_SRC, "relaxation_laws.jl"))
include(joinpath(_SRC, "dissipation.jl"))

# ------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------
function setup_logger!(; level::LogLevel=Logging.Info)
    global_logger(ConsoleLogger(stderr, level))
    return nothing
end








# ------------------------------------------------------------
# Admissibility (MOOD)
# ------------------------------------------------------------
@inline function admissible_state_fast(U::AbstractMatrix, i::Int, τ::Float64, model::IdealDiffViscModel;
                                       Emin::Float64=E_FLOOR,
                                       χ::Float64=χ_SrE)
    L = model.layout
    Dtau = U[L.iDtau,i]
    Sr   = U[L.iSr,i]
    E    = U[L.iE,i]

    if !(isfinite(Dtau) & isfinite(Sr) & isfinite(E)) || (Dtau < 0.0) || (E < Emin)
        return false
    end
    if L.hasNur   && !isfinite(U[L.iNur,i]);   return false; end
    if L.hasPi    && !isfinite(U[L.iPi,i]);    return false; end
    if L.hasPiR   && !isfinite(U[L.iPiR,i]);   return false; end
    if L.hasPiEta && !isfinite(U[L.iPiEta,i]); return false; end

    if abs(Sr) <= 0.98*χ*E
        return true
    end
    if E < 100Emin
        return true
    end

    prim = cons_to_prim_col(U, i, τ, model)
    return prim.ok
end

function any_bad(U, grid, τ, model::IdealDiffViscModel, work; Emin::Float64=E_FLOOR)
    ng = grid.nghost
    bad_tls = work.bad_tls
    fill!(bad_tls, false)
    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        tid = Threads.threadid()
        if !admissible_state_fast(U, i, τ, model; Emin=Emin)
            bad_tls[tid] = true
        end
    end
    @inbounds for t in 1:length(bad_tls)
        if bad_tls[t]
            return true
        end
    end
    return false
end

function any_bad(U, grid, τ, model::IdealDiffViscModel, work; Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    bad_tls = work.bad_tls
    fill!(bad_tls, false)
    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        tid = Threads.threadid()
        if !admissible_state_fast(U, i, τ, model; Emin=Emin, χ=χ)
            bad_tls[tid] = true
        end
    end
    @inbounds for t in 1:length(bad_tls)
        if bad_tls[t]
            return true
        end
    end
    return false
end

function mark_bad!(bad, U, grid, τ, model::IdealDiffViscModel; Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    fill!(bad, false)
    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        bad[i] = !admissible_state_fast(U, i, τ, model; Emin=Emin, χ=χ)
    end
    return nothing
end

function expand_bad!(bad::BitVector, tmp::BitVector, grid; radius::Int=1)
    ng = grid.nghost
    N  = length(bad)
    phys_lo = ng+1
    phys_hi = N-ng

    copyto!(tmp, bad)
    copyto!(bad, tmp)

    @inbounds for i in phys_lo:phys_hi
        if tmp[i]
            for k in 1:radius
                if i-k >= phys_lo; bad[i-k] = true; end
                if i+k <= phys_hi; bad[i+k] = true; end
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------
# Failure debugging (window dump)
# ------------------------------------------------------------
@inline function _bad_reason(U::AbstractMatrix, i::Int, τ::Float64, model::IdealDiffViscModel, work;
                            Emin::Float64=E_FLOOR,
                            χ::Float64=χ_SrE,
                            v_eps::Float64=1e-12,
                            Rmax::Float64=10.0)
    L = model.layout
    Dtau = U[L.iDtau,i]
    Sr   = U[L.iSr,i]
    E    = U[L.iE,i]

    if !(isfinite(Dtau) & isfinite(Sr) & isfinite(E))
        return :nonfinite_conserved
    end
    if Dtau < 0.0
        return :negative_Dtau
    end
    if E < Emin
        return :E_below_floor
    end

    if L.hasNur   && !isfinite(U[L.iNur,i]);   return :nonfinite_nur; end
    if L.hasPi    && !isfinite(U[L.iPi,i]);    return :nonfinite_Pi; end
    if L.hasPiR   && !isfinite(U[L.iPiR,i]);   return :nonfinite_piR; end
    if L.hasPiEta && !isfinite(U[L.iPiEta,i]); return :nonfinite_piEta; end

    if work !== nothing && work.ok[i]
        v = work.vC[i]
        if !(isfinite(v) && abs(v) < (1 - v_eps))
            return :superluminal_v
        end

        cap = max(work.P[i] + work.e[i], 0.0)
        if cap > 0
            if model.enable_bulk && L.hasPi
                Π = abs(phys_from_stored(U[L.iPi,i]))
                if Π > Rmax * cap
                    return :Pi_too_large
                end
            end
            if model.enable_shear
                if L.hasPiR
                    pr = abs(phys_from_stored(U[L.iPiR,i]))
                    if pr > Rmax * cap
                        return :piR_too_large
                    end
                end
                if L.hasPiEta
                    pe = abs(phys_from_stored(U[L.iPiEta,i]))
                    if pe > Rmax * cap
                        return :piEta_too_large
                    end
                end
            end
        end

        if model.enable_diff && L.hasNur
            y = work.y[i]
            if isfinite(y)
                uτ = cosh(y)
                ncap = max(work.n[i] * uτ, 0.0)
                if ncap > 0
                    ν = abs(phys_from_stored(U[L.iNur,i]))
                    if ν > Rmax * ncap
                        return :nur_too_large
                    end
                end
            end
        end
    end

    # Match admissible_state_fast: cheap Sr/E gate before running prim recovery.
    if abs(Sr) <= 0.98*χ*E
        return :unknown
    end
    if E < 100Emin
        return :unknown
    end

    prim = cons_to_prim_col(U, i, τ, model)
    return prim.ok ? :unknown : :prim_recovery_failed
end

function _find_first_bad(U, grid, τ, model::IdealDiffViscModel, work; Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    for i in (ng+1):(size(U,2)-ng)
        why = _bad_reason(U, i, τ, model, work; Emin=Emin, χ=χ)
        if why != :unknown
            return i, why
        end
    end
    return 0, :none
end

function _dump_failure_window_csv(path::String, U, grid, τ, model::IdealDiffViscModel, work, i::Int; radius::Int=5)
    L = model.layout
    ng = grid.nghost
    N  = size(U,2)
    ilo = max(i - radius, ng+1)
    ihi = min(i + radius, N-ng)

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "τ,i,rC,Dtau,Sr,E,nur,Pi,piR,piEta,yT,phi,mu,y,vC,n,e,P,ok")
        for j in ilo:ihi
            rC = grid.rC[j]
            Dtau = U[L.iDtau,j]
            Sr   = U[L.iSr,j]
            E    = U[L.iE,j]
            nur  = (L.hasNur   ? phys_from_stored(U[L.iNur,j])   : 0.0)
            Pi   = (L.hasPi    ? phys_from_stored(U[L.iPi,j])    : 0.0)
            piR  = (L.hasPiR   ? phys_from_stored(U[L.iPiR,j])   : 0.0)
            piEta= (L.hasPiEta ? phys_from_stored(U[L.iPiEta,j]) : 0.0)

            yT  = (work === nothing ? NaN : work.yT[j])
            phi = (work === nothing ? NaN : work.phi[j])
            mu  = (work === nothing ? NaN : work.mu[j])
            y   = (work === nothing ? NaN : work.y[j])
            vC  = (work === nothing ? NaN : work.vC[j])
            n   = (work === nothing ? NaN : work.n[j])
            e   = (work === nothing ? NaN : work.e[j])
            P   = (work === nothing ? NaN : work.P[j])
            ok  = (work === nothing ? true : work.ok[j])

            println(io, string(τ, ",", j, ",", rC, ",",
                               Dtau, ",", Sr, ",", E, ",",
                               nur, ",", Pi, ",", piR, ",", piEta, ",",
                               yT, ",", phi, ",", mu, ",", y, ",", vC, ",",
                               n, ",", e, ",", P, ",", ok))
        end
    end
    return nothing
end

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
    @inbounds U[L.iSr, i0] = 0.0

    i_last = Ntot - ng
    @inbounds for g in 1:ng
        iG = Ntot - ng + g
        for a in 1:length(L.names)
            U[a, iG] = U[a, i_last]
        end
    end
    return nothing
end

# ------------------------------------------------------------
# Fluxes + sources + wavespeeds
# ------------------------------------------------------------
@inline function flux_cell!(F::AbstractVector, prim::PrimIdealVisc, τ::Float64, model::IdealDiffViscModel)
    L = layout(model)
    fill!(F, 0.0)

    ur = prim.ur
    uτ = sqrt(1 + ur^2)
    v  = ur / max(uτ, 1e-50)

    Ptot = prim.P + prim.Pi
    weff = (prim.e + prim.P) + prim.Pi

    pi_tr = (uτ*ur) * prim.piR
    pi_rr = (uτ*uτ) * prim.piR

    F[L.iDtau] = τ * (prim.n * ur + prim.nur)

    F[L.iSr] = weff * ur^2 + Ptot + pi_rr
    F[L.iE]  = weff * ur * uτ + pi_tr

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

@inline function wavespeeds_from_prim(T::Float64, μ::Float64, ur::Float64, eos)
    cs = safe_sqrt(clamp(eos_cs2(T, μ, eos), 0.0, 0.999999))
    uτ = sqrt(1 + ur^2)
    v  = ur / max(uτ, 1e-50)
    ap = (v + cs) / (1 + v*cs + TINY)
    am = (v - cs) / (1 - v*cs + TINY)
    return am, ap
end

# ------------------------------------------------------------
# Reconstruction in (logT, φ, y)
# ------------------------------------------------------------
function reconstruct_muscl_prims!(UL, UR, σ, yT, φ, y, grid; limiter=mc_limiter)
    Ntot = length(yT)
    ng = grid.nghost
    fill!(σ, 0.0)

    @inbounds for i in (ng+2):(Ntot-ng-1)
        σ[1,i] = limiter(yT[i]-yT[i-1], yT[i+1]-yT[i])
        σ[2,i] = limiter(φ[i]-φ[i-1],   φ[i+1]-φ[i])
        σ[3,i] = limiter(y[i]-y[i-1],   y[i+1]-y[i])
    end

    @inbounds for i in 1:(Ntot-1)
        if i <= (ng+1) || i >= (Ntot-ng-1)
            UL[1,i] = yT[i];   UL[2,i] = φ[i];   UL[3,i] = y[i]
            UR[1,i] = yT[i+1]; UR[2,i] = φ[i+1]; UR[3,i] = y[i+1]
        else
            UL[1,i] = yT[i]   + 0.5*σ[1,i]
            UL[2,i] = φ[i]    + 0.5*σ[2,i]
            UL[3,i] = y[i]    + 0.5*σ[3,i]

            UR[1,i] = yT[i+1] - 0.5*σ[1,i+1]
            UR[2,i] = φ[i+1]  - 0.5*σ[2,i+1]
            UR[3,i] = y[i+1]  - 0.5*σ[3,i+1]

            # Monotonicity-preserving clamp: interface extrapolations must stay within
            # local neighbor bounds, otherwise MUSCL can introduce wiggles that remain
            # "admissible" but are physically/visually wrong.
            yTminL = min(yT[i-1], yT[i], yT[i+1]); yTmaxL = max(yT[i-1], yT[i], yT[i+1])
            φminL  = min(φ[i-1],  φ[i],  φ[i+1]);  φmaxL  = max(φ[i-1],  φ[i],  φ[i+1])
            yminL  = min(y[i-1],  y[i],  y[i+1]);  ymaxL  = max(y[i-1],  y[i],  y[i+1])

            yTminR = min(yT[i], yT[i+1], yT[i+2]); yTmaxR = max(yT[i], yT[i+1], yT[i+2])
            φminR  = min(φ[i],  φ[i+1],  φ[i+2]);  φmaxR  = max(φ[i],  φ[i+1],  φ[i+2])
            yminR  = min(y[i],  y[i+1],  y[i+2]);  ymaxR  = max(y[i],  y[i+1],  y[i+2])

            UL[1,i] = clamp(UL[1,i], yTminL, yTmaxL)
            UL[2,i] = clamp(UL[2,i], φminL,  φmaxL)
            UL[3,i] = clamp(UL[3,i], yminL,  ymaxL)

            UR[1,i] = clamp(UR[1,i], yTminR, yTmaxR)
            UR[2,i] = clamp(UR[2,i], φminR,  φmaxR)
            UR[3,i] = clamp(UR[3,i], yminR,  ymaxR)
        end
    end
    return nothing
end

# ------------------------------------------------------------
# prim->cons into column, HLLE, source
# ------------------------------------------------------------


@inline function hlle_flux_col_cons!(Fh::AbstractMatrix, col::Int,
                                    ULc::AbstractMatrix, URc::AbstractMatrix,
                                    primL::PrimIdealVisc, primR::PrimIdealVisc,
                                    eos, τ::Float64, model::IdealDiffViscModel,
                                    tmpFL::AbstractVector, tmpFR::AbstractVector)
    flux_cell!(tmpFL, primL, τ, model)
    flux_cell!(tmpFR, primR, τ, model)

    λmL, λpL = wavespeeds_from_prim(primL.T, primL.mu, primL.ur, eos)
    λmR, λpR = wavespeeds_from_prim(primR.T, primR.mu, primR.ur, eos)
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
                                eos, τ::Float64, model::IdealDiffViscModel,
                                tmpFL::AbstractVector, tmpFR::AbstractVector)
    flux_cell!(tmpFL, primL, τ, model)
    flux_cell!(tmpFR, primR, τ, model)

    λmL, λpL = wavespeeds_from_prim(primL.T, primL.mu, primL.ur, eos)
    λmR, λpR = wavespeeds_from_prim(primR.T, primR.mu, primR.ur, eos)
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
                                      rC::Float64, τ::Float64,
                                      model::IdealDiffViscModel)
    L = layout(model)
    @inbounds begin
        S[L.iSr, i] = 0.0
        S[L.iE,  i] = 0.0

        invτ = 1 / max(τ, 1e-50)
        S[L.iSr, i] += -Sr * invτ
        S[L.iE,  i] += -(E + (P + Pi_phys + piEta_phys)) * invτ

        piPhi = -piR_phys - piEta_phys
        S[L.iSr, i] += (P + Pi_phys + piPhi) / max(rC, 1e-50)
    end
    return nothing
end


# ------------------------------------------------------------
# Work arrays MOVED to src/work.jl
# ------------------------------------------------------------

# ------------------------------------------------------------
# RHS (unchanged from your version; kept for completeness)
# ------------------------------------------------------------
function rhs!(dU, U, grid, τ, model::IdealDiffViscModel, work::Work1D;
              Emin::Float64=E_FLOOR,
              force_first_order::Union{Nothing,BitVector}=nothing,
              diag::Union{Nothing,DiagCounters}=nothing)

    apply_bc!(U, grid, τ, model)

    Nvars, Ntot = size(U)
    L   = layout(model)
    eos = model.eos
    ng  = grid.nghost

    primfail_tls = work.primfail_tls
    fill!(primfail_tls, 0)

    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D  = U[L.iDtau,i] / max(τ, 1e-50)
        Sr = U[L.iSr,i]
        E  = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos;
            yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
            maxit=40
        )

        if ok
            Tm = max(T, T_MIN)
            φ  = (μ - hq_mass(eos)) / Tm

            work.yT[i]    = log(Tm)
            work.phi[i]   = φ
            work.mu[i]    = μ
            work.alpha[i] = μ / Tm
            work.y[i]     = asinh(ur)

            work.P[i]     = P
            work.n[i]     = n
            work.e[i]     = e
            work.ok[i]    = true

            uτ = sqrt(1 + ur^2)
            work.vC[i] = ur / max(uτ, 1e-50)

            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = work.phi[i]
            work.x0_y[i]   = work.y[i]
        else
            primfail_tls[tid] += 1

            if diag !== nothing
                # best-effort: record one failing cell (race is acceptable for diagnostics)
                diag.last_prim_fail_i = i
                diag.last_prim_fail_tau = τ
                diag.last_prim_fail_Dtau = U[L.iDtau,i]
                diag.last_prim_fail_Sr = U[L.iSr,i]
                diag.last_prim_fail_E = U[L.iE,i]
                diag.last_prim_fail_nur_stored = (L.hasNur ? U[L.iNur,i] : 0.0)
                diag.last_prim_fail_Pi_stored = (L.hasPi ? U[L.iPi,i] : 0.0)
                diag.last_prim_fail_piR_stored = (L.hasPiR ? U[L.iPiR,i] : 0.0)
                diag.last_prim_fail_piEta_stored = (L.hasPiEta ? U[L.iPiEta,i] : 0.0)
            end

            Tv = T_MIN
            μv = hq_mass(eos)
            Pv, _, ev = eos_Pne(Tv, μv, eos)

            work.yT[i]    = log(Tv)
            work.phi[i]   = 0.0
            work.mu[i]    = μv
            work.alpha[i] = μv / Tv
            work.y[i]     = 0.0

            work.P[i]     = Pv
            work.n[i]     = 0.0
            work.e[i]     = max(ev, 0.0)
            work.ok[i]    = true

            work.vC[i]    = 0.0
        end
    end

    if diag !== nothing
        diag_add!(diag; primfail=sum(primfail_tls))
    end

    i0 = ng + 1
    work.y[i0]  = 0.0
    work.vC[i0] = 0.0

    reconstruct_muscl_prims!(work.ULp, work.URp, work.σp, work.yT, work.phi, work.y, grid)

    fill!(work.Fh, 0.0)

    Threads.@threads for i in 1:(Ntot-1)
        tid   = Threads.threadid()
        tmpFL = work.tmpFL[tid]
        tmpFR = work.tmpFR[tid]

        use_pc = false
        if force_first_order !== nothing
            use_pc = force_first_order[i] || force_first_order[i+1]
        end

        if !use_pc
            yTL = work.ULp[1,i]; φL = work.ULp[2,i]; yL = work.ULp[3,i]
            yTR = work.URp[1,i]; φR = work.URp[2,i]; yR = work.URp[3,i]

            nurL_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, i])   : 0.0)
            nurR_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, i+1]) : 0.0)

            PiL_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, i])    : 0.0)
            PiR_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, i+1])  : 0.0)

            piRL_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, i])   : 0.0)
            piRR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, i+1]) : 0.0)

            piEtaL_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, i])   : 0.0)
            piEtaR_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, i+1]) : 0.0)

            okL, primL = prim_to_cons_col_ideal_phi_diff_visc!(work.ULc, i,
                                                              yTL, φL, yL,
                                                              nurL_phys, PiL_phys, piRL_phys, piEtaL_phys,
                                                              τ, eos, L)
            okR, primR = prim_to_cons_col_ideal_phi_diff_visc!(work.URc, i,
                                                              yTR, φR, yR,
                                                              nurR_phys, PiR_phys, piRR_phys, piEtaR_phys,
                                                              τ, eos, L)

            if okL && okR &&
               work.ULc[L.iDtau,i] ≥ 0.0 && work.URc[L.iDtau,i] ≥ 0.0 &&
               work.ULc[L.iE,i]    ≥ Emin && work.URc[L.iE,i]    ≥ Emin
                hlle_flux_col_cons!(work.Fh, i, work.ULc, work.URc, primL, primR, eos, τ, model, tmpFL, tmpFR)
                continue
            end
        end

        TL = exp(work.yT[i]);    μL = work.mu[i];    urL = sinh(work.y[i])
        TR = exp(work.yT[i+1]);  μR = work.mu[i+1];  urR = sinh(work.y[i+1])

        nurL_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, i])   : 0.0)
        nurR_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur, i+1]) : 0.0)
        PiL_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, i])    : 0.0)
        PiR_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi, i+1])  : 0.0)
        piRL_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, i])   : 0.0)
        piRR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR, i+1]) : 0.0)
        piEtaL_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, i])   : 0.0)
        piEtaR_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta, i+1]) : 0.0)

        primL = PrimIdealVisc(TL, μL, urL, work.n[i],   work.e[i],   work.P[i],
                              nurL_phys, PiL_phys, piRL_phys, piEtaL_phys, true)
        primR = PrimIdealVisc(TR, μR, urR, work.n[i+1], work.e[i+1], work.P[i+1],
                              nurR_phys, PiR_phys, piRR_phys, piEtaR_phys, true)

        hlle_flux_lr_U!(work.Fh, i, U, i, i+1, primL, primR, eos, τ, model, tmpFL, tmpFR)
    end

    need_theta = false
    if L.hasNur   && model.advect_nur; need_theta = true; end
    if L.hasPi    && model.advect_Pi;  need_theta = true; end
    if (L.hasPiR || L.hasPiEta) && model.advect_pi; need_theta = true; end

    if need_theta
        compute_theta!(work, grid, τ)
    else
        fill!(work.theta, 0.0)
    end

    fill!(work.S, 0.0)
    Threads.@threads for i in (ng+1):(Ntot-ng)
        Pi_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi,i])    : 0.0)
        piR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

        source_cell_fast_col!(work.S, i, U[L.iSr,i], U[L.iE,i],
                              work.P[i], Pi_phys, piR_phys, piEta_phys,
                              grid.rC[i], τ, model)

        if need_theta
            divrv = work.theta[i] - 1 / max(τ, 1e-50)

            if L.hasNur && model.advect_nur
                work.S[L.iNur, i] += U[L.iNur, i] * divrv
            end

            if L.hasPi && model.advect_Pi
                work.S[L.iPi, i] += U[L.iPi, i] * divrv
            end

            if model.advect_pi
                if L.hasPiR
                    work.S[L.iPiR, i] += U[L.iPiR, i] * divrv
                end
                if L.hasPiEta
                    work.S[L.iPiEta, i] += U[L.iPiEta, i] * divrv
                end
            end
        end
    end

    fill!(dU, 0.0)

    @inbounds for a in 1:Nvars
        dU[a,i0] = -2 * work.Fh[a,i0] / grid.dr + work.S[a,i0]
    end

    Threads.@threads for i in (i0+1):(Ntot-ng)
        rC  = grid.rC[i]
        rRp = grid.rF[i+1]
        rRm = grid.rF[i]
        invr = 1 / max(rC, 1e-50)

        @inbounds for a in 1:Nvars
            div = (rRp*work.Fh[a,i] - rRm*work.Fh[a,i-1]) / grid.dr
            dU[a,i] = -div * invr + work.S[a,i]
        end
    end

    return true
end

# ------------------------------------------------------------
# dt (CFL + diffusion cap for charge diffusion only)
# ------------------------------------------------------------


"""
    compute_dt_from_work(work, grid, τ, model; CFL, CFLτ)

Fast dt estimate using cached primitive fields in `work`. **Requires** work caches to be
valid for current `U` (they are updated at the end of `step_ssprk2!` via `relax_dissipative!`).
We call `prime_work_from_U!` once at τ0 to seed these caches.
"""
function compute_dt_from_work(work::Work1D, grid, τ, model; CFL::Float64=0.2, CFLτ::Float64=0.05)
    ng = grid.nghost
    amax = 1e-30
    kmax = 0.0

    @inbounds for i in (ng+1):(length(work.yT)-ng)
        work.ok[i] || continue
        T  = exp(work.yT[i])
        μ  = work.mu[i]
        ur = sinh(work.y[i])
        λm, λp = wavespeeds_from_prim(T, μ, ur, model.eos)
        a = max(abs(λm), abs(λp))

        if model.enable_diff
            uτ = sqrt(1 + ur^2)
            κ  = diff_kappa(T, μ, work.n[i], model)
            kmax = max(kmax, κ * (uτ^2))
            a = max(a, 0.999999)
        end
        amax = max(amax, a)
    end

    dt = min(CFL * grid.dr/(amax + TINY), CFLτ * τ)

    if model.enable_diff && kmax > 0
        dt = min(dt, model.diff_dt_coeff * grid.dr^2/(kmax + TINY))
    end
    return dt
end





# ------------------------------------------------------------
# SSPRK2 + MOOD (your implementation; unchanged)
# ------------------------------------------------------------
function step_ssprk2!(U, grid, τ, Δτ, model::IdealDiffViscModel, work::Work1D;
                      Emin::Float64=E_FLOOR,
                      χ::Float64=χ_SrE,
                      max_stage_retries::Int=3,
                      max_dt_halvings::Int=24,
                      diag::Union{Nothing,DiagCounters}=nothing)

    bad = work.bad
    bad_tmp = work.bad_tmp
    Δ = Δτ

    last_fail_stage = 0
    last_fail_tau   = NaN
    last_fail_Δ     = NaN
    last_fail_i     = 0
    last_fail_why   = :none
    last_fail_U     = nothing

    for halv in 1:max_dt_halvings
        diag === nothing || diag_add!(diag; halvings = (halv > 1 ? 1 : 0))

        local ok1 = false
        local force_mask::Union{Nothing,BitVector} = nothing

        for retry in 0:max_stage_retries
            rhs!(work.k, U, grid, τ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)

            @. work.U1 = U + Δ*work.k
            apply_bc!(work.U1, grid, τ+Δ, model)
            enforce_floors!(work.U1, grid, τ+Δ, model; Emin=Emin, diag=diag)
            enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, mask=nothing, diag=diag)

            relax_dissipative!(work.U1, grid, τ+Δ, Δ, model, work)
            apply_bc!(work.U1, grid, τ+Δ, model)
            enforce_floors!(work.U1, grid, τ+Δ, model; Emin=Emin, diag=diag)
            enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, mask=nothing, diag=diag)

            if SANITIZE_SCOPE == :global
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=nothing, diag=diag)
            end

            if !DO_MOOD || !any_bad(work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
                ok1 = true
                break
            end

            last_fail_stage = 1
            last_fail_tau   = τ + Δ
            last_fail_Δ     = Δ
            last_fail_i, last_fail_why = _find_first_bad(work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)

            mark_bad!(bad, work.U1, grid, τ+Δ, model; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood1=count(bad))

            if WRITE_BADMASK
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage1_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, mask=bad, diag=diag)
            end

            force_mask = bad
        end

        if !ok1
            last_fail_U = copy(work.U1)
            Δ *= 0.5
            continue
        end

        local ok2 = false
        force_mask = nothing
        for retry in 0:max_stage_retries
            rhs!(work.k, work.U1, grid, τ+Δ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)

            @. work.U2 = 0.5*U + 0.5*(work.U1 + Δ*work.k)
            apply_bc!(work.U2, grid, τ+Δ, model)
            enforce_floors!(work.U2, grid, τ+Δ, model; Emin=Emin, diag=diag)
            enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, mask=nothing, diag=diag)

            relax_dissipative!(work.U2, grid, τ+Δ, Δ, model, work)
            apply_bc!(work.U2, grid, τ+Δ, model)
            enforce_floors!(work.U2, grid, τ+Δ, model; Emin=Emin, diag=diag)
            enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, mask=nothing, diag=diag)

            if SANITIZE_SCOPE == :global
                sanitize_state!(work.U2, grid, τ+Δ, model; Emin=Emin, mask=nothing, diag=diag)
            end

            if !DO_MOOD || !any_bad(work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
                ok2 = true
                break
            end

            last_fail_stage = 2
            last_fail_tau   = τ + Δ
            last_fail_Δ     = Δ
            last_fail_i, last_fail_why = _find_first_bad(work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ)

            mark_bad!(bad, work.U2, grid, τ+Δ, model; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood2=count(bad))

            if WRITE_BADMASK
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage2_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U2, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, mask=bad, diag=diag)
            end

            force_mask = bad
        end

        if !ok2
            last_fail_U = copy(work.U2)
            Δ *= 0.5
            continue
        end

        U .= work.U2
        apply_bc!(U, grid, τ+Δ, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, mask=nothing, diag=diag)

        relax_dissipative!(U, grid, τ+Δ, Δ, model, work)
        apply_bc!(U, grid, τ+Δ, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, mask=nothing, diag=diag)

        if SANITIZE_SCOPE == :global
            sanitize_state!(U, grid, τ+Δ, model; Emin=Emin, mask=nothing, diag=diag)
        end

        return Δ
    end

    # Hard failure: dump a small window around the last known bad cell/state.
    Udbg  = (last_fail_U === nothing ? work.U2 : last_fail_U)
    τdbg  = (isfinite(last_fail_tau) ? last_fail_tau : (τ + Δ))
    Δdbg  = (isfinite(last_fail_Δ) ? last_fail_Δ : Δ)
    ibad  = last_fail_i
    why   = last_fail_why

    if ibad == 0
        ibad, why = _find_first_bad(Udbg, grid, τdbg, model, work; Emin=Emin, χ=χ)
    end

    if ibad != 0
        f = joinpath("debug_failures", @sprintf("failure_tau_%06.3f_stage_%d_i_%d.csv", τdbg, last_fail_stage, ibad))
        _dump_failure_window_csv(f, Udbg, grid, τdbg, model, work, ibad; radius=5)
        @error "dt-halving failure" τ=τdbg Δ=Δdbg stage=last_fail_stage i=ibad r=grid.rC[ibad] reason=why dump=f
    else
        @error "dt-halving failure" τ=τdbg Δ=Δdbg stage=last_fail_stage reason=:no_bad_cell_found
    end
    error("Time step failed: could not find admissible update even after dt halving.")
end

# ------------------------------------------------------------
# Conserved charge integral
# ------------------------------------------------------------
function charge_integral_Dtau(U, grid, model::IdealDiffViscModel)
    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng
    Q = 0.0
    @inbounds for i in i0:iL
        rL = max(grid.rF[i],   0.0)
        rR = max(grid.rF[i+1], 0.0)
        Ai = π * (rR*rR - rL*rL)
        Q += U[L.iDtau, i] * Ai
    end
    return Q
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function initialize!(U, grid, τ0, model::IdealDiffViscModel;
                     init_csv::Union{Nothing,String}=nothing,
                     fugacity_kind::Symbol = :alpha,
                     taper_width::Float64 = 1.0,
                     Emin::Float64=E_FLOOR,
                     χ::Float64=χ_SrE)

    eos  = model.eos
    L    = layout(model)
    ng   = grid.nghost
    Ntot = size(U,2)
    fill!(U, 0.0)

    itpT = nothing
    itpF = nothing
    if init_csv !== nothing
        itpT, itpF, rmax_data = load_initial_interpolants(init_csv; fugacity_kind=fugacity_kind, taper_width=taper_width)
        @info "Loaded CSV IC (vacuum extrap + taper)" init_csv=init_csv fugacity_kind=fugacity_kind rmax_data=rmax_data grid_rmax=grid.rmax taper_width=taper_width
    end

    @inbounds for i in (ng+1):(Ntot-ng)
        r = grid.rC[i]

        if init_csv === nothing
            T  = 0.35 + 0.25*exp(-(r/3.0)^2)
            μ  = 0.10*exp(-(r/5.0)^2)
        else
            T  = max(itpT(r), T_MIN)
            f0 = itpF(r)
            if fugacity_kind == :alpha
                μ = f0 * T
            else
                λ = max(f0, TINY)
                μ = log(λ) * T
            end
        end

        ur = 0.0
        uτ = 1.0
        v  = 0.0

        nur0_phys   = 0.0
        Pi0_phys    = 0.0
        piR0_phys   = 0.0
        piEta0_phys = 0.0

        P, n, e = eos_Pne(T, μ, eos)

        if !(isfinite(T) && isfinite(μ) && isfinite(P) && isfinite(n) && isfinite(e)) || e < Emin
            T = max(T_MIN, T)
            μ = 0.0
            P, n, e = eos_Pne(T, μ, eos)
            n = 0.0
            e = max(Emin, e)
        end

        w    = e + P
        Ptot = P + Pi0_phys
        weff = w + Pi0_phys

        D  = n*uτ + v*nur0_phys

        U[L.iDtau, i] = τ0 * D
        U[L.iSr,  i]  = weff * (uτ*ur)
        U[L.iE,   i]  = weff * (uτ^2) - Ptot

        if L.hasNur
            U[L.iNur, i] = stored_from_phys(nur0_phys)
        end
        if L.hasPi
            U[L.iPi, i] = stored_from_phys(Pi0_phys)
        end
        if L.hasPiR
            U[L.iPiR, i] = stored_from_phys(piR0_phys)
        end
        if L.hasPiEta
            U[L.iPiEta, i] = stored_from_phys(piEta0_phys)
        end
    end

    apply_bc!(U, grid, τ0, model)
    enforce_floors!(U, grid, τ0, model; Emin=Emin, diag=nothing)
    enforce_Sr_energy_constraint!(U, grid, model; χ=χ, mask=nothing, diag=nothing)
    return nothing
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
function run_sim_ideal_diff_visc(; outdir::String,
                                Nr::Int=300, rmax::Float64=20.0,
                                τ0::Float64=0.6, τfinal::Float64=3.0,
                                CFL::Float64=0.2, CFLτ::Float64=0.05,
                                dump_dt::Float64=0.2,
                                Emin::Float64=E_FLOOR,
                                χ::Float64=χ_SrE,
                                init_csv::Union{Nothing,String}=nothing,
                                fugacity_kind::Symbol = :alpha,
                                taper_width::Float64 = 1.0,
                                log_every::Int=50,
                                log_corrections_every::Int=50,
                                # charge diffusion
                                enable_diff::Bool=true,
                                kappa_coeff::Float64=0.1,
                                tauN_coeff::Float64=1.0,
                                deltaN_factor::Float64=0.0,
                                diff_dt_coeff::Float64=0.02,
                                nur_clip_factor::Float64=0.5,
                                alpha_filter_eps::Float64=0.0,
                                nur_filter_eps::Float64=0.0,
                                alpha_smooth_len::Float64=0.1,
                                nur_smooth_len::Float64=0.1,
                                do_axis_project_nur::Bool=true,
                                axis_project_nfit::Int=16,
                                advect_nur::Bool=true,
                                relax_advect_nur::Bool=false,
                                # viscosity
                                enable_shear::Bool=false,
                                enable_bulk::Bool=false,
                                eta_over_s::Float64=0.08,
                                zeta_over_s::Float64=0.00,
                                tauPi_coeff::Float64=1.0,
                                tauShear_coeff::Float64=1.0,
                                deltaPi_factor::Float64=0.0,
                                deltaShear_factor::Float64=0.0,
                                visc_filter_eps::Float64=0.0,
                                visc_smooth_len::Float64=0.3,
                                Pi_clip_factor::Float64=0.5,
                                pi_clip_factor::Float64=0.5,
                                advect_Pi::Bool=false,
                                advect_pi::Bool=false,
                                relax_advect_Pi::Bool=false,
                                relax_advect_pi::Bool=false,

                                # EOS choice
                                eos = ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0))

    grid = make_grid(Nr; rmax=rmax, nghost=3)

    layout = StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    shear_model = QGPViscosity(eta_over_s, tauShear_coeff)
    bulk_model  = (zeta_over_s == 0.0) ? ZeroBulkViscosity() : SimpleBulkViscosity(zeta_over_s, tauPi_coeff)

    _check_transport_flags!(;
        advect_nur=advect_nur, relax_advect_nur=relax_advect_nur,
        advect_Pi=advect_Pi,   relax_advect_Pi=relax_advect_Pi,
        advect_pi=advect_pi,   relax_advect_pi=relax_advect_pi
    )

    model  = IdealDiffViscModel(eos, layout, IdealPrimRec(),
        # charge diffusion
        enable_diff, kappa_coeff, tauN_coeff, deltaN_factor,
        diff_dt_coeff, nur_clip_factor,
        alpha_filter_eps, nur_filter_eps, alpha_smooth_len, nur_smooth_len,
        do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur,
        # viscosity
        enable_shear, enable_bulk, shear_model, bulk_model,
        deltaPi_factor, deltaShear_factor,
        visc_filter_eps, visc_smooth_len,
        Pi_clip_factor, pi_clip_factor,
        advect_Pi, advect_pi,
        # NEW
        relax_advect_Pi, relax_advect_pi
    )

    U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
    initialize!(U, grid, τ0, model; init_csv=init_csv, fugacity_kind=fugacity_kind, taper_width=taper_width, Emin=Emin, χ=χ)
    work = make_work(U)
    diag = DiagCounters()

    τ = τ0
    it = 0
    next_dump = τ0

    # Seed work caches once so dt-from-work is valid at the first step
    prime_work_from_U!(work, U, grid, τ, model)

    @info "Start IDEAL+DIFF+VISC" threads=Threads.nthreads() outdir=outdir Nr=Nr dr=grid.dr τ0=τ0 τfinal=τfinal rmax=rmax CFL=CFL CFLτ=CFLτ enable_diff=enable_diff enable_shear=enable_shear enable_bulk=enable_bulk eta_over_s=eta_over_s zeta_over_s=zeta_over_s DISS_SIGN=DISS_SIGN relax_advect_nur=relax_advect_nur

    mkpath(outdir)
    clear_dir!(outdir)
    write_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)), U, grid, τ, model)

    while τ < τfinal - 1e-12
        diag_reset_last!(diag)

        # IMPORTANT: honor run-time CFL values
        Δτ = compute_dt_from_work(work, grid, τ, model; CFL=CFL, CFLτ=CFLτ)
        if τ + Δτ > τfinal
            Δτ = τfinal - τ
        end

        Δused = step_ssprk2!(U, grid, τ, Δτ, model, work; Emin=Emin, χ=χ, diag=diag)
        τ += Δused
        it += 1

        if diag.last_prim_fail_cells > 0 && diag.last_prim_fail_i != 0
            i = diag.last_prim_fail_i
            r = (1 <= i <= length(grid.rC)) ? grid.rC[i] : NaN
            @warn "primitive recovery failed" it=it τ=τ i=i r=r Dtau=diag.last_prim_fail_Dtau Sr=diag.last_prim_fail_Sr E=diag.last_prim_fail_E nur_stored=diag.last_prim_fail_nur_stored Pi_stored=diag.last_prim_fail_Pi_stored piR_stored=diag.last_prim_fail_piR_stored piEta_stored=diag.last_prim_fail_piEta_stored
        end

        if (it % log_corrections_every) == 0
            @debug "corrections (last window)" it=it τ=τ Δτ=Δused primfail=diag.last_prim_fail_cells floorE=diag.last_floor_E_cells floorD=diag.last_floor_D_cells nanfix=diag.last_nanfix_cells SrScaled=diag.last_Sr_scaled_cells mood1=diag.last_mood_stage1_bad mood2=diag.last_mood_stage2_bad dtHalvings=diag.last_stage_dt_halvings
        end

        if τ >= next_dump - 1e-12
            fname = joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ))
            write_snapshot_csv(fname, U, grid, τ, model)

            @info "dump" τ=τ file=fname it=it Q_Dtau=charge_integral_Dtau(U, grid, model) primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells SrScaled_total=diag.Sr_scaled_cells mood1_total=diag.mood_stage1_bad mood2_total=diag.mood_stage2_bad
            next_dump += dump_dt
        end

        if (it % log_every) == 0
            @info "progress" τ=τ Δτ=Δused it=it Q_Dtau=charge_integral_Dtau(U, grid, model) primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells SrScaled_total=diag.Sr_scaled_cells
        end
    end

    @info "Done IDEAL+DIFF+VISC" τ=τ it=it outdir=outdir Q_Dtau=charge_integral_Dtau(U, grid, model) primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells floorD_total=diag.floor_D_cells nanfix_total=diag.nanfix_cells SrScaled_total=diag.Sr_scaled_cells mood1_total=diag.mood_stage1_bad mood2_total=diag.mood_stage2_bad dt_halvings_total=diag.stage_dt_halvings
    return nothing
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
function main()
    setup_logger!(level=Logging.Info)

    rmax    = 15.0
    Nr = ceil(Int, rmax / 25 * 500)
    τ0      = 0.4
    τfinal  = 15.
    dump_dt = 0.1
    Ntaper_target = 40
    taper_width = Ntaper_target * rmax/Nr

    init_csv      = "data/initial_profiles.csv"
    fugacity_kind = :alpha

    eos = RunningConformalHQEOS()
    #eos = ConformalHQEOS()

    run_sim_ideal_diff_visc(outdir="snapshots/snapshots_ideal_diff_visc_phi",
                            Nr=Nr, rmax=rmax, τ0=τ0, τfinal=τfinal,
                            CFL=0.1, CFLτ=0.1,
                            dump_dt=dump_dt,

                            #χ=0.99,

                            init_csv=init_csv, fugacity_kind=fugacity_kind,
                            taper_width=taper_width,
                            log_every=50,
                            log_corrections_every=50,

                            enable_diff=true,
                            kappa_coeff=0.1, tauN_coeff=1.0, deltaN_factor=0.0,
                            diff_dt_coeff=0.05,
                            nur_clip_factor=0.0,
                            alpha_filter_eps=0.0,
                            nur_filter_eps=0.0,
                            alpha_smooth_len=0.0,
                            nur_smooth_len=0.0,
                            do_axis_project_nur=true,
                            axis_project_nfit=15,
                            advect_nur=false,
                            relax_advect_nur=true,

                            enable_shear=true,
                            enable_bulk=true,
                            eta_over_s=0.1,
                            zeta_over_s=0.083,
                            tauPi_coeff=15.0,
                            tauShear_coeff=0.2,
                            deltaPi_factor=0.0,
                            deltaShear_factor=0.0,
                            visc_filter_eps=0.0,
                            visc_smooth_len=0.0,
                            Pi_clip_factor=0.0,
                            pi_clip_factor=0.0,
                            advect_Pi=false,
                            advect_pi=false,
                            relax_advect_Pi=true,
                            relax_advect_pi=true,

                            eos = eos)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module hydro