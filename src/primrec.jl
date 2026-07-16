# ==============================================================================
# src/primrec.jl
#
# Primitive recovery loop (Newton-Raphson) and helpers.
# ==============================================================================

# ------------------------------------------------------------
# Primitive recovery work (Newton unknowns: yT, φ, y)
# ------------------------------------------------------------

mutable struct PrimRecWork
    x::Vector{Float64}
    F::Vector{Float64}
    Fp::Vector{Float64}
    J::Matrix{Float64}
    xtrial::Vector{Float64}
    Fres::Vector{Float64}
    A::Matrix{Float64}
    δ::Vector{Float64}

    last_reason::PrimRecReason
    last_iters::Int
    last_resnorm::Float64
end
PrimRecWork() = PrimRecWork(
    zeros(3), zeros(3), zeros(3), zeros(3,3), zeros(3), zeros(3), zeros(3,4), zeros(3),
    PRR_UNSET, 0, NaN,
)

@inline function _allfinite3(v::Vector{Float64})
    @inbounds return isfinite(v[1]) & isfinite(v[2]) & isfinite(v[3])
end

@inline function _clamp_newton_x!(x::Vector{Float64}, yT_lo::Float64, yT_hi::Float64)
    @inbounds begin
        x[1] = clamp(x[1], yT_lo, yT_hi)
        x[2] = clamp(x[2], -PHI_CAP, PHI_CAP)
        x[3] = clamp(x[3], -Y_CAP, Y_CAP)
    end
    return nothing
end

@inline function _μ_guess_from_n(eos, T::Float64, n_target::Float64)
    Tuse = max(T, T_MIN)
    m = hq_mass(eos)
    nt = max(n_target, 0.0)
    if nt == 0.0
        return m
    end

    if eos isa ConformalHQEOS || eos isa RunningConformalHQEOS
        g = (eos isa ConformalHQEOS) ? (eos::ConformalHQEOS).g_hq : (eos::RunningConformalHQEOS).g_hq
        x = m / Tuse
        if !(isfinite(x)) || x <= 0.0
            return m
        end
        A   = g * m^2 / (2π^2)
        K2x = SpecialFunctions.besselkx(2, x)
        F   = A * Tuse * max(K2x, 0.0)
        if !(isfinite(F)) || F <= 0.0
            return m
        end
        μ = m + Tuse * log(nt / (F + TINY))
        return m + Tuse * clamp((μ - m) / Tuse, -PHI_CAP, PHI_CAP)
    elseif eos isa TabulatedHQEOS
        αlo = (eos::TabulatedHQEOS).αmin
        αhi = (eos::TabulatedHQEOS).αmax

        nlo = eos_Pne(Tuse, αlo*Tuse, eos)[2]
        nhi = eos_Pne(Tuse, αhi*Tuse, eos)[2]
        if nt <= nlo
            return αlo * Tuse
        elseif nt >= nhi
            return αhi * Tuse
        end

        lo = αlo
        hi = αhi
        for _ in 1:60
            mid = 0.5*(lo + hi)
            nmid = eos_Pne(Tuse, mid*Tuse, eos)[2]
            if nmid < nt
                lo = mid
            else
                hi = mid
            end
        end
        return 0.5*(lo + hi) * Tuse
    else
        return m
    end
end

@inline function evalF_ideal_phi_diff_visc!(F::Vector{Float64}, x::Vector{Float64},
                                           D::Float64, Sr::Float64, E::Float64, nur_phys::Float64,
                                           Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                           r::Float64, τ::Float64,
                                           eos)
    yT = x[1]; φ = x[2]; y = x[3]
    if !(isfinite3(yT, φ, y))
        F .= Inf
        return
    end

    T  = exp(yT)
    μ  = hq_mass(eos) + T*φ
    ur = sinh(y)
    uτ = cosh(y)
    v  = safe_div(ur, uτ)

    #T < 1e-100 && @warn "evalF_ideal_phi_diff_visc! very small T=$(T),yT=$(yT)" 
    #T > 1e100  && @warn "evalF_ideal_phi_diff_visc! very large T=$(T),yT=$(yT)" 

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite3(P, n, e))
        F .= Inf
        return
    end

    w    = e + P
    Ptot = P + Pi_phys
    weff = w + Pi_phys

    Π = shear_tensor_contravariant(ur, uτ, r, τ, piR_phys, piEta_phys)

    F[1] = n*uτ + v*nur_phys - D
    F[2] = weff * (uτ*ur) + Π.tr - Sr
    F[3] = weff*uτ^2 - Ptot + Π.tt - E
    return
end

# ------------------------------------------------------------
# Charge-less EOS handling
#
# If the EOS has no μ dependence (effectively n(T,μ) ≡ 0), the full (yT,φ,y)
# Newton system becomes singular because F[1] does not constrain φ.
# For such cases we solve only (yT,y) with φ fixed to 0 (μ = hq_mass(eos)).
# ------------------------------------------------------------

@inline eos_has_charge(eos) = true
@inline eos_has_charge(eos::ConformalHQEOS) = (eos.g_hq > 0.0)
@inline eos_has_charge(eos::RunningConformalHQEOS) = (eos.g_hq > 0.0)
@inline eos_has_charge(::TabulatedHQEOS) = true

@inline function _is_baryonless_conformal_eos(eos)
    return false
end

@inline function _is_baryonless_conformal_eos(eos::ConformalHQEOS)
    # With g_hq==0 the EOS becomes conformal, μ-independent, and baryonless.
    return eos.g_hq <= 0.0
end

"""Exact inversion for ideal *conformal baryonless* hydro from (Sr, E) -> (T, ur).

Assumes:
- EOS is conformal with e=3P and μ-independent
- no charge diffusion (n=0, nur=0)
- dissipatives are negligible (Pi=piR=piEta=0)

This is used as a robust primitive recovery path for Gubser validation.
"""
@inline function cons_to_prim_conformal_baryonless_ideal(
    Sr::Float64,
    E::Float64,
    r::Float64,
    τ::Float64,
    eos,
)
    # Guard vacuum-ish
    if !isfinite(E) || E <= E_VAC
        T  = T_MIN
        μ  = 0.0
        ur = 0.0
        P, _, _ = eos_Pne(T, μ, eos)
        return (T, μ, ur, 0.0, max(E, 0.0), P, true)
    end

    a = Sr / (E + TINY)
    if !isfinite(a)
        a = 0.0
    end

    # Solve a*v^2 - 4v + 3a = 0 with the root that gives v≈3a/4 for small |a|.
    v = 0.0
    if abs(a) > 1e-14
        disc = 16.0 - 12.0*a*a
        disc = disc < 0.0 ? 0.0 : disc
        v = (4.0 - sqrt(disc)) / (2.0*a)
    end
    v = clamp(v, -0.999999999, 0.999999999)

    uτ = 1.0 / sqrt(1.0 - v*v)
    ur = uτ * v

    # For conformal ideal: E = P*(3+v^2)/(1-v^2)  => P = E*(1-v^2)/(3+v^2)
    P = E * (1.0 - v*v) / (3.0 + v*v + TINY)
    P = max(P, 0.0)

    # Invert EOS: P = a_SB * T^4 * fmGeV3
    T = (P / (a_SB(eos) * fmGeV3 + TINY))^(0.25)
    T = max(T, T_MIN)

    μ = 0.0
    P2, n2, e2 = eos_Pne(T, μ, eos)
    if !(isfinite3(P2, n2, e2))
        return (T, μ, ur, 0.0, max(E, 0.0), max(P, 0.0), false)
    end

    return (T, μ, ur, 0.0, max(e2, 0.0), max(P2, 0.0), true)
end

@inline function _evalF_ideal_phi0!(F2::Base.RefValue{Float64}, F3::Base.RefValue{Float64},
                                   yT::Float64, y::Float64,
                                   Sr::Float64, E::Float64,
                                   Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                   r::Float64, τ::Float64,
                                   eos)
    T  = exp(yT)
    μ  = hq_mass(eos)  # φ = 0
    ur = sinh(y)
    uτ = cosh(y)

    P, _, e = eos_Pne(T, μ, eos)
    if !(isfinite3(P, e, uτ))
        F2[] = Inf
        F3[] = Inf
        return
    end

    w    = e + P
    Ptot = P + Pi_phys
    weff = w + Pi_phys

    Π = shear_tensor_contravariant(ur, uτ, r, τ, piR_phys, piEta_phys)

    F2[] = weff * (uτ*ur) + Π.tr - Sr
    F3[] = weff * (uτ^2) - Ptot + Π.tt - E
    return
end

function _cons_to_prim_ideal_phi0_y0_nocharge!(w::PrimRecWork,
                                               E::Float64,
                                               Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                               r::Float64, τ::Float64,
                                               eos;
                                               maxit::Int=80, tol_res::Float64=1e-14)
    yT_lo = log(T_MIN)
    yT_hi = log(T_SOLVE_MAX)
    F2 = Ref{Float64}(NaN)
    F3 = Ref{Float64}(NaN)

    eval_y0!(yT) = _evalF_ideal_phi0!(F2, F3, yT, 0.0, 0.0, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
    tolF = tol_res * (1 + abs(E))

    y_prev = yT_lo
    eval_y0!(y_prev)
    if !(isfinite(F2[]) && isfinite(F3[]))
        w.last_reason = PRR_NONFINITE_INITIAL
        w.last_iters = 0
        w.last_resnorm = NaN
        T = T_MIN
        μ = hq_mass(eos)
        P, _, e = eos_Pne(T, μ, eos)
        return (T, μ, 0.0, 0.0, max(e, 0.0), P, false)
    end

    f_prev = F3[]
    best_y = y_prev
    best_f = f_prev
    bracket_found = false
    y_left = y_prev
    y_right = y_prev
    f_left = f_prev
    f_right = f_prev

    for k in 1:64
        y_cur = yT_lo + (yT_hi - yT_lo) * (k / 64)
        eval_y0!(y_cur)
        if !(isfinite(F2[]) && isfinite(F3[]))
            continue
        end
        if abs(F3[]) < abs(best_f)
            best_y = y_cur
            best_f = F3[]
        end
        if signbit(f_prev) != signbit(F3[])
            y_left = y_prev
            y_right = y_cur
            f_left = f_prev
            f_right = F3[]
            bracket_found = true
            break
        end
        y_prev = y_cur
        f_prev = F3[]
    end

    y_sol = best_y
    f_sol = best_f
    if bracket_found
        for it in 1:maxit
            y_mid = 0.5 * (y_left + y_right)
            eval_y0!(y_mid)
            if !(isfinite(F2[]) && isfinite(F3[]))
                break
            end
            y_sol = y_mid
            f_sol = F3[]
            if abs(f_sol) < tolF
                w.last_reason = PRR_CONVERGED
                w.last_iters = it
                w.last_resnorm = abs(f_sol)
                T = clamp(exp(y_sol), T_MIN, T_SOLVE_MAX)
                μ = hq_mass(eos)
                P, _, e = eos_Pne(T, μ, eos)
                return (T, μ, 0.0, 0.0, max(e, 0.0), P, true)
            end
            if signbit(f_left) != signbit(f_sol)
                y_right = y_mid
                f_right = f_sol
            else
                y_left = y_mid
                f_left = f_sol
            end
        end
    end

    w.last_reason = abs(f_sol) < tolF ? PRR_CONVERGED : PRR_RESIDUAL_TOO_LARGE
    w.last_iters = bracket_found ? maxit : 0
    w.last_resnorm = abs(f_sol)
    T = clamp(exp(y_sol), T_MIN, T_SOLVE_MAX)
    μ = hq_mass(eos)
    P, _, e = eos_Pne(T, μ, eos)
    return (T, μ, 0.0, 0.0, max(e, 0.0), P, abs(f_sol) < tolF)
end

function cons_to_prim_ideal_phi0_nocharge!(w::PrimRecWork,
                                          Sr::Float64, E::Float64,
                                          Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                          r::Float64, τ::Float64,
                                          eos;
                                          yT0::Float64=NaN, y0::Float64=NaN,
                                          maxit::Int=80, tol::Float64=1e-15, tol_res::Float64=1e-14)

    yT_lo = log(T_MIN)
    yT_hi = log(T_SOLVE_MAX)

    w.last_reason = PRR_UNSET
    w.last_iters = 0
    w.last_resnorm = NaN

    # Initial guess
    if isfinite(yT0) && isfinite(y0)
        yT = clamp(yT0, yT_lo, yT_hi)
        y  = clamp(y0,  -Y_CAP, Y_CAP)
    else
        T0 = clamp(posden(E)^(0.25), T_MIN, T_SOLVE_MAX)
        yT = log(T0)
        P0, _, _ = eos_Pne(T0, hq_mass(eos), eos)
        denom = posden(E + P0 + abs(Pi_phys) + abs(piR_phys) + abs(piEta_phys))
        v0 = clamp(safe_div(Sr, denom), -0.9999, 0.9999)
        y  = atanh(v0)
    end

    # Newton in (yT,y)
    F2 = Ref{Float64}(NaN)
    F3 = Ref{Float64}(NaN)

    # Scratch Refs (avoid per-iteration allocations in the FD Jacobian)
    F2p = Ref{Float64}(NaN); F3p = Ref{Float64}(NaN)
    F2m = Ref{Float64}(NaN); F3m = Ref{Float64}(NaN)
    F2f = Ref{Float64}(NaN); F3f = Ref{Float64}(NaN)

    _evalF_ideal_phi0!(F2, F3, yT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
    if !(isfinite(F2[]) && isfinite(F3[]))
        w.last_reason = PRR_NONFINITE_INITIAL
        T = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
        μ = hq_mass(eos)
        P, _, _ = eos_Pne(T, μ, eos)
        return (T, μ, 0.0, 0.0, max(E, 0.0), P, false)
    end

    if abs(y) <= 1e-14 && abs(F2[]) <= tol_res * (1 + abs(Sr) + abs(E))
        return _cons_to_prim_ideal_phi0_y0_nocharge!(w, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos;
                                                     maxit=maxit, tol_res=tol_res)
    end

    if abs(y) <= 1e-12
        y = copysign(1e-8, abs(Sr) > 0.0 ? Sr : 1.0)
        _evalF_ideal_phi0!(F2, F3, yT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        if !(isfinite(F2[]) && isfinite(F3[]))
            w.last_reason = PRR_NONFINITE_INITIAL
            T = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
            μ = hq_mass(eos)
            P, _, _ = eos_Pne(T, μ, eos)
            return (T, μ, 0.0, 0.0, max(E, 0.0), P, false)
        end
    end

    converged = false
    last_nrm = NaN
    for it in 1:maxit
        nrm = sqrt(F2[]^2 + F3[]^2)
        last_nrm = nrm
        if nrm < tol || nrm < tol_res * (1 + abs(Sr) + abs(E))
            converged = true
            w.last_reason = PRR_CONVERGED
            w.last_iters = it - 1
            break
        end

        # finite-difference Jacobian in (yT, y)
        hT = 1e-6
        hy = 1e-6

        _evalF_ideal_phi0!(F2p, F3p, yT + hT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        _evalF_ideal_phi0!(F2m, F3m, yT - hT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        dF2_dyT = (F2p[] - F2m[]) / (2hT)
        dF3_dyT = (F3p[] - F3m[]) / (2hT)

        _evalF_ideal_phi0!(F2p, F3p, yT, y + hy, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        _evalF_ideal_phi0!(F2m, F3m, yT, y - hy, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        dF2_dy  = (F2p[] - F2m[]) / (2hy)
        dF3_dy  = (F3p[] - F3m[]) / (2hy)

        det = dF2_dyT*dF3_dy - dF2_dy*dF3_dyT
        if !(isfinite(det)) || abs(det) <= 1e-30
            if abs(y) <= 1e-8
                y = copysign(1e-6, abs(Sr) > 0.0 ? Sr : 1.0)
                _evalF_ideal_phi0!(F2, F3, yT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
                if isfinite(F2[]) && isfinite(F3[])
                    continue
                end
            end
            w.last_reason = PRR_LINSOLVE_FAILED
            break
        end

        # Solve 2x2 system for update δ
        b1 = -F2[]
        b2 = -F3[]
        δyT = ( b1*dF3_dy - b2*dF2_dy ) / det
        δy  = ( dF2_dyT*b2 - dF3_dyT*b1 ) / det

        # Mild damping if steps are huge
        α = 1.0
        if abs(δyT) > 1.0
            α = min(α, 1.0/abs(δyT))
        end
        if abs(δy) > 1.0
            α = min(α, 1.0/abs(δy))
        end

        yT = clamp(yT + α*δyT, yT_lo, yT_hi)
        y  = clamp(y  + α*δy,  -Y_CAP, Y_CAP)

        _evalF_ideal_phi0!(F2, F3, yT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
        if !(isfinite(F2[]) && isfinite(F3[]))
            w.last_reason = PRR_NONFINITE_JACOBIAN
            break
        end
    end

    w.last_resnorm = last_nrm
    if !converged
        # best-effort return
        T  = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
        μ  = hq_mass(eos)
        ur = sinh(y)
        P, _, e = eos_Pne(T, μ, eos)
        ok = false
        return (T, μ, ur, 0.0, max(e, 0.0), P, ok)
    end

    # Residual-based acceptance
    T  = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
    μ  = hq_mass(eos)
    ur = sinh(y)
    P, _, e = eos_Pne(T, μ, eos)
    _evalF_ideal_phi0!(F2f, F3f, yT, y, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
    resnorm = sqrt(F2f[]^2 + F3f[]^2)
    w.last_resnorm = resnorm
    if !(isfinite(resnorm) && resnorm < tol_res*(1 + abs(Sr) + abs(E)))
        w.last_reason = PRR_RESIDUAL_TOO_LARGE
        return (T, μ, ur, 0.0, max(e, 0.0), P, false)
    end

    w.last_reason = PRR_CONVERGED
    return (T, μ, ur, 0.0, max(e, 0.0), P, true)
end

function cons_to_prim_ideal_phi_diff_visc!(w::PrimRecWork,
                                          D::Float64, Sr::Float64, E::Float64, nur_phys::Float64,
                                          Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                          r::Float64, τ::Float64,
                                          eos;
                                          yT0::Float64=NaN, φ0::Float64=NaN, y0::Float64=NaN,
                                          maxit::Int=100, tol::Float64=1e-15, tol_res::Float64=1e-14)

    # ---- Newton domain guards (prevents T=exp(yT) from going insane) ----
    # Pick this comfortably above anything physical you expect.
    # If you know your EOS validity range, set it accordingly.

    yT_lo = log(T_MIN)
    yT_hi = log(T_SOLVE_MAX)

    w.last_reason = PRR_UNSET
    w.last_iters = 0
    w.last_resnorm = NaN

    # IMPORTANT: allow charge-less (D==0) matter.
    # Many validations (e.g. ideal Gubser flow) are for a baryonless conformal fluid.

    # We only treat the cell as vacuum if energy is vacuum-ish or D is invalid/negative.
    if !isfinite(E) || E <= E_VAC || !isfinite(D) || D < D_VAC
        w.last_reason = PRR_VACUUM
        w.last_iters = 0
        w.last_resnorm = 0.0
        T  = T_MIN
        μ  = 0.0
        ur = 0.0
        P, _, _ = eos_Pne(T, μ, eos)
        n  = 0.0
        e  = max(E, 0.0)
        return (T, μ, ur, n, e, P, true)
    end

    # If the EOS carries no charge dependence (μ-independent), avoid the singular (yT,φ,y) Newton.
    if !eos_has_charge(eos)
        # For the conformal baryonless EOS we can invert primitives *exactly* (ideal, no dissipatives).
        if _is_baryonless_conformal_eos(eos) &&
           abs(nur_phys) <= 1e-20 && abs(Pi_phys) <= 1e-20 && abs(piR_phys) <= 1e-20 && abs(piEta_phys) <= 1e-20
            return cons_to_prim_conformal_baryonless_ideal(Sr, E, r, τ, eos)
        end

        # Otherwise fall back to a 2-variable Newton with φ fixed.
        return cons_to_prim_ideal_phi0_nocharge!(w, Sr, E, Pi_phys, piR_phys, piEta_phys, r, τ, eos;
                                                 yT0=yT0, y0=y0, maxit=maxit, tol=tol, tol_res=tol_res)
    end

    # ---- initial guess ----
    use_cache = isfinite(yT0) && isfinite(φ0) && isfinite(y0)
    # If we previously failed and fell back to vacuum (T≈T_MIN), but the current
    # conservative state is clearly not vacuum, the cached guess can prevent
    # Newton from ever finding the correct basin of attraction.
    if use_cache && (yT0 <= log(T_MIN) + 1e-12) && (E > 1e6 * E_VAC)
        use_cache = false
    end

    if use_cache
        w.x[1] = yT0
        w.x[2] = φ0
        w.x[3] = y0
        _clamp_newton_x!(w.x, yT_lo, yT_hi)
    else
        # Temperature guess from E, but bounded to EOS domain
        T0 = clamp(posden(E)^(0.25), T_MIN, T_SOLVE_MAX)
        yTg = log(T0)

        # Use a v-guess (NOT ur-guess): v must be in (-1,1)
        # A decent scaling is Sr/(E+P) ~ v for moderate flows.
        P0, _, _ = eos_Pne(T0, hq_mass(eos), eos)  # μ≈m as a neutral guess
        denom = posden(E + P0 + abs(Pi_phys) + abs(piR_phys) + abs(piEta_phys))
        v0 = clamp(safe_div(Sr, denom), -0.9999, 0.9999)     # allow near-causal starts (needed when |Sr| is large)
        yg = atanh(v0)

        # Charge-density-consistent μ/φ guess using implied n ≈ (D - v*ν)/uτ at the velocity guess.
        uτ0 = cosh(yg)
        ntarget = safe_div((D - v0*nur_phys), uτ0)
        μg = _μ_guess_from_n(eos, T0, ntarget)
        φg = (μg - hq_mass(eos)) / max(T0, T_MIN)

        w.x[1] = yTg
        w.x[2] = φg
        w.x[3] = yg
        _clamp_newton_x!(w.x, yT_lo, yT_hi)
    end

    evalF_ideal_phi_diff_visc!(w.F, w.x, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
    if !_allfinite3(w.F)
        w.last_reason = PRR_NONFINITE_INITIAL
        # fallback: don’t crash, but signal failure
        T = clamp(exp(w.x[1]), T_MIN, T_SOLVE_MAX)
        μ = hq_mass(eos)
        P, _, _ = eos_Pne(T, μ, eos)
        e = max(E, 0.0)
        return (T, μ, 0.0, 0.0, e, P, false)
    end

    ok = true
    last_nrm = NaN
    converged = false
    for it in 1:maxit
        nrm = sqrt(w.F[1]^2 + w.F[2]^2 + w.F[3]^2)
        last_nrm = nrm
        if nrm < tol
            converged = true
            w.last_reason = PRR_CONVERGED
            w.last_iters = it - 1
            break
        end

        # ---- numerical Jacobian with clamped trial points ----
        for j in 1:3
            xj = w.x[j]
            dx = 1e-8 * max(abs(xj), 1.0)

            
            @inbounds begin
                w.xtrial[1] = w.x[1]
                w.xtrial[2] = w.x[2]
                w.xtrial[3] = w.x[3]
                w.xtrial[j] = xj + dx
            end
            _clamp_newton_x!(w.xtrial, yT_lo, yT_hi)

            evalF_ideal_phi_diff_visc!(w.Fp, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
            if !_allfinite3(w.Fp)
                ok = false
                w.last_reason = PRR_NONFINITE_JACOBIAN
                break
            end

            @inbounds begin
                w.J[1,j] = (w.Fp[1] - w.F[1]) / dx
                w.J[2,j] = (w.Fp[2] - w.F[2]) / dx
                w.J[3,j] = (w.Fp[3] - w.F[3]) / dx
            end
        end
        ok || break

        ok = solve3x3_gauss!(w.δ, w.J, w.F, w.A)   # solves J*δ = -F
        if !ok
            w.last_reason = PRR_LINSOLVE_FAILED
        end
        ok || break

        # ---- trust region (caps the step so yT can't jump by 1e200) ----
        dyT_cap = 0.5   # exp(±0.5) ~ ×1.65 change in T per Newton attempt
        dφ_cap  = 2.0
        dy_cap  = 0.7

        s = 1.0
        s = min(s, dyT_cap / max(abs(w.δ[1]), 1e-300))
        s = min(s, dφ_cap  / max(abs(w.δ[2]), 1e-300))
        s = min(s, dy_cap  / max(abs(w.δ[3]), 1e-300))
        w.δ .*= s

        # ---- backtracking line search with clamped candidate ----
        αls = 1.0
        improved = false
        for _ in 1:12
            @inbounds begin
                w.xtrial[1] = w.x[1] + αls*w.δ[1]
                w.xtrial[2] = w.x[2] + αls*w.δ[2]
                w.xtrial[3] = w.x[3] + αls*w.δ[3]
            end
            _clamp_newton_x!(w.xtrial, yT_lo, yT_hi)

            evalF_ideal_phi_diff_visc!(w.Fp, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
            if _allfinite3(w.Fp)
                nrm2 = sqrt(w.Fp[1]^2 + w.Fp[2]^2 + w.Fp[3]^2)
                if nrm2 < (1 - 1e-6*αls)*nrm
                    @inbounds begin
                        w.x .= w.xtrial
                        w.F .= w.Fp
                    end
                    improved = true
                    break
                end
            end
            αls *= 0.5
        end

        if !improved
            ok = false
            w.last_reason = PRR_LINESEARCH_FAILED
            break
        end
    end

    if ok && !converged
        w.last_reason = PRR_MAXIT
        w.last_iters = maxit
        w.last_resnorm = last_nrm
    end

    # ---- final primitives ----
    _clamp_newton_x!(w.x, yT_lo, yT_hi)
    yT = w.x[1]; φ = w.x[2]; y = w.x[3]

    T  = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
    μ  = hq_mass(eos) + T*φ
    ur = sinh(y)

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite(T) && isfinite(n) && isfinite(e) && isfinite(P) && T > 0 && e > 0)
        w.last_reason = PRR_EOS_INVALID
        return (T, 0.0, 0.0, 0.0, 0.0, 0.0, false)
    end

    @inbounds begin
        w.xtrial[1] = yT; w.xtrial[2] = φ; w.xtrial[3] = y
    end
        evalF_ideal_phi_diff_visc!(w.Fres, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, eos)
    resnorm = sqrt(w.Fres[1]^2 + w.Fres[2]^2 + w.Fres[3]^2)
    w.last_resnorm = resnorm
    if !(_allfinite3(w.Fres) &&
          resnorm < tol_res*(1 + abs(D) + abs(Sr) + abs(E)))
        w.last_reason = PRR_RESIDUAL_TOO_LARGE
        return (T, μ, ur, n, e, P, false)
    end

    w.last_reason = PRR_CONVERGED

    return (T, μ, ur, n, e, P, true)
end

mutable struct IdealPrimRec
    work::Vector{PrimRecWork}
end
# Size per-thread scratch to the full thread-id range, not just the default pool:
# Threads.@threads can run on the :interactive pool too, so threadid() may reach
# nthreads(:default)+nthreads(:interactive).  Using only Threads.nthreads() (default
# pool) under-sizes and triggers a BoundsError in the threaded primitive loops
# whenever an interactive thread exists.
_primrec_nslots() = Threads.nthreads() + Threads.nthreads(:interactive)
IdealPrimRec() = IdealPrimRec([PrimRecWork() for _ in 1:_primrec_nslots()])

@inline function cons_to_prim_col(U::AbstractMatrix, i::Int, r::Float64, τ::Float64, model::IdealDiffViscModel)
    L = model.layout
    tid = Threads.threadid()
    w = model.primrec.work[tid]

    D  = safe_div(U[L.iDtau, i], τ)
    Sr = U[L.iSr,i]
    E  = U[L.iE,i]

    nur_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur,i])   : 0.0)
    Pi_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi,i])    : 0.0)
    piR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
    piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

    T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(w, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, model.eos)
    return PrimIdealVisc(T, μ, ur, n, e, P, nur_phys, Pi_phys, piR_phys, piEta_phys, ok)
end

# ------------------------------------------------------------
# prim->cons into column
# ------------------------------------------------------------
@inline function prim_to_cons_col_ideal_phi_diff_visc!(Umat::AbstractMatrix, col::Int,
                                                      yT::Float64, φ::Float64, y::Float64,
                                                      nur_phys::Float64, Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                                      r::Float64,
                                                      τ::Float64,
                                                      eos, L::StateLayout)
    T  = max(exp(yT), T_MIN)
    μ  = hq_mass(eos) + T*φ
    ur = sinh(y)
    uτ = cosh(y)
    v  = safe_div(ur, uτ)

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite3(P,n,e)) || e <= 0.0
        return false, PrimIdealVisc(T, μ, 0.0, 0.0, 0.0, 0.0, nur_phys, Pi_phys, piR_phys, piEta_phys, false)
    end

    w    = e + P
    Ptot = P + Pi_phys
    weff = w + Pi_phys

    D = n*uτ + v*nur_phys

    Π = shear_tensor_contravariant(ur, uτ, r, τ, piR_phys, piEta_phys)

    @inbounds begin
        Umat[L.iDtau, col] = τ * D
        Umat[L.iSr,   col] = weff * (uτ*ur) + Π.tr
        Umat[L.iE,    col] = weff * (uτ^2) - Ptot + Π.tt

        if L.hasNur
            Umat[L.iNur, col] = stored_from_phys(nur_phys)
        end
        if L.hasPi
            Umat[L.iPi, col] = stored_from_phys(Pi_phys)
        end
        if L.hasPiR
            Umat[L.iPiR, col] = stored_from_phys(piR_phys)
        end
        if L.hasPiEta
            Umat[L.iPiEta, col] = stored_from_phys(piEta_phys)
        end
    end

    return true, PrimIdealVisc(T, μ, ur, n, e, P, nur_phys, Pi_phys, piR_phys, piEta_phys, true)
end
