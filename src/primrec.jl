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
end
PrimRecWork() = PrimRecWork(zeros(3), zeros(3), zeros(3), zeros(3,3), zeros(3), zeros(3), zeros(3,4), zeros(3))

@inline function evalF_ideal_phi_diff_visc!(F::Vector{Float64}, x::Vector{Float64},
                                           D::Float64, Sr::Float64, E::Float64, nur_phys::Float64,
                                           Pi_phys::Float64, piR_phys::Float64,
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
    v  = ur / max(uτ, 1e-50)

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

    F[1] = n*uτ + v*nur_phys - D
    F[2] = (weff + piR_phys) * (uτ*ur) - Sr
    F[3] = weff*uτ^2 - Ptot + (ur^2)*piR_phys - E
    return
end

function cons_to_prim_ideal_phi_diff_visc!(w::PrimRecWork,
                                          D::Float64, Sr::Float64, E::Float64, nur_phys::Float64,
                                          Pi_phys::Float64, piR_phys::Float64,
                                          eos;
                                          yT0::Float64=NaN, φ0::Float64=NaN, y0::Float64=NaN,
                                          maxit::Int=40, tol::Float64=1e-10, tol_res::Float64=1e-9)

    # ---- Newton domain guards (prevents T=exp(yT) from going insane) ----
    # Pick this comfortably above anything physical you expect.
    # If you know your EOS validity range, set it accordingly.

    yT_lo = log(T_MIN)
    yT_hi = log(T_SOLVE_MAX)

    @inline function clamp_x!(x::Vector{Float64})
        @inbounds begin
            x[1] = clamp(x[1], yT_lo, yT_hi)
            x[2] = clamp(x[2], -PHI_CAP, PHI_CAP)
            x[3] = clamp(x[3], -Y_CAP, Y_CAP)
        end
        return nothing
    end

    if !isfinite(E) || E <= E_VAC || !isfinite(D) || D <= D_VAC
        T  = T_MIN
        μ  = 0.0
        ur = 0.0
        P, _, _ = eos_Pne(T, μ, eos)
        n  = 0.0
        e  = max(E, 0.0)
        return (T, μ, ur, n, e, P, true)
    end

    # ---- initial guess ----
    if isfinite(yT0) && isfinite(φ0) && isfinite(y0)
        w.x[1] = yT0
        w.x[2] = φ0
        w.x[3] = y0
        clamp_x!(w.x)
    else
        @inline function μ_guess_from_n(T::Float64, n_target::Float64)
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
                K2x = safe_besselkx(2, x)
                F   = A * Tuse * max(K2x, 0.0)
                if !(isfinite(F)) || F <= 0.0
                    return m
                end
                μ = m + Tuse * log(nt / (F + TINY))
                # keep φ=(μ-m)/T within caps
                return m + Tuse * clamp((μ - m) / Tuse, -PHI_CAP, PHI_CAP)
            elseif eos isa TabulatedHQEOS
                # Tabulated EOS stores n(T,α). Invert α by bisection in [αmin, αmax].
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

        # Temperature guess from E, but bounded to EOS domain
        T0 = clamp(max(E, 1e-50)^(0.25), T_MIN, T_SOLVE_MAX)
        yTg = log(T0)

        # Use a v-guess (NOT ur-guess): v must be in (-1,1)
        # A decent scaling is Sr/(E+P) ~ v for moderate flows.
        P0, _, _ = eos_Pne(T0, hq_mass(eos), eos)  # μ≈m as a neutral guess
        denom = max(E + P0 + abs(Pi_phys) + abs(piR_phys), 1e-50)
        v0 = clamp(Sr / denom, -0.95, 0.95)       # safe start, not too close to 1
        yg = atanh(v0)

        # Charge-density-consistent μ/φ guess using implied n ≈ (D - v*ν)/uτ at the velocity guess.
        uτ0 = cosh(yg)
        ntarget = (D - v0*nur_phys) / max(uτ0, 1e-50)
        μg = μ_guess_from_n(T0, ntarget)
        φg = (μg - hq_mass(eos)) / max(T0, T_MIN)

        w.x[1] = yTg
        w.x[2] = φg
        w.x[3] = yg
        clamp_x!(w.x)
    end

    evalF_ideal_phi_diff_visc!(w.F, w.x, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos)
    if !all(isfinite, w.F)
        # fallback: don’t crash, but signal failure
        T = clamp(exp(w.x[1]), T_MIN, T_SOLVE_MAX)
        μ = hq_mass(eos)
        P, _, _ = eos_Pne(T, μ, eos)
        e = max(E, 0.0)
        return (T, μ, 0.0, 0.0, e, P, false)
    end

    ok = true
    for _ in 1:maxit
        nrm = sqrt(w.F[1]^2 + w.F[2]^2 + w.F[3]^2)
        if nrm < tol
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
            clamp_x!(w.xtrial)

            evalF_ideal_phi_diff_visc!(w.Fp, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos)
            if !all(isfinite, w.Fp)
                ok = false
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
            clamp_x!(w.xtrial)

            evalF_ideal_phi_diff_visc!(w.Fp, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos)
            if all(isfinite, w.Fp)
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
            break
        end
    end

    # ---- final primitives ----
    clamp_x!(w.x)
    yT = w.x[1]; φ = w.x[2]; y = w.x[3]

    T  = clamp(exp(yT), T_MIN, T_SOLVE_MAX)
    μ  = hq_mass(eos) + T*φ
    ur = sinh(y)

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite(T) && isfinite(n) && isfinite(e) && isfinite(P) && T > 0 && e > 0)
        return (T, 0.0, 0.0, 0.0, 0.0, 0.0, false)
    end

    @inbounds begin
        w.xtrial[1] = yT; w.xtrial[2] = φ; w.xtrial[3] = y
    end
    evalF_ideal_phi_diff_visc!(w.Fres, w.xtrial, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos)
    if !(all(isfinite, w.Fres) &&
          sqrt(w.Fres[1]^2 + w.Fres[2]^2 + w.Fres[3]^2) < tol_res*(1 + abs(D) + abs(Sr) + abs(E)))
        return (T, μ, ur, n, e, P, false)
    end

    return (T, μ, ur, n, e, P, true)
end

mutable struct IdealPrimRec
    work::Vector{PrimRecWork}
end
IdealPrimRec() = IdealPrimRec([PrimRecWork() for _ in 1:Threads.nthreads()])

@inline function cons_to_prim_col(U::AbstractMatrix, i::Int, τ::Float64, model::IdealDiffViscModel)
    L = model.layout
    tid = Threads.threadid()
    w = model.primrec.work[tid]

    D  = U[L.iDtau,i] / max(τ, 1e-50)
    Sr = U[L.iSr,i]
    E  = U[L.iE,i]

    nur_phys   = (L.hasNur   ? phys_from_stored(U[L.iNur,i])   : 0.0)
    Pi_phys    = (L.hasPi    ? phys_from_stored(U[L.iPi,i])    : 0.0)
    piR_phys   = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
    piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

    T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(w, D, Sr, E, nur_phys, Pi_phys, piR_phys, model.eos)
    return PrimIdealVisc(T, μ, ur, n, e, P, nur_phys, Pi_phys, piR_phys, piEta_phys, ok)
end

# ------------------------------------------------------------
# prim->cons into column
# ------------------------------------------------------------
@inline function prim_to_cons_col_ideal_phi_diff_visc!(Umat::AbstractMatrix, col::Int,
                                                      yT::Float64, φ::Float64, y::Float64,
                                                      nur_phys::Float64, Pi_phys::Float64, piR_phys::Float64, piEta_phys::Float64,
                                                      τ::Float64,
                                                      eos, L::StateLayout)
    T  = max(exp(yT), T_MIN)
    μ  = hq_mass(eos) + T*φ
    ur = sinh(y)
    uτ = cosh(y)
    v  = ur / max(uτ, 1e-50)

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite3(P,n,e)) || e <= 0.0
        return false, PrimIdealVisc(T, μ, 0.0, 0.0, 0.0, 0.0, nur_phys, Pi_phys, piR_phys, piEta_phys, false)
    end

    w    = e + P
    Ptot = P + Pi_phys
    weff = w + Pi_phys

    D = n*uτ + v*nur_phys

    pi_tr = (uτ*ur) * piR_phys
    pi_tt = (ur*ur) * piR_phys

    @inbounds begin
        Umat[L.iDtau, col] = τ * D
        Umat[L.iSr,   col] = weff * (uτ*ur) + pi_tr
        Umat[L.iE,    col] = weff * (uτ^2) - Ptot + pi_tt

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
