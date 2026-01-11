# ==============================================================================
# src/dissipation.jl
#
# Dissipative relaxation: diffusion, shear, bulk.
# ==============================================================================

# ------------------------------------------------------------
# Charge diffusion Coefficients
# ------------------------------------------------------------
@inline diff_kappa(T, μ, n, model::IdealDiffViscModel) =
    model.kappa_coeff * max(n,0.0) / max(T, 1e-50) / fmGeV

@inline function diff_tauN(T, μ, model::IdealDiffViscModel)
    Tm = max(T, 1e-50)
    m  = hq_mass(model.eos)
    z  = m / Tm
    z <= 0 && return 0.0

    # For very large z (cold regime), use asymptotic approximation to avoid
    # numerical instabilities from forward recurrence and catastrophic cancellation
    if z > 50.0
        # Asymptotic analysis: For large z, the besselkx functions have similar magnitude
        # and the numerator (2*K1 - 3*K3 + K5) has leading-order cancellation.
        # The ratio ~ O(1/z²) for large z, making τ ~ z³/Tm * O(1/z²) = z/Tm = m/Tm²
        #
        # To avoid overflow in z^3/Tm and ensure bounded behavior, we directly
        # compute a saturated timescale based on the asymptotic limit.
        # In the cold limit (T → 0), diffusion timescale should grow but remain finite.

        DsT  = model.kappa_coeff
        tauD = model.tauN_coeff

        # Asymptotic formula: τ ~ (DsT/48) * (m²/Tm²) with geometric mean regularization
        # This gives bounded behavior even as T → 0
        τ_GeVinv = (DsT / 48) * (m^2 / (Tm^2 + 1e-10))  # Regularized to avoid overflow
        return min((τ_GeVinv / fmGeV) * tauD, 1e20)  # Cap at very large but finite value
    end

    # For moderate z, use direct calculation of Bessel functions
    # (avoid forward recurrence which is unstable for large arguments)
    K1x = safe_besselkx(1, z)
    K2x = safe_besselkx(2, z)
    K3x = safe_besselkx(3, z)  # Direct calculation instead of recurrence
    K5x = safe_besselkx(5, z)  # Direct calculation instead of recurrence

    # Compute the ratio with protection against division by zero
    numerator = 2*K1x - 3*K3x + K5x
    denominator = max(abs(K2x), TINY)
    ratio = numerator / denominator * sign(K2x == 0 ? 1.0 : K2x)

    # Guard against numerical overflow in z^3/Tm term
    # If z is large enough that z^3/Tm would overflow, cap it
    z3_over_Tm = z^3 / Tm
    if !isfinite(z3_over_Tm) || z3_over_Tm > 1e50
        z3_over_Tm = 1e50  # Cap at a large but finite value
    end

    DsT  = model.kappa_coeff
    tauD = model.tauN_coeff

    τ_GeVinv = (DsT / 48) * z3_over_Tm * ratio

    # Final safety check: ensure result is finite and reasonable
    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)  # Cap at large but finite value
    end

    return (τ_GeVinv / fmGeV) * tauD
end

@inline diff_deltaNN(T, μ, model::IdealDiffViscModel) = model.deltaN_factor * diff_tauN(T, μ, model)

@inline function eps_from_len(len::Float64, dr::Float64; c::Float64=0.10)
    len <= 0 && return 0.0
    ϵ = c * (len/dr)^2
    return clamp(ϵ, 0.0, 0.24)
end

# ------------------------------------------------------------
# Diffusion / viscosity helpers (θ, grad α, smoothing)
# ------------------------------------------------------------
function compute_theta!(work::Work1D, grid, τ)
    ng = grid.nghost
    Ntot = length(work.vC)
    fill!(work.theta, 0.0)
    fill!(work.vF, 0.0)

    @inbounds for i in 1:(Ntot-1)
        work.vF[i] = 0.5*(work.vC[i] + work.vC[i+1])
    end

    i0 = ng+1
    work.theta[i0] = 1/max(τ,1e-50) + 2*work.vF[i0]/grid.dr

    Threads.@threads for i in (i0+1):(Ntot-ng)
        rC  = max(grid.rC[i], 1e-50)
        rRp = grid.rF[i+1]
        rRm = grid.rF[i]
        div = (rRp*work.vF[i] - rRm*work.vF[i-1]) / grid.dr
        work.theta[i] = 1/max(τ,1e-50) + div / rC
    end
    return nothing
end

function compute_dvdr!(work::Work1D, grid)
    ng = grid.nghost
    Ntot = length(work.vC)
    fill!(work.dvdr, 0.0)
    i0 = ng+1
    work.dvdr[i0] = (work.vF[i0] - work.vF[i0-1]) / grid.dr
    Threads.@threads for i in (i0+1):(Ntot-ng)
        work.dvdr[i] = (work.vF[i] - work.vF[i-1]) / grid.dr
    end
    return nothing
end

function smooth_alpha!(work::Work1D, grid, eps::Float64)
    eps <= 0 && return nothing
    ng = grid.nghost
    Ntot = length(work.alpha)
    copyto!(work.alpha_tmp, work.alpha)

    i0 = ng+1
    iL = Ntot-ng
    @inbounds for i in (i0+1):(iL-1)
        work.alpha[i] = work.alpha_tmp[i] + eps*(work.alpha_tmp[i-1] - 2*work.alpha_tmp[i] + work.alpha_tmp[i+1])
    end
    work.alpha[i0] = work.alpha_tmp[i0]
    work.alpha[iL] = work.alpha_tmp[iL]
    return nothing
end

function compute_alpha_grad_fv!(work::Work1D, grid; limiter=minmod)
    ng = grid.nghost
    Ntot = length(work.alpha)
    Nf   = Ntot - 1
    fill!(work.alphaF, 0.0)
    fill!(work.gradAlpha, 0.0)

    slope = work.gradAlpha
    fill!(slope, 0.0)

    @inbounds for i in (ng+2):(Ntot-ng-1)
        dL = work.alpha[i]   - work.alpha[i-1]
        dR = work.alpha[i+1] - work.alpha[i]
        slope[i] = limiter(dL, dR)
    end

    @inbounds for i in 1:Nf
        if i <= (ng+1) || i >= (Ntot-ng-1)
            work.alphaF[i] = 0.5*(work.alpha[i] + work.alpha[i+1])
        else
            αL = work.alpha[i]   + 0.5*slope[i]
            αR = work.alpha[i+1] - 0.5*slope[i+1]
            work.alphaF[i] = 0.5*(αL + αR)
        end
    end

    i0 = ng+1
    work.gradAlpha[i0] = (work.alphaF[i0] - work.alphaF[i0-1]) / grid.dr
    Threads.@threads for i in (i0+1):(Ntot-ng)
        work.gradAlpha[i] = (work.alphaF[i] - work.alphaF[i-1]) / grid.dr
    end
    return nothing
end

function smooth_field_centered!(U, idx::Int, worktmp::Vector{Float64}, grid, eps::Float64)
    eps <= 0 && return nothing
    ng = grid.nghost
    Ntot = size(U,2)
    @inbounds for i in 1:Ntot
        worktmp[i] = U[idx,i]
    end
    i0 = ng+1
    iL = Ntot-ng
    @inbounds for i in (i0+1):(iL-1)
        U[idx,i] = worktmp[i] + eps*(worktmp[i-1] - 2*worktmp[i] + worktmp[i+1])
    end
    return nothing
end

function smooth_n_for_clip!(work::Work1D, grid; eps::Float64=0.25)
    ng = grid.nghost
    Ntot = length(work.n)
    copyto!(work.n_tmp, work.n)

    i0 = ng+1
    iL = Ntot-ng
    @inbounds for i in (i0+1):(iL-1)
        work.n_tmp[i] = work.n[i] + eps*(work.n[i-1] - 2*work.n[i] + work.n[i+1])
    end
    work.n_tmp[i0] = work.n[i0]
    work.n_tmp[iL] = work.n[iL]
    return nothing
end

function axis_project_nur_tapered!(U::AbstractMatrix, grid::Grid1D, model::IdealDiffViscModel; nfit::Int=10)
    L = layout(model)
    (!L.hasNur) && return nothing

    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng
    nfit = clamp(nfit, 2, iL - i0 + 1)

    num = 0.0
    den = 0.0
    @inbounds for k in 0:(nfit-1)
        i = i0 + k
        r = grid.rC[i]
        ν_phys = phys_from_stored(U[L.iNur,i])
        num += r * ν_phys
        den += r * r
    end
    s = num / (den + TINY)

    @inbounds for k in 0:(nfit-1)
        i = i0 + k
        r = grid.rC[i]
        νfit_phys = s * r

        ξ = k / (nfit-1)
        w = cos(0.5*pi*ξ)^2

        νold_phys = phys_from_stored(U[L.iNur,i])
        νnew_phys = (1.0 - w) * νold_phys + w * νfit_phys
        U[L.iNur,i] = stored_from_phys(νnew_phys)
    end
    return nothing
end

# ------------------------------------------------------------
# Viscosity helpers (use explicit models)
# ------------------------------------------------------------
@inline function visc_eta(T, μ, n, e, P, model::IdealDiffViscModel)
    th = local_thermo(T, μ, n, e, P, model.eos)
    return viscosity(T, th, model.shear)
end

@inline function visc_zeta(T, μ, n, e, P, model::IdealDiffViscModel)
    th = local_thermo(T, μ, n, e, P, model.eos)
    return bulk_viscosity(T, th, model.bulk)
end

@inline function visc_tauShear(T, μ, n, e, P, model::IdealDiffViscModel)
    th = local_thermo(T, μ, n, e, P, model.eos)
    return τ_shear(T, th, model.shear)
end

@inline function visc_tauPi(T, μ, n, e, P, model::IdealDiffViscModel)
    th = local_thermo(T, μ, n, e, P, model.eos)
    return τ_bulk(T, th, model.bulk)
end

@inline function visc_deltaPi(T, μ, n, e, P, model::IdealDiffViscModel)
    return model.deltaPi_factor * visc_tauPi(T, μ, n, e, P, model)
end

@inline function visc_deltaShear(T, μ, n, e, P, model::IdealDiffViscModel)
    return model.deltaShear_factor * visc_tauShear(T, μ, n, e, P, model)
end

# ------------------------------------------------------------
# Dissipative relaxation: ν_r + Π + π (semi-implicit BE)
# ------------------------------------------------------------
function relax_dissipative!(U, grid, τ, Δ, model::IdealDiffViscModel, work::Work1D)
    L = model.layout
    Ntot = size(U,2)

    if !( (model.enable_diff && L.hasNur) || 
          (model.enable_shear && (L.hasPiR || L.hasPiEta)) || 
          (model.enable_bulk  && L.hasPi) )
        return nothing
    end

    # Recover primitives everywhere (and build vC, alpha, etc.)
    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D   = U[L.iDtau,i] / max(τ, 1e-50)
        Sr  = U[L.iSr,i]
        E   = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)

        yT0 = work.yT[i]
        φ0  = work.phi[i]
        y0  = work.y[i]

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, model.eos;
            yT0=yT0, φ0=φ0, y0=y0, maxit=40
        )

        if ok
            T = max(T, T_MIN)
            work.yT[i]  = log(T)
            work.mu[i]  = μ
            work.phi[i] = (μ - hq_mass(model.eos)) / T
            work.y[i]   = asinh(ur)

            work.n[i]  = n
            work.P[i]  = P
            work.e[i]  = e
            work.ok[i] = true

            uτ = sqrt(1 + ur^2)
            work.vC[i]    = ur / max(uτ, 1e-50)
            work.alpha[i] = μ / T
        else
            work.ok[i]    = false
            work.vC[i]    = 0.0
            work.alpha[i] = 0.0
            work.n[i]     = 0.0
            work.P[i]     = 0.0
            work.e[i]     = 0.0
            work.mu[i]    = 0.0
            work.yT[i]    = log(T_MIN)
            work.y[i]     = 0.0
            work.phi[i]   = 0.0
        end
    end

    ng = grid.nghost

    # Fail-safe: if primitive recovery fails, zero dissipative vars locally.
    # This prevents runaway stresses/fluxes in near-vacuum regions from poisoning dt/MOOD.
    Threads.@threads for i in (ng+1):(Ntot-ng)
        if !work.ok[i]
            if L.hasNur
                U[L.iNur, i] = stored_from_phys(0.0)
            end
            if L.hasPi
                U[L.iPi, i] = stored_from_phys(0.0)
            end
            if L.hasPiR
                U[L.iPiR, i] = stored_from_phys(0.0)
            end
            if L.hasPiEta
                U[L.iPiEta, i] = stored_from_phys(0.0)
            end
        end
    end

    # ---- diffusion ----
    if model.enable_diff && L.hasNur
        @inline function _soft_project_nur_phys(ν::Float64, n::Float64, uτ::Float64)
            νbound = max(n * uτ, 0.0)
            νbound <= 0.0 && return 0.0
            # Smooth projection into the physically-admissible domain |ν^r| <= n uτ.
            # This avoids hard clipping but prevents states with no valid primitive recovery.
            return νbound * tanh(ν / (νbound + TINY))
        end

        epsα = clamp(model.alpha_filter_eps + eps_from_len(model.alpha_smooth_len, grid.dr), 0.0, 0.24)
        epsν = clamp(model.nur_filter_eps   + eps_from_len(model.nur_smooth_len,   grid.dr), 0.0, 0.24)

        smooth_alpha!(work, grid, epsα)

        @inbounds for g in 1:ng
            il = ng + 1 - g
            ir = ng + g
            work.alpha[il] = work.alpha[ir]
        end

        compute_theta!(work, grid, τ)
        compute_alpha_grad_fv!(work, grid; limiter=mc_limiter)

        @inbounds for i in 1:Ntot
            work.nur_tmp[i] = phys_from_stored(U[L.iNur,i])
        end

        Threads.@threads for i in (ng+1):(Ntot-ng)
            work.ok[i] || continue

            ur = sinh(work.y[i])
            uτ = sqrt(1 + ur^2)
            v  = ur / max(uτ, 1e-50)

            θ  = work.theta[i]
            dα = work.gradAlpha[i]

            T  = exp(work.yT[i])
            μ  = work.mu[i]
            n  = work.n[i]

            DsT = model.kappa_coeff
            κ = (DsT / max(T,1e-50)) * max(n,0.0) / fmGeV

            τn = diff_tauN(T, μ, model)
            δ  = diff_deltaNN(T, μ, model)

            # Conventional sign: ν_NS = -κ uτ^2 ∂r α
            νNS_raw  = -κ * (uτ^2) * dα
            νNS_phys = _soft_project_nur_phys(νNS_raw, n, uτ)

            adv_src = 0.0
            if model.relax_advect_nur
                νi = work.nur_tmp[i]
                dν_dr = ifelse(v >= 0.0,
                               (νi - work.nur_tmp[i-1]) / grid.dr,
                               (work.nur_tmp[i+1] - νi) / grid.dr)
                adv_src = -Δ * v * dν_dr
            end

            νold_phys = phys_from_stored(U[L.iNur, i])
            
            νnew_phys = relaxation_update_nur_phys(DefaultRelaxationLaw();
                νold_phys=νold_phys, νNS_phys=νNS_phys, adv_src=adv_src,
                Δ=Δ, τn=τn, δ=δ, θ=θ, uτ=uτ
            )

            # Keep ν within the causal domain so (D,S,E,ν) remains solvable.
            νnew_phys = _soft_project_nur_phys(νnew_phys, n, uτ)

            U[L.iNur,i] = stored_from_phys(νnew_phys)
        end

        if model.do_axis_project_nur
            axis_project_nur_tapered!(U, grid, model; nfit=model.axis_project_nfit)
        end

        smooth_field_centered!(U, L.iNur, work.nur_tmp, grid, epsν)

        # Soft causality enforcement after post-processing (axis projection + smoothing).
        Threads.@threads for i in (ng+1):(Ntot-ng)
            work.ok[i] || continue
            ur = sinh(work.y[i])
            uτ = sqrt(1 + ur^2)
            νphys = phys_from_stored(U[L.iNur,i])
            νphys = _soft_project_nur_phys(νphys, work.n[i], uτ)
            U[L.iNur,i] = stored_from_phys(νphys)
        end

        if model.nur_clip_factor > 0
            smooth_n_for_clip!(work, grid)
            Threads.@threads for i in (ng+1):(Ntot-ng)
                work.ok[i] || continue
                #νmax = model.nur_clip_factor * max(work.n_tmp[i], 0.0)
                ur  = sinh(work.y[i])
                uτ  = sqrt(1 + ur^2)
                νmax = model.nur_clip_factor * max(work.n_tmp[i] * uτ, 0.0)   # <- key change

                νphys = phys_from_stored(U[L.iNur,i])
                νphys = clamp(νphys, -νmax, νmax)
                U[L.iNur,i] = stored_from_phys(νphys)
            end
        end
    end

    # ---- bulk + shear ----
    if (model.enable_shear && (L.hasPiR || L.hasPiEta)) || (model.enable_bulk && L.hasPi)
        epsV = clamp(model.visc_filter_eps + eps_from_len(model.visc_smooth_len, grid.dr), 0.0, 0.24)

        compute_theta!(work, grid, τ)
        compute_dvdr!(work, grid)

        if model.relax_advect_Pi && L.hasPi
            @inbounds for i in 1:Ntot
                work.visc_tmp1[i] = phys_from_stored(U[L.iPi,i])
            end
        end
        if model.relax_advect_pi
            if L.hasPiR
                @inbounds for i in 1:Ntot
                    work.visc_tmp2[i] = phys_from_stored(U[L.iPiR,i])
                end
            end
            if L.hasPiEta
                @inbounds for i in 1:Ntot
                    work.visc_tmp3[i] = phys_from_stored(U[L.iPiEta,i])
                end
            end
        end

        Threads.@threads for i in (ng+1):(Ntot-ng)
            work.ok[i] || continue

            dv   = work.dvdr[i]
            θ    = work.theta[i]
            invτ = 1 / max(τ, 1e-50)

            ur = sinh(work.y[i])
            uτ = sqrt(1 + ur^2)

            T = exp(work.yT[i])
            μ = work.mu[i]
            n = work.n[i]
            e = work.e[i]
            P = work.P[i]

            cap = max(e + P, 0.0)
            if cap < 1e6 * E_FLOOR
                if L.hasPi
                    U[L.iPi,i] = stored_from_phys(0.0)
                end
                if L.hasPiR
                    U[L.iPiR,i] = stored_from_phys(0.0)
                end
                if L.hasPiEta
                    U[L.iPiEta,i] = stored_from_phys(0.0)
                end
                continue
            end

            if model.enable_shear
                η  = visc_eta(T, μ, n, e, P, model)
                τπ = visc_tauShear(T, μ, n, e, P, model)
                δπ = visc_deltaShear(T, μ, n, e, P, model)
            else
                η, τπ, δπ = 0.0, 1.0, 0.0
            end

            if model.enable_bulk
                ζ  = visc_zeta(T, μ, n, e, P, model)
                τΠ = visc_tauPi(T, μ, n, e, P, model)
                δΠ = visc_deltaPi(T, μ, n, e, P, model)
            else
                 ζ, τΠ, δΠ = 0.0, 1.0, 0.0
            end

            v_over_r = θ - invτ - dv

            σr = dv        - θ/3
            σφ = v_over_r  - θ/3
            ση = invτ      - θ/3

            ΠNS_phys     = -ζ * θ
            # Sign convention: with our σ definitions, the Navier–Stokes shear targets
            # must use π_NS = -2η σ so that Bjorken flow yields piEta < 0.
            piRNS_phys   = -2η * σr
            piEtaNS_phys = -2η * ση

            v = ur / max(uτ, 1e-50)

            # --- Advection source terms (upwind) ---
            adv_Pi = 0.0
            if model.enable_bulk && model.relax_advect_Pi && L.hasPi
                val = work.visc_tmp1[i]
                dval = (v >= 0.0) ? (val - work.visc_tmp1[i-1])/grid.dr : (work.visc_tmp1[i+1] - val)/grid.dr
                adv_Pi = -Δ * v * dval
            end

            adv_piR = 0.0
            if model.enable_shear && model.relax_advect_pi && L.hasPiR
                val = work.visc_tmp2[i]
                dval = (v >= 0.0) ? (val - work.visc_tmp2[i-1])/grid.dr : (work.visc_tmp2[i+1] - val)/grid.dr
                adv_piR = -Δ * v * dval
            end

            adv_piEta = 0.0
            if model.enable_shear && model.relax_advect_pi && L.hasPiEta
                val = work.visc_tmp3[i]
                dval = (v >= 0.0) ? (val - work.visc_tmp3[i-1])/grid.dr : (work.visc_tmp3[i+1] - val)/grid.dr
                adv_piEta = -Δ * v * dval
            end

            if model.enable_bulk && L.hasPi
                Πold_phys = phys_from_stored(U[L.iPi, i])
                Πnew_phys = relaxation_update_Pi_phys(DefaultRelaxationLaw();
                    Πold_phys=Πold_phys, ΠNS_phys=ΠNS_phys, adv_src=adv_Pi,
                    Δ=Δ, τΠ=τΠ, δΠ=δΠ, θ=θ, uτ=uτ
                )
                U[L.iPi,i] = stored_from_phys(Πnew_phys)
            end

            if model.enable_shear
                pr_old = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
                pe_old = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
                pp_old = -(pr_old + pe_old)

                pr_NS = piRNS_phys
                pe_NS = piEtaNS_phys
                pp_NS = -(pr_NS + pe_NS)

                adv_pp = -(adv_piR + adv_piEta)

                pp_new, pe_new = relaxation_update_pi_phi_eta_phys(DefaultRelaxationLaw();
                   πφ_old_phys = pp_old,
                   πη_old_phys = pe_old,
                   πφNS_phys   = pp_NS,
                   πηNS_phys   = pe_NS,
                   adv_πφ      = adv_pp,
                   adv_πη      = adv_piEta,
                   Δ=Δ, τπ=τπ, δπ=δπ, θ=θ, uτ=uτ
                )

                pr_new = -(pp_new + pe_new)

                if L.hasPiR
                    U[L.iPiR,i] = stored_from_phys(pr_new)
                end
                if L.hasPiEta
                    U[L.iPiEta,i] = stored_from_phys(pe_new)
                end
            end
        end

        if epsV > 0
            if model.enable_bulk && L.hasPi
                smooth_field_centered!(U, L.iPi, work.visc_tmp1, grid, epsV)
            end
            if model.enable_shear && L.hasPiR
                smooth_field_centered!(U, L.iPiR, work.visc_tmp2, grid, epsV)
            end
            if model.enable_shear && L.hasPiEta
                smooth_field_centered!(U, L.iPiEta, work.visc_tmp3, grid, epsV)
            end
        end

        # Stability clip:
        # - If clip_factor == 0, use a conservative default (0.5).
        # - If clip_factor < 0, treat as disabled.
        if (model.enable_bulk && L.hasPi) || (model.enable_shear && (L.hasPiR || L.hasPiEta))
            Πfac = model.Pi_clip_factor
            πfac = model.pi_clip_factor
            do_Πclip = Πfac >= 0
            do_πclip = πfac >= 0

            Πfac = (Πfac == 0) ? 0.5 : Πfac
            πfac = (πfac == 0) ? 0.5 : πfac
            Threads.@threads for i in (ng+1):(Ntot-ng)
                work.ok[i] || continue
                cap = max(work.P[i] + work.e[i], 0.0)
                if do_Πclip && model.enable_bulk && L.hasPi
                    Πmax = Πfac * cap
                    Πphys = clamp(phys_from_stored(U[L.iPi,i]), -Πmax, Πmax)
                    U[L.iPi,i] = stored_from_phys(Πphys)
                end
                if do_πclip && model.enable_shear
                    pimax = πfac * cap
                    if L.hasPiR
                        pr = clamp(phys_from_stored(U[L.iPiR,i]), -pimax, pimax)
                        U[L.iPiR,i] = stored_from_phys(pr)
                    end
                    if L.hasPiEta
                        pe = clamp(phys_from_stored(U[L.iPiEta,i]), -pimax, pimax)
                        U[L.iPiEta,i] = stored_from_phys(pe)
                    end
                end
            end
        end
    end

    return nothing
end

