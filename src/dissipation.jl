# ==============================================================================
# src/dissipation.jl
#
# Dissipative relaxation: diffusion, shear, bulk.
# ==============================================================================

# ------------------------------------------------------------
# Charge diffusion Coefficients
# ------------------------------------------------------------
@inline diff_kappa(T, μ, n, model::IdealDiffViscModel) =
    model.kappa_coeff * safe_div(max(n, 0.0), T) / fmGeV

@inline function diff_dn_dT_at_fixed_alpha(T::Float64, α::Float64, model::IdealDiffViscModel;
                                           relstep::Float64=1e-4,
                                           absstep::Float64=1e-6)
    T0 = max(T, T_EOS_MIN)
    h  = max(relstep * T0, absstep)

    Tp = T0 + h
    μp = α * Tp
    _, np, _ = eos_Pne(Tp, μp, model.eos)

    if T0 - h <= T_EOS_MIN
        μ0 = α * T0
        _, n0, _ = eos_Pne(T0, μ0, model.eos)
        dndT = (np - n0) / h
    else
        Tm = T0 - h
        μm = α * Tm
        _, nm, _ = eos_Pne(Tm, μm, model.eos)
        dndT = (np - nm) / (Tp - Tm)
    end

    return isfinite(dndT) ? dndT : 0.0
end

@inline function _fluidum_single_hadron_normalization(T::Float64, α::Float64, eos::LatticeHRGEOS)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    b2 = SpecialFunctions.besselkx(2, z)
    ex = exp(clamp(α - z, -700.0, 700.0))
    return eos.g_hq * (Tm / (2π^2)) * m^2 * ex * b2 * fmGeV3
end

@inline function _diff_tauN_impl(Tm::Float64, α::Float64, eos::LatticeHRGEOS, DsT::Float64, tauD::Float64)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0.0 && return 0.0

    b1 = SpecialFunctions.besselkx(1, z)
    b2 = SpecialFunctions.besselkx(2, z)
    b3 = b1 + 4.0 / z * b2
    b4 = b2 + 6.0 / z * b3
    b5 = b3 + 8.0 / z * b4
    ex = exp(clamp(α - z, -700.0, 700.0))

    tauq = (DsT / (96.0 * π^2 * Tm^3)) * m^5 * ex * (2.0 * b1 - 3.0 * b3 + b5)
    norm = _fluidum_single_hadron_normalization(Tm, α, eos)
    abs(norm) <= TINY && return 0.0

    τn = tauq / norm * (fmGeV^2)
    if !isfinite(τn)
        return 0.0
    end
    return max(τn, 0.0) * tauD
end

@inline function _diff_tauN_impl(Tm::Float64, α::Float64, eos, DsT::Float64, tauD::Float64)
    m = hq_mass(eos)
    z = m / Tm
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

        # Asymptotic formula: τ ~ (DsT/48) * (m²/Tm²) with geometric mean regularization
        # This gives bounded behavior even as T → 0
        τ_GeVinv = (DsT / 48) * (m^2 / (Tm^2 + 1e-10))  # Regularized to avoid overflow
        return min((τ_GeVinv / fmGeV) * tauD, 1e20)  # Cap at very large but finite value
    end

    # For moderate z, use direct calculation of Bessel functions
    # (avoid forward recurrence which is unstable for large arguments)
    K1x = SpecialFunctions.besselkx(1, z)
    K2x = SpecialFunctions.besselkx(2, z)
    K3x = SpecialFunctions.besselkx(3, z)  # Direct calculation instead of recurrence
    K5x = SpecialFunctions.besselkx(5, z)  # Direct calculation instead of recurrence

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

    τ_GeVinv = (DsT / 48) * z3_over_Tm * ratio

    # Final safety check: ensure result is finite and reasonable
    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)  # Cap at large but finite value
    end

    return (τ_GeVinv / fmGeV) * tauD
end

@inline function diff_tauN(T, μ, model::IdealDiffViscModel)
    Tm = posden(T)
    DsT  = model.kappa_coeff
    tauD = model.tauN_coeff
    α = μ / Tm
    return _diff_tauN_impl(Tm, α, model.eos, DsT, tauD)
end

@inline diff_deltaNN(T, μ, model::IdealDiffViscModel) = model.deltaN_factor * diff_tauN(T, μ, model)

@inline function eps_from_len(len::Float64, dr::Float64; c::Float64=0.10)
    len <= 0 && return 0.0
    ϵ = c * (len/dr)^2
    return clamp(ϵ, 0.0, 0.24)
end

# ------------------------------------------------------------
# Density-frame charge diffusion (parabolic flux)
#
# In charge_mode=:density_frame there is no auxiliary ν^r field: μ is fixed by the
# on-slice charge density (the ν-less layout makes primitive recovery solve
# n u^τ = J^τ), and diffusion enters as the first-order-in-time parabolic flux
#
#   J^r_D = -κ (u^τ)^2 ∂_r α ,     κ = diff_kappa = DsT n / T ,
#
# derived from the heavy-quark Fokker–Planck equation in
# Tex/DensityFrame/df_fp_derivation.tex (Eq. dfflux).  We add τ J^r_D to the charge
# row of the face-flux array (work.Fh[iDtau, i] is the flux through the face between
# cells i and i+1), using a centred two-point gradient of α.  The cell-centred α, T,
# μ, n and rapidity y are already filled by the rhs! primitive-recovery loop.
# ------------------------------------------------------------
function add_density_frame_charge_flux!(work::Work1D, grid, τ::Float64, model::IdealDiffViscModel)
    L = layout(model)
    ng = grid.nghost
    Ntot = length(work.alpha)
    i0 = ng + 1
    iL = Ntot - ng
    invdr = 1.0 / grid.dr
    @inbounds for i in i0:iL
        (work.ok[i] && work.ok[i+1]) || continue
        dαdr = (work.alpha[i+1] - work.alpha[i]) * invdr
        Tf  = 0.5 * (exp(work.yT[i]) + exp(work.yT[i+1]))
        μf  = 0.5 * (work.mu[i]      + work.mu[i+1])
        nf  = 0.5 * (work.n[i]       + work.n[i+1])
        uτf = 0.5 * (cosh(work.y[i]) + cosh(work.y[i+1]))
        κf  = diff_kappa(Tf, μf, nf, model)
        JrD = -κf * uτf^2 * dαdr
        work.Fh[L.iDtau, i] += τ * JrD
    end
    return nothing
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
    work.theta[i0] = safe_inv(τ) + 2*work.vF[i0]/grid.dr

    Threads.@threads for i in (i0+1):(Ntot-ng)
        rC  = posden(grid.rC[i])
        rRp = grid.rF[i+1]
        rRm = grid.rF[i]
        div = (rRp*work.vF[i] - rRm*work.vF[i-1]) / grid.dr
        work.theta[i] = safe_inv(τ) + div / rC
    end
    return nothing
end

# ------------------------------------------------------------
# Proper 4-divergence θ = ∇_μ u^μ for MIS relaxation
#
# We use u^μ = (u^τ, u^r, 0, 0) with
#   u^r = ur = sinh(y),   u^τ = cosh(y)
# and approximate (dropping explicit ∂_τ u^τ term during the relaxation substep):
#   θ ≈ u^τ/τ + (1/r) ∂_r (r u^r)
#
# This is distinct from `compute_theta!`, which is used as a geometric divergence
# of the 3-velocity v for conservative advection source terms.
#
# Side effect: stores face-centered u^r in work.vF (scratch) and θ in work.theta.
# ------------------------------------------------------------
function compute_theta_u!(work::Work1D, grid, τ)
    ng = grid.nghost
    Ntot = length(work.y)
    fill!(work.theta, 0.0)
    fill!(work.vF, 0.0)

    @inbounds for i in 1:(Ntot-1)
        urL = sinh(work.y[i])
        urR = sinh(work.y[i+1])
        work.vF[i] = 0.5 * (urL + urR)
    end

    invτ = safe_inv(τ)
    i0 = ng + 1
    uτ0 = cosh(work.y[i0])
    work.theta[i0] = uτ0*invτ + 2*work.vF[i0]/grid.dr

    Threads.@threads for i in (i0+1):(Ntot-ng)
        uτ = cosh(work.y[i])
        rC  = posden(grid.rC[i])
        rRp = grid.rF[i+1]
        rRm = grid.rF[i]
        div = (rRp*work.vF[i] - rRm*work.vF[i-1]) / grid.dr
        work.theta[i] = uτ*invτ + div / rC
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

# Uses face-centered u^r stored in work.vF (by compute_theta_u!) to build ∂_r u^r.
function compute_durdr!(work::Work1D, grid)
    ng = grid.nghost
    Ntot = length(work.y)
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
    compute_field_grad_fv!(work.alphaF, work.gradAlpha, work.alpha, grid; limiter=limiter)
end

function compute_field_grad_fv!(face::AbstractVector, grad::AbstractVector, field::AbstractVector, grid; limiter=minmod)
    ng = grid.nghost
    Ntot = length(field)
    Nf   = Ntot - 1
    fill!(face, 0.0)
    fill!(grad, 0.0)

    slope = grad
    fill!(slope, 0.0)

    @inbounds for i in (ng+2):(Ntot-ng-1)
        dL = field[i]   - field[i-1]
        dR = field[i+1] - field[i]
        slope[i] = limiter(dL, dR)
    end

    @inbounds for i in 1:Nf
        if i <= (ng+1) || i >= (Ntot-ng-1)
            face[i] = 0.5*(field[i] + field[i+1])
        else
            fL = field[i]   + 0.5*slope[i]
            fR = field[i+1] - 0.5*slope[i+1]
            face[i] = 0.5*(fL + fR)
        end
    end

    i0 = ng + 1
    grad[i0] = (face[i0] - face[i0-1]) / grid.dr
    Threads.@threads for i in (i0+1):(Ntot-ng)
        grad[i] = (face[i] - face[i-1]) / grid.dr
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

@inline function _clip_shear_pair(pr::Float64, pe::Float64, pimax::Float64)
    pp = -(pr + pe)
    maxabs = max(abs(pr), abs(pe), abs(pp))
    if !(isfinite(pimax) && pimax > 0.0) || !isfinite(maxabs)
        return 0.0, 0.0, 0.0, true, Inf
    end
    if maxabs <= pimax
        return pr, pe, pp, false, maxabs / (pimax + TINY)
    end
    fac = pimax / (maxabs + TINY)
    pr_new = pr * fac
    pe_new = pe * fac
    pp_new = -(pr_new + pe_new)
    return pr_new, pe_new, pp_new, true, maxabs / (pimax + TINY)
end

# ------------------------------------------------------------
# Dissipative relaxation: ν_r + Π + π (semi-implicit BE)
# ------------------------------------------------------------
function relax_dissipative!(U, grid, τ, Δ, model::IdealDiffViscModel, work::Work1D;
                            diag::Union{Nothing,DiagCounters}=nothing)
    L = model.layout
    Ntot = size(U,2)

    # Cache previous-step rapidity so we can approximate D0 u^r = ∂_τ u^r
    # during this relaxation substep.
    copyto!(work.y_prev, work.y)
    # NB: work.alpha_prev is intentionally NOT refreshed here. It must hold the PREVIOUS timestep's α
    # so the covariant drive's temporal piece ∂τα = (α − α_prev)/Δ is nonzero. Copying it at entry
    # (as was done) made α_prev ≡ α ⇒ ∂τα ≡ 0 ⇒ ν^r ≈2× under-driven vs Fluidum. It is now stored at
    # the END of the charm-diffusion block below.

    if !( (model.enable_diff && L.hasNur) || 
          (model.enable_shear && (L.hasPiR || L.hasPiEta)) || 
          (model.enable_bulk  && L.hasPi) )
        return nothing
    end

    # Recover primitives everywhere (and build vC, alpha, etc.)
    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D   = safe_div(U[L.iDtau, i], τ)
        Sr  = U[L.iSr,i]
        E   = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

        yT0 = work.yT[i]
        φ0  = work.phi[i]
        y0  = work.y[i]

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, grid.rC[i], τ, model.eos;
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
            work.vC[i]    = safe_div(ur, uτ)
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

        # Drive field for ν_NS target.
        drive = model.diffusion_drive
        if drive === :alpha
            smooth_alpha!(work, grid, epsα)
            @inbounds for g in 1:ng
                il = ng + 1 - g
                ir = ng + g
                work.alpha[il] = work.alpha[ir]
            end
            compute_field_grad_fv!(work.alphaF, work.gradAlpha, work.alpha, grid; limiter=mc_limiter)
        elseif drive === :n
            # Build the density gradient and the temperature gradient separately.
            # The actual thermodynamic force is the density gradient at fixed α=μ/T:
            #   (∂r n)|α = ∂r n - (∂n/∂T)|α ∂r T.
            # Using raw ∂r n alone can flip the sign when the temperature profile
            # dominates, which is exactly what happens in the current ICs.
            @inbounds for g in 1:ng
                il = ng + 1 - g
                ir = ng + g
                work.n[il] = work.n[ir]
            end
            compute_field_grad_fv!(work.alphaF, work.n_tmp, work.n, grid; limiter=mc_limiter)

            @inbounds for i in 1:Ntot
                work.alpha_tmp[i] = exp(work.yT[i])
            end
            @inbounds for g in 1:ng
                il = ng + 1 - g
                ir = ng + g
                work.alpha_tmp[il] = work.alpha_tmp[ir]
            end
            compute_field_grad_fv!(work.alphaF, work.gradAlpha, work.alpha_tmp, grid; limiter=mc_limiter)
        else
            throw(ArgumentError("Unknown diffusion_drive=$drive. Use :alpha or :n"))
        end

        compute_theta_u!(work, grid, τ)
        compute_durdr!(work, grid)

        @inbounds for i in 1:Ntot
            work.nur_tmp[i] = phys_from_stored(U[L.iNur,i])
        end

        Threads.@threads for i in (ng+1):(Ntot-ng)
            work.ok[i] || continue

            ur = sinh(work.y[i])
            uτ = sqrt(1 + ur^2)
            v  = safe_div(ur, uτ)

            θ  = work.theta[i]

            # Full expansion scalar for second-order terms (include ∂τ u^τ).
            ur_prev = sinh(work.y_prev[i])
            durdτ = safe_div((ur - ur_prev), Δ)
            duτdτ = safe_div(ur, uτ) * durdτ
            θ_full = θ + duτdτ

            # Diagonal shear rate (mixed rr) used in ν⋅σ coupling.
            durdr = work.dvdr[i]
            σr = durdr - θ_full / 3

            # Projected comoving derivative term from
            #   Δ^r{}_ν (u·∇) ν^ν
            # with ν^τ = (u^r/u^τ) ν^r. In 1+1D this reduces to
            #   (u·∇)ν^r - v ν^r (u·∇)y,
            # where y is the radial flow rapidity.
            Dy = safe_div(durdτ + ur * durdr, uτ)

            T  = exp(work.yT[i])
            μ  = work.mu[i]
            n  = work.n[i]
            α  = work.alpha[i]

            DsT = model.kappa_coeff
            κ = safe_div(DsT, T) * max(n, 0.0) / fmGeV
            D = safe_div(DsT, T) / fmGeV

            τn = diff_tauN(T, μ, model)
            δ  = diff_deltaNN(T, μ, model)
            λNN = model.lambda_NN_factor * τn

            # Navier–Stokes target.
            # :alpha -> thermodynamic driving with the FULL covariant projector:
            #            ν_NS = -κ ∇^⟨r⟩α = -κ [uτ² ∂rα + u^r u^τ ∂τα]
            #          The temporal piece is part of Fluidum's covariant gradient and is NOT
            #          negligible on a cooling background (∂τα ~ m|Ṫ|/T² ~ O(1)/fm); without it the
            #          FiVo first-order current is ≈2× under-driven vs Fluidum's IS2 charm (Pb+Pb
            #          hydro-comparison). Restores the BIGRUN-2 Fluidum↔FiVo agreement (3–10%).
            # :n     -> density-based form:    ν_NS = -D uτ^2 [∂r n - (∂n/∂T)|α ∂r T]
            νNS_raw = if drive === :alpha
                αp = work.alpha_prev[i]
                # Guard: skip ∂τα on the very first substep (alpha_prev unprimed: NaN/0.0).
                dαdτ = (isfinite(αp) && αp != 0.0) ? safe_div(α - αp, Δ) : 0.0
                -κ * ((uτ^2) * work.gradAlpha[i] + ur * uτ * dαdτ)
            else
                dn_dr = work.n_tmp[i]
                dT_dr = work.gradAlpha[i]
                dn_dT_alpha = diff_dn_dT_at_fixed_alpha(T, α, model)
                dn_corr = dn_dr - dn_dT_alpha * dT_dr
                -D * (uτ^2) * dn_corr
            end
            νNS_phys = model.do_soft_project_nur ? _soft_project_nur_phys(νNS_raw, n, uτ) : νNS_raw

            # Projected relaxation equation for the radial component:
            #   τn Δ^r{}_ν (u·∇) ν^ν + ν^r = ν_NS^r
            # which becomes
            #   τn[(u^τ∂τ + u^r∂r)ν^r - v ν^r (u·∇)y] + ν^r = ν_NS^r,
            # plus the optional linear MIS couplings retained below.
            # Discretize ∂τ with backward-Euler in ν and upwind ∂r explicitly.
            νold = work.nur_tmp[i]

            dν_dr = 0.0
            if model.relax_advect_nur
                if ur >= 0.0
                    dν_dr = (νold - work.nur_tmp[i-1]) / grid.dr
                else
                    dν_dr = (work.nur_tmp[i+1] - νold) / grid.dr
                end
            end

            A = safe_div((τn * uτ), Δ)

            # Second-order linear damping terms (set factors to 0.0 to disable):
            #   +δ ν θ + λNN ν σr
            # The projected derivative contributes an additional linear term
            #   -τn v (u·∇)y ν,
            # which is part of the requested covariant equation.
            damp = δ * θ_full + λNN * σr - τn * v * Dy

            denom = A + 1 + damp
            denom = max(denom, 1e-12)
            νnew_phys = (A * νold - τn * ur * dν_dr + νNS_phys) / denom

            # Keep ν within the causal domain so (D,S,E,ν) remains solvable.
            if model.do_soft_project_nur
                νnew_phys = _soft_project_nur_phys(νnew_phys, n, uτ)
            end

            U[L.iNur,i] = stored_from_phys(νnew_phys)
        end

        if model.do_axis_project_nur
            axis_project_nur_tapered!(U, grid, model; nfit=model.axis_project_nfit)
        end

        smooth_field_centered!(U, L.iNur, work.nur_tmp, grid, epsν)

        # Soft causality enforcement after post-processing (axis projection + smoothing).
        if model.do_soft_project_nur
            Threads.@threads for i in (ng+1):(Ntot-ng)
                work.ok[i] || continue
                ur = sinh(work.y[i])
                uτ = sqrt(1 + ur^2)
                νphys = phys_from_stored(U[L.iNur,i])
                νphys = _soft_project_nur_phys(νphys, work.n[i], uτ)
                U[L.iNur,i] = stored_from_phys(νphys)
            end
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

        # Persist this step's (smoothed) α so the NEXT step's covariant drive has a true
        # temporal piece ∂τα = (α − α_prev)/Δ (restores ~Fluidum-level ν^r; see entry note).
        copyto!(work.alpha_prev, work.alpha)
    end

    # ---- bulk + shear ----
    if (model.enable_shear && (L.hasPiR || L.hasPiEta)) || (model.enable_bulk && L.hasPi)
        epsV = clamp(model.visc_filter_eps + eps_from_len(model.visc_smooth_len, grid.dr), 0.0, 0.24)

        compute_theta_u!(work, grid, τ)
        compute_durdr!(work, grid)

        if L.hasPi && model.relax_advect_Pi
            @inbounds for i in 1:Ntot
                work.visc_tmp1[i] = phys_from_stored(U[L.iPi,i])
            end
        end
        if model.enable_shear
            # We evolve the mixed diagonal components (π^φ_φ, π^η_η)
            # and reconstruct π^r_r by tracelessness.
            @inbounds for i in 1:Ntot
                pr = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
                pe = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
                work.visc_tmp2[i] = -(pr + pe)  # π^φ_φ
                work.visc_tmp3[i] = pe          # π^η_η
            end
        end

        Threads.@threads for i in (ng+1):(Ntot-ng)
            work.ok[i] || continue

            dur  = work.dvdr[i]
            θ    = work.theta[i]
            invτ = safe_inv(τ)

            ur = sinh(work.y[i])
            uτ = sqrt(1 + ur^2)

            # Full expansion scalar θ = ∇_μ u^μ.
            # `compute_theta_u!` drops ∂τ u^τ during the relaxation substep, so we add
            # it back using a backward difference for full/second-order terms (and bulk).
            ur_prev = sinh(work.y_prev[i])
            durdτ = safe_div((ur - ur_prev), Δ)
            duτdτ = safe_div(ur, uτ) * durdτ
            θ_full = θ + duτdτ

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
            else
                 ζ, τΠ = 0.0, 1.0
            end

            ΠNS_phys = -ζ * θ_full

            δΠ = visc_deltaPi(T, μ, n, e, P, model)
            λΠπ = model.lambda_Pi_pi_factor * τΠ
            τππ = model.taupi_pi_factor * τπ
            λπΠ = model.lambda_pi_Pi_factor * τπ

            # Build NS shear as a full contravariant tensor Π_NS^{μν}, then extract the
            # stored shear dofs from it. This single-sources the tensor structure.
            # Use θ_full (including ∂_τ u^τ) for consistency with bulk and shear blocks.
            ΠNS = shear_NS_target_contravariant(ur, uτ, grid.rC[i], τ, η, θ_full, dur)
            piRNS_phys, piEtaNS_phys = shear_dofs_from_contravariant(ur, uτ, grid.rC[i], τ, ΠNS)

            v = safe_div(ur, uτ)

            # --- Advection source terms (upwind) ---
            dΠ_dr = 0.0
            if model.enable_bulk && L.hasPi
                Πold = work.visc_tmp1[i]
                if model.relax_advect_Pi
                    dΠ_dr = (ur >= 0.0) ? (Πold - work.visc_tmp1[i-1]) / grid.dr : (work.visc_tmp1[i+1] - Πold) / grid.dr
                end
            end

            # Shear is evolved in PDE form; spatial derivatives are computed below when enabled.

            if model.enable_bulk && L.hasPi
                Πold = work.visc_tmp1[i]
                A = safe_div((τΠ * uτ), Δ)

                # Bulk/shear coupling term: +λΠπ (π:σ)
                πσ = 0.0
                if model.enable_shear && λΠπ != 0.0
                    pr = (L.hasPiR   ? phys_from_stored(U[L.iPiR,i])   : 0.0)
                    pe = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
                    pp = -(pr + pe)

                    rC = posden(grid.rC[i])
                    σr_loc = dur - θ_full / 3
                    σφ_loc = ur / rC - θ_full / 3
                    ση_loc = uτ * invτ - θ_full / 3
                    πσ = pr * σr_loc + pp * σφ_loc + pe * ση_loc
                end

                denom = A + 1 + (δΠ * θ_full)
                denom = max(denom, 1e-12)
                Πnew_phys = (A * Πold - τΠ * ur * dΠ_dr + ΠNS_phys + (λΠπ * πσ)) / denom

                U[L.iPi,i] = stored_from_phys(Πnew_phys)
            end

            if model.enable_shear
                # Israel–Stewart PDE form (with D0=∂τ, D1=∂r):
                #   τπ (u^τ ∂τ + u^r ∂r) π + π = π_NS
                # We discretize ∂τ with backward-Euler in π and upwind ∂r explicitly.

                # Mixed diagonal shear components:
                pp_old = work.visc_tmp2[i]   # π^φ_φ
                pe_old = work.visc_tmp3[i]   # π^η_η

                # Use the already-computed θ_full (from above) for consistency
                # with the bulk block and the contravariant NS target.
                rC = posden(grid.rC[i])

                # Mixed diagonal shear rates (Milne+cylindrical, 1D + boost-invariant):
                σφ = ur / rC - θ_full / 3
                ση = uτ * invτ - θ_full / 3

                pp_NS = -2η * σφ
                pe_NS = -2η * ση

                # Upwind spatial derivatives for the comoving term u^r ∂r π.
                dpp_dr = 0.0
                dpe_dr = 0.0
                if model.relax_advect_pi
                    if ur >= 0.0
                        dpp_dr = (pp_old - work.visc_tmp2[i-1]) / grid.dr
                        dpe_dr = (pe_old - work.visc_tmp3[i-1]) / grid.dr
                    else
                        dpp_dr = (work.visc_tmp2[i+1] - pp_old) / grid.dr
                        dpe_dr = (work.visc_tmp3[i+1] - pe_old) / grid.dr
                    end
                end

                A = safe_div((τπ * uτ), Δ)

                Πphys = (model.enable_bulk && L.hasPi) ? phys_from_stored(U[L.iPi,i]) : 0.0

                denom_pp = A + 1 + δπ * θ_full + τππ * σφ
                denom_pe = A + 1 + δπ * θ_full + τππ * ση
                denom_pp = max(denom_pp, 1e-12)
                denom_pe = max(denom_pe, 1e-12)

                pp_new = (A * pp_old - τπ * ur * dpp_dr + pp_NS + λπΠ * Πphys * σφ) / denom_pp
                pe_new = (A * pe_old - τπ * ur * dpe_dr + pe_NS + λπΠ * Πphys * ση) / denom_pe
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

            # If diagnostics are requested, do a deterministic serial pass so we can
            # count and locate clipping without thread-race bookkeeping.
            if diag !== nothing
                PiClip = 0
                PiClipMax = 0.0
                PiClipMaxI = 0
                piClip = 0
                piClipMax = 0.0
                piClipMaxI = 0

                @inbounds for i in (ng+1):(Ntot-ng)
                    work.ok[i] || continue
                    cap = max(work.P[i] + work.e[i], 0.0)
                    cap <= 0 && continue

                    if do_Πclip && model.enable_bulk && L.hasPi
                        Πmax = Πfac * cap
                        Πold = phys_from_stored(U[L.iPi,i])
                        Πnew = clamp(Πold, -Πmax, Πmax)
                        if Πnew != Πold
                            PiClip += 1
                            ratio = abs(Πold) / (Πmax + TINY)
                            if ratio > PiClipMax
                                PiClipMax = ratio
                                PiClipMaxI = i
                            end
                            U[L.iPi,i] = stored_from_phys(Πnew)
                        end
                    end

                    if do_πclip && model.enable_shear
                        pimax = πfac * cap
                        pr_old = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
                        pe_old = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
                        pr_new, pe_new, _, clipped, ratio = _clip_shear_pair(pr_old, pe_old, pimax)
                        if clipped
                            piClip += 1
                            if ratio > piClipMax
                                piClipMax = ratio
                                piClipMaxI = i
                            end
                            if L.hasPiR
                                U[L.iPiR,i] = stored_from_phys(pr_new)
                            end
                            if L.hasPiEta
                                U[L.iPiEta,i] = stored_from_phys(pe_new)
                            end
                        end
                    end
                end

                diag_add!(diag; PiClip=PiClip, PiClipMax=PiClipMax, PiClipMaxI=PiClipMaxI,
                                piClip=piClip, piClipMax=piClipMax, piClipMaxI=piClipMaxI)
            else
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
                        pr = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
                        pe = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)
                        pr, pe, _, _, _ = _clip_shear_pair(pr, pe, pimax)
                        if L.hasPiR
                            U[L.iPiR,i] = stored_from_phys(pr)
                        end
                        if L.hasPiEta
                            U[L.iPiEta,i] = stored_from_phys(pe)
                        end
                    end
                end
            end
        end
    end

    return nothing
end

