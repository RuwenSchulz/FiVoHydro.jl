# ------------------------------------------------------------
# Admissibility (MOOD)
# ------------------------------------------------------------
@inline function _sr_bound_max(E::Float64,
                              Pguess::Float64,
                              Pi_phys::Float64,
                              piR_phys::Float64,
                              piEta_phys::Float64,
                              χ::Float64)
    Peff = max(Pguess, 0.0)
    if isfinite(Pi_phys)
        Peff += abs(Pi_phys)
    end
    if isfinite(piR_phys)
        Peff += abs(piR_phys)
    end
    if isfinite(piEta_phys)
        Peff += abs(piEta_phys)
    end
    return χ * (E + Peff)
end

@inline function _update_work_cell_from_U!(work::Work1D, U::AbstractMatrix, i::Int, r::Float64, τ::Float64, model::IdealDiffViscModel)
    L   = layout(model)
    eos = model.eos

    D  = safe_div(U[L.iDtau, i], τ)
    Sr = U[L.iSr,i]
    E  = U[L.iE,i]

    nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
    Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
    piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
    piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

    # Use cached Newton guess stored in work.
    tid = Threads.threadid()
    wpr = model.primrec.work[tid]
    T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
        wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, r, τ, eos;
        yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
        maxit=80
    )

    if ok
        Tm = max(T, T_MIN)
        φ  = clamp((μ - hq_mass(eos)) / Tm, -PHI_CAP, PHI_CAP)

        work.yT[i]    = log(Tm)
        work.phi[i]   = φ
        work.mu[i]    = μ
        work.alpha[i] = μ / Tm
        work.y[i]     = asinh(ur)

        work.P[i]  = P
        work.n[i]  = n
        work.e[i]  = e
        work.ok[i] = true

        uτ = sqrt(1 + ur^2)
        work.vC[i] = safe_div(ur, uτ)

        work.x0_yT[i]  = work.yT[i]
        work.x0_phi[i] = work.phi[i]
        work.x0_y[i]   = work.y[i]
    else
        # Consistent vacuum fallback.
        Tv = T_MIN
        μv = 0.0
        Pv, _, ev = eos_Pne(Tv, μv, eos)

        work.yT[i]    = log(Tv)
        work.phi[i]   = 0.0
        work.mu[i]    = μv
        work.alpha[i] = 0.0
        work.y[i]     = 0.0

        work.P[i]  = Pv
        work.n[i]  = 0.0
        work.e[i]  = max(ev, 0.0)
        work.ok[i] = true
        work.vC[i] = 0.0

        work.x0_yT[i]  = work.yT[i]
        work.x0_phi[i] = 0.0
        work.x0_y[i]   = 0.0
    end
    return nothing
end

function _repair_Dtau_positivity!(Unew::AbstractMatrix,
                                  Ubase::AbstractMatrix,
                                  grid,
                                  model::IdealDiffViscModel;
                                  mask::Union{Nothing,BitVector}=nothing)
    L = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(Unew, 2) - ng
    Nvars = size(Unew, 1)

    modified_any = false

    @inbounds for i in i0:iL
        if mask !== nothing && !mask[i]
            continue
        end

        D1 = Unew[L.iDtau, i]
        if isfinite(D1) && D1 >= 0.0
            continue
        end

        D0 = Ubase[L.iDtau, i]
        if !(isfinite(D0) && D0 >= 0.0)
            for a in 1:Nvars
                Unew[a, i] = Ubase[a, i]
            end
            if !(isfinite(Unew[L.iDtau, i]) && Unew[L.iDtau, i] >= 0.0)
                Unew[L.iDtau, i] = 0.0
            end
            modified_any = true
            continue
        end

        if !isfinite(D1)
            for a in 1:Nvars
                Unew[a, i] = Ubase[a, i]
            end
            modified_any = true
            continue
        end

        θ = clamp(D0 / (D0 - D1 + TINY), 0.0, 1.0)
        for a in 1:Nvars
            Unew[a, i] = Ubase[a, i] + θ * (Unew[a, i] - Ubase[a, i])
        end

        if Unew[L.iDtau, i] < 0.0
            Unew[L.iDtau, i] = (Unew[L.iDtau, i] > -100 * TINY) ? 0.0 : max(Unew[L.iDtau, i], 0.0)
        end
        modified_any = true
    end

    return modified_any
end

function _repair_theta_admissibility!(Unew::AbstractMatrix,
                                     Ubase::AbstractMatrix,
                                     grid,
                                     τ::Float64,
                                     model::IdealDiffViscModel,
                                     work::Work1D;
                                     Emin::Float64=E_FLOOR,
                                     χ::Float64=χ_SrE,
                                     mask::Union{Nothing,BitVector}=nothing,
                                     diag::Union{Nothing,DiagCounters}=nothing)
    hydro_flags().repair_theta || return false

    L = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(Unew,2) - ng
    Nvars = size(Unew,1)

    # If we modify anything, we update work caches for those cells.
    modified_any = false
    scaled = 0
    maxratio = 0.0
    maxi = 0

    @inbounds for i in i0:iL
        if mask !== nothing && !mask[i]
            continue
        end

        # Base state is assumed admissible (it came from a previous accepted stage).
        E1  = Unew[L.iE,i]
        Sr1 = Unew[L.iSr,i]
        if !(isfinite(E1) && isfinite(Sr1))
            continue
        end
        if E1 < Emin
            # Floors should have handled this; fall back to base.
            for a in 1:Nvars
                Unew[a,i] = Ubase[a,i]
            end
            modified_any = true
            _update_work_cell_from_U!(work, Unew, i, grid.rC[i], τ, model)
            continue
        end

        Pi1   = (L.hasPi    ? phys_from_stored(Unew[L.iPi,i])    : 0.0)
        piR1  = (L.hasPiR   ? phys_from_stored(Unew[L.iPiR,i])   : 0.0)
        piEta1 = (L.hasPiEta ? phys_from_stored(Unew[L.iPiEta,i]) : 0.0)

        # Conservative quick check (P=0) to decide whether we need to do anything.
        χeff = sr_margin() * χ
        Srmax_quick = _sr_bound_max(E1, 0.0, Pi1, piR1, piEta1, χeff)
        if abs(Sr1) <= Srmax_quick
            continue
        end

        # Use actual primitive recovery (pressure) at θ=1 to decide whether this cell
        # really violates the bound (important with shear/bulk where P is not well
        # approximated by interpolation).
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        Dtau1 = Unew[L.iDtau,i]
        nur1  = (L.hasNur ? phys_from_stored(Unew[L.iNur,i]) : 0.0)

        T1, μ1, ur1, n1, e1, P1, okP1 = cons_to_prim_ideal_phi_diff_visc!(
            wpr, safe_div(Dtau1, τ), Sr1, E1, nur1, Pi1, piR1, piEta1, grid.rC[i], τ, model.eos;
            yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
            maxit=40
        )
        P1 = (okP1 && isfinite(P1)) ? max(P1, 0.0) : 0.0

        Srmax1 = _sr_bound_max(E1, P1, Pi1, piR1, piEta1, χeff)
        if !(isfinite(Srmax1) && Srmax1 > 0.0)
            # Degenerate: revert.
            for a in 1:Nvars
                Unew[a,i] = Ubase[a,i]
            end
            modified_any = true
            _update_work_cell_from_U!(work, Unew, i, grid.rC[i], τ, model)
            continue
        end
        if abs(Sr1) <= Srmax1
            # It's actually fine: update cache (so v/P are consistent for later checks).
            _update_work_cell_from_U!(work, Unew, i, grid.rC[i], τ, model)
            continue
        end

        # Now we do θ-bisection between base and new state.
        Dtau0 = Ubase[L.iDtau,i]
        Sr0   = Ubase[L.iSr,i]
        E0    = Ubase[L.iE,i]
        nur0  = (L.hasNur ? phys_from_stored(Ubase[L.iNur,i]) : 0.0)

        Pi0    = (L.hasPi    ? phys_from_stored(Ubase[L.iPi,i])    : 0.0)
        piR0   = (L.hasPiR   ? phys_from_stored(Ubase[L.iPiR,i])   : 0.0)
        piEta0 = (L.hasPiEta ? phys_from_stored(Ubase[L.iPiEta,i]) : 0.0)

        # Ensure base is at least floor-admissible.
        if E0 < Emin
            E0 = Emin
        end

        # For diagnostics: how far over the bound θ=1 is.
        ratio_pre = abs(Sr1) / max(Srmax1, 1e-300)

        # Check function using *actual* primitive recovery for the interpolated state.
        @inline function ok_theta(θ::Float64)
            Dtauθ = Dtau0 + θ*(Dtau1 - Dtau0)
            Eθ  = E0  + θ*(E1  - E0)
            Srθ = Sr0 + θ*(Sr1 - Sr0)
            if !(isfinite(Eθ) && isfinite(Srθ))
                return false
            end
            if Eθ < Emin
                return false
            end

            Piθ    = Pi0    + θ*(Pi1    - Pi0)
            piRθ   = piR0   + θ*(piR1   - piR0)
            piEtaθ = piEta0 + θ*(piEta1 - piEta0)
            nurθ   = nur0   + θ*(nur1   - nur0)

            # Primrec for pressure
            Tθ, μθ, urθ, nθ, eθ, Pθ, okP = cons_to_prim_ideal_phi_diff_visc!(
                wpr, safe_div(Dtauθ, τ), Srθ, Eθ, nurθ, Piθ, piRθ, piEtaθ, grid.rC[i], τ, model.eos;
                yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
                maxit=40
            )
            Pθ = (okP && isfinite(Pθ)) ? max(Pθ, 0.0) : 0.0

            Srmaxθ = _sr_bound_max(Eθ, Pθ, Piθ, piRθ, piEtaθ, χeff)
            return (isfinite(Srmaxθ) && Srmaxθ > 0.0 && abs(Srθ) <= Srmaxθ)
        end

        # If even θ=0 is bad (shouldn't happen), hard reset to base anyway.
        if !ok_theta(0.0)
            for a in 1:Nvars
                Unew[a,i] = Ubase[a,i]
            end
            modified_any = true
            scaled += 1
            if ratio_pre > maxratio
                maxratio = ratio_pre
                maxi = i
            end
            _update_work_cell_from_U!(work, Unew, i, grid.rC[i], τ, model)
            continue
        end

        # Bisection for the largest admissible θ.
        lo = 0.0
        hi = 1.0
        for _ in 1:18
            mid = 0.5*(lo + hi)
            if ok_theta(mid)
                lo = mid
            else
                hi = mid
            end
        end
        θ = lo

        # Apply blend.
        for a in 1:Nvars
            Unew[a,i] = Ubase[a,i] + θ*(Unew[a,i] - Ubase[a,i])
        end
        modified_any = true

        # Update work cache for this cell so MOOD sees consistent v/P.
        _update_work_cell_from_U!(work, Unew, i, grid.rC[i], τ, model)

        scaled += 1
        if ratio_pre > maxratio
            maxratio = ratio_pre
            maxi = i
        end
    end

    if diag !== nothing && scaled > 0
        diag_add!(diag; srscaled=scaled, srscaled_maxratio=maxratio, srscaled_max_i=maxi)
    end
    return modified_any
end

@inline function admissible_state_fast(U::AbstractMatrix, grid, i::Int, τ::Float64, model::IdealDiffViscModel,
                                       work::Union{Nothing,Work1D}=nothing;
                                       Emin::Float64=E_FLOOR,
                                       χ::Float64=χ_SrE,
                                       v_eps::Float64=1e-12)
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

    # If a work cache is available (it is in the time stepper), use it to
    # reject near-/superluminal states even when the cheap Sr/E gate would
    # skip primitive recovery. This matters especially with shear enabled,
    # where stresses can make the mapping from (Dtau,Sr,E) to v ill-conditioned.
    if work !== nothing && work.ok[i]
        v = work.vC[i]
        if !(isfinite(v) && abs(v) < (1 - v_eps))
            return false
        end
    end

    Pguess = 0.0
    if work !== nothing && work.ok[i]
        Ptmp = work.P[i]
        if isfinite(Ptmp)
            Pguess = max(Ptmp, 0.0)
        end
    end
    Π  = (L.hasPi    ? abs(phys_from_stored(U[L.iPi,i]))    : 0.0)
    pr = (L.hasPiR   ? abs(phys_from_stored(U[L.iPiR,i]))   : 0.0)
    pe = (L.hasPiEta ? abs(phys_from_stored(U[L.iPiEta,i])) : 0.0)
    χeff = sr_margin() * χ
    Srmax = χeff*(E + Pguess + Π + pr + pe)
    if abs(Sr) <= Srmax
        return true
    end
    if E < 100Emin
        return true
    end

    prim = cons_to_prim_col(U, i, grid.rC[i], τ, model)
    return prim.ok
end

function any_bad(U, grid, τ, model::IdealDiffViscModel, work; Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    bad_tls = work.bad_tls
    fill!(bad_tls, false)
    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        tid = Threads.threadid()
        if !admissible_state_fast(U, grid, i, τ, model, work; Emin=Emin, χ=χ)
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

function mark_bad!(bad, U, grid, τ, model::IdealDiffViscModel, work::Work1D;
                   Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    ng = grid.nghost
    fill!(bad, false)
    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        bad[i] = !admissible_state_fast(U, grid, i, τ, model, work; Emin=Emin, χ=χ)
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
