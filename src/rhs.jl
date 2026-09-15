# ------------------------------------------------------------
# RHS
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

    primfail_i_tls = work.primfail_i_tls
    fill!(primfail_i_tls, 0)
    primfail_tau_tls = work.primfail_tau_tls
    primfail_Dtau_tls = work.primfail_Dtau_tls
    primfail_Sr_tls = work.primfail_Sr_tls
    primfail_E_tls = work.primfail_E_tls
    primfail_nur_stored_tls = work.primfail_nur_stored_tls
    primfail_Pi_stored_tls = work.primfail_Pi_stored_tls
    primfail_piR_stored_tls = work.primfail_piR_stored_tls
    primfail_piEta_stored_tls = work.primfail_piEta_stored_tls
    primfail_reason_tls = work.primfail_reason_tls
    primfail_iters_tls = work.primfail_iters_tls
    primfail_resnorm_tls = work.primfail_resnorm_tls

    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D  = safe_div(U[L.iDtau,i], τ)
        Sr = U[L.iSr,i]
        E  = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys, grid.rC[i], τ, eos;
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

            work.P[i]     = P
            work.n[i]     = n
            work.e[i]     = e
            work.ok[i]    = true

            uτ = sqrt(1 + ur^2)
            work.vC[i] = safe_div(ur, uτ)

            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = work.phi[i]
            work.x0_y[i]   = work.y[i]
        else
            primfail_tls[tid] += 1

            if diag !== nothing && primfail_i_tls[tid] == 0
                # Record one failing cell per thread (race-free), reduced below.
                primfail_i_tls[tid] = i
                primfail_tau_tls[tid] = τ
                primfail_Dtau_tls[tid] = U[L.iDtau,i]
                primfail_Sr_tls[tid] = U[L.iSr,i]
                primfail_E_tls[tid] = U[L.iE,i]
                primfail_nur_stored_tls[tid] = (L.hasNur ? U[L.iNur,i] : 0.0)
                primfail_Pi_stored_tls[tid] = (L.hasPi ? U[L.iPi,i] : 0.0)
                primfail_piR_stored_tls[tid] = (L.hasPiR ? U[L.iPiR,i] : 0.0)
                primfail_piEta_stored_tls[tid] = (L.hasPiEta ? U[L.iPiEta,i] : 0.0)
                primfail_reason_tls[tid] = wpr.last_reason
                primfail_iters_tls[tid] = wpr.last_iters
                primfail_resnorm_tls[tid] = wpr.last_resnorm
            end

            Tv = T_MIN
            μv = 0.0
            Pv, _, ev = eos_Pne(Tv, μv, eos)

            work.yT[i]    = log(Tv)
            work.phi[i]   = 0.0
            work.mu[i]    = μv
            work.alpha[i] = 0.0#μv / Tv
            work.y[i]     = 0.0

            work.P[i]     = Pv
            work.n[i]     = 0.0
            work.e[i]     = max(ev, 0.0)
            work.ok[i]    = true

            work.vC[i]    = 0.0

            # Reset Newton initial guess caches to safe values.
            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = 0.0
            work.x0_y[i]   = 0.0
        end
    end

    _debug_check_shear_conserved!(U, grid, τ, model, work; diag=diag)

    if diag !== nothing
        diag_add!(diag; primfail=sum(primfail_tls))

        # Deterministic representative: choose the smallest failing i across threads.
        i_min = 0
        tid_min = 0
        @inbounds for tid in eachindex(primfail_i_tls)
            i = primfail_i_tls[tid]
            if i != 0 && (i_min == 0 || i < i_min)
                i_min = i
                tid_min = tid
            end
        end

        if i_min != 0
            @inbounds begin
                diag.last_prim_fail_i = primfail_i_tls[tid_min]
                diag.last_prim_fail_tau = primfail_tau_tls[tid_min]
                diag.last_prim_fail_Dtau = primfail_Dtau_tls[tid_min]
                diag.last_prim_fail_Sr = primfail_Sr_tls[tid_min]
                diag.last_prim_fail_E = primfail_E_tls[tid_min]
                diag.last_prim_fail_nur_stored = primfail_nur_stored_tls[tid_min]
                diag.last_prim_fail_Pi_stored = primfail_Pi_stored_tls[tid_min]
                diag.last_prim_fail_piR_stored = primfail_piR_stored_tls[tid_min]
                diag.last_prim_fail_piEta_stored = primfail_piEta_stored_tls[tid_min]
                diag.last_prim_fail_reason = primfail_reason_tls[tid_min]
                diag.last_prim_fail_iters = primfail_iters_tls[tid_min]
                diag.last_prim_fail_resnorm = primfail_resnorm_tls[tid_min]
            end
        end
    end

    i0 = ng + 1
    if !AXIS_CELL_EXACT
        work.y[i0]  = 0.0
        work.vC[i0] = 0.0
    end

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
                                                              grid.rC[i],
                                                              τ, eos, L)
            okR, primR = prim_to_cons_col_ideal_phi_diff_visc!(work.URc, i,
                                                              yTR, φR, yR,
                                                              nurR_phys, PiR_phys, piRR_phys, piEtaR_phys,
                                                              grid.rC[i+1],
                                                              τ, eos, L)

            if okL && okR &&
               work.ULc[L.iDtau,i] ≥ 0.0 && work.URc[L.iDtau,i] ≥ 0.0 &&
               work.ULc[L.iE,i]    ≥ Emin && work.URc[L.iE,i]    ≥ Emin
                hlle_flux_col_cons!(work.Fh, i, work.ULc, work.URc, primL, primR, eos, grid.rC[i], grid.rC[i+1], τ, model, tmpFL, tmpFR)
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

        hlle_flux_lr_U!(work.Fh, i, U, i, i+1, primL, primR, eos, grid.rC[i], grid.rC[i+1], τ, model, tmpFL, tmpFR)
    end

    # Density-frame charge sector: add the first-order parabolic diffusion flux
    # J^r_D = -κ (u^τ)^2 ∂_r α to the (purely advective) HLLE charge flux.
    if model.charge_mode === :density_frame && model.enable_diff
        add_density_frame_charge_flux!(work, grid, τ, model)
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
                              sinh(work.y[i]),
                              grid.rC[i], τ, model)

        if need_theta
            divrv = work.theta[i] - safe_inv(τ)

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
        invr = safe_inv(rC)

        @inbounds for a in 1:Nvars
            div = (rRp*work.Fh[a,i] - rRm*work.Fh[a,i-1]) / grid.dr
            dU[a,i] = -div * invr + work.S[a,i]
        end
    end

    return true
end
