# ------------------------------------------------------------
# SSPRK2 + MOOD
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
    last_fail_U::Union{Nothing,typeof(U)} = nothing

    for halv in 1:max_dt_halvings
        diag === nothing || diag_add!(diag; halvings = (halv > 1 ? 1 : 0))

        local ok1 = false
        local force_mask::Union{Nothing,BitVector} = nothing

        for retry in 0:max_stage_retries
            rhs!(work.k, U, grid, τ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)
            diag === nothing || _maybe_log_primfail!(diag, U, grid, τ, model, work, retry; Emin=Emin, χ=χ)
            diag === nothing || _abort_on_first_primfail!(diag, U, grid, τ, model, work, retry; Emin=Emin, χ=χ)

            @. work.U1 = U + Δ*work.k
            apply_bc!(work.U1, grid, τ+Δ, model)
            _repair_Dtau_positivity!(work.U1, U, grid, model)
            enforce_floors!(work.U1, grid, τ+Δ, model; Emin=Emin, diag=diag)
            _repair_theta_admissibility!(work.U1, U, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
            enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)

            apply_bc!(work.U1, grid, τ+Δ, model)

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

            mark_bad!(bad, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood1=count(bad))

            if hydro_flags().write_badmask
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage1_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=bad, diag=diag)
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
            diag === nothing || _maybe_log_primfail!(diag, work.U1, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)
            diag === nothing || _abort_on_first_primfail!(diag, work.U1, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)

            @. work.U2 = 0.5*U + 0.5*(work.U1 + Δ*work.k)
            apply_bc!(work.U2, grid, τ+Δ, model)
            _repair_Dtau_positivity!(work.U2, work.U1, grid, model)
            enforce_floors!(work.U2, grid, τ+Δ, model; Emin=Emin, diag=diag)
            _repair_theta_admissibility!(work.U2, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
            enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)

            apply_bc!(work.U2, grid, τ+Δ, model)

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

            mark_bad!(bad, work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood2=count(bad))

            if hydro_flags().write_badmask
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage2_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U2, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, P=work.P, mask=bad, diag=diag)
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
        _repair_Dtau_positivity!(U, work.U1, grid, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        _repair_theta_admissibility!(U, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)

        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)
        apply_bc!(U, grid, τ+Δ, model)

        relax_dissipative!(U, grid, τ+Δ, Δ, model, work; diag=diag)
        apply_bc!(U, grid, τ+Δ, model)
        _repair_Dtau_positivity!(U, work.U1, grid, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        _repair_theta_admissibility!(U, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)

        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)
        apply_bc!(U, grid, τ+Δ, model)


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
# SSPRK3 + MOOD (Shu-Osher)
#
# Notes:
# - We follow the same MOOD+dt-halving strategy as SSPRK2.
# - For simplicity (and to match the existing dissipative relaxation treatment),
#   we apply stage post-processing at the same stage time τ+Δ.
# ------------------------------------------------------------
function step_ssprk3!(U, grid, τ, Δτ, model::IdealDiffViscModel, work::Work1D;
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
    last_fail_U::Union{Nothing,typeof(U)} = nothing

    for halv in 1:max_dt_halvings
        diag === nothing || diag_add!(diag; halvings = (halv > 1 ? 1 : 0))

        local ok1 = false
        local force_mask::Union{Nothing,BitVector} = nothing

        # --- stage 1 ---
        for retry in 0:max_stage_retries
            rhs!(work.k, U, grid, τ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)
            diag === nothing || _maybe_log_primfail!(diag, U, grid, τ, model, work, retry; Emin=Emin, χ=χ)
            diag === nothing || _abort_on_first_primfail!(diag, U, grid, τ, model, work, retry; Emin=Emin, χ=χ)

            @. work.U1 = U + Δ*work.k
            apply_bc!(work.U1, grid, τ+Δ, model)
            _repair_Dtau_positivity!(work.U1, U, grid, model)
            enforce_floors!(work.U1, grid, τ+Δ, model; Emin=Emin, diag=diag)
            _repair_theta_admissibility!(work.U1, U, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
            enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)

            apply_bc!(work.U1, grid, τ+Δ, model)

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

            mark_bad!(bad, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood1=count(bad))

            if hydro_flags().write_badmask
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage1_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=bad, diag=diag)
            end

            force_mask = bad
        end

        if !ok1
            last_fail_U = copy(work.U1)
            Δ *= 0.5
            continue
        end

        # --- stage 2 ---
        local ok2 = false
        force_mask = nothing
        for retry in 0:max_stage_retries
            rhs!(work.k, work.U1, grid, τ+Δ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)
            diag === nothing || _maybe_log_primfail!(diag, work.U1, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)
            diag === nothing || _abort_on_first_primfail!(diag, work.U1, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)

            @. work.U2 = (3/4)*U + (1/4)*(work.U1 + Δ*work.k)
            apply_bc!(work.U2, grid, τ+Δ, model)
            _repair_Dtau_positivity!(work.U2, work.U1, grid, model)
            enforce_floors!(work.U2, grid, τ+Δ, model; Emin=Emin, diag=diag)
            _repair_theta_admissibility!(work.U2, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
            enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)

            apply_bc!(work.U2, grid, τ+Δ, model)

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

            mark_bad!(bad, work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            diag === nothing || diag_add!(diag; mood2=count(bad))

            if hydro_flags().write_badmask
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage2_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U2, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U2, grid, model; χ=χ, P=work.P, mask=bad, diag=diag)
            end

            force_mask = bad
        end

        if !ok2
            last_fail_U = copy(work.U2)
            Δ *= 0.5
            continue
        end

        # --- stage 3 (final) ---
        local ok3 = false
        force_mask = nothing
        for retry in 0:max_stage_retries
            rhs!(work.k, work.U2, grid, τ+Δ, model, work; Emin=Emin, force_first_order=force_mask, diag=diag)
            diag === nothing || _maybe_log_primfail!(diag, work.U2, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)
            diag === nothing || _abort_on_first_primfail!(diag, work.U2, grid, τ+Δ, model, work, retry; Emin=Emin, χ=χ)

            # Reuse U1 as the final buffer.
            @. work.U1 = (1/3)*U + (2/3)*(work.U2 + Δ*work.k)
            apply_bc!(work.U1, grid, τ+Δ, model)
            _repair_Dtau_positivity!(work.U1, work.U2, grid, model)
            enforce_floors!(work.U1, grid, τ+Δ, model; Emin=Emin, diag=diag)
            _repair_theta_admissibility!(work.U1, work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
            enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)

            apply_bc!(work.U1, grid, τ+Δ, model)

            if SANITIZE_SCOPE == :global
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=nothing, diag=diag)
            end

            if !DO_MOOD || !any_bad(work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
                ok3 = true
                break
            end

            last_fail_stage = 3
            last_fail_tau   = τ + Δ
            last_fail_Δ     = Δ
            last_fail_i, last_fail_why = _find_first_bad(work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)

            mark_bad!(bad, work.U1, grid, τ+Δ, model, work; Emin=Emin, χ=χ)
            expand_bad!(bad, bad_tmp, grid; radius=1+retry)
            # reuse mood2 counter for stage 3 as well (keeps diagnostics simple)
            diag === nothing || diag_add!(diag; mood2=count(bad))

            if hydro_flags().write_badmask
                write_badmask_csv(joinpath("debug_masks", @sprintf("bad_stage3_tau_%06.3f_retry_%d.csv", τ+Δ, retry)), bad, grid)
            end

            if SANITIZE_SCOPE == :local
                sanitize_state!(work.U1, grid, τ+Δ, model; Emin=Emin, mask=bad, diag=diag)
                enforce_Sr_energy_constraint!(work.U1, grid, model; χ=χ, P=work.P, mask=bad, diag=diag)
            end

            force_mask = bad
        end

        if !ok3
            last_fail_U = copy(work.U1)
            Δ *= 0.5
            continue
        end

        U .= work.U1
        apply_bc!(U, grid, τ+Δ, model)
        _repair_Dtau_positivity!(U, work.U2, grid, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        _repair_theta_admissibility!(U, work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)
        apply_bc!(U, grid, τ+Δ, model)

        relax_dissipative!(U, grid, τ+Δ, Δ, model, work; diag=diag)
        apply_bc!(U, grid, τ+Δ, model)
        _repair_Dtau_positivity!(U, work.U2, grid, model)
        enforce_floors!(U, grid, τ+Δ, model; Emin=Emin, diag=diag)
        _repair_theta_admissibility!(U, work.U2, grid, τ+Δ, model, work; Emin=Emin, χ=χ, mask=nothing, diag=diag)
        enforce_Sr_energy_constraint!(U, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)
        apply_bc!(U, grid, τ+Δ, model)

        if SANITIZE_SCOPE == :global
            sanitize_state!(U, grid, τ+Δ, model; Emin=Emin, mask=nothing, diag=diag)
        end

        return Δ
    end

    # Hard failure: dump a small window around the last known bad cell/state.
    Udbg  = (last_fail_U === nothing ? work.U1 : last_fail_U)
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
