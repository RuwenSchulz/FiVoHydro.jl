# ==============================================================================
# src2d/timestepper2d.jl — SSPRK2/3 method of lines + CFL.
#
# A dedicated stepper rather than a reuse of src/timestepper.jl: that file is
# index-agnostic (only two references to grid.rC, both in error messages) but its
# calls into `rhs!` and `relax_dissipative!` carry 1-D signatures. Reproducing the
# ~120 lines that matter is cleaner than threading a dispatch layer through a
# production file we have committed not to touch.
#
# The dt-halving retry mirrors the 1-D behaviour: a stage that produces an
# inadmissible state is not accepted, the step is halved and retried.
#
# Each stage is followed by `sanitize_stage_2d!` (floors -> S-E bound -> nu bound
# -> BCs), in the same order src/timestepper.jl applies them. Without it the
# solver fails on the FIRST step of the production IC, whose tapered vacuum tail
# sits at the energy floor.
#
# MOOD: if a stage still leaves inadmissible cells after sanitising, they are
# flagged, the mask is grown by one cell so the whole stencil that touched them is
# covered, and the stage is REDONE with piecewise-constant reconstruction there.
# Only if that also fails does the step halve. Same escalation as
# src/timestepper.jl (mark_bad! -> expand_bad! -> retry with force_first_order).
# ==============================================================================

"""
    compute_dt_2d(work, g, τ, model; CFL, CFLτ) -> Δτ

Transverse CFL from the largest HLLE signal speed seen in the last `rhs_2d!`,
plus a cap on the fractional change of the Bjorken clock. `min(dx,dy)` is used
because the scheme is unsplit — both directions are advanced with one Δτ.
"""
function compute_dt_2d(work::Work2D, g::Grid2D, τ::Float64;
                       CFL::Float64 = 0.2, CFLτ::Float64 = 0.05)
    amax = 0.0
    @inbounds for v in work.amax_tls
        v > amax && (amax = v)
    end
    amax = max(amax, 1e-6)
    dt_cfl = CFL * min(g.dx, g.dy) / amax
    dt_tau = CFLτ * τ
    return min(dt_cfl, dt_tau)
end

"""State admissibility: positive charge and above-floor energy on the interior."""
function state_ok_2d(U::AbstractMatrix, g::Grid2D, L::StateLayout2D; Emin::Float64 = E_FLOOR)
    ng = g.nghost
    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)
        (isfinite(U[L.iE,i]) && U[L.iE,i] >= Emin) || return false
        (isfinite(U[L.iDtau,i]) && U[L.iDtau,i] >= 0.0) || return false
        (isfinite(U[L.iSx,i]) && isfinite(U[L.iSy,i])) || return false
    end
    return true
end

"""
    step_ssprk2_2d!(U, g, τ, Δτ, model, work; ...) -> (ok, Δτ_used)

Heun / SSPRK2. On an inadmissible stage the step is halved and retried up to
`max_halve` times.
"""
function step_ssprk2_2d!(U::AbstractMatrix, g::Grid2D, τ::Float64, Δτ::Float64,
                         model::IdealDiffVisc2DModel, work::Work2D;
                         bc::Symbol = :outflow, max_halve::Int = 12,
                         max_mood::Int = 2)
    L = model.layout
    nv = nvars(L); Ntot = g.Ntot
    Δ = Δτ

    for _ in 0:max_halve
        copyto!(work.U2, U)                       # keep the entry state

        ok1 = _stage_2d!(work.U1, U, work.U2, U, g, τ, Δ, 1.0, 0.0, model, work, bc,
                         max_mood, nv, Ntot, L)
        if ok1
            ok2 = _stage_2d!(U, work.U1, work.U2, work.U1, g, τ + Δ, Δ, 0.5, 0.5,
                             model, work, bc, max_mood, nv, Ntot, L)
            ok2 && return true, Δ
        end

        copyto!(U, work.U2)                       # restore and retry smaller
        Δ *= 0.5
    end
    return false, Δ
end

"""
One RK stage with the MOOD escalation.

`Uout = wold*Uold + wnew*(Uin + Δ·L(Uin))`, evaluated on `Uin`, then sanitised.
If cells remain inadmissible they are flagged, the mask grown by one, and the
stage redone with first-order reconstruction there. `Uold` is the state the RK
weight `wold` multiplies (the step's entry state for the second SSPRK2 stage).
"""
function _stage_2d!(Uout::AbstractMatrix, Uin::AbstractMatrix,
                    Uold::AbstractMatrix, Ubase::AbstractMatrix,
                    g::Grid2D, τ::Float64, Δ::Float64, wold::Float64, wnew::Float64,
                    model::IdealDiffVisc2DModel, work::Work2D, bc::Symbol,
                    max_mood::Int, nv::Int, Ntot::Int, L::StateLayout2D)
    mask = nothing
    for attempt in 0:max_mood
        rhs_2d!(work.k, Ubase, g, τ, model, work; bc = bc, force_first_order = mask)
        if wnew == 0.0
            @inbounds for i in 1:Ntot, a in 1:nv
                Uout[a,i] = Uin[a,i] + Δ*work.k[a,i]
            end
        else
            @inbounds for i in 1:Ntot, a in 1:nv
                Uout[a,i] = wold*Uold[a,i] + wnew*(Uin[a,i] + Δ*work.k[a,i])
            end
        end
        sanitize_stage_2d!(Uout, g, τ, model, work; bc = bc)
        state_ok_2d(Uout, g, L) && return true

        attempt == max_mood && return false
        mark_bad_2d!(work.bad, Uout, g, model)
        expand_bad_2d!(work.bad, work.bad_tmp, g; radius = 1 + attempt)
        mask = work.bad
    end
    return false
end

"""
    step_ssprk3_2d!(U, g, τ, Δτ, model, work; ...) -> (ok, Δτ_used)

Shu-Osher SSPRK3. Used for the convergence gates, where the second-order stepper
would contaminate the spatial order measurement.
"""
function step_ssprk3_2d!(U::AbstractMatrix, g::Grid2D, τ::Float64, Δτ::Float64,
                         model::IdealDiffVisc2DModel, work::Work2D;
                         bc::Symbol = :outflow, max_halve::Int = 12)
    L = model.layout
    nv = nvars(L); Ntot = g.Ntot
    Δ = Δτ

    for _ in 0:max_halve
        copyto!(work.U2, U)

        rhs_2d!(work.k, U, g, τ, model, work; bc = bc)
        @inbounds for i in 1:Ntot, a in 1:nv
            work.U1[a,i] = U[a,i] + Δ*work.k[a,i]
        end
        sanitize_stage_2d!(work.U1, g, τ + Δ, model, work; bc = bc)
        if !state_ok_2d(work.U1, g, L)
            copyto!(U, work.U2); Δ *= 0.5; continue
        end

        rhs_2d!(work.k, work.U1, g, τ + Δ, model, work; bc = bc)
        @inbounds for i in 1:Ntot, a in 1:nv
            work.U1[a,i] = 0.75*work.U2[a,i] + 0.25*(work.U1[a,i] + Δ*work.k[a,i])
        end
        sanitize_stage_2d!(work.U1, g, τ + Δ, model, work; bc = bc)
        if !state_ok_2d(work.U1, g, L)
            copyto!(U, work.U2); Δ *= 0.5; continue
        end

        rhs_2d!(work.k, work.U1, g, τ + 0.5Δ, model, work; bc = bc)
        @inbounds for i in 1:Ntot, a in 1:nv
            U[a,i] = (work.U2[a,i] + 2.0*(work.U1[a,i] + Δ*work.k[a,i]))/3.0
        end
        sanitize_stage_2d!(U, g, τ + Δ, model, work; bc = bc)
        if state_ok_2d(U, g, L)
            return true, Δ
        end

        copyto!(U, work.U2); Δ *= 0.5
    end
    return false, Δ
end
