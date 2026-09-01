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
# Each stage is followed by `sanitize_stage_2d!` (floors -> S-E bound -> BCs), in
# the same order src/timestepper.jl applies them. Without it the solver fails on
# the FIRST step of the production IC, whose tapered vacuum tail sits at the
# energy floor.
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
                         bc::Symbol = :outflow, max_halve::Int = 12)
    L = model.layout
    nv = nvars(L); Ntot = g.Ntot
    Δ = Δτ

    for _ in 0:max_halve
        copyto!(work.U2, U)                       # keep the entry state

        rhs_2d!(work.k, U, g, τ, model, work; bc = bc)
        @inbounds for i in 1:Ntot, a in 1:nv
            work.U1[a,i] = U[a,i] + Δ*work.k[a,i]
        end
        sanitize_stage_2d!(work.U1, g, τ + Δ, model, work; bc = bc)

        if state_ok_2d(work.U1, g, L)
            rhs_2d!(work.k, work.U1, g, τ + Δ, model, work; bc = bc)
            @inbounds for i in 1:Ntot, a in 1:nv
                U[a,i] = 0.5*(work.U2[a,i] + work.U1[a,i] + Δ*work.k[a,i])
            end
            sanitize_stage_2d!(U, g, τ + Δ, model, work; bc = bc)
            if state_ok_2d(U, g, L)
                return true, Δ
            end
        end

        copyto!(U, work.U2)                       # restore and retry smaller
        Δ *= 0.5
    end
    return false, Δ
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
