# ==============================================================================
# src2d/floors2d.jl — floors, admissibility constraint, MOOD detection.
#
# Counterpart of src/floors.jl and the detection half of src/mood.jl.
#
# WHY THIS EXISTS: the bare scheme runs every controlled gate (Bjorken, Gubser-like
# uniform states, smooth bumps) without any of this. It fails on the FIRST STEP of
# the real production IC, because that IC has a vacuum tail — `load_initial_interpolants`
# tapers T down to `T_MIN = 1e-20`, so the outer half of the grid has `E` at or below
# the floor and `|S|` unbounded relative to it. The 1-D solver has carried floors,
# an S–E admissibility bound and MOOD from the start for exactly this reason.
#
# Conventions follow src/floors.jl term for term, including the `E <= 1e6*Emin`
# NEAR-VACUUM branch: cells that are above the hard floor but still tiny get their
# momenta, charge and dissipatives zeroed, because the primitive recovery and the
# relaxation are ill-conditioned there even though nothing is formally invalid.
# ==============================================================================

"""
    enforce_floors_2d!(U, g, model; Emin=E_FLOOR) -> (nfloorE, nfloorD, nnan)

Interior-only; call `apply_bc_2d!` afterwards to refill ghosts.
"""
function enforce_floors_2d!(U::AbstractMatrix, g::Grid2D, model::IdealDiffVisc2DModel;
                            Emin::Float64 = E_FLOOR)
    L = model.layout
    ng = g.nghost
    # Declare-vacuum threshold: the hard floor, or the energy of `T_vac_cut`,
    # whichever is larger. See the note on T_vac_cut in primitives2d.jl.
    Evac = max(Emin, model.E_vac_cut)
    rdom = model.r_domain
    rdom2 = isfinite(rdom) ? rdom*rdom : Inf
    nfloorE = 0; nfloorD = 0; nnan = 0

    @inline function blank!(i)
        @inbounds begin
            U[L.iSx,i] = 0.0
            U[L.iSy,i] = 0.0
            U[L.iDtau,i] = 0.0
            if L.hasNu;    U[L.iNux,i] = 0.0;  U[L.iNuy,i] = 0.0;  end
            if L.hasPi;    U[L.iPi,i] = 0.0;                        end
            if L.hasShear
                U[L.iPixx,i] = 0.0; U[L.iPixy,i] = 0.0
                U[L.iPiyy,i] = 0.0; U[L.iPieta,i] = 0.0
            end
        end
    end

    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)

        # Outside the disc: hold at vacuum. Applied EVERY stage, not just at t=0 —
        # seeding the exterior as vacuum once is not enough, because the interior
        # fluxes refill it and the instability regrows there.
        if isfinite(rdom)
            x = g.xC[ix]; y = g.yC[iy]
            if x*x + y*y > rdom2
                U[L.iE,i] = Emin; blank!(i); continue
            end
        end

        Ei = U[L.iE,i]

        if !isfinite(Ei)
            nnan += 1; U[L.iE,i] = Emin; blank!(i); continue
        end
        if Ei < Evac
            # Below the vacuum cut: collapse to the hard floor so the primitive
            # recovery takes its PRR_VACUUM branch instead of trying to solve a
            # state the EOS cannot represent.
            nfloorE += 1; U[L.iE,i] = Emin; blank!(i); continue
        end
        # Near-vacuum: above the cut but not usefully resolvable. The 1e6 factor
        # is calibrated against the HARD floor (1e-20 -> 1e-14), NOT against the
        # physical vacuum cut: applying it to E_vac_cut = 7.4e-7 would blank
        # everything below 0.74, which at tau ~ 4 is most of the fireball. That
        # mistake showed up immediately as L2 = 0.32 against the 1-D reference.
        if Ei <= 1e6 * Emin
            blank!(i)
        end

        Di = U[L.iDtau,i]
        if !isfinite(Di) || Di < 0.0
            nfloorD += 1; U[L.iDtau,i] = 0.0
        end
        if !isfinite(U[L.iSx,i]); nnan += 1; U[L.iSx,i] = 0.0; end
        if !isfinite(U[L.iSy,i]); nnan += 1; U[L.iSy,i] = 0.0; end

        if L.hasNu
            isfinite(U[L.iNux,i]) || (nnan += 1; U[L.iNux,i] = 0.0)
            isfinite(U[L.iNuy,i]) || (nnan += 1; U[L.iNuy,i] = 0.0)
        end
        if L.hasPi
            isfinite(U[L.iPi,i]) || (nnan += 1; U[L.iPi,i] = 0.0)
        end
        if L.hasShear
            isfinite(U[L.iPixx,i])  || (nnan += 1; U[L.iPixx,i]  = 0.0)
            isfinite(U[L.iPixy,i])  || (nnan += 1; U[L.iPixy,i]  = 0.0)
            isfinite(U[L.iPiyy,i])  || (nnan += 1; U[L.iPiyy,i]  = 0.0)
            isfinite(U[L.iPieta,i]) || (nnan += 1; U[L.iPieta,i] = 0.0)
        end
    end
    return nfloorE, nfloorD, nnan
end

"""
    enforce_S_energy_constraint_2d!(U, g, model; χ=χ_SrE, P=nothing) -> (nscaled, maxratio)

Bound the transverse momentum by `|S| ≤ χ (E + P_eff)`, rescaling the VECTOR
`(S^x, S^y)` so its direction is preserved. The 1-D version scales a scalar; in
2-D scaling the magnitude and keeping the direction is the only choice that does
not manufacture flow along an axis — the same reasoning that fixed the shear
storage (TWOD_PROGRAM.md §2).

`P` is the pressure from the last successful recovery (`work.P`); viscous terms
enter as absolute magnitudes, as in `enforce_Sr_energy_constraint!`.
"""
function enforce_S_energy_constraint_2d!(U::AbstractMatrix, g::Grid2D,
                                         model::IdealDiffVisc2DModel;
                                         χ::Float64 = χ_SrE,
                                         P::Union{Nothing,AbstractVector{Float64}} = nothing)
    L = model.layout
    ng = g.nghost
    nscaled = 0; maxratio = 0.0

    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)
        Ei = U[L.iE,i]; Sx = U[L.iSx,i]; Sy = U[L.iSy,i]
        (isfinite(Ei) && isfinite(Sx) && isfinite(Sy)) || continue
        Ei <= 0 && continue

        Peff = 0.0
        if P !== nothing
            p = P[i]
            isfinite(p) && (Peff = max(p, 0.0))
        end
        if L.hasPi
            v = phys_from_stored(U[L.iPi,i]); isfinite(v) && (Peff += abs(v))
        end
        if L.hasShear
            for k in (L.iPixx, L.iPixy, L.iPiyy, L.iPieta)
                v = phys_from_stored(U[k,i]); isfinite(v) && (Peff += abs(v))
            end
        end

        Smax = χ * (Ei + Peff)
        aS = hypot(Sx, Sy)
        if aS > Smax
            nscaled += 1
            maxratio = max(maxratio, aS/(Smax + TINY))
            sc = Smax/(aS + TINY)
            U[L.iSx,i] = Sx*sc
            U[L.iSy,i] = Sy*sc
        end
    end
    return nscaled, maxratio
end

"""
    enforce_nu_charge_constraint_2d!(U, g, model; χ=0.999) -> (nscaled, maxratio)

Bound the diffusion current by the charge it carries: `|ν_⊥| ≤ χ · J^τ`.

This is the charge-sector analogue of `enforce_S_energy_constraint_2d!`, and it
exists for a concrete reason. The recovery's charge row is
`n u^τ + ν^τ = J^τ`, so a solution with `n ≥ 0` exists only if `J^τ - ν^τ > 0`.
Since `|ν^τ| = |u·ν|/u^τ < |ν_⊥|`, bounding `|ν_⊥| ≤ χ J^τ` GUARANTEES that,
without needing the velocity (which is not yet known when this runs).

Without it the charge row becomes unsolvable in the dilute tail, and the staged
fallback in `primrec2d.jl` then keeps a hydro state whose charge does not match
its own conserved variable — measured on the production IC at N=150, τ=8: total
charge drift 1.85e+05. With it the row is always closable.

Like the momentum bound, this rescales the VECTOR and preserves its direction.
"""
function enforce_nu_charge_constraint_2d!(U::AbstractMatrix, g::Grid2D,
                                          model::IdealDiffVisc2DModel, τ::Float64;
                                          χ::Float64 = 0.999)
    L = model.layout
    L.hasNu || return (0, 0.0)
    ng = g.nghost
    nscaled = 0; maxratio = 0.0

    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)
        # U[iDtau] = τ J^τ; the τ cancels in the ratio, so compare directly.
        Dt = U[L.iDtau,i] / τ            # J^τ; U stores τ J^τ
        nux = phys_from_stored(U[L.iNux,i])
        nuy = phys_from_stored(U[L.iNuy,i])
        (isfinite(Dt) && isfinite(nux) && isfinite(nuy)) || continue

        if Dt <= 0.0
            (nux != 0.0 || nuy != 0.0) && (nscaled += 1)
            U[L.iNux,i] = 0.0; U[L.iNuy,i] = 0.0
            continue
        end

        cap = χ * Dt
        aν = hypot(nux, nuy)
        if aν > cap
            nscaled += 1
            maxratio = max(maxratio, aν/(cap + TINY))
            sc = cap/(aν + TINY)
            U[L.iNux,i] = stored_from_phys(nux*sc)
            U[L.iNuy,i] = stored_from_phys(nuy*sc)
        end
    end
    return nscaled, maxratio
end

"""
    sanitize_stage_2d!(U, g, τ, model, work; Emin, χ, bc)

The full post-stage repair sequence, in the order `src/timestepper.jl` applies it:
floors, then the S–E bound, then boundary conditions.
"""
function sanitize_stage_2d!(U::AbstractMatrix, g::Grid2D, τ::Float64,
                            model::IdealDiffVisc2DModel, work::Work2D;
                            Emin::Float64 = E_FLOOR, χ::Float64 = χ_SrE,
                            bc::Symbol = :outflow)
    enforce_floors_2d!(U, g, model; Emin = Emin)
    enforce_S_energy_constraint_2d!(U, g, model; χ = χ, P = work.P)
    enforce_nu_charge_constraint_2d!(U, g, model, τ)
    apply_bc_2d!(U, g, model.layout; bc = bc)
    return nothing
end

# ------------------------------------------------------------------------------
# ADMISSIBILITY — the predicate the MOOD escalation turns on
# ------------------------------------------------------------------------------
#
# 🔴 2026-09-08. Until this pass the 2-D MOOD escalation and the dt-halving retry
# were UNREACHABLE, and nothing said so. `state_ok_2d` tested exactly the three
# invariants `sanitize_stage_2d!` had just imposed one line earlier — every cell
# finite, `E >= Emin`, `D >= 0` — so it could not fail. MEASURED: over 2000
# deliberately pathological states (NaN, -Inf, huge negatives) pushed through
# `sanitize_stage_2d!`, `state_ok_2d` returned true 2000/2000, while the old
# `mark_bad_2d!` would have flagged cells in all 2000 — but it is only ever CALLED
# after `state_ok_2d` returns false. `force_first_order`, `mark_bad_2d!`,
# `expand_bad_2d!` and the Δ-halving retry were all dead code.
#
# That is very likely what main2D.jl's own end-of-run note was seeing: a
# single-event IC ran to completion with |pi|/P = 1e12 and max|u| = 58 and
# "succeeded by every criterion the solver had".
#
# The old `mark_bad_2d!` bound was independently too loose: `χ(E + |E|) = 2χE`,
# against a physical `|S| <= E + P` with `P <= E/3`. MEASURED: states with
# `1.0 <= |S|/E < 2.0` are NOT invertible by `cons_to_prim_2d!`
# (PRR_RESIDUAL_TOO_LARGE) and the old detector flagged none of them.
#
# The fix mirrors `src/mood.jl:admissible_state_fast`, which is the 1-D solver's
# answer and does not have this defect. Cheap filters first, and the arbiter for
# anything near the causal bound is the ONE test that actually matters: run the
# primitive recovery and see whether it inverts.
#
#   1. non-finite, or `E < Emin`, or `D < 0`                        -> inadmissible
#   2. any non-finite dissipative dof                               -> inadmissible
#   3. near-vacuum (`E <= 1e6 Emin`, the same branch enforce_floors_2d! uses)
#                                                                   -> admissible
#   4. last recovery's |v| >= 1                                     -> inadmissible
#   5. `|S| <= SR_MARGIN_2D χ (E + P + |Π| + Σ|π|)`                  -> admissible
#   6. otherwise: `cons_to_prim_2d!`, and PRR_VACUUM counts as admissible
#
# `SR_MARGIN_2D < 1` is what makes this NON-VACUOUS, and is the whole point:
# `enforce_S_energy_constraint_2d!` rescales an over-the-bound cell to EXACTLY
# `χ(E + Peff)`, so a test at `χ` would again be satisfied by construction. The
# margin leaves a band in which the sanitiser's repair is not taken on trust but
# checked by inversion. It is 1-D's `sr_margin()` device, one file over.
const SR_MARGIN_2D = 0.98

"""
    cell_admissible_2d(U, i, τ, model, work, L; Emin, χ, v_eps) -> Bool

Whether cell `i` carries a state the scheme can actually invert. See the block
above for why this is not the same question as "did the floors run".

`work` may be `nothing`; the velocity and pressure filters are then skipped and
more cells fall through to the recovery, which is slower but never wrong.
"""
function cell_admissible_2d(U::AbstractMatrix, i::Int, τ::Float64,
                            model::IdealDiffVisc2DModel,
                            work::Union{Nothing,Work2D}, L::StateLayout2D;
                            Emin::Float64 = E_FLOOR, χ::Float64 = χ_SrE,
                            v_eps::Float64 = 1e-12)
    @inbounds begin
        E = U[L.iE,i]; Sx = U[L.iSx,i]; Sy = U[L.iSy,i]; Dt = U[L.iDtau,i]
        (isfinite(E) & isfinite(Sx) & isfinite(Sy) & isfinite(Dt)) || return false
        (E >= Emin) || return false
        (Dt >= 0.0) || return false

        if L.hasNu
            (isfinite(U[L.iNux,i]) & isfinite(U[L.iNuy,i])) || return false
        end
        if L.hasPi
            isfinite(U[L.iPi,i]) || return false
        end
        if L.hasShear
            (isfinite(U[L.iPixx,i]) & isfinite(U[L.iPixy,i]) &
             isfinite(U[L.iPiyy,i]) & isfinite(U[L.iPieta,i])) || return false
        end

        # Near-vacuum: `enforce_floors_2d!` has already blanked momentum, charge
        # and the dissipatives here, and there is no state left to decide. The
        # 1-D predicate has the same escape (`E < 100 Emin`).
        E <= 1e6*Emin && return true

        if work !== nothing && work.ok[i]
            vx = work.vxC[i]; vy = work.vyC[i]
            (isfinite(vx) & isfinite(vy)) || return false
            (vx*vx + vy*vy) < (1 - v_eps)^2 || return false
        end

        Peff = 0.0
        if work !== nothing
            p = work.P[i]
            isfinite(p) && (Peff = max(p, 0.0))
        end
        if L.hasPi
            v = phys_from_stored(U[L.iPi,i]); isfinite(v) && (Peff += abs(v))
        end
        if L.hasShear
            for k in (L.iPixx, L.iPixy, L.iPiyy, L.iPieta)
                v = phys_from_stored(U[k,i]); isfinite(v) && (Peff += abs(v))
            end
        end
        hypot(Sx, Sy) <= SR_MARGIN_2D*χ*(E + Peff) && return true

        # Within the margin of the causal bound: settle it by inverting.
        wpr = model.primrec.work[Threads.threadid()]
        nux, nuy, Pi, pixx, pixy, piyy, pieta = dissipatives_at(U, i, L)
        _, _, _, _, _, _, _, ok = cons_to_prim_2d!(
            wpr, safe_div(Dt, τ), Sx, Sy, E, nux, nuy, Pi,
            pixx, pixy, piyy, pieta, τ, model.eos)
        return ok || wpr.last_reason == PRR_VACUUM
    end
end

# ------------------------------------------------------------------------------
# MOOD detection
# ------------------------------------------------------------------------------

"""
    mark_bad_2d!(bad, U, g, τ, model, work; Emin, χ) -> count

Flag every cell `cell_admissible_2d` rejects. Same predicate `state_ok_2d` uses,
so the mask always covers what made the stage unacceptable — before this pass the
two used different bounds and the mask could come back empty on a rejected stage.
"""
function mark_bad_2d!(bad::BitVector, U::AbstractMatrix, g::Grid2D, τ::Float64,
                      model::IdealDiffVisc2DModel,
                      work::Union{Nothing,Work2D} = nothing;
                      Emin::Float64 = E_FLOOR, χ::Float64 = χ_SrE)
    L = model.layout
    ng = g.nghost
    fill!(bad, false)
    nbad = 0
    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)
        if !cell_admissible_2d(U, i, τ, model, work, L; Emin = Emin, χ = χ)
            bad[i] = true; nbad += 1
        end
    end
    return nbad
end

"""Grow the bad mask by `radius` cells in the 4-neighbour sense, so the
first-order fallback covers the whole stencil that touched a bad cell."""
function expand_bad_2d!(bad::BitVector, tmp::BitVector, g::Grid2D; radius::Int = 1)
    ng = g.nghost
    for _ in 1:radius
        copyto!(tmp, bad)
        @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            tmp[i] && continue
            if bad[i-1] || bad[i+1] || bad[i-g.Nytot] || bad[i+g.Nytot]
                tmp[i] = true
            end
        end
        copyto!(bad, tmp)
    end
    return count(bad)
end
