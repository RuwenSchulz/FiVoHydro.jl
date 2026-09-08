# ==============================================================================
# src2d/primrec2d.jl — 2+1D primitive recovery.
#
# Counterpart of src/primrec.jl. The 1-D solver runs a 3-unknown Newton on
# (yT, φ, y = asinh u^r): the velocity is a SCALAR with a sign, and the stored
# shear parametrisation makes π^{τr} ∝ u^τu^r, so the shear contribution to the
# momentum is collinear with u by construction (src/primrec.jl:779).
#
# In 2+1D that collinearity is gone — with π^{xy} ≠ 0 the shear tilts the momentum
# away from the flow — so the flow DIRECTION is a genuine unknown and the system is
#
#     unknowns  x = (yT = log T,  φ = (μ-m)/T,  u^x,  u^y)
#
#     F1 = n u^τ + (u^xν^x + u^yν^y)/u^τ           - D      (charge)
#     F2 = w_eff u^τ u^x + π^{τx}                  - S^x    (x-momentum)
#     F3 = w_eff u^τ u^y + π^{τy}                  - S^y    (y-momentum)
#     F4 = w_eff (u^τ)²  - P - Π + π^{ττ}          - E      (energy)
#
# with u^τ = sqrt(1 + (u^x)² + (u^y)²) and the τ-row of π from orthogonality
# (src2d/shear2d.jl).
#
# WHY (u^x, u^y) AND NOT A RAPIDITY: the 1-D code carries y = asinh u^r to keep the
# velocity bounded. Its 2-D analogue would need a magnitude and an ANGLE, and the
# angle is singular at u = 0 — every cell of a symmetric IC starts there. Carrying
# the components directly has no such degeneracy, and |v| = |u|/u^τ < 1 holds
# automatically for any finite (u^x, u^y), so boundedness is not lost.
# ==============================================================================

# ------------------------------------------------------------------------------
# Small dense solver: Gaussian elimination with partial pivoting, N ≤ 4.
# Mirrors src/utils.jl:solve3x3_gauss! (scale-aware pivot tolerance, no allocation).
# ------------------------------------------------------------------------------
@inline function solve_gauss_n!(δ::Vector{Float64}, J::Matrix{Float64},
                                F::Vector{Float64}, A::Matrix{Float64}, n::Int)
    @inbounds begin
        for r in 1:n
            for c in 1:n
                A[r,c] = J[r,c]
            end
            A[r,n+1] = -F[r]
        end

        smax = 0.0
        for r in 1:n, c in 1:n
            smax = max(smax, abs(A[r,c]))
        end
        (!isfinite(smax) || smax == 0.0) && return false
        pivtol = 1e-15 * smax

        for k in 1:n
            p = k
            amax = abs(A[k,k])
            for r in (k+1):n
                if abs(A[r,k]) > amax
                    amax = abs(A[r,k]); p = r
                end
            end
            amax < pivtol && return false
            if p != k
                for c in k:(n+1)
                    A[k,c], A[p,c] = A[p,c], A[k,c]
                end
            end
            invp = 1.0 / A[k,k]
            for r in (k+1):n
                f = A[r,k] * invp
                f == 0.0 && continue
                for c in k:(n+1)
                    A[r,c] -= f * A[k,c]
                end
            end
        end

        for r in n:-1:1
            s = A[r,n+1]
            for c in (r+1):n
                s -= A[r,c] * δ[c]
            end
            δ[r] = s / A[r,r]
        end

        for r in 1:n
            isfinite(δ[r]) || return false
        end
    end
    return true
end

# ------------------------------------------------------------------------------
# Per-thread scratch
# ------------------------------------------------------------------------------
mutable struct PrimRecWork2D
    x::Vector{Float64}
    F::Vector{Float64}
    Fp::Vector{Float64}
    Fm::Vector{Float64}
    J::Matrix{Float64}
    A::Matrix{Float64}
    δ::Vector{Float64}

    last_reason::PrimRecReason
    last_iters::Int
    last_resnorm::Float64
    last_rows::Vector{Float64}      # per-row scaled residual at exit (diagnosis)
end

PrimRecWork2D() = PrimRecWork2D(zeros(4), zeros(4), zeros(4), zeros(4),
                                zeros(4,4), zeros(4,5), zeros(4),
                                PRR_UNSET, 0, NaN, zeros(4))

mutable struct IdealPrimRec2D
    work::Vector{PrimRecWork2D}
end
# maxthreadid(), NOT nthreads(): since Julia 1.9 the interactive pool means
# threadid() can exceed nthreads(:default), and indexing per-thread scratch by
# threadid() then runs off the end of the vector.
IdealPrimRec2D(nthreads::Int = Threads.maxthreadid()) =
    IdealPrimRec2D([PrimRecWork2D() for _ in 1:nthreads])

const U_CAP_2D = 1.0e4      # |u^i| guard; sinh-free, so this is a plain bound

# ------------------------------------------------------------------------------
# Does the EOS carry a μ-dependence? If not, F1 cannot constrain φ and the 4x4
# system is singular, so we drop φ and solve 3x3. Same predicate as
# src/primrec.jl:139-143. Named `_2d` so a test can include BOTH primrec chains
# (test_primrec2d_vs_1d.jl) without redefining the 1-D methods.
# ------------------------------------------------------------------------------
@inline eos_has_charge_2d(eos) = true
@inline eos_has_charge_2d(eos::ConformalHQEOS) = (eos.g_hq > 0.0)
@inline eos_has_charge_2d(eos::RunningConformalHQEOS) = (eos.g_hq > 0.0)
@inline eos_has_charge_2d(::TabulatedHQEOS) = true

# ------------------------------------------------------------------------------
# Seeding the Newton from the conserved variables.
#
# In production the previous timestep supplies an excellent guess, and the Newton
# converges in a handful of iterations. But MOOD repair, MUSCL face states and any
# cold start call the recovery with NO guess, and from a fixed seed (u = 0,
# T = 0.15 GeV) the iteration STALLS on fast-flow cells: measured 0.95% of
# physical-band states failed with PRR_MAXIT at iters = 0, every one of them with
# |u| ~ 1.5. The Jacobian at those solutions is perfectly well conditioned — the
# failure is global, not local, so the cure is a better starting point plus a line
# search (see the backtracking block in cons_to_prim_2d!).
#
# The seed uses the IDEAL relations, which invert in closed form given P:
#     M   = sqrt(Sx² + Sy²)
#     v   = M / (E + P)
#     e   = (E + P)(1 - v²) - P
# and closes the loop by inverting e(T) for T at fixed φ. Shear and bulk are
# ignored here; they are a correction to the seed, not to the answer.
# ------------------------------------------------------------------------------

"""Invert `e(T; φ)` for T by bisection on log T. Monotone in T, so this cannot
fail; used only to build a starting point."""
@inline function _invert_e_for_T(e_target::Float64, φ::Float64, eos)
    lo = log(T_MIN); hi = log(T_SOLVE_MAX)
    e_target <= 0.0 && return lo
    @inbounds for _ in 1:60
        mid = 0.5*(lo + hi)
        Tm  = exp(mid)
        _, _, em = eos_Pne(Tm, hq_mass(eos) + Tm*φ, eos)
        if !isfinite(em) || em > e_target
            hi = mid
        else
            lo = mid
        end
    end
    return 0.5*(lo + hi)
end

"""
    seed_from_conserved_2d(E, Sx, Sy, φ, eos) -> (yT, ux, uy)

Fixed-point seed for the Newton. Three passes are enough: the map
`P -> v -> e -> T -> P` contracts strongly for any equation of state with
`0 < dP/de < 1`.
"""
function seed_from_conserved_2d(E::Float64, Sx::Float64, Sy::Float64, φ::Float64, eos)
    M = sqrt(Sx*Sx + Sy*Sy)
    (!isfinite(E) || E <= 0.0) && return (log(max(0.15, T_MIN)), 0.0, 0.0)

    P  = E/3          # conformal opener
    yT = log(max(0.15, T_MIN))
    v  = 0.0
    @inbounds for _ in 1:3
        denom = E + P
        v = denom > TINY ? min(M/denom, 0.999999) : 0.0
        e = denom*(1 - v*v) - P
        e <= 0.0 && (e = max(E, TINY))
        yT = _invert_e_for_T(e, φ, eos)
        T  = exp(yT)
        Pn, _, _ = eos_Pne(T, hq_mass(eos) + T*φ, eos)
        isfinite(Pn) || break
        P = Pn
    end

    γ  = 1/sqrt(max(1 - v*v, 1e-16))
    us = γ*v                      # |u| = γ|v|
    ux = M > TINY ? us*Sx/M : 0.0
    uy = M > TINY ? us*Sy/M : 0.0
    return (yT, clamp(ux, -U_CAP_2D, U_CAP_2D), clamp(uy, -U_CAP_2D, U_CAP_2D))
end

# ------------------------------------------------------------------------------
# Residual
# ------------------------------------------------------------------------------
"""
    evalF_2d!(F, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos; ncharge)

Residual of the conserved-variable map. `ncharge=false` fixes φ = 0 (μ = m) and
leaves `F[1]` untouched — used for EOS without μ dependence, where F1 cannot
constrain φ and the 4×4 system would be singular. This mirrors the 1-D
`cons_to_prim_ideal_phi0_nocharge!` branch (src/primrec.jl:338).
"""
@inline function evalF_2d!(F::Vector{Float64}, x::Vector{Float64},
                           D::Float64, Sx::Float64, Sy::Float64, E::Float64,
                           nux::Float64, nuy::Float64, Pi::Float64,
                           pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
                           τ::Float64, eos; ncharge::Bool = true)
    yT = x[1]; φ = x[2]; ux = x[3]; uy = x[4]
    if !(isfinite(yT) && isfinite(φ) && isfinite(ux) && isfinite(uy))
        fill!(F, Inf); return
    end

    T  = exp(yT)
    μ  = hq_mass(eos) + T*φ
    uτ = sqrt(1 + ux*ux + uy*uy)

    P, n, e = eos_Pne(T, μ, eos)
    if !(isfinite3(P, n, e))
        fill!(F, Inf); return
    end

    weff = e + P + Pi
    Ptot = P + Pi

    Π = shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)

    F[1] = ncharge ? (n*uτ + nu_tau_2d(ux, uy, uτ, nux, nuy) - D) : 0.0
    F[2] = weff*uτ*ux + Π.tx - Sx
    F[3] = weff*uτ*uy + Π.ty - Sy
    F[4] = weff*uτ*uτ - Ptot + Π.tt - E
    return
end

# ------------------------------------------------------------------------------
# Staged fallback: recover the HYDRO block, then the charge, separately.
#
# The four rows are not equally robust. Rows 2-4 (momentum, momentum, energy) are
# well conditioned everywhere. Row 1 (charge) is nearly DEGENERATE in the dilute
# tail: n depends exponentially on phi, so a small error in D swings phi wildly.
# Solving all four together means the fragile row can veto the three robust ones —
# measured on the production IC at N=300, **68.2% of all recovery failures had the
# charge row alone out of tolerance while the hydro block had converged**, and 0%
# were hydro-only. Every one of those cells was then reset to vacuum, punching
# holes in a perfectly good temperature and velocity field.
#
# So: if the coupled solve fails, retry in stages.
#   1. (yT, u^x, u^y) from rows 2-4 with phi frozen.
#   2. phi from row 1 alone, by BISECTION — n is monotone in phi, so this cannot
#      diverge; if `D - nu^tau <= 0` no n >= 0 solves it and phi goes to its floor.
#
# For `LatticeHRGEOS` this is EXACT rather than an approximation: dP/dmu = 0
# identically (TWOD_PROGRAM.md §6b), so rows 2-4 do not contain phi at all. For an
# EOS where they do, the staged solve is approximate — which is why it is a
# FALLBACK after the coupled solve, never the primary path.
# ------------------------------------------------------------------------------

"""Bisect row 1 for φ at fixed (T, u). Returns (φ, ok)."""
@inline function _solve_phi_bisect(D::Float64, T::Float64, ux::Float64, uy::Float64,
                                   nux::Float64, nuy::Float64, eos)
    uτ = sqrt(1 + ux*ux + uy*uy)
    target = D - nu_tau_2d(ux, uy, uτ, nux, nuy)
    m = hq_mass(eos)
    if !(isfinite(target)) || target <= 0.0
        return -PHI_CAP, false          # no n >= 0 closes the row
    end
    lo = -PHI_CAP; hi = PHI_CAP
    @inbounds for _ in 1:200
        mid = 0.5*(lo + hi)
        _, nm, _ = eos_Pne(T, m + T*mid, eos)
        (isfinite(nm) && nm*uτ > target) ? (hi = mid) : (lo = mid)
    end
    φ = 0.5*(lo + hi)
    _, nf, _ = eos_Pne(T, m + T*φ, eos)
    return φ, isfinite(nf)
end

# ------------------------------------------------------------------------------
# cons -> prim
# ------------------------------------------------------------------------------
"""
    cons_to_prim_2d!(w, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos; ...)

Recover primitives, retrying from the FRESH SEED if the warm start fails.

🔴 2026-09-08. A warm start that has gone stale does not merely cost iterations —
it lands the Newton in a basin it cannot leave, and the cell was then declared a
failure and reset to COLD VACUUM (`T = T_MIN`, `u = 0`, `n = 0`) by
`_primitive_pass!`. That is a hole punched in `work.yT/ux/uy` for a cell holding
perfectly good matter, and MUSCL reconstructs its neighbours' faces through it.

MEASURED on the G9 single-event IC (N=200, 316 steps to τ=8): of the cells
`_primitive_pass!` could not recover, 9 429 560 were near-vacuum and EXPECTED,
and **8 625 held real matter — 27.3 per step. Re-running `cons_to_prim_2d!` on
exactly the same conserved state with a fresh seed inverted 100.0 % of them**
(0 genuinely non-invertible). That 8625 is exactly the `pf = 8625` gate G9
reports, so the whole of its "primfail" population was this, not bad states.

The fresh seed is a genuinely different starting point — `seed_from_conserved_2d`
inverts the IDEAL relations in closed form — so this is a second basin, not a
second pass at the same one. The retry runs only after a failure (27 cells per
step out of 40 000), so it is free.

No retry when the first attempt reported `PRR_VACUUM` (there is no state to find)
or when there was no warm start to begin with (the fresh seed is what already ran).
"""
function cons_to_prim_2d!(w::PrimRecWork2D,
                          D::Float64, Sx::Float64, Sy::Float64, E::Float64,
                          nux::Float64, nuy::Float64, Pi::Float64,
                          pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
                          τ::Float64, eos;
                          yT0::Float64 = NaN, φ0::Float64 = NaN,
                          ux0::Float64 = NaN, uy0::Float64 = NaN,
                          maxit::Int = 80, tol_res::Float64 = 1e-12)

    res = _cons_to_prim_attempt_2d!(w, D, Sx, Sy, E, nux, nuy, Pi,
                                    pixx, pixy, piyy, pieta, τ, eos;
                                    yT0 = yT0, φ0 = φ0, ux0 = ux0, uy0 = uy0,
                                    maxit = maxit, tol_res = tol_res)
    res[8] && return res                                   # converged
    w.last_reason === PRR_VACUUM && return res             # nothing to find
    (isfinite(yT0) && isfinite(ux0) && isfinite(uy0)) || return res   # was already cold

    return _cons_to_prim_attempt_2d!(w, D, Sx, Sy, E, nux, nuy, Pi,
                                     pixx, pixy, piyy, pieta, τ, eos;
                                     yT0 = NaN, φ0 = NaN, ux0 = NaN, uy0 = NaN,
                                     maxit = maxit, tol_res = tol_res)
end

"""
    _cons_to_prim_attempt_2d!(w, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos; ...)

ONE Newton solve. `cons_to_prim_2d!` above wraps this with the fresh-seed retry;
call that, not this.

Returns `(T, μ, ux, uy, n, e, P, ok)`.

`D` is `J^τ`, i.e. `U[iDtau]/τ` — the τ weight is stripped by the caller, exactly
as in the 1-D `rhs!` (src/rhs.jl:40).

`tol_res` is on the ROW-SCALED max-norm, so it is a relative tolerance on every
equation separately. It is 1e-12, not machine epsilon: the Jacobian is built by
finite differences, whose relative accuracy is ~1e-10 (h² truncation plus eps/h
roundoff at h = 1e-6), so the residual floor sits around 1e-14 and bounces. A
tolerance AT that floor rejected converged cells — measured on a realistic
non-symmetric IC, 1.5% of cell-updates were failed with
`PRR_LINESEARCH_FAILED` at `resnorm = 1.14e-14`, i.e. converged to machine
precision and thrown away. 1e-12 is still four orders tighter than the 1-D
production solver achieves at the dilute edge (3.3e-12, see
test_primrec2d_vs_1d.jl) and ten orders below any quoted number. The round-trip
gates measure what it actually costs rather than assuming.
"""
function _cons_to_prim_attempt_2d!(w::PrimRecWork2D,
                          D::Float64, Sx::Float64, Sy::Float64, E::Float64,
                          nux::Float64, nuy::Float64, Pi::Float64,
                          pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
                          τ::Float64, eos;
                          yT0::Float64 = NaN, φ0::Float64 = NaN,
                          ux0::Float64 = NaN, uy0::Float64 = NaN,
                          maxit::Int = 80, tol_res::Float64 = 1e-12)

    yT_lo = log(T_MIN)
    yT_hi = log(T_SOLVE_MAX)

    w.last_reason = PRR_UNSET
    w.last_iters = 0
    w.last_resnorm = NaN

    # ---- vacuum ----
    if !isfinite(E) || E <= E_VAC || !isfinite(D) || D < D_VAC
        w.last_reason = PRR_VACUUM
        w.last_resnorm = 0.0
        T = T_MIN; μ = 0.0
        P, _, _ = eos_Pne(T, μ, eos)
        return (T, μ, 0.0, 0.0, 0.0, max(E, 0.0), P, false)
    end

    ncharge = eos_has_charge_2d(eos)

    x = w.x; F = w.F; Fp = w.Fp; Fm = w.Fm; J = w.J; A = w.A; δ = w.δ
    n_unk = ncharge ? 4 : 3          # (yT, φ, ux, uy) or (yT, ux, uy)

    x[2] = (ncharge && isfinite(φ0)) ? clamp(φ0, -PHI_CAP, PHI_CAP) : 0.0
    if isfinite(yT0) && isfinite(ux0) && isfinite(uy0)
        x[1] = clamp(yT0, yT_lo, yT_hi)
        x[3] = clamp(ux0, -U_CAP_2D, U_CAP_2D)
        x[4] = clamp(uy0, -U_CAP_2D, U_CAP_2D)
    else
        sy, sx_, syy = seed_from_conserved_2d(E, Sx, Sy, x[2], eos)
        x[1] = clamp(sy, yT_lo, yT_hi); x[3] = sx_; x[4] = syy
    end

    # index of unknown k within x (skip φ when the EOS carries no charge):
    #   ncharge:  1->1 (yT)  2->2 (φ)  3->3 (u^x)  4->4 (u^y)
    #  !ncharge:  1->1 (yT)  2->3 (u^x) 3->4 (u^y)
    slot = ncharge ? (1, 2, 3, 4) : (1, 3, 4, 4)

    evalF_2d!(F, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
              ncharge = ncharge)
    if !all(isfinite, F)
        w.last_reason = PRR_NONFINITE_INITIAL
        T = clamp(exp(x[1]), T_MIN, T_SOLVE_MAX); μ = hq_mass(eos)
        P, _, e = eos_Pne(T, μ, eos)
        return (T, μ, 0.0, 0.0, 0.0, max(e, 0.0), P, false)
    end

    # ---- PER-EQUATION SCALES ----------------------------------------------
    # The four residual rows live on wildly different scales: with the charm EOS
    # the charge row is O(n) ~ 1e-4 while the energy row is O(e) ~ 1e3. Measuring
    # them against one global norm (as `1 + |Sx| + |Sy| + |E| + |D|` would) makes
    # the charge equation unconstrained at the requested tolerance, and — worse —
    # partial pivoting on the raw Jacobian then eliminates the charge direction
    # against an energy-sized pivot and reports a singular system. Both failure
    # modes were measured: 147/4000 states failed with PRR_LINSOLVE_FAILED at an
    # O(1) relative residual, and μ carried errors up to 1.4e-8 where it did
    # converge.
    #
    # Rows are therefore EQUILIBRATED: residual and Jacobian are both divided by
    # the row scale before the solve. Scaling rows does not change the Newton step
    # δ, so this is a pure conditioning fix, not a change of algorithm.
    sD = max(abs(D), 1e-12)
    sS = max(abs(Sx), abs(Sy), 1e-10*abs(E), 1e-12)
    sE = max(abs(E), 1e-12)
    rs1 = 1/sD; rs2 = 1/sS; rs3 = 1/sS; rs4 = 1/sE

    converged = false
    last_nrm = NaN
    r_lo = ncharge ? 1 : 2

    @inline scaled(r, v) = v * (r == 1 ? rs1 : r == 2 ? rs2 : r == 3 ? rs3 : rs4)

    for it in 1:maxit
        # convergence on the SCALED residual, active rows only
        nrm = 0.0
        for r in r_lo:4
            nrm = max(nrm, abs(scaled(r, F[r])))
        end
        last_nrm = nrm
        if nrm < tol_res
            converged = true
            w.last_reason = PRR_CONVERGED
            w.last_iters = it - 1
            break
        end

        # ---- finite-difference Jacobian (central) ----
        ok_jac = true
        for k in 1:n_unk
            s = slot[k]
            h = 1e-6 * max(1.0, abs(x[s]))
            xs = x[s]

            x[s] = xs + h
            evalF_2d!(Fp, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
                      ncharge = ncharge)
            x[s] = xs - h
            evalF_2d!(Fm, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
                      ncharge = ncharge)
            x[s] = xs

            if !(all(isfinite, Fp) && all(isfinite, Fm))
                ok_jac = false; break
            end
            inv2h = 1/(2h)
            for r in 1:n_unk
                rr = ncharge ? r : r + 1
                J[r,k] = scaled(rr, (Fp[rr] - Fm[rr]) * inv2h)
            end
        end
        if !ok_jac
            w.last_reason = PRR_NONFINITE_JACOBIAN
            break
        end

        # Pack the ACTIVE residual rows into scratch. Must not write through w.F,
        # which is aliased to F.
        Fact = Fm          # Fm is free once the Jacobian is assembled
        for r in 1:n_unk
            rr = ncharge ? r : r + 1
            δ[r] = 0.0
            Fact[r] = scaled(rr, F[rr])
        end

        if !solve_gauss_n!(δ, J, Fact, A, n_unk)
            w.last_reason = PRR_LINSOLVE_FAILED
            break
        end

        # ---- damped update with BACKTRACKING LINE SEARCH ----
        # A bare Newton step can increase the residual far from the solution; that
        # is precisely how the fixed-seed failures above stalled. Cap the raw step,
        # then halve until the scaled max-norm actually decreases.
        α = 1.0
        for k in 1:n_unk
            ad = abs(δ[k])
            ad > 1.0 && (α = min(α, 1.0/ad))
        end

        xs1 = x[1]; xs2 = x[2]; xs3 = x[3]; xs4 = x[4]
        accepted = false
        @inbounds for _ls in 1:12
            x[1] = xs1; x[2] = xs2; x[3] = xs3; x[4] = xs4
            for k in 1:n_unk
                x[slot[k]] += α*δ[k]
            end
            x[1] = clamp(x[1], yT_lo, yT_hi)
            ncharge && (x[2] = clamp(x[2], -PHI_CAP, PHI_CAP))
            x[3] = clamp(x[3], -U_CAP_2D, U_CAP_2D)
            x[4] = clamp(x[4], -U_CAP_2D, U_CAP_2D)

            evalF_2d!(F, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
                      ncharge = ncharge)
            if all(isfinite, F)
                trial = 0.0
                for r in r_lo:4
                    trial = max(trial, abs(scaled(r, F[r])))
                end
                # Armijo-lite: any decrease is enough for a Newton direction
                if trial < nrm*(1 - 1e-4*α)
                    accepted = true
                    break
                end
            end
            α *= 0.5
        end

        if !accepted
            # No downhill step along the Newton direction. Keep the last state and
            # let the residual test below reject it, rather than wandering.
            w.last_reason = PRR_LINESEARCH_FAILED
            break
        end
    end

    w.last_resnorm = last_nrm

    T  = clamp(exp(x[1]), T_MIN, T_SOLVE_MAX)
    μ  = ncharge ? (hq_mass(eos) + T*x[2]) : hq_mass(eos)
    ux = x[3]; uy = x[4]
    P, n, e = eos_Pne(T, μ, eos)

    # ---- RESIDUAL IS THE ARBITER, NOT HOW THE LOOP EXITED ----------------
    # A converged state cannot be improved further, so the backtracking line
    # search necessarily "fails" once the residual reaches its floor. Returning
    # `false` there rejected perfectly good cells: on a realistic non-symmetric IC
    # that produced a 1.5% "primfail" rate whose representative cell carried
    # reason = PRR_LINESEARCH_FAILED at resnorm = 1.14e-14 — converged, and thrown
    # away. The 1-D solver avoids this by always falling through to the residual
    # test (src/primrec.jl, after the `if ok && !converged` block); do the same.
    if !converged
        w.last_reason = (w.last_reason == PRR_UNSET) ? PRR_MAXIT : w.last_reason
    end

    # residual-based acceptance on the final state
    evalF_2d!(F, x, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
              ncharge = ncharge)
    resnorm = 0.0
    for r in r_lo:4
        w.last_rows[r] = abs(scaled(r, F[r]))
        resnorm = max(resnorm, w.last_rows[r])
    end
    w.last_resnorm = resnorm

    if !(isfinite(resnorm) && resnorm < tol_res)
        # ---- STAGED FALLBACK (see the block above) ----
        hyd = max(w.last_rows[2], w.last_rows[3], w.last_rows[4])
        if ncharge && isfinite(hyd) && hyd < tol_res
            # rows 2-4 already converged: keep them, close row 1 on its own.
            φb, okb = _solve_phi_bisect(D, T, ux, uy, nux, nuy, eos)
            x[2] = clamp(φb, -PHI_CAP, PHI_CAP)
            μ = hq_mass(eos) + T*x[2]
            P, n, e = eos_Pne(T, μ, eos)
            w.last_reason = okb ? PRR_CONVERGED : PRR_RESIDUAL_TOO_LARGE
            # The hydro block is trusted either way; that is the whole point.
            return (T, μ, ux, uy, max(n, 0.0), max(e, 0.0), P, true)
        end
        w.last_reason = PRR_RESIDUAL_TOO_LARGE
        return (T, μ, ux, uy, n, max(e, 0.0), P, false)
    end

    w.last_reason = PRR_CONVERGED
    return (T, μ, ux, uy, n, max(e, 0.0), P, true)
end

# ------------------------------------------------------------------------------
# prim -> cons
# ------------------------------------------------------------------------------
"""
    prim_to_cons_2d!(Umat, col, T, μ, ux, uy, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos, L)

Forward map into column `col` of a conserved-state matrix. Returns
`(ok, PrimIdealVisc2D)`.
"""
@inline function prim_to_cons_2d!(Umat::AbstractMatrix, col::Int,
                                  T::Float64, μ::Float64, ux::Float64, uy::Float64,
                                  nux::Float64, nuy::Float64, Pi::Float64,
                                  pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
                                  τ::Float64, eos, L::StateLayout2D)
    Tc = max(T, T_MIN)
    uτ = sqrt(1 + ux*ux + uy*uy)

    P, n, e = eos_Pne(Tc, μ, eos)
    if !(isfinite3(P, n, e)) || e <= 0.0
        return false, PrimIdealVisc2D(Tc, μ, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      nux, nuy, Pi, pixx, pixy, piyy, pieta, false)
    end

    weff = e + P + Pi
    Ptot = P + Pi
    Π = shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)

    Jt = n*uτ + nu_tau_2d(ux, uy, uτ, nux, nuy)

    @inbounds begin
        Umat[L.iDtau, col] = τ * Jt
        Umat[L.iSx,   col] = weff*uτ*ux + Π.tx
        Umat[L.iSy,   col] = weff*uτ*uy + Π.ty
        Umat[L.iE,    col] = weff*uτ*uτ - Ptot + Π.tt

        if L.hasNu
            Umat[L.iNux, col] = stored_from_phys(nux)
            Umat[L.iNuy, col] = stored_from_phys(nuy)
        end
        if L.hasPi
            Umat[L.iPi, col] = stored_from_phys(Pi)
        end
        if L.hasShear
            Umat[L.iPixx,  col] = stored_from_phys(pixx)
            Umat[L.iPixy,  col] = stored_from_phys(pixy)
            Umat[L.iPiyy,  col] = stored_from_phys(piyy)
            Umat[L.iPieta, col] = stored_from_phys(pieta)
        end
    end

    return true, PrimIdealVisc2D(Tc, μ, ux, uy, n, e, P,
                                 nux, nuy, Pi, pixx, pixy, piyy, pieta, true)
end
