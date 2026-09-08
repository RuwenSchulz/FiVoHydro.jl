#!/usr/bin/env julia
# =============================================================================
# main2IS2.jl — 5-component Israel–Stewart HQ solver on a fixed background
#
# Evolves [α, νr, πQr, πQperp, PiQ] using a FiVo-local second-moment builder
# derived from Fluidum's HQ_const_BG_2nd_moment mode, but expressed in the
# physical second-moment variables.
#
# Entry point for the static comparison: run_static_IS2_test(...)
# =============================================================================
module hydro_current_IS2

using LinearAlgebra
using Printf
using Logging
using CSV
using Tables
using JLD2
using Interpolations
using SpecialFunctions

const _SRC = joinpath(@__DIR__, "src")
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "grid.jl"))
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "io.jl"))
include(joinpath(_SRC, "is2_second_moment_builder.jl"))
include(joinpath(_SRC, "hq_consistent_firstmoment.jl"))
include(joinpath(_SRC, "hq_consistent_m2.jl"))
include(joinpath(_SRC, "logging_setup.jl"))

const TWO_PI = 2π
const IS2_FREEZE_PDE_ALPHA = get(ENV, "FIVO_FREEZE_PDE_ALPHA", "1") == "1"
const IS2_USE_Q_RECOVERY = get(ENV, "FIVO_USE_Q_RECOVERY", "1") == "1"
# Sign of the 2nd-moment back-coupling c_M in the assembled (LHS-form) system.  AUDIT (CMExperiment
# DERIVATION_CHECK.md): the derivation (Tex/HydroFPderivation/FP_Hydro_matching.tex eq:diffusion_split)
# has the feedback as −c_M ∂π on the RHS with c_M=+D_s/T; build_IS2_system! moves all derivative terms
# to the LHS, flipping it to +c_M ⇒ the implemented coefficient must be +D_s/T ⇒ default +1.0.  The old
# default −1.0 was the ELLIPTIC/Hadamard branch (grid-refinement blows up faster as Nr↑).  Env-overridable
# so the elliptic branch stays reproducible for the well-posedness scan.
const IS2_CM_SIGN = parse(Float64, get(ENV, "FIVO_CM_SIGN", "+1.0"))
# τ_M/η_M convention. τ_M and τ_n share the SAME normalization so the kinetic ratio τ_M/τ_n = ½K4K2/K3²
# ≈ 0.55 (FP_Hydro_matching eq:tauM_Bessel) is preserved.
#
# ⚠ CORRECTED 2026-08-01 — this paragraph used to state that "the LP1 Pb+Pb PRODUCTION uses the
# degeneracy-weighted (÷g_hq) convention ... is2_dropin sets FIVO_IS2_TAUN_DEGENERACY=1 +
# FIVO_IS2_TAUM_DEGENERACY=1". That was true only until 2026-07-21, when Fluidum's τ_diffusion_hadron
# itself went BARE (Eq. 30 of 2205.07692). NEITHER production path sets these flags any more:
# LangevinPaper1/is2_dropin.jl says so explicitly ("default to 0 = bare, so we set nothing here") and
# LangevinPaperOO/is2_dropin.jl sets nothing either, so BOTH systems evolve with the bare module
# default and are mutually consistent. Verified by measurement, not by reading: the cached
# tau_diff_spline reads ≈1.81 fm at T=T_fo in both projects (bare); an 0.30 fm reading means a
# pre-07-21 stale cache, which is what O+O's diagnostic field turned out to be.
# The flags remain for the convention study only. Keep τ_n and τ_M in the SAME convention (mixing
# breaks the 0.55 ratio by g_hq=6).
const IS2_TAUM_DEGENERACY = get(ENV, "FIVO_IS2_TAUM_DEGENERACY", "0") == "1"
# FIVO_IS2_TAUN_DEGENERACY=1 divides τ_n by g_hq too, so FiVo's τ_n MATCHES Fluidum's second-moment
# τ_diffusion_hadron EXACTLY (which carries the ÷normalization ∝ g_hq). Use TOGETHER with
# FIVO_IS2_TAUM_DEGENERACY=1 (keeps τ_M/τ_n = ½K4K2/K3² ≈ 0.55) and FIVO_IS2_CAUSAL_COEFF=0 (no clamp,
# so τ_n is exactly Fluidum's). This is the "match Fluidum's τ_n/τ_M" convention (operator, 2026-07-19).
const IS2_TAUN_DEGENERACY = get(ENV, "FIVO_IS2_TAUN_DEGENERACY", "0") == "1"
# Pure multiplier on τ_n, DIAGNOSTIC ONLY. DEFAULT 1.0 = production (byte-identical).
# FIVO_IS2_TAUN_SCALE=0.25 → quarter the memory; →0 approaches the Navier–Stokes limit ν^r → ν_NS.
# See the note at the τn use site for what this is for. Scales τ_n ONLY, so keep c_M = 0 when using it.
const IS2_TAUN_SCALE = parse(Float64, get(ENV, "FIVO_IS2_TAUN_SCALE", "1.0"))
# FIVO_IS2_TAUPI_REL=1 gives the TRACE projection (Π_Q) its own relaxation time instead of reusing
# τ_M.  The 14-moment ansatz fixes τ_π = τ_M from the rank-2 sector but says NOTHING about the trace:
# its only scalar is removed by Landau matching, so the inertia coefficient of the Π_Q equation is a
# 0/0 and the production code simply SETS τ_Π = τ_π = τ_M.  Restoring the minimal matching-orthogonal
# scalar φ_Π = k² − 3I₃₁/I₁₀ gives (LangevinPaper1 main.tex, Eq. B22)
#     τ_Π/τ_π = 5/2 − (3/2)·K₃²(z)/[K₂(z)K₄(z)] ,   z = M/T,
# i.e. 1.14 at freeze-out (z=9.6) rising to ~1.27 in the hot core — the trace sector is SLOWER.
# DEFAULT 0 = production (τ_Π ≡ τ_M, byte-identical).  DIAGNOSTIC: see is2_taupi_experiment.jl.
const IS2_TAUPI_REL = get(ENV, "FIVO_IS2_TAUPI_REL", "0") == "1"
const IS2_ORIGIN_ODD_FIRST_ORDER_CELLS = parse(Int, get(ENV, "FIVO_ORIGIN_ODD_FIRST_ORDER_CELLS", "4"))
const IS2_VACUUM_N_LO = parse(Float64, get(ENV, "FIVO_VACUUM_N_LO", "1e-6"))
# 2026-07-26: PRODUCTION DEFAULT LOWERED 1e-2 → 2e-3.  The old 1e-2 damped every dU by 2-5× across the
# whole freeze-out shell (measured w = 0.76-0.86 at r=7 fm, 0.18-0.45 at r=8, 0.02-0.17 at r=9), leaving
# the shell charm ∫2πrτn dr|_{7..9fm} up to 12% below Fluidum on the identical background.  2e-3 is the
# SMALLEST threshold at which the α=±200 clamp still never engages — below it the clamp binds at
# r≈8.6 fm, i.e. inside the shell, not in harmless far vacuum (1e-3 → 25 cells, 5e-4 → 319 cells and
# charm drift 8× worse).  Measured at 2e-3: shell 12% low → 1.1% low (linear τ=4) and 9% low → 0.0%
# (linear τ=8), const 0.995/0.999/0.989, charm drift unchanged at 5.8e-5, zero clamped cells, zero
# solver failures.  See Julia/Projects/LangevinPaper1/HydroFieldsDiagnostic/README.md for the full scan.
# ⚠️ This CHANGES every FiVo charm result — all FiVo charm products must be re-solved and re-ingested.
const IS2_VACUUM_N_HI = parse(Float64, get(ENV, "FIVO_VACUUM_N_HI", "2e-3"))
# ── TEMPERATURE-GATED vacuum ramp (2026-07-26) ──────────────────────────────────────────────────
# The density-gated ramp below (`_vacuum_weight`) cannot separate "vacuum" from "dilute but physical
# fluid", because the charm freeze-out density and the regularization threshold overlap:
#   • leaving the T=T_fo contour undamped needs n_hi ≲ 2.7e-4 (measured, min n on the linear contour);
#   • keeping the α=±200 clamp from engaging needs n_hi ≳ 2e-3 (1e-3 already pins 25 cells).
# Those windows miss each other by ~an order of magnitude, so at the shipped n_hi=1e-2 the ramp damps
# every dU by 2-5x at r≈7-9 fm — i.e. across the entire freeze-out surface, where FiVo's density then
# sits ~40% below Fluidum's.
# ⚠️ TESTED AND IT DOES NOT WORK ON THE LP1 BACKGROUND — kept as a documented negative result, OFF by
# default. The idea was that T discriminates where n cannot (fluid ⇔ T ≥ T_fo). It fails because this
# background has a WARM FLAT TAIL: at τ=12, T = 0.146 at r=10 and 0.138 at r=12 while n has fallen to
# 7e-4 and below; at r=15, T = 0.103 with n = 2e-27. So any T gate low enough to leave the freeze-out
# contour alone also leaves the whole far tail undamped, exactly where dn_dα → 0 — and α runs straight
# into the ±200 clamp (measured, linear/Nr=600/cM=0: T_HI=0.12 → 551 clamped cells, T_HI=0.08 → 428,
# charm drift degraded 5.6e-5 → 4.6e-3). Density really is the conditioning parameter of the α
# equation, so the regularizer has to be density-based; the practical fix is simply a SMALLER n_hi
# (3e-3 gives 0 clamped cells, drift 5.6e-5, and r=8 density within 2% of Fluidum).
# Set FIVO_VACUUM_T_HI > 0 to use the T ramp INSTEAD of the density ramp; the hard n_lo cutoff is kept
# either way as a pure-vacuum backstop. T_HI=0 (default) reproduces the historical density-gated
# behaviour bit-for-bit (verified: identical to 5 s.f. on the linear Nr=300 audit).
const IS2_VACUUM_T_HI = parse(Float64, get(ENV, "FIVO_VACUUM_T_HI", "0.0"))
const IS2_VACUUM_T_LO = parse(Float64, get(ENV, "FIVO_VACUUM_T_LO", "0.0"))

# ── RELATIVE vacuum ramp (2026-08-02) ───────────────────────────────────────────────────────────
# FIVO_VACUUM_N_REL_HI > 0 measures dilution against the CURRENT SLICE MAXIMUM n_ref(τ) instead of
# against an absolute density, i.e. w ramps over n/n_ref ∈ [REL_LO, REL_HI].  DEFAULT 0 = off =
# the absolute thresholds, byte-identical production.
#
# 🔴 WHY THIS EXISTS.  The absolute N_HI=2e-3 was calibrated on Pb+Pb, which carries ~24 charm quarks;
# O+O carries 0.27.  Measured on the O+O freeze-out contour, the MEDIAN density is 1.2-1.4e-4 — below
# the Pb+Pb contour MINIMUM of 2.7e-4 quoted above — so w ≈ 0.06 over the outer branch and ≈ 0.44-0.55
# over the re-entrant shell, and by τ≈5 every cell inside r=5 fm is damped.  Since the weight
# multiplies the WHOLE RHS including dU[2]=dν^r/dτ, that is a time dilation of the charm sector by
# 1/w, i.e. an EFFECTIVE τ_n = τ_n/w: ≈27 fm on the outer branch and ≈4.2 fm on the inner shell
# against the physical 1.81 fm.  Measured effect on ν^r at the freeze-out surface (identical solves,
# only this threshold changed — `Projects/LangevinPaperOO/diagnose_vacuum_ramp.jl`):
#   const  outer |ν^r/n| 0.028 vs 0.230 undamped (8× suppressed) | inner 0.168 vs 0.118 (1.4× inflated)
#   linear outer 0.022 vs 0.639 (29× suppressed)                 | inner 0.261 vs 0.214 (1.2× inflated)
# — the current is wrong in BOTH directions at once, which distorts the branch-to-branch SHAPE by
# 12-29× and is what the λ-scan reads as "λ>0 outside, λ≈−1 on the re-entrant shell".
#
# A ratio is the right variable because the regularizer's actual job is to switch off the far
# exterior where dn_dα → 0 *relative to the fluid*, and that notion is scale-free in both system size
# and Bjorken dilution, neither of which an absolute density tracks.
#
# ⚠ NOT VALIDATED ON Pb+Pb YET.  Default stays off until a Pb+Pb solve confirms the α clamp still
# never engages and the charm drift is unchanged.  See the T-gate above for a documented case of a
# reasonable-looking reparametrization that failed on the Pb+Pb background.
const IS2_VACUUM_N_REL_HI = parse(Float64, get(ENV, "FIVO_VACUUM_N_REL_HI", "0.0"))
const IS2_VACUUM_N_REL_LO = parse(Float64, get(ENV, "FIVO_VACUUM_N_REL_LO", "1e-4"))
# Slice reference density n_ref(τ), refreshed once per RHS evaluation at the top of _compute_dUdt!
# (which also covers _second_moment_eigen_rhs!, called from inside it).
const IS2_VACUUM_NREF = Ref(0.0)

# Kinematic bound |nu^r| <= FIVO_IS2_NU_BOUND * n, enforced in _recover_alpha_from_q! (see there for
# why that is the only place it matters). 0 = OFF = production, byte-identical. 1.0 is the exact
# bound; a slightly smaller value (0.9) leaves margin against the recovery hitting its density floor.
# This and the RELATIVE ramp above are two halves of ONE fix and should be evaluated together: the
# absolute ramp is currently doing two unrelated jobs at once — masking this runaway AND damping the
# physical freeze-out region — and only the second is a mis-calibration.
const IS2_NU_BOUND = parse(Float64, get(ENV, "FIVO_IS2_NU_BOUND", "0.0"))
# Frame the bound is written in: "lab" reproduces the original |nu^r| <= f n, "lrf" the physical
# |nu*_r| <= f n <=> |nu^r| <= f n u^tau. See the derivation at the clamp in _recover_alpha_from_q!.
#
# 🔴 THE DEFAULT IS THE *WRONG* FRAME ON PURPOSE, AND f DOES NOT TRANSFER BETWEEN FRAMES.
# "lrf" is the physically correct statement (the charm drifts slower than light RELATIVE TO THE
# FLUID). But the production f = 0.7 was calibrated EMPIRICALLY IN THE LAB FRAME against the
# depletion runaway (dpm_recipes.jl: 0.9 -> 127 runaway cells, 0.7 -> 0), and "lrf" loosens the
# same f by one u^tau (~1.37 on Sigma_fo). Measured A/B, O+O bulk, f = 0.7, Nr = 300, tau 0.4->5:
#     lab: max|nu^r|/n = 0.700   J^tau<=0 in  579 cell-steps,  7 cells on the alpha floor
#     lrf: max|nu^r|/n = 1.167   J^tau<=0 in 1884 cell-steps, 16 cells on the alpha floor
# i.e. at fixed f the "correct" frame puts the lab-frame ratio ABOVE 1 -- it stops enforcing the
# admissibility bound it exists for -- and re-opens the depletion feedback. So switching the frame
# is not a free correction: it needs its own calibration (f_lrf ~ f_lab/u^tau ~ 0.5, which the
# recipe's own scan already shows is runaway-free) and re-mints every O+O charm product.
# Until that scan is run, "lab" stays the default and the paper must quote the LAB-frame bound
# (f = 0.7 in lab variables is a physical drift bound of f/u^tau ~ 0.51 c, not 0.7 c).
const IS2_NU_BOUND_LRF = lowercase(get(ENV, "FIVO_IS2_NU_BOUND_FRAME", "lab")) == "lrf"
# Saturate smoothly (b*tanh(nu/b)) instead of clamping. Off = the exact projection, as before.
const IS2_NU_BOUND_SMOOTH = get(ENV, "FIVO_IS2_NU_BOUND_SMOOTH", "0") == "1"
# Knee: the fraction of the bound below which the smooth map is EXACTLY the identity. Must be < 1.
const IS2_NU_BOUND_KNEE = parse(Float64, get(ENV, "FIVO_IS2_NU_BOUND_KNEE", "0.8"))
# Floor J^tau at this fraction of the slice reference density instead of zeroing nu^r when the
# transported charge comes out non-positive. 0 = historical (zero the current).
const IS2_JTAU_FLOOR = parse(Float64, get(ENV, "FIVO_IS2_JTAU_FLOOR", "0.0"))
# Project the (n, nu*_r) pair onto the admissible cone |nu*_r| <= f n instead of clamping a single
# component. 0 = off. This is the covariant statement (N^mu timelike) and, unlike the clamp, it is
# the identity ON the cone, so it engages without a kink. See _recover_alpha_from_q!.
const IS2_CONE_PROJECT = parse(Float64, get(ENV, "FIVO_IS2_CONE_PROJECT", "0.0"))
const IS2_CONE_HITS    = Ref(0)
# How often the bound actually DID something. A bound that never binds is inert and the solution is
# the solver's; a bound that binds on a large fraction of cell-steps is SHAPING the answer, and any
# comparison against another code has to say so. `IS2_JTAU_NEG` counts the harsher branch: J^tau <= 0
# means the recovered density would be negative, and the code responds by ZEROING the current there.
const IS2_NU_BOUND_HITS = Ref(0)
const IS2_JTAU_NEG      = Ref(0)

# ── THE SAME BOUND, MADE UNREACHABLE INSTEAD OF ENFORCED (FIVO_NU_RAPIDITY) ─────────────────────
# `IS2_NU_BOUND` above is a PROJECTION: it lets the RK step produce a superluminal nu^r and then puts
# it back on the boundary. Exact, but it discards information every time it binds, and a state that
# sits ON the bound is one the closure has no business being in.
#
# The alternative is a CHANGE OF INTEGRATION VARIABLE. nu = nu*_r/n = nu^r/(n u^tau) is a VELOCITY,
# so the object with no kinematic bound is its RAPIDITY, phi = artanh(nu). Integrate phi and the
# constraint |nu| < 1 is not enforced, it is unreachable: phi ranges over all of R and
# nu = tanh(phi) can never leave (-1, 1) whatever the right-hand side does.
#
#   phi = artanh(nu^r / N),   N = n u^tau = n_eq e^alpha u^tau      (frozen at the step's start)
#   dphi/dtau = (dnu^r/dtau) / N / (1 - (nu^r/N)^2)
#   nu^r_new  = N tanh(phi_new)
#
# WHAT THIS IS AND IS NOT. It is not a clip: there is no branch, no threshold, no max/min, and tanh
# is applied at every step whatever the state. It is not a new closure either -- phi is a smooth
# bijection of nu on (-1,1), so RK4 in phi is the SAME fourth-order scheme in a different chart, and
# it reduces to the current update to O(nu^2) since tanh(phi) = phi + O(phi^3). What it costs is that
# N is held fixed across the step, so the bound realised is |nu| < N(tau)/N(tau+dt) = 1 + O(dt)
# rather than exactly 1; each step re-anchors N, so there is no drift.
#
# 🔴 CELLS THAT ARE ALREADY OUTSIDE cannot be mapped (artanh is undefined), and those fall back to
# the plain additive update rather than being silently moved. They are COUNTED in the diagnostics as
# `nu_rapidity_fallbacks` -- a nonzero count means the run STARTED outside the bound, which is a
# statement about the initial condition, not about this scheme.
#
# 🔴 MEASURED 2026-08-03 (Pb+Pb, `CMExperiment/diag_cm_subluminal.jl`) — DO NOT PROMOTE THIS AS IS.
# It behaves exactly as designed and is still NOT SUFFICIENT on its own, for a reason worth writing
# down: `_recover_alpha_from_q!` runs AFTER the step and re-derives n from the conserved J^tau, so a
# bound imposed on nu^r during the RK stages is undone one line later when n moves. In the
# regulator-free c_M = 0 linear run the chart changes nothing (max|nu| 1.9e15 either way, with 79755
# fallback cell-steps = cells that were outside before the step began), because the runaway lives in
# cells whose DENSITY has underflowed, not in cells whose current is too large -- and those carry
# 0.00% of the emission. The existing `IS2_NU_BOUND` clamp, which acts inside the recovery, does
# better on the contour (fraction of tau above |nu| = 0.8 falls 19.0% -> 4.1%) and still does not
# remove it. The place for this chart, if it is ever wanted, is INSIDE the recovery beside that
# clamp, where n and nu^r are determined together.
# ✅ VERIFIED: flag off reproduces the shipped solution exactly (max|nu| 0.2213 / 0.0824 to 4 d.p.);
# flag on with healthy physics is inert (identical numbers, 0 fallbacks). The real cure for the
# superluminal current is c_M = D_s/T, which brings max|nu| to 0.074 (const) / 0.082 (linear)
# REGULATOR-INDEPENDENTLY -- a bound is a safety net, not the fix.
#
# Default off; when off, not one arithmetic operation below changes.
const IS2_NU_RAPIDITY = get(ENV, "FIVO_NU_RAPIDITY", "0") == "1"

# ── THE FUGACITY FLOOR ───────────────────────────────────────────────────────────────────────────
# 🔴 2026-08-03. Both α call sites used `clamp(α, -200.0, 200.0)`. That is an OVERFLOW GUARD, not a
# physical bound: real charm fugacities on Σ run −1…−5, so ±200 sits ~40× outside anything the fluid
# ever reaches. In the dilute exterior α → −∞ (α = log(n/n_th) diverges logarithmically as n → 0),
# so the guard pinned whole shells at exactly −200 and the spline through that wall overshot to
# −200.325 and worse.
#
# The damage is not in the hydro — it is in every CONSUMER, because −200.325 is a finite Float64 and
# so participates silently in arithmetic:
#   * differentiate it  → |∂α/∂r| ≈ 5000 vs 73 in a clean field (wrecked the O+O δ_cM figure)
#   * exponentiate it   → e^−200 ≈ 0, so a masked cell is silently deleted from a Cooper–Frye integral
#   * α_h + log(n/n_h)  → the −200 CANCELS, resurrecting a masked cell (blew the O+O moment floor up
#                         by 6.5e12 and made "the floor" score 3× worse than plain hydro f₀)
# Measured extent: O+O linear 548/156821 cells; Pb+Pb const 3436 and linear 6164 of 152071, reaching
# −214 and −228 at the r=12 grid edge. This is solver-wide, not an O+O quirk.
#
# THE FIX IS A FLOOR, NOT A TIGHTER CLAMP. The distinction that matters is whether α REACHES the
# bound continuously. Descending from −5, α passes through −20 smoothly and then flatlines: that is
# a kink (C⁰, ∂α → 0), not a cliff. At −200 the evolution never gets there by physics — it lands
# there as garbage, so the field jumps and the derivative explodes. e^−20 ≈ 2e-9 keeps n finite and
# representable (no 1e-91 underflow) while being utterly negligible physically.
#
# Set FIVO_IS2_ALPHA_MIN=-200 to recover the pre-2026-08-03 behaviour exactly.
const IS2_ALPHA_MIN = parse(Float64, get(ENV, "FIVO_IS2_ALPHA_MIN", "-20.0"))
const IS2_ALPHA_MAX = parse(Float64, get(ENV, "FIVO_IS2_ALPHA_MAX", "200.0"))
# Softness of the floor, in units of alpha. 0 = hard max() (the 2026-08-03 first form).
const IS2_ALPHA_SOFT = parse(Float64, get(ENV, "FIVO_IS2_ALPHA_SOFT", "1.0"))

"""
    _alpha_floor(α) -> Float64

Apply the fugacity floor SMOOTHLY, and the (genuine) overflow guard on the high side hard.

A hard `max(α, α_min)` is a KINK: ∂α drops discontinuously to zero where it activates, which is
the residual |∂α/∂r| ≈ 456 left after the first version of this fix (against 73 in a field that
never touches the floor). Softplus removes it:

    α = α_min + s·log1p(exp((α_raw − α_min)/s))

C^∞ everywhere, → α_raw far above the floor, → α_min far below. It is TRANSPARENT where the physics
lives: with s = 1 and α_min = −20, a physical α = −5 sits 15 above the floor and is shifted by
log1p(e^−15) ≈ 3e−7. Only the dilute exterior, where α was running to −200 and meaning nothing, is
affected at all.

The high side stays a hard clamp: α_max = 200 is a true overflow guard, physical α never approaches
it, and a soft cap there would perturb nothing while costing an exp() per cell.
"""
@inline function _alpha_floor(α::Float64)
    isfinite(α) || return IS2_ALPHA_MIN
    if IS2_ALPHA_SOFT > 0.0
        Δ = (α - IS2_ALPHA_MIN) / IS2_ALPHA_SOFT
        # log1p(exp(Δ)) overflows for large Δ and underflows harmlessly for very negative Δ
        αf = IS2_ALPHA_MIN + IS2_ALPHA_SOFT * (Δ > 30.0 ? Δ : log1p(exp(Δ)))
    else
        αf = max(α, IS2_ALPHA_MIN)
    end
    return min(αf, IS2_ALPHA_MAX)
end

"""
Damping weight for the dilute/vacuum exterior, applied to every dU.  Returns 1 in the fluid.
`n` is the local charm density, `T` the local (floored) temperature.
"""
@inline function _vacuum_weight(n::Float64, T::Float64)
    n <= IS2_VACUUM_N_LO && return 0.0          # true vacuum backstop, ALL modes
    if IS2_VACUUM_T_HI > 0.0
        T >= IS2_VACUUM_T_HI && return 1.0
        lo = IS2_VACUUM_T_LO
        return T <= lo ? 0.0 : clamp((T - lo) / (IS2_VACUUM_T_HI - lo), 0.0, 1.0)
    end
    if IS2_VACUUM_N_REL_HI > 0.0 && IS2_VACUUM_NREF[] > 0.0
        hi = IS2_VACUUM_N_REL_HI * IS2_VACUUM_NREF[]
        lo = max(IS2_VACUUM_N_REL_LO * IS2_VACUUM_NREF[], IS2_VACUUM_N_LO)
        n >= hi && return 1.0
        return n <= lo ? 0.0 : clamp((n - lo) / (hi - lo), 0.0, 1.0)
    end
    n >= IS2_VACUUM_N_HI && return 1.0
    return clamp((n - IS2_VACUUM_N_LO) / (IS2_VACUUM_N_HI - IS2_VACUUM_N_LO), 0.0, 1.0)
end
# Cap on the characteristic speed used for the CFL Δτ estimate.
# AUDIT CORRECTION (Projects/FiVoBenchmark/bench_is2_causality.jl, finding I-1): the eigenvalues
# |λ|>1 of At⁻¹·Ax are NOT vacuum artifacts — they occur for PHYSICAL high-T/high-α states where the
# IS2 diffusion-sector rest-frame v_sig=√(κ/(χτ_n))≈1.07>c (a side-effect of mixing grand-canonical κ
# with canonical χ), boost-amplified to ~24c at v≈0.95. Vacuum cells are actually causal (|λ|≈0.98).
# Capping the CFL at 10c while the HLL/Rusanov dissipation uses the TRUE c_face under-resolves the
# timestep (c·Δτ/Δr>1 ⇒ explicit instability). So for PHYSICAL cells the CFL now uses the true speed;
# the cap is kept only as a guard for genuinely pathological (near-singular A_t) cells.
const IS2_MAX_SIGNAL_SPEED = parse(Float64, get(ENV, "FIVO_MAX_SIGNAL_SPEED", "10.0"))
# FIVO_IS2_CAUSAL_CFL=0 restores the old cap-everything behaviour (for regression comparison).
const IS2_CAUSAL_CFL = get(ENV, "FIVO_IS2_CAUSAL_CFL", "1") == "1"
@inline _cfl_speed(c::Float64, n::Float64) =
    (IS2_CAUSAL_CFL && n >= IS2_VACUUM_N_LO) ? c : min(c, IS2_MAX_SIGNAL_SPEED)
# COEFFICIENT-LEVEL causality fix (audit finding I-1): the diffusion rest-frame signal speed
# v_sig²=κ/(χ τ_n) exceeds c for high-T/α because κ is grand-canonical while χ=dn/dα is canonical.
# Enforcing τ_n ≥ κ/χ makes v_sig ≤ c everywhere while leaving the (already subluminal) majority of
# states untouched. This intentionally diverges from Fluidum's (acausal) τ_n ONLY where Fluidum is
# acausal. FIVO_IS2_CAUSAL_COEFF=0 restores the exact Fluidum-matched τ_n (for rhs_compare parity).
const IS2_CAUSAL_COEFF = get(ENV, "FIVO_IS2_CAUSAL_COEFF", "1") == "1"
const IS2_DRIVE_DISSIP     = parse(Float64, get(ENV, "FIVO_DRIVE_DISSIP", "1.0"))   # Rusanov coeff for the intra-cell drive (rows 1-2)
const IS2_DRIVE_ENABLE     = get(ENV, "FIVO_DRIVE_ENABLE", "1") == "1"              # intra-cell drive on (set 0 to reproduce the pre-fix bug)

# Charm-diffusion κ normalization. Fluidum's κ = D_s · normalization(T,μ), where the charm hadron
# list (MainFluidum/FluidumInit/charm.data) is a SINGLE species (Charm: q=1, deg=6, m=1.5) → the
# "hadron sum" Σ q² n_i is just the grand-canonical charm density. FiVo's ConformalHQEOS/LatticeHRGEOS
# charm density (g_hq=6, m=1.5) is algebraically identical, so with a grand-canonical charm EOS
# (canon_factor=1.0) the two κ definitions coincide EXACTLY → factor is 1, not measured. The old
# "1.05" merely compensated the LatticeHRGEOS default canon_factor≈0.952 (hardcoded N=21.55).
const HQ_KAPPA_RESONANCE_FACTOR = parse(Float64, get(ENV, "FIVO_KAPPA_RESONANCE_FACTOR", "1.0"))

# ── THERMODYNAMICALLY CONSISTENT FIRST MOMENT (2026-08-25). DEFAULT OFF = byte-identical
# production. Adds to src[2] the five source terms the homogeneous-rest-frame derivation of the
# shipped ν^r row drops (pressure-gradient ∇⊥T channel, inertial τ_n n a^r, ν·∇u, τ_n ν D ln h,
# geometric dilution τ_n ν(u^τ/τ+u^r/r)) — see src/hq_consistent_firstmoment.jl for the physics
# and provenance (xAct: Julia/tools/derive_hq_consistent.wls 7/7; Fluidum twin gated to 1e-15).
# A Ref so a driver can toggle per-solve without reloading the module
# (runner: Tex/MaxEntHydro/run_fivo_consistent.jl; gates: Tex/MaxEntHydro/diag_fivo_consistent_gates.jl).
#
# ⚠ TWO NAMES, ONE FLAG (documented 2026-09-08). The live production switch is NOT this module's
# `FIVO_HQ_CONSISTENT` — nothing in phd-git has ever set it; a repo-wide grep finds only this line.
# O+O production turns the closure on by ASSIGNING THE REF from the project side,
# `Projects/LangevinPaperOO/is2_dropin.jl:139`, which reads `FIVO_IS2_CONSISTENT` (and refuses to
# run without OO_IS2_OUT_SUFFIX, so a different closure cannot overwrite the shipped solve).
# `FIVO_IS2_CONSISTENT` is accepted here too, so reading the module no longer suggests a switch that
# does nothing. Both default to "0" ⇒ default-off behaviour is unchanged and byte-identical.
# Pb+Pb does NOT come through here at all: LP1_CLOSURE=consistent resolves Fluidum's
# :HQ_const_BG_consistent_5f matrix (charm_hydro_consistent), not this module.
const IS2_CONSISTENT_FM = Ref(get(ENV, "FIVO_IS2_CONSISTENT",
                                  get(ENV, "FIVO_HQ_CONSISTENT", "0")) == "1")
# ── THE CONSISTENT SECOND MOMENT (2026-09-08), src/hq_consistent_m2.jl ────────────────────────────
# INDEPENDENT of IS2_CONSISTENT_FM on purpose: the first and second moments switch separately, which
# is what makes a four-way attribution (shipped/consistent × moment) possible. Default OFF = the
# shipped `_second_moment_eigen_rhs!` rows, byte-identical to every number produced before this file.
#
# ⚠ The shipped rows are NOT the ∇ν-only limit of the consistent system: their σ_(ν) drive carries the
# OPPOSITE SIGN (measured, exactly −1× — Tex/MaxEntHydro/diag_fivo_m2_gates.jl gate M3 reports
# |ratio−1| = 2.0000 at all 24 points) and they carry coordinate-transport couplings among
# (π_r, π_perp, Π_Q) that Δ-projected Dπ does not have. So this is a REPLACEMENT of rows 3-5, not an
# addition to them. Gate M4: the replacement reproduces Fluidum's `hqc_m2_rows(:consistent)` to
# 1.7e-15 over 24 points × 3 rows.
const IS2_CONSISTENT_M2 = Ref(get(ENV, "FIVO_IS2_CONSISTENT_M2", "0") == "1")
# ── Consistent 5-field (2026-08-25): with IS2_CONSISTENT_FM AND use_cM, the c_M back-coupling is
# applied as an EXPLICIT SOURCE built from the complete abstract-route row (hq_cm_force — includes
# the hoop-stress and τ-redshift geometric pieces the legacy basis-route matrix entries miss), and
# the legacy matrix cM entries are zeroed. The ∂τζ part uses the previous RK stage's dU[3]/dU[5]
# (scratch below) — FiVo's split computes dπ/dτ after dν/dτ, so an in-stage value does not exist;
# O(dt) lag on one subdominant O(ur) term. Damped by n/(n+IS2_CM_NFLOOR) exactly like the Fluidum
# twin (HQC_CM_NFLOOR, three orders below n_fo — <0.2% at the freeze-out surface). Flag off, or
# use_cM=false: byte-identical production.
const IS2_CM_NFLOOR = parse(Float64, get(ENV, "HQC_CM_NFLOOR", "1e-6"))
const IS2_CM_DT3 = Ref(Float64[])     # lagged dU[3] (dπQr/dτ) for the consistent cM source
const IS2_CM_DT5 = Ref(Float64[])     # lagged dU[5] (dΠQ/dτ)

# ═══════════════════════ Grid ═══════════════════════════════════════
struct IS2Grid1D
    r::Vector{Float64}
    rF::Vector{Float64}
    dr::Vector{Float64}
end

function IS2Grid1D(Nr::Int, rmax::Float64)
    dr_val = rmax / Nr
    r = [(i - 0.5) * dr_val for i in 1:Nr]
    rF = [(i - 1) * dr_val for i in 1:(Nr + 1)]
    return IS2Grid1D(r, rF, fill(dr_val, Nr))
end

# ═══════════════════════ Background ═════════════════════════════════
struct IS2Background
    r_grid::Vector{Float64}
    t_grid::Vector{Float64}
    T_spl::Any
    ur_spl::Any
    v_spl::Any
    α_spl::Any
    n_spl::Any
    nur_spl::Any
    kappa_spl::Any
    tau_diff_spl::Any
    dtT_spl::Any       # optional: pre-computed derivative splines
    drT_spl::Any
    dtur_spl::Any
    drur_spl::Any
end

function load_IS2_background(path::AbstractString)
    isfile(path) || error("Background file not found: $path")
    jldopen(path, "r") do f
        _k(s) = haskey(f, s) ? s : haskey(f, s*"1") ? s*"1" : s
        IS2Background(
            Float64.(f["r_grid"]),
            Float64.(f["t_grid"]),
            f[_k("T_spline")],
            haskey(f, "ur_spline") || haskey(f, "ur_spline1") ? f[_k("ur_spline")] : nothing,
            haskey(f, "v_spline") || haskey(f, "v_spline1") ? f[_k("v_spline")] : nothing,
            haskey(f, "α_spline") || haskey(f, "α_spline1") ? f[_k("α_spline")] : nothing,
            haskey(f, "n_spline") || haskey(f, "n_spline1") ? f[_k("n_spline")] : nothing,
            haskey(f, "nur_spline") || haskey(f, "nur_spline1") ? f[_k("nur_spline")] : nothing,
            haskey(f, "kappa_spline") || haskey(f, "kappa_spline1") ? f[_k("kappa_spline")] : nothing,
            haskey(f, "tau_diff_spline") || haskey(f, "tau_diff_spline1") ? f[_k("tau_diff_spline")] : nothing,
            haskey(f, "dtT_spline") || haskey(f, "dtT_spline1") ? f[_k("dtT_spline")] : nothing,
            haskey(f, "drT_spline") || haskey(f, "drT_spline1") ? f[_k("drT_spline")] : nothing,
            haskey(f, "dtur_spline") || haskey(f, "dtur_spline1") ? f[_k("dtur_spline")] : nothing,
            haskey(f, "drur_spline") || haskey(f, "drur_spline1") ? f[_k("drur_spline")] : nothing,
        )
    end
end

@inline function _u_from_v(v::Real)
    vC = clamp(Float64(v), -0.999999, 0.999999)
    uτ = 1.0 / sqrt(1.0 - vC^2 + 1e-12)
    ur = uτ * vC
    return uτ, ur, vC
end

@inline function _u_from_ur(ur::Real)
    urf = Float64(ur)
    uτ = sqrt(1.0 + urf^2)
    v = urf / uτ
    return uτ, urf, v
end

@inline function _clamp_eval(spl, r, τ, rg, tg)
    rr = clamp(Float64(r), Float64(first(rg)), Float64(last(rg)))
    tt = clamp(Float64(τ), Float64(first(tg)), Float64(last(tg)))
    return Float64(spl(rr, tt))
end

@inline function _fd_eval(spl, r, τ, rg, tg; wrt::Symbol)
    rr = clamp(Float64(r), Float64(first(rg)), Float64(last(rg)))
    tt = clamp(Float64(τ), Float64(first(tg)), Float64(last(tg)))
    if wrt === :r
        h = max(abs(rr - Float64(first(rg))), abs(Float64(last(rg)) - rr))
        h = min(h, (Float64(last(rg)) - Float64(first(rg))) / max(length(rg), 2)) * 0.5
        h = max(h, 1e-6)
        rp = clamp(rr + h, Float64(first(rg)), Float64(last(rg)))
        rm = clamp(rr - h, Float64(first(rg)), Float64(last(rg)))
        return (Float64(spl(rp, tt)) - Float64(spl(rm, tt))) / max(rp - rm, 1e-12)
    else
        h = max(abs(tt - Float64(first(tg))), abs(Float64(last(tg)) - tt))
        h = min(h, (Float64(last(tg)) - Float64(first(tg))) / max(length(tg), 2)) * 0.5
        h = max(h, 1e-6)
        tp = clamp(tt + h, Float64(first(tg)), Float64(last(tg)))
        tm = clamp(tt - h, Float64(first(tg)), Float64(last(tg)))
        return (Float64(spl(rr, tp)) - Float64(spl(rr, tm))) / max(tp - tm, 1e-12)
    end
end

@inline function bg_fields(bg::IS2Background, τ::Real, r::Real)
    T   = _clamp_eval(bg.T_spl,  r, τ, bg.r_grid, bg.t_grid)
    ur  = _clamp_eval(bg.ur_spl, r, τ, bg.r_grid, bg.t_grid)

    dtT  = bg.dtT_spl  !== nothing ? _clamp_eval(bg.dtT_spl,  r, τ, bg.r_grid, bg.t_grid) : _fd_eval(bg.T_spl,  r, τ, bg.r_grid, bg.t_grid; wrt=:t)
    drT  = bg.drT_spl  !== nothing ? _clamp_eval(bg.drT_spl,  r, τ, bg.r_grid, bg.t_grid) : _fd_eval(bg.T_spl,  r, τ, bg.r_grid, bg.t_grid; wrt=:r)
    dtur = bg.dtur_spl !== nothing ? _clamp_eval(bg.dtur_spl, r, τ, bg.r_grid, bg.t_grid) : _fd_eval(bg.ur_spl, r, τ, bg.r_grid, bg.t_grid; wrt=:t)
    drur = bg.drur_spl !== nothing ? _clamp_eval(bg.drur_spl, r, τ, bg.r_grid, bg.t_grid) : _fd_eval(bg.ur_spl, r, τ, bg.r_grid, bg.t_grid; wrt=:r)

    return T, ur, dtT, drT, drur, dtur
end

@inline bg_T(bg::IS2Background, τ, r)  = _clamp_eval(bg.T_spl,  r, τ, bg.r_grid, bg.t_grid)

@inline function bg_ur(bg::IS2Background, τ, r)
    if bg.ur_spl !== nothing
        return _clamp_eval(bg.ur_spl, r, τ, bg.r_grid, bg.t_grid)
    end
    bg.v_spl === nothing && error("Background file needs either ur_spline or v_spline")
    _, ur, _ = _u_from_v(_clamp_eval(bg.v_spl, r, τ, bg.r_grid, bg.t_grid))
    return ur
end

@inline function _bg_eval_optional(spl, bg::IS2Background, τ, r, default)
    spl === nothing && return default
    return _clamp_eval(spl, r, τ, bg.r_grid, bg.t_grid)
end

@inline bg_alpha(bg::IS2Background, τ, r, default=0.0) = _bg_eval_optional(bg.α_spl, bg, τ, r, default)
@inline bg_n(bg::IS2Background, τ, r, default=0.0) = _bg_eval_optional(bg.n_spl, bg, τ, r, default)
@inline bg_nur(bg::IS2Background, τ, r, default=0.0) = _bg_eval_optional(bg.nur_spl, bg, τ, r, default)
@inline bg_kappa(bg::IS2Background, τ, r, default=0.0) = _bg_eval_optional(bg.kappa_spl, bg, τ, r, default)
@inline bg_tau_diff(bg::IS2Background, τ, r, default=0.0) = _bg_eval_optional(bg.tau_diff_spl, bg, τ, r, default)

function initial_state_from_background(bg::IS2Background, τ0::Float64, r::Float64, T_floor::Float64, eos)
    T0 = max(bg_T(bg, τ0, r), T_floor)
    α0 = bg_alpha(bg, τ0, r, 0.0)
    μ0 = α0 * T0
    n0 = bg.n_spl === nothing ? eos_Pne(T0, μ0, eos)[2] : bg_n(bg, τ0, r, 0.0)
    νr0 = bg_nur(bg, τ0, r, 0.0)
    return α0, max(n0, 0.0), νr0
end

# ═══════════════════════ Slope limiter ══════════════════════════════
function build_limited_slopes!(slopes::Vector{Float64}, u::Vector{Float64}, r::Vector{Float64})
    Nr = length(u)
    fill!(slopes, 0.0)
    Nr <= 1 && return nothing
    slopes[1]  = (u[2] - u[1]) / max(r[2] - r[1], 1e-12)
    slopes[Nr] = (u[Nr] - u[Nr-1]) / max(r[Nr] - r[Nr-1], 1e-12)
    Nr <= 2 && return nothing
    @inbounds for i in 2:(Nr-1)
        ΔC = max(0.5 * (r[i+1] - r[i-1]), 1e-12)
        slopes[i] = mc_limiter(u[i] - u[i-1], u[i+1] - u[i]) / ΔC
    end
    return nothing
end

mutable struct IS2Diagnostics
    linear_failures::Int
    eigen_failures::Int
    warned_linear::Int
    warned_eigen::Int
    # cell-steps on which FIVO_NU_RAPIDITY could not chart the state because it was ALREADY outside
    # |nu^r| < n u^tau; those fall back to the plain additive update. Nonzero ⇒ the run started
    # outside the bound, which is a statement about the initial condition.
    nu_rapidity_fallbacks::Int
    steps::Int            # time steps taken by solve_IS2 (for the perf baseline / step-count diagnostics)
end

IS2Diagnostics() = IS2Diagnostics(0, 0, 0, 0, 0, 0)

function _warn_or_fail_solver!(kind::Symbol, diagnostics::IS2Diagnostics, τ::Float64, r::Float64;
    message::AbstractString,
    fail_fast::Bool,
    max_solver_warnings::Int,
    details...)
    if kind === :linear
        diagnostics.linear_failures += 1
        warned = diagnostics.warned_linear
        if warned < max_solver_warnings
            diagnostics.warned_linear += 1
            @warn message τ=τ r=r details...
        end
    else
        diagnostics.eigen_failures += 1
        warned = diagnostics.warned_eigen
        if warned < max_solver_warnings
            diagnostics.warned_eigen += 1
            @warn message τ=τ r=r details...
        end
    end

    fail_fast && error("$message at τ=$τ, r=$r")
    return nothing
end

# ═══════════════════════ Transport / Thermodynamics ═════════════════
function build_eos(kind::String)
    k = lowercase(kind)
    if k in ("lattice", "latticehrg", "lhrg")
        return LatticeHRGEOS()
    elseif k in ("conformal", "conformalhq", "chq")
        return ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    elseif k in ("running", "runningconformal", "rconformal")
        return RunningConformalHQEOS()
    else
        @warn "Unknown EOS, falling back to LatticeHRGEOS" kind=kind
        return LatticeHRGEOS()
    end
end

@inline function _bessel_kx_safe(z)
    z <= 0.0 && return (0.0, 0.0)
    K1x = SpecialFunctions.besselkx(1, z)
    K2x = SpecialFunctions.besselkx(2, z)
    return K1x, K2x
end

"""Compute (n, dn_dα, dn_dT, κ, τn, τM, ηM) at given T, α."""
function transport_all(T::Float64, α::Float64, DsT_val::Float64, eos)
    Tm = max(T, T_MIN)
    m  = hq_mass(eos)
    z  = m / Tm

    K1x, K2x = _bessel_kx_safe(z)

    # --- number density via eos_Pne (includes all factors) ---
    μ  = α * Tm
    _, n_val, _ = eos_Pne(Tm, μ, eos)
    n_val = max(n_val, 0.0)

    # Degeneracy normalization g_hq, and the canonical factor f_can baked into n by eos_Pne. To match
    # Fluidum's IS2 matrix EXACTLY (audit_coefficients.jl): it uses the CANONICAL density
    # (thermodynamic.pressure, ×f_can) for n / dn_dα / dn_dT, the GRAND-CANONICAL `normalization`
    # (no f_can) for κ, and divides τ_n / τ_M / η_M by the degeneracy. We replicate all three.
    transport_norm = hasproperty(eos, :g_hq) ? eos.g_hq : 6.0
    cf = hasproperty(eos, :canon_factor) ? eos.canon_factor : 1.0

    # --- dn/dα = n (Boltzmann fugacity); canonical (= n_val), like Fluidum thermodynamic n ---
    dn_dα = n_val

    # --- dn/dT at fixed α (canonical) ---
    dn_dT = abs(K2x) > 1e-30 ? n_val / Tm * (3.0 + z * K1x / K2x) : 0.0

    # --- κ = Ds · n_GC / fmGeV : GRAND-CANONICAL (divide out canon_factor), matching Fluidum
    #     diffusion_hadron(T,α) = DsT/T·normalization/fmGeV which carries NO canonical factor. ---
    κ = DsT_val / Tm * (HQ_KAPPA_RESONANCE_FACTOR * n_val / cf) / fmGeV

    # --- diffusion relaxation time τn = _tauN, BARE (2026-07-16 fix; was `/ transport_norm`).
    #     τ_n = D_s·I_31/(T·P_0) = D_s·z·K₃/K₂ is a RATIO of equilibrium moments ⇒ the degeneracy MUST
    #     CANCEL. The old ÷g_hq copied Fluidum's τ_diffusion_hadron, which divides by a `normalization`
    #     ∝ Degeneracy — harmless there because its pseudoscalar D mesons have Degeneracy=1, but WRONG
    #     for FiVo's single-species charm quark with g_hq=6. Same bug fixed in main2.jl:224
    #     (diff_tauN_bg) on 2026-07-16; main2IS2.jl is a separate module and was missed in that pass.
    #     (An earlier version of this comment claimed the file was untracked and so never showed in the
    #     diff — false: main2IS2.jl IS tracked in the FiVoHydro submodule.)
    #     VERIFIED: bare _tauN ≡ the fixed HC.diff_tauN_bg to ratio 1.0000 at every (T,DsT,α) — e.g.
    #     T=0.156/DsT=0.11634 → 1.808805 fm, matching LangevInMedium's τ_n = 15.55·DsT. The old value
    #     (0.301468 fm) was exactly 1/6 of it, at every T: the bug's fingerprint.
    #     τ_M/η_M below MUST stay in the SAME convention (now bare too) or the kinetic ratio breaks. ---
    τn = _tauN(Tm, z, DsT_val, K1x, K2x)

    # Optionally divide τ_n by the degeneracy g_hq so it MATCHES Fluidum's second-moment τ_diffusion_hadron
    # (which carries ÷normalization ∝ g_hq). Off by default (bare τ_n is the physically-correct ratio-of-
    # moments value ≡ diff_tauN_bg ≡ 15.55·DsT); on for the Fluidum-matched comparison convention.
    if IS2_TAUN_DEGENERACY
        τn /= transport_norm
    end

    # DIAGNOSTIC-ONLY multiplier on τ_n. DEFAULT 1.0 = production, byte-identical.
    # Added 2026-08-02 to test whether the charm first moment carries too much MEMORY. The IS2 solve
    # under-spreads the charm relative to the Langevin transport on the same background and the same
    # u^r (⟨r⟩ ratio at freeze-out 0.956 const / 0.802 linear for O+O, 0.916 for Pb+Pb linear), and at
    # late times the transport's charge-weighted diffusive drift tracks the INSTANTANEOUS ν_NS almost
    # exactly (+0.038 vs +0.035 at τ=5.5, linear) while the IS2 current sits at −0.201 — a relic, since
    # τ_n ≈ 1.9 fm exceeds the time left to freeze-out for the whole last third of the evolution.
    # Scaling τ_n down interpolates toward the Navier–Stokes limit and tests that directly.
    # ⚠ This scales τ_n ONLY. τ_M is left alone, which breaks the kinetic ratio τ_M/τ_n ≈ 0.55 — harmless
    # while c_M = 0 (production), where the second moments are passive and never feed back into ν^r,
    # but do NOT use this together with FIVO_IS2_USE_CM=1.
    τn *= IS2_TAUN_SCALE

    # Causality clamp (finding I-1): v_sig²=κ/(dn_dα·τn) must be ≤ 1. Where the grand-canonical-κ /
    # canonical-χ convention pushes it >1 (high T/α), raise τ_n to the causal floor κ/dn_dα. Leaves
    # the subluminal majority unchanged. Disable with FIVO_IS2_CAUSAL_COEFF=0 (exact Fluidum τ_n).
    if IS2_CAUSAL_COEFF && dn_dα > 0.0 && τn > 0.0
        τn = max(τn, κ / dn_dα)
    end

    # --- Ds in same convention as Fluidum ---
    Ds = DsT_val / Tm / fmGeV

    # --- second-moment relaxation τ_M, shear viscosity η_M (FP_Hydro_matching.tex eq:tauM_etaM_expanded):
    #     τ_M = (D_s/2)·[6z K1+(z²+24)K2]/[z K1+4K2] = (D_s z/2) K4/K3 ,  η_M = (D_s z/2) K3/K2 (·T convention).
    #     The boxed forms are the BARE (per-particle) coefficients. The FP derivation defines τ_n AND τ_M in
    #     the SAME bare convention (eq:tauM_Bessel), so τ_M/τ_n = ½·K4K2/K3² ≈ 0.55 (NR limit τ_M→τ_n/2).
    #     Since the 2026-07-16 g_hq fix, τ_n above is BARE, so τ_M/η_M are bare too — the SAME convention,
    #     which is what the kinetic ratio requires: τ_M/τ_n = 0.5501 (T=0.156) / 0.5891 (T=0.300), measured.
    #     WHAT MATTERS IS THAT BOTH SHARE A CONVENTION. Mixing them breaks the ratio by 6× either way
    #     (bare τ_M with ÷g_hq τ_n → 3.3, i.e. the 2nd moment fails to relax, over-produces shear and
    #     inflates c_M; the reverse → 0.09). Do NOT re-introduce a ÷transport_norm on one side alone. ---
    denom = z * K1x + 4 * K2x
    if abs(denom) > 1e-30
        τM = Ds / 2 * (6*z*K1x + (z^2 + 24)*K2x) / denom
        ηM = abs(K2x) > 1e-30 ? (Ds*Tm/2) * (4 + z*K1x/K2x) : 0.0
        if IS2_TAUM_DEGENERACY            # legacy ÷g_hq convention — only valid with a ÷g_hq τ_n (default off)
            τM /= transport_norm
            ηM /= transport_norm
        end
    else
        τM = max(τn / 2, 0.0)
        ηM = 0.0
    end

    # --- τ_Π: relaxation time of the TRACE projection Π_Q. ---------------------------------------
    #     Production (IS2_TAUPI_REL=false) sets τ_Π = τ_π = τ_M, which is what the 14-moment ansatz
    #     leaves undetermined (its only scalar dies on Landau matching ⇒ the inertia coefficient of
    #     the Π_Q equation is 0/0).  With the minimal matching-orthogonal scalar φ_Π = k² − 3I₃₁/I₁₀
    #     restored, the ratio is closed-form (main.tex Eq. B22):
    #         τ_Π/τ_π = 5/2 − (3/2)·K₃²/(K₂K₄),
    #     ≥ 1 by log-convexity of K_n in its order, → 1 in the NR limit.  K₃/K₄ come from K₁/K₂ by the
    #     recurrence K_{ν+1} = K_{ν-1} + (2ν/z)K_ν, so the exponential scaling of `besselkx` cancels
    #     in the ratio.  Verified against the direct moment integrals (1.1365 at z=9.62).
    τΠ = τM
    if IS2_TAUPI_REL && z > 0.0 && abs(K2x) > 1e-30
        K3x = K1x + (4 / z) * K2x
        K4x = K2x + (6 / z) * K3x
        d24 = K2x * K4x
        if abs(d24) > 1e-300
            τΠ = τM * max(1.0, 2.5 - 1.5 * K3x^2 / d24)
        end
    end

    return (n=n_val, dn_dα=dn_dα, dn_dT=dn_dT, κ=κ, τn=τn, τM=τM, τΠ=τΠ, ηM=ηM, Ds=Ds)
end

@inline function _state_to_q!(q::Vector{Float64}, α::Vector{Float64}, νr::Vector{Float64}, τ::Float64,
    grid::IS2Grid1D, bg::IS2Background; T_floor::Float64, eos)
    @inbounds for i in eachindex(grid.r)
        r = grid.r[i]
        T = max(bg_T(bg, τ, r), T_floor)
        μ = α[i] * T
        _, n_val, _ = eos_Pne(T, μ, eos)
        uτ, ur, _ = _u_from_ur(bg_ur(bg, τ, r))
        q[i] = r * (uτ * max(n_val, 0.0) + (ur / uτ) * νr[i])
    end
    return nothing
end

@inline function _build_q_flux!(flux_q::Vector{Float64}, q::Vector{Float64}, νr::Vector{Float64}, τ::Float64,
    grid::IS2Grid1D, bg::IS2Background)
    Nr = length(grid.r)
    flux_q[1] = 0.0

    @inbounds for i in 2:Nr
        rface = grid.rF[i]
        urface = bg_ur(bg, τ, rface)
        uτface, _, vface = _u_from_ur(urface)
        if vface >= 0.0
            Jtau_up = q[i - 1] / max(grid.r[i - 1], 1e-12)
            nu_up = νr[i - 1]
        else
            Jtau_up = q[i] / max(grid.r[i], 1e-12)
            nu_up = νr[i]
        end
        Jr_face = vface * Jtau_up + nu_up / (uτface^2)
        flux_q[i] = rface * Jr_face
    end

    rface = grid.rF[Nr + 1]
    urface = bg_ur(bg, τ, rface)
    uτface, _, vface = _u_from_ur(urface)
    Jtau_up = q[Nr] / max(grid.r[Nr], 1e-12)
    nu_up = νr[Nr]
    Jr_face = vface * Jtau_up + nu_up / (uτface^2)
    flux_q[Nr + 1] = rface * Jr_face
    return nothing
end

@inline function _transport_update_q!(q::Vector{Float64}, q_old::Vector{Float64}, flux_q::Vector{Float64},
    τ::Float64, dt::Float64, grid::IS2Grid1D)
    τ_safe = max(τ, 1e-12)
    @inbounds for i in eachindex(grid.r)
        q[i] = q_old[i] - dt * (((flux_q[i + 1] - flux_q[i]) / grid.dr[i]) + q_old[i] / τ_safe)
    end
    return nothing
end

@inline function _recover_alpha_from_q!(α::Vector{Float64}, q::Vector{Float64}, νr::Vector{Float64}, τ::Float64,
    grid::IS2Grid1D, bg::IS2Background; T_floor::Float64, eos)
    @inbounds for i in eachindex(grid.r)
        r = grid.r[i]
        T = max(bg_T(bg, τ, r), T_floor)
        uτ, ur, _ = _u_from_ur(bg_ur(bg, τ, r))
        Jtau = q[i] / max(r, 1e-12)
        # ── PHYSICAL BOUND ON THE CURRENT (FIVO_IS2_NU_BOUND, default 0 = off) ──────────────────
        # nu^r/n is a VELOCITY, so |nu^r| <= n is a kinematic bound, and the IS2 system enforces
        # none. It matters HERE and nowhere else: this line recovers n from the conserved J^tau as
        # n = (J^tau - (u^r/u^tau) nu^r)/u^tau and floors it at 1e-300, so an oversized nu^r pushes
        # the recovered density onto the floor and the "conserved" charge silently leaks.
        # 🔴 That is a REAL defect the vacuum ramp was masking, not a cosmetic one. Relaxing the ramp
        # on O+O exposes it immediately (linear law, dilute exterior): max|nu^r|/n = 2528 c with 395
        # superluminal cells and 221 cells at J^tau < 0 at N_REL_HI=0.01, 7155 c / 822 / 213 at
        # 0.001, against 0.411 c and ZERO violations at the production absolute threshold — and the
        # resulting charm drift GROWS with resolution (+3.1% at Nr=300, +3.8% at 600, +5.6% at 1200),
        # which is how you tell a genuine runaway from an under-resolution artifact.
        # ✅ SOLVED EXACTLY, not iterated. The naive clamp |nu| <= f*n caps against the n implied by
        # the PRE-clamp nu^r, so it undershoots and leaves residual violations (4 cells survived of
        # 395 in the O+O linear run). But n depends on nu^r linearly, so the constraint closes:
        #     n = (J - a nu)/u,   a = u^r/u^tau,  u = u^tau,   require |nu| <= f n
        #   nu > 0:  nu u <= f(J - a nu)  =>  nu <= +f J / (u + f a)
        #   nu < 0: -nu u <= f(J - a nu)  =>  nu >= -f J / (u - f a)
        # an ASYMMETRIC window — the flow direction makes an outward current cheaper to sustain than
        # an inward one, which the symmetric clamp got wrong. Exact in one step, no iteration.
        # J^tau <= 0 means the cell is already inconsistent: fall back to nu^r = 0 (=> n = J/u).
        # 🔴 WHICH FRAME? (FIVO_IS2_NU_BOUND_FRAME, "lab" = the original, "lrf" = corrected.)
        # The kinematic statement is that the charm drifts slower than light RELATIVE TO THE FLUID,
        # i.e. |nu*_r| < n with nu*_r the REST-FRAME current. The lab component carries one more
        # u^tau, nu^r = u^tau nu*_r, so the bound in lab variables is |nu^r| <= f n u^tau -- and the
        # form below imposed |nu^r| <= f n, which is tighter by u^tau (a factor 1.37 on Sigma_fo at
        # tau ~ 5, so f = 1 was really bounding the physical drift at 0.73 c). Solving the closed
        # constraint again with the extra u^tau, using n u^tau = Jtau - a nu:
        #   nu > 0:  nu <= f(J - a nu)      =>  nu <=  f J / (1 + f a)
        #   nu < 0: -nu <= f(J - a nu)      =>  nu >= -f J / (1 - f a)
        # i.e. exactly the old expressions with the leading u^tau removed from each denominator.
        # Default stays "lab" so no existing run changes meaning without being asked.
        if IS2_NU_BOUND > 0.0
            f = IS2_NU_BOUND
            a = ur / uτ
            if Jtau <= 0.0
                IS2_JTAU_NEG[] += 1
                # J^tau <= 0 means the transported charge in this cell is already inconsistent. The
                # historical response is to ZERO the current, which is the most violent intervention
                # in this routine -- far more so than the bound, which merely trims. With
                # FIVO_IS2_JTAU_FLOOR > 0 the charge is instead floored at that fraction of the slice
                # reference density and the current is left alone, so the cell rejoins the solution
                # smoothly instead of having its dipole deleted. Default 0 = historical.
                if IS2_JTAU_FLOOR > 0.0 && IS2_VACUUM_NREF[] > 0.0
                    Jtau = IS2_JTAU_FLOOR * IS2_VACUUM_NREF[]
                else
                    νr[i] = 0.0
                end
            end
            if Jtau > 0.0
                den = IS2_NU_BOUND_LRF ? 1.0 : uτ
                hi = f * Jtau / (den + f * a)
                lo = (den - f * a) > 0.0 ? -f * Jtau / (den - f * a) : -hi
                (νr[i] > hi || νr[i] < lo) && (IS2_NU_BOUND_HITS[] += 1)
                # SOFT-KNEE SATURATION vs HARD CLAMP (FIVO_IS2_NU_BOUND_SMOOTH).
                # `clamp` is C^0: a cell crossing the bound has its derivative replaced by zero, and
                # the boundary of the bounded REGION shows up in the solution as a kink -- the narrow
                # spikes at the entry and exit of the superluminal window.
                # 🔴 THE OBVIOUS SMOOTHING IS WRONG. `x -> b tanh(x/b)` is smooth and saturates at the
                # same b, but it is NOT the identity below the bound: it shaves x by (x/b)^2/3, and
                # this map is applied to the STATE once per step, so over 3e4 steps that compounds
                # into a global damping. Measured: it drives the linear-law current to 0.19 at tau=2
                # and monotonically down to 0.07, i.e. it destroys the very current it was meant to
                # leave alone. A saturation used as a state map must be EXACTLY the identity away
                # from the bound.
                # Hence a knee: identity for |nu| <= kappa*b, then a tanh that matches value AND
                # derivative at the knee (tanh'(0) = 1) and saturates at b. C^1 everywhere, no
                # compounding below the knee, and strictly |nu| < b above it.
                if IS2_NU_BOUND_SMOOTH
                    b = νr[i] >= 0.0 ? hi : -lo
                    if b > 0.0
                        κb = IS2_NU_BOUND_KNEE * b
                        x  = abs(νr[i])
                        x > κb && (νr[i] = sign(νr[i]) *
                                   (κb + (b - κb) * tanh((x - κb) / max(b - κb, 1e-300))))
                    end
                else
                    νr[i] = clamp(νr[i], lo, hi)
                end
            end
        end
        nu_tau = (ur / uτ) * νr[i]
        n_val = max((Jtau - nu_tau) / uτ, 1e-300)

        # ── CONE PROJECTION (FIVO_IS2_CONE_PROJECT) ─────────────────────────────────────────────
        # Everything above repairs ONE COMPONENT and then recovers the other, which is why the
        # regularized solution shows kinks where the repair switches on: nu^r is clamped at fixed
        # J^tau, n is whatever falls out, and the pair jumps.
        # The admissibility condition is not a statement about a component. The charge four-current
        # N^mu = n u^mu + nu*_r ubar^mu must be TIMELIKE for a rest frame to exist at all, i.e.
        #     N.N = n^2 - nu*^2 > 0   <=>   |nu*_r| < n,
        # which is the same |nu| < 1 written as a property of the VECTOR. So repair the vector:
        # if the pair (n, nu*) has left the cone, project it onto the ray |nu*| = f n, which changes
        # BOTH entries, is the closest admissible state in the (n, nu*) plane, and is the IDENTITY on
        # the boundary -- so the scheme switches on continuously instead of jumping.
        #     d = (1, f)/sqrt(1+f^2),  (n', nu*') = ((n + f|nu*|)/(1+f^2)) (1, f sign nu*)
        # At |nu*| = f n this returns (n, nu*) exactly, so there is no kink where it engages.
        # It also subsumes the J^tau <= 0 branch: a non-positive charge with a finite current is just
        # a point far outside the cone, and the projection returns a positive n rather than needing a
        # special case that deletes the dipole.
        if IS2_CONE_PROJECT > 0.0
            f = IS2_CONE_PROJECT
            νstar = νr[i] / uτ
            if abs(νstar) > f * n_val
                IS2_CONE_HITS[] += 1
                np = (n_val + f * abs(νstar)) / (1.0 + f * f)
                if np > 0.0
                    n_val  = np
                    νr[i]  = sign(νstar) * f * np * uτ
                end
            end
        end
        n_eq = max(eos_Pne(T, 0.0, eos)[2], 1e-300)
        # FLOOR AT THE SOURCE. alpha = log(n/n_th) diverges as n -> 0, so every site that CREATES
        # or ADVANCES alpha must floor it -- flooring only the local `alpha_safe` copy used for
        # transport (as of the first attempt at this fix) leaves the EVOLVED and EXPORTED field
        # unbounded, which is what every downstream consumer actually reads.
        α[i] = _alpha_floor(log(n_val / n_eq))
    end
    return nothing
end

@inline function _tauN(Tm, z, DsT_val, K1x, K2x)
    z <= 0.0 && return 0.0
    if z > 50.0
        τ_GeVinv = (DsT_val / 48.0) * (z^2 * Tm)
        return min(τ_GeVinv / fmGeV, 1e20)
    end
    K3x = K1x + 4.0/z * K2x
    K4x = K2x + 6.0/z * K3x
    K5x = K3x + 8.0/z * K4x
    num = 2*K1x - 3*K3x + K5x
    den = max(abs(K2x), TINY)
    ratio = num / den * sign(K2x == 0 ? 1.0 : K2x)
    z3_over_Tm = z^3 / Tm
    if !isfinite(z3_over_Tm) || z3_over_Tm > 1e50
        z3_over_Tm = 1e50
    end
    τ_GeVinv = (DsT_val / 48.0) * z3_over_Tm * ratio
    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)
    end
    return τ_GeVinv / fmGeV
end

# ═══════════════════════ 5×5 Matrix assembly ═══════════════════════
"""
Build At (5×5), Ax (5×5), source (5)  — exact Fluidum formulation.

State vector U = [α, νr, πQr, πQperp, PiQ].
System: At * ∂U/∂τ + Ax * ∂U/∂r = source.
"""
# ═══════════════════════ Characteristic speeds ═════════════════════
"""
Compute the maximum characteristic speed |λ| from the 5×5 system
At·∂U/∂τ + Ax·∂U/∂r = S via eigenvalues of At⁻¹·Ax.
Includes imaginary parts since RK2 has finite stability on the imaginary axis.
"""
@inline function _max_signal_speed(B::Matrix{Float64})
    evs = eigvals(B)
    cmax = 0.0
    @inbounds for ev in evs
        rv = abs(ev)
        if isfinite(rv)
            cmax = max(cmax, rv)
        end
    end
    return cmax
end

function _build_reduced_system!(
    B::Matrix{Float64},
    src_term::Vector{Float64},
    At::Matrix{Float64},
    Ax::Matrix{Float64},
    src::Vector{Float64},
    U::Vector{Float64},
    τ::Float64,
    r::Float64,
    bg::IS2Background;
    DsT,
    T_floor::Float64,
    T_cM_min::Float64,
    eos,
    use_cM::Bool,
    diagnostics::IS2Diagnostics,
    fail_on_linear_failure::Bool,
    fail_on_eigen_failure::Bool,
    max_solver_warnings::Int,
    cm_dtz::Float64 = 0.0,           # ∂τ(πQr+ΠQ), stage-lagged (consistent 5-field only)
    cm_drz::Float64 = 0.0,           # ∂r(πQr+ΠQ) from the MUSCL slopes (center calls only)
)
    T, ur_val, dtT, drT, drur, dtur = bg_fields(bg, τ, r)
    T = max(T, T_floor)

    α_safe = _alpha_floor(U[1])   # smooth physical floor, not an overflow guard
    # DsT may be a scalar or a T-dependent law (e.g. linear D_sT(T)); evaluate locally.
    DsT_val = DsT isa Function ? Float64(DsT(T)) : DsT
    tp = transport_all(T, α_safe, DsT_val, eos)
    κ_eff = bg.kappa_spl === nothing ? tp.κ : bg_kappa(bg, τ, r, tp.κ)
    τn_eff = bg.tau_diff_spl === nothing ? tp.τn : bg_tau_diff(bg, τ, r, tp.τn)
    # c_M = D_s/T is the O(Kn²) back-coupling of the 2nd moment 𝓜 onto ν^r (FP_Hydro_matching.tex
    # eq:transport).  The derivation puts it as −c_M ∂π on the RHS (eq:diffusion_split); build_IS2_system!
    # assembles in LHS form (Aₜ∂_τU+Aₓ∂_rU=src), which flips the sign ⇒ the LHS coefficient is +c_M=+D_s/T,
    # i.e. IS2_CM_SIGN=+1 (the default, and the well-posed/hyperbolic branch — see CMExperiment audit).
    cM = (use_cM && T > T_cM_min) ? (IS2_CM_SIGN * tp.Ds / T) : 0.0
    # Consistent 5-field: the cM force goes in as an explicit source (below) built from the
    # COMPLETE abstract-route row; the legacy matrix cM entries (basis-route, missing the
    # geometric hoop/redshift pieces) are zeroed. The CFL vM fold further down stays keyed
    # on the physical cM either way.
    cM_matrix = (IS2_CONSISTENT_FM[] && use_cM) ? 0.0 : cM

    r_safe = max(abs(r), 1e-6)
    r3 = r_safe^3
    r4 = r_safe^4

    build_IS2_system!(At, Ax, src,
        (α_safe, U[2], U[3], _piQperp_physical(U[4], r_safe), U[5]),
        τ, r, ur_val, T, dtT, drT, drur, dtur,
        tp.n, tp.dn_dα, tp.dn_dT, κ_eff, τn_eff, tp.τM, tp.ηM, cM_matrix)

    # Consistent first moment: the five derived source terms (flag, default off — see the
    # IS2_CONSISTENT_FM const and src/hq_consistent_firstmoment.jl). Sources only: At/Ax and
    # the CFL machinery are untouched, so flag-off is byte-identical production.
    if IS2_CONSISTENT_FM[]
        h_hqc, hp_hqc = hq_consistent_h_hp(T, DsT_val, τn_eff, eos)
        src[2] += hq_consistent_extras(τ, r_safe, ur_val, T, dtT, drT, drur, dtur,
                                       tp.n, tp.dn_dT, U[2], τn_eff, tp.Ds, h_hqc, hp_hqc)
        # The consistent c_M back-coupling as an explicit source (complete row, geometric
        # pieces included), Fluidum-parity vacuum damping n/(n+nfloor). Face calls carry
        # cm_dtz = cm_drz = 0 and their src_face_term is never consumed.
        if cM != 0.0
            src[2] += cM * (tp.n / (tp.n + IS2_CM_NFLOOR)) *
                      hq_cm_force(τ, r_safe, ur_val, dtur, drur,
                                  U[3], _piQperp_physical(U[4], r_safe), U[5], cm_dtz, cm_drz)
        end
    end

    ax14 = Ax[1,4]; ax24 = Ax[2,4]; ax34 = Ax[3,4]; ax44 = Ax[4,4]; ax54 = Ax[5,4]
    @inbounds for row in 1:5
        At[row,4] *= r4
        Ax[row,4] *= r4
    end
    src[1] -= ax14 * (4.0 * r3 * U[4])
    src[2] -= ax24 * (4.0 * r3 * U[4])
    src[3] -= ax34 * (4.0 * r3 * U[4])
    src[4] -= ax44 * (4.0 * r3 * U[4])
    src[5] -= ax54 * (4.0 * r3 * U[4])

    # The second moments (rows 3-5) are evolved in the bounded physical-eigenvalue basis by an
    # explicit override (_second_moment_eigen_rhs!), NOT by this matrix.  Trivialise these rows
    # so the (α, ν_r) block solve stays well-conditioned and the matrix contributes nothing to
    # dU[3:5].  With c_M=0 the second moments are passive (At/Ax[1:2,3:5]=0), so rows 1-2 are
    # unaffected.
    @inbounds for row in 3:5
        for c in 1:5
            At[row,c] = 0.0
            Ax[row,c] = 0.0
        end
        At[row,row] = 1.0
        src[row] = 0.0
    end

    F = lu(At; check=false)
    if !issuccess(F)
        _warn_or_fail_solver!(:linear, diagnostics, τ, r;
            message="IS2 linear solve failed",
            fail_fast=fail_on_linear_failure,
            max_solver_warnings=max_solver_warnings,
            pivmin=minimum(abs.(diag(F.U))), α=α_safe, T=T, ur=ur_val)
        return false, 0.0, tp.n
    end

    copyto!(B, F \ Ax)
    copyto!(src_term, F \ src)

    try
        cmax = _max_signal_speed(B)
        # The reduced matrix trivialises the 𝓜 rows (3-5) — the return coupling ν→𝓜 is done by
        # _second_moment_eigen_rhs!, NOT the matrix — so eigvals(B) MISS the ν↔𝓜 telegraph wave that
        # c_M≠0 introduces.  Fold in its analytic high-k characteristic speed v_M² = (channel)·c_M^phys·
        # η_M/(τ_n τ_M) (rest-frame; c_M^phys=D_s/T magnitude), using the WORST-CASE combined ν↔(π_r+Π_Q)
        # channel factor 4/3+5/3 (cf. CMExperiment/dispersion_analysis.jl), relativistically boosted to the
        # lab frame, so the CFL Δτ is bounded by it.  NB v_M can exceed c in the hot core (T≳0.42 GeV) —
        # a known acausality of this O(Kn²) closure; the cap below keeps the timestep finite there.
        if cM != 0.0
            vM_lrf = sqrt(max(0.0, (4.0/3.0 + 5.0/3.0) * (tp.Ds / T) * tp.ηM / (τn_eff * tp.τM)))
            vflow  = abs(ur_val) / sqrt(1.0 + ur_val^2)
            vM_lab = (vM_lrf + vflow) / (1.0 + vM_lrf * vflow)
            cmax = max(cmax, vM_lab)
        end
        return true, cmax, tp.n
    catch
        _warn_or_fail_solver!(:eigen, diagnostics, τ, r;
            message="IS2 eigensystem failed",
            fail_fast=fail_on_eigen_failure,
            max_solver_warnings=max_solver_warnings,
            α=α_safe, T=T, ur=ur_val)
        return false, 0.0, tp.n
    end
end

@inline function _set_state!(U::Vector{Float64}, αv::Real, νrv::Real, piQrv::Real, piQperpv::Real, PiQv::Real)
    U[1] = αv
    U[2] = νrv
    U[3] = piQrv
    U[4] = piQperpv
    U[5] = PiQv
    return nothing
end

@inline _piQperp_regularized(qperp::Real, r::Real) = Float64(qperp)
# FiVo evolves π_perp^Q as the bounded physical eigenvalue (paper Eq. app-piQ-components),
# NOT the /r⁴-scaled variable: storage and physical value coincide (identity).  The /r⁴ basis
# was numerically unstable in the explicit FV scheme at large r (π_perp = qperp·r⁴ amplified
# noise ~r⁴, blowing up at the larger Pb+Pb radii).  See _second_moment_eigen_rhs!.
@inline _piQperp_physical(qperp::Real, r::Real) = Float64(qperp)
@inline _odd_regularized(u::Real, r::Real) = Float64(u) / max(Float64(r), 1e-12)
@inline _odd_physical(ureg::Real, r::Real) = Float64(ureg) * Float64(r)

@inline function _reconstruct_face_left!(U::Vector{Float64}, i::Int, grid::IS2Grid1D,
    α::Vector{Float64}, νr::Vector{Float64}, piQr::Vector{Float64}, piQperp::Vector{Float64}, PiQ_field::Vector{Float64},
    slopes::NTuple{5, Vector{Float64}})
    Δ = grid.r[i] - grid.rF[i]
    _set_state!(U,
        α[i] - Δ * slopes[1][i],
        _odd_physical(_odd_regularized(νr[i], grid.r[i]) - Δ * slopes[2][i], grid.rF[i]),
        _odd_physical(_odd_regularized(piQr[i], grid.r[i]) - Δ * slopes[3][i], grid.rF[i]),
        _piQperp_regularized(piQperp[i], grid.r[i]) - Δ * slopes[4][i],
        PiQ_field[i] - Δ * slopes[5][i],
    )
    return nothing
end

@inline function _reconstruct_face_right!(U::Vector{Float64}, i::Int, grid::IS2Grid1D,
    α::Vector{Float64}, νr::Vector{Float64}, piQr::Vector{Float64}, piQperp::Vector{Float64}, PiQ_field::Vector{Float64},
    slopes::NTuple{5, Vector{Float64}})
    Δ = grid.rF[i + 1] - grid.r[i]
    _set_state!(U,
        α[i] + Δ * slopes[1][i],
        _odd_physical(_odd_regularized(νr[i], grid.r[i]) + Δ * slopes[2][i], grid.rF[i + 1]),
        _odd_physical(_odd_regularized(piQr[i], grid.r[i]) + Δ * slopes[3][i], grid.rF[i + 1]),
        _piQperp_regularized(piQperp[i], grid.r[i]) + Δ * slopes[4][i],
        PiQ_field[i] + Δ * slopes[5][i],
    )
    return nothing
end

@inline function _set_origin_boundary_state!(U::Vector{Float64}, α::Vector{Float64}, piQperp::Vector{Float64}, PiQ_field::Vector{Float64})
    U[1] = α[1]
    U[2] = 0.0
    U[3] = 0.0
    U[4] = piQperp[1]
    U[5] = PiQ_field[1]
    return nothing
end

@inline function _set_outer_boundary_state!(U::Vector{Float64}, α::Vector{Float64}, νr::Vector{Float64}, piQr::Vector{Float64}, piQperp::Vector{Float64}, PiQ_field::Vector{Float64}, i::Int)
    _set_state!(U, α[i], νr[i], piQr[i], piQperp[i], PiQ_field[i])
    return nothing
end

@inline function _midpoint_state!(U_mid::Vector{Float64}, U_left::Vector{Float64}, U_right::Vector{Float64})
    @inbounds for k in 1:5
        U_mid[k] = 0.5 * (U_left[k] + U_right[k])
    end
    return nothing
end

@inline function _jump_state!(jump::Vector{Float64}, U_left::Vector{Float64}, U_right::Vector{Float64})
    @inbounds for k in 1:5
        jump[k] = U_right[k] - U_left[k]
    end
    return nothing
end

# ═══════════════ second-moment evolution (bounded eigenvalue basis) ═══════════════
# Evolve the physical eigenvalues π_r^Q, π_perp^Q, Π_Q (paper Eq. app-piQ-components) via the
# rederived covariant MIS equations (Julia/tools/derive_2nd_moment_eigenbasis.wl).  Each field
# pure-advects at v = u^r/u^τ and relaxes toward its Navier–Stokes target built from gradients of
# the diffusion current ν^r — clean (only 1/r, 1/τ geometric factors), with NO /r⁴ amplification.
# The fields are passive (c_M=0) so dν_r/dτ (= dU[2]) is already known when this is called.
# Overrides dU[3:5].
function _second_moment_eigen_rhs!(
    dU::NTuple{5, Vector{Float64}},
    α::Vector{Float64}, νr::Vector{Float64},
    piQr::Vector{Float64}, piQperp::Vector{Float64}, PiQ_field::Vector{Float64},
    τ::Float64, grid::IS2Grid1D, bg::IS2Background; DsT, T_floor::Float64, eos)
    Nr = length(grid.r)
    @inbounds for i in 2:(Nr-1)
        r = grid.r[i]
        T, ur, dtT, drT, drur, dtur = bg_fields(bg, τ, r)
        T = max(T, T_floor)
        DsT_val = DsT isa Function ? Float64(DsT(T)) : DsT
        tp = transport_all(T, α[i], DsT_val, eos)
        tauM = tp.τM; tauPi = tp.τΠ; etaM = tp.ηM; n_local = tp.n
        if !(tauM > 0.0) || !(r > 0.0)
            dU[3][i] = 0.0; dU[4][i] = 0.0; dU[5][i] = 0.0; continue
        end
        ut = sqrt(1.0 + ur^2); v = ur / ut
        nur = νr[i]; piR = piQr[i]; piP = piQperp[i]; bPi = PiQ_field[i]
        dtnur = dU[2][i]
        drnur = (νr[i+1] - νr[i-1]) / (grid.r[i+1] - grid.r[i-1])     # central (ν_r odd ⇒ →0 at axis)
        # upwind advection gradients (v ≥ 0 outward ⇒ backward difference)
        drR = v >= 0 ? (piQr[i]-piQr[i-1])/(grid.r[i]-grid.r[i-1]) : (piQr[i+1]-piQr[i])/(grid.r[i+1]-grid.r[i])
        drP = v >= 0 ? (piQperp[i]-piQperp[i-1])/(grid.r[i]-grid.r[i-1]) : (piQperp[i+1]-piQperp[i])/(grid.r[i+1]-grid.r[i])
        drB = v >= 0 ? (PiQ_field[i]-PiQ_field[i-1])/(grid.r[i]-grid.r[i-1]) : (PiQ_field[i+1]-PiQ_field[i])/(grid.r[i+1]-grid.r[i])
        ut2 = ut*ut; ut4 = ut2*ut2; ut5 = ut4*ut
        # Second-moment evolution = the Fluidum HQ_const_BG_2nd_moment Israel–Stewart system, in the
        # bounded orthonormal eigenbasis (π_r,π_perp,Π_Q).  Derived by inverting Fluidum's 5×5 matrix
        # (At⁻¹(source − At[:,2]∂_τν − Ax∂_rX)) and splitting off the diagonal advection −v∂_rX (which
        # is exactly −uʳ/uᵗ for all three fields).  Verified to machine precision against the Fluidum
        # matrix function in Julia/tools/verify_fivo_vs_fluidum_2nd_moment.wl (+ Julia/tools/rhs_compare.jl).
        # The Src terms carry the ν-gradient driving (2η_Q σ_(ν)) AND the π_perp/Π_Q cross-coupling in
        # the relaxation that the previous eigenbasis rederivation was missing.  c_M = 0.
        SrcR = (4*etaM*r*τ*(-(dtur*nur*ur^2) + ut*(drnur - drur*nur*ur + drnur*ur^2) + dtnur*(ur + ur^3))
                + 2*piP*tauM*(r + r*ur^2 - τ*ur*ut)*ut2 - 2*bPi*tauM*(r + r*ur^2 + τ*ur*ut)*ut2
                + piR*r*ut2*(-3*τ*ut + 2*tauM*ut2)) / (3*r*τ*tauM*ut4)
        SrcP = (-2*drnur*etaM*r*τ*ut4 + piP*(-3*r*τ + 4*τ*tauM*ur + 2*r*tauM*ut)*ut4
                + 2*(drur*etaM*nur*r*τ*ur*ut2 + bPi*tauM*(2*τ*ur - r*ut)*ut4
                     + r*ut*(-(etaM*τ*ur*(dtnur - dtur*nur*ur + dtnur*ur^2)) + piR*tauM*ut4))) / (3*r*τ*tauM*ut5)
        # Trace (Π_Q) row.  Every τ_M in the NUMERATOR cancels the 1/τ_M of the prefactor: those are the
        # geometric (1/r, 1/τ) Christoffel couplings, which carry no relaxation time.  What is genuinely
        # divided by τ_M is only the driving ζ_Q θ_(ν) (the 5·η_M terms, ζ_Q = 5/3 η_M) and the diagonal
        # relaxation −Π_Q/(τ_M u^τ).  Splitting SrcB = A_B/τ + G_B therefore isolates exactly the rate,
        # and IS2_TAUPI_REL swaps τ_M → τ_Π there and NOWHERE else.  The two branches below are
        # algebraically identical at τ_Π = τ_M; production keeps the original single expression so it
        # stays bit-for-bit unchanged.
        SrcB = if IS2_TAUPI_REL
            A_B = 5*dtur*etaM*nur*r*τ + ut2*(5*etaM*r*τ*(dtnur*ur + drnur*ut) - 3*bPi*r*τ*ut)
            G_B = ut2*(2*bPi*τ*ur*ut - 2*piP*(r + r*ur^2 - τ*ur*ut) - 2*piR*r*ut2 + 2*bPi*r*ut2)
            (A_B / tauPi + G_B) / (3*r*τ*ut4)
        else
            (5*dtur*etaM*nur*r*τ + ut2*(2*bPi*τ*tauM*ur*ut
                + 5*etaM*r*τ*(dtnur*ur + drnur*ut) - 2*piP*tauM*(r + r*ur^2 - τ*ur*ut)
                - 2*piR*r*tauM*ut2 + bPi*r*(-3*τ*ut + 2*tauM*ut2))) / (3*r*τ*tauM*ut4)
        end
        d3 = -v*drR + SrcR
        d4 = -v*drP + SrcP
        d5 = -v*drB + SrcB

        # ── THE CONSISTENT SECOND MOMENT (src/hq_consistent_m2.jl) ────────────────────────────────
        # A REPLACEMENT of the three rows above, not an addition: the shipped system carries the
        # σ_(ν) drive with the opposite sign and coordinate-transport couplings the covariant
        # reduction does not have (gate M3), so the two cannot be superposed.
        # ∂_rα is central here for the same reason drnur is: it feeds a source, not a flux, and the
        # upwind bias that the advection terms need would put a one-sided error into a symmetric term.
        if IS2_CONSISTENT_M2[]
            dralpha = (α[i+1] - α[i-1]) / (grid.r[i+1] - grid.r[i-1])
            h_m2, hp_m2 = hq_consistent_h_hp(T, DsT_val, tp.τn, eos)
            d3, d4, d5 = hq_consistent_m2_rhs(
                τ, r, ur, T, dtT, drT, drur, dtur,
                piR, piP, bPi, nur,
                dU[1][i], dralpha, dtnur, drnur, (drR, drP, drB),
                n_local, tp.τn, tp.Ds, h_m2, hp_m2, tauM, etaM, hq_mass(eos))
        end
        # vacuum ramp (same as the first-moment fields)
        w = _vacuum_weight(n_local, T)
        dU[3][i] = d3*w; dU[4][i] = d4*w; dU[5][i] = d5*w
    end
    dU[3][1] = 0.0; dU[4][1] = 0.0; dU[5][1] = 0.0
    dU[3][Nr] = 0.0; dU[4][Nr] = 0.0; dU[5][Nr] = 0.0
    return nothing
end

# ═══════════════════════ RHS computation ═══════════════════════════
"""
Compute dU/dτ for the 5-component IS2 system at the given state.
Fills `dU[k][i]` for each field k and grid point i.
Returns the maximum characteristic speed across all grid points.
"""
function _compute_dUdt!(
    dU::NTuple{5, Vector{Float64}},
    α::Vector{Float64},
    νr::Vector{Float64},
    piQr::Vector{Float64},
    piQperp::Vector{Float64},
    PiQ_field::Vector{Float64},
    odd_proxy_νr::Vector{Float64},
    odd_proxy_piQr::Vector{Float64},
    qperp_proxy::Vector{Float64},
    τ::Float64,
    grid::IS2Grid1D,
    bg::IS2Background;
    DsT,
    T_floor::Float64,
    T_cM_min::Float64,
    eos,
    use_cM::Bool,
    σ_KO::Float64,
    slopes::NTuple{5, Vector{Float64}},
    At::Matrix{Float64},
    Ax::Matrix{Float64},
    B::Matrix{Float64},
    src::Vector{Float64},
    src_term::Vector{Float64},
    src_face_term::Vector{Float64},
    rhs::Vector{Float64},
    U_left::Vector{Float64},
    U_right::Vector{Float64},
    U_left_in::Vector{Float64},
    U_right_in::Vector{Float64},
    U_mid::Vector{Float64},
    jump_state::Vector{Float64},
    D_left::Vector{Float64},
    D_right::Vector{Float64},
    diagnostics::IS2Diagnostics,
    fail_on_linear_failure::Bool,
    fail_on_eigen_failure::Bool,
    max_solver_warnings::Int,
)
    Nr = length(grid.r)
    max_charspeed = 0.0

    # Slice reference density for the RELATIVE vacuum ramp. Computed once per RHS evaluation and
    # stashed where _vacuum_weight can see it (it only receives (n,T)). Skipped entirely when the
    # relative mode is off, so the absolute path costs nothing.
    if IS2_VACUUM_N_REL_HI > 0.0
        nref = 0.0
        @inbounds for i in 1:Nr
            T = max(bg_T(bg, τ, grid.r[i]), T_floor)
            _, n_i, _ = eos_Pne(T, α[i] * T, eos)
            isfinite(n_i) && n_i > nref && (nref = n_i)
        end
        IS2_VACUUM_NREF[] = nref
    end

    build_limited_slopes!(slopes[1], α, grid.r)
    @inbounds for i in eachindex(grid.r)
        odd_proxy_νr[i] = _odd_regularized(νr[i], grid.r[i])
        odd_proxy_piQr[i] = _odd_regularized(piQr[i], grid.r[i])
        qperp_proxy[i] = _piQperp_regularized(piQperp[i], grid.r[i])
    end
    build_limited_slopes!(slopes[2], odd_proxy_νr, grid.r)
    build_limited_slopes!(slopes[3], odd_proxy_piQr, grid.r)
    build_limited_slopes!(slopes[4], qperp_proxy, grid.r)
    build_limited_slopes!(slopes[5], PiQ_field, grid.r)
    if IS2_ORIGIN_ODD_FIRST_ORDER_CELLS > 0
        nfix = min(IS2_ORIGIN_ODD_FIRST_ORDER_CELLS, Nr)
        @inbounds for i in 1:nfix
            slopes[2][i] = 0.0
            slopes[3][i] = 0.0
        end
    end

    @inbounds for i in 1:Nr
        r = grid.r[i]
        _set_state!(U_mid, α[i], νr[i], piQr[i], piQperp[i], PiQ_field[i])
        # consistent 5-field: ζ-gradient from the MUSCL slopes (πQr in its odd basis, like ν),
        # ∂τζ from the previous stage's eigen-RHS (the lag documented at IS2_CM_DT3)
        cm_dtz = 0.0; cm_drz = 0.0
        if IS2_CONSISTENT_FM[] && use_cM
            if length(IS2_CM_DT3[]) == Nr
                cm_dtz = IS2_CM_DT3[][i] + IS2_CM_DT5[][i]
            end
            cm_drz = piQr[i] / max(r, 1e-12) + r * slopes[3][i] + slopes[5][i]
        end
        ok_center, c_center, n_local = _build_reduced_system!(
            B, src_term, At, Ax, src, U_mid, τ, r, bg;
            DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM,
            diagnostics=diagnostics,
            fail_on_linear_failure=fail_on_linear_failure,
            fail_on_eigen_failure=fail_on_eigen_failure,
            max_solver_warnings=max_solver_warnings,
            cm_dtz=cm_dtz, cm_drz=cm_drz,
        )

        if ok_center
            max_charspeed = max(max_charspeed, _cfl_speed(c_center, n_local))
            fill!(rhs, 0.0)

            # ── Path-conservative intra-cell smooth-gradient drive (rows 1-2) ────────────
            # The HLL face fluctuations below use B·(reconstructed face jump), which VANISHES
            # for smooth fields (MUSCL reconstruction makes the faces continuous → jump→0).
            # For a quasi-linear non-conservative system ∂_τU = -(At⁻¹src + At⁻¹Ax·∂_rU), the
            # smooth advective/diffusive drive B·∂_rU (B = At⁻¹Ax at the CELL CENTRE, still in
            # `B` here before the face rebuilds) must be added explicitly — otherwise the
            # diffusion drive κ·∂_rα on the ν^r row is lost and ν^r is grossly under-driven.
            # (rows 3-5 are overwritten by _second_moment_eigen_rhs!, so only 1-2 are needed.)
            if IS2_DRIVE_ENABLE; let invr = 1.0 / max(r, 1e-12)
                gα = slopes[1][i]                       # physical ∂_rα (MC-limited slope)
                gν = νr[i] * invr + r * slopes[2][i]    # physical ∂_rν^r in the odd (ν/r) basis
                rhs[1] += B[1,1]*gα + B[1,2]*gν
                rhs[2] += B[2,1]*gα + B[2,2]*gν
                # Rusanov (LLF) dissipation: the centred drive above carries no upwind bias, so at
                # fine grids it can seed grid-scale oscillations. Add 0.5·c·∂²_rU (2nd-order, ∝ dr →
                # vanishes in the continuum, does not bias the converged solution; strongly damps the
                # 2-cell mode). ν^r uses its odd proxy ν/r so the origin parity is respected.
                if 1 < i < Nr && IS2_DRIVE_DISSIP > 0
                    cd = 0.5 * IS2_DRIVE_DISSIP * min(c_center, IS2_MAX_SIGNAL_SPEED) / max(grid.dr[i], 1e-12)
                    rhs[1] -= cd * (α[i+1] - 2*α[i] + α[i-1])
                    qm = νr[i-1] / max(grid.r[i-1], 1e-12); qp = νr[i+1] / max(grid.r[i+1], 1e-12)
                    rhs[2] -= cd * r * (qp - 2*(νr[i]*invr) + qm)
                end
            end; end

            if i > 1
                _reconstruct_face_right!(U_left_in, i - 1, grid, α, νr, piQr, piQperp, PiQ_field, slopes)
                _reconstruct_face_left!(U_left, i, grid, α, νr, piQr, piQperp, PiQ_field, slopes)
                _midpoint_state!(U_mid, U_left_in, U_left)
                _jump_state!(jump_state, U_left_in, U_left)
                ok_face, c_face, n_face = _build_reduced_system!(
                    B, src_face_term, At, Ax, src, U_mid, τ, grid.rF[i], bg;
                    DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM,
                    diagnostics=diagnostics,
                    fail_on_linear_failure=fail_on_linear_failure,
                    fail_on_eigen_failure=fail_on_eigen_failure,
                    max_solver_warnings=max_solver_warnings,
                )
                if ok_face
                    max_charspeed = max(max_charspeed, _cfl_speed(c_face, n_face))
                    mul!(D_left, B, jump_state)
                    @inbounds for k in 1:5
                        rhs[k] += 0.5 * (D_left[k] + c_face * jump_state[k]) / max(grid.dr[i], 1e-12)
                    end
                end
            end

            if i < Nr
                _reconstruct_face_right!(U_right, i, grid, α, νr, piQr, piQperp, PiQ_field, slopes)
                _reconstruct_face_left!(U_right_in, i + 1, grid, α, νr, piQr, piQperp, PiQ_field, slopes)
                _midpoint_state!(U_mid, U_right, U_right_in)
                _jump_state!(jump_state, U_right, U_right_in)
                ok_face, c_face, n_face = _build_reduced_system!(
                    B, src_face_term, At, Ax, src, U_mid, τ, grid.rF[i + 1], bg;
                    DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM,
                    diagnostics=diagnostics,
                    fail_on_linear_failure=fail_on_linear_failure,
                    fail_on_eigen_failure=fail_on_eigen_failure,
                    max_solver_warnings=max_solver_warnings,
                )
                if ok_face
                    max_charspeed = max(max_charspeed, _cfl_speed(c_face, n_face))
                    mul!(D_right, B, jump_state)
                    @inbounds for k in 1:5
                        rhs[k] += 0.5 * (D_right[k] - c_face * jump_state[k]) / max(grid.dr[i], 1e-12)
                    end
                end
            end

            @inbounds for k in 1:5
                dU[k][i] = -(src_term[k] + rhs[k])
            end
        else
            dU[1][i] = 0.0; dU[2][i] = 0.0; dU[3][i] = 0.0
            dU[4][i] = 0.0; dU[5][i] = 0.0
        end

        # Smooth damping in the dilute/vacuum exterior.
        # At[1,1] = dn_dα ≈ n → 0 makes α ill-conditioned; cM-driven νr perturbations get amplified
        # by 1/n, causing α blow-up. Ramp the evolution off there — gated on T when
        # FIVO_VACUUM_T_HI > 0, else on n (see _vacuum_weight for why T is the better discriminator).
        # T is recomputed here exactly as _build_reduced_system! does; it returns n_local but not T.
        w_vac = _vacuum_weight(n_local, max(bg_T(bg, τ, grid.r[i]), T_floor))
        if w_vac < 1.0
            dU[1][i] *= w_vac
            dU[2][i] *= w_vac
            dU[3][i] *= w_vac
            dU[4][i] *= w_vac
            dU[5][i] *= w_vac
        end

        if IS2_FREEZE_PDE_ALPHA
            # alpha is corrected from the conservative q-transport after each full
            # step; freezing the PDE alpha increment avoids mixing two inconsistent
            # update paths inside the RK stages.
            dU[1][i] = 0.0
        end
    end

    # Override dU[3:5] (second moments) with the bounded-eigenvalue scheme; dU[2] (=dν_r/dτ)
    # is now complete and feeds the Navier–Stokes source.  Replaces the unstable /r⁴ matrix rows.
    _second_moment_eigen_rhs!(dU, α, νr, piQr, piQperp, PiQ_field, τ, grid, bg;
                              DsT=DsT, T_floor=T_floor, eos=eos)

    # stash dπQr/dτ and dΠQ/dτ for the NEXT stage's consistent cM source (see IS2_CM_DT3).
    # Deliberately taken BEFORE the Kreiss-Oliger block below: the ∂τζ of the abstract-route
    # derivation is the PHYSICAL rate, and KO is a numerical regulator — the stored lag is the
    # pre-dissipation value (audit note 2026-08-25).
    if IS2_CONSISTENT_FM[] && use_cM
        if length(IS2_CM_DT3[]) != Nr
            IS2_CM_DT3[] = zeros(Nr); IS2_CM_DT5[] = zeros(Nr)
        end
        copyto!(IS2_CM_DT3[], dU[3]); copyto!(IS2_CM_DT5[], dU[5])
    end

    # ── Kreiss-Oliger dissipation: damp grid-scale oscillations ──────
    # Adds -σ_KO/(16·dx) * (U[i-2] - 4U[i-1] + 6U[i] - 4U[i+1] + U[i+2])
    # to dU/dτ for each field. Standard 4th-order KO for 2nd-order schemes.
    if σ_KO > 0 && Nr > 4
        fields = (α, νr, piQr, piQperp, PiQ_field)
        dx = grid.dr[1]
        ko_fac = σ_KO / (16.0 * dx)
        @inbounds for k in 1:5
            if k == 2
                U = odd_proxy_νr
                for i in 3:(Nr-2)
                    stencil = U[i-2] - 4*U[i-1] + 6*U[i] - 4*U[i+1] + U[i+2]
                    dU[2][i] -= ko_fac * _odd_physical(stencil, grid.r[i])
                end
            elseif k == 3
                U = odd_proxy_piQr
                for i in 3:(Nr-2)
                    stencil = U[i-2] - 4*U[i-1] + 6*U[i] - 4*U[i+1] + U[i+2]
                    dU[3][i] -= ko_fac * _odd_physical(stencil, grid.r[i])
                end
            elseif k == 4
                U = qperp_proxy
                for i in 3:(Nr-2)
                    stencil = U[i-2] - 4*U[i-1] + 6*U[i] - 4*U[i+1] + U[i+2]
                    dU[4][i] -= ko_fac * stencil
                end
            else
                U = fields[k]
                for i in 3:(Nr-2)
                    dU[k][i] -= ko_fac * (U[i-2] - 4*U[i-1] + 6*U[i] - 4*U[i+1] + U[i+2])
                end
            end
        end
    end

    return max_charspeed
end

# ═══════════════════════ Time stepper (RK4) ════════════════════════
"""
Step the 5-component IS2 system forward by dt using classical RK4.
RK4 is stable on the imaginary axis for |ω| ≤ 2√2 ≈ 2.83, which is
needed when cM ≠ 0 introduces oscillatory modes with purely imaginary
eigenvalues that forward Euler and RK2 (Heun) cannot handle.
Returns the maximum characteristic speed across all grid points.
"""
function step_IS2!(
    α::Vector{Float64},
    νr::Vector{Float64},
    piQr::Vector{Float64},
    piQperp::Vector{Float64},
    PiQ_field::Vector{Float64},
    odd_proxy_νr::Vector{Float64},
    odd_proxy_piQr::Vector{Float64},
    qperp_proxy::Vector{Float64},
    τ::Float64, dt::Float64,
    grid::IS2Grid1D,
    bg::IS2Background;
    DsT,
    T_floor::Float64,
    T_cM_min::Float64,
    eos,
    use_cM::Bool,
    σ_KO::Float64,
    # workspace arrays (pre-allocated)
    slopes::NTuple{5, Vector{Float64}},
    dU::NTuple{5, Vector{Float64}},
    dU2::NTuple{5, Vector{Float64}},
    U_pred::NTuple{5, Vector{Float64}},
    At::Matrix{Float64},
    Ax::Matrix{Float64},
    B::Matrix{Float64},
    src::Vector{Float64},
    src_term::Vector{Float64},
    src_face_term::Vector{Float64},
    rhs::Vector{Float64},
    U_left::Vector{Float64},
    U_right::Vector{Float64},
    U_left_in::Vector{Float64},
    U_right_in::Vector{Float64},
    U_mid::Vector{Float64},
    jump_state::Vector{Float64},
    D_left::Vector{Float64},
    D_right::Vector{Float64},
    q_old::Vector{Float64},
    q_tmp::Vector{Float64},
    flux_q::Vector{Float64},
    diagnostics::IS2Diagnostics,
    fail_on_linear_failure::Bool,
    fail_on_eigen_failure::Bool,
    max_solver_warnings::Int,
)
    Nr = length(grid.r)
    _state_to_q!(q_old, α, νr, τ, grid, bg; T_floor=T_floor, eos=eos)
    rhs_kw = (DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM,
              σ_KO=σ_KO, slopes=slopes, At=At, Ax=Ax, B=B,
              src=src, src_term=src_term, src_face_term=src_face_term, rhs=rhs,
              U_left=U_left, U_right=U_right, U_left_in=U_left_in, U_right_in=U_right_in,
              U_mid=U_mid, jump_state=jump_state, D_left=D_left, D_right=D_right,
              diagnostics=diagnostics,
              fail_on_linear_failure=fail_on_linear_failure,
              fail_on_eigen_failure=fail_on_eigen_failure,
              max_solver_warnings=max_solver_warnings)

    # ── FIVO_NU_RAPIDITY: chart component 2 by its rapidity for the duration of this step ────────
    # N is frozen at the step's start, so the map nu^r <-> phi is a fixed diffeomorphism of
    # (-N, N) onto R throughout the four stages and RK4 stays fourth-order in phi. `φ0[i] = NaN`
    # marks a cell that is already outside; those keep the plain additive update.
    nurap = IS2_NU_RAPIDITY
    N0   = nurap ? Vector{Float64}(undef, Nr) : Float64[]
    φ0   = nurap ? Vector{Float64}(undef, Nr) : Float64[]
    φacc = nurap ? Vector{Float64}(undef, Nr) : Float64[]
    dφ   = nurap ? Vector{Float64}(undef, Nr) : Float64[]
    if nurap
        @inbounds for i in 1:Nr
            T = max(bg_T(bg, τ, grid.r[i]), T_floor)
            uτ, _, _ = _u_from_ur(bg_ur(bg, τ, grid.r[i]))
            n_eq = max(eos_Pne(T, 0.0, eos)[2], 1e-300)
            N0[i] = max(n_eq * exp(α[i]) * uτ, 1e-300)
            x = νr[i] / N0[i]
            if abs(x) < 1.0 - 1e-12
                φ0[i] = atanh(x)
            else
                φ0[i] = NaN
                diagnostics.nu_rapidity_fallbacks += 1
            end
        end
    end
    # dphi/dtau from dnu^r/dtau, evaluated at the state the slope was computed at (`νs`).
    @inline nu_dphi(dν, νs, i) = (x = νs / N0[i]; g = 1.0 - x*x;
                                  g <= 0.0 ? 0.0 : (dν / N0[i]) / g)
    # phi advanced from the step's start by `h * s`, mapped back to nu^r. `s` is the stage slope for
    # a predictor and the RK4 accumulation for the final update.
    @inline nu_from_phi(h, s, i, νfallback) = isnan(φ0[i]) ? νfallback :
                                              N0[i] * tanh(φ0[i] + h * s)

    # We accumulate the weighted sum in dU2: dU2 = (k1 + 2k2 + 2k3 + k4)/6
    # Using dU for each stage's evaluation and U_pred for the intermediate state.

    # Stage 1: k1 = f(τ, U)
    max_cs = _compute_dUdt!(dU, α, νr, piQr, piQperp, PiQ_field, odd_proxy_νr, odd_proxy_piQr, qperp_proxy, τ, grid, bg; rhs_kw...)
    # Accumulate: dU2 = k1
    @inbounds for i in 1:Nr
        dU2[1][i] = dU[1][i]; dU2[2][i] = dU[2][i]; dU2[3][i] = dU[3][i]
        dU2[4][i] = dU[4][i]; dU2[5][i] = dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            dφ[i] = nu_dphi(dU[2][i], νr[i], i); φacc[i] = dφ[i]
        end
    end

    # Stage 2: k2 = f(τ + dt/2, U + dt/2 * k1)
    @inbounds for i in 1:Nr
        U_pred[1][i] = α[i]         + 0.5*dt*dU[1][i]
        U_pred[2][i] = νr[i]        + 0.5*dt*dU[2][i]
        U_pred[3][i] = piQr[i]      + 0.5*dt*dU[3][i]
        U_pred[4][i] = piQperp[i]   + 0.5*dt*dU[4][i]
        U_pred[5][i] = PiQ_field[i] + 0.5*dt*dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            U_pred[2][i] = nu_from_phi(0.5*dt, dφ[i], i, U_pred[2][i])
        end
    end
    max_cs = max(max_cs, _compute_dUdt!(dU, U_pred[1], U_pred[2], U_pred[3], U_pred[4], U_pred[5], odd_proxy_νr, odd_proxy_piQr, qperp_proxy,
                                         τ + 0.5*dt, grid, bg; rhs_kw...))
    # Accumulate: dU2 += 2*k2
    @inbounds for i in 1:Nr
        dU2[1][i] += 2*dU[1][i]; dU2[2][i] += 2*dU[2][i]; dU2[3][i] += 2*dU[3][i]
        dU2[4][i] += 2*dU[4][i]; dU2[5][i] += 2*dU[5][i]
    end
    if nurap                       # U_pred[2] still holds the state k2 was evaluated at
        @inbounds for i in 1:Nr
            dφ[i] = nu_dphi(dU[2][i], U_pred[2][i], i); φacc[i] += 2*dφ[i]
        end
    end

    # Stage 3: k3 = f(τ + dt/2, U + dt/2 * k2)
    @inbounds for i in 1:Nr
        U_pred[1][i] = α[i]         + 0.5*dt*dU[1][i]
        U_pred[2][i] = νr[i]        + 0.5*dt*dU[2][i]
        U_pred[3][i] = piQr[i]      + 0.5*dt*dU[3][i]
        U_pred[4][i] = piQperp[i]   + 0.5*dt*dU[4][i]
        U_pred[5][i] = PiQ_field[i] + 0.5*dt*dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            U_pred[2][i] = nu_from_phi(0.5*dt, dφ[i], i, U_pred[2][i])
        end
    end
    max_cs = max(max_cs, _compute_dUdt!(dU, U_pred[1], U_pred[2], U_pred[3], U_pred[4], U_pred[5], odd_proxy_νr, odd_proxy_piQr, qperp_proxy,
                                         τ + 0.5*dt, grid, bg; rhs_kw...))
    # Accumulate: dU2 += 2*k3
    @inbounds for i in 1:Nr
        dU2[1][i] += 2*dU[1][i]; dU2[2][i] += 2*dU[2][i]; dU2[3][i] += 2*dU[3][i]
        dU2[4][i] += 2*dU[4][i]; dU2[5][i] += 2*dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            dφ[i] = nu_dphi(dU[2][i], U_pred[2][i], i); φacc[i] += 2*dφ[i]
        end
    end

    # Stage 4: k4 = f(τ + dt, U + dt * k3)
    @inbounds for i in 1:Nr
        U_pred[1][i] = α[i]         + dt*dU[1][i]
        U_pred[2][i] = νr[i]        + dt*dU[2][i]
        U_pred[3][i] = piQr[i]      + dt*dU[3][i]
        U_pred[4][i] = piQperp[i]   + dt*dU[4][i]
        U_pred[5][i] = PiQ_field[i] + dt*dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            U_pred[2][i] = nu_from_phi(dt, dφ[i], i, U_pred[2][i])
        end
    end
    max_cs = max(max_cs, _compute_dUdt!(dU, U_pred[1], U_pred[2], U_pred[3], U_pred[4], U_pred[5], odd_proxy_νr, odd_proxy_piQr, qperp_proxy,
                                         τ + dt, grid, bg; rhs_kw...))
    # Accumulate: dU2 += k4
    @inbounds for i in 1:Nr
        dU2[1][i] += dU[1][i]; dU2[2][i] += dU[2][i]; dU2[3][i] += dU[3][i]
        dU2[4][i] += dU[4][i]; dU2[5][i] += dU[5][i]
    end
    if nurap
        @inbounds for i in 1:Nr
            φacc[i] += nu_dphi(dU[2][i], U_pred[2][i], i)
        end
    end

    # Final RK4 update: U += dt/6 * (k1 + 2k2 + 2k3 + k4)
    c = dt / 6.0
    @inbounds for i in 1:Nr
        α[i]         = _alpha_floor(α[i] + c * dU2[1][i])
        νr[i]        = nurap ? nu_from_phi(c, φacc[i], i, νr[i] + c * dU2[2][i]) :
                               νr[i] + c * dU2[2][i]
        piQr[i]      += c * dU2[3][i]
        piQperp[i]   += c * dU2[4][i]
        PiQ_field[i] += c * dU2[5][i]
    end

    # Regularity BC at r = 0 (α, πQperp, PiQ are even; νr, πQr are odd)
    if Nr >= 2
        fac = grid.r[1] / max(grid.r[2], 1e-12)
        α[1]         = α[2]
        νr[1]        = νr[2] * fac
        piQr[1]      = piQr[2] * fac
        piQperp[1]   = piQperp[2]
        PiQ_field[1] = PiQ_field[2]
    end

    if IS2_USE_Q_RECOVERY
        _build_q_flux!(flux_q, q_old, νr, τ + dt, grid, bg)
        _transport_update_q!(q_tmp, q_old, flux_q, τ + dt, dt, grid)
        _recover_alpha_from_q!(α, q_tmp, νr, τ + dt, grid, bg; T_floor=T_floor, eos=eos)
        if Nr >= 2
            α[1] = α[2]
        end
    end

    return max_cs
end

# ═══════════════════════ Solver loop ═══════════════════════════════
function solve_IS2(
    grid::IS2Grid1D,
    α0::Vector{Float64}, νr0::Vector{Float64},
    piQr0::Vector{Float64}, piQperp0::Vector{Float64}, PiQ0::Vector{Float64},
    τ0::Float64, τf::Float64,
    bg::IS2Background;
    CFL::Float64, CFLτ::Float64, save_dt::Float64,
    log_every::Int = 50,
    DsT, T_floor::Float64, T_cM_min::Float64,
    eos, use_cM::Bool, σ_KO::Float64,
    fail_on_linear_failure::Bool = false,
    fail_on_eigen_failure::Bool = false,
    max_solver_warnings::Int = 20,
)
    Nr = length(grid.r)

    # Working arrays
    α = copy(α0); νr = copy(νr0)
    piQr = copy(piQr0); piQperp = copy(piQperp0); PiQ_field = copy(PiQ0)

    # Pre-allocate workspace
    slopes = ntuple(_ -> zeros(Nr), 5)
    dU = ntuple(_ -> zeros(Nr), 5)
    dU2 = ntuple(_ -> zeros(Nr), 5)       # RK2 stage 2
    U_pred = ntuple(_ -> zeros(Nr), 5)    # RK2 predicted state
    At = zeros(5, 5)
    Ax = zeros(5, 5)
    B = zeros(5, 5)
    src = zeros(5)
    src_term = zeros(5)
    src_face_term = zeros(5)
    rhs = zeros(5)
    U_left = zeros(5)
    U_right = zeros(5)
    U_left_in = zeros(5)
    U_right_in = zeros(5)
    U_mid = zeros(5)
    jump_state = zeros(5)
    D_left = zeros(5)
    D_right = zeros(5)
    q_old = zeros(Nr)
    q_tmp = zeros(Nr)
    flux_q = zeros(Nr + 1)
    odd_proxy_νr = zeros(Nr)
    odd_proxy_piQr = zeros(Nr)
    qperp_proxy = zeros(Nr)
    diagnostics = IS2Diagnostics()

    τ = τ0; it = 0; next_dump = τ0 + save_dt

    # Storage (initial state always saved)
    τs = Float64[τ]
    αs = Vector{Float64}[copy(α)]
    νrs = Vector{Float64}[copy(νr)]
    piQrs = Vector{Float64}[copy(piQr)]
    piQperps = Vector{Float64}[copy(piQperp)]
    PiQs = Vector{Float64}[copy(PiQ_field)]

    # Initial signal speed estimate (flow velocity baseline)
    prev_charspeed = 0.0
    for i in 1:Nr
        ur_val = bg_ur(bg, τ, grid.r[i])
        uτ = sqrt(1.0 + ur_val^2)
        prev_charspeed = max(prev_charspeed, abs(ur_val / uτ))
    end
    if use_cM
        # Compute eigenvalue-based signal speed from initial state
        for i in 1:max(1, Nr÷20):Nr
            r = grid.r[i]
            T_val, ur_val, dtT, drT, drur, dtur = bg_fields(bg, τ, r)
            T_val = max(T_val, T_floor)
            α_safe = _alpha_floor(α[i])
            _set_state!(U_mid, α_safe, νr[i], piQr[i], piQperp[i], PiQ_field[i])
            ok_state, c_state, n_state = _build_reduced_system!(
                B, src_term, At, Ax, src, U_mid, τ, r, bg;
                DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM,
                diagnostics=diagnostics,
                fail_on_linear_failure=fail_on_linear_failure,
                fail_on_eigen_failure=fail_on_eigen_failure,
                max_solver_warnings=max_solver_warnings,
            )
            if ok_state
                prev_charspeed = max(prev_charspeed, _cfl_speed(c_state, n_state))
            end
        end
    end
    prev_charspeed = max(prev_charspeed, 1e-8)
    nsteps = 0

    while τ < τf - 1e-12
        nsteps += 1
        # CFL from max characteristic speed (eigenvalue-based when cM≠0)
        dt_adv = CFL * minimum(grid.dr) / prev_charspeed
        dt_tau = CFLτ * τ
        Δτ = min(dt_adv, dt_tau)
        if τ + Δτ > τf
            Δτ = τf - τ
        end

        τ_eval = τ + Δτ
        charspeed = step_IS2!(α, νr, piQr, piQperp, PiQ_field, odd_proxy_νr, odd_proxy_piQr, qperp_proxy, τ, Δτ, grid, bg;
                  DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min,
                  eos=eos, use_cM=use_cM, σ_KO=σ_KO,
                  slopes=slopes, dU=dU, dU2=dU2, U_pred=U_pred,
                  At=At, Ax=Ax, B=B, src=src,
                  src_term=src_term, src_face_term=src_face_term, rhs=rhs,
                  U_left=U_left, U_right=U_right, U_left_in=U_left_in, U_right_in=U_right_in,
                  U_mid=U_mid, jump_state=jump_state, D_left=D_left, D_right=D_right,
                  q_old=q_old, q_tmp=q_tmp, flux_q=flux_q,
                  diagnostics=diagnostics,
                  fail_on_linear_failure=fail_on_linear_failure,
                  fail_on_eigen_failure=fail_on_eigen_failure,
                  max_solver_warnings=max_solver_warnings)
        prev_charspeed = max(charspeed, 1e-8)
        τ = τ_eval
        it += 1

        if τ >= next_dump - 1e-12
            push!(τs, τ); push!(αs, copy(α)); push!(νrs, copy(νr))
            push!(piQrs, copy(piQr)); push!(piQperps, copy(piQperp))
            push!(PiQs, copy(PiQ_field))
            next_dump += save_dt
        end

        if (it % log_every) == 0
            @info "IS2 progress" τ=τ Δτ=Δτ it=it charspeed=prev_charspeed
        end
    end

    if τs[end] != τ
        push!(τs, τ); push!(αs, copy(α)); push!(νrs, copy(νr))
        push!(piQrs, copy(piQr)); push!(piQperps, copy(piQperp))
        push!(PiQs, copy(PiQ_field))
    end

    diagnostics.steps = nsteps
    return τs, αs, νrs, piQrs, piQperps, PiQs, diagnostics
end

# ═══════════════════════ Entry point ══════════════════════════════
"""
    run_static_IS2_test(; kwargs...)

Run the 5-component IS2 evolution on a static (or flowing) background.
Returns a `Dict` with keys: `"r_grid"`, `"t_grid"`, `"n"`, `"nur"`,
`"piQr"`, `"piQperp"`, `"PiQ"` (2D arrays [Nr × Nt]).
"""
function run_static_IS2_test(;
    background_file::String,
    DsT = 0.1163,   # canonical const D_sT = 0.116 (= 1.765·T_fo − 0.159 at T_fo = 0.156)
    τ0::Float64  = 0.4,
    τfinal::Float64 = 8.0,
    Nr::Int = 300,
    rmax::Float64 = 25.0,
    CFL::Float64  = 0.15,
    CFLτ::Float64 = 0.03,
    dump_dt::Float64 = 0.1,
    T_floor::Float64 = 1e-6,
    init_mode::Symbol = :auto,
    n_profile = r -> exp(-r^2 / (2.0 * 4.0^2)),
    nur_profile = nothing,
    ic_dict = nothing,
    use_cM::Bool = false,
    T_cM_min::Float64 = 0.12,   # disable cM below this T [GeV] to avoid stiffness
    σ_KO::Float64 = 0.2,   # modest Kreiss-Oliger dissipation damps residual grid-scale wiggles
    outdir::String = "",
    # Grand-canonical charm EOS (canon_factor=1.0): κ = D_s·n_GC matches Fluidum's grand-canonical
    # `normalization` exactly. The physical charm number N is enforced by the caller's conserved-
    # current rescale (is2_dropin: scale = adv0/is2_0), mirroring Fluidum's fugacity/N matching —
    # NOT by suppressing κ. The old LatticeHRGEOS() default carried canon_factor≈0.952 (hardcoded
    # N=21.55), the source of the spurious ~1.05 κ mismatch.
    eos = LatticeHRGEOS(canon_factor = 1.0),
    log_every::Int = 100,
    fail_on_linear_failure::Bool = false,
    fail_on_eigen_failure::Bool = false,
    max_solver_warnings::Int = 20,
)
    bg = load_IS2_background(background_file)
    rmax_use = min(rmax, last(bg.r_grid))
    grid = IS2Grid1D(Nr, rmax_use)

    use_background_ic = init_mode === :background || (init_mode === :auto && bg.α_spl !== nothing)

    # ── Initialise α from background or n_profile ────────────────
    α = zeros(Nr)
    νr = zeros(Nr)
    for i in eachindex(grid.r)
        r = grid.r[i]
        if use_background_ic
            α[i], _, νr[i] = initial_state_from_background(bg, τ0, r, T_floor, eos)
        else
            T = max(bg_T(bg, τ0, r), T_floor)
            n_want = n_profile(r)
            _, n_eq0, _ = eos_Pne(T, 0.0, eos)
            α[i] = n_eq0 > TINY ? _alpha_floor(log(max(n_want / n_eq0, TINY))) : 0.0
            νr[i] = nur_profile === nothing ? 0.0 : Float64(nur_profile(r))
        end
    end

    # ── Initialise second moments from ic_dict ────────────────────
    piQr = zeros(Nr)
    piQperp = zeros(Nr)
    PiQ_field = zeros(Nr)

    if ic_dict !== nothing
        ic_r = Float64.(ic_dict["r_grid"])
        for (field, ic_key) in ((piQr, "piQr"), (PiQ_field, "PiQ"))
            ic_vals = Float64.(ic_dict[ic_key])
            itp = extrapolate(interpolate((ic_r,), ic_vals, Gridded(Linear())), Flat())
            for i in eachindex(grid.r)
                field[i] = itp(grid.r[i])
            end
        end
        piQperp_vals = Float64.(ic_dict["piQperp"])
        piQperp_itp = extrapolate(interpolate((ic_r,), piQperp_vals, Gridded(Linear())), Flat())
        for i in eachindex(grid.r)
            piQperp[i] = piQperp_itp(grid.r[i])   # bounded eigenvalue π_perp^Q (no /r⁴)
        end
        if Nr >= 2
            piQperp[1] = piQperp[2]
        end
    end

    @info "Starting IS2 evolution" Nr=Nr rmax=rmax_use τ0=τ0 τfinal=τfinal DsT=DsT use_cM=use_cM T_cM_min=T_cM_min σ_KO=σ_KO

    # Clear the consistent-cM lag scratch HERE so no driver has to remember: a stale
    # same-length array from a previous solve would pass the reader's length check and feed
    # the first stage an unrelated solve's ∂τζ (audit finding E2, 2026-08-25).
    IS2_CM_DT3[] = Float64[]; IS2_CM_DT5[] = Float64[]

    τs, αs, νrs, piQrs, piQperps, PiQs, diagnostics = solve_IS2(
        grid, α, νr, piQr, piQperp, PiQ_field,
        τ0, τfinal, bg;
        CFL=CFL, CFLτ=CFLτ, save_dt=dump_dt, log_every=log_every,
        DsT=DsT, T_floor=T_floor, T_cM_min=T_cM_min, eos=eos, use_cM=use_cM, σ_KO=σ_KO,
        fail_on_linear_failure=fail_on_linear_failure,
        fail_on_eigen_failure=fail_on_eigen_failure,
        max_solver_warnings=max_solver_warnings)

    # ── Build result dict ────────────────────────────────────────
    Nt = length(τs)
    n_arr      = zeros(Nr, Nt)
    nur_arr    = zeros(Nr, Nt)
    piQr_arr   = zeros(Nr, Nt)
    piQperp_arr = zeros(Nr, Nt)
    PiQ_arr    = zeros(Nr, Nt)
    alpha_arr  = zeros(Nr, Nt)   # grand-canonical fugacity α=μ/T=log(n/n_eq), solved coupled with νr

    for it in 1:Nt
        τ = τs[it]
        for i in 1:Nr
            T = max(bg_T(bg, τ, grid.r[i]), T_floor)
            α_val = clamp(αs[it][i], -200.0, 200.0)
            _, nv, _ = eos_Pne(max(T, T_MIN), α_val * max(T, T_MIN), eos)
            n_arr[i, it] = max(nv, 0.0)
            alpha_arr[i, it] = α_val
        end
        nur_arr[:, it]  .= νrs[it]
        piQr_arr[:, it] .= piQrs[it]
        PiQ_arr[:, it]  .= PiQs[it]
        @inbounds for i in 1:Nr
            piQperp_arr[i, it] = _piQperp_physical(piQperps[it][i], grid.r[i])
        end
    end

    results = Dict{String, Any}(
        "r_grid"  => copy(grid.r),
        "t_grid"  => copy(τs),
        "n"       => n_arr,
        "nur"     => nur_arr,
        "alpha"   => alpha_arr,
        "piQr"    => piQr_arr,
        "piQperp" => piQperp_arr,
        "PiQ"     => PiQ_arr,
        "diagnostics" => Dict(
            "steps" => diagnostics.steps,
            "linear_failures" => diagnostics.linear_failures,
            "eigen_failures" => diagnostics.eigen_failures,
            "nu_rapidity_fallbacks" => diagnostics.nu_rapidity_fallbacks,
            "nu_bound_hits" => IS2_NU_BOUND_HITS[],
            "jtau_nonpositive" => IS2_JTAU_NEG[],
            # cell-steps on which the CONE PROJECTION actually moved the state. Zero means the
            # solution never left the admissible cone and the projection is inert -- which is the
            # difference between "the solver stayed physical" and "the cap held it there".
            "cone_projections" => IS2_CONE_HITS[],
        ),
    )

    if !isempty(outdir)
        mkpath(outdir)
        @info "IS2 results computed" τs_range=(first(τs), last(τs)) Nt=Nt diagnostics=results["diagnostics"]
    elseif diagnostics.linear_failures > 0 || diagnostics.eigen_failures > 0
        @warn "IS2 run completed with solver diagnostics" diagnostics=results["diagnostics"]
    end

    return results
end

end # module hydro_current_IS2
