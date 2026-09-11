
# =============================================================================
# src/hq_consistent_m2.jl — THE THERMODYNAMICALLY CONSISTENT SECOND MOMENT for
# the charm sector (2026-09-08). FiVo-side twin of Fluidum's
# Julia/Fluidum.jl/src/Matrix/HQ_const_BG_consistent_m2.jl (`hqc_m2_rows`);
# derivation Tex/LangevinPaper1/M2_CONSISTENT_DERIVATION.md, symbolic twin
# Julia/tools/derive_hq_consistent_m2.wls.
#
# Companion of src/hq_consistent_firstmoment.jl, one moment up: that file
# corrects rows 1-2 (α, ν^r); this one corrects rows 3-5 (π_r, π_perp, Π_Q).
#
# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS: THE SHIPPED ROWS ARE NOT THE ∇ν-ONLY LIMIT
# -----------------------------------------------------------------------------
# FiVo's `_second_moment_eigen_rhs!` was derived by inverting Fluidum's SHIPPED
# 5×5 second-moment matrix, and it reproduces it to machine precision. But the
# shipped rows are not the clean covariant ∇ν-only reduction, and the difference
# is not small. MEASURED by Tex/MaxEntHydro/diag_fivo_m2_gates.jl (gate M3):
#
#     the ∂_rν drive coefficient of FiVo's shipped rows is EXACTLY MINUS the
#     covariant one — |ratio - 1| = 2.0000 at every one of the 24 points.
#
# That is the mostly-minus notebook heritage HQ_const_BG_consistent_m2.jl's
# header names (same family as the W0/C0 sign traps), and the shipped rows also
# carry coordinate-transport couplings among (π_r, π_perp, Π_Q) that the
# parallel-transported basis shows to be absent from Δ-projected Dπ. So the
# shipped system cannot be repaired by adding terms to it; the consistent rows
# are built here from the covariant reduction directly.
#
# -----------------------------------------------------------------------------
# THE BASIS — AND WHY NO TRANSFORMATION IS NEEDED
# -----------------------------------------------------------------------------
# Fluidum's (p_l, p_φ, Π_Q) is the parallel-transported orthonormal triad
#     l = (u^r, u^τ, 0, 0),  φ̂ = (0, 0, 1/r, 0),  η̂ = (0, 0, 0, 1/τ),
# for which D l = a_l u (so Δ·Dl = 0) and D φ̂ = D η̂ = 0, hence Δ-projected Dπ is
# DIAGONAL with entries D p_i. FiVo's (π_r, π_perp, Π_Q) comes from
# Julia/tools/derive_2nd_moment_eigenbasis.wl, which builds
#     π^{μν} = π_r (u^r,u^τ)⊗(u^r,u^τ) + (π_perp/r²) φφ + (π_η/τ²) ηη
# — the same tensor. So p_l ≡ π_r and p_φ ≡ π_perp IDENTICALLY, with π_η =
# -(π_r + π_perp) by tracelessness in both. Gate M0 checks this on the one entry
# that cannot lie about it (the relaxation slot must be τ_M u^τ in both codes)
# and measures 0.00e+00.
#
# -----------------------------------------------------------------------------
# THE EQUATIONS (all-on-LHS = 0, D = u^τ∂_τ + u^r∂_r, D_l = u^r∂_τ + u^τ∂_r)
# -----------------------------------------------------------------------------
# Traceless channels i ∈ {l, φ}  (the η channel follows by tracelessness):
#     τ_M D p_i + p_i + 2 η_Q σν_i                                     (legacy)
#       + τ_M [ ((5/3)θ + D ln C) p_i + 2(p_i s_i - π:σ/3) + 2 Π_Q s_i ]
#       + 2 η̄ s_i
#       + [ 2 λ_a a_l ν_l + (D_s/T) ν_l D_l(Th) ] (δ_il - 1/3)
#     = 0
# Trace channel:
#     τ_M D Π_Q + Π_Q + ζ_Q θ_ν                                        (legacy)
#       + τ_M ((5/3)θ + D ln C) Π_Q + (2/3) τ_M π:σ
#       + (τ_n P₀/2) [ Dα + (A/B) D ln T + (5/3)θ ]
#       + (D_s A/3T) a·ν + (5 D_s/6T) ν_l D_l(Th)
#     = 0
#
# COEFFICIENTS (z = m/T):
#     τ_M = τ_n K₄K₂/(2K₃²)      η_Q = T τ_n/2 (= η_M)     ζ_Q = (5/3) η_Q
#     η̄ = τ_n P₀/2, P₀ = nT      λ_a = (D_s/2T)(m² + 6Th)
#     A = m² + 5Th               A/B = 5 + z K₂/K₃
#     D ln C = z[(K₃+K₅)/(2K₄) - (K₂+K₄)/(2K₃)] D ln T,  K₅ = K₃ + 8K₄/z
#     D_l(Th) = (h + T h′) D_l T
#
# -----------------------------------------------------------------------------
# WHAT THIS RETURNS, AND THE SIGN CONVENTION
# -----------------------------------------------------------------------------
# `hq_consistent_m2_rhs` returns ∂_τ(π_r, π_perp, Π_Q) DIRECTLY — the same thing
# `_second_moment_eigen_rhs!` produces — not LHS matrix rows. Fluidum's rows are
# `At ∂_τφ + Ax ∂_rφ + src = 0` with At diagonal in the moments (τ_M u^τ), so
#
#     ∂_τ p_i = -( at[i,1] ∂_τα + at[i,2] ∂_τν + ax[i,1] ∂_rα
#                + ax[i,2] ∂_rν + ax[i,2+i] ∂_r p_i + src_i ) / (τ_M u^τ)
#
# and that is exactly the assembly below. Gate M4 checks it against the Fluidum
# rows on identical states and identical handed-in gradients.
#
# FLAG: `IS2_CONSISTENT_M2` (main2IS2.jl), env FIVO_IS2_CONSISTENT_M2, default
# OFF = the shipped rows, byte-identical to every number produced before this
# file. It is INDEPENDENT of IS2_CONSISTENT_FM: the first and second moments can
# be switched separately, which is what makes the four-way attribution possible.
# =============================================================================

# The per-term switches (`Terms`) live in the shared register. Every solver
# includes it first; the guard lets a script include this file on its own
# (test_consistent_*2d.jl, plot_operator_closure_ratio.jl do).
@isdefined(Terms) || include(joinpath(@__DIR__, "terms.jl"))

"""
    hq_m2_coeffs(T, n, τn, Ds, h, hp, m) -> NamedTuple

The consistent second moment's coefficients at one state. `τ_M` and `η_M` are
NOT recomputed here — the caller passes the ones `transport_all` already built,
so this file cannot drift from the row's own relaxation time.

`dlnC_dlnT` is the logarithmic derivative of C = I₅₂/I₄₂ that the (iii) class
needs; the closed form is `z[(K₃+K₅)/(2K₄) - (K₂+K₄)/(2K₃)]` with
`K₅ = K₃ + 8K₄/z` from the Bessel recurrence.
"""
@inline function hq_m2_coeffs(T::Float64, n::Float64, τn::Float64, Ds::Float64,
                              h::Float64, m::Float64)
    Tm = max(T, T_MIN)
    z  = m / Tm
    K2 = SpecialFunctions.besselkx(2, z)
    K3 = SpecialFunctions.besselkx(3, z)
    K4 = SpecialFunctions.besselkx(4, z)
    K5 = K3 + 8 * K4 / z
    dlnC_dlnT = (abs(K4) > TINY && abs(K3) > TINY) ?
                z * ((K3 + K5) / (2 * K4) - (K2 + K4) / (2 * K3)) : 0.0
    ηbar = τn * n * Tm / 2                       # τ_n P₀/2, P₀ = nT
    λa   = Ds * (m^2 + 6 * Tm * h) / (2 * Tm)    # (D_s/2T)(A+B)
    Aco  = m^2 + 5 * Tm * h                      # A = I₄₁/P₀
    ABr  = (abs(K3) > TINY) ? 5 + z * K2 / K3 : 5.0   # A/B = dln I₃₁/dln T
    return (dlnC_dlnT = isfinite(dlnC_dlnT) ? dlnC_dlnT : 0.0,
            ηbar = ηbar, λa = λa, Aco = Aco,
            ABr = isfinite(ABr) ? ABr : 5.0)
end

"""
    hq_consistent_m2_rhs(τ, r, ur, T, dtT, drT, drur, dtur,
                         α, ν, p_l, p_φ, PiQ,
                         dtα, drα, dtν, drν, drp,
                         n, τn, Ds, h, hp, τM, ηM, m) -> (dp_l, dp_φ, dPiQ)

∂_τ of the three second moments under the CONSISTENT closure, in FiVo's
(π_r, π_perp, Π_Q) basis — which is Fluidum's (p_l, p_φ, Π_Q) identically.

Gradients are arguments, not differenced here, so the caller controls the
discretisation and the cross-code gate can hand both codes identical data.
`drp` is the triple (∂_r p_l, ∂_r p_φ, ∂_r Π_Q).

⚠ `τM` and `ηM` must be the SAME ones the first-moment rows use (`transport_all`
supplies both); passing a rescaled τ_M here and not there would put the two
halves of one system on different clocks.
"""
@inline function hq_consistent_m2_rhs(τ::Float64, r::Float64, ur::Float64, T::Float64,
                                      dtT::Float64, drT::Float64, drur::Float64, dtur::Float64,
                                      p_l::Float64, p_φ::Float64, PiQ::Float64, ν::Float64,
                                      dtα::Float64, drα::Float64, dtν::Float64, drν::Float64,
                                      drp::NTuple{3,Float64},
                                      n::Float64, τn::Float64, Ds::Float64,
                                      h::Float64, hp::Float64,
                                      τM::Float64, ηM::Float64, m::Float64;
                                      terms::Terms = Terms())
    Tm = max(T, T_MIN)
    uτ = sqrt(1.0 + ur^2)
    (τM > 0.0 && r > 0.0) || return (0.0, 0.0, 0.0)

    p_η = -(p_l + p_φ)

    # ── background kinematics in the transported triad ────────────────────────
    Dlur = ur * dtur + uτ * drur          # D_l u^r
    θ_l  = Dlur / uτ
    θ_φ  = ur / r
    θ_η  = uτ / τ
    θ    = θ_l + θ_φ + θ_η
    s_l  = θ_l - θ / 3
    s_φ  = θ_φ - θ / 3
    s_η  = θ_η - θ / 3
    a_l  = (uτ * dtur + ur * drur) / uτ   # a·l = (D u^r)/u^τ
    ν_l  = ν / uτ

    # ── σ_(ν): the ∇ν drive ───────────────────────────────────────────────────
    # (∇ν)_ll = D_l ν^r/u^τ - u^r ν^r (D_l u^r)/u^τ³, and D_l acts on ν as
    # u^r ∂_τν + u^τ ∂_rν — the handed-in gradients, so no differencing here.
    Dlν  = ur * dtν + uτ * drν
    gll  = Dlν / uτ - ur * ν * Dlur / uτ^3
    gφφ  = ν / r
    # (∇ν)_ηη = u^r ν/(u^τ τ) is NOT formed: only the l and φ channels are
    # evolved (the η one follows by tracelessness) and the trace it would
    # contribute is already in θν below, via the identity Σ_i (∇ν)_ii = θν - a·ν.
    aν   = a_l * ν_l
    # θ_ν = ∂_τν^τ + ∂_rν^r + ν^τ/τ + ν^r/r with ν^τ = u^r ν/u^τ.
    # ⚠ ∂_τν^τ = (u^r/u^τ)∂_τν + ν ∂_τ(u^r/u^τ), and ∂_τ(u^r/u^τ) = ∂_τu^r/u^τ³
    # (since u^τ² = 1 + u^r²). The ∂_τν coefficient is therefore u^r/u^τ, NOT u^r
    # — an earlier version here dropped the 1/u^τ and was wrong by that factor,
    # 1.4 at u^r = 0.97. Gate M4 caught it: 1.5/1.2/14.6 % on the three rows.
    θν   = (ur / uτ) * dtν + ν * dtur / uτ^3 + drν + ur * ν / (uτ * τ) + ν / r
    trν  = θν - aν                        # the trace that σ_(ν) removes
    σν_l = gll - trν / 3
    σν_φ = gφφ - trν / 3

    ηQ = ηM
    ζQ = (5.0 / 3.0) * ηQ

    cf = hq_m2_coeffs(Tm, n, τn, Ds, h, m)

    DlnT = (uτ * dtT + ur * drT) / Tm
    DlnC = cf.dlnC_dlnT * DlnT
    DlTh = (h + Tm * hp) * (ur * dtT + uτ * drT)      # D_l(Th)

    πσ  = p_l * s_l + p_φ * s_φ + p_η * s_η
    geo = τM * ((5.0 / 3.0) * θ + DlnC)
    rank1 = 2 * cf.λa * a_l * ν_l + (Ds / Tm) * ν_l * DlTh

    t = terms
    if t.m2_nu_gradient && t.m2_bg_gradu && t.m2_bg_DlnT && t.m2_bg_Dalpha && t.m2_expansion &&
       t.m2_pi_sigma && t.m2_PiQ_sigma && t.m2_accel_nu && t.m2_nu_gradTh
        # ALL ON — the rows exactly as they stood before the switches (bit-identical).
        # ── the LHS sources (Fluidum's src[1..3]) ─────────────────────────────
        src_l = p_l + 2 * ηQ * σν_l +
                geo * p_l + 2 * τM * (p_l * s_l - πσ / 3) + 2 * τM * PiQ * s_l +
                2 * cf.ηbar * s_l + rank1 * (2.0 / 3.0)
        src_φ = p_φ + 2 * ηQ * σν_φ +
                geo * p_φ + 2 * τM * (p_φ * s_φ - πσ / 3) + 2 * τM * PiQ * s_φ +
                2 * cf.ηbar * s_φ + rank1 * (-1.0 / 3.0)
        src_B = PiQ + ζQ * θν +
                geo * PiQ + (2.0 / 3.0) * τM * πσ +
                cf.ηbar * (cf.ABr * DlnT + (5.0 / 3.0) * θ) +
                (Ds * cf.Aco / (3 * Tm)) * aν + (5.0 / 6.0) * (Ds / Tm) * ν_l * DlTh

        # ── the Dα entry of the trace row (Fluidum's at3[3,1] / ax3[3,1]) ─────
        src_B += cf.ηbar * (uτ * dtα + ur * drα)
    else
        # SWITCHED — one `Terms` field per term (src/terms.jl), the names of the 2-D
        # `consistent_m2_source_2d`. `m2_vorticity` and `m2_projector` have no
        # counterpart here: both vanish identically in the transported triad
        # (l, φ̂, η̂) these channels are written in (see this file's header and
        # src/terms.jl, "RADIAL SYMMETRY"). Gate test_terms1d.jl T2: the pieces sum
        # to the all-on value.
        g  = t.m2_expansion ? geo : 0.0
        r1 = (t.m2_accel_nu  ? 2 * cf.λa * a_l * ν_l : 0.0) +
             (t.m2_nu_gradTh ? (Ds / Tm) * ν_l * DlTh : 0.0)
        src_l = p_l + (t.m2_nu_gradient ? 2 * ηQ * σν_l : 0.0) +
                g * p_l + (t.m2_pi_sigma ? 2 * τM * (p_l * s_l - πσ / 3) : 0.0) +
                (t.m2_PiQ_sigma ? 2 * τM * PiQ * s_l : 0.0) +
                (t.m2_bg_gradu ? 2 * cf.ηbar * s_l : 0.0) + r1 * (2.0 / 3.0)
        src_φ = p_φ + (t.m2_nu_gradient ? 2 * ηQ * σν_φ : 0.0) +
                g * p_φ + (t.m2_pi_sigma ? 2 * τM * (p_φ * s_φ - πσ / 3) : 0.0) +
                (t.m2_PiQ_sigma ? 2 * τM * PiQ * s_φ : 0.0) +
                (t.m2_bg_gradu ? 2 * cf.ηbar * s_φ : 0.0) + r1 * (-1.0 / 3.0)
        src_B = PiQ + (t.m2_nu_gradient ? ζQ * θν : 0.0) +
                g * PiQ + (t.m2_pi_sigma ? (2.0 / 3.0) * τM * πσ : 0.0) +
                cf.ηbar * ((t.m2_bg_DlnT ? cf.ABr * DlnT : 0.0) +
                           (t.m2_bg_gradu ? (5.0 / 3.0) * θ : 0.0)) +
                (t.m2_accel_nu  ? (Ds * cf.Aco / (3 * Tm)) * aν : 0.0) +
                (t.m2_nu_gradTh ? (5.0 / 6.0) * (Ds / Tm) * ν_l * DlTh : 0.0)
        src_B += t.m2_bg_Dalpha ? cf.ηbar * (uτ * dtα + ur * drα) : 0.0
    end

    # ── solve At ∂_τp + Ax ∂_rp + src = 0 for ∂_τp ────────────────────────────
    # At is diagonal in the moments with entry τ_M u^τ; Ax likewise with τ_M u^r.
    # The ∇ν drive's own ∂_τν entries are already inside src via Dlν and θν, so
    # what remains is the advective column and the relaxation.
    den  = τM * uτ
    dp_l = -(τM * ur * drp[1] + src_l) / den
    dp_φ = -(τM * ur * drp[2] + src_φ) / den
    dPiQ = -(τM * ur * drp[3] + src_B) / den
    return (dp_l, dp_φ, dPiQ)
end
