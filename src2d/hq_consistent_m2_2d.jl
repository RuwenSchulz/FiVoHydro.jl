
# =============================================================================
# src2d/hq_consistent_m2_2d.jl — THE CONSISTENT SECOND MOMENT IN 2+1D
# (2026-09-08). Transverse-Cartesian counterpart of src/hq_consistent_m2.jl,
# and the second-moment companion of src2d/hq_consistent_firstmoment2d.jl.
#
# Derivation: Tex/LangevinPaper1/M2_CONSISTENT_DERIVATION.md (the covariant
# equation, chart-independent); geometry and algebraic structure gated in
# Julia/tools/derive_hq_m2_2p1d.wls (9/9).
#
# -----------------------------------------------------------------------------
# WHY THIS IS A TENSOR HERE AND THREE SCALARS THERE
# -----------------------------------------------------------------------------
# The 1-D reduction evolves (p_l, p_φ, Π_Q) because azimuthal symmetry supplies a
# PARALLEL-TRANSPORTED orthonormal triad — l = (u^r,u^τ,0,0), φ̂, η̂ — for which
# D l = a_l u and D φ̂ = D η̂ = 0, so Δ-projected Dπ is DIAGONAL and three scalars
# close the system.
#
# The Cartesian transverse chart has no such triad: the flow picks no preferred
# transverse direction, so π_Q must be carried as a TENSOR. Boost invariance kills
# every η-mixed component (odd under η → −η), leaving ττ, τx, τy, xx, xy, yy, ηη
# = seven, cut by three orthogonality relations and one trace condition to THREE
# independent dofs — exactly the count `src2d/shear2d.jl` already uses for the
# MEDIUM shear (TWOD_PROGRAM.md §2).
#
# ⚠ SO THIS IS NOT src/hq_consistent_m2.jl WITH x FOR r. The 1-D file's three
# scalar equations are that tensor equation CONTRACTED on a triad that does not
# exist here. What ports is the covariant statement, not the reduced form.
#
# -----------------------------------------------------------------------------
# FIELDS — the medium shear's storage, reused exactly
# -----------------------------------------------------------------------------
#     pQxx, pQxy, pQyy   the transverse block π_Q^{ij}
#     pQeta              π_Q^η_η = τ² π_Q^{ηη}  (mixed component; ONE REDUNDANT
#                        dof, restored by projection)
#     PiQ                the trace channel
#
# The τ-row is DERIVED from the transverse block, never stored, so
# u_μ π_Q^{μν} = 0 holds by construction (gate G2car-orth). Tracelessness is the
# one constraint carried numerically and is restored by correcting pQeta alone —
# `project_shear_traceless_2d`, whose `pieta = Π^{tt} − π^{xx} − π^{yy}` was
# verified in the .wls to be algebraically identical to solving g_{μν}π^{μν} = 0.
# Correcting pQeta alone leaves the transverse block untouched and so preserves
# x↔y symmetry exactly, which is what lets gate Gm2 measure that symmetry.
#
# Because the storage and the constraint handling are IDENTICAL to the medium
# shear, this file REUSES `shear_tensor_contravariant_2d` and
# `project_shear_traceless_2d` rather than restating them. One convention, one
# implementation, and `test_shear2d_algebra.jl` already gates it.
#
# -----------------------------------------------------------------------------
# THE EQUATIONS (all-on-LHS = 0, D = u^μ∇_μ)
# -----------------------------------------------------------------------------
# Traceless sector, as a tensor:
#     τ_M Δ-proj D π_Q^{μν} + π_Q^{μν} + 2 η_Q σ_(ν)^{μν}                (legacy)
#       + τ_M [ ((5/3)θ + D ln C) π_Q^{μν}
#               + 2(π_Q^{⟨μ}_λ σ^{ν⟩λ} − (π:σ/3)Δ^{μν})
#               + 2 Π_Q σ^{μν} ]                                    (iii + dM-ext)
#       + 2 η̄ σ^{μν}                                                      (ii)
#       + [ 2 λ_a a^{⟨μ}ν^{ν⟩} + (D_s/T) ν^{⟨μ}∇^{ν⟩}(Th) ]              (iv)
#     = 0
# Trace channel (a scalar, so identical in form to 1-D):
#     τ_M D Π_Q + Π_Q + ζ_Q θ_ν
#       + τ_M ((5/3)θ + D ln C) Π_Q + (2/3) τ_M π:σ
#       + (τ_n P₀/2)[ Dα + (A/B) D ln T + (5/3)θ ]
#       + (D_s A/3T) a·ν + (5 D_s/6T) ν·∇(Th)
#     = 0
#
# with σ_(ν)^{μν} the shear of the CURRENT — the same construction as σ^{μν} with
# u → ν, which is why `ns_shear_target_2d`'s algebra is reused for it.
#
# ⚠ Dα IS THE TRACE ROW'S DOMINANT TERM. In 1-D, reading a FROZEN Dα (zeroed by
# IS2_FREEZE_PDE_ALPHA before the second-moment RHS ran) put Π_Q on the wrong
# side of its own fixed point — the fixed point moves ACROSS ZERO for a 0.3 %
# change in Dα. The 2-D solver does not freeze α, but the sensitivity is the
# same, so the caller must pass the PHYSICAL rate.
#
# FLAG: `IdealDiffVisc2DModel.consistent_m2`, default false. INDEPENDENT of
# `consistent_fm`, so the two moments can be attributed separately.
#
# -----------------------------------------------------------------------------
# WHAT WAS WRONG BEFORE IT PASSED (2026-09-08) — five errors, all one family
# -----------------------------------------------------------------------------
# Gate Gm1 (the axisymmetric reduction) took this file from 65 % to 2.3e-15 in
# five steps. FOUR of the five were the SAME mistake: contracting or tracing
# CONTRAVARIANT components as if they were orthonormal, i.e. forgetting that
# lowering an index on a u-orthogonal tensor brings in its τ-row.
#
#   1. σ_(ν) built BY ANALOGY with the u-shear — 65 %. The projector acting on
#      ∇^{(a}ν^{b)} drags in gradients of U through ν^τ = (u·ν_⊥)/u^τ. FIXED by
#      generating it (derive_signu_2p1d.wls, 3/3).
#   2. the class-(iv) rank-1 block symmetrised and traced by hand — 2.69x in the
#      φ channel. FIXED by `rank1_traceless_2d`, generated.
#   3. π:σ as `pxx σxx + 2 pxy σxy + pyy σyy + peta σeta` — 3-5 % on Π_Q.
#      Needs both indices lowered. Generated.
#   4. the (iii) coupling π^{(i}_λ σ^{j)λ} contracted over transverse indices
#      only — sub-1 % on p_l. Needs the metric. Generated.
#   5. a·ν and ν·∇(Th) as `ax νx + ay νy` — they carry −(a·u)(ν·u)/(u^τ)².
#   6. the π:σ trace subtracted as a bare `πσ/3` instead of `Δ^{ij} πσ/3` —
#      the last 0.006-0.57 % on p_l.
#
# ⛔ THE LESSON, written here because it will recur in any further 2-D tensor
# sector: in the 1-D chart every field is an ORTHONORMAL triad component, so
# contractions look like plain products and the metric is invisible. In the
# Cartesian chart the stored fields are CONTRAVARIANT and every contraction,
# trace and symmetrisation needs the metric explicitly. Generate them; do not
# transcribe the 1-D forms.
#
# The gate that caught all six is the 1-D limit. Rotational covariance (Gm2,
# 7.8e-16) catches a different class — an x/y transposition — that the 1-D limit
# is structurally blind to, because it has one transverse direction.
# =============================================================================


"""
    hq_m2_coeffs_2d(T, n, τn, Ds, h, m) -> NamedTuple

The consistent second moment's coefficients. Identical closed forms to the 1-D
`hq_m2_coeffs` — these are scalars built from the EoS, and nothing about them is
chart-dependent. Restated here rather than shared because `src/` cannot be
included into `hydro2d` (it drags in the 1-D solver types), the same reason
`transport2d.jl` restates the transport chain; `test_consistent_m22d.jl` gates
the two against each other, which is what keeps the duplication honest.
"""
@inline function hq_m2_coeffs_2d(T::Float64, n::Float64, τn::Float64, Ds::Float64,
                                 h::Float64, m::Float64)
    Tm = max(T, T_MIN)
    z  = m / Tm
    K2 = SpecialFunctions.besselkx(2, z)
    K3 = SpecialFunctions.besselkx(3, z)
    K4 = SpecialFunctions.besselkx(4, z)
    K5 = K3 + 8 * K4 / z
    dlnC_dlnT = (abs(K4) > TINY && abs(K3) > TINY) ?
                z * ((K3 + K5) / (2 * K4) - (K2 + K4) / (2 * K3)) : 0.0
    ηbar = τn * n * Tm / 2
    λa   = Ds * (m^2 + 6 * Tm * h) / (2 * Tm)
    Aco  = m^2 + 5 * Tm * h
    ABr  = (abs(K3) > TINY) ? 5 + z * K2 / K3 : 5.0
    return (dlnC_dlnT = isfinite(dlnC_dlnT) ? dlnC_dlnT : 0.0,
            ηbar = ηbar, λa = λa, Aco = Aco,
            ABr = isfinite(ABr) ? ABr : 5.0)
end

"""
    sigma_nu_2d(uxv, uyv, uT, tau, nux, nuy,
                dtnx, dxnx, dynx, dtny, dxny, dyny,
                dtux, dxux, dyux, dtuy, dxuy, dyuy)
        -> (σxx, σxy, σyy, σeta)

`σ_(ν)^{ij}`, the transverse-projected traceless symmetric gradient of the
current — GENERATED, not hand-written.

⚠ THIS IS NOT THE u-SHEAR WITH u → ν. The first version of this file assumed it
was and gate Gm1 measured **65 %** on p_l. The projector `Δ^i_a Δ^j_b` acts on the
FULL `∇^{(a}ν^{b)}`, whose τ-row involves `ν^τ = (u·ν_⊥)/u^τ` — a function of u.
Projecting therefore drags in terms carrying gradients of **U**, not of ν, which
no u → ν substitution can produce. The 1-D file shows the same structure in its
own chart: its `(∇ν)_ll = D_lν^r/u^τ − u^rν(D_lu^r)/u^τ³` has exactly such a
second term.

The expressions below are emitted by `Julia/tools/derive_signu_2p1d.wls`, which
forms `Δ^i_a Δ^j_b sym^{ab} − (1/3)Δ^{ij}(θ_ν − a·ν)` symbolically in Cartesian
Milne and gates (S1) that the full projection is what it prints, plus that
`Dν^i` carries no Christoffel piece for a transverse index (S2) and that
`θ_ν = ∂_τν^τ + ∂_iν^i + ν^τ/τ` (S3). 3/3.

Arguments are the solver's own symbols: `uT` is u^τ, `dtnx` = ∂_τν^x, `dxux` =
∂_xu^x, and so on. `σeta` is the MIXED component τ²σ_(ν)^{ηη}, matching how
`pQeta` is stored.
"""
@inline function sigma_nu_2d(uxv::Float64, uyv::Float64, uT::Float64, tau::Float64,
                             nux::Float64, nuy::Float64,
                             dtnx::Float64, dxnx::Float64, dynx::Float64,
                             dtny::Float64, dxny::Float64, dyny::Float64,
                             dtux::Float64, dxux::Float64, dyux::Float64,
                             dtuy::Float64, dxuy::Float64, dyuy::Float64)
    σxx = (tau*(1 + uxv^2 + uyv^2)*(2*dtnx*uxv^3 - dtny*uyv + 2*dxnx*uT - dyny*uT - uxv^2*(dtny*uyv - 2*dxnx*uT + dyny*uT) + uxv*(2*dtnx + 3*dtnx*uyv^2 + 3*dynx*uyv*uT)) + nuy*((-1 - dtux*tau*uxv - uxv^2 + 2*dtux*tau*uxv^3)*uyv^3 - 2*tau*uxv*(1 + uxv^2)^2*(dtuy*uxv + dxuy*uT) - tau*(-1 + 2*uxv^2)*uyv^2*(dtuy + dtuy*uxv^2 - dyux*uxv*uT) + (1 + uxv^2)*uyv*(-1 + 2*dtux*tau*uxv^3 + dyuy*tau*uT + uxv^2*(-1 + 2*dxux*tau*uT - 2*dyuy*tau*uT))) + nux*(uxv^5*(-1 + 2*dtuy*tau*uyv) + tau*uyv*(1 + uyv^2)*(dtux*uyv + dyux*uT) + 2*tau*uxv^4*(-dtux - dtux*uyv^2 + dxuy*uyv*uT) - 2*tau*uxv^2*(dtux + 2*dtux*uyv^2 + dtux*uyv^4 + (-dxuy + dyux)*uyv*uT + dyux*uyv^3*uT) - uxv*(1 + dtuy*tau*uyv^3 + 2*dxux*tau*uT + uyv^2*(1 + 2*dxux*tau*uT + dyuy*tau*uT)) + uxv^3*(2*dtuy*tau*uyv + 2*dtuy*tau*uyv^3 - 2*(1 + dxux*tau*uT) + uyv^2*(-1 - 2*dxux*tau*uT + 2*dyuy*tau*uT))))/(3*tau*(1 + uxv^2 + uyv^2)^(3/2))
    σxy = (tau*(1 + uxv^2 + uyv^2)*(3*dtny*uxv^3 + 3*dynx*(1 + uyv^2)*uT + 3*(dtnx*uyv*(1 + uyv^2) + dxny*uT) + uxv^2*(dtnx*uyv + 3*dxny*uT) + uxv*(3*dtny + dtny*uyv^2 + (dxnx + dyny)*uyv*uT)) - nuy*(4*dtuy*tau*uxv^5*uyv + 3*dxuy*tau*uyv*uT + 4*tau*uxv^4*uyv*(-(dtux*uyv) + dxuy*uT) - tau*uxv^2*uyv*(-7*dxuy*uT + 2*dtux*uyv*(3 + 2*uyv^2) + dyux*uT*(3 + 4*uyv^2)) + uxv^3*(10*dtuy*tau*uyv + 4*dtuy*tau*uyv^3 + 3*dyuy*tau*uT + uyv^2*(2 - 4*dxux*tau*uT + 4*dyuy*tau*uT)) + uxv*(6*dtuy*tau*uyv + 4*dtuy*tau*uyv^3 + 2*uyv^4 + 3*dyuy*tau*uT + uyv^2*(2 - 3*dxux*tau*uT + 4*dyuy*tau*uT))) + nux*(2*uxv^4*uyv*(-1 + 2*dtuy*tau*uyv) - 3*dxux*tau*uyv*(1 + uyv^2)*uT - 4*tau*uxv^3*uyv*(dtux + dtux*uyv^2 - dxuy*uyv*uT) + uxv^2*uyv*(-2 + 6*dtuy*tau*uyv + 4*dtuy*tau*uyv^3 - 4*dxux*tau*uT + 3*dyuy*tau*uT + uyv^2*(-2 - 4*dxux*tau*uT + 4*dyuy*tau*uT)) - tau*uxv*(dyux*uT*(3 + 7*uyv^2 + 4*uyv^4) + uyv*(6*dtux + 10*dtux*uyv^2 + 4*dtux*uyv^4 - 3*dxuy*uyv*uT))))/(6*tau*(1 + uxv^2 + uyv^2)^(3/2))
    σyy = (tau*(1 + uxv^2 + uyv^2)*(-(dtnx*uxv) + 2*dtny*uyv + 3*dtny*uxv^2*uyv - dtnx*uxv*uyv^2 + 2*dtny*uyv^3 + 3*dxny*uxv*uyv*uT - dxnx*(1 + uyv^2)*uT + 2*dyny*(1 + uyv^2)*uT) + nux*(uxv^3*(-1 - dtuy*tau*uyv - uyv^2 + 2*dtuy*tau*uyv^3) - 2*tau*uyv*(1 + uyv^2)^2*(dtux*uyv + dyux*uT) - tau*uxv^2*(-1 + 2*uyv^2)*(dtux + dtux*uyv^2 - dxuy*uyv*uT) + uxv*(1 + uyv^2)*(-1 + 2*dtuy*tau*uyv^3 + dxux*tau*uT + uyv^2*(-1 - 2*dxux*tau*uT + 2*dyuy*tau*uT))) + nuy*((-1 + 2*dtux*tau*uxv)*uyv^5 + tau*uxv*(1 + uxv^2)*(dtuy*uxv + dxuy*uT) + 2*tau*uyv^4*(-dtuy - dtuy*uxv^2 + dyux*uxv*uT) - 2*tau*uyv^2*(dtuy + 2*dtuy*uxv^2 + dtuy*uxv^4 + (dxuy - dyux)*uxv*uT + dxuy*uxv^3*uT) + uyv^3*(2*dtux*tau*uxv + 2*dtux*tau*uxv^3 + uxv^2*(-1 + 2*dxux*tau*uT - 2*dyuy*tau*uT) - 2*(1 + dyuy*tau*uT)) - uyv*(1 + dtux*tau*uxv^3 + 2*dyuy*tau*uT + uxv^2*(1 + dxux*tau*uT + 2*dyuy*tau*uT))))/(3*tau*(1 + uxv^2 + uyv^2)^(3/2))
    σeta = (-dxnx - dyny + ((dtux*uxv + dtuy*uyv)*(nux*uxv + nuy*uyv))/(1 + uxv^2 + uyv^2)^(3/2) - (dtux*nux + dtuy*nuy + dtnx*uxv + dtny*uyv)/uT + (2*(nux*uxv + nuy*uyv))/(tau*uT) + nux*(dxux*uxv + dyux*uyv + dtux*uT) + nuy*(dxuy*uxv + dyuy*uyv + dtuy*uT) - ((nux*uxv + nuy*uyv)*(dxux*uxv^2 + uxv*((dxuy + dyux)*uyv + dtux*uT) + uyv*(dyuy*uyv + dtuy*uT)))/(1 + uxv^2 + uyv^2))/3
    return (σxx, σxy, σyy, σeta)
end

"""
    rank1_traceless_2d(Ax, Ay, Bx, By, ux, uy, uτ) -> (qxx, qxy, qyy)

Symmetric, traceless, transverse part of `A^{(i}B^{j)}` for two u-orthogonal
transverse vectors — i.e. `A^{⟨i}B^{j⟩}`.

Both vectors' τ-components follow from u-orthogonality, so lowering an index to
take the trace brings them in: the Δ-trace is

    tr = [A^xB^x(1+u_y²) + A^yB^y(1+u_x²) − (A^xB^y + A^yB^x)u_xu_y] / (u^τ)²

and NOT `A^xB^x + A^yB^y`. Generated symbolically; the hand-written version was
wrong by 2.69x in the φ channel.
"""
@inline function rank1_traceless_2d(Ax::Float64, Ay::Float64,
                                    Bx::Float64, By::Float64,
                                    ux::Float64, uy::Float64, uτ::Float64)
    u2 = uτ*uτ
    tr = (Ay*(By + By*ux^2 - Bx*ux*uy) + Ax*(Bx - By*ux*uy + Bx*uy^2)) / u2
    qxx = Ax*Bx - (1 + ux^2)*tr/3
    qyy = Ay*By - (1 + uy^2)*tr/3
    qxy = (Ay*Bx + Ax*By)/2 - (ux*uy)*tr/3
    return (qxx, qxy, qyy)
end

"""
    consistent_m2_source_2d(...) -> (dpxx, dpxy, dpyy, dPiQ)

∂_τ of the transverse second-moment block and the trace, under the CONSISTENT
closure. `pQeta` is NOT returned: it is redundant and is restored from the
transverse block by `project_shear_traceless_2d` after the update, exactly as
the medium shear does.

Gradients are arguments so the caller owns the discretisation and the cross-code
gate can hand both codes identical data.
"""
@inline function consistent_m2_source_2d(ux::Float64, uy::Float64, uτ::Float64, τ::Float64,
                                         T::Float64, dxT::Float64, dyT::Float64, dtT::Float64,
                                         νx::Float64, νy::Float64,
                                         pxx::Float64, pxy::Float64, pyy::Float64,
                                         peta::Float64, PiQ::Float64,
                                         dtα::Float64, dxα::Float64, dyα::Float64,
                                         θν::Float64, aνx::Float64, aνy::Float64,
                                         dtνx::Float64, dtνy::Float64,
                                         dxνx::Float64, dxνy::Float64,
                                         dyνx::Float64, dyνy::Float64,
                                         θ::Float64, ax::Float64, ay::Float64,
                                         dtux::Float64, dtuy::Float64,
                                         dxux::Float64, dxuy::Float64,
                                         dyux::Float64, dyuy::Float64,
                                         n::Float64, τn::Float64, Ds::Float64,
                                         h::Float64, hp::Float64,
                                         τM::Float64, ηM::Float64, m::Float64)
    Tm = max(T, T_MIN)
    (τM > 0.0) || return (0.0, 0.0, 0.0, 0.0)

    ηQ = ηM
    ζQ = (5.0 / 3.0) * ηQ
    cf = hq_m2_coeffs_2d(Tm, n, τn, Ds, h, m)

    # ── the medium's own shear, at unit viscosity (σ^{μν} itself) ─────────────
    # ns_shear_target_2d returns -2η σ, so η = -1/2 hands back σ directly. One
    # implementation of σ for the medium and the charm sector, not two.
    σxx, σxy, σyy, σeta = ns_shear_target_2d(ux, uy, uτ, τ, θ, ax, ay,
                                             dxux, dxuy, dyux, dyuy, -0.5)

    # ── the current's shear (generated; see sigma_nu_2d) ─────────────────────
    # a·ν = a_μ ν^μ. ⚠ NOT `ax νx + ay νy`: both vectors are u-orthogonal, so
    # lowering the index brings in their τ-components and the contraction picks up
    # −(a·u)(ν·u)/(u^τ)². Derived symbolically alongside π:σ and the (iii)
    # coupling — the same lowering error, in the term that dominates the trace
    # channel's ν dependence.
    aνdot = ax*νx + ay*νy - ((ax*ux + ay*uy)*(νx*ux + νy*uy))/(uτ*uτ)
    sνxx, sνxy, sνyy, _ = sigma_nu_2d(ux, uy, uτ, τ, νx, νy,
                                      dtνx, dxνx, dyνx, dtνy, dxνy, dyνy,
                                      dtux, dxux, dyux, dtuy, dxuy, dyuy)

    # ── scalars shared by both sectors ────────────────────────────────────────
    DT   = uτ*dtT + ux*dxT + uy*dyT
    DlnT = DT / Tm
    DlnC = cf.dlnC_dlnT * DlnT
    Dα   = uτ*dtα + ux*dxα + uy*dyα
    geo  = τM * ((5.0/3.0)*θ + DlnC)

    # π:σ = π_{μν}σ^{μν}, with BOTH indices lowered by the metric.
    # ⚠ NOT `pxx σxx + 2 pxy σxy + pyy σyy + peta σeta`. The stored components are
    # CONTRAVARIANT, so lowering drags in the τ-rows (themselves determined by
    # orthogonality) and the contraction acquires u-dependent weights. The naive
    # form is wrong by O(u²) — it is what left Π_Q 3-5 % off in gate Gm1 after
    # every other term was exact. Generated symbolically from π_{μν}σ^{μν} with
    # the τ-row derived; it reduces to the 1-D `p_l s_l + p_φ s_φ + p_η s_η` in
    # the axisymmetric limit, which Gm1 checks.
    πσ = (2*pxy*σxy + pyy*σyy + 2*pxy*σxy*ux^2 + 2*pyy*σyy*ux^2 + pyy*σyy*ux^4 - 2*pxy*σxx*ux*uy - 2*pyy*σxy*ux*uy - 2*pxy*σyy*ux*uy - 2*pyy*σxy*ux^3*uy - 2*pxy*σyy*ux^3*uy + 2*pxy*σxy*uy^2 + pyy*σxx*ux^2*uy^2 + 4*pxy*σxy*ux^2*uy^2 - 2*pxy*σxx*ux*uy^3 + pxx*σxx*(1 + uy^2)^2 + peta*σeta*(1 + ux^2 + uy^2)^2 + pxx*ux*uy*(σyy*ux*uy - 2*σxy*(1 + uy^2)))/(1 + ux^2 + uy^2)^2

    # ── the (iii) coupling π_Q^{(i}_λ σ^{j)λ}, symmetrised ───────────────────
    # ⚠ The index on π must be LOWERED by the metric before contracting, and the
    # τ-row then contributes: the correction is −(π·u)^i(σ·u)^j/(u^τ)², which the
    # naive transverse-only product omits. Same class of error as π:σ above.
    # Generated symbolically alongside it.
    c_xx = pxx*σxx + pxy*σxy - ((pxx*ux + pxy*uy)*(σxx*ux + σxy*uy))/(1 + ux^2 + uy^2)
    c_yy = pxy*σxy + pyy*σyy - ((pxy*ux + pyy*uy)*(σxy*ux + σyy*uy))/(1 + ux^2 + uy^2)
    c_xy = (pyy*(σxy + σxy*ux^2 - σxx*ux*uy) + pxy*(σxx + σyy + σyy*ux^2 - 2*σxy*ux*uy + σxx*uy^2) + pxx*(σxy - σyy*ux*uy + σxy*uy^2))/(2*(1 + ux^2 + uy^2))

    # ── class (iv): the rank-1 ν⊗a and ν⊗∇(Th) structures ────────────────────
    # 2 λ_a a^{⟨i}ν^{j⟩} + (D_s/T) ν^{⟨i}∇^{j⟩}(Th): the symmetric TRACELESS
    # TRANSVERSE part of each rank-1 product, built by `rank1_traceless_2d`, which
    # lowers with the metric and removes the Δ-trace over all THREE directions.
    # ⚠ Doing this by hand — symmetrising as 2A^iB^i on the diagonal and averaging
    # the trace over the two transverse directions — was wrong by 2.69x in the φ
    # channel (measured by gate Gm1).
    gTx = (h + Tm*hp) * (dxT + ux*DT)          # ∇^{⟨x⟩}(Th)
    gTy = (h + Tm*hp) * (dyT + uy*DT)
    qa_xx, qa_xy, qa_yy = rank1_traceless_2d(ax, ay, νx, νy, ux, uy, uτ)
    qg_xx, qg_xy, qg_yy = rank1_traceless_2d(νx, νy, gTx, gTy, ux, uy, uτ)
    q_xx = 2*cf.λa*qa_xx + (Ds/Tm)*qg_xx
    q_yy = 2*cf.λa*qa_yy + (Ds/Tm)*qg_yy
    q_xy = 2*cf.λa*qa_xy + (Ds/Tm)*qg_xy

    # ── the traceless LHS sources ────────────────────────────────────────────
    sxx = pxx + 2*ηQ*sνxx + geo*pxx +
          2*τM*(c_xx - (1 + ux*ux)*πσ/3) + 2*τM*PiQ*σxx +
          2*cf.ηbar*σxx + q_xx
    syy = pyy + 2*ηQ*sνyy + geo*pyy +
          2*τM*(c_yy - (1 + uy*uy)*πσ/3) + 2*τM*PiQ*σyy +
          2*cf.ηbar*σyy + q_yy
    sxy = pxy + 2*ηQ*sνxy + geo*pxy +
          2*τM*(c_xy - (ux*uy)*πσ/3) + 2*τM*PiQ*σxy +
          2*cf.ηbar*σxy + q_xy

    # ── the trace channel (a scalar: same form as 1-D) ───────────────────────
    sB = PiQ + ζQ*θν + geo*PiQ + (2.0/3.0)*τM*πσ +
         cf.ηbar*(cf.ABr*DlnT + (5.0/3.0)*θ) +
         (Ds*cf.Aco/(3*Tm))*aνdot + (5.0/6.0)*(Ds/Tm)*(νx*gTx + νy*gTy - ((νx*ux + νy*uy)*(gTx*ux + gTy*uy))/(uτ*uτ)) +
         cf.ηbar*Dα

    # ── solve τ_M u^τ ∂_τ X + src = 0 ─────────────────────────────────────────
    den = τM * uτ
    return (-sxx/den, -sxy/den, -syy/den, -sB/den)
end
