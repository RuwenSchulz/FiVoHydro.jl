
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
# 🔴 INCOMPLETE — DO NOT ENABLE. WHAT IS DONE AND WHAT IS NOT (2026-09-08)
# -----------------------------------------------------------------------------
# The flag exists and the file loads, but `sigma_nu_2d` below is WRONG and gate
# Gm1 (test_consistent_m22d.jl) fails on it. Nothing may be run with
# consistent_m2 = true until that is closed.
#
# ESTABLISHED, and gated:
#   * the GEOMETRY and algebraic structure — derive_hq_m2_2p1d.wls, 9/9: the
#     tau-row is determined by orthogonality, tracelessness fixes pieta (and the
#     .wls confirms that formula is algebraically identical to
#     `project_shear_traceless_2d`'s), Delta-projected D pi is quasi-linear.
#   * the FIELD COUNT: 3 independent dofs stored as 4, exactly the medium
#     shear's convention, so `shear_tensor_contravariant_2d` and
#     `project_shear_traceless_2d` are reused rather than restated.
#   * the BACKGROUND sector is EXACT against the 1-D reduction: with the fields
#     and nu zero, every gradient combination reproduces
#     `hq_consistent_m2_rhs` to 3e-15, at u^r up to -2.5. So theta, sigma, the
#     eta-bar term, D ln C, D ln T and the geometric pieces are all right.
#   * the FIELD-PROPORTIONAL terms are nearly exact: Pi_Q to 1e-15, and the
#     pi:sigma couplings to 1.4e-3 / 5.8e-4.
#
# 🔴 NOT DONE — `sigma_nu_2d`:
#   With nu != 0 and every other input zero, gate Gm1 measures 65 % on p_l and
#   9 % on p_phi. The cause is identified: this file builds sigma_(nu) BY ANALOGY
#   with `ns_shear_target_2d` (the shear of u), and that analogy is false. The
#   1-D file carries
#       (grad nu)_ll = D_l nu^r / u^tau - u^r nu (D_l u^r)/u^tau^3
#   whose SECOND term is the connection piece from parallel-transporting l — it
#   involves gradients of U, not of nu, and has no counterpart in the u-shear.
#   The exact Cartesian sigma_(nu)^{ij} was derived symbolically (the scratch
#   .wls of this session) and is a large expression carrying u-gradient terms
#   this implementation omits entirely.
#
# THE FIX is to generate sigma_(nu)^{ij} from the symbolic derivation into Julia
# rather than hand-writing it, exactly as HQ_2p1d_BG_generated.jl is generated
# from derive_hq_2p1d.wls. That is the next step and it is mechanical; what is
# NOT safe is to keep patching the analogy by hand.
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
    sigma_nu_2d(νx, νy, ντ, ux, uy, uτ, τ, θν, aνx, aνy,
                dxνx, dxνy, dyνx, dyνy) -> (σxx, σxy, σyy, σeta)

The shear of the CURRENT, `σ_(ν)^{μν}`: the traceless transverse-projected
symmetric gradient of ν, built exactly as `ns_shear_target_2d` builds `σ^{μν}`
from u — with u → ν throughout, so the "acceleration" slot takes `a_ν^i = Dν^i`
(a VECTOR, one component per transverse direction) and the expansion slot takes
`θ_ν − a·ν`.

⚠ `a_ν^i` is per-component. An earlier version of this file passed the SCALAR
`a·ν` into both slots, which is dimensionally fine and physically wrong: it makes
σ_(ν) blind to the direction of Dν and destroys x↔y covariance. Gate Gm2b (the
rotation test) is what catches that class of error; the 1-D limit cannot, because
it has one transverse direction.

The `−a·ν` in the trace is not cosmetic either: the identity
`Σ_i (∇ν)_ii = θ_ν − a·ν` is what makes the trace removed here the same one the
1-D file removes (its `Strν`).
"""
@inline function sigma_nu_2d(νx::Float64, νy::Float64,
                             ux::Float64, uy::Float64, uτ::Float64, τ::Float64,
                             θν::Float64, aνdot::Float64,
                             aνx::Float64, aνy::Float64,
                             dxνx::Float64, dxνy::Float64,
                             dyνx::Float64, dyνy::Float64)
    third = (θν - aνdot) / 3
    σxx = dxνx + ux*aνx - third*(1 + ux*ux)
    σyy = dyνy + uy*aνy - third*(1 + uy*uy)
    σxy = 0.5*(dxνy + dyνx) + 0.5*(ux*aνy + uy*aνx) - third*(ux*uy)
    σeta = -third                      # ν^η = 0, so only the trace part survives
    return (σxx, σxy, σyy, σeta)
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
                                         dxνx::Float64, dxνy::Float64,
                                         dyνx::Float64, dyνy::Float64,
                                         θ::Float64, ax::Float64, ay::Float64,
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

    # ── the current's shear ───────────────────────────────────────────────────
    aνdot = ax*νx + ay*νy                      # a·ν (ν^η = 0, transverse metric)
    sνxx, sνxy, sνyy, _ = sigma_nu_2d(νx, νy, ux, uy, uτ, τ, θν, aνdot,
                                      aνx, aνy, dxνx, dxνy, dyνx, dyνy)

    # ── scalars shared by both sectors ────────────────────────────────────────
    DT   = uτ*dtT + ux*dxT + uy*dyT
    DlnT = DT / Tm
    DlnC = cf.dlnC_dlnT * DlnT
    Dα   = uτ*dtα + ux*dxα + uy*dyα
    geo  = τM * ((5.0/3.0)*θ + DlnC)

    # π:σ — the FULL contraction, η channel included. That channel is where the
    # 1-D p_η lives; dropping it silently loses a third of the coupling.
    πσ = pxx*σxx + 2*pxy*σxy + pyy*σyy + peta*σeta

    # ── the (iii) coupling π_Q^{⟨μ}_λ σ^{ν⟩λ} on the transverse block ────────
    # Raised-index contraction over the transverse directions; the τ and η rows
    # contribute nothing to the transverse block (π^{iη} = σ^{iη} = 0, and the
    # τ-row is not independent). Symmetrised, with its trace removed below.
    c_xx = pxx*σxx + pxy*σxy
    c_yy = pxy*σxy + pyy*σyy
    c_xy = 0.5*((pxx*σxy + pxy*σyy) + (pxy*σxx + pyy*σxy))

    # ── class (iv): the rank-1 ν⊗a and ν⊗∇(Th) structures, symmetrised ───────
    gTx = (h + Tm*hp) * (dxT + ux*DT)          # ∇^{⟨x⟩}(Th)
    gTy = (h + Tm*hp) * (dyT + uy*DT)
    q_xx = 2*cf.λa*(2*ax*νx)      + (Ds/Tm)*(2*νx*gTx)
    q_yy = 2*cf.λa*(2*ay*νy)      + (Ds/Tm)*(2*νy*gTy)
    q_xy = 2*cf.λa*(ax*νy + ay*νx) + (Ds/Tm)*(νx*gTy + νy*gTx)
    # its transverse trace, removed so the block stays traceless
    q_tr = (q_xx*(1) + q_yy*(1)) / 3

    # ── the traceless LHS sources ────────────────────────────────────────────
    sxx = pxx + 2*ηQ*sνxx + geo*pxx +
          2*τM*(c_xx - πσ/3) + 2*τM*PiQ*σxx +
          2*cf.ηbar*σxx + (q_xx - q_tr)/2
    syy = pyy + 2*ηQ*sνyy + geo*pyy +
          2*τM*(c_yy - πσ/3) + 2*τM*PiQ*σyy +
          2*cf.ηbar*σyy + (q_yy - q_tr)/2
    sxy = pxy + 2*ηQ*sνxy + geo*pxy +
          2*τM*c_xy + 2*τM*PiQ*σxy +
          2*cf.ηbar*σxy + q_xy/2

    # ── the trace channel (a scalar: same form as 1-D) ───────────────────────
    sB = PiQ + ζQ*θν + geo*PiQ + (2.0/3.0)*τM*πσ +
         cf.ηbar*(cf.ABr*DlnT + (5.0/3.0)*θ) +
         (Ds*cf.Aco/(3*Tm))*aνdot + (5.0/6.0)*(Ds/Tm)*(νx*gTx + νy*gTy) +
         cf.ηbar*Dα

    # ── solve τ_M u^τ ∂_τ X + src = 0 ─────────────────────────────────────────
    den = τM * uτ
    return (-sxx/den, -sxy/den, -syy/den, -sB/den)
end
