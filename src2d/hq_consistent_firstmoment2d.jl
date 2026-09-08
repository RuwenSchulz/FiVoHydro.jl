
# =============================================================================
# src2d/hq_consistent_firstmoment2d.jl — THE THERMODYNAMICALLY CONSISTENT FIRST
# MOMENT in 2+1D (2026-09-08). Transverse-Cartesian counterpart of
# src/hq_consistent_firstmoment.jl, and of the same xAct derivation
# Julia/tools/derive_hq_consistent.wls.
#
# -----------------------------------------------------------------------------
# WHAT IT ADDS
# -----------------------------------------------------------------------------
# The shipped charge row relaxes ν^i toward the Navier-Stokes target
# ν_NS^i = -κ ∇^{⟨i⟩}α, i.e. its ONLY drive is the fugacity gradient. That is the
# homogeneous-rest-frame reduction of
#
#     (D_s/T) Δ^i_λ ∇_μ T_Q^{μλ} + ν^i = 0 ,
#     T_Q = ε_Q u u + P_0 Δ + h (u ν + ν u),   P_0 = nT,   ε_Q + P_0 = n h ,
#
# with h the enthalpy per particle. Removing the homogeneity step adds the same
# sources the 1-D solver carries, written here with no assumption of axisymmetry.
# (The wls NAMES five; they assemble into four additive pieces, because the
# geometric dilution is part of the full divergence θ — see below.)
#
# -----------------------------------------------------------------------------
# THE DECOMPOSITION, AND HOW THIS FILE WAS DERIVED
# -----------------------------------------------------------------------------
# `derive_hq_consistent.wls` names the five terms (its `extras`, gate W2):
#
#     extras^i = (D_s/T)(n + T ∂n/∂T) gradTperp^i        pressure gradient, T channel
#              + τ_n n acc^i                              inertial
#              + τ_n nugradu^i                            ν riding flow gradients
#              + ν^i (τ_n θ + (D_s/T) ∂h/∂T · DT)         expansion + coefficient transport
#                                                          (θ is the FULL divergence:
#                                                           the geometric dilution,
#                                                           the wls's fifth named
#                                                           term, lives inside it)
#
# with, covariantly,
#
#     acc^i       = Δ^i_λ u^m ∇_m u^λ                  (= a^i, the four-acceleration)
#     nugradu^i   = Δ^i_λ ν^m ∇_m u^λ
#     gradTperp^i = Δ^{iq} ∇_q T = ∂_i T + u^i DT
#     DT          = u^m ∂_m T
#     θ           = ∇_μ u^μ
#
# ⚠ THE 1-D FUNCTION IS NOT A TEMPLATE YOU CAN RETYPE WITH x FOR r. Its printed
# form is the above CONTRACTED in radial Milne, where several distinct covariant
# objects collapse onto one another. Two collapses in particular:
#
#   * θ_flat and nugradu/ν are NUMERICALLY EQUAL in 1-D and are NOT equal here.
#     With one transverse direction, ν^m∇_m u^r/ν = ∂_r u^r + (u^r/u^τ)∂_τ u^r,
#     which is exactly θ_flat = ∂_τu^τ + ∂_r u^r. In 2-D, ν^m∇_m u^i is a genuine
#     contraction over BOTH transverse directions and picks up ∂_y u^x etc., so
#     the two terms must be built separately. Retyping the 1-D grouping would
#     silently double one of them and drop the cross-derivatives.
#
#   * the GEOMETRIC TERMS APPEAR ONCE, INSIDE θ. The wls lists "expansion" and
#     "geometric dilution" as two of its five named terms, and it is tempting to
#     read that as two additive contributions. It is not: the ν-proportional
#     coefficient of the assembled row is exactly
#
#         τ_n θ + (D_s/T) h' DT ,     θ = ∇_μ u^μ the FULL divergence,
#
#     and θ already carries every Christoffel piece (u^τ/τ + u^r/r in radial
#     coordinates, u^τ/τ here — the 1-D u^r/r being part of ∂_ju^j in Cartesian
#     coordinates, the same σ^{yy} ↔ σ^φ_φ correspondence dissipation2d.jl
#     records for the shear). The wls's fifth term is what the DEFECTIVE
#     basis-route row omitted (its gate W0), not a second copy to be added to a
#     θ that already has it.
#
#     ⚠ THIS FILE GOT IT WRONG FIRST TIME. The initial version added θ_full AND a
#     separate τ_n ν^i u^τ/τ, double-counting the dilution. Against the 1-D
#     reference that is a 2.5-25 % error, largest where u^τ/τ dominates the other
#     gradients — a plausible-looking, smooth, entirely wrong source. Gate Gc1
#     caught it; nothing else would have.
#
# `test_consistent_fm2d.jl` is that gate: it evaluates THIS code in the u^y = 0,
# ∂_y = 0 limit against the 1-D `hq_consistent_extras` and requires agreement to
# round-off, exactly as `test_charge2d.jl` does for the copied transport
# coefficients.
#
# -----------------------------------------------------------------------------
# WHAT IS NOT HERE
# -----------------------------------------------------------------------------
# The c_M second-moment back-coupling (`hq_cm_force` in the 1-D file) has NO 2-D
# counterpart, because the 2-D solver has no second-moment sector: there is no
# π_Q, no Π_Q, nothing to couple back. The consistent projection corrects the
# FIRST moment only, which is the whole of what the 2-D charge sector carries.
# In O+O production the back-coupling is off (`use_cM = 0`) in any case.
#
# FLAG: `IdealDiffVisc2DModel.consistent_fm`, default `false` = the shipped
# ∇α-only drive, byte-identical to every 2-D number produced before this file.
# =============================================================================

"""
    hq_h_hprime_2d(T, eos) -> (h, hp)

Enthalpy per particle `h = m K₃(z)/K₂(z)` in GeV and `hp = dh/dT`, for the
Boltzmann charm sector (`z = m/T`).

This is the SAME `h` the 1-D chain encodes, reached by a different route. The
1-D `hq_consistent_h_hp` DEFINES h through the row's own relaxation time,
`h = τ_n T/D_s`, so that a rescaled or clamped τ_n stays exactly consistent with
the h that multiplies it; it then differentiates the bare chain numerically. Here
the closed form is used directly, because

  * `τ_n = D_s z K₃/K₂ / T` in `_diff_tauN_impl_2d` already IS the bare chain, so
    `τ_n T/D_s = m K₃/K₂` identically — the two definitions coincide. MEASURED
    against the solver's own `diff_coeffs_2d` at `D_sT = 0.1163`, `α = -4.2`:
    `τ_n T/D_s` = 1.76170, 1.91887, 2.04388, 2.34277, 2.81603, 3.31253 GeV at
    T = 0.10, 0.1565, 0.20, 0.30, 0.45, 0.60 GeV, equal to `h` in every digit
    printed. And
  * there is no causality clamp on the 2-D charge sector, so no T-dependent
    factor for the tie to protect against.

⚠ `tauN_coeff ≠ 1` BREAKS THE TIE, and is refused rather than run. It multiplies
τ_n (`transport2d.jl:138`) but not this `h`, so `τ_n T/D_s` would no longer equal
`h` and the row's own relaxation time and the enthalpy multiplying its sources
would sit on different clocks — silently, and only in the sources. The 1-D
`hq_consistent_h_hp` survives such a rescale precisely because it DEFINES h from
the used τ_n. `reject_unwired_knobs_2d` enforces this (it is a consistent_fm-only
restriction: with the closure off, `tauN_coeff` is a legitimate diagnostic dial).

`hp` is the analytic derivative of `m K₃/K₂`:

    K₃'(z) = -(K₂ + K₄)/2 ,   K₂'(z) = -(K₁ + K₃)/2 ,   dz/dT = -z/T

so  dh/dT = -m (z/T) d/dz(K₃/K₂)
          = (m z /T) · [ (K₂+K₄)K₂ - (K₁+K₃)K₃ ] / (2 K₂²) .

Using the scaled `besselkx` (all Kₙ carry the same e^{+z} factor, which cancels
in every ratio here) keeps this finite at large z.
"""
@inline function hq_h_hprime_2d(T::Float64, eos)
    Tm = max(T, T_MIN)
    m  = hq_mass(eos)
    z  = m / Tm
    z <= 0.0 && return (0.0, 0.0)

    K1 = SpecialFunctions.besselkx(1, z)
    K2 = SpecialFunctions.besselkx(2, z)
    K3 = SpecialFunctions.besselkx(3, z)
    K4 = SpecialFunctions.besselkx(4, z)
    abs(K2) <= TINY && return (0.0, 0.0)

    h  = m * K3 / K2
    hp = (m * z / Tm) * ((K2 + K4) * K2 - (K1 + K3) * K3) / (2.0 * K2 * K2)
    return (isfinite(h) ? h : 0.0, isfinite(hp) ? hp : 0.0)
end

"""
    hq_dn_dT_2d(T, n, eos) -> ∂n/∂T at fixed α

`n ∝ T³ z² K₂(z) e^α` for the Boltzmann list, so

    ∂n/∂T |_α = (n/T)(3 + z K₁/K₂) ,

the same closed form `main2IS2.jl::transport_all` uses (`dn_dT`). Returned
rather than differenced so the pressure-gradient channel carries no step noise.
"""
@inline function hq_dn_dT_2d(T::Float64, n::Float64, eos)
    Tm = max(T, T_MIN)
    m  = hq_mass(eos)
    z  = m / Tm
    z <= 0.0 && return 0.0
    K1 = SpecialFunctions.besselkx(1, z)
    K2 = SpecialFunctions.besselkx(2, z)
    abs(K2) <= TINY && return 0.0
    v = (n / Tm) * (3.0 + z * K1 / K2)
    return isfinite(v) ? v : 0.0
end

"""
    consistent_fm_source_2d(ux, uy, uτ, T, dxT, dyT, dtT,
                            nux, nuy, n, dn_dT, τn, Ds, h, hp,
                            θ, ax, ay, dxux, dxuy, dyux, dyuy) -> (sx, sy)

The consistent first-moment sources as a transverse vector `(s^x, s^y)`, in the
same sign convention as `ns_diffusion_target_2d`: ADD to the numerator of the
ν relaxation update, beside `ν_NS`.

Four additive pieces, not five: the wls's "geometric dilution" is not a separate
contribution here, it is inside the full divergence θ (see the header).

`θ`, `a^i` and the velocity gradients come straight from `kinematics_2d`, so the
sources are built from exactly the kinematics the shear sector already uses —
one definition of θ and a^i in the file, not two.

`ν^τ` follows from orthogonality `u·ν = 0` (`nu_tau_2d`), as it does in the
projected-derivative correction a few lines below the call site.
"""
@inline function consistent_fm_source_2d(ux::Float64, uy::Float64, uτ::Float64,
                                         T::Float64, dxT::Float64, dyT::Float64, dtT::Float64,
                                         nux::Float64, nuy::Float64,
                                         n::Float64, dn_dT::Float64,
                                         τn::Float64, Ds::Float64, h::Float64, hp::Float64,
                                         θ::Float64, ax::Float64, ay::Float64,
                                         dxux::Float64, dxuy::Float64,
                                         dyux::Float64, dyuy::Float64)
    Tm = max(T, T_MIN)

    # D T = u^m ∂_m T
    DT = uτ*dtT + ux*dxT + uy*dyT

    # gradTperp^i = Δ^{iq}∇_q T = ∂_i T + u^i DT
    gTx = dxT + ux*DT
    gTy = dyT + uy*DT

    # ν^τ from u·ν = 0
    nut = nu_tau_2d(ux, uy, uτ, nux, nuy)

    # nugradu^i = Δ^i_λ ν^m ∇_m u^λ.
    # In Milne with u^η = 0 and ∂_η = 0, ∇_m u^i = ∂_m u^i for transverse i, and
    # the ∂_τ u^i piece is carried through the acceleration identity below rather
    # than differenced again: ν^m∂_m u^i = ν^τ∂_τu^i + ν^x∂_xu^i + ν^y∂_yu^i, and
    # ∂_τ u^i = (a^i - u^x∂_x u^i - u^y∂_y u^i)/u^τ from a^i = u^m∂_m u^i.
    dtux = (ax - ux*dxux - uy*dyux) * safe_inv(uτ)
    dtuy = (ay - ux*dxuy - uy*dyuy) * safe_inv(uτ)
    ngux = nut*dtux + nux*dxux + nuy*dyux
    nguy = nut*dtuy + nux*dxuy + nuy*dyuy
    # NO further projection. The wls writes this term with Δ^k_l in front, but for
    # THIS vector the projector is the identity: with V^λ = ν^m∇_m u^λ,
    #     Δ^i_λ V^λ = V^i + u^i (u_λ V^λ) ,   u_λ V^λ = ½ ν^m ∇_m (u·u) = 0 ,
    # so the correction vanishes identically (checked numerically: 0.0e+00, not
    # merely small). An earlier version here subtracted u^i V^τ instead of u^i
    # times that vanishing bracket — not the projector, and a real 0.4-1.6 %
    # error against the 1-D reference. Gate Gc1 caught that too.

    # ν^i (τ_n θ + (D_s/T) h' DT): expansion + coefficient transport, AND the
    # geometric dilution. These are ONE term, not two — see this file's header.
    # θ is the FULL divergence ∇_μu^μ, so it already carries every geometric
    # piece (u^τ/τ here; u^τ/τ + u^r/r in radial coordinates). Adding a separate
    # dilution on top double-counts it: measured, that is a 12-25 % error against
    # the 1-D reference, which is what gate Gc1 caught.
    br = τn*θ + (Ds/Tm)*hp*DT

    pref = (Ds/Tm) * (n + Tm*dn_dT)
    sx = pref*gTx + τn*n*ax + τn*ngux + nux*br
    sy = pref*gTy + τn*n*ay + τn*nguy + nuy*br
    return sx, sy
end
