# =============================================================================
# src/hq_consistent_firstmoment.jl — THE THERMODYNAMICALLY CONSISTENT FIRST
# MOMENT for the charm sector (2026-08-25). Companion of the Fluidum twin
# Julia/Fluidum.jl/src/Matrix/HQ_const_BG_consistent.jl; both transcribe the
# xAct derivation Julia/tools/derive_hq_consistent.wls (7/7 gates, abstract-
# index + BasisExpand route — the basis route silently drops the Christoffel
# part of the divergence, gate W0).
#
# WHAT IT ADDS. The shipped first-moment row (build_IS2_system! rows 1-2, the
# Fluidum HQ_const_BG algebra) is the homogeneous-rest-frame reduction of
#
#     (D_s/T) Δ^r_λ ∇_μ T_Q^{μλ} + ν^r = 0 ,
#     T_Q = εQ u u + P0 Δ + h (u ν + ν u),  P0 = nT,  εQ + P0 = n h ,
#
# with h the enthalpy per particle (= τ_n T/D_s by Eq. (30) of 2205.07692 —
# the shipped τ_n already contains it). Removing the homogeneity step adds
# FIVE source terms (verbatim from the .wls, gate W2):
#
#   (D_s/T)(n + T ∂n/∂T) ∇⊥T        pressure gradient — the T channel of the
#                                    full dP0 = nT dα + (n h/T) dT
#   τ_n n a^r                        inertial (enthalpy density × acceleration)
#   τ_n (ν·∇u)^r                     current riding flow gradients
#   τ_n ν (θ_grad + D ln h)          expansion + coefficient transport
#   τ_n ν (u^τ/τ + u^r/r)            geometric dilution (Bjorken: ∂_τ(τν)=0)
#
# All five are SOURCES (background gradients × fields): At/Ax and the CFL
# machinery are untouched; the consistent and shipped systems share their
# characteristic structure exactly (dn_dα = n for the Boltzmann charm sector,
# so the α-drive coefficient is unchanged — Fluidum gate G4, n01 == n).
#
# Every term is exactly ZERO in the homogeneous static flat-space frame the
# shipped derivation was performed in; on an IDEAL baryon-free background the
# first two combine to (τ_n n/T)(∇⊥T + T a) = 0 by Euler, which is the hidden
# justification of the shipped fugacity drive — and why it fails on a viscous
# background.
#
# GATES: Tex/MaxEntHydro/diag_fivo_consistent_gates.jl — the 24 xAct reference
# points of Tex/MaxEntHydro/results/hq_consistent_refpoints.csv (shared with
# the Fluidum gates), the h == m K3/K2 tie, the exact identity
# n + T dn/dT == n h/T, and the cross-code comparison against Fluidum's
# hqc_matrices on the real production background.
#
# FLAG: IS2_CONSISTENT_FM (main2IS2.jl), default OFF = byte-identical
# production. Runner: Tex/MaxEntHydro/run_fivo_consistent.jl.
# =============================================================================

"""
    hq_consistent_h_hp(T, DsT_val, τn_used) -> (h, hp)

Enthalpy per particle h = τn_used·T/D_s in GeV — the h the row's own τ_n
encodes (τ_n = D_s h/T defines h; for the bare `_tauN` chain this equals
m·K₃(z)/K₂(z) exactly, since (z³/48)(2K₁−3K₃+K₅) = z·K₃ by the Bessel
recurrences). hp = dh/dT through the SAME `_tauN` chain by central
difference (DsT cancels exactly between τ_n ∝ DsT and D_s ∝ DsT, so the
value of DsT_val is irrelevant to h), rescaled by the ratio of the used to
the chain τ_n so T-independent knobs (IS2_TAUN_SCALE, degeneracy
conventions) keep D ln h exact. ⚠ The rescale is NOT exact where
τn_used/chain is itself T-dependent — the causality clamp (which never
binds for charm, z·K₃/K₂ > 1) or a tau_diff spline override with its own
T-dependence; there the c′(T)·h₀ piece of dh/dT is dropped (audit A6).
"""
@inline function hq_consistent_h_hp(T::Float64, DsT_val::Float64, τn_used::Float64, eos)
    Tm = max(T, T_MIN)
    m  = hq_mass(eos)
    Ds = DsT_val / Tm / fmGeV
    h  = τn_used * Tm / Ds                      # the exact tie to the row's τ_n

    hchain = function (TT)
        z = m / TT
        K1x, K2x = _bessel_kx_safe(z)
        _tauN(TT, z, DsT_val, K1x, K2x) * TT / (DsT_val / TT / fmGeV)
    end
    δ = max(1e-4, 1e-3 * Tm)
    h0 = hchain(Tm)
    hp = (hchain(Tm + δ) - hchain(Tm - δ)) / (2δ)
    if h0 > 0.0 && isfinite(h0)
        hp *= h / h0                            # keep D ln h exact under τ_n rescalings
    end
    return h, hp
end

"""
    hq_consistent_extras(τ, r, ur, T, dtT, drT, drur, dtur, n, dn_dT, ν, τn, Ds, h, hp)

The five derived source terms, VERBATIM from derive_hq_consistent.wls (and
identical to the Fluidum twin hqc_matrices): pressure-gradient T channel +
inertial (field-independent), ν·∇u + expansion + D ln h (∝ ν), and the
geometric dilution τ_n ν (u^τ/τ + u^r/r). Source-on-the-LHS sign convention,
i.e. ADD the return value to src[2] of build_IS2_system!.
Pass r already floored (r_safe): the u^r/r dilution is finite at the origin
because ν and u^r are both odd.
"""
@inline function hq_consistent_extras(τ::Float64, r::Float64, ur::Float64, T::Float64,
                                      dtT::Float64, drT::Float64, drur::Float64, dtur::Float64,
                                      n::Float64, dn_dT::Float64, ν::Float64,
                                      τn::Float64, Ds::Float64, h::Float64, hp::Float64)
    uτ = sqrt(1.0 + ur^2)
    #   pressure-gradient T channel + inertial   (field-independent sources)
    ex = (Ds / T) * (drur * h * n * ur + dtur * h * n * uτ +
                     dtT * n * ur * uτ + dtT * dn_dT * T * ur * uτ +
                     drT * (n + dn_dT * T) * uτ^2)
    #   nu.grad u + expansion + D ln h           (proportional to nu)
    ex += (Ds / T) * ν * (2.0 * drur * h + drT * hp * ur +
                          (2.0 * dtur * h * ur + dtT * hp * uτ^2) / uτ)
    #   geometric dilution of the current (the W0/WB term)
    ex += τn * ν * (uτ / τ + ur / r)
    return ex
end

"""
    hq_cm_force(τ, r, ur, dtur, drur, piQr, piQperp, PiQ, dtζ, drζ)

The COMPLETE c_M coupling row at unit c_M, abstract-index route
(Julia/tools/derive_hq_cm_coupling.wls 3/3; Fluidum twin `hqc_cm_row2`, gated
G5 to 1e-10 on results/hq_cm_coupling_refpoints.csv):

    ur·uτ·∂τζ + uτ²·∂rζ + (drur·ur + dtur·uτ)·ζ
    + uτ²·(πQr − πQperp)/r + ur·uτ·(2πQr + πQperp)/τ ,     ζ = πQr + Π_Q.

The hoop-stress and τ-redshift terms (the last two) are the geometric pieces
the notebook/legacy basis-route coupling drops — `build_IS2_system!`'s own cM
entries carry only the first three, so the consistent 5-field zeroes those and
applies this force as an explicit source instead (source-on-the-LHS: multiply
by c_M and any regulator, ADD to src[2]). ∂τζ is supplied one RK stage lagged
by the caller (FiVo's operator split computes dπ/dτ after dν/dτ; the lag is an
O(dt) error on this one subdominant O(ur) term — documented at the wiring in
`_compute_dUdt!`). Pass r already floored.
"""
@inline function hq_cm_force(τ::Float64, r::Float64, ur::Float64, dtur::Float64, drur::Float64,
                             piQr::Float64, piQperp::Float64, PiQ::Float64,
                             dtζ::Float64, drζ::Float64)
    uτ = sqrt(1.0 + ur^2)
    ζ = piQr + PiQ
    return ur * uτ * dtζ + uτ^2 * drζ + (drur * ur + dtur * uτ) * ζ +
           uτ^2 * (piQr - piQperp) / r + ur * uτ * (2.0 * piQr + piQperp) / τ
end
