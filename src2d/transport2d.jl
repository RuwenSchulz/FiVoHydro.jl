# ==============================================================================
# src2d/transport2d.jl — charge-diffusion transport coefficients.
#
# `τ_n` and the Fluidum single-hadron normalisation are PURE functions of
# (T, α, eos, D_sT, τ_D) — nothing about them is dimension-dependent. They live
# inside `src/dissipation.jl`, which cannot be included here because it also
# defines the 1-D `relax_dissipative!` and drags in `Grid1D`, `Work1D` and
# `IdealDiffViscModel`.
#
# They are therefore COPIED VERBATIM, with `_2d` suffixes so both can coexist in
# one session, and `test_charge2d.jl` includes the 1-D chain and checks the copies
# agree with the originals to round-off across the production (T, α) range. That
# gate is what keeps the duplication honest: if `src/dissipation.jl` changes, the
# copy stops matching and the gate fails.
#
# Source: src/dissipation.jl:77-158 (2026-09-01).
# ==============================================================================

@inline function _fluidum_single_hadron_normalization_2d(T::Float64, α::Float64, eos::LatticeHRGEOS)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    b2 = SpecialFunctions.besselkx(2, z)
    ex = exp(clamp(α - z, -700.0, 700.0))
    return eos.g_hq * (Tm / (2π^2)) * m^2 * ex * b2 * fmGeV3
end

@inline function _diff_tauN_impl_2d(Tm::Float64, α::Float64, eos::LatticeHRGEOS,
                                    DsT::Float64, tauD::Float64)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0.0 && return 0.0

    b1 = SpecialFunctions.besselkx(1, z)
    b2 = SpecialFunctions.besselkx(2, z)
    b3 = b1 + 4.0 / z * b2
    b4 = b2 + 6.0 / z * b3
    b5 = b3 + 8.0 / z * b4
    ex = exp(clamp(α - z, -700.0, 700.0))

    tauq = (DsT / (96.0 * π^2 * Tm^3)) * m^5 * ex * (2.0 * b1 - 3.0 * b3 + b5)
    norm = _fluidum_single_hadron_normalization_2d(Tm, α, eos)
    abs(norm) <= TINY && return 0.0

    τn = tauq / norm * (fmGeV^2)
    if !isfinite(τn)
        return 0.0
    end
    return max(τn, 0.0) * tauD
end

@inline function _diff_tauN_impl_2d(Tm::Float64, α::Float64, eos,
                                    DsT::Float64, tauD::Float64)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0 && return 0.0

    if z > 50.0
        τ_GeVinv = (DsT / 48) * (m^2 / (Tm^2 + 1e-10))
        return min((τ_GeVinv / fmGeV) * tauD, 1e20)
    end

    K1x = SpecialFunctions.besselkx(1, z)
    K2x = SpecialFunctions.besselkx(2, z)
    K3x = SpecialFunctions.besselkx(3, z)
    K5x = SpecialFunctions.besselkx(5, z)

    numerator = 2*K1x - 3*K3x + K5x
    denominator = max(abs(K2x), TINY)
    ratio = numerator / denominator * sign(K2x == 0 ? 1.0 : K2x)

    z3_over_Tm = z^3 / Tm
    if !isfinite(z3_over_Tm) || z3_over_Tm > 1e50
        z3_over_Tm = 1e50
    end

    τ_GeVinv = (DsT / 48) * z3_over_Tm * ratio

    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)
    end

    return (τ_GeVinv / fmGeV) * tauD
end

"""
    diff_coeffs_2d(T, μ, n, model) -> (κ, τ_n, δ_N)

`κ = (D_sT/T)·n/fmGeV` and `τ_n` exactly as `src/dissipation.jl` builds them in
the relaxation block (`κ = safe_div(DsT,T)*max(n,0)/fmGeV`).
"""
@inline function diff_coeffs_2d(T::Float64, μ::Float64, n::Float64,
                                model::IdealDiffVisc2DModel)
    model.enable_diff || return (0.0, 0.0, 0.0)
    Tm = posden(T)
    α  = μ / Tm
    DsT = model.kappa_coeff
    κ  = safe_div(DsT, Tm) * max(n, 0.0) / fmGeV
    τn = _diff_tauN_impl_2d(Tm, α, model.eos, DsT, model.tauN_coeff)
    δN = model.deltaN_factor * τn
    return κ, τn, δN
end

"""
    vacuum_weight_2d(n, model)

Density gate for the charge sector: 0 in the vacuum, 1 in the fluid, linear in
between. Mirrors `main2IS2.jl:_vacuum_weight` (absolute-n branch).
"""
@inline function vacuum_weight_2d(n::Float64, model::IdealDiffVisc2DModel)
    lo = model.vacuum_n_lo; hi = model.vacuum_n_hi
    hi <= lo && return 1.0
    n <= lo && return 0.0
    n >= hi && return 1.0
    return (n - lo)/(hi - lo)
end
