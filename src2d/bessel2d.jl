# ==============================================================================
# src2d/bessel2d.jl — a fast scaled Bessel K₂ for the 2+1D equation of state.
#
# WHY. The charm density of `LatticeHRGEOS` is n ∝ T³ z² K₂(z) e^α with z = m/T,
# and SpecialFunctions evaluates `besselkx(2, z)` for a REAL z through the complex
# Amos routine: 163 ns a call. The primitive recovery evaluates the EOS inside a
# Newton solve on every cell three times per step, and profiled on the production
# IC those Bessel calls were ~50 % of the whole solver after the bit-identical
# EOS memo in primrec2d.jl had removed the repeated ones.
#
# WHAT. K₂ from the upward recurrence K₂ = K₀ + (2/x) K₁ (stable upward for K),
# with K₀ₓ = eˣK₀ and K₁ₓ = eˣK₁ from the rational approximations of Bessels.jl
# v0.2.8 (`besselk0x`, `besselk1x`; coefficients from Boost, after Holoborodko),
# vendored below rather than added as a dependency because the 2-D scripts run
# under two different project environments. 8.6 ns a call (19× faster).
#
# WHAT IT COSTS. Agreement with `SpecialFunctions.besselkx(2, x)`: max relative
# difference 1.6e-15 over x ∈ [0.15, 1e4] (200 000 log-spaced points; bit-identical
# at 36 % of them). On a full solve the conserved fields move by ≤ 4e-14 absolute
# (~2e-15 relative) — round-off, ten orders below anything the gates resolve. The
# 2-D solver is therefore NOT bit-identical to its pre-2026-09-11 self in the charm
# density; pressure and energy are (they involve no Bessel function).
# Gate: test_primrec2d.jl (the accuracy sweep and the EOS comparison).
#
# SCOPE. 2-D only. `src/eos.jl` — shared with the 1-D production solver — still
# calls SpecialFunctions, so every 1-D number stays byte-identical. For
# x ≥ BESSELKX_ASYM_X (T below 0.15 MeV: vacuum and floors) this uses the SAME
# asymptotic series as `safe_besselkx`, so that regime is bit-identical too.
#
# Bessels.jl is © 2021-2022 Michael Helton, Oscar Smith and contributors, MIT
# License (https://github.com/JuliaMath/Bessels.jl). The coefficient tables and the
# two evaluation formulas below are copied from its src/constants.jl and
# src/besselk.jl (v0.2.8), Float64 only.
# ==============================================================================

const _K0_P1 = (-1.372509002685546267e-1, 2.574916117833312855e-1,
                1.395474602146869316e-2, 5.445476986653926759e-4,
                7.125159422136622118e-6)
const _K0_Q1 = (1.000000000000000000e+00, -5.458333438017788530e-02,
                1.291052816975251298e-03, -1.367653946978586591e-05)
const _K0_P2 = (1.159315156584124484e-01, 2.789828789146031732e-01,
                2.524892993216121934e-02, 8.460350907213637784e-04,
                1.491471924309617534e-05, 1.627106892422088488e-07,
                1.208266102392756055e-09, 6.611686391749704310e-12)
const _K0_P3 = (2.533141373155002416e-1, 3.628342133984595192e0,
                1.868441889406606057e1, 4.306243981063412784e1,
                4.424116209627428189e1, 1.562095339356220468e1,
                -1.810138978229410898e0, -1.414237994269995877e0,
                -9.369168119754924625e-2)
const _K0_Q3 = (1.000000000000000000e0, 1.494194694879908328e1,
                8.265296455388554217e1, 2.162779506621866970e2,
                2.845145155184222157e2, 1.851714491916334995e2,
                5.486540717439723515e1, 6.118075837628957015e0,
                1.586261269326235053e-1)
const _K0_Y  = 1.137250900268554688

const _K1_Y  = 8.69547128677368164e-2
const _K1_Y2 = 1.45034217834472656
const _K1_P1 = (-3.62137953440350228e-3, 7.11842087490330300e-3,
                1.00302560256614306e-5, 1.77231085381040811e-6)
const _K1_Q1 = (1.00000000000000000e0, -4.80414794429043831e-2,
                9.85972641934416525e-4, -8.91196859397070326e-6)
const _K1_P2 = (-3.07965757829206184e-1, -7.80929703673074907e-02,
                -2.70619343754051620e-3, -2.49549522229072008e-5)
const _K1_Q2 = (1.00000000000000000e0, -2.36316836412163098e-2,
                2.64524577525962719e-4, -1.49749618004162787e-6)
const _K1_P3 = (-1.97028041029226295e-1, -2.32408961548087617e0,
                -7.98269784507699938e0, -2.39968410774221632e0,
                3.28314043780858713e1, 5.67713761158496058e1,
                3.30907788466509823e1, 6.62582288933739787e0,
                3.08851840645286691e-1)
const _K1_Q3 = (1.00000000000000000e0, 1.41811409298826118e1,
                7.35979466317556420e1, 1.77821793937080859e2,
                2.11014501598705982e2, 1.19425262951064454e2,
                2.88448064302447607e1, 2.27912927104139732e0,
                2.50358186953478678e-2)

"`eˣ K₀(x)` for `x > 0` (Bessels.jl `besselk0x`)."
@inline function besselk0x_2d(x::Float64)
    if x <= 1.0
        a = x * x / 4
        s = muladd(evalpoly(a, _K0_P1), inv(evalpoly(a, _K0_Q1)), _K0_Y)
        a = muladd(s, a, 1)
        return muladd(-a, log(x), evalpoly(x * x, _K0_P2)) * exp(x)
    else
        return muladd(evalpoly(inv(x), _K0_P3), inv(evalpoly(inv(x), _K0_Q3)), 1.0) / sqrt(x)
    end
end

"`eˣ K₁(x)` for `x > 0` (Bessels.jl `besselk1x`)."
@inline function besselk1x_2d(x::Float64)
    if x <= 1.0
        z = x * x
        a = z / 4
        pq = muladd(evalpoly(a, _K1_P1), inv(evalpoly(a, _K1_Q1)), _K1_Y)
        pq = muladd(pq * a, a, (a / 2 + 1))
        a = pq * x / 2
        pq = muladd(evalpoly(z, _K1_P2) / evalpoly(z, _K1_Q2), x, inv(x))
        return muladd(a, log(x), pq) * exp(x)
    else
        return muladd(evalpoly(inv(x), _K1_P3), inv(evalpoly(inv(x), _K1_Q3)), _K1_Y2) / sqrt(x)
    end
end

"""
    safe_besselk2x_2d(x) -> eˣ K₂(x)

The 2-D solver's `safe_besselkx(2, x)`: same guards, same asymptotic branch for
`x ≥ BESSELKX_ASYM_X`, and the fast recurrence `K₀ₓ + (2/x)K₁ₓ` below it.
"""
@inline function safe_besselk2x_2d(x::Float64)
    (isfinite(x) && x > 0.0) || return 0.0
    x < BESSELKX_ASYM_X && return besselk0x_2d(x) + (2/x)*besselk1x_2d(x)
    return safe_besselkx(2, x)          # the shared asymptotic series, bit-identical
end
