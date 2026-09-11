# ==============================================================================
# test/test_bjorken1d.jl — gates A1/A2: Bjorken flow, 1+1D bulk solver.
#
# A1  IDEAL Bjorken
#     a  conformal, charge-free: T τ^{1/3} = const exactly; the time order of both
#        integrators (SSPRK2 → 2, SSPRK3 → 3). 🔴 SSPRK3 evaluated its last stage at
#        τ+Δ instead of τ+Δ/2 until 2026-09-11, which made it FIRST order on every
#        explicitly τ-dependent source; this is the regression test.
#     b  with a conserved charge (LatticeHRGEOS): n τ conserved to round-off, and
#        (T, α)(τ) against the independent (e, n) ODE (`bjorken_background`).
# A2  VISCOUS Bjorken, the full 0+1D DNMR set (`bjorken_dnmr`), term by term:
#     shear alone, + δ_ππ, + τ_ππ, bulk alone, + δ_ΠΠ, and the shear–bulk couplings
#     λ_πΠ, λ_Ππ. Each configuration at three step sizes: the error must fall FIRST
#     order (the relaxation is operator-split), and its Richardson extrapolation
#     2q(Δ/2) − q(Δ) must sit on the referee (< 1e-3).
#     🔴 λ_πΠ and λ_Ππ had DNMR's mostly-minus sign and τ_ππ lacked the −π:σ/3 trace
#     until 2026-09-11 (default 0, no caller set them); the "sign" rows below fail
#     on the old code — part (c) records by how much a WRONG-sign referee misses.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_bjorken1d.jl
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(@__DIR__, "analytic_referees.jl"))
using .hydro
const H = hydro

const NR = 40
const RMAX = 10.0

"""Mean T, and the conserved quantities, over the interior at the end of a uniform run."""
function run_uniform(m; T0, α0, τ0, τ1, CFLτ, integrator = :ssprk2)
    g = H.make_grid_1d(NR; rmax = RMAX)
    U = H.allocate_state(g, m)
    H.initialize_uniform!(U, g, m, τ0; T0, alpha0 = α0)
    # CFL huge: the Bjorken clock CFLτ is the only step limit (a uniform state has no
    # transverse signal to resolve)
    res = H.run_sim_1d!(U, g, m; τ0, τfinal = τ1, CFL = 10.0, CFLτ, integrator)
    f = H.fields_1d(g, U, m; τ = res.τ, work = res.work)
    k = length(f.r) ÷ 2
    return (; res, T = f.T[k], α = f.alpha[k], e = f.e[k], P = f.P[k], n = f.n[k],
              φ = -f.piEta[k], Π = f.Pi[k], nonunif = (maximum(f.T) - minimum(f.T))/f.T[k])
end

# ------------------------------------------------------------------------------
function gate_A1()
    println("\nA1 — ideal Bjorken")
    # a: conformal, charge-free
    m = H.build_model_1d(; eos = H.ConformalHQEOS(m_hq = 0.0, g_hq = 0.0))
    Tex = 0.4*(1/5)^(1/3)
    for (integ, pmin) in ((:ssprk2, 1.9), (:ssprk3, 2.8))
        errs = [run_uniform(m; T0 = 0.4, α0 = 0.0, τ0 = 1.0, τ1 = 5.0, CFLτ = c, integrator = integ).T/Tex - 1
                for c in (0.04, 0.02, 0.01)]
        p = log2(abs(errs[2]/errs[3]))
        @printf("  a  %-6s rel. error in T at τ=5: %+.2e %+.2e %+.2e   order %.2f\n", integ, errs..., p)
        @test p > pmin
        @test abs(errs[3]) < (integ === :ssprk2 ? 1e-5 : 1e-7)
    end
    # b: with a conserved charge
    eos = H.LatticeHRGEOS()
    mc = H.build_model_1d(; eos)
    τs, Tref, αref = bjorken_background((T, μ) -> H.eos_Pne(T, μ, eos), 0.35, -1.0, 0.6, 3.0)
    r = run_uniform(mc; T0 = 0.35, α0 = -1.0, τ0 = 0.6, τ1 = 3.0, CFLτ = 0.005, integrator = :ssprk3)
    _, n0, _ = H.eos_Pne(0.35, -0.35, eos)
    dnτ = r.n*3.0/(n0*0.6) - 1
    @printf("  b  LatticeHRGEOS, α0 = −1: T %.6f vs %.6f (%.1e), α %.6f vs %.6f (%.1e), nτ drift %.1e, non-uniformity %.1e\n",
            r.T, Tref(3.0), r.T/Tref(3.0) - 1, r.α, αref(3.0), r.α - αref(3.0), dnτ, r.nonunif)
    @test abs(r.T/Tref(3.0) - 1) < 1e-6
    @test abs(r.α - αref(3.0)) < 1e-6
    @test abs(dnτ) < 1e-12
    @test r.nonunif < 1e-12
    return nothing
end

# ------------------------------------------------------------------------------
"""One viscous configuration against the DNMR referee at three CFLτ; returns the errors."""
function viscous_case(label; eos, α0 = -20.0, T0 = 0.45, τ0 = 0.6, τ1 = 3.0, kw_model, kw_ref)
    m = H.build_model_1d(; eos, kw_model...)
    th(T) = (P = H.eos_Pne(T, α0*T, eos); H.local_thermo(T, α0*T, P[2], P[3], P[1], eos))
    η(T)  = m.enable_shear ? H.viscosity(T, th(T), m.shear) : 0.0
    τπ(T) = m.enable_shear ? H.τ_shear(T, th(T), m.shear) : 0.0
    ζ(T)  = m.enable_bulk  ? H.bulk_viscosity(T, th(T), m.bulk) : 0.0
    τΠ(T) = m.enable_bulk  ? H.τ_bulk(T, th(T), m.bulk) : 1.0
    eos_eP(T) = (P = H.eos_Pne(T, α0*T, eos); (P[3], P[1]))
    function Tof_e(e)                         # bisection on the light-dominated e(T)
        lo, hi = 0.01, 2.0
        for _ in 1:100
            mid = 0.5(lo + hi); eos_eP(mid)[1] < e ? (lo = mid) : (hi = mid)
        end
        0.5(lo + hi)
    end
    e0 = eos_eP(T0)[1]
    _, eref, φref, Πref = bjorken_dnmr(; eos_eP, Tof_e, η, τπ, ζ, τΠ, e0, τ0, τ1, kw_ref...)
    raw = map((0.02, 0.01, 0.005)) do c
        r = run_uniform(m; T0, α0, τ0, τ1, CFLτ = c)
        (e = r.e, φ = r.φ, Π = r.Π)
    end
    rel(q, ref, on) = on ? q/ref - 1 : 0.0
    errs(o) = (e = rel(o.e, eref(τ1), true), φ = rel(o.φ, φref(τ1), m.enable_shear),
               Π = rel(o.Π, Πref(τ1), m.enable_bulk))
    out = map(errs, raw)
    # the scheme is first order (measured: the `order` column), so the Richardson
    # extrapolation 2q(Δ/2) − q(Δ) removes the O(Δ) error and leaves the defect
    ext = errs((e = 2raw[3].e - raw[2].e, φ = 2raw[3].φ - raw[2].φ, Π = 2raw[3].Π - raw[2].Π))
    worst(o) = max(abs(o.e), abs(o.φ), abs(o.Π))
    p = log2(worst(out[2])/worst(out[3]))
    @printf("  %-34s  rel. err (e, φ, Π): CFLτ=0.005 %+.1e %+.1e %+.1e | extrapolated %+.1e %+.1e %+.1e | order %.2f\n",
            label, out[3].e, out[3].φ, out[3].Π, ext.e, ext.φ, ext.Π, p)
    return out, p, ext
end

function gate_A2()
    println("\nA2 — viscous Bjorken vs the 0+1D DNMR ODEs (first order: the relaxation is split)")
    conf = H.ConformalHQEOS(m_hq = 0.0, g_hq = 0.0)
    lat  = H.LatticeHRGEOS()
    sh = (enable_shear = true, eta_over_s = 0.2, tauShear_coeff = 0.2)
    bu = (enable_bulk = true, zeta_over_s = 0.1, tauPi_coeff = 15.0)
    cases = [
        ("shear, δ_ππ = 0",               conf, 0.0,   (; sh..., deltaShear_factor = 0.0), NamedTuple()),
        ("shear, δ_ππ = 4/3",             conf, 0.0,   (; sh..., deltaShear_factor = 4/3), (δππ = 4/3,)),
        ("shear, δ_ππ = 4/3, τ_ππ = 10/7", conf, 0.0,  (; sh..., deltaShear_factor = 4/3, taupi_pi_factor = 10/7),
                                                       (δππ = 4/3, τππ = 10/7)),
        ("bulk",                          lat, -20.0,  (; bu..., deltaShear_factor = 0.0), NamedTuple()),
        ("bulk, δ_ΠΠ = 2/3",              lat, -20.0,  (; bu..., deltaPi_factor = 2/3), (δΠΠ = 2/3,)),
        ("shear+bulk, λ_πΠ = 6/5",        lat, -20.0,  (; sh..., bu..., deltaShear_factor = 4/3, lambda_pi_Pi_factor = 6/5),
                                                       (δππ = 4/3, λπΠ = 6/5)),
        ("shear+bulk, λ_Ππ = 1.2",        lat, -20.0,  (; sh..., bu..., deltaShear_factor = 4/3, lambda_Pi_pi_factor = 1.2),
                                                       (δππ = 4/3, λΠπ = 1.2)),
    ]
    for (label, eos, α0, km, kr) in cases
        out, p, ext = viscous_case(label; eos, α0, kw_model = km, kw_ref = kr)
        @test max(abs(out[3].e), abs(out[3].φ), abs(out[3].Π)) < 2.5e-2     # raw, finest step
        @test max(abs(ext.e), abs(ext.φ), abs(ext.Π)) < 1e-3               # extrapolated to Δ → 0
        @test p > 0.7                   # first order (the split); exactly 1 in the limit
    end
    # (c) the sign rows discriminate: the same solve against a referee with the
    #     coupling's sign FLIPPED (i.e. what the pre-2026-09-11 code integrated)
    for (label, km, kgood, kbad, key) in
            (("λ_πΠ", (; sh..., bu..., deltaShear_factor = 4/3, lambda_pi_Pi_factor = 6/5),
                      (δππ = 4/3, λπΠ = 6/5), (δππ = 4/3, λπΠ = -6/5), :φ),
             ("λ_Ππ", (; sh..., bu..., deltaShear_factor = 4/3, lambda_Pi_pi_factor = 1.2),
                      (δππ = 4/3, λΠπ = 1.2), (δππ = 4/3, λΠπ = -1.2), :Π))
        _, _, eg = viscous_case("  $label, derived sign"; eos = lat, α0 = -20.0, kw_model = km, kw_ref = kgood)
        _, _, eb = viscous_case("  $label, FLIPPED sign (old code)"; eos = lat, α0 = -20.0, kw_model = km, kw_ref = kbad)
        @test abs(getfield(eb, key)) > 10*abs(getfield(eg, key))
    end
    return nothing
end

function main()
    @testset "A1/A2 — Bjorken (1+1D)" begin
        gate_A1()
        gate_A2()
    end
end
main()
