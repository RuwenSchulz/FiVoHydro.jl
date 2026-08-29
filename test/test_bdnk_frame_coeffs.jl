#!/usr/bin/env julia
# test_bdnk_frame_coeffs.jl — the BDNK general-frame coefficients of the charm current.
# Run as an isolated subprocess from runtests.jl. Exits 0 on pass, 1 on failure.
#
# WHY THIS EXISTS (2026-08-29). `FiVoBenchmark/CAUSAL_HYDRO_AUDIT.md` finding B-4 records
# that BOTH shipped BDNK backgrounds have ∂_rT = 0 AND u^r = 0. Those two conditions kill
# the Soret term *and* the acceleration term identically, so no existing gate can see either
# coefficient. G5 below reproduces that blindness explicitly, which is the point: the gate is
# only worth anything on a background carrying both gradients.
#
# The physics, verified symbolically in `Julia/tools/derive_bdnk_frame_coeffs.wl` (19/19):
#   the Chapman–Enskog vector source is  k^<μ>( ∇_μα + (E/T)[∇_μ lnT ∓ u̇_μ] ),
#   i.e. the acceleration carries the SAME moment weight E/T as the temperature gradient.
#   Hence  σ_T = κ_n z K₃/K₂ = n τ_n   and   σ_a = -σ_T = -n τ_n,
#   the second being the terminal-velocity lag −n τ_n a^r that `AttractorHydro`
#   App. app:overdamped derives independently from the Smoluchowski limit.
#
# Gate flags live inside a function on purpose: `ok = false` inside a top-level `for` makes a
# NEW LOCAL and the gate then prints its failures and reports PASS (CLAUDE.md; bitten twice).

include(joinpath(@__DIR__, "..", "main2.jl"))
using .hydro_current
const HC = hydro_current

function run_gate()
    ok   = true
    fail(msg) = (ok = false; println("  FAIL  ", msg))
    pass(msg) = println("  PASS  ", msg)

    eos  = HC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    M    = HC.hq_mass(eos)
    DsT  = 0.116                      # the constant prescription, LangevinPaper1 §II
    Ts   = [0.16, 0.20, 0.25, 0.30, 0.40, 0.50]

    println("="^70)
    println(" BDNK general-frame coefficients of the charm current   (M = $M GeV)")
    println("="^70)

    # ---------------------------------------------------------------- G1
    # σ_T = n τ_n : ties two INDEPENDENTLY CODED shipped functions.
    # diff_sigmaT_bg builds κ·z·K₃/K₂ directly; diff_tauN_bg builds
    # (DsT/48)(z³/T)(2K₁−3K₃+K₅)/K₂, which equals D_s z K₃/K₂ only through the
    # LatticeHRG degeneracy identity 2K₁−3K₃+K₅ = (48/z³)(4K₂ + K₁z).
    let worst = 0.0
        for T in Ts
            n  = 1.0
            σT = HC.diff_sigmaT_bg(T, 0.0, n, DsT, eos)
            τn = HC.diff_tauN_bg(T, 0.0, DsT, eos)
            worst = max(worst, abs(σT/(n*τn) - 1))
        end
        worst < 1e-12 ? pass("G1  σ_T == n·τ_n  (two shipped functions agree)   max rel = $worst") :
                        fail("G1  σ_T != n·τ_n : max rel = $worst")
    end

    # ---------------------------------------------------------------- G2
    # |σ_a| == |σ_T|. Forced by Tolman–Ehrenfest: ∇_μ lnT ∓ u̇_μ is the combination
    # that vanishes in global equilibrium, so the two coefficients cannot be independent.
    let worst = 0.0, tbl = String[]
        for T in Ts
            n  = 1.0
            σT = HC.diff_sigmaT_bg(T, 0.0, n, DsT, eos)
            σa = HC.diff_sigmaa_bg(T, 0.0, n, DsT, eos)
            worst = max(worst, abs(abs(σa)/abs(σT) - 1))
            push!(tbl, "        T = $T GeV,  z = $(round(M/T, digits=2)):  σ_a/σ_T = $(round(σa/σT, sigdigits=6))")
        end
        if worst < 1e-12
            pass("G2  |σ_a| == |σ_T|   max rel = $worst")
        else
            fail("G2  |σ_a| != |σ_T| : max rel dev = $worst   (expected σ_a = -σ_T = -n τ_n)")
            println.(tbl)
            println("        the ratio is not a pure number ⇒ σ_a is dimensionally wrong.")
        end
    end

    # ---------------------------------------------------------------- A1/A2
    # THE TWO INVARIANT ANCHORS. Everything above is a relation between coefficients; these two
    # pin the coefficients themselves to physics, in a form that no transcription, signature or
    # unit convention can rotate. The repo carried THREE readings of κ_n (D_s n/T, D_s n,
    # D_s P₀/T) and two overall sign conventions for the σ terms; A1 and A2 decide between them
    # by measurement instead of by reading. Adopt whichever convention passes them.
    #
    #   A1  Fick.  Static, isothermal, no flow  ⇒  ν^r = -D_s ∂_r n   exactly.
    #             This is what fixes κ_n = D_s n (and kills the D_s n/T reading, which is
    #             2.0-6.3x too large over T = 0.5 -> 0.16 GeV).
    #   A2  The lag.  Uniform n and T, flowing and accelerating  ⇒  ν^r = -n τ_n a^r  exactly.
    #             The tracer drifts BEHIND the accelerating fluid at the terminal velocity where
    #             drag balances inertia. This is what fixes σ_a, sign and magnitude, and it is
    #             the same statement AttractorHydro App. app:overdamped reaches independently.
    let worst = 0.0
        for T in Ts, n in (0.5, 2.0)
            κ  = HC.diff_kappa_bg(T, 0.0, n, DsT, eos)
            Ds = DsT/T*0.1973269804                       # the spatial diffusion coefficient [fm]
            # ν^r = -κ ∂_rα ; isothermal ⇒ ∂_rα = ∂_r ln n = (∂_r n)/n. Take ∂_r n = 1 fm^-4.
            νr   = -κ * (1.0/n)
            want = -Ds * 1.0
            worst = max(worst, abs(νr/want - 1))
        end
        worst < 1e-12 ? pass("A1  Fick: ν^r = -D_s ∂_r n on a static isothermal cell   max rel = $worst") :
                        fail("A1  κ_n is not D_s·n : max rel dev = $worst " *
                             "(D_s n/T would fail by 1/T = 2.0-6.3x here)")
    end
    let worst = 0.0
        for T in Ts
            n  = 1.0
            σa = HC.diff_sigmaa_bg(T, 0.0, n, DsT, eos)
            τn = HC.diff_tauN_bg(T, 0.0, DsT, eos)
            # uniform n, T ⇒ only the acceleration term survives: ν^r = σ_a a^r, want -n τ_n a^r
            worst = max(worst, abs(σa/(-n*τn) - 1))
        end
        worst < 1e-12 ? pass("A2  lag:  ν^r = -n τ_n a^r on a uniform accelerating cell  max rel = $worst") :
                        fail("A2  σ_a is not -n·τ_n : max rel dev = $worst")
    end

    # ---------------------------------------------------------------- G3/G4
    # The hydrostatic null. Static (∂_τ = 0), α = const, u^r = sinh θ(r), and T chosen so
    # the Tolman combination vanishes:  (u^τ)² ∂_r lnT = -u^r ∂_r u^r  ⇒  T = T₀/cosh θ.
    # A medium in hydrostatic equilibrium carries NO diffusion current, at any flow velocity.
    # This is the constitutive-relation twin of ConventionNote's equilibrium null.
    #
    # ν^r is the shipped constitutive relation, eq:milne_bdnk_current with ∂_τ = 0:
    #     ν^r = -κ (u^τ)² ∂_rα  -  σ_T (u^τ)² ∂_r lnT  +  σ_a (u^r ∂_r u^r)
    nu_r(θ, dθ, T0, n0) = begin
        uτ, ur = cosh(θ), sinh(θ)
        T      = T0/cosh(θ)                       # hydrostatic
        dlnT   = -tanh(θ)*dθ                      # d/dr ln T
        κ      = HC.diff_kappa_bg(T, 0.0, n0, DsT, eos)
        σT     = HC.diff_sigmaT_bg(T, 0.0, n0, DsT, eos)
        σa     = HC.diff_sigmaa_bg(T, 0.0, n0, DsT, eos)
        (-σT*uτ^2*dlnT + σa*(ur*cosh(θ)*dθ), κ, σT, σa, T, uτ)
    end

    # θ(r) = θ_max * tanh(r/w) : smooth, u^r up to ~1.2 (the Pb+Pb freeze-out surface runs
    # at u^r ≃ 0.4–1, ConventionNote §III), and ∂_rT ≠ 0 everywhere it matters.
    θmax, w, T0, n0 = 1.0, 4.0, 0.30, 1.0
    rs = range(0.5, 12.0; length = 200)
    let worst = 0.0, scale = 0.0, urmax = 0.0, dTmax = 0.0
        for r in rs
            θ  = θmax*tanh(r/w); dθ = θmax*(1 - tanh(r/w)^2)/w
            ν, κ, σT, σa, T, uτ = nu_r(θ, dθ, T0, n0)
            worst = max(worst, abs(ν))
            scale = max(scale, abs(σT*uτ^2*tanh(θ)*dθ))     # size of either term alone
            urmax = max(urmax, sinh(θ))
            dTmax = max(dTmax, abs(-tanh(θ)*dθ))
        end
        println("        background: max u^r = $(round(urmax, digits=3)), " *
                "max |∂_r lnT| = $(round(dTmax, digits=4)) fm⁻¹  (both nonzero — this is the point)")
        rel = worst/max(scale, eps())
        rel < 1e-12 ? pass("G4  hydrostatic null: ν^r ≡ 0 at α = const   max|ν^r|/scale = $rel") :
                      fail("G4  hydrostatic null VIOLATED: max|ν^r|/scale = $rel " *
                           "(a medium in equilibrium is carrying a diffusion current)")
    end

    # ---------------------------------------------------------------- G5
    # The blindness proof. On the two shipped BDNK backgrounds (u^r = 0, ∂_rT = 0) the
    # Soret and acceleration terms vanish identically, so ANY value of σ_a passes. This is
    # why B-4 could not see it, and why G4 must be run on a flowing, non-isothermal cell.
    let θ = 0.0, dθ = 0.0
        ν, κ, σT, σa, T, uτ = nu_r(θ, dθ, T0, n0)
        if abs(ν) < 1e-14
            pass("G5  blindness reproduced: on u^r = 0, ∂_rT = 0 the current is ν^r = $ν")
            println("        ⇒ σ_T and σ_a are unconstrained there. B-4's backgrounds " *
                    "could not have caught this.")
        else
            fail("G5  expected an identically zero current on the flat background, got $ν")
        end
    end

    println("="^70)
    println(ok ? " GATE PASSED" : " GATE FAILED")
    println("="^70)
    return ok
end

exit(run_gate() ? 0 : 1)
