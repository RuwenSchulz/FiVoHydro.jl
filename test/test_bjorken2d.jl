# ==============================================================================
# test/test_bjorken2d.jl — gate G0.
#
# A transversely uniform state in 2+1D Milne is Bjorken flow. It exercises
# exactly the pieces that are NEW relative to the 1-D solver's Bjorken test:
# the two-direction unsplit flux divergence must cancel to round-off, and the
# Cartesian geometric source must reproduce the analytic dilution.
#
# EXACT REFERENCE (no integrator, no fit). Ideal Bjorken conserves
#
#       τ n = const        (charge)
#       τ s = const        (entropy, ideal flow)
#
# so at any τ the state solves n(T,μ) = n₀τ₀/τ and s(T,μ) = s₀τ₀/τ. That is two
# algebraic equations for (T, μ) — solved here by an independently written 2-D
# Newton, NOT by the solver's own primitive recovery.
#
# What each check catches:
#   * τJ^τ exactly constant  -> the transverse fluxes cancel and the charge
#                               source is genuinely zero
#   * u^x = u^y ≡ 0          -> no spurious transverse drift from the unsplit
#                               reconstruction (would be invisible in 1-D)
#   * uniformity preserved   -> the BCs do not leak gradients inward
#   * E(τ) vs exact          -> the geometric source -(E+P+Π+τ²π^{ηη})/τ is right
#   * order of convergence   -> the source is integrated, not merely present
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_bjorken2d.jl
# ==============================================================================

using Printf
using Test

include(joinpath(@__DIR__, "..", "main2D.jl"))
using .hydro2d

const H = hydro2d

# ------------------------------------------------------------------------------
# Analytic reference.
#
# CAREFUL — the entropy that is conserved here is NOT `eos_entropy`. The
# production EOS carries charm as a pure TRACER: measured, `LatticeHRGEOS` has
# `dP/dmu = 0` EXACTLY while `n != 0`, so charm contributes nothing to P or e.
# `eos_entropy(T,mu,n,e,P) = (e+P-mu*n)/T` therefore subtracts a term the pressure
# never contained, and `tau*s` built from it is NOT an invariant of the ideal
# Bjorken ODE. Using it puts a dt-INDEPENDENT floor of 3e-4 on this gate — which
# is how the convergence test below caught the mistake.
#
# The correct statement: the EOS satisfies the Euler relation `e = T dP/dT - P`
# (verified to 2e-10), hence `e + P = T dP/dT` and
#
#       s = (e + P)/T           tau * s  = const
#       n                       tau * n  = const
#
# Proof that `tau*s` is exactly the invariant of the solver's ODE: with
# `de/dtau = -(e+P)/tau` and `s = dP/dT`, `ds/dT = d2P/dT2 = (de/dT)/T`, so
# `ds/dtau = (1/T) de/dtau = -s/tau`, giving `d(tau*s)/dtau = 0`.
#
# T is then fixed by tau*s alone (s is monotone in T) and mu follows from n.
# Both are found by bisection — no use of the solver's own recovery.
# ------------------------------------------------------------------------------

"""Entropy consistent with this EOS family: s = (e+P)/T = dP/dT."""
@inline s_of_T(T, mu, eos) = ((P, n, e) = H.eos_Pne(T, mu, eos); (e + P)/T)

function bjorken_exact(eos, T0, mu0, tau0, tau)
    P0, n0, e0 = H.eos_Pne(T0, mu0, eos)
    s0   = (e0 + P0)/T0
    star = s0 * tau0/tau
    ntar = n0 * tau0/tau

    # T from tau*s = const  (s increases monotonically with T)
    lo, hi = 1e-4, 5.0
    for _ in 1:200
        mid = 0.5*(lo + hi)
        s_of_T(mid, mu0, eos) > star ? (hi = mid) : (lo = mid)
    end
    T = 0.5*(lo + hi)

    # mu from n(T,mu) = ntar  (n increases monotonically with mu)
    mlo, mhi = -40.0, 40.0
    for _ in 1:200
        mid = 0.5*(mlo + mhi)
        (_, nm, _) = H.eos_Pne(T, mid, eos)
        nm > ntar ? (mhi = mid) : (mlo = mid)
    end
    mu = 0.5*(mlo + mhi)

    P, n, e = H.eos_Pne(T, mu, eos)
    return (T = T, mu = mu, n = n, e = e, P = P)
end

"""Run a uniform (Bjorken) 2-D state and report deviations from exact."""
function run_bjorken(; Nx = 24, Ny = 24, τ0 = 0.4, τf = 4.0, T0 = 0.50, alpha0 = -4.2,
                       CFLτ = 0.02, integrator = :ssprk2, eos = H.LatticeHRGEOS())
    g = H.make_grid2d(Nx, Ny; xmax = 6.0, ymax = 6.0)
    m = H.build_model_2d(; eos = eos)
    L = m.layout
    U = H.allocate_state(g, m)
    @assert H.initialize_uniform!(U, g, m, τ0; T0 = T0, alpha0 = alpha0) == 0

    Dtau0 = U[L.iDtau, H.lin(g, g.nghost+1, g.nghost+1)]

    res = H.run_sim_2d!(U, g, m; τ0 = τ0, τfinal = τf, CFL = 0.2, CFLτ = CFLτ,
                        integrator = integrator)
    @assert res.ok

    # interior statistics
    ng = g.nghost
    maxDerr = 0.0; maxU = 0.0; Espread = 0.0
    Emin = Inf; Emax = -Inf
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        maxDerr = max(maxDerr, abs(U[L.iDtau,i] - Dtau0)/abs(Dtau0))
        maxU    = max(maxU, abs(res.work.ux[i]), abs(res.work.uy[i]))
        Emin = min(Emin, U[L.iE,i]); Emax = max(Emax, U[L.iE,i])
    end
    Espread = (Emax - Emin)/abs(Emax)

    ic = H.lin(g, ng + g.Nx÷2, ng + g.Ny÷2)
    ex = bjorken_exact(eos, T0, alpha0*T0, τ0, res.τ)
    Eerr = abs(U[L.iE,ic] - ex.e)/ex.e
    Terr = abs(exp(res.work.yT[ic]) - ex.T)/ex.T
    nerr = abs(res.work.n[ic] - ex.n)/ex.n

    return (τ = res.τ, nsteps = res.nsteps, primfail = res.nprimfail,
            maxDerr = maxDerr, maxU = maxU, Espread = Espread,
            Eerr = Eerr, Terr = Terr, nerr = nerr)
end

@testset "G0 — Bjorken (transversely uniform 2+1D)" begin

    @testset "exact invariants" begin
        r = run_bjorken(; CFLτ = 0.02)
        @printf("  steps %d  primfail %d\n", r.nsteps, r.primfail)
        @printf("  tau*J^tau drift     = %.3e   (must be round-off: no flux, no source)\n", r.maxDerr)
        @printf("  max |u^x|,|u^y|     = %.3e   (must be identically zero)\n", r.maxU)
        @printf("  transverse E spread = %.3e   (uniformity must be preserved)\n", r.Espread)
        @test r.primfail == 0
        @test r.maxDerr < 1e-13
        @test r.maxU == 0.0
        @test r.Espread < 1e-13
    end

    @testset "energy against the analytic solution" begin
        r = run_bjorken(; CFLτ = 0.005)
        @printf("  at tau = %.3f:  E err %.3e   T err %.3e   n err %.3e\n",
                r.τ, r.Eerr, r.Terr, r.nerr)
        @test r.Eerr < 5e-4
        @test r.Terr < 5e-4
        @test r.nerr < 5e-4
    end

    @testset "convergence order of the geometric source" begin
        # Halving CFLτ must reduce the error like the integrator's order. This is
        # what separates "the source is present" from "the source is integrated
        # correctly" — a wrong source coefficient converges to the WRONG answer at
        # full order, so the order test is run together with the absolute test above.
        for (integ, expect) in ((:ssprk2, 2.0), (:ssprk3, 3.0))
            errs = Float64[]
            for c in (0.02, 0.01, 0.005)
                push!(errs, run_bjorken(; CFLτ = c, integrator = integ).Eerr)
            end
            p1 = log2(errs[1]/errs[2]); p2 = log2(errs[2]/errs[3])
            @printf("  %-7s errors %.3e %.3e %.3e   observed order %.2f, %.2f\n",
                    integ, errs[1], errs[2], errs[3], p1, p2)
            @test p2 > expect - 0.6
        end
    end
end
