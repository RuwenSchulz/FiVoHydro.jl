# ==============================================================================
# test/test_bjorken_bulk2d.jl — GATE G0b: NONLINEAR BULK against a semi-analytic
# reference.
#
# The bulk sector's only quantitative check was the sound gate, which is LINEAR:
# a 1e-4 perturbation about a uniform background. Gubser cannot help — it is
# conformal, so zeta = 0 there. Nothing tested the bulk relaxation in a regime
# where Pi is a finite fraction of P and the background is evolving.
#
# Bjorken flow does. Transversally uniform, u^i = 0, theta = 1/tau, so the 2-D
# solver's bulk sector reduces to a closed pair of ODEs:
#
#     de/dtau  = -(e + P + Pi)/tau
#     dPi/dtau = ( -Pi - zeta/tau ) / tau_Pi
#
# (the code's bulk law is tau_Pi D Pi + Pi = -zeta theta, with no delta_PiPi
# term - read off dissipation2d.jl rather than assumed). Integrated by RK4 to a
# tolerance far below the solver's, that is an independent reference.
#
# WHAT THIS TESTS AND WHAT IT DOES NOT
#
# The reference calls the solver's OWN `bulk_coeffs_2d` for zeta and tau_Pi, so
# this gate is NOT a check on those formulas — gate Gs already pins zeta's
# magnitude and its Lorentzian temperature dependence against sound attenuation
# to ~1%. What this adds is the NONLINEAR dynamics those coefficients drive: the
# implicit relaxation update, its coupling back into the energy equation, and the
# resulting entropy production, at |Pi|/P up to tens of percent.
#
# T is recovered from e by bisection. That is legitimate here because the
# production EOS carries charm as a pure tracer - dP/dmu = 0 exactly - so e and P
# depend on T alone and the inversion is one-dimensional. That is asserted below
# rather than assumed.
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

# T0 = 0.30 and tau_f = 6 on purpose: zeta is a Lorentzian peaked at T = 0.175,
# and Bjorken cooling from 0.30 over tau = 0.4 -> 6 carries the system straight
# through that peak, so |Pi|/P is O(10%) rather than the O(0.5%) it would be if
# the run stayed hot. A bulk test that never enters the bulk regime is not one.
const T0    = 0.30
const ALPHA = -4.2
const TAU0  = 0.4

"""T from e by bisection; valid because dP/dmu = 0 for this EOS (asserted)."""
function T_of_e(e_target, eos)
    lo, hi = 1e-4, 5.0
    for _ in 1:200
        mid = 0.5*(lo+hi)
        (_, _, em) = H.eos_Pne(mid, 0.0, eos)
        (em < e_target) ? (lo = mid) : (hi = mid)
    end
    return 0.5*(lo+hi)
end

"""RK4 the Bjorken+bulk pair. Returns e(tau_f), Pi(tau_f), T(tau_f)."""
function bjorken_bulk_exact(model, τf; n = 400_000)
    eos = model.eos
    P0, n0, e0 = H.eos_Pne(T0, ALPHA*T0, eos)
    h = (τf - TAU0)/n
    e = e0; Π = 0.0; τ = TAU0
    rhs(τ, e, Π) = begin
        T = T_of_e(e, eos)
        P, nn, ee = H.eos_Pne(T, ALPHA*T, eos)
        ζ, τΠ = H.bulk_coeffs_2d(T, ALPHA*T, nn, ee, P, model)
        (-(e + P + Π)/τ, (-Π - ζ/τ)/max(τΠ, 1e-12))
    end
    for _ in 1:n
        k1 = rhs(τ, e, Π)
        k2 = rhs(τ+h/2, e+h/2*k1[1], Π+h/2*k1[2])
        k3 = rhs(τ+h/2, e+h/2*k2[1], Π+h/2*k2[2])
        k4 = rhs(τ+h,   e+h*k3[1],   Π+h*k3[2])
        e += h/6*(k1[1]+2k2[1]+2k3[1]+k4[1])
        Π += h/6*(k1[2]+2k2[2]+2k3[2]+k4[2])
        τ += h
    end
    return e, Π, T_of_e(e, eos)
end

function run_solver(ζs, τf; N = 24, CFL = 0.15)
    g = H.make_grid2d(N, N; xmax = 4.0, ymax = 4.0)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_shear = false,
                           enable_bulk = true, zeta_over_s = ζs, tauPi_coeff = 15.0,
                           enable_diff = false)
    L = m.layout
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_uniform!(U, g, m, TAU0; T0 = T0, alpha0 = ALPHA)
    r = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = CFL, CFLτ = 0.05, work = wk)
    @assert r.ok
    H.update_primitives_2d!(U, g, r.τ, m, wk)
    ng = g.nghost; i0 = H.lin(g, ng + N÷2, ng + N÷2)
    # transverse uniformity is part of the statement: nothing may vary in x,y
    Tspread = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        Tspread = max(Tspread, abs(exp(wk.yT[i]) - exp(wk.yT[i0]))/exp(wk.yT[i0]))
    end
    return (T = exp(wk.yT[i0]), e = wk.e[i0], P = wk.P[i0],
            Π = H.phys_from_stored(U[L.iPi, i0]), τ = r.τ, Tspread, m)
end

@testset "G0b — nonlinear bulk on Bjorken flow" begin
    eos = H.LatticeHRGEOS()

    # the inversion T(e) is one-dimensional only if charm is a pure tracer
    P1, n1, e1 = H.eos_Pne(0.30,  0.0, eos)
    P2, n2, e2 = H.eos_Pne(0.30, -2.0, eos)
    @printf("  dP/dmu = 0 check: P(mu=0) = %.10e vs P(mu=-2) = %.10e  (n = %.3e, %.3e)\n",
            P1, P2, n1, n2)
    @test abs(P1 - P2)/P1 < 1e-12
    @test abs(e1 - e2)/e1 < 1e-12

    for ζs in (0.05, 0.15, 0.30)
        τf = 6.0
        r = run_solver(ζs, τf)
        ee, Πe, Te = bjorken_bulk_exact(r.m, τf)
        dT = abs(r.T - Te)/Te
        dΠ = abs(r.Π - Πe)/max(abs(Πe), 1e-30)
        @printf("  zeta/s=%.2f | T %.8f vs %.8f (%.2e) | Pi %+.8f vs %+.8f (%.2e) | |Pi|/P = %.4f | x,y spread %.1e\n",
                ζs, r.T, Te, dT, r.Π, Πe, dΠ, abs(r.Π)/r.P, r.Tspread)
        @test r.Tspread < 1e-12          # Bjorken must stay transversally uniform
        @test dT < 1e-3
        # sub-percent at the PRODUCTION timestep (measured 0.52 / 0.75 / 0.16 %);
        # the CFL scan below shows it is the timestep and converges away.
        @test dΠ < 1e-2
        @test abs(r.Π)/r.P > 0.05        # the test must actually be in a bulk regime
    end

    # CONVERGENCE: the residual must be the solver's timestep, not a modelling gap
    println("  --- convergence in CFL at zeta/s = 0.15 ---")
    ee, Πe, Te = bjorken_bulk_exact(run_solver(0.15, 6.0; CFL = 0.15).m, 6.0)
    prev = 0.0; prevP = 0.0; lastP = 0.0
    for CFL in (0.20, 0.10, 0.05, 0.025)
        r = run_solver(0.15, 6.0; CFL = CFL)
        dT = abs(r.T - Te)/Te
        dP = abs(r.Π - Πe)/abs(Πe)
        @printf("    CFL=%.3f  |dT|/T = %.3e%s   |dPi|/Pi = %.3e%s\n", CFL, dT,
                prev  > 0 ? @sprintf(" (x%.2f)", prev/dT)  : "     ", dP,
                prevP > 0 ? @sprintf(" (x%.2f)", prevP/dP) : "")
        prev = dT; prevP = dP; lastP = dP
    end
    @test prev  < 1e-4
    @test lastP < 1e-3
end
