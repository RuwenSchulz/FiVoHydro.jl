# ==============================================================================
# test/test_gubser_viscous2d.jl — GATE G1v: VISCOUS Gubser flow.
#
# G1 covers IDEAL Gubser. The plan asked for "Gubser ideal + viscous"; this is the
# viscous half, and it is the only test in the ladder that compares the NONLINEAR
# shear sector against an exact solution. (Sound waves test it in the LINEAR
# regime; everything else tests it against the 1-D solver or against a sign.)
#
# ---------------------------------------------------------------------------
# THE REFERENCE, DERIVED RATHER THAN QUOTED
#
# Gubser flow is a fixed point of the conformal symmetry, so the KINEMATICS
# (u^mu) are the same as the ideal case whatever the viscosity: only T and pi
# change. Weyl-rescaling Milne by 1/tau^2 and going to de Sitter coordinates
# leaves the fluid AT REST, with
#     theta_hat = 2 tanh(rho),   sigma_hat^eta_eta = -(2/3) tanh(rho),
#     sigma_hat^theta_theta = sigma_hat^phi_phi = +(1/3) tanh(rho).
# Writing pibar = pi^eta_eta/(e+P) (a Weyl-invariant ratio, so it is the SAME
# number in Milne), energy conservation and the IS shear equation with
# delta_pipi = (4/3) tau_pi reduce to
#
#     dT_hat/drho  = -(2/3) T_hat tanh(rho) - (1/3) T_hat pibar tanh(rho)
#     dpibar/drho  =  (4/3) pibar^2 tanh(rho) - pibar/tau_pi_hat - (4/15) tanh(rho)
#
# with tau_pi_hat = 5 (eta/s) / T_hat. The delta_pipi term cancels the (8/3)
# pibar tanh(rho) from the chain rule exactly, which is why the second equation
# is this clean.
#
# TWO INDEPENDENT VALIDATIONS OF THAT REFERENCE, both run before the solver:
#
#  1. eta/s -> 0 must integrate to T_hat = T_hat0 cosh^{-2/3}(rho), which via the
#     identity cosh^2(rho) = Den/(4 q^2 tau^2) is EXACTLY the code's own
#     `gubser_temperature`. So the ODE solver, the rho(tau,r) map and the Weyl
#     weight are all checked against an independently-written formula.
#  2. at small tau_pi the shear must relax onto Navier-Stokes,
#     pibar_NS = -(4/3)(eta/s) tanh(rho)/T_hat.
#
# ⚠ The comparison is run at SMALL eta/s on purpose. There the solution sits near
# the NS limit, so the reference is insensitive to the tau_pi convention — which
# is the one thing here that cannot be checked from inside this file (the sound
# gate established empirically that the realised tau_pi is the natural-units one).
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

const QG    = 1.0
const TAU0  = 1.0
const TH0   = 0.6            # T_hat at rho = 0  (= T(tau0, r=0) since tau0 = 1)
const ALPHA = -20.0
const BOX   = 10.0
const RCMP  = 3.0
const EOS   = H.ConformalHQEOS()
# ⚠ tau_pi = 5 (eta/s) / T with T in NATURAL units (fm^-1), not GeV. Using GeV
# makes tau_pi -- and hence the Navier-Stokes fixed point pibar_NS =
# (4/3)(eta/s) tanh(rho)/T_hat_nat -- a factor invfmGeV = 5.07 too large, which is
# exactly the discrepancy the first version of this gate showed against the solver
# (solver 0.00345 vs reference 0.01477). Same trap as the sound gate.
const INVFMGEV = 1/0.1973269804

ρ_of(τ, r) = asinh(-(1 - QG^2*τ^2 + QG^2*r^2)/(2QG*τ))
urana(τ, r) = hydro.gubser_ur(τ, r, QG)

"""RK4 the (T_hat, pibar) system, FORWARD IN RHO ONLY.

Integrating backward is exponentially unstable and that is not a bug in the
solver, it is the physics: the relaxation term -pibar/tau_pi_hat is anti-damping
in reverse, so round-off grows like exp(|drho|/tau_pi_hat). Measured, the
integration dies at |drho|/tau_pi_hat = 16-19 for every eta/s tried (rho = -0.32
at eta/s = 0.002, -0.69 at 0.005, -2.77 at 0.02) — exactly where 1e-16 has been
amplified to O(1). Smaller eta/s dies SOONER, which is the giveaway.

Forward is the attracting direction: pibar relaxes onto the hydrodynamic
attractor within a few tau_pi_hat whatever it starts at, which is why the seed
below is the Navier-Stokes value and why starting from 0 instead must give the
same answer (asserted).
"""
function solve_ode(ηs; ρmin = -6.0, ρmax = 2.0, n = 400_000, seed_ns = true)
    h = (ρmax - ρmin)/n
    ρs = collect(range(ρmin, ρmax; length = n+1))
    Th = zeros(n+1); pb = zeros(n+1)
    f(ρ, T, p) = begin
        th = tanh(ρ)
        τπ = ηs > 0 ? 5*ηs/(max(T, 1e-12)*INVFMGEV) : 0.0
        dT = -(2/3)*T*th + (1/3)*T*p*th
        dp = ηs > 0 ? -(4/3)*p*p*th - p/τπ + (4/15)*th : 0.0
        (dT, dp)
    end
    # T_hat of the IDEAL solution is closed form, so the starting temperature can
    # be set analytically with no backward integration at all.
    Th[1] = TH0*cosh(ρmin)^(-2/3)
    pb[1] = (ηs > 0 && seed_ns) ? (4/3)*ηs*tanh(ρmin)/(Th[1]*INVFMGEV) : 0.0
    for i in 1:n
        T = Th[i]; p = pb[i]; ρ = ρs[i]
        k1 = f(ρ, T, p); k2 = f(ρ+h/2, T+h/2*k1[1], p+h/2*k1[2])
        k3 = f(ρ+h/2, T+h/2*k2[1], p+h/2*k2[2]); k4 = f(ρ+h, T+h*k3[1], p+h*k3[2])
        Th[i+1] = T + h/6*(k1[1]+2k2[1]+2k3[1]+k4[1])
        pb[i+1] = p + h/6*(k1[2]+2k2[2]+2k3[2]+k4[2])
    end
    return ρs, Th, pb
end

"""Linear lookup on the uniform rho grid."""
function mk_interp(ρs, v)
    lo = ρs[1]; h = ρs[2]-ρs[1]; n = length(ρs)
    return function (ρ)
        t = (ρ - lo)/h
        j = clamp(floor(Int, t) + 1, 1, n-1)
        w = t - (j-1)
        (1-w)*v[j] + w*v[j+1]
    end
end

"""Seed the 2-D state from the ODE solution at proper time tau."""
function seed!(U, g, m, τ, That, pibar)
    L = m.layout
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        ρ = ρ_of(τ, r)
        T = That(ρ)/τ
        ur = urana(τ, r); γ = sqrt(1 + ur^2)
        ux = r > 0 ? ur*x/r : 0.0; uy = r > 0 ? ur*y/r : 0.0
        P, _, e = H.eos_Pne(T, ALPHA*T, m.eos)
        Πη = pibar(ρ)*(e + P)
        # pi^{mu nu} = Pi_eta [ -(1/2)(e_r e_r + e_phi e_phi) + e_eta e_eta ], with
        # e_r = (u_r, gamma x/r, gamma y/r, 0), e_phi = (0, -y/r, x/r, 0).
        if r > 1e-12
            c = x/r; sφ = y/r
            pixx = Πη*(-(1/2)*(γ^2*c^2 + sφ^2))
            piyy = Πη*(-(1/2)*(γ^2*sφ^2 + c^2))
            pixy = Πη*(-(1/2)*c*sφ*(γ^2 - 1))
        else
            pixx = -Πη/2; piyy = -Πη/2; pixy = 0.0
        end
        H.set_cell!(U, H.lin(g, ix, iy), T, ALPHA, ux, uy, τ, m;
                    pixx = pixx, pixy = pixy, piyy = piyy)
    end
    H.finalize_ic!(U, g, m; τ0 = τ)
end

"""Run and return the errors in T and in pibar against the ODE at tau_f."""
function run_visc(N, ηs, τf, That, pibar)
    g = H.make_grid2d(N, N; xmax = BOX, ymax = BOX)
    m = H.build_model_2d(; eos = EOS, enable_shear = true, eta_over_s = ηs,
                           tauShear_coeff = 0.2, deltaShear_factor = 4/3,
                           enable_bulk = false, enable_diff = false)
    L = m.layout
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    seed!(U, g, m, TAU0, That, pibar)
    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05, work = wk)
    @assert res.ok
    H.update_primitives_2d!(U, g, res.τ, m, wk)
    τ = res.τ; ng = g.nghost
    sT = 0.0; sTa = 0.0; sp = 0.0; spa = 0.0; asym = 0.0; nc = 0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y); r <= RCMP || continue
        i = H.lin(g, ix, iy); nc += 1
        ρ = ρ_of(τ, r)
        Ta = That(ρ)/τ; Tn = exp(wk.yT[i])
        sT += (Tn - Ta)^2; sTa += Ta^2
        Pn = wk.P[i]; en = wk.e[i]
        pbn = H.phys_from_stored(U[L.iPieta, i])/(en + Pn)
        pba = pibar(ρ)
        sp += (pbn - pba)^2; spa += pba^2
        j = H.lin(g, ng+1+(iy-ng-1), ng+1+(ix-ng-1))
        a = exp(wk.yT[i]); b = exp(wk.yT[j])
        asym = max(asym, abs(a-b)/max(a,b))
    end
    L2T = sqrt(sT/sTa); L2p = spa > 0 ? sqrt(sp/spa) : NaN
    @printf("    N=%3d eta/s=%.3f tau=%.1f | L2(T)=%.3e  L2(pibar)=%.3e | x<->y=%.2e | %5.1fs\n",
            N, ηs, τ, L2T, L2p, asym, time()-t0)
    return (; L2T, L2p, asym, nc)
end

@testset "G1v — viscous Gubser" begin
    # ---------- validation 1: eta/s -> 0 reproduces the code's ideal formula ----------
    ρs0, Th0, pb0 = solve_ode(0.0)
    Ti0 = mk_interp(ρs0, Th0)
    Tscale = hydro.gubser_Tscale_from_center_T(TAU0, QG, Ti0(0.0))
    worst = 0.0
    for τ in (1.0, 1.5, 2.0), r in 0.0:0.25:RCMP
        a = Ti0(ρ_of(τ, r))/τ
        b = hydro.gubser_temperature(τ, r, QG, Tscale)
        worst = max(worst, abs(a-b)/b)
    end
    @printf("  ODE at eta/s=0 vs the code's analytic ideal Gubser: max rel diff %.3e\n", worst)
    @test worst < 1e-8
    @test maximum(abs.(pb0)) == 0.0

    # ---------- validation 2: the shear relaxes onto Navier-Stokes ----------
    for ηs in (0.005, 0.02)
        ρs, Th, pb = solve_ode(ηs)
        Ti = mk_interp(ρs, Th); Pi = mk_interp(ρs, pb)
        w = 0.0
        for ρ in -3.0:0.1:1.0
            abs(ρ) > 0.2 || continue
            ns = (4/3)*ηs*tanh(ρ)/(Ti(ρ)*INVFMGEV)
            w = max(w, abs(Pi(ρ) - ns)/abs(ns))
        end
        @printf("  eta/s=%.3f: max deviation of pibar from Navier-Stokes = %.3e  (max|pibar| = %.4f)\n",
                ηs, w, maximum(abs.(pb)))
        @test w < 0.5
        @test all(isfinite, pb) && all(>(0), Th)

        # the attractor: starting pibar at 0 instead of at NS must give the same
        # solution once the transient has decayed
        _, _, pb2 = solve_ode(ηs; seed_ns = false)
        Pi2 = mk_interp(ρs, pb2)
        d = maximum(abs(Pi(ρ) - Pi2(ρ)) for ρ in -3.0:0.05:1.0)
        @printf("    attractor: |pibar(seed=NS) - pibar(seed=0)| over the comparison range = %.3e\n", d)
        @test d < 5e-2
    end

    # ---------- the solver ----------
    for ηs in (0.005, 0.02)
        println("  --- viscous Gubser, eta/s = $ηs, tau = 1 -> 2 ---")
        ρs, Th, pb = solve_ode(ηs)
        Ti = mk_interp(ρs, Th); Pi = mk_interp(ρs, pb)
        rs = [run_visc(N, ηs, 2.0, Ti, Pi) for N in (100, 200, 400)]
        for r in rs
            @test r.nc > 100
            @test r.asym < 1e-11
        end
        oT = log2(rs[2].L2T/rs[3].L2T)
        op = log2(rs[2].L2p/rs[3].L2p)
        @printf("    convergence order  T: %.2f   pibar: %.2f\n", oT, op)
        @test rs[end].L2T < 5e-3
        @test rs[end].L2p < 8e-2
        @test oT > 1.0
        @test op > 0.7
    end
end
