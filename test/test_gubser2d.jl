# ==============================================================================
# test/test_gubser2d.jl — GATE G1: GUBSER FLOW, the analytic 2-D acceptance test.
#
# The original ladder (TWOD_PROGRAM.md §6) had "G2 | Gubser ideal + viscous |
# analytic". The implemented ladder renumbered around it and it was never
# written, which left the 2-D solver with NO comparison against an exact
# solution: G0 is Bjorken (transversally uniform, so it exercises none of the
# 2-D machinery), and every other gate compares against the 1-D solver, against
# itself at another resolution, or against a physical expectation.
#
# Gubser flow is the standard acceptance test for a 2+1D relativistic hydro code:
# an exact solution of IDEAL CONFORMAL hydrodynamics in Milne coordinates with
# non-trivial radial flow. It is azimuthally symmetric, which is a feature here —
# the solver is given no hint of that, so the run tests
#
#   (a) accuracy against truth, and its CONVERGENCE ORDER;
#   (b) that a Cartesian (x,y) discretisation does not manufacture azimuthal
#       structure out of a radially symmetric solution.
#
# ---------------------------------------------------------------------------
# THE REFERENCE IS VALIDATED, NOT ASSUMED
#
# Two preliminaries run before the solver is touched at all, because a gate that
# compares a solver against an unchecked formula tests neither:
#
#   1. the EOS actually used is conformal to a measured tolerance (the light
#      sector is a_SB T^4 but `ConformalHQEOS` carries a MASSIVE heavy-quark
#      Boltzmann sector on top, which is not conformal — it is suppressed here
#      by the tracer fugacity, and the residual e/(3P) - 1 is measured);
#   2. the analytic solution satisfies ideal conformal hydrodynamics, checked by
#      finite-differencing the entropy current: for ideal flow d_mu (s u^mu) = 0
#      exactly, which in Milne with s ~ T^3 is
#          (1/tau) d_tau (tau s u^tau) + (1/r) d_r (r s u^r) = 0.
#      This validates gubser_temperature and gubser_ur TOGETHER.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_gubser2d.jl
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))      # 1-D module: the Gubser analytic formulas
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

# ---- Gubser parameters -------------------------------------------------------
const QG    = 1.0        # 1/fm, the standard choice
const TAU0  = 1.0        # fm/c
const TC0   = 0.6        # GeV at (tau0, r=0)
const ALPHA = -20.0      # tracer fugacity: mu = alpha*T, suppresses the HQ sector
const BOX   = 10.0       # fm; comparison is restricted well inside this
const RCMP  = 3.0        # fm, comparison radius

const EOS    = H.ConformalHQEOS()
const TSCALE = hydro.gubser_Tscale_from_center_T(TAU0, QG, TC0)

Tana(τ, r)  = hydro.gubser_temperature(τ, r, QG, TSCALE)
urana(τ, r) = hydro.gubser_ur(τ, r, QG)

# ------------------------------------------------------------------ preliminary 1
"""How conformal is the EOS we actually integrate? Returns max|e/(3P) - 1|."""
function conformality_residual()
    worst = 0.0; Tw = 0.0
    for τ in (TAU0, 2.0, 3.0), r in 0.0:0.25:RCMP
        T = Tana(τ, r)
        P, _, e = H.eos_Pne(T, ALPHA*T, EOS)
        d = abs(e/(3P) - 1)
        d > worst && (worst = d; Tw = T)
    end
    return worst, Tw
end

# ------------------------------------------------------------------ preliminary 2
"""FD residual of d_mu(s u^mu) = 0 on the ANALYTIC solution, relative to the size
of the individual terms. s ~ T^3 for a conformal fluid (any constant cancels)."""
function analytic_entropy_residual()
    h = 1e-4
    worst = 0.0
    for τ in (1.2, 2.0, 3.0), r in 0.5:0.25:RCMP
        s(τ_, r_) = Tana(τ_, r_)^3
        uτ(τ_, r_) = sqrt(1 + urana(τ_, r_)^2)
        f1(τ_) = τ_ * s(τ_, r) * uτ(τ_, r)
        f2(r_) = r_ * s(τ, r_) * urana(τ, r_)
        d1 = (f1(τ+h) - f1(τ-h))/(2h) / τ
        d2 = (f2(r+h) - f2(r-h))/(2h) / r
        scale = max(abs(d1), abs(d2))
        scale > 0 && (worst = max(worst, abs(d1 + d2)/scale))
    end
    return worst
end

# ------------------------------------------------------------------ the solver run
"""Evolve ideal Gubser on an (x,y) grid; return errors vs analytic at τf."""
function run_gubser(N, τf)
    g = H.make_grid2d(N, N; xmax = BOX, ymax = BOX)
    m = H.build_model_2d(; eos = EOS, enable_shear = false, enable_bulk = false,
                           enable_diff = false)
    L = m.layout
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        ur = urana(TAU0, r)
        ux = r > 0 ? ur*x/r : 0.0
        uy = r > 0 ? ur*y/r : 0.0
        H.set_cell!(U, H.lin(g, ix, iy), Tana(TAU0, r), ALPHA, ux, uy, TAU0, m)
    end

    ng = g.nghost
    # entropy at tau0, for the ideal-conservation check: tau * int s u^tau dx dy
    Sof(work) = begin
        acc = 0.0
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H.lin(g, ix, iy)
            hypot(g.xC[ix], g.yC[iy]) <= RCMP || continue
            T = exp(work.yT[i]); uτ = sqrt(1 + work.ux[i]^2 + work.uy[i]^2)
            acc += T^3 * uτ
        end
        acc * g.dx * g.dy
    end
    H.rhs_2d!(wk.k, U, g, TAU0, m, wk)
    S0 = TAU0 * Sof(wk)

    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05, work = wk)
    @assert res.ok
    H.update_primitives_2d!(U, g, res.τ, m, wk)
    τ = res.τ

    # ---- errors against the analytic solution, inside r <= RCMP ----
    sT = 0.0; sTa = 0.0; sU = 0.0; sUa = 0.0
    eTinf = 0.0; eUinf = 0.0; ncmp = 0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        r <= RCMP || continue
        i = H.lin(g, ix, iy); ncmp += 1
        Tn = exp(wk.yT[i]); Ta = Tana(τ, r)
        ur = urana(τ, r)
        uxa = r > 0 ? ur*x/r : 0.0; uya = r > 0 ? ur*y/r : 0.0
        sT += (Tn - Ta)^2; sTa += Ta^2
        sU += (wk.ux[i] - uxa)^2 + (wk.uy[i] - uya)^2
        sUa += uxa^2 + uya^2
        eTinf = max(eTinf, abs(Tn - Ta)/Ta)
        eUinf = max(eUinf, hypot(wk.ux[i]-uxa, wk.uy[i]-uya)/max(hypot(uxa,uya), 1e-3))
    end
    L2T = sqrt(sT/sTa); L2U = sqrt(sU/max(sUa, 1e-30))

    # ---- azimuthal symmetry: the solution is radially symmetric, so x<->y and
    # the spread of T around a ring are both properties the SCHEME must preserve
    asym = 0.0
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        i = H.lin(g, ng+1+k, ng+1+l); j = H.lin(g, ng+1+l, ng+1+k)
        hypot(g.xC[ng+1+k], g.yC[ng+1+l]) <= RCMP || continue
        a = exp(wk.yT[i]); b = exp(wk.yT[j])
        asym = max(asym, abs(a-b)/max(a, b))
    end
    # Grid imprinting: the spread AT FIXED RADIUS of T normalised by the analytic
    # T at that cell's own radius. Normalising is the whole point — binning raw T
    # by radius measures the genuine radial variation of T across the bin width,
    # which does not converge and is not an error. (First version of this gate did
    # exactly that and read a flat 0.11 at every resolution.)
    rings = Dict{Int,Vector{Float64}}()
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        r = hypot(g.xC[ix], g.yC[iy]); r <= RCMP || continue
        r > 0.5 || continue                       # skip the origin, where r-binning is meaningless
        k = round(Int, r/0.25)
        push!(get!(rings, k, Float64[]),
              exp(wk.yT[H.lin(g, ix, iy)])/Tana(τ, r))
    end
    ringspread = 0.0
    for (_, v) in rings
        length(v) >= 8 || continue
        ringspread = max(ringspread, maximum(v) - minimum(v))
    end

    # ENTROPY. tau * int_{r<=RCMP} s u^tau is NOT conserved: Gubser flow expands,
    # so entropy legitimately leaves the fixed disc, and the drift converges to a
    # physical -11.6%, not to zero. The meaningful statement is the error against
    # the ANALYTIC entropy in the same disc at the same tau.
    Sana = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        r = hypot(g.xC[ix], g.yC[iy]); r <= RCMP || continue
        Sana += Tana(τ, r)^3 * sqrt(1 + urana(τ, r)^2)
    end
    Sana *= τ * g.dx * g.dy
    S1 = τ * Sof(wk)
    dS = (S1 - S0)/S0                     # physical outflow, reported not asserted
    eS = (S1 - Sana)/Sana                 # the actual error
    @printf("  N=%3d dx=%.4f tau=%.1f | L2(T)=%.3e Linf(T)=%.3e | L2(u)=%.3e Linf(u)=%.3e | x<->y=%.2e ring=%.3e | S err=%+.3e (outflow %+.3f) | %5.1fs\n",
            N, g.dx, τ, L2T, eTinf, L2U, eUinf, asym, ringspread, eS, dS, time()-t0)
    return (; L2T, L2U, eTinf, eUinf, asym, ringspread, dS, eS, ncmp)
end

@testset "G1 — Gubser flow (analytic 2-D acceptance test)" begin
    # ---------- preliminary 1: is the EOS conformal? ----------
    cres, Tw = conformality_residual()
    @printf("  EOS conformality over the Gubser range: max|e/(3P)-1| = %.3e (at T=%.3f GeV)\n", cres, Tw)
    @test cres < 1e-6

    # ---------- preliminary 2: does the analytic solution solve the equations? ----------
    ares = analytic_entropy_residual()
    @printf("  analytic solution, FD residual of d_mu(s u^mu)=0: %.3e (relative)\n", ares)
    @test ares < 1e-6

    # ---------- the solver ----------
    println("  --- ideal Gubser, tau = 1 -> 2 fm/c ---")
    rs = [run_gubser(N, 2.0) for N in (100, 200, 400)]

    for r in rs
        @test r.ncmp > 100
        @test isfinite(r.L2T) && isfinite(r.L2U)
    end

    # accuracy at the finest resolution
    @test rs[end].L2T < 5e-3
    @test rs[end].L2U < 5e-2

    # CONVERGENCE ORDER. MUSCL + SSPRK2 is formally 2nd order on smooth data;
    # limiters clip extrema so the measured order is typically 1.5-2.
    oT = [log2(rs[k].L2T/rs[k+1].L2T) for k in 1:2]
    oU = [log2(rs[k].L2U/rs[k+1].L2U) for k in 1:2]
    @printf("  convergence order  T: %.2f, %.2f   u: %.2f, %.2f\n", oT[1], oT[2], oU[1], oU[2])
    @test oT[2] > 1.3
    @test oU[2] > 1.3

    # AZIMUTHAL SYMMETRY. x<->y must be at round-off: the grid and the solution
    # are both symmetric under that exchange, so anything else is the scheme.
    for r in rs
        @test r.asym < 1e-12
    end
    # Grid imprinting must CONVERGE AWAY, not merely be small.
    @printf("  ring spread of T/T_analytic: %.3e -> %.3e -> %.3e (must converge)\n",
            rs[1].ringspread, rs[2].ringspread, rs[3].ringspread)
    @test rs[3].ringspread < rs[1].ringspread
    @test rs[3].ringspread < 5e-3

    # ENTROPY. The drift out of the fixed disc is physical (~-11.6%, and the same
    # at every resolution, which is how you can tell). The error against the
    # analytic disc entropy is what must vanish, and it must CONVERGE.
    @printf("  entropy: physical outflow %+.3f (resolution-independent) | error vs analytic %+.3e -> %+.3e -> %+.3e\n",
            rs[3].dS, rs[1].eS, rs[2].eS, rs[3].eS)
    @test abs(rs[3].eS) < abs(rs[1].eS)
    @test abs(rs[3].eS) < 1e-3
end
