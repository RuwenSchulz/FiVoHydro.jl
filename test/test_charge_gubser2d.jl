# ==============================================================================
# test/test_charge_gubser2d.jl — GATE G3g: CHARGE TRANSPORT ON GUBSER FLOW.
#
# The charge sector's quantitative checks were a Fick-diffusion measurement on a
# STATIC UNIFORM background (2%) and an exact but one-dimensional reduction (D2).
# Neither exercises charge transport on a non-trivial 2-D flow. This does.
#
# ---------------------------------------------------------------------------
# THE EXACT STATEMENT
#
# With kappa = 0 the charge obeys d_mu (n u^mu) = 0. For IDEAL CONFORMAL flow the
# entropy obeys the same equation, and s = a T^3, so
#
#       n / T^3   is conserved along the flow, exactly.
#
# Seed it UNIFORM and it must stay uniform in space and constant in time, on a
# background with u^r reaching ~0.9c. That is a sharper statement than comparing
# n to a profile: it is a single number that must not move, so there is nothing
# to fit and no normalisation to get wrong.
#
# The seeding is closed-form. For the HQ Boltzmann tracer n(T,mu) = A(T) e^{mu/T}
# e^{-m/T}, so n(T,mu)/n(T,0) = e^{alpha} and
#
#       alpha = log( n_target / n(T, mu=0) )
#
# needs no knowledge of A(T) at all.
#
# 🔴 THE VACUUM CUT DESTROYS CHARGE, AND IT IS NOT SMALL HERE.
#
# With the production T_vac_cut = 0.05, Gubser's tail crosses that isotherm at
# r = 7.9 fm at tau = 2 - INSIDE the box - and every cell beyond it is zeroed,
# taking its charge. Measured, with everything else fixed:
#
#     T_vac_cut   0.05      0.02      0.01      0.005
#     dQ/Q        6.18e-3   4.01e-4   9.95e-5   9.95e-5
#     core inv    3.44e-3   3.44e-3   3.44e-3   3.44e-3
#
# The loss falls 62x and then saturates as the cut radius leaves the box, while
# the core invariant does not move at all - so the two effects separate cleanly,
# and the residual 1e-4 is genuine outflow. It is also why dQ was independent of
# BOX SIZE (10 vs 14 fm) and of resolution, which is what ruled out both boundary
# outflow and discretisation before this scan was run.
#
# Practically: charge is lost wherever the fluid is colder than T_vac_cut. On the
# production IC that is small (dQ ~ 8e-6) because its cold tail carries almost no
# charm, but it is not a property of the scheme - it is a property of how much
# charge sits below the cut.
#
# ⚠ This tests ADVECTION, not diffusion. With kappa > 0 the exact solution is
# destroyed: alpha = log(C T^3 / n(T,0)) is not spatially uniform (n(T,0) is not
# proportional to T^3), so grad(alpha) != 0 and the diffusion current acts. That
# is why the gate runs kappa = 0, and why the Fick test on a uniform background
# remains the diffusion check. Together they cover the two halves.
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

const QG   = 1.0
const TAU0 = 1.0
const TC0  = 0.6
# BOX = 14, not 10: Gubser expands to v ~ 0.9 and with outflow boundaries the
# charge that leaves the box is REAL, not an error. At +-10 fm it costs 6.1e-3 of
# the total by tau = 2, and being physical it is resolution-INDEPENDENT (measured
# 6.22 / 6.12 / 6.07 e-3 at N = 100 / 200 / 400), which is how you can tell.
const BOX  = 14.0
const RCMP = 3.0
const EOS  = H.ConformalHQEOS()
const TSCALE = hydro.gubser_Tscale_from_center_T(TAU0, QG, TC0)
const CN   = 1.0/TC0^3          # so n = 1 at the centre at tau0

Tana(τ, r)  = hydro.gubser_temperature(τ, r, QG, TSCALE)
urana(τ, r) = hydro.gubser_ur(τ, r, QG)
"""alpha that puts the density at n_target for this T."""
function alpha_for(T, ntarget)
    _, n0, _ = H.eos_Pne(T, 0.0, EOS)
    return log(ntarget/n0)
end

function run_charge(N, τf)
    g = H.make_grid2d(N, N; xmax = BOX, ymax = BOX)
    # kappa = 0: pure advection. T_vac_cut lowered to 0.01 so the test measures
    # TRANSPORT rather than the vacuum treatment - see the note below.
    m = H.build_model_2d(; eos = EOS, enable_shear = false, enable_bulk = false,
                           enable_diff = false, T_vac_cut = 0.01)
    L = m.layout
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        T = Tana(TAU0, r); ur = urana(TAU0, r)
        α = alpha_for(T, CN*T^3)
        H.set_cell!(U, H.lin(g, ix, iy), T, α, r>0 ? ur*x/r : 0.0, r>0 ? ur*y/r : 0.0,
                    TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    ng = g.nghost
    Q0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q0 += U[L.iDtau, H.lin(g, ix, iy)]
    end
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05, work = wk)
    @assert res.ok
    H.update_primitives_2d!(U, g, res.τ, m, wk)
    τ = res.τ

    # the invariant, and the density against the analytic profile
    worst = 0.0; sN = 0.0; sNa = 0.0; nc = 0; asym = 0.0; Q1 = 0.0
    # the same, restricted to the DENSE CORE: the invariant converges slowly at the
    # dilute edge, where alpha is large and the floors bite, and mixing the two
    # hides which is which.
    worst_c = 0.0; sNc = 0.0; sNac = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        Q1 += U[L.iDtau, i]
        r = hypot(g.xC[ix], g.yC[iy]); r <= RCMP || continue
        nc += 1
        T = exp(wk.yT[i]); nn = wk.n[i]
        worst = max(worst, abs(nn/(CN*T^3) - 1))
        na = CN*Tana(τ, r)^3
        sN += (nn - na)^2; sNa += na^2
        if r <= 2.0
            worst_c = max(worst_c, abs(nn/(CN*T^3) - 1))
            sNc += (nn - na)^2; sNac += na^2
        end
        j = H.lin(g, ng+1+(iy-ng-1), ng+1+(ix-ng-1))
        asym = max(asym, abs(nn - wk.n[j])/max(nn, wk.n[j]))
    end
    L2n = sqrt(sN/sNa); dQ = abs(Q1 - Q0)/abs(Q0)
    L2c = sqrt(sNc/sNac)
    umax = maximum(hypot(wk.ux[H.lin(g,ix,iy)], wk.uy[H.lin(g,ix,iy)])
                   for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny))
    @printf("    N=%3d dx=%.4f | r<=3: inv %.3e L2 %.3e | r<=2: inv %.3e L2 %.3e | x<->y %.1e | dQ %.2e | max|u| %.2f\n",
            N, g.dx, worst, L2n, worst_c, L2c, asym, dQ, umax)
    return (; worst, L2n, worst_c, L2c, asym, dQ, nc)
end

@testset "G3g — charge transport on Gubser flow" begin
    # the seeding identity: alpha = log(n_target/n(T,0)) must land exactly
    for T in (0.6, 0.35, 0.2)
        nt = CN*T^3
        α = alpha_for(T, nt)
        _, ngot, _ = H.eos_Pne(T, α*T, EOS)
        @test abs(ngot/nt - 1) < 1e-12
    end
    println("  seeding identity alpha = log(n_target/n(T,0)) exact to 1e-12")

    println("  --- ideal Gubser, charge advected, tau = 1 -> 2 ---")
    rs = [run_charge(N, 2.0) for N in (100, 200, 400)]
    for r in rs
        @test r.nc > 100
        @test r.asym < 1e-11          # the flow is symmetric; so must the charge be
    end
    # THE INVARIANT: n/T^3 must not move
    @test rs[end].worst_c < 5e-3
    oi = log2(rs[2].worst_c/rs[3].worst_c)
    oc = log2(rs[2].L2c/rs[3].L2c)
    @printf("  core (r<=2): invariant %.3e -> %.3e -> %.3e (order %.2f) ; L2(n) %.3e -> %.3e -> %.3e (order %.2f)\n",
            rs[1].worst_c, rs[2].worst_c, rs[3].worst_c, oi,
            rs[1].L2c, rs[2].L2c, rs[3].L2c, oc)
    @test rs[end].L2c < 5e-3
    # ⚠ NOT second order. The invariant converges at ~0.9 and L2 at ~0.4 toward a
    # floor near 2e-3; asserted as a bound plus a requirement that it still
    # improves, rather than claimed as an order it does not have.
    @test oc > 0.2
    @test rs[3].worst_c < rs[1].worst_c
    # charge lost through the outflow boundary is PHYSICAL - Gubser expands past
    # the box - so this is bounded, not asserted at round-off, and its
    # resolution-independence is what identifies it as outflow.
    @printf("  charge lost: %.2e %.2e %.2e (with T_vac_cut = 0.01; at the production 0.05 it is 6.2e-3)\n",
            rs[1].dQ, rs[2].dQ, rs[3].dQ)
    for r in rs; @test r.dQ < 3e-4; end
end
