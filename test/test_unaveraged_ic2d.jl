# ==============================================================================
# test/test_unaveraged_ic2d.jl — GATE G8: the UN-AVERAGED production IC (F1).
#
# Every other gate runs either the azimuthally symmetric production profile or a
# synthetic deformation. This one runs the real thing: the Pb+Pb 0-5% initial
# condition built WITHOUT the φ-average, by
# `Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl`.
#
# That builder keeps the binary-collision midpoint VECTORS which
# `MCGCollisionDensity.jl` reduces to radii before depositing `azimuthal_gauss`.
# The φ-average in that one line is where the azimuthal information was being
# destroyed — upstream of every solver, which is why
# `data/initial_profiles_physical.csv` is `r, T0, alpha0`.
#
# The IC is a GENERATED artefact. If it is absent this gate skips rather than
# fails; regenerate with
#     julia --project=Julia Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl
#
# ---------------------------------------------------------------------------
# WHAT IS CHECKED
#
# There is still no oracle for a deformed IC, so this gate checks that the IC is
# faithfully inherited and that the response is physical:
#
#   * the seeded state carries a real ε₂ (the whole point of F1);
#   * momentum anisotropy is positive and BUILDS WITH TIME — spatial eccentricity
#     converting to momentum anisotropy is the defining hydrodynamic response, and
#     a solver that produced it instantly, or not at all, would be wrong;
#   * it is resolution-stable;
#   * conservation and admissibility hold as on every other IC.
#
# ⚠ The IC's NORMALISATION is inherited from the 1-D calibration via a
# density -> T map, not re-derived in 2-D (BuildIC2D.jl header). It reproduces the
# 1-D charm count to 1.0% (23.792 pairs against the 1-D build's 24.038) and the
# central temperature to 1.1% (T_max 0.5877 against 0.5814) -- corrected 2026-09-04
# from "0.2%" and "exactly", neither of which the builder has ever printed. The
# pion yield of a 2-D run built this way has NOT been checked against ALICE.
#
# ---------------------------------------------------------------------------
# `b.dQ` RE-BASELINED 1e-4 -> 5e-4 on 2026-09-05, after the analytic-alpha IC landed.
# The reasoning and the bound's physical meaning are at the assertion itself; the
# short version is below. Resolved deliberately, not by nudging until green.
#
# ALICE_IC_Creation replaced a tabulated charm A(T) — read from a stale copy of
# FiVoHydro's own 1-D IC — with the CLOSED FORM of this package's charm EOS. The
# new IC is strictly more correct: integrating `eos_Pne` over it returns 23.7924
# charm pairs against the 23.7924 the IC is built to carry (+0.00%), where the
# table version returned 24.1485 (+1.50%).
#
# The drift it costs is NOT a defect in that IC. Demonstrated, not inferred — the
# same IC with the charm stripped from every cell below T_fo (0.011176 pairs,
# 9493 cells, T untouched):
#
#     IC                                  dQ (N=150, tau=8)     gate
#     table alpha (pre-2026-09-04)             5.64e-06         19/19
#     analytic alpha, faithful                 1.15e-04         18/19
#     analytic alpha, sub-T_fo charm stripped  6.49e-06         19/19
#
# So all of the excess comes from charm the IC now places BELOW FREEZE-OUT, in
# dilute cells (T ~ 0.07, above `T_vac_cut` so they are evolved, far below T_fo so
# hydro has no business describing them) where the charge sector leaks it. The old
# IC passed because its A(T) was wrong out there and under-represented that charm
# by ~96% in the 9-10 fm annulus.
#
# CHOSEN: keep the IC faithful (it matches the 1-D pipeline, which also carries charm
# below T_fo, and cutting it would break the exact-N_charm anchoring), and set this
# gate's bound to the charm the IC seeds below freeze-out — 4.70e-4 of the total —
# because that is precisely the matter the cold-edge machinery deletes by design.
# ---------------------------------------------------------------------------
#
# The IC this gate reads was REBUILT on 2026-09-04 after two fixes in
# ALICE_IC_Creation: a half-bin binning bias in `azimuthal_average` and an
# ill-conditioned exponent window in `loglog_extended_map`. The collision field and
# its eccentricities are bit-identical to the previous version -- the deposit never
# moved -- but T is now up to 16% lower at the fireball edge, so the freeze-out
# geometry a run on this IC sees has genuinely changed. This gate passes 19/19 on
# it; G9's regulator-inertness block does NOT (see test_fluctuating_ic2d.jl).
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const IC2D = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation",
                               "PbPb", "data", "ic2d_00-05.csv"))
const TAU0 = 0.4
const THOT = 0.05

function run_unaveraged(N, τf)
    g = H.make_grid2d(N, N; xmax = 20.0, ymax = 20.0)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0)
    L = m.layout
    U = H.allocate_state(g, m)
    ic = H.initialize_from_grid_csv!(U, g, m, TAU0, IC2D)
    ng = g.nghost

    # spatial eccentricity of the SEEDED energy density, before evolution
    sc = 0.0; ss = 0.0; sr = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy); x = g.xC[ix]; y = g.yC[iy]
        r = hypot(x, y); r < 1e-9 && continue
        w = U[L.iE,i]*r^2; φ = atan(y, x)
        sc += w*cos(2φ); ss += w*sin(2φ); sr += w
    end
    eps2_ic = sr > 0 ? hypot(sc, ss)/sr : 0.0

    Q0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q0 += U[L.iDtau, H.lin(g, ix, iy)]
    end

    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok

    Q1 = 0.0; sx = 0.0; sy = 0.0; maxu = 0.0; minPtot = Inf
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy); Q1 += U[L.iDtau, i]
        exp(res.work.yT[i]) < THOT && continue
        w = U[L.iE,i]
        sx += w*res.work.ux[i]^2; sy += w*res.work.uy[i]^2
        maxu = max(maxu, hypot(res.work.ux[i], res.work.uy[i]))
        minPtot = min(minPtot, res.work.P[i] + H.phys_from_stored(U[L.iPi,i]))
    end
    ep = (sx + sy) > 0 ? (sx - sy)/(sx + sy) : 0.0
    dQ = abs(Q1 - Q0)/abs(Q0)

    @printf("  N=%3d tau=%.1f | ic bad=%d vacuum=%d | pf=%6d | T(0)=%.4f max|u|=%.3f | eps2(IC)=%.4f -> p-anisotropy=%+.5f (response %.3f) | dQ=%.2e\n",
            N, res.τ, ic.nbad, ic.nvacuum, res.nprimfail,
            exp(res.work.yT[H.lin(g, ng+g.Nx÷2, ng+g.Ny÷2)]), maxu,
            eps2_ic, ep, ep/max(eps2_ic, 1e-12), dQ)
    return (; eps2_ic, ep, dQ, maxu, minPtot, ic, res)
end

@testset "G8 — the un-averaged production IC" begin
    if !isfile(IC2D)
        @info "skipping G8: $(IC2D) absent. Regenerate with BuildIC2D.jl."
        @test_skip false
    else
        a = run_unaveraged(150, 2.0)
        b = run_unaveraged(150, 8.0)
        c = run_unaveraged(250, 8.0)

        # the IC is faithfully inherited: no cell with matter failed to seed
        for r in (a, b, c)
            @test r.ic.nbad == 0
            @test r.res.nprimfail < 10_000
            @test r.maxu < 3.0
            @test r.minPtot > 0.0
        end

        # ---- charge: a MEASURED TRADE-OFF, not a free tolerance ----
        # The vacuum cut and the density-gated ramp both delete charge at the cold
        # edge, and the effect grows with resolution and time: at N=250, tau=8 the
        # drift is ~3e-3 against ~5e-6 at N=150. Loosening the thresholds fixes it
        # (2.1e-6 with T_vac_cut=0.02, n_lo=1e-9) but brings D6 straight back —
        # measured, the elliptic IC then returns x<->y = 1.00 instead of 1e-13.
        # So the loss is the PRICE of a stable charge sector. Assert tightly where
        # the edge is well away from the fireball, loosely where it is not, and
        # leave this comment so nobody "fixes" it by loosening the thresholds.
        # 2026-09-05: `b` re-baselined 1e-4 -> 5e-4, DELIBERATELY, and the bound is a
        # physical statement rather than the measured number plus headroom.
        #
        # The comment above already had this mechanism right — the cold edge deletes
        # charge and that is the price of a stable charge sector. What changed is the
        # INPUT: ALICE_IC_Creation's analytic-alpha IC (2026-09-04) carries exactly the
        # charm it should (+0.00% against 23.7924 pairs, where the retired table-alpha
        # version carried +1.50%), and part of getting that right is representing the
        # n_hard tail past the fireball, which the old A(T) under-represented by ~96%
        # in the 9-10 fm annulus. More charm at the cold edge, more for the edge to
        # delete: dQ went 5.64e-06 -> 1.15e-04 at N=150, tau=8.
        #
        # Demonstrated, not inferred: the SAME IC with the charm stripped from every
        # cell below T_fo (0.011176 pairs, 9493 cells, T untouched) returns dQ =
        # 6.49e-06 and this gate to 19/19. All of the excess is sub-freeze-out charm.
        #
        # So the bound is: the drift may not exceed the charm the IC seeds BELOW
        # FREEZE-OUT, which is 0.011176 / 23.7924 = 4.70e-4 of the total and which the
        # cold-edge machinery deletes by design. 5e-4 is that number. A drift above it
        # would be eating charm from matter hydro can actually describe, and that would
        # be a real defect — which is what this test is for.
        #
        # The IC is deliberately NOT cut at T_fo: it matches the 1-D pipeline, which
        # also carries charm below freeze-out, and cutting it would break the exact
        # N_charm anchoring the whole normalisation chain rests on.
        @test a.dQ < 1e-4          # N=150, tau=2 — edge still far from the fireball
        @test b.dQ < 5e-4          # N=150, tau=8 — see above; measured 1.15e-4
        @test c.dQ < 2e-2          # N=250, tau=8 — edge-loss dominated
        @printf("  charge drift: N=150 tau=2 %.2e | N=150 tau=8 %.2e | N=250 tau=8 %.2e\n",
                a.dQ, b.dQ, c.dQ)

        # it carries a REAL deformation — this is what F1 bought
        @test a.eps2_ic > 0.05

        # the hydrodynamic response: spatial eccentricity converts to momentum
        # anisotropy, and it BUILDS with time rather than appearing instantly
        @test a.ep > 0.0
        @test b.ep > a.ep

        # resolution stability of the response
        @printf("  anisotropy at tau=8: N=150 %.5f, N=250 %.5f (%.2f%%)\n",
                b.ep, c.ep, 100*abs(c.ep - b.ep)/b.ep)
        @test abs(c.ep - b.ep)/b.ep < 0.05
    end
end
