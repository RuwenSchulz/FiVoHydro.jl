# ==============================================================================
# test/test_fluctuating_ic2d.jl — GATE G9: a SINGLE-EVENT (fluctuating) IC.
#
# Every other gate in the ladder runs a SMOOTH initial condition: Bjorken, a
# synthetic bump, a synthetic ellipse, or a production IC that has been averaged
# over 400-800 collisions. Averaging events after rotating each into its own
# participant plane preserves eps2 by construction and destroys everything else —
# the ensemble ICs measure eps3 = 0.001-0.007, i.e. zero.
#
# So nothing in the ladder had ever run a LUMPY field, and the solver did not
# survive one. That is what this gate is for.
#
# ---------------------------------------------------------------------------
# WHAT WENT WRONG (measured 2026-09-01, TWOD_PROGRAM.md §6j)
#
# A single event's hot spots are set by the sub-nucleon width W = 0.5 fm. The
# shear sector could not resolve their gradients: |pi|/P ran past 1 and then to
# 1e7-1e12, P + Pi went negative, primitive recovery began failing in bulk and the
# solve cascaded — max|u| ~ 20-58 (v -> 0.9995) and 16-60% of the charge lost.
#
# Two things made it worse than a loud crash:
#   * `run_sim_2d!` returned ok = true throughout. The run "succeeded" while
#     losing a third of the conserved charge.
#   * throwing resolution at it does NOT fix it. Measured on ev02, clip off:
#     max|u| = 28.1 at N = 200 and 32.7 at N = 300, dQ = 0.39 and 0.54, with 729k
#     and 1.17M recovery failures, and the v_n it reports differ at every
#     resolution — not a measurement of anything.
#
# THE FIX IS THE SHEAR REGULATOR, `pi_clip_factor` (dissipation2d.jl), which caps
# the evolved pi component-wise at f|P|. It is off by default, inherited from the
# 1-D solver whose smooth radial profile never needs it. At f = 1:
#     v3 = 0.14651 / 0.14473 / 0.14402 at N = 200 / 300 / 400  (converged to 1.2%)
#     max|u| = 1.39, dQ = 3.1e-6, min(P+Pi) = +0.038, max|pi|/P = 0.41
# and v3 does not depend on f: 0.14651 / 0.14605 / 0.14605 / 0.14605 at
# f = 1 / 2 / 5 / 10. Any finite cap stops the runaway; the answer is the same.
# and it is 3.3-10x FASTER, because a diverging run spends its time on MOOD
# escalation retries.
#
# It is also INERT where it is not needed: on the smooth ensemble IC, clip on vs
# off agrees to every printed digit at both N = 200 and N = 300. That inertness is
# asserted below, because it is the whole licence for using the regulator at all.
#
# The ICs are GENERATED artefacts; if absent this gate skips. Regenerate with
#     julia --project=Julia Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl events 20-30 6
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const DATA  = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation", "PbPb", "data"))
const EVENT = joinpath(DATA, "ic2d_20-30_ev02.csv")
const ENSEM = joinpath(DATA, "ic2d_20-30.csv")
const TAU0  = 0.4
const T_FO  = 0.1565

"""Run one IC and return the admissibility set plus (eps_n, v_n)."""
function run_fluct(ic, N, clip; τf = 8.0)
    g = H.make_grid2d(N, N; xmax = 16.0, ymax = 16.0)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0,
        pi_clip_factor = clip)
    L = m.layout; U = H.allocate_state(g, m); wk = H.make_work(g, m)
    icr = H.initialize_from_grid_csv!(U, g, m, TAU0, ic)
    ng = g.nghost
    idx = [H.lin(g, ix, iy) for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    xs  = [g.xC[ix] for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    ys  = [g.yC[iy] for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]

    # eps_n of the SEEDED energy density, the standard r^n-weighted definition
    function eps_n(n)
        sc = 0.0; ss = 0.0; sw = 0.0
        for (k, i) in enumerate(idx)
            r = hypot(xs[k], ys[k]); r < 1e-9 && continue
            w = U[L.iE, i]*r^n; φ = atan(ys[k], xs[k])
            sc += w*cos(n*φ); ss += w*sin(n*φ); sw += w
        end
        return sw > 0 ? hypot(sc, ss)/sw : 0.0
    end
    e2 = eps_n(2); e3 = eps_n(3)
    Q0 = sum(U[L.iDtau, i] for i in idx)

    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05, work = wk)
    el = time() - t0
    H.update_primitives_2d!(U, g, res.τ, m, wk)

    # v_n from the transverse momentum density above freeze-out
    maxu = 0.0; minP = Inf; mpi = 0.0
    sc2 = 0.0; ss2 = 0.0; sc3 = 0.0; ss3 = 0.0; sw = 0.0
    for i in idx
        exp(wk.yT[i]) > T_FO || continue
        maxu = max(maxu, hypot(wk.ux[i], wk.uy[i]))
        minP = min(minP, wk.P[i] + H.phys_from_stored(U[L.iPi, i]))
        if wk.P[i] > 0
            mpi = max(mpi, max(abs(H.phys_from_stored(U[L.iPixx,i])),
                               abs(H.phys_from_stored(U[L.iPixy,i])),
                               abs(H.phys_from_stored(U[L.iPiyy,i])))/wk.P[i])
        end
        sx = U[L.iSx,i]; sy = U[L.iSy,i]; w = hypot(sx, sy); w > 0 || continue
        φ = atan(sy, sx)
        sc2 += w*cos(2φ); ss2 += w*sin(2φ); sc3 += w*cos(3φ); ss3 += w*sin(3φ); sw += w
    end
    v2 = sw > 0 ? hypot(sc2, ss2)/sw : 0.0
    v3 = sw > 0 ? hypot(sc3, ss3)/sw : 0.0
    dQ = abs(sum(U[L.iDtau, i] for i in idx) - Q0)/abs(Q0)

    @printf("  %-9s N=%3d clip=%-3s %5.1fs | eps2=%.4f eps3=%.4f -> v2=%.4f v3=%.4f | max|u|=%6.2f dQ=%8.1e min(P+Pi)=%+9.2e max|pi|/P=%9.3g pf=%6d\n",
            basename(ic)[6:end-4], N, clip > 0 ? "on" : "off", el,
            e2, e3, v2, v3, maxu, dQ, minP, mpi, res.nprimfail)
    return (; e2, e3, v2, v3, maxu, dQ, minP, mpi, nbad = icr.nbad,
              pf = res.nprimfail, el)
end

@testset "G9 — a single-event (fluctuating) IC" begin
    if !isfile(EVENT) || !isfile(ENSEM)
        @info "skipping G9: $(EVENT) absent. Regenerate with BuildIC2D.jl events 20-30 6."
        @test_skip false
    else
        ev200 = run_fluct(EVENT, 200, 1.0)
        ev300 = run_fluct(EVENT, 300, 1.0)
        en_on  = run_fluct(ENSEM, 200, 1.0)
        en_off = run_fluct(ENSEM, 200, -1.0)

        # ---- REPRODUCIBILITY: this IC must be the event it claims to be ----
        # `MCGCollisionDensity2D` samples nucleus configurations from a stateful
        # Metropolis-Hastings chain whose FIRST draw was taking randomness from the
        # global RNG rather than from the seeded `rng`. One unstable draw out of
        # 8000 events was enough to move that event's rank in the multiplicity sort
        # and so change WHICH collision a centrality window selects: two runs of
        # the identical build command picked event 2044 (N_coll 734) and event 2992
        # (N_coll 891). Five of six events reproduced, so it read as a one-off.
        #
        # Fixed by seeding the global RNG and burning the chains' first draw
        # (MCGCollisionDensity2D.jl). This pins it: if a regeneration picks a
        # different collision, the numbers below describe a different event and the
        # gate must fail rather than quietly re-measure something else.
        meta = replace(EVENT, ".csv" => "_meta.txt")
        if isfile(meta)
            sel = ""
            for ln in eachline(meta)
                startswith(ln, "selected_events") && (sel = strip(split(ln, "=", limit=2)[2]))
            end
            @printf("  provenance: selected_events = %s (expected [2992])\n", sel)
            @test sel == "[2992]"
        else
            @info "no meta file beside $(EVENT); provenance not checked"
        end

        # ---- the IC really does carry a fluctuation ----
        # eps2 survives ensemble averaging; eps3 does not. This is what separates a
        # single event from everything else in the ladder.
        @test ev200.nbad == 0
        @test ev200.e3 > 0.05

        # ---- admissibility, the quantities the divergence moved ----
        for (r, N) in ((ev200, 200), (ev300, 300))
            @test r.maxu < 3.0            # was 28.1 / 32.7 unregulated
            @test r.minP > 0.0            # was negative
            @test r.mpi < 2.0             # was 1e7-1e12
            # 🔑 PER CELL, not an absolute count. `pf` accumulates cell-failures over
            # steps, so an absolute bound applied at two resolutions is stricter at
            # the finer one for no physical reason. Measured, the RATE is flat:
            # 8294/200^2 = 0.207 and 17528/300^2 = 0.195 -- the same fixed cold-tail
            # region, counted on more cells. The old `pf < 10_000` passed N=200 and
            # failed N=300 on exactly that arithmetic.
            @test r.pf/N^2 < 0.35
        end

        # ---- charge conservation ----
        # 2026-09-03, RE-BASELINED, and the cause is measured rather than assumed.
        # The bound was 1e-4, set when tau_n was a factor g_hq = 6 too short (D8).
        # With the corrected tau_n the current persists ~6x further into the cold
        # edge before decaying, and TWOD_PROGRAM.md 6v established that the vacuum
        # cut is a CHARGE SINK -- charge is lost wherever the fluid is colder than
        # T_vac_cut. So more charge reaches the sink. Attributed three ways on the
        # ensemble IC (N=200, tau=8): old tau_n 1.97e-5, corrected tau_n 5.21e-4,
        # corrected + the D9 relaxation ramp 2.36e-4 -- i.e. D9 HALVES it and the
        # residual is what the corrected physics costs.
        #
        # It is a knob, not a defect: T_vac_cut 0.05 -> 0.02 -> 0.01 gives
        # dQ 2.4e-4 -> 6.4e-10 -> 2.6e-12, at the price of primfail 2497 -> 16860
        # -> 256927 as far more cold cells become "fluid". Which side of that trade
        # production sits on is a physics decision and is NOT made here.
        #
        # 1e-3 is ~4x the measured 2.7e-4: a regression guard, not a restatement.
        for r in (ev200, ev300)
            @test r.dQ < 1e-3
        end

        # ---- v3 is a MEASUREMENT, i.e. resolution-converged ----
        @printf("  v3 convergence: N=200 %.5f, N=300 %.5f (%.2f%%)\n",
                ev200.v3, ev300.v3, 100*abs(ev300.v3 - ev200.v3)/ev200.v3)
        @test abs(ev300.v3 - ev200.v3)/ev200.v3 < 0.02
        @test abs(ev300.v2 - ev200.v2)/ev200.v2 < 0.02

        # ---- triangular flow is a FLUCTUATION observable ----
        # The ensemble IC has eps3 ~ 0 and must produce ~no v3; the single event
        # must produce an order of magnitude more. A solver that gave the ensemble
        # a large v3 would be manufacturing it.
        @printf("  v3: single event %.5f vs ensemble %.5f (ratio %.1fx)\n",
                ev200.v3, en_on.v3, ev200.v3/max(en_on.v3, 1e-12))
        @test en_on.e3 < 0.02
        @test en_on.v3 < 0.03
        @test ev200.v3 > 0.10
        @test ev200.v3 > 5*en_on.v3

        # ---- the regulator must be INERT where it is not needed ----
        # This is the licence for switching it on at all: on a smooth IC it may not
        # move the answer. Measured, it agrees to every printed digit.
        @printf("  regulator inertness on the smooth ensemble IC: dv2=%.2e dv3=%.2e dmax|u|=%.2e\n",
                abs(en_on.v2 - en_off.v2), abs(en_on.v3 - en_off.v3),
                abs(en_on.maxu - en_off.maxu))
        @test abs(en_on.v2 - en_off.v2) < 1e-4
        @test abs(en_on.v3 - en_off.v3) < 1e-4
        @test abs(en_on.maxu - en_off.maxu) < 1e-3
        @test abs(en_on.mpi - en_off.mpi) < 1e-3
    end
end
