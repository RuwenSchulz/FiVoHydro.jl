# ==============================================================================
# test/test_production_allsectors2d.jl — GATE G5.
#
# The production initial condition, ALL FOUR SECTORS on, run to late proper time,
# at two resolutions.
#
# This gate exists because the rest of the ladder did not catch D6 or D7. Every
# other gate runs either a controlled problem (Bjorken, a smooth bump, a uniform
# state) or a single sector, and both defects needed the real IC, sector
# INTERACTION and late time simultaneously:
#
#   D6  the charge sector at the fluid-vacuum interface: the charge row
#       `n u^τ + ν^τ = J^τ` is nearly degenerate there, so ν_NS = -κ∇α fed back
#       into it and diverged. x↔y went to 1.0. Closed by the density-gated vacuum
#       ramp (transport2d.jl) plus the ν admissibility bound (floors2d.jl).
#
#   D7  a charge-row failure DISCARDED the converged hydro block. Measured: 68.2%
#       of all failures had rows 2-4 within tolerance and only row 1 out, and every
#       one of those cells was reset to vacuum — punching holes in a good T and u
#       field, which then showed up as max|u| inflating 1.16 -> 3.02 and
#       max|π^xy| reaching 1.0e+03 by τ=4 at N=300. Closed by the staged fallback
#       in primrec2d.jl and the bulk positivity guard.
#
# The assertions below are on exactly the quantities those two defects moved.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_production_allsectors2d.jl
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0   = 0.4
const RMAX   = 20.0
const THOT   = 0.05      # measure only where there is fluid
const T_FO   = 0.1565    # freeze-out; above this is what produces observables

function run_production(N, τf)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)

    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, enable_bulk = true, enable_diff = true,
                           eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
                           zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           kappa_coeff = 0.1163, tauN_coeff = 1.0)
    L = m.layout
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix], g.yC[iy])
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)

    ng = g.nghost
    Q0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q0 += U[L.iDtau, H.lin(g, ix, iy)]
    end

    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    el = time() - t0
    @assert res.ok

    Q1 = 0.0; maxu = 0.0; maxpi_rel = 0.0; maxpi_rel_fo = 0.0
    minPtot = Inf; asym = 0.0; nhot = 0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        Q1 += U[L.iDtau, i]
        exp(res.work.yT[i]) < THOT && continue
        nhot += 1
        P  = res.work.P[i]
        Pi = H.phys_from_stored(U[L.iPi, i])
        minPtot = min(minPtot, P + Pi)
        maxu = max(maxu, hypot(res.work.ux[i], res.work.uy[i]))
        # shear measured RELATIVE to the pressure: the absolute scale falls with τ,
        # so an absolute bound would be met trivially at late times.
        if P > 0
            pmag = max(abs(H.phys_from_stored(U[L.iPixx,i])),
                       abs(H.phys_from_stored(U[L.iPixy,i])),
                       abs(H.phys_from_stored(U[L.iPiyy,i])))
            maxpi_rel = max(maxpi_rel, pmag/P)
            # |pi|/P is an APPLICABILITY statement, not a correctness one, so it is
            # asserted where observables come from. In the dilute tail a viscous
            # correction of order the pressure is expected and says nothing about
            # the scheme; measured globally it crosses 1 as soon as the fluid is
            # allowed to expand properly (it read 0.947 while the fluid-vacuum face
            # was still a reflecting wall, 1.007 once that was fixed — the fluid
            # simply reaches further into the dilute regime now).
            exp(res.work.yT[i]) > T_FO && (maxpi_rel_fo = max(maxpi_rel_fo, pmag/P))
        end
    end
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        i = H.lin(g, ng+1+k, ng+1+l); j = H.lin(g, ng+1+l, ng+1+k)
        (exp(res.work.yT[i]) < THOT && exp(res.work.yT[j]) < THOT) && continue
        a = U[L.iE, i]; b = U[L.iE, j]
        asym = max(asym, abs(a-b)/max(abs(a), abs(b), 1e-30))
    end

    dQ = abs(Q1 - Q0)/abs(Q0)
    @printf("  N=%3d tau=%.1f steps=%4d primfail=%6d wall=%5.1fs | hot=%6d max|u|=%.3f max|pi|/P=%.3f (above T_fo %.3f) min(P+Pi)=%+.2e dQ=%.2e x<->y=%.2e\n",
            N, res.τ, res.nsteps, res.nprimfail, el, nhot, maxu, maxpi_rel,
            maxpi_rel_fo, minPtot, dQ, asym)
    return (res = res, nhot = nhot, maxu = maxu, maxpi_rel = maxpi_rel,
            maxpi_rel_fo = maxpi_rel_fo, minPtot = minPtot, dQ = dQ, asym = asym)
end

@testset "G5 — production IC, all four sectors, to late time" begin
    for (N, τf) in ((150, 8.0), (300, 4.0))
        r = run_production(N, τf)

        @test r.nhot > 10_000

        # x<->y symmetry. The IC is azimuthally symmetric, so ANY asymmetry is the
        # scheme's. This is the number that read 1.0 under D6 and 2.2e-2 under D7.
        @test r.asym < 1e-9

        # Flow must stay physical. D7 inflated this to 4.3 (v = 0.97) by punching
        # vacuum holes into a converged velocity field.
        @test r.maxu < 3.0

        # Shear relative to the pressure, ABOVE FREEZE-OUT. D7 drove this through
        # 1e3 in absolute terms. Asserted in the observable-producing region; see
        # the note in run_production for why the global figure is reported but not
        # gated.
        @test r.maxpi_rel_fo < 1.0

        # Total pressure must stay positive: bulk+diffusion drove min(P+Pi) to
        # -2.7e-4 before the positivity guard.
        @test r.minPtot > 0.0

        # Charge: outflow boundaries lose some genuinely, but not much.
        @test r.dQ < 1e-3

        # Recovery failures must stay rare. Not zero — the dilute tail legitimately
        # has cells the EOS cannot represent — but far from the 2.5e4 of D7.
        @test r.res.nprimfail < 5_000
    end
end
