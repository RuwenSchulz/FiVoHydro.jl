# ==============================================================================
# test/test_elliptic2d.jl — GATE G6: a genuinely NON-AXISYMMETRIC initial condition.
#
# Every other gate in the ladder runs an azimuthally symmetric problem, because
# that is the only case with an independent trusted answer (the 1-D solver). But
# azimuthal structure is the entire point of a 2+1D code, so it has to be
# exercised even though nothing can be compared against.
#
# ---------------------------------------------------------------------------
# WHAT CAN AND CANNOT BE CHECKED HERE
#
# There is NO reference solution for a deformed IC anywhere in this repo — that
# is precisely why the 2-D solver is being built. So this gate cannot check
# correctness against an oracle. It checks the four things that ARE decidable:
#
#   1. the deformation is what produces the anisotropy — the ε₂ = 0 control must
#      give identically zero, so the scheme is not manufacturing it;
#   2. the anisotropy CONVERGES with resolution, so it is a property of the
#      equations and not of the grid;
#   3. reflection in x and in y are exact symmetries of this IC, so any breaking
#      is the scheme's;
#   4. conservation and admissibility hold as they do on the symmetric problem.
#
# The IC is the production radial profile evaluated on an area-preserving
# elliptical radius, so it inherits the real profile's steep gradient and vacuum
# tail — the features that exposed D6 and D7 — while adding ε₂ ≠ 0.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_elliptic2d.jl
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
const TAUF   = 6.0
const RMAX   = 20.0
const THOT   = 0.05
const T_FO   = 0.1565      # freeze-out temperature; above this is what produces observables

function run_elliptic(N; eps2 = 0.25, τf = TAUF)
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

    a = sqrt(1 - eps2); b = sqrt(1 + eps2)          # area-preserving
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix]/a, g.yC[iy]/b)
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)

    ng = g.nghost
    Q0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q0 += U[L.iDtau, H.lin(g, ix, iy)]
    end

    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok

    Q1 = 0.0; sx = 0.0; sy = 0.0; maxu = 0.0; minPtot = Inf
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        Q1 += U[L.iDtau, i]
        exp(res.work.yT[i]) < THOT && continue
        w = U[L.iE, i]                                # energy-weighted, as v₂ is
        sx += w*res.work.ux[i]^2
        sy += w*res.work.uy[i]^2
        maxu = max(maxu, hypot(res.work.ux[i], res.work.uy[i]))
        minPtot = min(minPtot, res.work.P[i] + H.phys_from_stored(U[L.iPi,i]))
    end

    # Reflection symmetry, measured TWICE: globally, and restricted to the region
    # that produces observables (T > T_fo). The distinction is not cosmetic —
    # measured at N=300, eps2=0.25 the global figure is 1.8e-3 while ABOVE
    # FREEZE-OUT it is 5.3e-15. Every cell contributing to the global number sits
    # at r in [18.6, 24.7] with T in [0.067, 0.116], i.e. in the box CORNERS
    # (outside r = rmax, a region the 1-D grid does not have) and below freeze-out.
    # Gating on the global number would be gating on cells nobody uses; reporting
    # only the restricted one would hide a real feature of the scheme. So: assert
    # on the physics region, report both.
    refx = 0.0; refy = 0.0; refx_fo = 0.0; refy_fo = 0.0
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        i  = H.lin(g, ng+1+k,        ng+1+l)
        jx = H.lin(g, ng+g.Nx-k,     ng+1+l)
        jy = H.lin(g, ng+1+k,        ng+g.Ny-l)
        e = U[L.iE, i]
        e < 1e-12 && continue
        dx = abs(e - U[L.iE,jx])/max(abs(e), 1e-30)
        dy = abs(e - U[L.iE,jy])/max(abs(e), 1e-30)
        refx = max(refx, dx); refy = max(refy, dy)
        if exp(res.work.yT[i]) > T_FO
            refx_fo = max(refx_fo, dx); refy_fo = max(refy_fo, dy)
        end
    end

    aniso = (sx + sy) > 0 ? (sx - sy)/(sx + sy) : 0.0
    dQ = abs(Q1 - Q0)/abs(Q0)
    @printf("  N=%3d eps2=%.2f | pf=%6d max|u|=%.3f | anisotropy=%+.5f | dQ=%.2e | refl global %.2e/%.2e  above T_fo %.2e/%.2e\n",
            N, eps2, res.nprimfail, maxu, aniso, dQ, refx, refy, refx_fo, refy_fo)
    return (aniso = aniso, dQ = dQ, refx = refx, refy = refy,
            refx_fo = refx_fo, refy_fo = refy_fo,
            maxu = maxu, minPtot = minPtot, res = res)
end

@testset "G6 — non-axisymmetric IC" begin

    @testset "the deformation produces the anisotropy, not the scheme" begin
        c = run_elliptic(200; eps2 = 0.0)
        @printf("  eps2 = 0 control: anisotropy = %+.3e (must be identically zero)\n", c.aniso)
        @test abs(c.aniso) < 1e-10
        @test c.refx_fo < 1e-11
        @test c.refy_fo < 1e-11
    end

    @testset "anisotropy converges with resolution" begin
        # No oracle exists for this problem, so resolution-independence is the
        # substitute: a grid artefact would not converge.
        a = run_elliptic(150)
        b = run_elliptic(300)
        d1 = abs(b.aniso - a.aniso)/abs(a.aniso)
        @printf("  anisotropy %.5f -> %.5f  (change %.2f%%)\n", a.aniso, b.aniso, 100*d1)
        @test a.aniso > 0.15            # an oblate IC must give a positive, sizeable signal
        @test d1 < 0.02                 # converged to better than 2%

        for r in (a, b)
            @test r.res.nprimfail < 5_000
            @test r.maxu < 3.0
            @test r.minPtot > 0.0
            @test r.dQ < 1e-3
            # Reflection in x and y are exact symmetries of this IC. Asserted in
            # the observable-producing region; see the note in run_elliptic.
            @test r.refx_fo < 1e-11
            @test r.refy_fo < 1e-11
        end
    end
end
