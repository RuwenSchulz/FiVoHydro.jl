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

    refx = 0.0; refy = 0.0
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        i  = H.lin(g, ng+1+k,        ng+1+l)
        jx = H.lin(g, ng+g.Nx-k,     ng+1+l)
        jy = H.lin(g, ng+1+k,        ng+g.Ny-l)
        e = U[L.iE, i]
        e < 1e-12 && continue
        refx = max(refx, abs(e - U[L.iE,jx])/max(abs(e), 1e-30))
        refy = max(refy, abs(e - U[L.iE,jy])/max(abs(e), 1e-30))
    end

    aniso = (sx + sy) > 0 ? (sx - sy)/(sx + sy) : 0.0
    dQ = abs(Q1 - Q0)/abs(Q0)
    @printf("  N=%3d eps2=%.2f | pf=%6d max|u|=%.3f | anisotropy=%+.5f | dQ=%.2e refl_x=%.2e refl_y=%.2e\n",
            N, eps2, res.nprimfail, maxu, aniso, dQ, refx, refy)
    return (aniso = aniso, dQ = dQ, refx = refx, refy = refy,
            maxu = maxu, minPtot = minPtot, res = res)
end

@testset "G6 — non-axisymmetric IC" begin

    @testset "the deformation produces the anisotropy, not the scheme" begin
        c = run_elliptic(200; eps2 = 0.0)
        @printf("  eps2 = 0 control: anisotropy = %+.3e (must be identically zero)\n", c.aniso)
        @test abs(c.aniso) < 1e-10
        @test c.refx < 1e-9
        @test c.refy < 1e-9
    end

    @testset "anisotropy converges with resolution" begin
        # No oracle exists for this problem, so resolution-independence is the
        # substitute: a grid artefact would not converge.
        a = run_elliptic(100)
        b = run_elliptic(200)
        d1 = abs(b.aniso - a.aniso)/abs(a.aniso)
        @printf("  anisotropy %.5f -> %.5f  (change %.2f%%)\n", a.aniso, b.aniso, 100*d1)
        @test a.aniso > 0.15            # an oblate IC must give a positive, sizeable signal
        @test d1 < 0.02                 # converged to better than 2%

        for r in (a, b)
            @test r.res.nprimfail < 5_000
            @test r.maxu < 3.0
            @test r.minPtot > 0.0
            @test r.dQ < 1e-3
            # reflection in x and y are exact symmetries of this IC
            @test r.refx < 1e-8
            @test r.refy < 1e-8
        end
    end
end
