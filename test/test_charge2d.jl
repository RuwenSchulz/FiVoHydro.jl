# ==============================================================================
# test/test_charge2d.jl — gate G3, the charge/diffusion sector (P3, closes D2).
#
#   (a) the copied transport coefficients still match src/dissipation.jl
#   (b) ν_NS reduces EXACTLY to the 1-D production expression when u^y = 0
#   (c) charge is conserved to round-off with diffusion switched on
#   (d) diffusion flows DOWN the fugacity gradient and flattens a bump
#   (e) an azimuthally symmetric IC stays x<->y symmetric to round-off
#
# (a) and (b) test the derivation and guard the duplication; (c)-(e) test the code.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_charge2d.jl
# ==============================================================================

using Printf
using Random
using Test
using SpecialFunctions

const _ROOT = normpath(joinpath(@__DIR__, ".."))

# ---- the 1-D chain, for the coefficient comparison in (a) --------------------
module OneD
    using SpecialFunctions
    const R = normpath(joinpath(@__DIR__, "..", ".."))
    const P = normpath(joinpath(@__DIR__, "..", "..", "..", "FiVoHydro.jl"))
    _root = normpath(joinpath(@__DIR__, ".."))
    include(joinpath(_root, "src", "constants.jl"))
    include(joinpath(_root, "src", "utils.jl"))
    include(joinpath(_root, "src", "eos.jl"))
    include(joinpath(_root, "src", "primitives.jl"))
    include(joinpath(_root, "src", "state_layout.jl"))
    include(joinpath(_root, "src", "grid.jl"))
    include(joinpath(_root, "src", "shear_tensor.jl"))
    include(joinpath(_root, "src", "diagnostics.jl"))
    include(joinpath(_root, "src", "primrec.jl"))
    include(joinpath(_root, "src", "work.jl"))
    include(joinpath(_root, "src", "floors.jl"))
    include(joinpath(_root, "src", "dissipation.jl"))
end

include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const RNG = MersenneTwister(20260901)

# ------------------------------------------------------------------------------
@testset "G3a — copied transport coefficients match src/dissipation.jl" begin
    # src2d/transport2d.jl duplicates two pure functions because src/dissipation.jl
    # cannot be included alongside the 2-D module. This gate is what keeps the copy
    # honest: it fails the moment the original changes.
    eos1 = OneD.LatticeHRGEOS()
    eos2 = H.LatticeHRGEOS()
    worst_tau = 0.0; worst_norm = 0.0
    for T in (0.128, 0.15, 0.20, 0.30, 0.40, 0.50, 0.5633), α in (-1.75, -0.5, 0.5, 1.89)
        for DsT in (0.05, 0.1163, 0.3), tauD in (0.5, 1.0, 2.0)
            a = OneD._diff_tauN_impl(T, α, eos1, DsT, tauD)
            b = H._diff_tauN_impl_2d(T, α, eos2, DsT, tauD)
            worst_tau = max(worst_tau, abs(a-b)/max(abs(a), 1e-300))
        end
        n1 = OneD._fluidum_single_hadron_normalization(T, α, eos1)
        n2 = H._fluidum_single_hadron_normalization_2d(T, α, eos2)
        worst_norm = max(worst_norm, abs(n1-n2)/max(abs(n1), 1e-300))
    end
    @printf("  max rel diff  tau_n %.3e   normalization %.3e\n", worst_tau, worst_norm)
    @test worst_tau == 0.0
    @test worst_norm == 0.0
end

# ------------------------------------------------------------------------------
@testset "G3b — nu_NS reduces to the 1-D production form (closes D2)" begin
    # 1-D:  nu_NS = -kappa[ (u^tau)^2 d_r alpha + u^r u^tau d_tau alpha ]
    # ours: nu_NS^i = -kappa[ d_i alpha + u^i D alpha ],  D = u^tau d_tau + u^k d_k
    # With u^y = 0 these are the same expression, via 1 + (u^x)^2 = (u^tau)^2.
    worst = 0.0; worst_y = 0.0
    for _ in 1:5000
        ux = 2.0*(2rand(RNG)-1)
        uτ = sqrt(1 + ux^2)
        dxa = 3.0*(2rand(RNG)-1)
        dta = 3.0*(2rand(RNG)-1)
        κ   = 0.01 + rand(RNG)

        nsx, nsy = H.ns_diffusion_target_2d(ux, 0.0, uτ, dxa, 0.0, dta, κ)
        oned = -κ*((uτ^2)*dxa + ux*uτ*dta)      # written out independently

        # Scale by the MAGNITUDE OF THE TERMS, not by the result. The two terms
        # can cancel to near zero, and a relative measure against the cancelled
        # sum reports the cancellation rather than the disagreement.
        scale = κ*((uτ^2)*abs(dxa) + abs(ux*uτ*dta))
        worst   = max(worst, abs(nsx - oned)/max(scale, 1e-30))
        worst_y = max(worst_y, abs(nsy))
    end
    @printf("  max |nu_NS^x - 1-D form| / term scale = %.3e\n", worst)
    @printf("  max |nu_NS^y| on a y-symmetric state = %.3e\n", worst_y)
    @test worst < 1e-13
    @test worst_y == 0.0
end

# ------------------------------------------------------------------------------
# Runs with the charge sector live.
# ------------------------------------------------------------------------------

"""Gaussian fugacity bump on a uniform-T background, well inside the box."""
function run_bump(; Nx = 48, Ny = 48, box = 8.0, τ0 = 0.6, τf = 2.0,
                    T0 = 0.40, a0 = -4.5, da = 1.5, w = 2.0,
                    DsT = 0.1163, enable_diff = true, CFLτ = 0.01, bc = :outflow)
    g = H.make_grid2d(Nx, Ny; xmax = box, ymax = box)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_diff = enable_diff,
                           kappa_coeff = DsT, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    nbad = 0
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix], g.yC[iy])
        α = a0 + da*exp(-(r/w)^2)
        H.set_cell!(U, H.lin(g, ix, iy), T0, α, 0.0, 0.0, τ0, m) || (nbad += 1)
    end
    @assert nbad == 0

    L = m.layout; ng = g.nghost
    Q0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q0 += U[L.iDtau, H.lin(g, ix, iy)]
    end

    res = H.run_sim_2d!(U, g, m; τ0 = τ0, τfinal = τf, CFL = 0.2, CFLτ = CFLτ, bc = bc)
    @assert res.ok

    Q1 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        Q1 += U[L.iDtau, H.lin(g, ix, iy)]
    end

    return (g = g, m = m, U = U, res = res, L = L, Q0 = Q0, Q1 = Q1)
end

@testset "G3c — charge conservation with diffusion on" begin
    # tau*J^tau summed over the interior is the total charge, and the Dtau source
    # is identically zero, so the flux divergence telescopes: any drift MUST be
    # flux through the boundary. The decisive test is therefore a CLOSED domain.
    #
    # With outflow boundaries the state is not compactly supported — the charm
    # background n(T, alpha0) fills the whole box, so charge genuinely leaves and a
    # drift is physical, not a scheme error. Both numbers are reported so the
    # distinction is on the record rather than asserted.
    per = run_bump(; bc = :periodic)
    out = run_bump(; bc = :outflow)
    dper = abs(per.Q1 - per.Q0)/abs(per.Q0)
    dout = abs(out.Q1 - out.Q0)/abs(out.Q0)
    @printf("  steps %d  primfail %d\n", per.res.nsteps, per.res.nprimfail)
    @printf("  charge drift, closed domain (periodic) = %.3e   <- the conservation test\n", dper)
    @printf("  charge drift, outflow boundaries       = %.3e   <- physical outflow\n", dout)
    @test per.res.nprimfail == 0
    @test out.res.nprimfail == 0
    @test dper < 1e-12
end

@testset "G3d — diffusion runs DOWN the fugacity gradient and flattens the bump" begin
    on  = run_bump(; enable_diff = true)
    off = run_bump(; enable_diff = false)
    g = on.g; L = on.L; ng = g.nghost
    ic = H.lin(g, ng + g.Nx÷2, ng + g.Ny÷2)

    # peak charge density must fall further with diffusion than without
    peak_on  = on.U[L.iDtau, ic]
    peak_off = off.U[L.iDtau, ic]
    @printf("  central tau*J^tau:  diffusion off %.6e   on %.6e   ratio %.5f\n",
            peak_off, peak_on, peak_on/peak_off)
    @test peak_on < peak_off

    # nu^x must oppose d_x alpha (nu_NS = -kappa grad alpha)
    nsame = 0; ntot = 0
    for ix in (ng+2):(ng+g.Nx-1), iy in (ng+2):(ng+g.Ny-1)
        i = H.lin(g, ix, iy)
        dxa = (on.res.work.alpha[i+g.Nytot] - on.res.work.alpha[i-g.Nytot])
        nux = H.phys_from_stored(on.U[L.iNux, i])
        (abs(dxa) < 1e-10 || abs(nux) < 1e-14) && continue
        ntot += 1
        (nux*dxa > 0) && (nsame += 1)
    end
    @printf("  cells with nu^x pointing UP-gradient: %d / %d\n", nsame, ntot)
    @test ntot > 100
    @test nsame == 0
end

@testset "G3e — azimuthally symmetric IC stays x<->y symmetric" begin
    # The IC depends only on r, so the state must be invariant under x <-> y at
    # every time. This is the small-scale rehearsal for gate G4: any asymmetry
    # here is the SCHEME's, since the physics has none.
    r = run_bump()
    g = r.g; L = r.L; U = r.U; ng = g.nghost
    worst = 0.0
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        i = H.lin(g, ng+1+k, ng+1+l)
        j = H.lin(g, ng+1+l, ng+1+k)          # transpose
        for (a, b) in ((L.iDtau, L.iDtau), (L.iE, L.iE), (L.iSx, L.iSy), (L.iNux, L.iNuy))
            sc = max(abs(U[a,i]), abs(U[b,j]), 1e-30)
            worst = max(worst, abs(U[a,i] - U[b,j])/sc)
        end
    end
    @printf("  max x<->y asymmetry over all fields = %.3e\n", worst)
    @test worst < 1e-12
end
