# ==============================================================================
# test/test_reproduction2d.jl — GATE G4, the reproduction gate.
#
# The 2+1D solver, seeded from the production initial condition
# (data/initial_profiles_physical.csv), must reproduce the 1+1D PRODUCTION run
# from the same IC.
#
# This is the money gate of the whole ladder, and also the one whose meaning is
# easiest to overstate. The production IC is azimuthally symmetric BY
# CONSTRUCTION — `Julia/Projects/ALICE_IC_Creation/MCGCollisionDensity.jl:34`
# deposits every binary collision as `azimuthal_gauss(r, ρ, w)`, the φ-average of
# a unit Gaussian. So this test contains NO new physics. What it establishes is
# that the 2-D solver, on the problem where an independent trusted answer exists,
# gives that answer — which is the only way to validate a 2-D viscous solver
# against something other than itself.
#
# Both solvers are run here, in-process, from the SAME loader: the IC comes from
# `hydro.load_initial_interpolants`, the 1-D code's own function, so an IC
# mismatch is impossible by construction.
#
# What is compared: the azimuthal average of T(x,y) binned onto the 1-D radial
# grid, over the region where the fireball is hot. Agreement must IMPROVE with
# 2-D resolution — a fixed offset would indicate a systematic difference, which a
# single-resolution comparison could not distinguish from discretisation error.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_reproduction2d.jl
# ==============================================================================

using Printf
using Test
using DelimitedFiles

const _ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(_ROOT, "main.jl"))     # module hydro   — 1-D production
include(joinpath(_ROOT, "main2D.jl"))   # module hydro2d — under test
using .hydro
using .hydro2d
const H = hydro2d

const IC_CSV  = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0    = 0.4
const TAUF    = 2.5
const RMAX    = 20.0
const TAPER_W = 1.0

# ------------------------------------------------------------------------------
"""Run the 1-D production solver and return its final snapshot."""
function reference_1d(outdir)
    hydro.run_sim_ideal_diff_visc(
        outdir = outdir,
        Nr = 800, rmax = RMAX, τ0 = TAU0, τfinal = TAUF,
        CFL = 0.15, CFLτ = 0.05, time_integrator = :ssprk2, dump_dt = 0.5,
        init_csv = IC_CSV, fugacity_kind = :alpha, taper_width = TAPER_W,
        eos = hydro.LatticeHRGEOS(),
        enable_diff = false, enable_shear = false, enable_bulk = false,
        init_good_range = false, expand_grid = false,
        postprocess = false, log_every = 10_000_000)

    snaps = filter(f -> startswith(f, "snapshot_tau_") && !endswith(f, "_meta.csv"),
                   readdir(outdir))
    @assert !isempty(snaps)
    raw, hdr = readdlm(joinpath(outdir, sort(snaps)[end]), ','; header = true)
    col(name) = Float64.(raw[:, findfirst(==(name), vec(hdr))])
    return (r = col("r"), T = col("T"), n = col("n"), tau = col("tau")[1])
end

"""Run the 2-D solver on the same IC to the same proper time."""
function run_2d(N, τf)
    # the 1-D code's OWN loader: identical IC by construction
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = TAPER_W, interp_kind = :linear)

    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS())
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix], g.yC[iy])
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)

    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    return g, m, U, res, time() - t0
end

"""Azimuthally average the 2-D temperature onto the 1-D radial grid and compare."""
function compare(N, ref)
    g, m, U, res, el = run_2d(N, ref.tau)
    L = m.layout; ng = g.nghost
    nb = length(ref.r); dr = ref.r[2] - ref.r[1]

    sT = zeros(nb); sT2 = zeros(nb); cnt = zeros(Int, nb)
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        k = Int(floor(hypot(g.xC[ix], g.yC[iy])/dr)) + 1
        (k < 1 || k > nb) && continue
        T = exp(res.work.yT[i])
        sT[k] += T; sT2[k] += T*T; cnt[k] += 1
    end

    num = 0.0; den = 0.0; linf = 0.0; azmax = 0.0
    for k in 1:nb
        (cnt[k] < 8 || ref.T[k] < 0.15) && continue     # hot region, populated bins
        Tm = sT[k]/cnt[k]
        v  = max(sT2[k]/cnt[k] - Tm^2, 0.0)
        d  = abs(Tm - ref.T[k])/ref.T[k]
        num += d^2; den += 1; linf = max(linf, d)
        azmax = max(azmax, sqrt(v)/Tm)
    end

    asym = 0.0
    for k in 0:(g.Nx-1), l in 0:(g.Ny-1)
        a = U[L.iE, H.lin(g, ng+1+k, ng+1+l)]
        b = U[L.iE, H.lin(g, ng+1+l, ng+1+k)]
        asym = max(asym, abs(a-b)/max(abs(a), 1e-30))
    end

    L2 = sqrt(num/max(den, 1))
    @printf("  N=%4d dx=%.3f steps=%4d wall=%5.1fs | T vs 1-D: L2=%.3e Linf=%.3e | azim spread=%.3e | x<->y=%.1e\n",
            N, g.dx, res.nsteps, el, L2, linf, azmax, asym)
    return (L2 = L2, linf = linf, azmax = azmax, asym = asym, res = res)
end

# ------------------------------------------------------------------------------
@testset "G4 — 2+1D reproduces the 1+1D production run on the production IC" begin
    outdir = mktempdir()
    ref = reference_1d(outdir)
    @printf("  1-D reference: tau=%.5f, %d radial points, dr=%.4f, T(r=0)=%.4f\n",
            ref.tau, length(ref.r), ref.r[2]-ref.r[1], ref.T[1])

    a = compare(100, ref)
    b = compare(200, ref)

    order = log2(a.L2/b.L2)
    @printf("  convergence toward the 1-D answer: order %.2f in dx\n", order)

    # The 2-D answer must APPROACH the 1-D one as the grid refines. A systematic
    # difference would show as a flat L2; discretisation error must converge.
    @test b.L2 < a.L2
    @test order > 0.8

    # Absolute agreement at the coarser of the two resolutions (dx = 0.4, i.e. 16x
    # coarser than the 1-D grid), so the bound is not resolution-flattering.
    @test a.L2   < 5e-3
    @test a.linf < 2e-2
    @test b.L2   < 2e-3

    # The IC has no azimuthal structure, so any is the scheme's: it must stay small
    # and must not grow with refinement.
    @test b.azmax < 5e-3
    @test b.azmax <= a.azmax * 1.5

    # x<->y exchange symmetry of the raw 2-D field must hold to round-off.
    @test a.asym < 1e-11
    @test b.asym < 1e-11

    rm(outdir; recursive = true, force = true)
end
