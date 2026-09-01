# ==============================================================================
# test/test_dissipative_vs_1d.jl — GATE G7 (closes P6).
#
# The DISSIPATIVE sectors against the 1-D production solver, on the production IC,
# with shear + bulk + diffusion all enabled on BOTH sides at the Pb+Pb operating
# point. Gate G4 does the same thing for the ideal sector only, and until this
# gate existed the dissipative sectors had never been compared against an
# independent solver on a realistic problem — only against derivations, an
# independently integrated Bjorken ODE, and their own internal consistency.
#
# ---------------------------------------------------------------------------
# WHAT THIS FOUND, AND WHY THE ASSERTIONS ARE RADIUS-RESOLVED
#
# A single global L2 over the fireball is MISLEADING here. Measured at N=200:
#
#     field   global L2     r<2      2-4      4-6      6-8
#     alpha    1.18e-01   5.1e-03  7.8e-03  5.1e-02  2.5e-01
#     Pi       1.14e-01   4.0e-03  2.3e-02  9.5e-02  1.2e-01
#     pirr     1.25e-01   3.8e-03  4.1e-02  2.7e-01  4.1e-01
#     pieta    2.93e-02   4.8e-03  9.2e-03  5.7e-02  1.3e-01
#     nur      4.87e-02   1.4e-02  1.5e-02  3.3e-02  1.0e-01
#     (local T: 0.327, 0.308, 0.259, 0.191 — freeze-out is 0.1565)
#
# The global number is dominated by the outer shells, where the fields are small,
# the profile is steepest, and the 2-D solver carries guards the 1-D does not have
# at all: the density-gated vacuum ramp on the charge sector, the bulk positivity
# guard, and the E_vac_cut. Those were not optional — without them the 2-D solver
# does not run on this IC (TWOD_PROGRAM.md §6d-§6e). So a difference there is
# EXPECTED, and asserting a tight global bound would either fail or force the
# guards out.
#
# In the core the agreement is sub-percent to a few percent across every
# dissipative field, which is the statement worth guarding.
#
# ⚠ CONSEQUENCE FOR USE: do not quote the 2-D dissipative fields from the outer
# region (r ≳ 6 fm, T ≲ 0.19 GeV) without further work. The bulk (T) is fine
# everywhere — 1.4e-3 globally.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_dissipative_vs_1d.jl
# ==============================================================================

using Printf
using Test
using DelimitedFiles

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0 = 0.4;  const TAUF = 2.5;  const RMAX = 20.0

# Pb+Pb production operating point, passed explicitly on BOTH sides so that no
# default on either side can silently differ.
const DST = 0.1163; const ETAS = 0.10; const TAUSH = 0.2; const DSHEAR = 4/3
const ZETAS = 0.10; const TAUPI = 15.0

const FIELDS = (:T, :alpha, :ur, :Pi, :pirr, :pieta, :nur)

function reference_1d(outdir)
    hydro.run_sim_ideal_diff_visc(outdir = outdir,
        Nr = 800, rmax = RMAX, τ0 = TAU0, τfinal = TAUF,
        CFL = 0.15, CFLτ = 0.05, dump_dt = 0.5,
        init_csv = IC_CSV, fugacity_kind = :alpha, taper_width = 1.0,
        eos = hydro.LatticeHRGEOS(),
        enable_diff  = true, kappa_coeff = DST, tauN_coeff = 1.0, deltaN_factor = 0.0,
        enable_shear = true, eta_over_s = ETAS, tauShear_coeff = TAUSH, deltaShear_factor = DSHEAR,
        enable_bulk  = true, zeta_over_s = ZETAS, tauPi_coeff = TAUPI, deltaPi_factor = 0.0,
        taupi_pi_factor = 0.0, lambda_Pi_pi_factor = 0.0,
        lambda_pi_Pi_factor = 0.0, lambda_NN_factor = 0.0,
        init_good_range = false, expand_grid = false,
        postprocess = false, log_every = 10_000_000)

    sn = sort(filter(f -> startswith(f, "snapshot_tau_") && !endswith(f, "_meta.csv"),
                     readdir(outdir)))
    raw, hdr = readdlm(joinpath(outdir, sn[end]), ','; header = true)
    c(n) = Float64.(raw[:, findfirst(==(n), vec(hdr))])
    ur = c("ur")
    return (r = c("r"), tau = c("tau")[1], T = c("T"), alpha = c("alpha"), ur = ur,
            Pi = c("Pi"),
            # Π^{rr} in the 1-D convention: piR is the LRF amplitude, boosted —
            # Π^{rr} = (u^τ)² piR (src/shear_tensor.jl; verified in
            # test_shear2d_algebra.jl "reduces to the 1-D production parametrisation")
            pirr = [(1 + ur[k]^2)*c("piR")[k] for k in eachindex(ur)],
            pieta = c("piEta"), nur = c("nur"))
end

function run_2d(N, τf)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = ETAS, tauShear_coeff = TAUSH, deltaShear_factor = DSHEAR,
        enable_bulk  = true, zeta_over_s = ZETAS, tauPi_coeff = TAUPI, deltaPi_factor = 0.0,
        enable_diff  = true, kappa_coeff = DST, tauN_coeff = 1.0, deltaN_factor = 0.0)
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix], g.yC[iy])
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok
    return g, m, U, res
end

"""Azimuthally average the 2-D state onto the 1-D radial grid, projecting the
tensors onto the radial direction so both sides are the same objects."""
function binned_2d(N, ref)
    g, m, U, res = run_2d(N, ref.tau)
    L = m.layout; ng = g.nghost
    nb = length(ref.r); dr = ref.r[2] - ref.r[1]
    S = Dict(k => zeros(nb) for k in FIELDS); C = zeros(Int, nb)

    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        k = Int(floor(r/dr)) + 1
        (k < 1 || k > nb || r < 1e-9) && continue
        cp = x/r; sp = y/r
        pxx = H.phys_from_stored(U[L.iPixx,i]); pxy = H.phys_from_stored(U[L.iPixy,i])
        pyy = H.phys_from_stored(U[L.iPiyy,i])
        S[:T][k]     += exp(res.work.yT[i])
        S[:alpha][k] += res.work.alpha[i]
        S[:ur][k]    += res.work.ux[i]*cp + res.work.uy[i]*sp
        S[:Pi][k]    += H.phys_from_stored(U[L.iPi,i])
        S[:pirr][k]  += pxx*cp*cp + 2*pxy*sp*cp + pyy*sp*sp
        S[:pieta][k] += H.phys_from_stored(U[L.iPieta,i])
        S[:nur][k]   += H.phys_from_stored(U[L.iNux,i])*cp + H.phys_from_stored(U[L.iNuy,i])*sp
        C[k] += 1
    end
    return S, C, res
end

"""Relative L2 of (2-D − 1-D) over radial shell [rlo, rhi), scaled by the 1-D rms."""
function shell_err(S, C, ref, k, rlo, rhi)
    num = 0.0; den = 0.0; n = 0
    for j in eachindex(ref.r)
        (C[j] < 8 || ref.T[j] < 0.15) && continue
        (ref.r[j] < rlo || ref.r[j] >= rhi) && continue
        v2 = S[k][j]/C[j]; v1 = getfield(ref, k)[j]
        num += (v2 - v1)^2; den += v1^2; n += 1
    end
    return (n == 0 || den <= 0) ? 0.0 : sqrt(num/den), n
end

@testset "G7 — dissipative sectors vs the 1-D production solver (P6)" begin
    outdir = mktempdir()
    ref = reference_1d(outdir)
    @printf("  1-D reference (all sectors): tau=%.4f  T(0)=%.4f  Pi(0)=%.3e  piEta(0)=%.3e\n",
            ref.tau, ref.T[1], ref.Pi[1], ref.pieta[1])

    S, C, res = binned_2d(200, ref)
    @test res.nprimfail < 5_000

    @printf("  %-7s %10s %10s %10s %10s %10s\n", "field", "global", "r<2", "2-4", "4-6", "6-8")
    core = Dict{Symbol,Float64}()
    for k in FIELDS
        gl, _ = shell_err(S, C, ref, k, 0.0, 1e9)
        e1, _ = shell_err(S, C, ref, k, 0.0, 2.0)
        e2, _ = shell_err(S, C, ref, k, 2.0, 4.0)
        e3, _ = shell_err(S, C, ref, k, 4.0, 6.0)
        e4, _ = shell_err(S, C, ref, k, 6.0, 8.0)
        core[k] = max(e1, e2)
        @printf("  %-7s %10.2e %10.2e %10.2e %10.2e %10.2e\n", String(k), gl, e1, e2, e3, e4)
    end

    # ---- the CORE is what is guarded (r < 4 fm, T > 0.3 GeV) ----
    # Bounds are ~2x the measured values, so this is a regression guard, not a
    # restatement of the measurement.
    @test core[:T]     < 5e-3
    @test core[:alpha] < 2e-2
    @test core[:ur]    < 5e-2
    @test core[:Pi]    < 5e-2
    @test core[:pirr]  < 8e-2
    @test core[:pieta] < 2e-2
    @test core[:nur]   < 4e-2

    # The BULK must agree everywhere, not only in the core: T carries no guard
    # that the 1-D lacks, so a global disagreement there would be a real defect.
    glT, _ = shell_err(S, C, ref, :T, 0.0, 1e9)
    @test glT < 5e-3

    rm(outdir; recursive = true, force = true)
end
