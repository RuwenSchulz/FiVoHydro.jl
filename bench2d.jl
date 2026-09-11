# ==============================================================================
# bench2d.jl — cost model for the 2+1D solver.
#
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/bench2d.jl
#
# Reports ns per cell-update, so numbers are comparable across grid sizes, plus
# the marginal cost of each dissipative sector and the scaling with resolution.
# All runs use the production IC so the primitive recovery does the work it
# actually does (a uniform state would flatter the recovery, which is the
# dominant cost).
# ==============================================================================

using Printf
using Statistics

const _ROOT = @__DIR__
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H = hydro2d

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0 = 0.4

function bench(N, τf; shear = true, bulk = true, diff = true, cfm = false, m2 = false,
               label = "", reps = 1)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
    g = H.make_grid2d(N, N; xmax = 20.0, ymax = 20.0)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = shear, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = bulk,  zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = diff,  kappa_coeff = 0.1163, tauN_coeff = 1.0,
        consistent_fm = cfm, consistent_m2 = m2)

    best = Inf; nsteps = 0; pf = 0
    for _ in 1:reps
        U = H.allocate_state(g, m)
        for ix in 1:g.Nxtot, iy in 1:g.Nytot
            r = hypot(g.xC[ix], g.yC[iy])
            H.set_cell!(U, H.lin(g,ix,iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
        end
        H.finalize_ic!(U, g, m; τ0 = TAU0)
        t0 = time()
        res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
        el = time() - t0
        @assert res.ok
        el < best && (best = el; nsteps = res.nsteps; pf = res.nprimfail)
    end
    ncell = g.Nx*g.Ny
    ns_per_cellstep = best/(nsteps*ncell)*1e9
    @printf("  %-22s N=%4d  cells=%7d  steps=%4d  wall=%6.2fs  %7.1f ns/cell-step  pf=%7d\n",
            label, N, ncell, nsteps, best, ns_per_cellstep, pf)
    return (; ncell, nsteps, wall = best, ns = ns_per_cellstep)
end

function main()
    println("Threads: ", Threads.nthreads(), "   (", Sys.CPU_THREADS, " CPU threads)\n")

    # WARM-UP. Without it the first row carries the compilation of the whole solver and reads
    # ~6x the steady cost (measured: N=100 at 4107 ns/cell-step against 594 at N=300, 2026-09-11).
    bench(60, 0.6; label = "warm-up (discarded)")
    println()

    println("Resolution scaling — all sectors, production IC, tau 0.4 -> 4.0")
    rs = [bench(N, 4.0; label = "all sectors") for N in (100, 150, 200, 300)]
    println()
    @printf("  cells x%.1f from N=100 to N=300; ns/cell-step %.1f -> %.1f (%.2fx)\n",
            rs[end].ncell/rs[1].ncell, rs[1].ns, rs[end].ns, rs[end].ns/rs[1].ns)
    @printf("  steps grow %.2fx for %.1fx in N. dt = min(CFL dx/a, CFLtau tau), and WHICH one binds\n",
            rs[end].nsteps/rs[1].nsteps, 3.0)
    println("  depends on the box and the tau range: steps ~ N means the transverse CFL binds (cost ~N^3),")
    println("  steps flat means the Bjorken clock does (cost ~N^2). On +-14 fm to tau = 2 the clock binds")
    println("  below N ~ 150 (COMPARISON_2P1D.md 34.2). This line asserted ~N^3 until 2026-09-10.")

    println("\nMarginal cost of each sector at N=200, tau 0.4 -> 4.0")
    # reps = 3 (best of): with one repetition this section reported diffusion as CHEAPER than
    # ideal and "all three" cheaper than shear alone — noise, not a measurement (2026-09-11).
    b0 = bench(200, 4.0; shear=false, bulk=false, diff=false, label = "ideal only",  reps = 3)
    bs = bench(200, 4.0; shear=true,  bulk=false, diff=false, label = "+ shear",     reps = 3)
    bb = bench(200, 4.0; shear=false, bulk=true,  diff=false, label = "+ bulk",      reps = 3)
    bd = bench(200, 4.0; shear=false, bulk=false, diff=true,  label = "+ diffusion", reps = 3)
    ba = bench(200, 4.0; label = "all sectors", reps = 3)
    # the charm closures (2026-09-08/09): the consistent first moment adds four source
    # terms per cell; the second moment adds five fields and FIVE source evaluations per
    # cell per step (one plus one per channel, to measure the implicit coefficient).
    bf = bench(200, 4.0; cfm = true,            label = "all + consistent_fm",      reps = 3)
    bm = bench(200, 4.0; cfm = true, m2 = true, label = "all + consistent_fm + m2", reps = 3)
    println()
    for (nm, b) in (("shear", bs), ("bulk", bb), ("diffusion", bd), ("all three", ba),
                    ("+ cons. fm", bf), ("+ fm + m2", bm))
        @printf("  %-10s costs %+6.1f%% over ideal (per cell-step)\n", nm, 100*(b.ns/b0.ns - 1))
    end

    println("\nProjection to a production-resolution run:")
    ns = ba.ns
    for (N, τf) in ((400, 13.0), (800, 13.0))
        # steps scale ~ N (CFL) x the tau range
        steps = ba.nsteps * (N/200) * (13.0-0.4)/(4.0-0.4)
        cells = N*N
        secs = ns*1e-9*steps*cells
        @printf("  N=%4d (dx=%.3f fm), tau -> %.0f : ~%.0f steps, ~%.1f min\n",
                N, 40.0/N, τf, steps, secs/60)
    end
end

main()
