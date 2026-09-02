# ==============================================================================
# plot2d_evolution.jl — T(x,y,τ) and charm n(x,y,τ) on the un-averaged IC.
#
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/plot2d_evolution.jl
#
# Runs the 2+1D solver from the genuinely non-homogeneous initial condition built
# by `Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl` — in which the temperature and
# the charm density carry DIFFERENT geometries (entropy/participants vs binary
# collisions), so both profiles are non-trivial and neither is slaved to the other.
#
# CONFIGURATION, stated explicitly because it is the point of the exercise:
#   * bulk hydro with shear + bulk viscosity (eta/s = zeta/s = 0.10);
#   * charm carried as a FIRST-ORDER MIS diffusion current (nu^x, nu^y). The IS2
#     second-moment charm system does not exist in the 2-D solver at all, so
#     "second moments" are off by construction, not by a flag;
#   * NO back-reaction of the current on the fluid — automatic here rather than
#     imposed: `LatticeHRGEOS` has dP/dmu = 0 identically (TWOD_PROGRAM.md §6b), so
#     the charm sector cannot push on the bulk. Measured, not assumed.
#
# Writes PNG + PDF into `plots2d/`.
# ==============================================================================

using Printf
using Plots
using Statistics

const _ROOT = @__DIR__
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

# Which centrality class to run. `BuildIC2D.jl all` writes 00-05, 20-30 and 30-40;
# the peripheral classes are far more almond-shaped (eps2 0.08 -> ~0.30 on the
# entropy field), which is the point of running them.
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/plot2d_evolution.jl 20-30
const TAG  = isempty(ARGS) ? "00-05" : ARGS[1]
const SUF  = TAG == "00-05" ? "" : "_" * TAG
const IC2D = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation",
                               "PbPb", "data", "ic2d_$(TAG).csv"))
const OUT  = joinpath(_ROOT, "plots2d")
const TAUS = [0.4, 1.0, 2.0, 4.0, 6.0, 8.0]
const T_FO = 0.1565      # freeze-out; above this is what produces observables

gr()
default(fontfamily = "sans-serif", grid = false, framestyle = :box)

"""Run and capture (T, n) on the interior grid at each requested proper time."""
function run_and_capture(; N = 200, box = 16.0)
    g = H.make_grid2d(N, N; xmax = box, ymax = box)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0,
        # SHEAR REGULATOR. Off by default in the solver (1-D parity); required for
        # any single-event IC, where hot-spot gradients drive |pi|/P past 1e7 and
        # the solve loses a third of its charge while still reporting ok = true.
        # Inert on smooth ICs — measured, agrees to every printed digit. Gate G9.
        pi_clip_factor = 1.0)
    L = m.layout
    U = H.allocate_state(g, m)
    ic = H.initialize_from_grid_csv!(U, g, m, TAUS[1], IC2D)
    @printf("IC: %d cells with matter failed, %d vacuum\n", ic.nbad, ic.nvacuum)
    @assert ic.nbad == 0

    ng = g.nghost
    xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]
    ys = [g.yC[iy] for iy in (ng+1):(ng+g.Ny)]

    frames = Dict{Float64,NamedTuple}()
    wk = H.make_work(g, m)

    function grab(τ, work)
        T = zeros(g.Nx, g.Ny); n = zeros(g.Nx, g.Ny)
        for (a, ix) in enumerate((ng+1):(ng+g.Nx)), (b, iy) in enumerate((ng+1):(ng+g.Ny))
            i = H.lin(g, ix, iy)
            T[a,b] = exp(work.yT[i]); n[a,b] = work.n[i]
        end
        # the diffusion current, reported so the run ATTESTS that the charge
        # sector is live rather than merely flagged on: nu = 0 everywhere would
        # mean enable_diff was cosmetic. It lives in the STATE, not the work
        # arrays, and is stored rescaled (`phys_from_stored`).
        #
        # |nu|/n is reported BOTH globally and above freeze-out, because the two
        # differ by an order of magnitude and only the second is an applicability
        # statement. Measured on all three centrality classes: the global figure
        # crosses 1 (peak 1.25 at tau=1, 20-30%) but every one of those maxima
        # sits at n ~ 2e-6 fm^-3 and T ~ 0.09-0.12 GeV — five decades below the
        # peak density and BELOW T_fo, i.e. in the dilute tail that produces no
        # observables. Above T_fo the ratio peaks at 0.21-0.27 and falls
        # monotonically to 0.01-0.04 by tau=8.
        numax = 0.0; nurel = 0.0; nurel_fo = 0.0
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H.lin(g, ix, iy)
            nu = hypot(H.phys_from_stored(U[L.iNux, i]),
                       H.phys_from_stored(U[L.iNuy, i]))
            numax = max(numax, nu)
            if work.n[i] > 1e-6
                nurel = max(nurel, nu/work.n[i])
                exp(work.yT[i]) > T_FO && (nurel_fo = max(nurel_fo, nu/work.n[i]))
            end
        end
        frames[τ] = (T = T, n = n, numax = numax, nurel = nurel, nurel_fo = nurel_fo)
    end

    τ = TAUS[1]
    H.rhs_2d!(wk.k, U, g, τ, m, wk)          # populate primitives for the τ0 frame
    grab(τ, wk)
    t0 = time(); nsteps = 0
    for τt in TAUS[2:end]
        res = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = τt, CFL = 0.15, CFLτ = 0.05, work = wk)
        @assert res.ok
        τ = res.τ; nsteps += res.nsteps
        H.update_primitives_2d!(U, g, τ, m, wk)
        grab(τ, wk)
        @printf("  tau=%5.2f  steps=%4d  primfail=%6d  T(0)=%.4f  n_max=%.4e  max|nu|=%.3e  max|nu|/n=%.4f (above T_fo %.4f)\n",
                τ, res.nsteps, res.nprimfail, frames[τ].T[g.Nx÷2, g.Ny÷2],
                maximum(frames[τ].n), frames[τ].numax, frames[τ].nurel, frames[τ].nurel_fo)
    end
    @printf("total %d steps, %.1f s wall (%d x %d)\n", nsteps, time()-t0, N, N)
    return xs, ys, frames, g
end

"""Panel of heatmaps, one per proper time, on a shared colour scale."""
function panel(xs, ys, frames, key, title, cb_label;
               logscale = false, cmap = :inferno, contour_at = nothing)
    vals = [frames[τ][key] for τ in TAUS]
    lo = minimum(minimum(v[v .> 0]) for v in vals)
    hi = maximum(maximum(v) for v in vals)
    plots = Any[]
    for (k, τ) in enumerate(TAUS)
        A = copy(vals[k])
        if logscale
            A = log10.(max.(A, lo))
            clims = (log10(max(hi*1e-4, lo)), log10(hi))
        else
            clims = (0.0, hi)
        end
        # colourbar on EVERY panel: with it on only some, those panels are shrunk
        # to make room and the grid no longer reads as a sequence.
        p = heatmap(xs, ys, permutedims(A);
                    c = cmap, clims = clims, aspect_ratio = 1,
                    xlims = extrema(xs), ylims = extrema(ys),
                    title = @sprintf("τ = %.1f fm/c", τ), titlefontsize = 10,
                    xlabel = k > 3 ? "x [fm]" : "", ylabel = k in (1,4) ? "y [fm]" : "",
                    colorbar = true, colorbar_title = k in (3,6) ? cb_label : "",
                    colorbar_titlefontsize = 8,
                    tickfontsize = 7, guidefontsize = 8)
        # freeze-out contour: the boundary of the region that produces observables
        if contour_at !== nothing
            Traw = frames[τ][key]
            contour!(p, xs, ys, permutedims(Traw); levels = [contour_at],
                     c = :white, lw = 1.2, colorbar_entry = false, label = "")
        end
        push!(plots, p)
    end
    return plot(plots...; layout = (2,3), size = (1320, 760),
                plot_title = title, plot_titlefontsize = 13, left_margin = 4Plots.mm,
                bottom_margin = 4Plots.mm, right_margin = 2Plots.mm)
end

"""Slices along x and y through the centre — the anisotropy made quantitative.

Colour encodes proper time, line style encodes direction (solid x, dashed y), so
the legend stays readable with six times. For the charm density the y-range is
CLIPPED to the fireball: the vacuum floor sits 12 decades below the peak and an
unclipped log axis shows nothing but that floor.
"""
function slices(xs, ys, frames, key, ylab; logy = false, floor_frac = 1e-4)
    ix0 = length(xs) ÷ 2 + 1; iy0 = length(ys) ÷ 2 + 1
    peak = maximum(maximum(frames[τ][key]) for τ in TAUS)
    lo = logy ? floor_frac*peak : 0.0

    p = plot(; xlabel = "distance from centre [fm]", ylabel = ylab,
             legend = :topright, size = (700, 470),
             yscale = logy ? :log10 : :identity,
             ylims = logy ? (lo, 2peak) : (0, 1.05peak),
             xlims = (-13, 13),
             tickfontsize = 8, guidefontsize = 10, legendfontsize = 8,
             title = "solid: along x   ·   dashed: along y", titlefontsize = 9)

    cols = palette(:viridis, length(TAUS))
    for (k, τ) in enumerate(TAUS)
        A = frames[τ][key]
        vx = A[:, iy0]; vy = A[ix0, :]
        if logy
            vx = [v > lo ? v : NaN for v in vx]
            vy = [v > lo ? v : NaN for v in vy]
        end
        plot!(p, xs, vx; c = cols[k], lw = 2,  label = @sprintf("τ = %.1f fm/c", τ))
        plot!(p, ys, vy; c = cols[k], lw = 2, ls = :dash, label = "")
    end
    return p
end

function main()
    mkpath(OUT)
    xs, ys, frames, g = run_and_capture()

    cent = occursin("_ev", TAG) ?
        replace(TAG, "_ev" => "%, single event #") : TAG * "%"
    figs = Dict(
        "T_evolution$(SUF)"  => panel(xs, ys, frames, :T,
                                "Temperature evolution — Pb+Pb $(cent), un-averaged IC (white: T = T_fo = 0.1565 GeV)",
                                "T [GeV]"; cmap = :inferno, contour_at = 0.1565),
        "n_evolution$(SUF)"  => panel(xs, ys, frames, :n, "Charm density evolution — same run",
                                "log₁₀ n [fm⁻³]"; logscale = true, cmap = :viridis),
        "T_slices$(SUF)"     => slices(xs, ys, frames, :T, "T [GeV]"),
        "n_slices$(SUF)"     => slices(xs, ys, frames, :n, "charm density n [fm⁻³]"; logy = true),
    )
    for (name, f) in figs
        savefig(f, joinpath(OUT, name * ".png"))
        savefig(f, joinpath(OUT, name * ".pdf"))
        println("  wrote ", joinpath(OUT, name * ".png"))
    end

    # quantify the anisotropy the plots show
    ix0 = length(xs) ÷ 2 + 1; iy0 = length(ys) ÷ 2 + 1
    println("\nhalf-width where the field falls to 1/2 of its central value:")
    @printf("  %6s | %8s %8s %8s | %8s %8s %8s\n", "tau", "T x", "T y", "T y/x", "n x", "n y", "n y/x")
    for τ in TAUS
        A = frames[τ]
        function halfwidth(v, coord, c0)
            k = findfirst(j -> v[j] < 0.5*c0, (length(coord)÷2+1):length(coord))
            k === nothing ? NaN : coord[length(coord)÷2 + k]
        end
        Tx = halfwidth(A.T[:, iy0], xs, A.T[ix0, iy0])
        Ty = halfwidth(A.T[ix0, :], ys, A.T[ix0, iy0])
        nx = halfwidth(A.n[:, iy0], xs, A.n[ix0, iy0])
        ny = halfwidth(A.n[ix0, :], ys, A.n[ix0, iy0])
        @printf("  %6.1f | %8.2f %8.2f %8.3f | %8.2f %8.2f %8.3f\n", τ, Tx, Ty, Ty/Tx, nx, ny, ny/nx)
    end
end

main()
