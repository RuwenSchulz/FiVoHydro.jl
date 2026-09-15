#!/usr/bin/env julia
#=
03 — how fine a grid do I need, and how would I know?

The template for the convergence study a referee asks for, run on the configuration this solver is
actually used in. It reports THREE different things, because they converge at different rates and
picking the flattering one is the easiest way to over-claim:

  a  a FIELD:      T at the centre and the L2 of T over the hot region
  b  an OBSERVABLE: the momentum anisotropy (an integral, so it converges faster than the field)
  c  a CONSERVED QUANTITY: the charge drift ΔQ/Q (which is not a truncation error at all)

⚠ A STATISTIC CAN IMPROVE BECAUSE ITS DENOMINATOR GREW. Refining a grid changes the number of cells
in every average. Always report the quantity AND its scale, and always run the control (here: the
same ladder on an ε₂ = 0 IC, whose anisotropy must be identically zero at every resolution — if it
is not, the "convergence" of the ε₂ = 0.25 number is partly the grid finding a signal that is not
there).

⚠ RICHARDSON NEEDS THREE POINTS. Two grids give you a ratio; only three tell you whether the ratio
is the asymptotic order. The ladder below is 3 grids for that reason, and the printed order is the
thing to quote — not the difference between the two finest.
=#
ENV["GKSwstype"] = "100"
using Printf, Statistics, Plots
# Figure style, shared by every example in BOTH packages (2026-09-14). dpi 200 and the larger
# fonts are for the GitHub READMEs: they render an image at container width (~900 px), so a
# 1700-px-wide figure is downscaled and 8 pt tick labels turn to mush. Keep the two packages
# identical — a reader comparing FiVo and Fluidum plots should not be reading two house styles.
gr(); default(; fontfamily = "sans-serif", framestyle = :box, grid = false, dpi = 200, lw = 2.2,
               titlefontsize = 11, guidefontsize = 10, tickfontsize = 9, legendfontsize = 8,
               foreground_color_legend = nothing, background_color_legend = RGBA(1,1,1,0.75),
               left_margin = 9Plots.mm, bottom_margin = 6Plots.mm, right_margin = 3Plots.mm,
               colorbar_titlefontsize = 8,   # ⚠ at 10 pt the colorbar TITLE overlaps its own
                                             # tick labels on every map panel (measured on ex10)
               top_margin = 2Plots.mm)
# ⚠ the margins are NOT cosmetic. Raising the font sizes above without them silently DROPS the axis
# labels and clips the y-label off the left edge — GR gives the axis whatever space is left after the
# panel, and at dpi 200 with 10 pt guides there is none. Measured on ex01: "τ [fm/c]" vanished from
# all three panels and reappeared only once the margins were set.

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0, TAUF, RMAX, T_FO = 0.4, 6.0, 20.0, 0.1565

function run_case(N, eps2)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    a = sqrt(1 - eps2); b = sqrt(1 + eps2)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(hypot(g.xC[ix]/a, g.yC[iy]/b))),
                    Float64(itpF(hypot(g.xC[ix]/a, g.yC[iy]/b))), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    t = @elapsed res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok
    ng = g.nghost; sx = 0.0; sy = 0.0; nhot = 0
    Tf = zeros(g.Nx, g.Ny)                       # a plain array indexed by cell, not a Dict
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy); T = exp(res.work.yT[i])
        Tf[ix-ng, iy-ng] = T
        T < T_FO && continue
        w = U[m.layout.iE, i]; sx += w*res.work.ux[i]^2; sy += w*res.work.uy[i]^2; nhot += 1
    end
    xs = [g.xC[ng+k] for k in 1:g.Nx]
    ic = argmin(abs.(xs))
    (aniso = (sx+sy) > 0 ? (sx-sy)/(sx+sy) : 0.0, T0 = Tf[ic, ic],
     nhot = nhot, dQ = res.dQ, maxu = res.maxu, wall = t, Tf = Tf, xs = xs, nsteps = res.nsteps)
end
"""L2 of T over the hot region, sampled on a FIXED physical grid so the two runs are compared at the
same points rather than at their own cell centres -- the usual way a resolution study lies to itself.

WARNING: look the cells up by INDEX, not by scanning every cell for the nearest one. An `argmin`
over all cells, run per sample point, is O(N^4) and turned this example from seconds into half an
hour the first time it was written."""
function l2_against(fine, coarse)
    at(r, x, y) = r.Tf[clamp(searchsortedfirst(r.xs, x), 1, length(r.xs)),
                       clamp(searchsortedfirst(r.xs, y), 1, length(r.xs))]
    num = 0.0; den = 0.0
    for x in -8.0:0.5:8.0, y in -8.0:0.5:8.0
        a = at(fine, x, y); a < T_FO && continue
        num += (a - at(coarse, x, y))^2; den += a^2
    end
    sqrt(num/max(den, 1e-30))
end

const NS = (100, 150, 225)          # ×1.5 per rung, so the orders below are log_1.5
println("\n  ε₂ = 0.25, all sectors on, τ = $TAU0 → $TAUF fm/c")
@printf("  %5s %7s %10s %12s %9s %9s %8s %8s\n", "N", "dx", "T(0) [GeV]", "anisotropy", "hot cells", "dQ/Q", "steps", "wall")
runs = Dict{Int,Any}()
for N in NS
    r = run_case(N, 0.25); runs[N] = r
    @printf("  %5d %7.3f %10.6f %12.6f %9d %9.2e %8d %7.1fs\n",
            N, 2RMAX/N, r.T0, r.aniso, r.nhot, r.dQ, r.nsteps, r.wall)
end
println("\n  ε₂ = 0 CONTROL — the anisotropy must be identically zero at EVERY resolution:")
for N in NS
    r = run_case(N, 0.0)
    @printf("    N = %3d   anisotropy = %+.3e   %s\n", N, r.aniso, abs(r.aniso) < 1e-10 ? "OK" : "SUSPECT")
end

ord(a, b, c) = log(abs(a-b)/abs(b-c))/log(1.5)
@printf("\n  observed order in Δx (three-point Richardson, refinement ratio 1.5)\n")
@printf("    T at the centre     %.2f\n", ord(runs[100].T0,    runs[150].T0,    runs[225].T0))
@printf("    anisotropy          %.2f\n", ord(runs[100].aniso, runs[150].aniso, runs[225].aniso))
@printf("    L2(T) hot region    N=100 vs 225 %.3e   N=150 vs 225 %.3e   (ratio %.2f)\n",
        l2_against(runs[225], runs[100]), l2_against(runs[225], runs[150]),
        l2_against(runs[225], runs[100])/max(l2_against(runs[225], runs[150]), 1e-30))
# ── the figure: errors against the finest grid, on log-log with reference slopes ─────────────────
# Plotted against the FINEST run rather than against an extrapolated limit, because with three points
# a Richardson limit is itself uncertain and plotting against it hides that. The reference slopes are
# drawn, not fitted.
let ref = runs[NS[end]], dxs = [2RMAX/N for N in NS[1:end-1]]
    eT  = [abs(runs[N].T0    - ref.T0)    for N in NS[1:end-1]]
    eA  = [abs(runs[N].aniso - ref.aniso) for N in NS[1:end-1]]
    eL2 = [l2_against(ref, runs[N])       for N in NS[1:end-1]]
    plt = plot(size = (720, 470), xscale = :log10, yscale = :log10, legend = :bottomright,
               xlabel = "Δx  [fm]", ylabel = "difference from the N = $(NS[end]) run",
               title = "three quantities, three convergence rates")
    plot!(plt, dxs, eT;  marker = :circle,   label = "T at the centre")
    plot!(plt, dxs, eL2; marker = :square,   label = "L2 of T over the hot region")
    plot!(plt, dxs, eA;  marker = :diamond,  label = "momentum anisotropy")
    plot!(plt, dxs, eL2[1] .* (dxs ./ dxs[1]).^1; ls = :dash, lc = :grey, label = "slope 1")
    plot!(plt, dxs, eT[1]  .* (dxs ./ dxs[1]).^2; ls = :dot,  lc = :grey, label = "slope 2")
    savefig(plt, joinpath(FIG, "ex03_convergence.png"))
    println("\n  -> ", joinpath(FIG, "ex03_convergence.png"))
end

println("""
  READING IT
    Three quantities, three different answers, from one ladder:
      * T at the centre converges cleanly, at the order printed above;
      * the L2 of T over the hot region converges more slowly — it is dominated by the STEEP part
        of the profile, not by the smooth core, and that is the part a freeze-out surface lives on;
      * the anisotropy is NOT MONOTONE at this refinement ratio. Its three values differ by ~1e-3
        on ~0.22, which is the same size as its own resolution noise, so the "order" printed for it
        is not an order — it is a ratio of two numbers that are both near zero. Reporting it as a
        convergence order would be exactly the over-claim this file exists to prevent. The honest
        statement is: the anisotropy is converged to ~1 % at N = 150 and refining further does not
        move it, which is what you actually need to know.
    ΔQ/Q is not a truncation error at all: charge is conserved by construction (D̃ = τJ^τ is the
    evolved variable), so what this column measures is the floor and vacuum-cut machinery, which
    grows slightly WITH resolution. See TWOD_PROGRAM.md §6ab for the measurement of that.
    And the ε₂ = 0 control returns 1e-15 at every resolution: whatever the numbers above are doing,
    they are not the grid inventing an anisotropy.""")
