#!/usr/bin/env julia
#=
05 — the dissipative sectors one at a time: what each does, what each costs, and the regulators.

`build_model_2d` starts from the BARE ideal scheme and every sector is opt-in. This file turns them
on one at a time on the same fireball and reports, for each: the change in the field, the size the
dissipative fields actually reach, the cost, and how hard the safety machinery had to work.

⚠ REPORT |π|/P, NOT π. The dimensionless ratio is what says whether second-order hydrodynamics is
being used inside or outside its domain, and it is what the regulators key off. On a production IC
it reaches ~0.5 in the dilute edge and ~1 transiently at high resolution (gate G5 of the ladder).

⚠ THE REGULATORS ARE OFF BY DEFAULT AND SHOULD STAY THAT WAY UNLESS MEASURED. `pi_clip_factor` and
`Pi_clip_factor` default to −1 (disabled). Switching one on changes the answer, so the honest use is
to run BOTH and show the difference is negligible — which is what the last block does. A clip that
changes the observable is not a safety net, it is a model.

⚠ `Pi_clip_factor = 1.0` DOES NOT GUARANTEE P + Π > 0. It bounds |Π| against P at the moment it is
applied; the advection step can then take P down while Π rides along (TWOD_PROGRAM.md §6r). The
run diagnostic `res.minPtot` is the thing to look at, and it is printed below.
=#
ENV["GKSwstype"] = "100"
using Printf, Plots
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
const TAU0, TAUF, RMAX, T_FO, N = 0.4, 8.0, 20.0, 0.1565, 150

function run_case(; shear, bulk, diff, piclip = -1.0, Piclip = -1.0, eps2 = 0.25)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = shear, eta_over_s = 0.10, tauShear_coeff = 0.2,
                                                 deltaShear_factor = 4/3, pi_clip_factor = piclip,
                           enable_bulk  = bulk,  zeta_over_s = 0.10, tauPi_coeff = 15.0,
                                                 Pi_clip_factor = Piclip,
                           enable_diff  = diff,  kappa_coeff = 0.1163, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    a = sqrt(1 - eps2); b = sqrt(1 + eps2)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix]/a, g.yC[iy]/b)
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    taus = Float64[]; an = Float64[]
    aniso_of(wk, Uu) = begin
        ax = 0.0; ay = 0.0; ng = g.nghost
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H.lin(g, ix, iy); exp(wk.yT[i]) < T_FO && continue
            w = Uu[m.layout.iE, i]; ax += w*wk.ux[i]^2; ay += w*wk.uy[i]^2
        end
        (ax + ay) > 0 ? (ax - ay)/(ax + ay) : 0.0
    end
    t = @elapsed res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15, CFLτ = 0.05,
        dump_dt = 0.25, on_dump = (τ, Uu, wk) -> (push!(taus, τ); push!(an, aniso_of(wk, Uu))))
    @assert res.ok
    ng = g.nghost; sx = 0.0; sy = 0.0; mpi = 0.0; mPi = 0.0; T0 = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy); T = exp(res.work.yT[i])
        hypot(g.xC[ix], g.yC[iy]) < 2RMAX/N && (T0 = T)
        T < T_FO && continue
        P = res.work.P[i]
        w = U[m.layout.iE, i]; sx += w*res.work.ux[i]^2; sy += w*res.work.uy[i]^2
        if shear
            mpi = max(mpi, abs(H.phys_from_stored(U[m.layout.iPieta, i]))/P)
        end
        bulk && (mPi = max(mPi, abs(H.phys_from_stored(U[m.layout.iPi, i]))/P))
    end
    (aniso = (sx+sy) > 0 ? (sx-sy)/(sx+sy) : 0.0, T0 = T0, maxpi = mpi, maxPi = mPi,
     wall = t, res = res, taus = taus, an = an)
end

println("\n  the sector ladder, ε₂ = 0.25, N = $N, τ = $TAU0 → $TAUF fm/c")
@printf("  %-26s %10s %10s %9s %9s %8s %9s %8s\n",
        "sectors", "T(0) GeV", "anisotropy", "max|π|/P", "max|Π|/P", "primfail", "min(P+Π)", "wall")
# ⚠ draw the figure FROM THESE RUNS. The first draft of this file re-ran four of the six cases just
# to plot them, which doubled the cost for numbers that were already in hand.
plt = plot(size = (760, 470), xlabel = "τ  [fm/c]", ylabel = "momentum anisotropy",
           title = "which sector moves the flow", legend = :bottomright)
for (lab, sh, bu, df) in (("ideal",                    false, false, false),
                          ("charge diffusion only",    false, false, true),
                          ("shear only",               true,  false, false),
                          ("bulk only",                false, true,  false),
                          ("shear + bulk",             true,  true,  false),
                          ("shear + bulk + diffusion", true,  true,  true))
    r = run_case(; shear = sh, bulk = bu, diff = df)
    @printf("  %-26s %10.6f %+10.5f %9.3f %9.3f %8d %9.2e %7.1fs\n",
            lab, r.T0, r.aniso, r.maxpi, r.maxPi, r.res.nprimfail,
            isfinite(r.res.minPtot) ? r.res.minPtot : NaN, r.wall)
    lab in ("ideal", "shear only", "bulk only", "shear + bulk") && plot!(plt, r.taus, r.an; label = lab)
end
savefig(plt, joinpath(FIG, "ex05_sectors.png"))
println("\n  are the regulators inert?  (they must be, or they are part of the model)")
@printf("  %-26s %12s %12s %12s\n", "setting", "anisotropy", "T(0)", "Δ vs off")
off = run_case(; shear = true, bulk = true, diff = true)
@printf("  %-26s %12.6f %12.6f %12s\n", "clips off (default)", off.aniso, off.T0, "—")
for (lab, pc, Pc) in (("pi_clip = 1.0", 1.0, -1.0), ("Pi_clip = 1.0", -1.0, 1.0),
                      ("both = 1.0",    1.0,  1.0))
    r = run_case(; shear = true, bulk = true, diff = true, piclip = pc, Piclip = Pc)
    @printf("  %-26s %12.6f %12.6f %12.2e\n", lab, r.aniso, r.T0, abs(r.aniso - off.aniso))
end

println("\n  -> ", joinpath(FIG, "ex05_sectors.png"))
println("""
  READING IT
    Measured on this fireball, at N = $N:
      * shear is the sector that moves the anisotropy, and it moves it DOWN (0.26601 -> 0.25025):
        it resists the differential expansion that builds the anisotropy, which is why v2
        constrains eta/s;
      * bulk ALONE moves it UP (0.26601 -> 0.27749), and bulk ON TOP OF shear moves it down
        (0.25025 -> 0.23948). The sectors are not additive, and quoting either one alone as
        "the bulk effect" would be wrong in sign;
      * none of them moves T(0), which spans 0.19200-0.19232 across all six settings. Bulk does
        work against the expansion, but on this fireball that shows up in the flow, not the
        central temperature;
      * charge diffusion changes neither the anisotropy nor T(0) at this level -- it is a passive
        tracer riding the medium, and the reason it is here at all is the charm sector.
    The clip block is the pattern to copy for any regulator: run with and without, quote the
    difference. Here it is 5.8e-6 on 0.239, i.e. inert -- which is the condition for calling them
    safety nets rather than model choices.
    CROSS-CODE NOTE. Fluidum's 2+1D solver, run bulk-only on an equivalent elliptic fireball, does
    NOT reproduce this: its anisotropy collapses by a factor 50 on a coarse grid and recovers under
    refinement (Julia/Fluidum.jl/examples/2p1d_viscous/03_elliptic_fireball.jl). FiVo's bulk-only
    column above is what a converged answer looks like, and it is the comparison that identified
    the other code's number as a grid artefact rather than as physics.""")
