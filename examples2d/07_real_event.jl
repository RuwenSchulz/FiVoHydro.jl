#!/usr/bin/env julia
#=
07 — a real Pb+Pb event, from the file to freeze-out, as animations.

Examples 02 and 04 build their initial conditions in this file. This one reads a REAL one: a single
un-averaged MC-Glauber event written by `Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl`, a CSV of
`x,y,T0,alpha0` carrying genuine ε₂ and ε₃ and genuine lumps. It runs to freeze-out and writes two
animations — the temperature with its freeze-out contour, and the charm density.

WHAT THE EXAMPLE IS ACTUALLY ABOUT: the `smooth_fm` argument of `initialize_from_grid_csv!`. A real
event carries sub-nucleon structure (`BuildIC2D.jl` deposits W = 0.5 fm sources) that a dx ≳ 0.2 fm
grid cannot resolve, and an unresolved hot spot is what drives |π| past P and puts the run on the
regulators. The file is loaded twice, raw and lightly blurred, and both runs are measured: what the
blur costs in ε₂, ε₃ and the flow response against what it buys in max|u| and recovery failures.
Neither is "the" right choice; the point is that it is a choice, and that it is cheap to measure.

⚠ THE COLOUR SCALE IS FIXED ACROSS FRAMES, and taken from cells ABOVE FREEZE-OUT. Rescaling per
frame makes a cooling fireball look static — the eye reads the colour, not the colourbar. And the
scale must not come from the whole grid: the vacuum tail carries |u| several times the fireball's
(COMPARISON_2P1D.md §35.3), and one such cell flattens every frame to one colour. Both lessons come
from `Projects/FiVoFluidumComparison/animate_fluctuating_fields.jl`, where they were learned.

⚠ THE ANIMATIONS ARE NOT TRACKED BY GIT (`figures/anim/` is ignored). They are regenerable, and
this package untracked 6.9 MB of figures once already.

Equations and knobs: ../EQUATIONS2D.md, ../README2D.md.
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
include(joinpath(_ROOT, "main2D.jl")); using .hydro2d; const H = hydro2d
const FIG  = joinpath(@__DIR__, "figures");  isdir(FIG)  || mkpath(FIG)
const ANIM = joinpath(FIG, "anim");          isdir(ANIM) || mkpath(ANIM)

const EVENT = joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation", "PbPb", "data",
                       "ic2d_20-30_ev01.csv")
const N, LBOX, TAU0, TAUF, T_FO = 200, 14.0, 0.4, 8.0, 0.1565
const NFRAME, FPS = 40, 10
const INK, MUTED = colorant"#1a1a19", colorant"#8a8980"

# ── one run of the event ────────────────────────────────────────────────────────────────────────
function run_event(smooth_fm; nframe = NFRAME)
    g = H.make_grid2d(N, N; xmax = LBOX, ymax = LBOX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = 0.1163,
                           pi_clip_factor = 1.0)     # a lumpy event needs it (example 04)
    U = H.allocate_state(g, m)
    H.initialize_from_grid_csv!(U, g, m, TAU0, EVENT; smooth_fm = smooth_fm)
    frames = NamedTuple[]
    on_dump = (τ, Uu, wk) -> begin
        f = H.fields_2d(g, Uu, wk, m)
        push!(frames, (τ = τ, T = f.T, n = f.n, e = f.e, ux = f.ux, uy = f.uy))
    end
    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15,
                        on_dump = on_dump, dump_dt = (TAUF - TAU0)/(nframe - 1))
    res.ok || error("run failed at τ = $(res.τ)")
    xs = H.fields_2d(g, U, res.work, m).x
    return (; g, m, U, res, frames, x = xs, y = xs, secs = time() - t0)
end

"ε_n and Ψ_n of the energy density, about its own centre of mass."
function eccentricity(x, y, e, n)
    cx = 0.0; cy = 0.0; mtot = 0.0
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        d = e[i,j]; mtot += d; cx += d*xi; cy += d*yj
    end
    cx /= mtot; cy /= mtot
    num = 0.0 + 0im; den = 0.0
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        d = e[i,j]; r = hypot(xi-cx, yj-cy); φ = atan(yj-cy, xi-cx)
        num += d*r^n*cis(n*φ); den += d*r^n
    end
    (abs(num)/den, angle(num)/n + π/n)
end

"""
    momentum_anisotropy(f) -> ε_p

Energy-weighted momentum anisotropy over cells above freeze-out,

    ε_p = |Σ e u⊥² e^{2iφ_u}| / Σ e u⊥² ,

which is example 02's `(⟨u_x²⟩−⟨u_y²⟩)/(⟨u_x²⟩+⟨u_y²⟩)` written so that it does not depend on how
the event happens to lie on the grid — a real event's Ψ₂ points somewhere random, and the aligned
form would throw most of the signal away (example 04's header makes the same point about ε₃).

⚠⚠ `Tcut` DEFAULTS TO ZERO — the whole fireball — AND THAT MATTERS LATE. With the freeze-out mask
(`Tcut = T_FO`) this quantity RISES steeply after τ ≈ 6 on this event, and the rise is the MASK, not
the flow: as the interior cools through T_fo the surviving cells are the fast, anisotropic outer
edge, so the average climbs while the real anisotropy falls. Measured on the smoothed run:

      τ        ε_p(T > T_fo)   cells      ε_p(no mask)   cells
      5.67        0.257         8454         0.229       15322
      8.00        0.457         5291         0.213       22094

Same run, same field, opposite conclusions. It is this repo's standing failure mode — a statistic
that moves because its DENOMINATOR moved (CLAUDE.md) — and the reason TWOD_PROGRAM.md §6 says the
T_fo-restricted v₂ is not usable late and to use ε_p over the fireball. The figure draws both, so
the artefact is visible rather than described.
"""
function momentum_anisotropy(f; Tcut = 0.0)
    num = 0.0 + 0im; den = 0.0
    for i in eachindex(f.T)
        f.T[i] < Tcut && continue
        u2 = f.ux[i]^2 + f.uy[i]^2; u2 < 1e-24 && continue
        w = f.e[i]*u2; den += w
        num += w*cis(2*atan(f.uy[i], f.ux[i]))
    end
    den > 0 ? abs(num)/den : 0.0
end

"max over cells above freeze-out — never over the grid (see the header)."
function max_above_fo(A, T)
    m = 0.0
    for i in eachindex(A)
        T[i] > T_FO && (m = max(m, abs(A[i])))
    end
    return m
end

"area above freeze-out, in fm²"
area_above_fo(f, dx, dy) = count(>(T_FO), f.T)*dx*dy

# ── the two runs ────────────────────────────────────────────────────────────────────────────────
println("\n  A REAL Pb+Pb EVENT (20-30 %, ic2d_20-30_ev01.csv): raw vs lightly smoothed")
@printf("  %d² on ±%.0f fm, τ %.1f → %.1f, shear+bulk+diffusion, pi_clip_factor = 1\n\n",
        N, LBOX, TAU0, TAUF)
raw = run_event(0.0)
smo = run_event(0.4)

function report(r)
    f0 = r.frames[1]
    dxr = r.x[2] - r.x[1]
    usable = [k for k in eachindex(r.frames) if area_above_fo(r.frames[k], dxr, dxr) >= 25.0]
    fe = r.frames[isempty(usable) ? length(r.frames) : usable[end]]   # last frame the proxy can read
    e2, _ = eccentricity(r.x, r.y, f0.e, 2)
    e3, _ = eccentricity(r.x, r.y, f0.e, 3)
    εp = momentum_anisotropy(fe)
    dx = r.x[2] - r.x[1]
    (T0max = maximum(f0.T), e2 = e2, e3 = e3, εp = εp,
     umax = max_above_fo(hypot.(fe.ux, fe.uy), fe.T),
     pf = r.res.nprimfail, dQ = r.res.dQ, secs = r.secs,
     area = area_above_fo(fe, dx, dx), τread = fe.τ)
end
R, S = report(raw), report(smo)
@printf("  %-26s %12s %12s\n", "", "raw", "σ = 0.4 fm")
for (lbl, k, fmt) in (("T_max(τ₀)  [GeV]", :T0max, "%12.4f"), ("ε₂ of the IC", :e2, "%12.4f"),
                      ("ε₃ of the IC", :e3, "%12.4f"), ("ε_p at τ_f", :εp, "%12.4f"),
                      ("max |u| above T_fo", :umax, "%12.3f"),
                      ("area above T_fo  [fm²]", :area, "%12.1f"),
                      ("recovery failures", :pf, "%12d"), ("charge drift ΔQ/Q", :dQ, "%12.2e"),
                      ("τ of the ε_p read  [fm/c]", :τread, "%12.2f"),
                      ("wall time  [s]", :secs, "%12.1f"))
    @eval @printf($("  %-26s " * fmt * fmt * "\n"), $lbl, $(getfield(R, k)), $(getfield(S, k)))
end
@printf("\n  the blur moves  ε₂ %+.1f %%   ε₃ %+.1f %%   ε_p %+.1f %%\n",
        100*(S.e2/R.e2 - 1), 100*(S.e3/R.e3 - 1), 100*(S.εp/R.εp - 1))
@printf("  and the RESPONSE  ε_p/ε₂  %.3f → %.3f  (%+.1f %%) — the medium's answer with the initial\n",
        R.εp/R.e2, S.εp/S.e2, 100*((S.εp/S.e2)/(R.εp/R.e2) - 1))
println("  state divided out, which is what says whether the blur changed the PHYSICS or the INPUT.")

# ── animations ──────────────────────────────────────────────────────────────────────────────────
# One fixed colour range per animation, from the frames themselves, over cells above freeze-out.
function animate_field(run, getfield_, tag, label, cmap; logscale = false)
    hi = maximum(max_above_fo(getfield_(f), f.T) for f in run.frames)
    lo = logscale ? hi*1e-3 : 0.0
    anim = @animate for f in run.frames
        Z = permutedims(getfield_(f))                     # heatmap wants [iy, ix]
        Z = logscale ? clamp.(Z, lo, hi) : Z
        heatmap(run.x, run.y, Z; c = cmap, clims = (lo, hi), aspect_ratio = 1,
                xlims = (-12, 12), ylims = (-12, 12), xlabel = "x  [fm]", ylabel = "y  [fm]",
                colorbar_title = label, size = (560, 480),
                title = @sprintf("τ = %5.2f fm/c    max = %.3g", f.τ, max_above_fo(getfield_(f), f.T)),
                titlefontsize = 10, colorbar_scale = logscale ? :log10 : :identity)
        contour!(run.x, run.y, permutedims(f.T); levels = [T_FO], c = MUTED, lw = 1.5,
                 colorbar_entry = false)
    end
    path = joinpath(ANIM, "ex07_$(tag).gif")
    gif(anim, path; fps = FPS, show_msg = false)
    return path
end

p1 = animate_field(smo, f -> f.T, "temperature", "T  [GeV]", :inferno)
p2 = animate_field(smo, f -> f.n, "charm_density", "n  [fm⁻³]", :viridis)
println("\n  wrote ", p1, "\n        ", p2)

# ── a static figure: the two initial conditions, and the flow response in time ──────────────────
f0r, f0s = raw.frames[1], smo.frames[1]
cl = (0.0, maximum(f0r.T))
pa = heatmap(raw.x, raw.y, permutedims(f0r.T); c = :inferno, clims = cl, aspect_ratio = 1,
             xlims = (-10, 10), ylims = (-10, 10), xlabel = "x  [fm]", ylabel = "y  [fm]",
             title = "raw event, τ₀", titlefontsize = 10, colorbar_title = "T  [GeV]")
pb = heatmap(smo.x, smo.y, permutedims(f0s.T); c = :inferno, clims = cl, aspect_ratio = 1,
             xlims = (-10, 10), ylims = (-10, 10), xlabel = "x  [fm]",
             title = "smoothed, σ = 0.4 fm", titlefontsize = 10, colorbar_title = "T  [GeV]")
τs = [f.τ for f in smo.frames]
pc = plot([f.τ for f in raw.frames], [momentum_anisotropy(f) for f in raw.frames];
          label = "raw event", c = colorant"#2a78d6", ls = :dash,
          xlabel = "τ  [fm/c]", ylabel = "ε_p  (energy-weighted)",
          title = "the momentum anisotropy, and a mask artefact", titlefontsize = 10,
          legend = :topleft, ylims = (0, 0.55))
plot!(pc, τs, [momentum_anisotropy(f) for f in smo.frames]; label = "smoothed, σ = 0.4 fm",
      c = colorant"#2a78d6")
plot!(pc, τs, [momentum_anisotropy(f; Tcut = T_FO) for f in smo.frames];
      label = "smoothed, restricted to T > T_fo", c = MUTED, ls = :dot, lw = 2)
annotate!(pc, 2.2, 0.055, text("the dotted rise is the MASK eating the interior,\nnot the flow — the source has the numbers", 7, :left, MUTED))
plt = plot(pa, pb, pc; layout = @layout([a b c]), size = (1450, 430),
           left_margin = 6Plots.mm, bottom_margin = 7Plots.mm)
savefig(plt, joinpath(FIG, "ex07_real_event.png"))
println("        ", joinpath(FIG, "ex07_real_event.png"))

println("""

  READING IT
    The blur is a REGULARISATION OF STRUCTURE THE GRID CANNOT CARRY, not a physics model, and the
    table above is what it did — read it, do not assume it. Two things are worth knowing in advance:

    * ε₂ is GEOMETRY and barely moves. ε₃ can move either way, and here it goes UP: ε_n carries an
      r^n weight, so ε₃ is dominated by the outskirts, while the blur mostly lowers the central peak
      — which is nearly isotropic and therefore contributes little to ε₃ but plenty to its
      normalisation. "Smoothing removes lumps, so ε₃ must fall" is the intuition; it is not what the
      measurement says on this event.
    * The RESPONSE ε_p/ε₂ is the medium's answer with the initial state divided out, so it is the
      number that says whether the blur changed the PHYSICS or only the input.
    * ε_p is measured over the FIREBALL, not over T > T_fo. The dotted curve in panel (c) is the
      masked version of the same run: it doubles after τ ≈ 6 purely because the mask is eating the
      interior. Any late-time number from a masked average needs its cell count printed beside it.

    ⚠ It smooths T and α as given, NOT the entropy density the event was built from (EQUATIONS2D
    and the `initialize_from_grid_csv!` docstring say so); at σ = 0.4 fm that is a reshaping of the
    peak, at σ ≳ 1 fm it would be a different initial state.""")
