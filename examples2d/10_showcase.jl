#!/usr/bin/env julia
#=
10 — the showcase: one real Pb+Pb event, every sector on, the vorticity coupling enabled, rendered
     for the eye rather than for a table.

Examples 07, 08 and 09 each measure ONE thing and draw the minimum needed to check it. This one is
the opposite: it is the picture you put on a slide. Same solver, same event, nothing switched off,
`m2_vorticity = true`, run long (τ → 8 fm/c) at high resolution, with the fields chosen because the
structure is beautiful and survives to late times.

⚠ WHAT "VORTICITY ENABLED" DOES AND DOES NOT DO. The charm second moment is PASSIVE at c_M = 0: it
is driven by the medium and feeds nothing back. So `m2_vorticity = true` moves π_Q and Π_Q and
NOTHING else — T, u^i and ν^i are bit-identical to a run with it off (example 08 measures exactly
that: ΔT/T = Δν/ν = 0 at every frame, to the last bit). A temperature movie therefore looks the
same either way. The panel that actually shows the term is the charm-stress one, and it is here for
that reason. Do not read the fireball panels as evidence of vorticity; read the π_Q one.

THE FIELDS, AND WHY THESE:

  * `n·τ` — the charm density with Bjorken dilution divided out. Raw `n` falls like 1/τ and a movie
    of it is a movie of the expansion, which the eye reads as "everything fades"; `n·τ` holds its
    scale, so what moves on screen is REDISTRIBUTION — diffusion and advection — rather than
    dilution. This is the panel where the lumps of a real event survive longest.
  * `n·(r+r₀)·τ` — the same thing weighted by radius. The r-weight is an EYE weight, not a physics
    one: it lifts the dilute outer skirt to the same visual footing as the dense core, which is
    where the late-time structure actually is (the core is smooth by τ ≈ 4, the edge never becomes
    smooth). The offset r₀ = 2 fm is there because a bare `r` sends the weight to ZERO at the origin
    and punches a black hole through the middle of the fireball — a rendering artefact that looks
    like a physical cavity. ⚠ it is NOT a conserved density and NOT the radial charm profile
    r·dN/dr — the azimuthal integral is missing. It is a rendering choice; no number comes from it.
  * `|ω|/|σ|` — where the flow swirls, the quantity example 08 measures. Shown here on the whole box
    at full strength, because the outer edge is where it is largest (see below).

⚠ THE OUTER EDGE IS THE MOST VORTICAL PART OF THE BOX, and that is a measurement, not a look. At
τ ≈ 7.6 on this event the cells with |ω|/|σ| > 0.8 sit at r ≈ 9–12 fm and carry |σ| = 2.0e-1 against
an all-cell median 1.07e-1 — TWICE the median, not a vanishing denominator — while |ω| = 2.4e-1
against a median 6.7e-4, a factor ~350 in the numerator. The fluid's edge swirls far harder than the
fireball it came from. `examples2d/08_vorticity.jl` with `EX08_DIAG=1` prints that check.

RESOLUTION AND TIME are ENV-overridable; the committed defaults are a compromise that still runs in
a couple of minutes. The slide version:

    EX10_N=480 EX10_NFRAME=80 EX10_SIZE=900 EX10_FPS=14 julia -t auto --project=. examples2d/10_showcase.jl

Every knob goes into the filename, so no two renders overwrite each other.

This is an EXAMPLE, not a gate. The validation ladder is ../test/run2d_gates.jl; the term shown here
is gated by Gt4 in ../test/test_terms2d.jl and measured by example 08.
=#
ENV["GKSwstype"] = "100"
using Printf, Statistics, Plots   # RGB and cgrad come from Plots' re-export of Colors
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
const N      = parse(Int,     get(ENV, "EX10_N", "240"))
const NFRAME = parse(Int,     get(ENV, "EX10_NFRAME", "48"))
const ASIZE  = parse(Int,     get(ENV, "EX10_SIZE", "640"))
const FPS    = parse(Int,     get(ENV, "EX10_FPS", "12"))
const TAUF   = parse(Float64, get(ENV, "EX10_TAUF", "8.0"))
const LBOX, TAU0, T_FO, DST = 14.0, 0.4, 0.1565, 0.1163
const T_VAC  = 0.050                      # GeV: below this there is no fluid, only the floor
const SFX    = string("_N$(N)", TAUF == 8.0 ? "" : @sprintf("_tau%.0f", TAUF))
const MUTED  = colorant"#8a8980"

# ── the run: everything on, vorticity included ──────────────────────────────────────────────────
function run_showcase()
    g = H.make_grid2d(N, N; xmax = LBOX, ymax = LBOX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = DST,
                           consistent_fm = true, consistent_m2 = true,
                           pi_clip_factor = 1.0,                 # a real lumpy event needs it
                           terms = (m2_vorticity = true,))       # ← the point of this file
    U = H.allocate_state(g, m)
    H.initialize_from_grid_csv!(U, g, m, TAU0, EVENT; smooth_fm = 0.4)   # example 07 explains σ
    frames = NamedTuple[]
    on_dump = (τ, Uu, wk) -> begin
        f = H.fields_2d(g, Uu, wk, m)
        push!(frames, (τ = τ, T = f.T, n = f.n, ux = f.ux, uy = f.uy,
                       pQxx = f.pQxx, pQxy = f.pQxy, pQyy = f.pQyy))
    end
    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15,
                        on_dump = on_dump, dump_dt = (TAUF - TAU0)/(NFRAME - 1))
    res.ok || error("run failed at τ = $(res.τ)")
    return (; g, m, frames, secs = time() - t0)
end

println("\n  SHOWCASE — one real Pb+Pb event (20-30 %), every sector on, vorticity coupling ENABLED")
@printf("  %d² on ±%.0f fm, τ %.1f → %.1f, %d frames\n", N, LBOX, TAU0, TAUF, NFRAME)
run = run_showcase()
@printf("  ran in %.1f s\n", run.secs)

xs  = run.g.xC[(run.g.nghost+1):(run.g.nghost+run.g.Nx)]
rad = [hypot(x, y) for x in xs, y in xs]

# ── |ω| and |σ| from consecutive dumps (example 08 derives this; same algebra) ───────────────────
"""‖ω‖ and ‖σ‖ on the grid from two dumps a real interval apart — ∂_τu^i is a difference, not zero.
⚠ ω is NOT ∂_xu^y − ∂_yu^x: a boost-invariant transverse flow carries the acceleration terms too."""
function kinematic_norms(fa, fb, τ, dτ, dx, dy)
    nx, ny = size(fa.T)
    ωn = zeros(nx, ny); σn = zeros(nx, ny)
    gm = (-1.0, 1.0, 1.0)
    for i in 2:nx-1, j in 2:ny-1
        ux = fa.ux[i,j]; uy = fa.uy[i,j]; ut = sqrt(1 + ux^2 + uy^2)
        dtux = (fb.ux[i,j] - ux)/dτ;  dtuy = (fb.uy[i,j] - uy)/dτ
        dxux = (fa.ux[i+1,j] - fa.ux[i-1,j])/(2dx); dxuy = (fa.uy[i+1,j] - fa.uy[i-1,j])/(2dx)
        dyux = (fa.ux[i,j+1] - fa.ux[i,j-1])/(2dy); dyuy = (fa.uy[i,j+1] - fa.uy[i,j-1])/(2dy)
        dtut = (ux*dtux + uy*dtuy)/ut
        dxut = (ux*dxux + uy*dxuy)/ut
        dyut = (ux*dyux + uy*dyuy)/ut
        ax = ut*dtux + ux*dxux + uy*dyux
        ay = ut*dtuy + ux*dxuy + uy*dyuy
        at = (ux*ax + uy*ay)/ut
        θ  = dtut + dxux + dyuy + ut/τ
        A = ((-dtut + ut*at, -dtux + ut*ax, -dtuy + ut*ay),
             ( dxut + ux*at,  dxux + ux*ax,  dxuy + ux*ay),
             ( dyut + uy*at,  dyux + uy*ax,  dyuy + uy*ay))
        u = (ut, ux, uy)
        ω2 = 0.0; σ2 = 0.0
        for α in 1:3, ν in 1:3
            Δαν = (α == ν ? gm[α] : 0.0) + u[α]*u[ν]
            w  = 0.5*(A[α][ν] - A[ν][α])
            sg = 0.5*(A[α][ν] + A[ν][α]) - Δαν*θ/3
            ω2 += w*w*gm[α]*gm[ν]
            σ2 += sg*sg*gm[α]*gm[ν]
        end
        σ2 += (ut/τ - θ/3)^2
        ωn[i,j] = sqrt(max(ω2, 0.0)); σn[i,j] = sqrt(max(σ2, 0.0))
    end
    return ωn, σn
end

# ── the fields ──────────────────────────────────────────────────────────────────────────────────
# ⚠ THE SOLVER CAN EMIT A DUPLICATE FINAL DUMP: with `dump_dt = (τf − τ0)/(nframe − 1)` the last
# scheduled dump and the end-of-run dump can both land on τf, giving two frames at the SAME τ. ω and
# σ here are built from CONSECUTIVE dumps — ∂_τu^i is a real difference — so a zero interval divides
# by zero and turns that whole frame into NaN, which `heatmap` then paints as the colormap's bottom
# colour: a completely black panel that looks like "no vorticity" rather than like a bug. Drop any
# pair that is not separated in time.
dx = xs[2] - xs[1]
nF = let m = length(run.frames) - 1
    while m > 1 && run.frames[m+1].τ - run.frames[m].τ <= 0
        m -= 1
    end
    m
end
length(run.frames) - 1 > nF &&
    @printf("  (dropped %d duplicate final dump(s) at τ = %.3f)\n",
            length(run.frames) - 1 - nF, run.frames[end].τ)
τs = [run.frames[k].τ for k in 1:nF]
Ts = [run.frames[k].T for k in 1:nF]

const R0 = 2.0                                   # fm: keeps the r-weight off zero at the origin
ntau  = [run.frames[k].n .* run.frames[k].τ           for k in 1:nF]   # dilution divided out
nrtau = [ntau[k] .* (rad .+ R0)                       for k in 1:nF]   # ... and radius-weighted
ratio = Matrix{Float64}[]
for k in 1:nF
    ωn, σn = kinematic_norms(run.frames[k], run.frames[k+1], τs[k],
                             run.frames[k+1].τ - τs[k], dx, dx)
    push!(ratio, map((w, s) -> w/max(s, 1e-30), ωn, σn))
end
# the charm shear stress, the one field the vorticity term actually moves
pQ = [map(hypot, run.frames[k].pQxx, run.frames[k].pQxy) for k in 1:nF]

# ── rendering: fade to the vacuum floor, never a hard rim ────────────────────────────────────────
# The fade is the lesson of examples 08/09: a `T > T_fo ? v : NaN` mask draws the fireball with a
# razor edge that moves frame to frame and reads as an artefact of the plot. Here the colour is
# faded to the background as T approaches the vacuum floor, so the fluid ends where the fluid ends.
# GR gives a heatmap only ONE scalar alpha for the whole series (a per-cell matrix is silently
# collapsed — measured), so each panel is an RGB image; the colorbar then needs its own zero-size
# series, since Plots has no bar for an image.
# ⚠ the weight touches the ALPHA only, never the value: the bright outer edge of |ω|/|σ| is real
# (see the header), and damping it would erase a measurement to tidy a picture.
const BG = RGB(1.0, 1.0, 1.0)

function faded_image(Z, T, cmap, lo, hi)
    g = cgrad(cmap)
    img = Matrix{RGB{Float64}}(undef, size(Z, 2), size(Z, 1))
    for j in axes(Z, 2), i in axes(Z, 1)
        w = clamp((T[i,j] - 0.6T_VAC)/(0.4T_VAC), 0.0, 1.0)
        c = RGB(get(g, clamp((Z[i,j] - lo)/(hi - lo), 0.0, 1.0)))
        img[j, i] = RGB(w*c.r + (1-w)*BG.r, w*c.g + (1-w)*BG.g, w*c.b + (1-w)*BG.b)
    end
    return img
end

"one panel: the faded image, a colorbar, and the freeze-out contour as a LINE"
function panel(Z, T, cmap, hi, label, title; lo = 0.0, contour_fo = true)
    p = plot(xs, xs, faded_image(Z, T, cmap, lo, hi); aspect_ratio = 1, yflip = false,
             xlims = (-LBOX, LBOX), ylims = (-LBOX, LBOX), xlabel = "x  [fm]", ylabel = "y  [fm]",
             title = isempty(label) ? title : "$title\n$label",
             titlefontsize = 10, framestyle = :box, legend = false)
    # ⚠ NO `colorbar_title`. GR puts it immediately right of the colorbar's TICK LABELS, and these
    # panels carry long ones (0.00050, 0.225) — the rotated title then sits ON the numbers. Shrinking
    # the title font does not help, because the collision is with the ticks, not the panel. The
    # quantity goes in the panel TITLE instead, which also reads better on a slide.
    scatter!(p, [NaN], [NaN]; zcolor = [lo], c = cmap, clims = (lo, hi), ms = 0,
             colorbar = true, label = "")
    contour_fo && contour!(p, xs, xs, permutedims(T); levels = [T_FO], c = :white, lw = 1,
                           alpha = 0.7, colorbar_entry = false)
    return p
end

# ⚠ the scale is a quantile over the cells that HOLD FLUID, fixed across frames. Per-frame
# rescaling makes a cooling fireball look static, and including the vacuum floor flattens every
# frame to one colour — both learned in
# Projects/FiVoFluidumComparison/animate_fluctuating_fields.jl.
#
# ⚠ AND THE QUANTILE IS NOT ALWAYS p99.5. The charm densities are smooth and a p99.5 frames them
# well. |ω|/|σ| and |π_Q| are not: both span orders of magnitude, with a thin filamentary tail that
# carries the top percentile (|ω|/|σ| reaches ~1 at the fluid's edge while the fireball sits at
# ~1e-2). Scaled to p99.5 those panels render BLACK — the whole fireball crushed into the bottom
# 1 % of the colour map by a few edge cells. A p98/p90 puts the structure on screen and lets the
# extreme filaments saturate, which is the right trade for a picture: the extremes are measured in
# example 08, not read off a colour here.
function scale(zs; q = 0.995)
    v = filter(isfinite, reduce(vcat, [vec(zs[k][Ts[k] .> T_VAC]) for k in 1:nF]))
    return quantile(v, q)
end

function animate(zs, tag, label, cmap, hi)
    anim = @animate for k in 1:nF
        panel(zs[k], Ts[k], cmap, hi, label,
              @sprintf("τ = %5.2f fm/c    (N = %d, vorticity ON)", τs[k], N))
        plot!(; size = (ASIZE, round(Int, 0.86ASIZE)))
    end
    path = joinpath(ANIM, "ex10_$(tag)$(SFX).gif"); gif(anim, path; fps = FPS, show_msg = false)
    return path
end

hi_nt, hi_nrt = scale(ntau), scale(nrtau)                  # smooth fields: p99.5
# ⚠ these two quantiles are RESOLUTION-DEPENDENT and were tuned at N = 400. Refinement resolves
# more filaments, so the same quantile cuts lower relative to the structure and the panel saturates
# into flat white: p90 was right at N = 100 and washed out at N = 400. If you change N a lot, look
# at the picture and move them. They set nothing but the colour.
hi_r,  hi_pQ  = scale(ratio; q = 0.98), scale(pQ; q = 0.90)  # filamentary fields (see above)
@printf("\n  colour scales, fixed across frames, over cells with T > %.0f MeV:\n", 1e3T_VAC)
@printf("    n·τ  %.3e (p99.5)   n·(r+r₀)·τ  %.3e (p99.5)\n", hi_nt, hi_nrt)
@printf("    |ω|/|σ|  %.3e (p98)   |π_Q|  %.3e (p90)   — these two span decades; p99.5 would\n",
        hi_r, hi_pQ)
@printf("    render them black (p99.5 is %.3e and %.3e, set by a few edge filaments)\n",
        scale(ratio), scale(pQ))

paths = [animate(ntau,  "charm_ntau",  "n · τ   [fm⁻²]",        :turbo,   hi_nt),
         animate(nrtau, "charm_nrtau", "n · (r+r₀) · τ  [fm⁻¹]", :turbo,  hi_nrt),
         animate(ratio, "omega_sigma", "|ω| / |σ|",             :magma,   hi_r),
         animate(pQ,    "charm_stress","|π_Q|  (vorticity ON)", :viridis, hi_pQ)]
println("\n  wrote")
for p in paths; println("        ", p); end

# ── the static figure: the four fields at the last frame ────────────────────────────────────────
# ⚠ 2x2, not 1x4. Four square panels in a row leave the figure mostly white margin and shrink each
# map to a thumbnail; the eye needs the panels large, since the whole point here is the structure.
k = nF
plt = plot(panel(ntau[k],  Ts[k], :turbo,   hi_nt,  "n · τ",       "charm density, dilution removed"),
           panel(nrtau[k], Ts[k], :turbo,   hi_nrt, "n·(r+r₀)·τ",  "... weighted toward the edge"),
           panel(ratio[k], Ts[k], :magma,   hi_r,   "|ω| / |σ|",   "where the flow swirls"),
           panel(pQ[k],    Ts[k], :viridis, hi_pQ,  "|π_Q|",       "charm shear stress (vorticity ON)");
           layout = (2, 2), size = (1150, 1000), left_margin = 8Plots.mm,
           bottom_margin = 8Plots.mm, right_margin = 4Plots.mm)
let f = joinpath(FIG, "ex10_showcase$(SFX).png")
    savefig(plt, f); println("        ", f)
end

@printf("""

  READING IT
    n·τ is the charm density with the Bjorken 1/τ divided out, so what moves is redistribution and
    not dilution — the lumps of a real event, surviving. The second panel weights it by (r + 2 fm),
    which is an EYE weight: it lifts the dilute skirt onto the same footing as the core, because by
    τ ≈ 4 fm/c the core is smooth and the edge is where the structure still lives. The offset is not
    cosmetic fussiness — a bare r zeroes the weight at the origin and draws a black cavity through
    the fireball's middle. Neither field is a conserved density; nothing is quoted from them.

    |ω|/|σ| is the one with the physics. It is largest at the FLUID'S OUTER EDGE, not in the
    fireball: at r ≈ 9–12 fm the vorticity runs ~350x the all-cell median while the shear runs only
    2x it (example 08, EX08_DIAG=1). That edge is one or two cells thick at this resolution, which
    is why it looks grainy — refine and it sharpens rather than dissolves.

    |π_Q| is the ONLY panel the vorticity coupling changes. At c_M = 0 the charm second moment is
    passive: switching `m2_vorticity` moves π_Q and Π_Q and leaves T, u and the current bit-for-bit
    identical (example 08 measures ΔT/T = Δν/ν = 0 exactly). So the fireball panels would look the
    same with the term off — they are here because they are the event, not because they are evidence.

    The colour scales are printed above and fixed across frames. The two smooth charm fields take a
    p99.5; |ω|/|σ| a p98 and |π_Q| a p90, because they span decades and a p99.5 set by a few edge
    filaments renders the whole fireball black. That means the brightest filaments SATURATE in those
    two panels — deliberately. Read magnitudes from example 08's diagnostics, never off a colour.
""")
