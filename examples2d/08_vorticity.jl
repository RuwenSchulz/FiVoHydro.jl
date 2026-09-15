#!/usr/bin/env julia
#=
08 — the vorticity coupling of the charm second moment: `m2_vorticity`, on against off.

The derived second moment (M2_CONSISTENT_DERIVATION.md §4.1, EQUATIONS2D.md §5) contains

    2 τ_M π_Q^{λ⟨i} ω_λ^{j⟩} ,    ω^{αν} = ½(∇⊥^α u^ν − ∇⊥^ν u^α) ,

the same corotation term DNMR carries for the medium shear. It is OFF BY DEFAULT here, and this
example is how that default is meant to be revisited: run the same event twice and measure.

WHY IT IS OFF. It vanishes IDENTICALLY in 1+1D — a radial flow has no transverse vorticity — so the
1-D solver never needed it, both 2-D codes were written from the 1-D reduction, and neither carried
it until 2026-09-10. Every 1-D-limit gate is structurally blind to its absence, and the cross-code
gate agreed at 2.2e-16 because BOTH codes omitted it. Fluidum's `HQ_2p1d_BG_m2.jl` gained it on
2026-09-11 and the two now agree at 4.9e-16 with it ON in both (gate_2p1d_m2_newterms.jl M2) — so
turning it on no longer breaks parity, but it IS a physics decision, to be taken in both codes at
once and on a measurement. This file is that measurement.

WHAT THIS FILE MEASURES, on one real Pb+Pb event:
  * |ω| against |σ| over the fireball — the size of the term's coefficient relative to its sibling,
    the shear coupling, which is already in both codes;
  * what switching it on does to π_Q and Π_Q, in time;
  * that it does NOTHING to the medium or to the current (the second moment is passive at c_M = 0),
    which is a check on the wiring, not a physics result.

⚠ ω IS NOT ∂_x u^y − ∂_y u^x. The transverse vorticity of a boost-invariant flow carries the
acceleration terms too, ω^{xy} = ½(∂_xu^y − ∂_yu^x + u^x a^y − u^y a^x), and a^i needs ∂_τ u^i. This
file gets that from two dumps a short interval apart rather than pretending the flow is static.

HIGH RESOLUTION. `EX08_N` overrides the grid (default 160, inside the suite's time budget):

    EX08_N=480 EX08_NFRAME=60 EX08_SIZE=900 EX08_FPS=12 julia -t auto --project=. examples2d/08_vorticity.jl

N = 480 is 3x and costs ~5 min, N = 640 is 4x and ~12 min; both write their own `*_N<N>.gif` and
`ex08_vorticity_N<N>.png` so a coarse run is never overwritten by a fine one. That is not only
cosmetic: better-resolved lumps carry MORE vorticity, so the question "is this term small?" is a
question about the continuum limit, and the answer has to survive refinement. Measured at τ ≈ 5.9
on the same event:

    N     dx [fm]   |ω|/|σ| median    p90      →  π_Q moves   Π_Q moves
    160    0.175       3.01e-2      1.02e-1        1.24e-2     2.26e-4
    480    0.058       2.84e-2      9.15e-2        1.00e-2     1.66e-4
    640    0.044       2.79e-2      9.04e-2        9.91e-3     1.67e-4

It CONVERGES. 160 → 480 moves the median 6 % and the π_Q shift 19 %; 480 → 640 moves them 1.8 % and
1.0 %. So the answer is a property of the equations on this event, not of the grid: the term is a
~1 % effect on π_Q. The `max` column is the exception and is not converged — it grows with
resolution because the sharpest filaments are exactly what refinement resolves. Read the median and
the p90; the max is a thin-filament statistic with no continuum limit here.

THE BRIGHT FRINGE IS REAL, TWICE OVER. |ω|/|σ| peaks in a thin band just inside the freeze-out
contour, and again at the outer edge of the fluid — both places where a ratio is most likely to be a
vanishing denominator dressed up as a signal. Neither is (`EX08_DIAG=1` prints both checks):

  * inside T_fo, σ at the brightest cells is 1.45e-1 against a hot-cell median of 1.11e-1;
  * at the fluid's outer edge (r ≈ 9–12 fm), σ is 2.0e-1 against an all-cell median of 1.07e-1
    while |ω| is 2.4e-1 against a median of 6.7e-4 — a factor ~350 in the NUMERATOR.

The flow genuinely swirls hardest where it meets vacuum. Nor does the inner band set the headline;
dropping it moves the median 3.01e-2 → 2.60e-2. That is why the maps fade the COLOUR out past
freeze-out but never damp the VALUE.

SEEING THE WHOLE TAIL. The default maps frame the fireball and fade past freeze-out, because that is
where the quoted numbers live. `EX08_NOFADE=1` instead shows every cell that holds fluid at full
strength, out to the vacuum floor, on the whole ±14 fm box — that is how the outer vorticity above
was found. `EX08_TAUF` runs longer than the default 6 fm/c. Both change the filename, never a
number: every statistic printed is over `T > T_fo` cells either way.

    EX08_NOFADE=1 EX08_TAUF=8.0 EX08_N=640 EX08_NFRAME=80 EX08_SIZE=900 EX08_FPS=14 julia …
        → ex08_omega_over_sigma_N640_tau8_full.gif

⚠ a nofade gif and a faded one carry DIFFERENT colorbars by construction: each is scaled to the
cells it actually draws (p99.5), so brightness is not comparable between the two.

Equations: ../EQUATIONS2D.md §5. The term's gate: ../test/test_terms2d.jl (Gt4).
=#
ENV["GKSwstype"] = "100"
using Printf, Statistics, LinearAlgebra, Plots   # RGB and cgrad come from Plots' re-export of Colors
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
# Resolution and frame count are ENV-overridable: the committed defaults keep this example inside
# the suite's time budget (~30 s), and a high-quality render is one command —
#     EX08_N=480 EX08_NFRAME=60 EX08_SIZE=900 julia -t auto --project=… examples2d/08_vorticity.jl
# N = 480 is 3x the default resolution and costs ~5 min; N = 640 (4x) ~12 min. The physics question
# refinement answers is in the header: better-resolved lumps carry MORE vorticity, so |ω|/|σ| and
# the term's effect are themselves resolution-dependent, and the table below prints both so the two
# runs can be compared directly.
const N      = parse(Int, get(ENV, "EX08_N", "160"))
const NFRAME = parse(Int, get(ENV, "EX08_NFRAME", "32"))
const ASIZE  = parse(Int, get(ENV, "EX08_SIZE", "560"))
const FPS    = parse(Int, get(ENV, "EX08_FPS", "8"))
const LBOX, TAU0, T_FO, DST = 14.0, 0.4, 0.1565, 0.1163
const TAUF   = parse(Float64, get(ENV, "EX08_TAUF", "6.0"))
# `EX08_NOFADE=1` shows the maps EVERYWHERE, with no fade past freeze-out — the whole outer tail at
# full strength. The fade exists so the fireball does not end in a hard rim; switching it off is the
# way to see what is out there, which is a real question (ω does not stop at T_fo, σ does fall).
# ⚠ nothing quoted in the table is affected either way: every printed number is a statistic over
# `T > T_fo` cells, fade or no fade. This only changes what the pictures show.
const NOFADE = get(ENV, "EX08_NOFADE", "") == "1"
# every non-default knob goes in the filename, so no two runs overwrite each other's output
const SFX = string(N == 160 ? "" : "_N$(N)", TAUF == 6.0 ? "" : @sprintf("_tau%.0f", TAUF),
                   NOFADE ? "_full" : "")
const BLUE, ORANGE, MUTED = colorant"#2a78d6", colorant"#eb6834", colorant"#8a8980"

# ── the run ─────────────────────────────────────────────────────────────────────────────────────
"""One event with the charm sector on. `vort` switches the coupling; everything else is identical,
including the initial condition and the step sequence."""
function run_case(vort::Bool; nframe = NFRAME)
    g = H.make_grid2d(N, N; xmax = LBOX, ymax = LBOX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = DST,
                           consistent_fm = true, consistent_m2 = true,
                           pi_clip_factor = 1.0,
                           terms = (m2_vorticity = vort,))
    U = H.allocate_state(g, m)
    H.initialize_from_grid_csv!(U, g, m, TAU0, EVENT; smooth_fm = 0.4)   # example 07 explains σ
    frames = NamedTuple[]
    on_dump = (τ, Uu, wk) -> push!(frames,
        (τ = τ, f = H.fields_2d(g, Uu, wk, m)))
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15,
                        on_dump = on_dump, dump_dt = (TAUF - TAU0)/(nframe - 1))
    res.ok || error("run failed at τ = $(res.τ)")
    return (; g, m, U, res, frames)
end

# ── |ω| and |σ|, the honest way ─────────────────────────────────────────────────────────────────
"""
    kinematic_norms(fa, fb, τ) -> (|ω|, |σ|, T)

‖ω‖ = √(ω_{μν}ω^{μν}) and ‖σ‖ = √(σ_{μν}σ^{μν}) on the grid, from two dumps `fa` (at τ) and `fb`
(the next one) so that ∂_τu^i is a real difference and not zero. Built by plain index algebra on
the (τ,x,y) block with the metric — the η components of ω vanish by boost invariance.
"""
function kinematic_norms(fa, fb, τ, dτ, dx, dy)
    nx, ny = size(fa.T)
    ωn = zeros(nx, ny); σn = zeros(nx, ny)
    gm = (-1.0, 1.0, 1.0)                       # g_ττ, g_xx, g_yy (and g^ττ = −1)
    for i in 2:nx-1, j in 2:ny-1
        ux = fa.ux[i,j]; uy = fa.uy[i,j]; ut = sqrt(1 + ux^2 + uy^2)
        dtux = (fb.ux[i,j] - ux)/dτ;  dtuy = (fb.uy[i,j] - uy)/dτ
        dxux = (fa.ux[i+1,j] - fa.ux[i-1,j])/(2dx); dxuy = (fa.uy[i+1,j] - fa.uy[i-1,j])/(2dx)
        dyux = (fa.ux[i,j+1] - fa.ux[i,j-1])/(2dy); dyuy = (fa.uy[i,j+1] - fa.uy[i,j-1])/(2dy)
        # ∂_α u^τ follows from u^τ = √(1 + u⊥²)
        dtut = (ux*dtux + uy*dtuy)/ut
        dxut = (ux*dxux + uy*dxuy)/ut
        dyut = (ux*dyux + uy*dyuy)/ut
        ax = ut*dtux + ux*dxux + uy*dyux        # a^i = u^α ∂_α u^i
        ay = ut*dtuy + ux*dxuy + uy*dyuy
        at = (ux*ax + uy*ay)/ut                 # from u·a = 0
        θ  = dtut + dxux + dyuy + ut/τ
        # A^{αν} = ∇⊥^α u^ν = g^{αα} ∂_α u^ν + u^α a^ν, derivative index FIRST
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
        σ2 += (ut/τ - θ/3)^2                    # σ^η_η, the one component outside the block
        ωn[i,j] = sqrt(max(ω2, 0.0)); σn[i,j] = sqrt(max(σ2, 0.0))
    end
    return ωn, σn
end

# ── run both ────────────────────────────────────────────────────────────────────────────────────
println("\n  THE VORTICITY COUPLING, ON AGAINST OFF — one real Pb+Pb event (20-30 %, smoothed)")
@printf("  %d² on ±%.0f fm, τ %.1f → %.1f, shear+bulk+diffusion + the consistent charm closures\n\n",
        N, LBOX, TAU0, TAUF)
off = run_case(false)
on  = run_case(true)
dx  = off.frames[1].f.x[2] - off.frames[1].f.x[1]

rel(a, b, mask) = begin                      # ‖a−b‖₂ / ‖b‖₂ over the masked cells
    num = 0.0; den = 0.0
    for i in eachindex(b)
        mask[i] || continue
        num += (a[i] - b[i])^2; den += b[i]^2
    end
    den > 0 ? sqrt(num/den) : 0.0
end

println("   τ      |ω|/|σ| (median, p90, max)        Δπ_Q/π_Q     ΔΠ_Q/Π_Q     Δν/ν      ΔT/T")
rows = NamedTuple[]
for k in 1:(length(off.frames)-1)
    fa, fb = off.frames[k].f, off.frames[k+1].f
    τ = off.frames[k].τ
    # ⚠ THE SOLVER CAN EMIT A DUPLICATE FINAL DUMP at τ = τ_f (the last scheduled dump and the
    # end-of-run dump coincide). ω and σ come from CONSECUTIVE dumps, so a zero interval divides by
    # zero and NaNs the whole frame. That used to be swallowed by the `isempty(r)` guard below —
    # the frame silently vanished and the table simply stopped one dump early, which is why the
    # numbers quoted here are at τ ≈ 5.84 and not 6.00. Skip it explicitly instead.
    off.frames[k+1].τ - τ > 0 || continue
    ωn, σn = kinematic_norms(fa, fb, τ, off.frames[k+1].τ - τ, dx, dx)
    hot = fa.T .> T_FO
    r = [ωn[i]/σn[i] for i in eachindex(ωn) if hot[i] && σn[i] > 0]
    isempty(r) && continue
    fon = on.frames[k].f
    πoff = vcat(fa.pQxx, fa.pQxy, fa.pQyy);  πon = vcat(fon.pQxx, fon.pQxy, fon.pQyy)
    hot3 = vcat(hot, hot, hot)
    dπ  = rel(πon, πoff, hot3)
    dΠ  = rel(fon.PiQ, fa.PiQ, hot)
    dν  = rel(vcat(fon.nux, fon.nuy), vcat(fa.nux, fa.nuy), vcat(hot, hot))
    dT  = rel(fon.T, fa.T, hot)
    push!(rows, (τ = τ, med = median(r), p90 = quantile(r, 0.9), mx = maximum(r),
                 dπ = dπ, dΠ = dΠ, dν = dν, dT = dT, ω = ωn, σ = σn, hot = hot))
    k % 4 == 1 && @printf("  %5.2f   %9.2e %9.2e %9.2e   %10.2e %12.2e %10.2e %9.2e\n",
                          τ, median(r), quantile(r, 0.9), maximum(r), dπ, dΠ, dν, dT)
end

# ── is the bright fringe at the freeze-out surface an artefact? (EX08_DIAG=1) ────────────────────
# The |ω|/|σ| map is brightest in a thin fringe just inside the T_fo contour, exactly where a ratio
# is most likely to be a 0/0 — σ is small in a dilute rim. That suspicion is testable, and it is
# FALSE on this event: σ at the brightest cells is ABOVE the hot-cell median, so ω is genuinely
# large there (the flow shears against the vacuum). This is the diagnostic behind the comment in
# `faded_image` that refuses to damp those cells, and behind the statement that the fringe does not
# drive the headline median. Re-run it on a new event before re-using either claim.
if get(ENV, "EX08_DIAG", "") == "1"
    rr = rows[end]
    fa = off.frames[length(rows)].f
    hot = rr.hot
    rat = map((w, s) -> w/max(s, 1e-30), rr.ω, rr.σ)
    hi = quantile(rat[hot], 0.995)
    bright = hot .& (rat .> hi)
    near = hot .& (fa.T .< 1.1*T_FO)                 # "the fringe": within 10 % of T_fo
    @printf("\n  FRINGE DIAGNOSTIC at τ = %.2f — hot cells %d, brightest 0.5 %% = %d of them\n",
            rr.τ, count(hot), count(bright))
    @printf("    |σ| there   %.3e   vs hot-cell median %.3e   → %s\n",
            median(rr.σ[bright]), median(rr.σ[hot]),
            median(rr.σ[bright]) > median(rr.σ[hot]) ? "NOT a vanishing denominator" : "σ IS small — suspect")
    @printf("    T there     %.4f   (T_fo = %.4f, hot-cell median %.4f)\n",
            median(fa.T[bright]), T_FO, median(fa.T[hot]))
    @printf("    %.0f %% of them sit in the fringe, which is %.0f %% of the hot cells\n",
            100*count(bright .& near)/max(count(bright), 1), 100*count(near)/count(hot))
    @printf("    median |ω|/|σ| without the fringe: %.3e   (quoted, with it: %.3e)\n",
            median(rat[hot .& .!near]), rr.med)
    # and what is OUTSIDE freeze-out, which is what EX08_NOFADE=1 puts on screen
    xg = off.frames[1].f.x
    rad = [hypot(x, y) for x in xg, y in xg]
    println("\n    radial profile of the tail (the cells the nofade render shows):")
    println("      r [fm]     cells        T          |σ|        |ω|/|σ|   p99 of ratio")
    for (r0, r1) in ((0,4), (4,7), (7,9), (9,10), (10,11), (11,12), (12,14))
        sel = (rad .>= r0) .& (rad .< r1) .& (rr.σ .> 0)
        count(sel) == 0 && continue
        @printf("     %2d–%2d   %7d   %.5f   %.3e   %.3e   %.3e\n", r0, r1, count(sel),
                median(fa.T[sel]), median(rr.σ[sel]), median(rat[sel]), quantile(rat[sel], 0.99))
    end
    # the speckle: which cells actually carry the huge ratios, and what is small there?
    spk = (rat .> 0.8) .& (rr.σ .> 0)
    @printf("\n    cells with |ω|/|σ| > 0.8: %d\n", count(spk))
    if count(spk) > 0
        @printf("      their T:  median %.5f  min %.5f  max %.5f\n",
                median(fa.T[spk]), minimum(fa.T[spk]), maximum(fa.T[spk]))
        @printf("      their r:  median %.2f  min %.2f  max %.2f  fm\n",
                median(rad[spk]), minimum(rad[spk]), maximum(rad[spk]))
        @printf("      their |σ|: median %.3e   (all-cell median %.3e)\n",
                median(rr.σ[spk]), median(rr.σ[rr.σ .> 0]))
        @printf("      their |ω|: median %.3e   (all-cell median %.3e)\n",
                median(rr.ω[spk]), median(rr.ω[rr.σ .> 0]))
    end
end

last = rows[end]
@printf("\n  N = %d, dx = %.3f fm, at τ = %.2f:  |ω|/|σ| median %.2e, p90 %.2e, max %.2e\n",
        N, 2LBOX/N, last.τ, last.med, last.p90, last.mx)
@printf("                       →   π_Q moves %.2e, Π_Q %.2e   [compare across resolutions]\n",
        last.dπ, last.dΠ)
@printf("  the medium and the current do not move at all: ΔT/T = %.1e, Δν/ν = %.1e — the second\n",
        last.dT, last.dν)
println("  moment is PASSIVE at c_M = 0, so anything else would be a wiring bug, not physics.")

# ── animations ──────────────────────────────────────────────────────────────────────────────────
# ⚠ NO HARD RIM. These maps are only meaningful where there is a fluid, so they used to be masked
# with `T > T_fo ? value : NaN` — which draws the fireball with a razor edge that moves frame to
# frame and reads as a rendering artefact rather than as a freeze-out surface. Instead the colour is
# FADED into the background over a temperature band [T_FADE, T_FO]: fully opaque above T_fo, gone
# below T_FADE. `heatmap` cannot do per-cell alpha, so the panel is built as an RGB image and drawn
# with `plot(xs, ys, img)`; the colorbar is then drawn separately (Plots has no bar for an image) by
# a zero-size scatter carrying the right `clims`. The T_fo contour is kept on top, so the surface
# the mask used to assert is still visible — as a line, where it belongs.
const T_FADE = 0.120                       # GeV: colour is fully gone here, full strength at T_FO
const T_VAC  = 0.050                       # GeV: below this there is no coherent fluid, only floor
const BG     = RGB(1.0, 1.0, 1.0)          # the panel background the fade blends into
xs = off.frames[1].f.x

"colour `Z` with `cmap` over [lo,hi], alpha-faded to the background as T runs T_FO → T_FADE"
function faded_image(Z, T, cmap, lo, hi)
    g = cgrad(cmap)
    img = Matrix{RGB{Float64}}(undef, size(Z, 2), size(Z, 1))
    for j in axes(Z, 2), i in axes(Z, 1)
        # NOFADE shows the whole tail at full strength and stops only at the vacuum floor
        # (T_VAC = 50 MeV; past r ≈ 11 fm on this event T is EXACTLY 0, and a ratio there is 0/0).
        #
        # ⚠ THE GRAINY BRIGHT EDGE AT r ≈ 9–12 fm IS NOT THAT 0/0, AND IS NOT SMOOTHED AWAY HERE.
        # Measured (EX08_DIAG=1, τ = 7.6): those cells have |σ| = 2.0e-1, nearly TWICE the all-cell
        # median, and |ω| = 2.4e-1 against an all-cell median of 6.7e-4 — a factor ~350. The edge is
        # bright because the vorticity is genuinely huge where the expanding fluid meets vacuum, not
        # because the denominator died. It is grainy because at 0.175 fm per cell that edge is one
        # or two cells thick; refine and it sharpens rather than disappears.
        w = NOFADE ? clamp((T[i,j] - 0.6T_VAC)/(0.4T_VAC), 0.0, 1.0) :
                     clamp((T[i,j] - T_FADE)/(T_FO - T_FADE), 0.0, 1.0)
        # ⚠ the weight touches ONLY the alpha, never the value. The bright fringe just inside the
        # freeze-out contour looks like a 0/0 and is not: measured at τ = 5.8 on this event, σ at the
        # brightest 0.5 % of hot cells is 1.45e-1, ABOVE the hot-cell median 1.11e-1 — those cells
        # are bright because ω is genuinely large where the flow shears against the vacuum, not
        # because σ vanishes. Damping the value there would erase a real signal to tidy the picture.
        # (Nor does the fringe drive the headline: excluding the whole T < 1.1 T_fo band moves the
        # median from 3.01e-2 to 2.60e-2.)
        c = RGB(get(g, clamp((Z[i,j] - lo)/(hi - lo), 0.0, 1.0)))
        # ⚠ row j of the image is y[j], and `yflip = false` in the panel below undoes the image
        # convention (first row at the top) that would otherwise print the y axis upside down.
        img[j, i] = RGB(w*c.r + (1-w)*BG.r, w*c.g + (1-w)*BG.g, w*c.b + (1-w)*BG.b)
    end
    return img
end

"the image panel plus a colorbar and the T_fo contour"
function faded_panel(Z, T, cmap, lo, hi, label, title)
    # with the fade off the point IS the outer tail, so show the whole box rather than the ±10 fm
    # crop that frames the fireball
    lim = NOFADE ? LBOX : 10.0
    p = plot(xs, xs, faded_image(Z, T, cmap, lo, hi); aspect_ratio = 1, yflip = false,
             xlims = (-lim, lim), ylims = (-lim, lim), xlabel = "x  [fm]", ylabel = "y  [fm]",
             titlefontsize = 10, title = title, framestyle = :box, legend = false)
    # the colorbar: an invisible series that only exists to carry the scale
    scatter!(p, [NaN], [NaN]; zcolor = [lo], c = cmap, clims = (lo, hi), ms = 0,
             colorbar = true, colorbar_title = label, label = "")
    contour!(p, xs, xs, permutedims(T); levels = [T_FO], c = :white, lw = 1, alpha = 0.7,
             colorbar_entry = false)
    return p
end

function animate(zs, Ts, τs, tag, label, cmap, hi; lo = 0.0)
    anim = @animate for (k, Z) in enumerate(zs)
        faded_panel(Z, Ts[k], cmap, lo, hi, label,
                    @sprintf("τ = %5.2f fm/c    (N = %d)", τs[k], N))
        plot!(; size = (ASIZE, round(Int, 0.86ASIZE)))
    end
    path = joinpath(ANIM, "ex08_$(tag)$(SFX).gif"); gif(anim, path; fps = FPS, show_msg = false); path
end

τs    = [r.τ for r in rows]
Ts    = [off.frames[k].f.T for k in 1:length(rows)]
ratio = [map((w, sg) -> w/max(sg, 1e-30), r.ω, r.σ) for r in rows]   # keeps the 2-D shape
# The colour scale is set on the cells the picture actually SHOWS. With the fade on that is the
# fireball, and the faded skirt must not be allowed to stretch the colorbar. With `EX08_NOFADE=1`
# the tail is the subject, and scaling to the fireball would saturate it to a single colour — the
# same mistake the README warns about for |u|. Either way it is a p99.5, never a max.
# ⚠ the two are NOT comparable: a nofade gif and a faded one carry different colorbars by design.
scale_cells(zs) = reduce(vcat, [vec(zs[k][NOFADE ? (Ts[k] .> T_VAC) : rows[k].hot])
                                for k in eachindex(rows)])
hiR  = quantile(filter(isfinite, scale_cells(ratio)), 0.995)
p1 = animate(ratio, Ts, τs, "omega_over_sigma", "|ω| / |σ|", :magma, hiR)

zs = Matrix{Float64}[]
for k in 1:length(rows)
    fa, fon = off.frames[k].f, on.frames[k].f
    sc = maximum(abs, fa.pQxx[fa.T .> T_FO]; init = 1e-30)
    push!(zs, map((a, b) -> abs(b - a)/sc, fa.pQxx, fon.pQxx))
end
hiD = quantile(filter(isfinite, scale_cells(zs)), 0.995)
p2 = animate(zs, Ts, τs, "delta_piQxx", "|Δπ_Q^{xx}| / max|π_Q^{xx}|", :viridis, hiD)
println("\n  wrote ", p1, "\n        ", p2)

# ── figure ──────────────────────────────────────────────────────────────────────────────────────
# ⚠ drop the first frame from the LOG panels: at τ₀ every difference is exactly 0 (nothing has been
# integrated yet), and a single zero collapses a log axis to 10^-Inf — it takes the whole figure
# with it. Plotting it as a floor would be worse: it would draw a number that is not a measurement.
ok = 2:length(rows)
pa = plot(τs[ok], [rows[k].med for k in ok]; label = "median", c = BLUE, xlabel = "τ  [fm/c]",
          ylabel = "|ω| / |σ|  (cells above T_fo)", yscale = :log10,
          title = "how big the coupling's kinematics are", titlefontsize = 10, legend = :bottomright)
plot!(pa, τs[ok], [rows[k].p90 for k in ok]; label = "90th percentile", c = BLUE, ls = :dash)
plot!(pa, τs[ok], [rows[k].mx for k in ok];  label = "max", c = ORANGE)
pb = plot(τs[ok], [rows[k].dπ for k in ok]; label = "π_Q^{ij}", c = BLUE, xlabel = "τ  [fm/c]",
          ylabel = "relative change when switched on", yscale = :log10,
          title = "what the term actually moves", titlefontsize = 10, legend = :bottomright)
plot!(pb, τs[ok], [rows[k].dΠ for k in ok]; label = "Π_Q", c = ORANGE)
annotate!(pb, τs[ok][1] + 0.2, 2e-4,
          text("ν^i and T: EXACTLY zero at every frame\n(the second moment is passive at c_M = 0)",
               7, :left, MUTED))
k = length(rows)
pc = faded_panel(ratio[k], Ts[k], :magma, 0.0, hiR, "|ω| / |σ|",
                 @sprintf("|ω|/|σ| at τ = %.1f fm/c", τs[k]))
plt = plot(pa, pb, pc; layout = @layout([a b c]), size = (1500, 430),
           left_margin = 7Plots.mm, bottom_margin = 7Plots.mm)
let f = joinpath(FIG, "ex08_vorticity$(SFX).png")
    savefig(plt, f); println("        ", f)
end

println("""

  READING IT
    |ω|/|σ| is the honest measure of the term's size: the vorticity coupling and the shear coupling
    sit side by side in the same bracket of the same equation, with the same τ_M in front, so their
    ratio is what says whether omitting one of them matters. The number to carry away is the one
    printed above for THIS event.

    ⚠ THE RATIO RISES BUT THE EFFECT DOES NOT. |ω|/|σ| does grow with τ, because σ falls as the
    fireball dilutes while ω does not. The shift it causes in π_Q behaves differently: run to τ = 8
    (EX08_TAUF=8.0) and it PEAKS near τ ≈ 5.5 and then falls, 1.01e-2 at τ = 5.8 down to 5.06e-3 at
    τ = 7.7 — halving over the last 2 fm/c, with Π_Q falling faster still (1.76e-4 → 4.93e-5). The
    default τ_f = 6 window stops right at the peak and so makes the trend look monotone; that alone
    is a reason to run this file past 6 fm/c before quoting a trend from it.

    WHY the effect turns over is NOT established here, and the obvious guess is wrong: τ_M does not
    collapse with the temperature. Its Bessel factor K₄K₂/2K₃² runs only 0.589 → 0.550 between
    T = 300 and 155 MeV (7 %), so τ_M simply tracks τ_n. Something else — the decay of π_Q itself,
    which the shift is measured RELATIVE to, is the first candidate — sets the turnover. Measure it
    before repeating an explanation.

    The effect on π_Q is far below the disagreements this sector already has with Fluidum (19 % on
    π_Q^{xx}, COMPARISON_2P1D.md §31.5). That is the case for leaving `m2_vorticity = false` as the
    default — not that the term is negligible in principle, but that on this event it is smaller
    than the errors already on the table, and turning it on breaks the cross-code comparison that is
    currently the sector's main check. Rerun this file on YOUR event before relying on that.""")
