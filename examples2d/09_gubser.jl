#!/usr/bin/env julia
#=
09 — Gubser flow: the one 2-D problem whose answer is known in closed form.

Every other example measures the solver against itself — a convergence study, a control, a sector
switched off. This one measures it against an EXACT SOLUTION. Gubser flow is ideal conformal
hydrodynamics with a symmetry (SO(3)_q × SO(1,1) × Z₂) that fixes T(τ,r) and u^r(τ,r) completely,
so at every cell and every time there is a right answer to subtract.

    T(τ, r) = T̂ / τ · [ … ] ,    u^r(τ, r) = sinh(…)      (src/gubser.jl, from the 1-D module)

What you get: an animation of the solver beside the exact solution and their difference, and the
convergence of the error with resolution — the quantity that says "second order" rather than
"looks right".

⚠ THE REFERENCE IS ONLY EXACT FOR THE EQUATIONS GUBSER SOLVES: ideal, conformal, no charge
diffusion. So the shear, bulk and charm sectors are OFF here, and the heavy-quark part of the EOS
is suppressed with α = −20 (the EOS is then conformal to 4e-13, which gate G1 measures). Turning a
dissipative sector on and comparing to this reference would be comparing two different problems.

⚠ AND THE COMPARISON IS RESTRICTED TO r ≤ 3 fm. Gubser flow extends to infinity and the box does
not; near the edge the outflow boundary and the vacuum floor are doing the talking, not the scheme.
Gate G1 uses the same radius for the same reason.

This is the example version of gate G1 (`../test/test_gubser2d.jl`), which asserts the numbers this
file draws. Run the gate to check the solver; run this to see what it is checking.
=#
ENV["GKSwstype"] = "100"
using Printf, Statistics, Plots   # RGB and cgrad come from Plots' re-export of Colors
gr(); default(; fontfamily = "sans-serif", framestyle = :box, grid = false, dpi = 150, lw = 2)

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))        # the 1-D module carries the analytic Gubser formulas
include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d
const FIG  = joinpath(@__DIR__, "figures");  isdir(FIG)  || mkpath(FIG)
const ANIM = joinpath(FIG, "anim");          isdir(ANIM) || mkpath(ANIM)

const QG, TAU0, TAUF, TC0, ALPHA = 1.0, 1.0, 3.0, 0.6, -20.0
const BOX, RCMP = 10.0, 3.0
const NFRAME, FPS = 32, 8
const EOS    = H.ConformalHQEOS()
const TSCALE = hydro.gubser_Tscale_from_center_T(TAU0, QG, TC0)
Tana(τ, r)  = hydro.gubser_temperature(τ, r, QG, TSCALE)
urana(τ, r) = hydro.gubser_ur(τ, r, QG)
const BLUE, ORANGE, MUTED = colorant"#2a78d6", colorant"#eb6834", colorant"#8a8980"

# ── one run ─────────────────────────────────────────────────────────────────────────────────────
function run_gubser(N; τf = TAUF, nframe = 0)
    g = H.make_grid2d(N, N; xmax = BOX, ymax = BOX)
    m = H.build_model_2d(; eos = EOS)          # ideal: no shear, no bulk, no diffusion
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        ur = urana(TAU0, r)
        H.set_cell!(U, H.lin(g, ix, iy), Tana(TAU0, r), ALPHA,
                    r > 0 ? ur*x/r : 0.0, r > 0 ? ur*y/r : 0.0, TAU0, m)
    end
    frames = NamedTuple[]
    on_dump = nframe == 0 ? nothing :
        (τ, Uu, wk) -> push!(frames, (τ = τ, f = H.fields_2d(g, Uu, wk, m)))
    t0 = time()
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05,
                        on_dump = on_dump, dump_dt = nframe == 0 ? 1e9 : (τf - TAU0)/(nframe - 1))
    res.ok || error("run failed at τ = $(res.τ)")
    f = H.fields_2d(g, U, res.work, m)
    return (; g, m, U, res, f, frames, secs = time() - t0)
end

"relative L2 of T and of |u| against the exact solution, inside r ≤ RCMP"
function errors(f, τ)
    sT = 0.0; sTa = 0.0; sU = 0.0; sUa = 0.0
    for (i, x) in enumerate(f.x), (j, y) in enumerate(f.y)
        r = hypot(x, y); r <= RCMP || continue
        Te = Tana(τ, r); ue = urana(τ, r)
        sT += (f.T[i,j] - Te)^2; sTa += Te^2
        sU += (hypot(f.ux[i,j], f.uy[i,j]) - ue)^2; sUa += ue^2
    end
    (sqrt(sT/sTa), sqrt(sU/sUa))
end

# ── the convergence ladder: the claim "second order", measured ──────────────────────────────────
println("\n  GUBSER FLOW — the solver against an exact solution")
@printf("  ideal conformal, q = %.1f /fm, τ %.1f → %.1f, comparison inside r ≤ %.0f fm\n\n",
        QG, TAU0, TAUF, RCMP)
# ⚠ IN A FUNCTION, not a top-level `for`: `prev = (eT, eU)` inside a top-level loop creates a new
# local every iteration and the next one cannot read it (CLAUDE.md trap 1; example 04 says the same).
function convergence_ladder()
    @printf("  %6s %9s %12s %9s %12s %9s %9s\n", "N", "dx [fm]", "L2(T)", "order", "L2(|u|)", "order", "wall [s]")
    prev = (NaN, NaN)
    ladder = NamedTuple[]
    for N in (100, 200, 400)
        r = run_gubser(N)
        eT, eU = errors(r.f, r.res.τ)
        oT = isnan(prev[1]) ? NaN : log2(prev[1]/eT)
        oU = isnan(prev[2]) ? NaN : log2(prev[2]/eU)
        @printf("  %6d %9.4f %12.3e %9s %12.3e %9s %9.1f\n", N, 2BOX/N, eT,
                isnan(oT) ? "—" : @sprintf("%.2f", oT), eU, isnan(oU) ? "—" : @sprintf("%.2f", oU), r.secs)
        prev = (eT, eU)
        push!(ladder, (N = N, dx = 2BOX/N, eT = eT, eU = eU))
    end
    return ladder
end
ladder = convergence_ladder()
println("  the exponent is the measurement; \"it looks right\" is not one. Gate G1 asserts ≥ 1.8.")

# ── the animation: solver, exact, and the difference ────────────────────────────────────────────
run = run_gubser(200; nframe = NFRAME)
xs = run.frames[1].f.x
exact_T(τ) = [Tana(τ, hypot(x, y)) for x in xs, y in xs]
const radius = [hypot(x, y) for x in xs, y in xs]

# ⚠ THE COMPARISON RADIUS IS A RING, NOT A CUT. The error panel used to be masked with
# `r ≤ 3 fm ? e : NaN`, which draws a hard circular rim in every frame — that reads as an artefact
# of the plot, not as the statement it is ("the numbers above are quoted here"). Instead the error
# is faded out over the last 0.6 fm and the radius is drawn as a dashed ring.
#
# The fade is done by damping the VALUE toward zero, not by alpha: GR gives a heatmap only ONE
# scalar alpha for the whole series (a per-cell matrix is silently collapsed — measured), and a
# per-cell fade would mean building an RGB image, which does not share the 3-panel layout cleanly.
# Damping the value costs nothing here because `:balance` is diverging and its neutral colour IS the
# background — zero error and "no data" render identically, which is exactly the intended reading.
# Unlike example 08's fringe, the error out here really is meaningless: it is the outflow boundary
# and the vacuum floor talking, it saturates ±3 % long before the box edge, and showing it at full
# strength makes the panel about the box rather than about the scheme.
faded_err(Z, r) = map((z, rr) -> z*clamp((RCMP + 0.3 - rr)/0.6, 0.0, 1.0), Z, r)

hiT = maximum(maximum(f.f.T) for f in run.frames)
# ⚠ the error scale is MEASURED, not assumed. It used to be a hardcoded ±3 %, which at N = 200 is a
# factor ~10 larger than anything inside the ring — so the panel rendered as a blank white disc and
# said nothing. Take the largest |error| any frame reaches inside the comparison radius, rounded up.
errscale = let m = 0.0
    for fr in run.frames
        Te = exact_T(fr.τ)
        for i in eachindex(Te)
            radius[i] <= RCMP && (m = max(m, abs(100*(fr.f.T[i] - Te[i])/Te[i])))
        end
    end
    max(0.05, round(1.15m; sigdigits = 2))
end
@printf("  the error panel is scaled to ±%.2g %% — the largest excursion inside r ≤ %.0f fm\n",
        errscale, RCMP)
anim = @animate for fr in run.frames
    Te = exact_T(fr.τ)
    err = @. 100*(fr.f.T - Te)/Te
    p1 = heatmap(xs, xs, permutedims(fr.f.T); c = :inferno, clims = (0, hiT), aspect_ratio = 1,
                 xlims = (-6, 6), ylims = (-6, 6), title = @sprintf("solver   τ = %4.2f fm/c", fr.τ),
                 titlefontsize = 9, xlabel = "x  [fm]", ylabel = "y  [fm]", colorbar_title = "T  [GeV]")
    p2 = heatmap(xs, xs, permutedims(Te); c = :inferno, clims = (0, hiT), aspect_ratio = 1,
                 xlims = (-6, 6), ylims = (-6, 6), title = "exact (Gubser)", titlefontsize = 9,
                 xlabel = "x  [fm]", colorbar_title = "T  [GeV]")
    p3 = heatmap(xs, xs, permutedims(faded_err(err, radius)); c = :balance,
                 clims = (-errscale, errscale), aspect_ratio = 1, xlims = (-6, 6), ylims = (-6, 6),
                 title = "(solver − exact)/exact;  quoted inside the ring", titlefontsize = 9,
                 xlabel = "x  [fm]", colorbar_title = "[%]")
    let θ = range(0, 2π; length = 200)
        plot!(p3, RCMP .* cos.(θ), RCMP .* sin.(θ); c = MUTED, lw = 1, ls = :dash, label = "",
              colorbar_entry = false)
    end
    plot(p1, p2, p3; layout = (1, 3), size = (1180, 400), bottom_margin = 6Plots.mm,
         left_margin = 4Plots.mm)
end
path = joinpath(ANIM, "ex09_gubser.gif"); gif(anim, path; fps = FPS, show_msg = false)
println("\n  wrote ", path)

# ── the static figure ───────────────────────────────────────────────────────────────────────────
fe = run.frames[end]; τe = fe.τ
mid = length(xs) ÷ 2 + 1
pa = plot(xs, [Tana(τe, abs(x)) for x in xs]; label = "exact", c = MUTED, lw = 3,
          xlabel = "x  [fm]  (y = 0)", ylabel = "T  [GeV]", xlims = (-6, 6),
          title = @sprintf("the profile at τ = %.1f fm/c", τe), titlefontsize = 10)
plot!(pa, xs, fe.f.T[:, mid]; label = "solver, N = 200", c = BLUE, ls = :dash)
vspan!(pa, [-RCMP, RCMP]; c = MUTED, alpha = 0.08, label = "compared here")
pb = plot(xs, [urana(τe, abs(x))*sign(x) for x in xs]; label = "exact", c = MUTED, lw = 3,
          xlabel = "x  [fm]  (y = 0)", ylabel = "u^x", xlims = (-6, 6),
          title = "the flow it builds", titlefontsize = 10, legend = :topleft)
plot!(pb, xs, fe.f.ux[:, mid]; label = "solver", c = ORANGE, ls = :dash)
pc = plot([l.dx for l in ladder], [l.eT for l in ladder]; xscale = :log10, yscale = :log10,
          marker = :circle, ms = 6, c = BLUE, label = "L2(T)",
          xlabel = "dx  [fm]", ylabel = "relative L2 error inside r ≤ 3 fm",
          title = "second order, measured", titlefontsize = 10, legend = :bottomright)
plot!(pc, [l.dx for l in ladder], [l.eU for l in ladder]; marker = :square, ms = 6, c = ORANGE,
      label = "L2(|u|)")
let d = [l.dx for l in ladder], e = [l.eT for l in ladder]
    plot!(pc, d, e[1] .* (d ./ d[1]).^2; c = MUTED, ls = :dash, label = "∝ dx²")
end
plt = plot(pa, pb, pc; layout = @layout([a b c]), size = (1450, 430),
           left_margin = 7Plots.mm, bottom_margin = 7Plots.mm)
savefig(plt, joinpath(FIG, "ex09_gubser.png"))
println("        ", joinpath(FIG, "ex09_gubser.png"))

@printf("""

  READING IT
    The middle panel of the animation never changes shape relative to the left one, and the right
    panel never leaves ±%.2g %% inside the ring — that is the whole point: an exact solution turns
    "the code ran" into "the code is right, to this order, on this problem". The panel's colour
    scale is that measured excursion, not a round number chosen in advance.

    ⚠ The error is NOT uniform in radius. It grows outward, and outside r ≈ 3 fm it stops being a
    statement about the scheme at all: Gubser flow fills the plane, the box does not, and the
    outflow boundary and the vacuum floor take over. That is why both this file and gate G1 quote a
    radius with every number, and why the panel fades the error out across that radius instead of
    cutting it off there.""", errscale)
println("""

    What this does NOT test: everything dissipative. Gubser is ideal and conformal, so shear, bulk
    and the charm sectors are off here. Their references are elsewhere — G2 and Gs for shear and
    sound, G0b for nonlinear bulk, Gc/Gm/Gt and the closed-form referees in
    Projects/FiVoFluidumComparison for the charm closures.""")
