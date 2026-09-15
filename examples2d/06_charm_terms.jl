#!/usr/bin/env julia
#=
06 — the charm sector term by term: what each term of the consistent first and second moments
does, and how to switch any single one of them off.

The charge sector (`enable_diff`) evolves the charm current ν^i. With `consistent_fm = true` its
drive is the full-∇P first moment — the fugacity gradient plus four more terms (pressure gradient,
inertia, the current riding the flow gradient, expansion + transport of h). With
`consistent_m2 = true` the charm second moment (π_Q^{ij}, Π_Q) is evolved beside it, with ten terms
of its own. Every one of them is behind a switch in `Terms2D`:

    m = build_model_2d(; ..., consistent_fm = true, terms = (fm_inertial = false,))
    show_equations(m)          # prints every equation, every term [x]/[ ], and its knob

This file runs one fireball with everything on, then once more with each term removed, and reports
how much each one moves the answer. That is an ATTRIBUTION, not a decomposition: removing a term
changes the state the other terms see, so the effects do not add up to 100 %.

⚠ THE SIGN OF THE FIRST MOMENT WAS WRONG UNTIL 2026-09-10. The consistent source is returned
source-on-the-LHS and the solver added it to the right-hand target, so every consistent term entered
with the wrong sign — the expansion term anti-damped the current. Gate Gc7 now evolves a Bjorken
state and checks it. Any 2-D number with `consistent_fm = true` produced before that date is wrong.

⚠ `m2_vorticity` IS OFF BY DEFAULT. It is part of the derived second-moment equation, vanishes
identically in 1-D, and neither 2-D code carried it before 2026-09-10; the last block measures it.

⚠ READ THE CURRENT OVER T > T_fo ONLY. The dilute tail is where the vacuum ramp throttles the
charge sector (README2D.md, "regulators"); a norm over the whole box is dominated by it.

Equations, term by term, with the code and the gate for each: ../EQUATIONS2D.md.
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
include(joinpath(_ROOT, "main2D.jl")); using .hydro2d; const H = hydro2d
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

const N, LBOX, TAU0, TAUF, T_FO, DST = 64, 10.0, 0.4, 2.0, 0.1565, 0.1163
const BLUE, ORANGE, INK, MUTED = colorant"#2a78d6", colorant"#eb6834", colorant"#1a1a19", colorant"#8a8980"

# ── the fireball ────────────────────────────────────────────────────────────────────────────────
# An elliptic medium (so the flow has a^i, σ^{ij} and a transverse direction) carrying charm with
# its own, off-centre fugacity lump (so ∇α and ∇T point in different directions and every drive
# term is live). Shear and bulk are on: the second moment's class-(ii) terms are driven by the
# MEDIUM's shear and expansion, and would be trivially small on an ideal background.
function run_case(terms)
    g = H.make_grid2d(N, N; xmax = LBOX, ymax = LBOX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10,
                           enable_bulk  = true, zeta_over_s = 0.10,
                           enable_diff  = true, kappa_coeff = DST,
                           consistent_fm = true, consistent_m2 = true, terms = terms)
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]
        env = exp(-(x^2/1.4 + y^2/0.7)/(2*2.8^2))
        T = max(0.06 + 0.42*env^(1/3), 0.06)
        α = -4.0 + 1.2*exp(-((x - 1.5)^2 + (y + 0.5)^2)/(2*1.6^2))
        H.set_cell!(U, H.lin(g, ix, iy), T, α, 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)       # floors + admissibility: never optional (example 01)
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF)
    res.ok || error("run failed at τ = $(res.τ)")
    return g, m, U, res.work
end

"The fields compared, restricted to cells above freeze-out."
function fields(g, m, U, w)
    L = m.layout; ng = g.nghost
    ν = Float64[]; πQ = Float64[]; ΠQ = Float64[]
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        exp(w.yT[i]) > T_FO || continue
        push!(ν, U[L.iNux, i], U[L.iNuy, i])
        push!(πQ, U[L.iPQxx, i], U[L.iPQxy, i], U[L.iPQyy, i])
        push!(ΠQ, U[L.iPiQ, i])
    end
    return (; ν, πQ, ΠQ)
end
rel(a, b) = sqrt(sum(abs2, a .- b)/sum(abs2, b))

# ── the reference: everything on ────────────────────────────────────────────────────────────────
t0 = time()
g, m, Uref, wref = run_case(H.Terms2D())
show_equations(m)
ref = fields(g, m, Uref, wref)
@printf("\nreference run: %d² to τ = %.1f in %.1f s, %d cells above T_fo\n\n",
        N, TAUF, time() - t0, length(ref.ΠQ))

# ── remove one term at a time ───────────────────────────────────────────────────────────────────
# `H.TERMS_2D` is the register every switch is listed in (name, sector, formula); iterating it is
# how this stays complete when a term is added.
fm_names = [t[1] for t in H.TERMS_2D if t[2] in (:enable_diff, :consistent_fm)]
m2_names = [t[1] for t in H.TERMS_2D if t[2] === :consistent_m2 && t[1] !== :m2_vorticity]

println("first moment — remove one term, measure the charm current ν over T > T_fo")
fm_eff = Float64[]
for k in fm_names
    gk, mk, Uk, wk = run_case(NamedTuple{(k,)}((false,)))
    e = rel(fields(gk, mk, Uk, wk).ν, ref.ν); push!(fm_eff, e)
    @printf("  %-14s off  →  |Δν|/|ν| = %7.2f %%\n", k, 100e)
end

println("\nsecond moment — remove one term, measure π_Q^{ij} and Π_Q")
m2_eff_π = Float64[]; m2_eff_Π = Float64[]
for k in m2_names
    gk, mk, Uk, wk = run_case(NamedTuple{(k,)}((false,)))
    f = fields(gk, mk, Uk, wk)
    push!(m2_eff_π, rel(f.πQ, ref.πQ)); push!(m2_eff_Π, rel(f.ΠQ, ref.ΠQ))
    @printf("  %-15s off  →  |Δπ_Q|/|π_Q| = %7.2f %%   |ΔΠ_Q|/|Π_Q| = %7.2f %%\n",
            k, 100m2_eff_π[end], 100m2_eff_Π[end])
end

println("""
  How to read these. fm_gradT and fm_inertial each move ν by ~240 % because they nearly CANCEL:
  on an ideal baryon-free fluid Euler gives a^i = −∇^⟨i⟩T/T and n + T∂n/∂T = nh/T, so the two are
  equal and opposite exactly; the viscous medium leaves the residual. Removing one leaves the other
  unopposed. The trace channel Π_Q has the same structure: its medium drives (bg_gradu, bg_DlnT,
  bg_Dalpha) are each large and nearly cancel, which is why Π_Q is the delicate channel.""")

# ── the one term that is OFF by default ─────────────────────────────────────────────────────────
gv, mv, Uv, wv = run_case((m2_vorticity = true,))
fv = fields(gv, mv, Uv, wv)
@printf("\nm2_vorticity ON (default off)  →  |Δπ_Q|/|π_Q| = %.2e   |ΔΠ_Q|/|Π_Q| = %.2e\n",
        rel(fv.πQ, ref.πQ), rel(fv.ΠQ, ref.ΠQ))
println("  (Π_Q moves only through π_Q: the vorticity coupling has no trace. A smooth fireball has")
println("   little transverse vorticity; a lumpy event carries roughly ten times more — EQUATIONS2D.md §5.)")

# ── figure ──────────────────────────────────────────────────────────────────────────────────────
# (a) the charm current of the reference run; (b)(c) what each term is worth. Effects span three
# decades, so a dot on a log axis, one row per term, reads better than bars.
L = m.layout; ng = g.nghost
xs = g.xC[(ng+1):(ng+g.Nx)]; ys = g.yC[(ng+1):(ng+g.Ny)]
νmag = [hypot(Uref[L.iNux, H.lin(g, ix, iy)], Uref[L.iNuy, H.lin(g, ix, iy)])
        for iy in (ng+1):(ng+g.Ny), ix in (ng+1):(ng+g.Nx)]
Tmap = [exp(wref.yT[H.lin(g, ix, iy)]) for iy in (ng+1):(ng+g.Ny), ix in (ng+1):(ng+g.Nx)]

pa = heatmap(xs, ys, νmag; c = cgrad([colorant"#f4f8fd", BLUE, colorant"#0d366b"]),
             aspect_ratio = 1, xlims = (-8, 8), ylims = (-8, 8),
             xlabel = "x  [fm]", ylabel = "y  [fm]", colorbar_title = "|ν|  [fm⁻³]",
             title = "charm current at τ = $(TAUF) fm/c", titlefontsize = 10)
contour!(pa, xs, ys, Tmap; levels = [T_FO], c = MUTED, lw = 1.5, colorbar_entry = false)
# the DIRECTION of ν on a coarse subset of cells above freeze-out: a segment from the cell centre,
# a dot at its head. (GR's `quiver!` draws its arrowheads in data units and turns vectors this
# short into zigzags — plain segments are what reads.)
step = 4; sx = Float64[]; sy = Float64[]; hx = Float64[]; hy = Float64[]
νmax = max(maximum(νmag), 1e-300)
sc = 1.6/νmax                                 # the longest arrow is 1.6 fm
for (jx, ix) in enumerate((ng+1):(ng+g.Nx)), (jy, iy) in enumerate((ng+1):(ng+g.Ny))
    (jx % step == 0 && jy % step == 0) || continue
    i = H.lin(g, ix, iy); exp(wref.yT[i]) > T_FO || continue
    hypot(Uref[L.iNux, i], Uref[L.iNuy, i]) > 0.1νmax || continue    # skip arrows that are dots
    x0 = g.xC[ix]; y0 = g.yC[iy]
    x1 = x0 + sc*Uref[L.iNux, i]; y1 = y0 + sc*Uref[L.iNuy, i]
    append!(sx, (x0, x1, NaN)); append!(sy, (y0, y1, NaN)); push!(hx, x1); push!(hy, y1)
end
plot!(pa, sx, sy; c = INK, lw = 1, label = false)
scatter!(pa, hx, hy; c = INK, ms = 2, msw = 0, label = false)

lab(names) = replace.(string.(names), "_" => " ")
yfm = collect(1:length(fm_names))
pb = scatter(100 .* fm_eff, yfm; xscale = :log10, yticks = (yfm, lab(fm_names)), yflip = true,
             ms = 7, c = BLUE, msc = :white, msw = 1.5, label = false,
             xlabel = "change in ν when the term is removed  [%]",
             title = "first moment", titlefontsize = 10, xlims = (0.1, 1e3))
ym2 = collect(1:length(m2_names))
pc = scatter(100 .* m2_eff_π, ym2 .- 0.14; xscale = :log10, yticks = (ym2, lab(m2_names)),
             yflip = true, ms = 7, c = BLUE, msc = :white, msw = 1.5, marker = :circle,
             label = "π_Q", xlabel = "change when the term is removed  [%]",
             title = "second moment", titlefontsize = 10, legend = :bottomright,
             xlims = (1e-2, 1e4))
scatter!(pc, 100 .* m2_eff_Π, ym2 .+ 0.14; ms = 6, c = ORANGE, msc = :white, msw = 1.5,
         marker = :square, label = "Π_Q")

plt = plot(pa, pb, pc; layout = @layout([a{0.38w} b c]), size = (1500, 470),
           left_margin = 6Plots.mm, bottom_margin = 7Plots.mm)
savefig(plt, joinpath(FIG, "ex06_charm_terms.png"))
println("\nwrote ", joinpath(FIG, "ex06_charm_terms.png"))
