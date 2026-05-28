#!/usr/bin/env julia
#
# plot_is_vs_bdnk.jl — Compare IS and BDNK current-only evolution results
# Generates fig26_is_vs_bdnk_evolution.pdf for the paper
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # FiVoHydro.jl

using JLD2, Plots; gr()
using LaTeXStrings, Printf

const FIGDIR = joinpath(@__DIR__, "..", "..", "MainFluidum", "Modes", "tex", "figures")

# ═══════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════
println("Loading IS current-only data …")
is = JLD2.load(joinpath(@__DIR__, "..",
    "snapshots/current_only/hydro_currents_FiVo_current-only_tau0_0p400_tauf_15p000_rmax_25p000_nr_300_dst_0p240_20260324_193803.jld2"))

println("Loading BDNK current-only data …")
bdnk = JLD2.load(joinpath(@__DIR__, "..",
    "snapshots/current_only_bdnk/hydro_currents_BDNK_current-only_tau0_0p400_tauf_15p000_rmax_25p000_nr_300_dst_0p240_20260326_160347.jld2"))

r_is   = is["r"]
tau_is = is["tau"]
r_bd   = bdnk["r"]
tau_bd = bdnk["tau"]

println("  IS:   r ∈ [$(round(r_is[1],digits=2)), $(round(r_is[end],digits=2))], τ ∈ [$(tau_is[1]), $(tau_is[end])], Nr=$(length(r_is)), Nτ=$(length(tau_is))")
println("  BDNK: r ∈ [$(round(r_bd[1],digits=2)), $(round(r_bd[end],digits=2))], τ ∈ [$(tau_bd[1]), $(tau_bd[end])], Nr=$(length(r_bd)), Nτ=$(length(tau_bd))")

# ═══════════════════════════════════════════════════════════════════
#  Helper: find nearest index
# ═══════════════════════════════════════════════════════════════════
nearest(arr, val) = argmin(abs.(arr .- val))

# ═══════════════════════════════════════════════════════════════════
#  Fig 26: IS vs BDNK evolution — 4 panels
#  (a) J^τ vs r at several τ
#  (b) ν_r vs r at several τ
#  (c) J^τ(τ) at r=1 fm
#  (d) ν_r(τ) at r=1 fm
# ═══════════════════════════════════════════════════════════════════
println("\n── Generating Fig 26: IS vs BDNK evolution comparison ──")

tau_slices = [1.0, 3.0, 6.0, 10.0]
r_ref = 1.0

# Colors: blue for IS, red for BDNK
# Opacity: lighter (early) → darker (late)
Nt = length(tau_slices)
alphas = range(0.25, 1.0, length=Nt)  # 0.25 at τ=1, 1.0 at τ=10

col_is   = :blue
col_bdnk = :red

# Panel (a): J^τ vs r
pa = plot(xlabel=L"r\;[\mathrm{fm}]", ylabel=L"J^\tau\;[\mathrm{fm}^{-3}]",
          title=L"(a)\;\;J^\tau(r)", legend=:topright, xlim=(0, 15))
for (i, τ) in enumerate(tau_slices)
    it_is = nearest(tau_is, τ)
    it_bd = nearest(tau_bd, τ)
    α = alphas[i]
    τ_str = string(round(τ, digits=0) == τ ? Int(τ) : round(τ, digits=1))
    # IS: solid; BDNK: dashed.  Label only first & last for legend clarity.
    lbl_is   = i == 1 ? L"\mathrm{IS}" : ""
    lbl_bdnk = i == 1 ? L"\mathrm{BDNK}" : ""
    plot!(pa, r_is, is["Jau"][:, it_is], lw=2, color=col_is, ls=:solid,
          alpha=α, label=lbl_is)
    plot!(pa, r_bd, bdnk["Jau"][:, it_bd], lw=2, color=col_bdnk, ls=:dash,
          alpha=α, label=lbl_bdnk)
end
# Add a text annotation for the time ordering
annotate!(pa, 11.5, maximum(is["Jau"][:, nearest(tau_is, 1.0)])*0.85,
          text(L"\tau = 1 \to 10\;\mathrm{fm}/c", 7, :black))
annotate!(pa, 12.0, maximum(is["Jau"][:, nearest(tau_is, 1.0)])*0.7,
          text("(light → dark)", 6, :gray))

# Panel (b): ν_r vs r
pb = plot(xlabel=L"r\;[\mathrm{fm}]", ylabel=L"\nu_r\;[\mathrm{fm}^{-2}]",
          title=L"(b)\;\;\nu_r(r)", legend=:topright, xlim=(0, 15))
for (i, τ) in enumerate(tau_slices)
    it_is = nearest(tau_is, τ)
    it_bd = nearest(tau_bd, τ)
    α = alphas[i]
    lbl_is   = i == 1 ? L"\mathrm{IS}" : ""
    lbl_bdnk = i == 1 ? L"\mathrm{BDNK}" : ""
    plot!(pb, r_is, is["nu_r"][:, it_is], lw=2, color=col_is, ls=:solid,
          alpha=α, label=lbl_is)
    plot!(pb, r_bd, bdnk["nu_r"][:, it_bd], lw=2, color=col_bdnk, ls=:dash,
          alpha=α, label=lbl_bdnk)
end

# Panel (c): J^τ(τ) at r=1 fm
ir_is = nearest(r_is, r_ref)
ir_bd = nearest(r_bd, r_ref)
pc = plot(xlabel=L"\tau\;[\mathrm{fm}/c]", ylabel=L"J^\tau\;[\mathrm{fm}^{-3}]",
          title=L"(c)\;\;J^\tau(\tau)\;\mathrm{at}\;r=1\;\mathrm{fm}", legend=:best)
plot!(pc, tau_is, is["Jau"][ir_is, :], lw=2.5, color=col_is, label=L"\mathrm{IS}")
plot!(pc, tau_bd, bdnk["Jau"][ir_bd, :], lw=2.5, color=col_bdnk, ls=:dash, label=L"\mathrm{BDNK}")

# Panel (d): ν_r(τ) at r=1 fm
pd = plot(xlabel=L"\tau\;[\mathrm{fm}/c]", ylabel=L"\nu_r\;[\mathrm{fm}^{-2}]",
          title=L"(d)\;\;\nu_r(\tau)\;\mathrm{at}\;r=1\;\mathrm{fm}", legend=:best)
plot!(pd, tau_is, is["nu_r"][ir_is, :], lw=2.5, color=col_is, label=L"\mathrm{IS}")
plot!(pd, tau_bd, bdnk["nu_r"][ir_bd, :], lw=2.5, color=col_bdnk, ls=:dash, label=L"\mathrm{BDNK}")

fig26 = plot(pa, pb, pc, pd, layout=(2,2), size=(900, 700),
             left_margin=5Plots.mm, bottom_margin=4Plots.mm)
savefig(fig26, joinpath(FIGDIR, "fig26_is_vs_bdnk_evolution.pdf"))
println("  → fig26_is_vs_bdnk_evolution.pdf")

# ═══════════════════════════════════════════════════════════════════
#  Fig 27: Relative difference |IS - BDNK| / |IS|
# ═══════════════════════════════════════════════════════════════════
println("\n── Generating Fig 27: Relative difference ──")

# Use common tau grid (they should be the same)
# For J^τ at several radii
r_compare = [1.0, 3.0, 5.0]
colors_r = [:blue, :red, :green]

p27 = plot(xlabel=L"\tau\;[\mathrm{fm}/c]",
           ylabel=L"|J^\tau_{\mathrm{IS}} - J^\tau_{\mathrm{BDNK}}|\,/\,|J^\tau_{\mathrm{IS}}|",
           title="Relative difference in charge density",
           legend=:best, yscale=:log10, ylim=(1e-4, 10))

for (i, rv) in enumerate(r_compare)
    ir_i = nearest(r_is, rv)
    ir_b = nearest(r_bd, rv)
    # Match tau grids
    Nt = min(length(tau_is), length(tau_bd))
    Jtau_is = is["Jau"][ir_i, 1:Nt]
    Jtau_bd = bdnk["Jau"][ir_b, 1:Nt]
    rel = abs.(Jtau_is .- Jtau_bd) ./ (abs.(Jtau_is) .+ 1e-30)
    plot!(p27, tau_is[1:Nt], rel, lw=2, color=colors_r[i],
          label=L"r = %$(round(rv, digits=1))\;\mathrm{fm}")
end

fig27 = plot(p27, size=(600, 420), left_margin=5Plots.mm, bottom_margin=4Plots.mm)
savefig(fig27, joinpath(FIGDIR, "fig27_is_bdnk_reldiff.pdf"))
println("  → fig27_is_bdnk_reldiff.pdf")

println("\nDone.")
