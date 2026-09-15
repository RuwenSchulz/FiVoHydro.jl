#!/usr/bin/env julia
#=
01 — the 1+1D solver in one screen: build a model, look at its equations, run a fireball,
read every field back.

The library interface (src/api1d.jl) has the same names and defaults as the 2+1D solver's:

    g = make_grid_1d(Nr; rmax)                     # radial Milne, axis at a face
    m = build_model_1d(; sectors, coefficients, terms)
    show_equations(m)                              # what m integrates, term by term
    U = allocate_state(g, m); initialize_from_radial!(U, g, m, τ0, T(r), α(r))
    res = run_sim_1d!(U, g, m; τ0, τfinal)         # in memory; res.ok, res.dQ, res.maxu, …
    f = fields_1d(g, U, m; τ = res.τ, work = res.work)

Defaults are the bare ideal scheme: every sector is opt-in. This example turns on shear, bulk
and charge diffusion with the thermodynamically consistent first moment, prints the equations,
evolves a Pb+Pb-like fireball from τ = 0.4 to 8 fm and plots T, u^r, the shear channels and
the current at four times.

What the figure shows, so it is read right: the fireball expands into a COLD TAIL (T = 0.05 GeV,
the profile's floor), and its edge steepens into a relativistic front — u^r up to 1.8, the shear
spiking towards −(e + P) — which the bare scheme (no stabilizers) carries as a narrow feature.
The kinks near T ≈ 0.17–0.19 GeV sit in the lattice EoS's soft region. Read dissipative fields over
the fluid, T > T_fo, as the printed numbers do (README.md, "read before quoting a number").

Run:  julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples1d/01_first_run.jl
Equations, term by term: ../EQUATIONS1D.md.
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
include(joinpath(_ROOT, "main.jl")); using .hydro; const H = hydro
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

function main()
    g = H.make_grid_1d(300; rmax = 15.0)
    m = H.build_model_1d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10,      # τ_π = η/(0.2 T s), δ_ππ = 4/3 τ_π
                           enable_bulk  = true, zeta_over_s = 0.10,
                           enable_diff  = true, kappa_coeff = 0.1163,   # D_s T
                           consistent_fm = true)                        # the full-∇P charm first moment
    H.show_equations(m)

    U = H.allocate_state(g, m)
    Tof(r) = 0.05 + 0.42*exp(-r^2/(2*3.2^2))          # a smooth Pb+Pb-sized profile, GeV
    αof(r) = -4.0 + 1.2*exp(-r^2/(2*2.5^2))            # charm fugacity μ/T
    H.initialize_from_radial!(U, g, m, 0.4, Tof, αof)

    snaps = Dict{Float64,Any}()
    t0 = time()
    res = H.run_sim_1d!(U, g, m; τ0 = 0.4, τfinal = 8.0, dump_dt = 0.1,
                        on_dump = (τ, U, wk) -> begin
                            for ts in (0.4, 2.0, 4.0, 8.0)
                                abs(τ - ts) < 0.05 && !haskey(snaps, ts) &&
                                    (snaps[ts] = H.fields_1d(g, U, m; τ, work = wk))
                            end
                        end)
    haskey(snaps, 8.0) || (snaps[8.0] = H.fields_1d(g, U, m; τ = res.τ, work = res.work))
    # dQ is not round-off here: charge leaves the box through the outflow boundary at r_max
    @printf("\nrun: ok = %s, %d steps in %.1f s, charge drift %.1e, max u^r = %.3f, recovery failures %d\n",
            res.ok, res.nsteps, time() - t0, res.dQ, res.maxu, res.primfail)

    # read dissipative fields over the FLUID only (T > T_fo): the dilute tail is dominated by
    # regulators and floors (README.md). By τ = 8 fm the whole fireball is below T_fo.
    f = snaps[4.0]
    cells = f.T .> 0.1565
    @printf("τ = 4 fm: T_max = %.3f GeV, fluid (T > T_fo) out to r = %.2f fm, max|π^η_η|/P = %.3f, max|ν^r|/n = %.3f\n",
            maximum(f.T), maximum(f.r[cells]), maximum(abs.(f.piEta[cells]) ./ f.P[cells]),
            maximum(abs.(f.nur[cells]) ./ f.n[cells]))
    @printf("τ = 8 fm: T_max = %.3f GeV — below T_fo everywhere\n", maximum(snaps[8.0].T))

    ps = [plot(; xlabel = "r [fm]", ylabel = lab, legend = k == 1 ? :topright : false)
          for (k, lab) in enumerate(("T [GeV]", "u^r", "π^η_η / (e+P)", "ν^r / n"))]
    for ts in sort(collect(keys(snaps)))
        s = snaps[ts]; lbl = @sprintf("τ = %.1f fm", ts)
        plot!(ps[1], s.r, s.T; label = lbl)
        plot!(ps[2], s.r, s.ur; label = lbl)
        plot!(ps[3], s.r, s.piEta ./ (s.e .+ s.P); label = lbl)
        plot!(ps[4], s.r, s.nur ./ max.(s.n, 1e-12); label = lbl)
    end
    for p in ps; xlims!(p, 0, 12); end
    ylims!(ps[4], -0.5, 0.5)
    fig = plot(ps...; layout = (2, 2), size = (1000, 700), plot_title = "FiVo 1+1D — a viscous, diffusing fireball")
    savefig(fig, joinpath(FIG, "ex01_first_run.png"))
    println("wrote ", joinpath(FIG, "ex01_first_run.png"))
    return res.ok
end
main() || exit(1)
