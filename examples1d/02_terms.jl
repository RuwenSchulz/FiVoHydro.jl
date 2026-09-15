#!/usr/bin/env julia
#=
02 — switching terms off: the charm closures one term at a time, in both 1+1D solvers.

Every solver in this package takes a `terms` keyword and accepts the same spellings
(src/terms.jl; `show_terms()` prints the register):

    terms = :default                           # the equations as they stand
    terms = :homogeneous                       # a homogeneous medium at rest: no ∇T, no a, no ∇u, no DT
    terms = without(:acceleration)             # drop the inertial terms
    terms = without(:vorticity, :acceleration) # drop by physical ingredient
    terms = (fm_dlnh = false,)                 # one named term
    terms = (preset = :full, without = (:acceleration,))

Part A runs a fireball in the BULK solver with the consistent first moment and removes its terms
by ingredient. Part B takes that fireball's T(τ, r) and u^r(τ, r) as the frozen background of the
CHARM IS2 solver — the production chain, bulk → charm, in one script — and does the same for
the consistent second moment.

Three things this shows that are easy to get wrong:
  * `:homogeneous` IS the shipped row: with `consistent_fm = true` it reproduces
    `consistent_fm = false` to the last bit (asserted below).
  * the pressure-gradient (∇T) and inertial (a) terms are each LARGE and cancel as a pair on a
    nearly ideal fluid — so dropping one of them is a much bigger change than dropping both.
  * in 1+1D the vorticity couplings vanish identically: `with = (:vorticity,)` changes nothing
    (asserted), and says so in `show_equations`. The 2+1D solver carries them.

Run:  julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples1d/02_terms.jl
=#
ENV["GKSwstype"] = "100"
using Printf, Plots, Interpolations
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
include(joinpath(_ROOT, "main.jl"));     using .hydro;             const H = hydro
include(joinpath(_ROOT, "main2IS2.jl")); using .hydro_current_IS2; const HI = hydro_current_IS2
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

const TAU0, TAUF, T_FO, DST = 0.4, 5.0, 0.1565, 0.1163
Tof(r) = 0.05 + 0.40*exp(-r^2/(2*3.0^2))
αof(r) = -4.0 + 1.0*exp(-r^2/(2*2.5^2))

function fireball(; terms = :default, consistent_fm = true, record = false)
    g = H.make_grid_1d(200; rmax = 12.0)
    m = H.build_model_1d(; enable_shear = true, eta_over_s = 0.1, enable_diff = true,
                           kappa_coeff = DST, consistent_fm, terms)
    U = H.allocate_state(g, m)
    H.initialize_from_radial!(U, g, m, TAU0, Tof, αof)
    τs = Float64[]; Ts = Vector{Float64}[]; urs = Vector{Float64}[]
    res = H.run_sim_1d!(U, g, m; τ0 = TAU0, τfinal = TAUF, dump_dt = record ? 0.05 : Inf,
                        on_dump = record ? (τ, U, wk) -> begin
                            f = H.fields_1d(g, U, m; τ, work = wk)
                            push!(τs, τ); push!(Ts, f.T); push!(urs, f.ur)
                        end : nothing)
    f = H.fields_1d(g, U, m; τ = res.τ, work = res.work)
    return (; f, U, m, τs, Ts, urs, r = f.r)
end

function main()
    println("="^90, "\nPart A — the bulk solver, consistent first moment\n", "="^90)
    H.show_equations(H.build_model_1d(; enable_diff = true, kappa_coeff = DST, consistent_fm = true,
                                        terms = H.without(:acceleration)))
    ref = fireball(; record = true)
    shipped = fireball(; consistent_fm = false)
    homog = fireball(; terms = :homogeneous)
    @assert homog.U == shipped.U          # the shipped row IS the homogeneous-medium reduction
    println("\n  consistent_fm = true, terms = :homogeneous  ==  consistent_fm = false, bit for bit ✓\n")

    cases = [("default (all terms)", ref),
             (":homogeneous (= shipped)", homog),
             ("without(:acceleration)", fireball(; terms = H.without(:acceleration))),
             ("without(:temperature_gradient)", fireball(; terms = H.without(:temperature_gradient))),
             ("without both (the Euler pair)", fireball(; terms = H.without(:acceleration, :temperature_gradient))),
             ("without(:expansion)", fireball(; terms = H.without(:expansion)))]
    fluid = ref.f.T .> T_FO
    sc = maximum(abs, ref.f.nur[fluid])
    @printf("  %-32s  max|Δν^r| over T > T_fo, rel. to max|ν^r| (τ = %.0f fm)\n", "configuration", TAUF)
    for (lbl, c) in cases
        @printf("  %-32s  %8.3f\n", lbl, maximum(abs, (c.f.nur .- ref.f.nur)[fluid])/sc)
    end
    # plot the FLUID (T > T_fo): beyond it the dilute edge, where the vacuum floor and the front
    # dominate, would set the scale and hide the differences that matter
    rfo = maximum(ref.r[fluid])
    pA = plot(; xlabel = "r [fm]", ylabel = "ν^r / n", legend = :bottomleft, xlims = (0, rfo),
              title = @sprintf("bulk solver, τ = %.0f fm, T > T_fo", TAUF))
    for (lbl, c) in cases
        sel = c.r .<= rfo
        plot!(pA, c.r[sel], (c.f.nur ./ max.(c.f.n, 1e-12))[sel]; label = lbl, ls = lbl[1] == ':' ? :dash : :solid)
    end

    println("\n", "="^90, "\nPart B — the charm IS2 solver on Part A's medium\n", "="^90)
    # the bulk run's T(τ, r), u^r(τ, r) as a frozen background: `analytic_background` takes any
    # functions of (τ, r) — here a bilinear interpolant of the recorded slices
    Tg = reduce(hcat, ref.Ts); ug = reduce(hcat, ref.urs)
    iT = extrapolate(interpolate((ref.r, ref.τs), Tg, Gridded(Linear())), Flat())
    iu = extrapolate(interpolate((ref.r, ref.τs), ug, Gridded(Linear())), Flat())
    bg = HI.analytic_background(; T = (τ, r) -> iT(r, τ), ur = (τ, r) -> iu(r, τ),
                                  r_grid = ref.r, t_grid = ref.τs)
    HI.show_equations_IS2(; consistent_fm = true, consistent_m2 = true, terms = HI.without(:acceleration))
    run(; terms = :default) = HI.run_static_IS2_test(; background = bg, DsT = DST, τ0 = TAU0 + 0.05,
                                  τfinal = TAUF - 0.2, Nr = 120, rmax = 10.0, dump_dt = 0.5,
                                  init_mode = :n_profile, n_profile = r -> 0.02*exp(-r^2/(2*2.8^2)),
                                  consistent_fm = true, consistent_m2 = true, terms, log_every = 10^9)
    rd = run()
    rv = run(; terms = (with = (:vorticity,),))
    @assert rv["piQr"] == rd["piQr"] && rv["PiQ"] == rd["PiQ"]
    println("\n  with = (:vorticity,) changes nothing in 1+1D (a radial flow has no vorticity) ✓\n")
    bcases = [("default", rd), (":homogeneous", run(; terms = :homogeneous)),
              ("without(:acceleration)", run(; terms = HI.without(:acceleration))),
              ("without(:shear)", run(; terms = HI.without(:shear))),
              ("without(:cooling)", run(; terms = HI.without(:cooling)))]
    j = size(rd["PiQ"], 2); r = rd["r_grid"]
    sP = maximum(abs, rd["PiQ"][:, j]); sπ = maximum(abs, rd["piQr"][:, j]); sν = maximum(abs, rd["nur"][:, j])
    @printf("  %-26s  max|Δν^r|/max|ν^r|  max|Δπ_Q^r|/max|π_Q^r|  max|ΔΠ_Q|/max|Π_Q|   (τ = %.1f fm)\n",
            "configuration", rd["t_grid"][j])
    for (lbl, c) in bcases
        @printf("  %-26s  %12.3f  %18.3f  %18.3f\n", lbl,
                maximum(abs, c["nur"][:, j] .- rd["nur"][:, j])/sν,
                maximum(abs, c["piQr"][:, j] .- rd["piQr"][:, j])/sπ,
                maximum(abs, c["PiQ"][:, j] .- rd["PiQ"][:, j])/sP)
    end
    pB = plot(; xlabel = "r [fm]", ylabel = "Π_Q", title = "charm IS2, consistent m2", xlims = (0, 8), legend = :bottomright)
    for (lbl, c) in bcases
        plot!(pB, r, c["PiQ"][:, j]; label = lbl, ls = lbl[1] == ':' ? :dash : :solid)
    end
    fig = plot(pA, pB; layout = (1, 2), size = (1150, 450), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    savefig(fig, joinpath(FIG, "ex02_terms.png"))
    println("\nwrote ", joinpath(FIG, "ex02_terms.png"))
    return true
end
main() || exit(1)
