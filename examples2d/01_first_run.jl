#!/usr/bin/env julia
#=
01 — the minimal correct 2+1D run, and the check to do before any other.

Everything here is the API you will copy: grid, model, state, initial condition, driver, readout.
The physics is deliberately trivial — a transversely UNIFORM state, which must evolve as Bjorken —
because that is the one configuration where 2+1D has a closed-form answer, so it separates "the
solver works" from "my initial condition is what I think it is".

⚠ THE ORDER MATTERS.  `set_cell!` writes primitives into conserved variables cell by cell; it does
NOT apply floors, and a profile with a vacuum tail starts below the energy floor in its outer cells.
`finalize_ic!` is what makes the IC admissible, and skipping it is the single most common way to get
a run that dies in the first few steps.  `initialize_uniform!` calls it for you; a hand-built IC
must call it itself, as example 02 does.
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

const T0, ALPHA0, TAU0, TAUF = 0.45, -3.0, 0.4, 8.0
# α = μ/T is held FIXED in the reference: on a uniform state with no charge gradient the charge
# equation reduces to τn = const, and for the lattice-HRG list dln n/dα = 1 exactly, so α is
# constant along the Bjorken solution. Gate Gk of the ladder is what establishes that identity.

# ── the model ───────────────────────────────────────────────────────────────────────────────────
# Every dissipative sector is OFF by default (`build_model_2d` reproduces the bare ideal scheme),
# so each one has to be asked for explicitly along with its coefficients.  The layout shrinks with
# the sectors: with `enable_shear=false` the four shear dofs are not in the state vector at all,
# which is what lets the ideal gates run the bare scheme rather than a viscous one carrying zeros.
g = H.make_grid2d(48, 48; xmax = 12.0, ymax = 12.0)
m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                       enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                       enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff  = 15.0,
                       enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0)
U = H.allocate_state(g, m)
H.initialize_uniform!(U, g, m, TAU0; T0 = T0, alpha0 = ALPHA0)

# ── the reference: Bjorken with shear and bulk, integrated here ──────────────────────────────────
# Reference and solver share only the equation of state.  A uniform state has no transverse
# gradient, so the 2+1D solver must integrate exactly this:
#     De = −(e + P + Π + π^η_η)/τ ,   τ_π Dπ^η_η = −π^η_η − 4η/(3τ) − δ_ππ π^η_η/τ ,
#     τ_Π DΠ  = −Π − ζ/τ
# ⚠ FiVo runs DNMR, not plain Israel–Stewart: `deltaShear_factor = 4/3` puts a −δ_ππ π θ term in
# the shear equation.  Leave it out of the reference and the comparison is 10 % off — that term is
# also the whole reason a Fluidum-vs-FiVo comparison must be run with δ_ππ = 0 to be like-for-like
# (Julia/Projects/FiVoFluidumComparison/COMPARISON_2P1D.md §3).
function bjorken_reference(tau0, tau1, T0, alpha0, m; nsub = 20_000)
    eos = m.eos
    # RK4, not Euler: at this resolution the reference must be far more accurate than the solver,
    # otherwise the comparison measures the REFERENCE's truncation error. (Euler at nsub = 40_000
    # is already 8e-3 off in pi^eta_eta, which is the same size as the solver's own error.)
    function rhs(tau, y)
        T, pieta, Pi = y
        P, n, e = H.eos_Pne(T, alpha0*T, eos)
        eta, taupi, dpi = H.shear_coeffs_2d(T, alpha0*T, n, e, P, m)
        zeta, tauPi     = H.bulk_coeffs_2d( T, alpha0*T, n, e, P, m)
        h = 1e-6*T
        _, _, ep = H.eos_Pne(T + h, alpha0*(T + h), eos)
        _, _, em = H.eos_Pne(T - h, alpha0*(T - h), eos)
        ( (-(e + P + Pi + pieta)/tau)/((ep - em)/(2h)),
          (-pieta - dpi*pieta/tau - 4eta/(3tau))/taupi,
          (-Pi - zeta/tau)/tauPi )
    end
    y = (T0, 0.0, 0.0); dtau = (tau1 - tau0)/nsub
    for k in 0:nsub-1
        tau = tau0 + k*dtau
        k1 = rhs(tau, y)
        k2 = rhs(tau + dtau/2, y .+ (dtau/2) .* k1)
        k3 = rhs(tau + dtau/2, y .+ (dtau/2) .* k2)
        k4 = rhs(tau + dtau,   y .+  dtau    .* k3)
        y  = y .+ (dtau/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    y
end

# ── run, sampling the centre as we go ────────────────────────────────────────────────────────────
τs = Float64[]; Ts = Float64[]; πs = Float64[]; Πs = Float64[]
ic = H.lin(g, g.nghost + g.Nx÷2, g.nghost + g.Ny÷2)
res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.2, dump_dt = 0.2,
    on_dump = (τ, Uu, wk) -> begin
        push!(τs, τ); push!(Ts, exp(wk.yT[ic]))
        push!(πs, H.phys_from_stored(Uu[m.layout.iPieta, ic]))
        push!(Πs, H.phys_from_stored(Uu[m.layout.iPi,    ic]))
    end)
@assert res.ok

Tr, πr, Πr = bjorken_reference(TAU0, TAUF, T0, ALPHA0, m)
@printf("\n  at τ = %.1f fm/c        solver          reference       relative\n", TAUF)
@printf("    T          %14.9f  %14.9f  %9.2e\n", Ts[end], Tr, abs(Ts[end]-Tr)/Tr)
@printf("    π^η_η      %14.9f  %14.9f  %9.2e\n", πs[end], πr, abs(πs[end]-πr)/abs(πr))
@printf("    Π          %14.9f  %14.9f  %9.2e\n", Πs[end], Πr, abs(Πs[end]-Πr)/abs(Πr))

# The three checks a uniform run must pass, and which no reference is needed for:
ng = g.nghost
Tsp = let v = [exp(res.work.yT[H.lin(g,ix,iy)]) for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)]
    (maximum(v) - minimum(v))/maximum(v) end
umx = maximum(hypot(res.work.ux[H.lin(g,ix,iy)], res.work.uy[H.lin(g,ix,iy)])
              for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny))
@printf("\n  transverse spread of T   %.2e   (a uniform state must stay uniform)\n", Tsp)
@printf("  max |u|                  %.2e   (a uniform state must stay at rest)\n", umx)
@printf("  charge drift ΔQ/Q        %.2e   (D̃ = τJ^τ is conserved by construction)\n", res.dQ)
@printf("  steps %d, primitive-recovery failures %d\n", res.nsteps, res.nprimfail)

# ── why the agreement is 1e-3 and not 1e-13 ─────────────────────────────────────────────────────
# The advection is SSPRK2 and converges at second order (gate G0 of the ladder measures exactly
# 2.00). The DISSIPATIVE sectors are not advanced by the same integrator: they are relaxed by an
# operator split, once per accepted step, so the whole scheme is FIRST order in the step size
# whenever shear, bulk or diffusion is on. That is what the ladder below shows, and it is the reason
# a viscous run needs a smaller CFL than an ideal one for the same accuracy.
println("\n  operator-splitting order — the step size is set by CFLτ here, since a uniform")
println("  state has no transverse signal at all:")
# ⚠ JULIA SOFT SCOPE. `prev = e` inside a TOP-LEVEL `for` creates a new local every iteration, so
# reading `prev` from the previous one throws (or, in a PASS/FAIL gate, silently reports PASS —
# that is trap #1 in this repo's CLAUDE.md). Put the loop in a function, as here.
function splitting_ladder(g, m, ic, Tr)
    @printf("    %8s %14s %12s %8s\n", "CFLτ", "T(τ_f)", "|ΔT|/T", "steps")
    prev = NaN
    for cflt in (0.16, 0.08, 0.04, 0.02)
        Uq = H.allocate_state(g, m); H.initialize_uniform!(Uq, g, m, TAU0; T0 = T0, alpha0 = ALPHA0)
        r  = H.run_sim_2d!(Uq, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.2, CFLτ = cflt)
        Tq = exp(r.work.yT[ic]); e = abs(Tq - Tr)/Tr
        @printf("    %8.3f %14.9f %12.3e %8d%s\n", cflt, Tq, e, r.nsteps,
                isnan(prev) ? "" : @sprintf("   (%.2fx)", prev/e))
        prev = e
    end
end
splitting_ladder(g, m, ic, Tr)
println("    the ratio approaches 2 — first order, as the splitting requires. The coarsest rung is")
println("    not asymptotic yet, which is exactly why a convergence claim needs THREE points.")

plt = plot(layout = (1,3), size = (1150, 440), legend = :topright)
plot!(plt[1], τs, Ts; label = "solver", xlabel = "τ  [fm/c]", ylabel = "T  [GeV]", title = "temperature")
scatter!(plt[1], [TAUF], [Tr]; label = "Bjorken reference", ms = 6, mc = :black)
plot!(plt[2], τs, πs; label = "solver", xlabel = "τ  [fm/c]", ylabel = "π^η_η  [GeV/fm³]", title = "shear")
scatter!(plt[2], [TAUF], [πr]; label = "reference", ms = 6, mc = :black)
plot!(plt[3], τs, Πs; label = "solver", xlabel = "τ  [fm/c]", ylabel = "Π  [GeV/fm³]", title = "bulk")
scatter!(plt[3], [TAUF], [Πr]; label = "reference", ms = 6, mc = :black)
savefig(plt, joinpath(FIG, "ex01_bjorken.png"))
println("\n  -> ", joinpath(FIG, "ex01_bjorken.png"))
