#!/usr/bin/env julia
#=
04 — a single lumpy event: triangular flow, and why it needs one event at a time.

v₃ has no place to come from in an azimuthally symmetric or an elliptic initial condition. It comes
from LUMPS, and lumps survive only in a single event: average many events and ε₃ falls away, because
the lumps sit somewhere different each time. Running events one at a time is the whole point.

⚠ IN THIS GENERATOR ε₂ IS ALSO PURE FLUCTUATION, and that is an artefact of the setup, not physics.
Every event here is centred — there is no impact parameter — so the 24-event mean washes out ε₂
(0.327 → 0.012) just as thoroughly as ε₃ (0.191 → 0.071). In a real non-central collision ε₂ has a
GEOMETRIC part from the almond overlap that survives averaging, and ε₃ does not. So read the two
columns below as "both are fluctuation-driven here"; do not read them as "ε₂ and ε₃ behave the same
way in data", because they do not.

⚠⚠ THE INITIAL CONDITION IS BUILT BY MODULATING THE PRODUCTION RADIAL PROFILE, NOT FROM SCRATCH, AND
THAT IS NOT A STYLISTIC CHOICE. The first version of this file built T(x,y) from a sum of Gaussian
hot spots directly — T = T_hot·(ρ/ρ_max)^{1/3} with a floor. It ran, it produced numbers, and the
numbers were garbage: max|u| = 14–78 (v > 0.99) and 4×10⁴–3×10⁶ primitive-recovery failures, against
8×10³ for the whole of gate G9 on a real fluctuating event. Measured on this harness, holding
everything else fixed:

    synthetic from scratch, diffusion on,  r_domain = Inf     max|u| = 13.8   primfail = 41025
    synthetic from scratch, diffusion off, r_domain = Inf     max|u| = 29.9   primfail = 38317
    synthetic from scratch, diffusion on,  r_domain = 12      max|u| = 13.8   primfail = 41025
    PRODUCTION radial profile (the control)                   max|u| =  0.88  primfail =     0
    production profile × the SAME lumpy modulation            max|u| =  2.46  primfail =   453

Switching diffusion off or clipping the domain changed nothing; only the profile did. The lesson is
not "synthetic ICs are bad" — it is that `data/initial_profiles_physical.csv` carries a taper, an
edge and a matched α(r) that the solver's floors and vacuum cut were tuned against, and a hand-built
profile has to earn those. The cheapest way to earn them is to inherit them, as below.

⚠ HARMONICS ARE MEASURED ABOUT THE PARTICIPANT PLANE, NOT THE GRID AXES. A lumpy event's ε₃ points
somewhere random; projecting onto cos 3φ with φ from +x throws most of it away and makes the answer
depend on how the event landed on the grid. Both ε_n and the response below carry their own Ψ_n.
This is invisible at ε₂ (a symmetric IC has Ψ₂ = 0) and catastrophic at ε₃.
=#
ENV["GKSwstype"] = "100"
using Printf, Random, Statistics, Plots
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
const TAU0, TAUF, RMAX, T_FO = 0.4, 8.0, 16.0, 0.1565

"A lumpy participant density: `nsrc` Gaussian hot spots inside a Woods–Saxon envelope."
function lumpy(seed, nsrc; R = 5.5, w = 0.9)
    rng = MersenneTwister(seed); src = Tuple{Float64,Float64,Float64}[]
    while length(src) < nsrc
        x, y = 2R*(rand(rng) - 0.5)*1.3, 2R*(rand(rng) - 0.5)*1.3
        rand(rng) < 1/(1 + exp((hypot(x, y) - R)/0.5)) && push!(src, (x, y, 0.7 + 0.6rand(rng)))
    end
    (x, y) -> sum(a*exp(-((x-cx)^2 + (y-cy)^2)/(2w^2)) for (cx, cy, a) in src)
end
"the modulation that turns the smooth production profile into one event; mean ≈ 1 by construction"
function modulation(dens; L = RMAX)
    dmax = maximum(dens(x, y) for x in -L:0.25:L, y in -L:0.25:L)
    (x, y) -> 0.55 + 0.45*min(dens(x, y)/dmax/0.35, 1.6)
end

"""ε_n and Ψ_n of the ENERGY DENSITY the solver is actually handed.

⚠ Not of the source density: the map from participants to T is nonlinear, so the two differ, and the
response ε_n → v_n is only meaningful against the field that was evolved."""
function eccentricity(Tof, n; L = RMAX, N = 160)
    xs = range(-L, L; length = N)
    function e(x, y)                        # energy density at this cell
        _, _, ee = H.eos_Pne(Tof(x, y), 0.0, H.LatticeHRGEOS())
        ee
    end
    cx = 0.0; cy = 0.0; m = 0.0
    for x in xs, y in xs; d = e(x, y); m += d; cx += d*x; cy += d*y; end
    cx /= m; cy /= m                          # ⚠ about the centre of mass, not the grid origin
    num = 0.0 + 0im; den = 0.0
    for x in xs, y in xs
        d = e(x, y); r = hypot(x-cx, y-cy); φ = atan(y-cy, x-cx)
        num += d*r^n*cis(n*φ); den += d*r^n
    end
    (abs(num)/den, angle(num)/n + π/n)
end

function run_event(N, Tof, αof; τf = TAUF)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    # ⚠ `pi_clip_factor = 1` IS REQUIRED ON A LUMPY EVENT (added 2026-09-10). Without it this run
    # stops at τ = 4.05: the shear grows past |π| ~ P in a hot spot and the state stops being
    # invertible. It USED to finish, because until commit 0c27f6f (2026-09-08) the admissibility
    # test was a tautology and MOOD/dt-halving could never fire — the example ran through an
    # inadmissible state and reported success (bisected: 0c27f6f^ passes, 0c27f6f fails at the
    # same τ bit for bit). The cap is what gate G9 uses; TWOD_PROGRAM.md §6j measured it inert on
    # smooth ICs (Δv₂ 4e-6) and v₃ independent of its value (0.1465 / 0.1461 at f = 1 / 2, 5, 10).
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           pi_clip_factor = 1.0,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        H.set_cell!(U, H.lin(g, ix, iy), Tof(g.xC[ix], g.yC[iy]), αof(g.xC[ix], g.yC[iy]),
                    0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok
    # ⚠ NOT a particle v_n. This is the anisotropy of the FLOW DIRECTION, weighted by E·|u| over
    # cells above freeze-out. A real v_n needs a freeze-out surface and Cooper-Frye (`freezeout2d.jl`);
    # this is the medium-side proxy that responds to the same physics, and its normalisation differs.
    ng = g.nghost; v = zeros(ComplexF64, 4); wsum = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy); exp(res.work.yT[i]) < T_FO && continue
        ux, uy = res.work.ux[i], res.work.uy[i]; up = hypot(ux, uy)
        up < 1e-12 && continue
        w = U[m.layout.iE, i]*up; wsum += w
        for n in 2:3; v[n] += w*cis(n*atan(uy, ux)); end
    end
    (v2 = abs(v[2])/wsum, v3 = abs(v[3])/wsum, res = res, g = g, U = U, m = m)
end

itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
    fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
αof(x, y) = Float64(itpF(hypot(x, y)))
event(seed, nsrc) = (mo = modulation(lumpy(seed, nsrc));
                     (x, y) -> Float64(itpT(hypot(x, y)))*mo(x, y))

println("\n  ONE EVENT vs the AVERAGE of many — the same generator, 24 seeds")
T1 = event(2992, 25)
e2, ψ2 = eccentricity(T1, 2); e3, ψ3 = eccentricity(T1, 3)
r1 = run_event(200, T1, αof)
@printf("    single event   ε₂ = %.4f (Ψ₂ = %+.2f)  ε₃ = %.4f (Ψ₃ = %+.2f)  ->  v₂ = %.4f  v₃ = %.4f\n",
        e2, ψ2, e3, ψ3, r1.v2, r1.v3)
mods = [modulation(lumpy(s, 25)) for s in 1:24]
Tavg(x, y) = Float64(itpT(hypot(x, y)))*mean(mo(x, y) for mo in mods)
a2, _ = eccentricity(Tavg, 2); a3, _ = eccentricity(Tavg, 3)
ra = run_event(200, Tavg, αof)
@printf("    24-event mean  ε₂ = %.4f                 ε₃ = %.4f                 ->  v₂ = %.4f  v₃ = %.4f\n",
        a2, a3, ra.v2, ra.v3)
@printf("    ratio single/averaged:  ε₃ %.1f×   v₃ %.1f×\n", e3/a3, r1.v3/max(ra.v3, 1e-12))

println("\n  resolution check on the single event (a lumpy IC is the demanding one):")
@printf("    %5s %8s %10s %10s %11s %9s\n", "N", "dx", "v₂", "v₃", "primfail", "max|u|")
# A function, not a top-level loop: the ladder's numbers are kept for the text at the end, and an
# accumulator in a top-level `for` is a fresh local every iteration (CLAUDE.md, trap 1).
function resolution_ladder(T1, αof)
    rows = NamedTuple[]
    for N in (150, 200, 300)
        r = run_event(N, T1, αof)
        @printf("    %5d %8.3f %10.5f %10.5f %11d %9.3f\n",
                N, 2RMAX/N, r.v2, r.v3, r.res.nprimfail, r.res.maxu)
        push!(rows, (v2 = r.v2, v3 = r.v3, pf = r.res.nprimfail))
    end
    rows
end
ladder = resolution_ladder(T1, αof)

g = r1.g; ng = g.nghost
xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]; ys = [g.yC[iy] for iy in (ng+1):(ng+g.Ny)]
T0 = [T1(x, y) for y in ys, x in xs]
Tf = [exp(r1.res.work.yT[H.lin(g, ix, iy)]) for iy in (ng+1):(ng+g.Ny), ix in (ng+1):(ng+g.Nx)]
plt = plot(layout = (1,2), size = (1000, 430))
heatmap!(plt[1], xs, ys, T0; c = :inferno, title = "T at τ = $TAU0 fm/c",
         xlabel = "x [fm]", ylabel = "y [fm]", aspect_ratio = 1)
heatmap!(plt[2], xs, ys, Tf; c = :inferno, title = "T at τ = $TAUF fm/c",
         xlabel = "x [fm]", aspect_ratio = 1)
savefig(plt, joinpath(FIG, "ex04_fluctuating_event.png"))
println("\n  -> ", joinpath(FIG, "ex04_fluctuating_event.png"))
println("""
  READING IT
    ε₃ survives in one event and averages away over many; the flow anisotropy follows it. That is the
    whole argument for event-by-event running, in two lines of output — with the caveat in the header
    that ε₂ washes out too here only because this generator has no impact parameter.
    The harmonics converge by N = 150 — see the ladder above; a lumpy IC is demanding for the
    recovery, not for the harmonics.
    The `primfail` column is a real diagnostic, not noise: those cells fall back to a floor, and the
    number to watch is whether it grows FASTER than the cell count as you refine. If it does, the run
    is being held together by the floors rather than by the scheme — which is exactly what the
    from-scratch initial condition in this file's header was doing, at 40 000 failures.""")
@printf("  This run: v₂ %s, v₃ %s at N = 150/200/300, primfail %s.\n",
        join((@sprintf("%.4f", r.v2) for r in ladder), "/"),
        join((@sprintf("%.4f", r.v3) for r in ladder), "/"),
        join((string(r.pf) for r in ladder), "/"))
println("  (With `pi_clip_factor = 1`; before the cap was required, commit 0c27f6f^ gave v₃ 0.1466/0.1455/")
println("   0.1453 with 1182/1995/3975 failures — the same harmonics, and the cap removes the failures.)")
