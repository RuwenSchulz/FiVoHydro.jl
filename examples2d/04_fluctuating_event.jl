#!/usr/bin/env julia
#=
04 — a single lumpy event: triangular flow, and why it needs one event at a time.

v₃ has no place to come from in an azimuthally symmetric or an elliptic initial condition. It comes
from LUMPS, and lumps survive only in a single event: average a few hundred events in the lab frame
and ε₃ averages to something an order of magnitude smaller, because the lumps are in different
places each time. The whole point of running events one at a time is here.

The initial condition is built in this file from hot spots rather than read from `data/`, for two
reasons: it is self-contained, and the repo's stored profiles are φ-AVERAGED upstream of FiVo
(TWOD_PROGRAM.md §0) so they carry neither ε₂ nor ε₃ by construction.

⚠ HARMONICS ARE MEASURED ABOUT THE PARTICIPANT PLANE, NOT THE GRID AXES. A lumpy event's ε₃ points
somewhere random; projecting onto cos 3φ with φ measured from +x throws most of it away and the
answer then depends on how the event happened to land on the grid. Both ε_n and the response below
carry their own Ψ_n. Getting this wrong is invisible at ε₂ (a symmetric IC has Ψ₂ = 0) and
catastrophic at ε₃.
=#
ENV["GKSwstype"] = "100"
using Printf, Random, Statistics, Plots
gr(); default(; fontfamily = "sans-serif", framestyle = :box, grid = false, dpi = 150, lw = 2)

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

const TAU0, TAUF, RMAX, T_FO = 0.4, 8.0, 16.0, 0.1565
const T_HOT, ALPHA0 = 0.46, -3.0

"A lumpy participant density: `nsrc` Gaussian hot spots inside a Woods–Saxon envelope."
function lumpy(seed, nsrc; R = 5.5, w = 0.6)
    rng = MersenneTwister(seed); src = Tuple{Float64,Float64,Float64}[]
    while length(src) < nsrc
        x, y = 2R*(rand(rng) - 0.5)*1.3, 2R*(rand(rng) - 0.5)*1.3
        rand(rng) < 1/(1 + exp((hypot(x, y) - R)/0.5)) && push!(src, (x, y, 0.7 + 0.6rand(rng)))
    end
    (x, y) -> sum(a*exp(-((x-cx)^2 + (y-cy)^2)/(2w^2)) for (cx, cy, a) in src)
end
"ε_n and Ψ_n of a density, weighted by r^n as the definition requires."
function eccentricity(dens, n; L = RMAX, N = 120)
    xs = range(-L, L; length = N); num = 0.0 + 0im; den = 0.0
    cx = 0.0; cy = 0.0; m = 0.0
    for x in xs, y in xs
        d = dens(x, y); m += d; cx += d*x; cy += d*y
    end
    cx /= m; cy /= m                       # ⚠ about the centre of mass, not the grid origin
    for x in xs, y in xs
        d = dens(x, y); r = hypot(x-cx, y-cy); φ = atan(y-cy, x-cx)
        num += d*r^n*cis(n*φ); den += d*r^n
    end
    (abs(num)/den, angle(num)/n + π/n, cx, cy)
end

function run_event(N, dens; τf = TAUF)
    dmax = maximum(dens(x, y) for x in -RMAX:0.25:RMAX, y in -RMAX:0.25:RMAX)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2,
                           enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        # T ∝ (participant density)^{1/3}: entropy ∝ density and s ∝ T³ for the light sector
        T = T_HOT*max(dens(g.xC[ix], g.yC[iy])/dmax, 1e-8)^(1/3)
        H.set_cell!(U, H.lin(g, ix, iy), max(T, 0.02), ALPHA0, 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    @assert res.ok
    # the flow harmonic, about ITS OWN plane
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

println("\n  ONE EVENT vs the AVERAGE of many — the same generator, 24 seeds")
d1 = lumpy(2992, 14)
e2, ψ2, _, _ = eccentricity(d1, 2); e3, ψ3, _, _ = eccentricity(d1, 3)
r1 = run_event(200, d1)
@printf("    single event  ε₂ = %.4f (Ψ₂ = %+.2f)  ε₃ = %.4f (Ψ₃ = %+.2f)  ->  v₂ = %.4f  v₃ = %.4f\n",
        e2, ψ2, e3, ψ3, r1.v2, r1.v3)
davg = let ds = [lumpy(s, 14) for s in 1:24]; (x, y) -> mean(d(x, y) for d in ds) end
a2, _, _, _ = eccentricity(davg, 2); a3, _, _, _ = eccentricity(davg, 3)
ra = run_event(200, davg)
@printf("    24-event mean ε₂ = %.4f                ε₃ = %.4f                ->  v₂ = %.4f  v₃ = %.4f\n",
        a2, a3, ra.v2, ra.v3)
@printf("    ratio single/averaged:  ε₃ %.1f×   v₃ %.1f×\n", e3/a3, r1.v3/max(ra.v3, 1e-12))

println("\n  resolution check on the single event (a lumpy IC is the demanding one):")
for N in (150, 200, 300)
    r = run_event(N, d1)
    @printf("    N = %3d  dx = %.3f  v₂ = %.5f  v₃ = %.5f  primfail = %6d  max|u| = %.3f\n",
            N, 2RMAX/N, r.v2, r.v3, r.res.nprimfail, r.res.maxu)
end

g = r1.g; ng = g.nghost
xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]; ys = [g.yC[iy] for iy in (ng+1):(ng+g.Ny)]
dmax1 = maximum(d1(a, b) for a in -RMAX:0.25:RMAX, b in -RMAX:0.25:RMAX)   # hoisted out of the
T0 = [T_HOT*max(d1(x, y)/dmax1, 1e-8)^(1/3) for y in ys, x in xs]          # comprehension: O(N^4) inside it
Tf = [exp(r1.res.work.yT[H.lin(g, ix, iy)]) for iy in (ng+1):(ng+g.Ny), ix in (ng+1):(ng+g.Nx)]
plt = plot(layout = (1,2), size = (1000, 430))
heatmap!(plt[1], xs, ys, T0; c = :inferno, title = "T at τ = $TAU0 fm/c", xlabel = "x [fm]", ylabel = "y [fm]", aspect_ratio = 1)
heatmap!(plt[2], xs, ys, Tf; c = :inferno, title = "T at τ = $TAUF fm/c", xlabel = "x [fm]", aspect_ratio = 1)
savefig(plt, joinpath(FIG, "ex04_fluctuating_event.png"))
println("\n  -> ", joinpath(FIG, "ex04_fluctuating_event.png"))
println("""
  READING IT
    ε₃ survives in one event and averages away over many; v₃ follows it. The lumps also make this
    the hardest IC for the primitive recovery — the `primfail` column is a real diagnostic, not
    noise: those cells fall back to a floor, and if the count grows faster than the cell count as
    you refine, the run is being held together by the floors rather than by the scheme.""")
