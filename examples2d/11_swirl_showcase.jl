#!/usr/bin/env julia
#=
11 — the swirl showcase: a violently fluctuating event with an initial rotation, every transport
     sector on, run as long as the fluid lasts, rendered for the eye.

Example 10 is the same idea on a REAL MC-Glauber event. This one asks a different question: what
does the medium do when you hand it far more structure than a real event carries, AND spin it? The
initial condition is deliberately extreme — twice the hot spots of example 04, a stronger
modulation, and a rigid-body swirl laid on top — so that the vortical and dissipative physics is
large enough to watch rather than to measure.

⚠ THIS IS A PICTURE, NOT A MEASUREMENT. The IC is synthetic and tuned for visual drama; no number
here belongs in a paper. The validated configurations are examples 04 (fluctuating), 07 (a real
event) and 10 (the showcase), and the gates are ../test/run2d_gates.jl.

THE TRAP THIS FILE INHERITS — and it is measured, not folklore. Example 04's header records what
happens if you build T(x,y) from a bare sum of Gaussians: max|u| = 14–78 (v > 0.99) and 4e4–3e6
primitive-recovery failures, against 8e3 for a whole gate on a real event. Switching diffusion off
or clipping the domain changed nothing; only the profile did. So the IC here is built the way 04
earned:

    T(x,y) = (production radial profile from data/initial_profiles_physical.csv) × modulation(x,y)

The CSV carries a taper, an edge and a matched α(r) that the solver's floors and vacuum cut were
tuned against. A hand-built profile has to earn those; inheriting them is the cheap way.

⚠ `pi_clip_factor = 1.0` IS REQUIRED on a lumpy event (example 04, added 2026-09-10). Without it a
run like this stops partway: the shear grows past |π| ~ P in a hot spot and the state stops being
invertible. It is what gate G9 uses, and TWOD_PROGRAM.md §6j measured it inert on smooth ICs.

⚠ THE SWIRL IS AN INITIAL CONDITION, NOT A CONSERVED SPIN. Milne coordinates with boost invariance
carry no angular-momentum conservation law for the transverse plane — the swirl is free to decay,
and it does. What survives is the vorticity it seeds in the lumps, which is the point.

⚠ AT c_M = 0 THE CHARM SECOND MOMENT IS PASSIVE: `m2_vorticity = true` moves π_Q and Π_Q and
NOTHING else — T, u^i and ν^i are bit-identical to a run with it off (example 08 measures exactly
that). Read the π_Q panel for the vorticity coupling, not the fireball ones.

Knobs, all ENV-overridable, all recorded in the filename so no two renders collide:

    EX11_N=480 EX11_TAUF=12 EX11_NFRAME=90 EX11_SIZE=900 EX11_FPS=15 \
        julia -t auto --project=. examples2d/11_swirl_showcase.jl

The committed defaults run in a few minutes. `EX11_SWIRL=0` turns the rotation off for comparison.
=#
ENV["GKSwstype"] = "100"
using Printf, Statistics, Random, Plots

# Figure style, shared by every example in BOTH packages (2026-09-14). See example 10's header for
# why the margins are not cosmetic.
gr(); default(; fontfamily = "sans-serif", framestyle = :box, grid = false, dpi = 200, lw = 2.2,
               titlefontsize = 11, guidefontsize = 10, tickfontsize = 9, legendfontsize = 8,
               foreground_color_legend = nothing, background_color_legend = RGBA(1,1,1,0.75),
               left_margin = 9Plots.mm, bottom_margin = 6Plots.mm, right_margin = 3Plots.mm,
               top_margin = 2Plots.mm, colorbar_titlefontsize = 8)

const _ROOT = normpath(joinpath(@__DIR__, ".."))
# main.jl too: the IC loader `load_initial_interpolants` lives in the 1-D module, and every 2-D
# example that starts from the production profile includes both (see example 04).
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d; const H = hydro2d
const FIG  = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)
const ANIM = joinpath(FIG, "anim");        isdir(ANIM) || mkpath(ANIM)

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0   = 0.4
const TAUF   = parse(Float64, get(ENV, "EX11_TAUF",   "10.0"))
const N      = parse(Int,     get(ENV, "EX11_N",      "320"))
const NFRAME = parse(Int,     get(ENV, "EX11_NFRAME", "60"))
const ASIZE  = parse(Int,     get(ENV, "EX11_SIZE",   "760"))
const FPS    = parse(Int,     get(ENV, "EX11_FPS",    "12"))
const SWIRL  = parse(Float64, get(ENV, "EX11_SWIRL",  "0.22"))   # rigid-body u^φ at r = R_SW
const SEED   = parse(Int,     get(ENV, "EX11_SEED",   "20260915"))
const NSRC   = parse(Int,     get(ENV, "EX11_NSRC",   "50"))     # 2x example 04
const LBOX   = 16.0
const T_FO   = 0.1565
const T_VAC  = 0.100      # the fade threshold for rendering, NOT a physics cut
const R_SW   = 6.0        # radius at which the swirl reaches its nominal strength
# ⚠ the box is ±16 fm but the FLUID never reaches the corners. Plotting the full box leaves each
# panel small inside a white frame, so the renders crop to where the matter is; the colour fade
# already marks the vacuum. Measured on this event: T > T_VAC stays inside ±13 fm to the last frame.
const LPLOT  = parse(Float64, get(ENV, "EX11_LPLOT", "13.0"))
const SFX    = "_N$(N)_tau$(replace(string(TAUF), "." => "p"))_sw$(replace(string(SWIRL), "." => "p"))"

# ── the initial condition ───────────────────────────────────────────────────────────────────────
"`nsrc` Gaussian hot spots inside a Woods–Saxon envelope (example 04's generator, more sources)."
function lumpy(seed, nsrc; R = 5.5, w = 0.75)
    rng = MersenneTwister(seed); src = Tuple{Float64,Float64,Float64}[]
    while length(src) < nsrc
        x, y = 2R*(rand(rng) - 0.5)*1.3, 2R*(rand(rng) - 0.5)*1.3
        rand(rng) < 1/(1 + exp((hypot(x, y) - R)/0.5)) && push!(src, (x, y, 0.6 + 0.8rand(rng)))
    end
    (x, y) -> sum(a*exp(-((x - cx)^2 + (y - cy)^2)/(2w^2)) for (cx, cy, a) in src)
end

# ⚠ a STRONGER modulation than example 04's (0.40 + 0.60·… against 0.55 + 0.45·…). The mean is
# still ≈ 1 by construction, so the event carries the production profile's total energy; what
# changes is the contrast between hot spot and valley. Pushed much past this the run needs the
# clip to work hard and the picture stops being physics.
function modulation(dens; L = LBOX)
    dmax = maximum(dens(x, y) for x in -L:0.25:L, y in -L:0.25:L)
    (x, y) -> 0.40 + 0.60*min(dens(x, y)/dmax/0.32, 1.8)
end

itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
    fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)

const MOD = modulation(lumpy(SEED, NSRC))
Tof(x, y) = Float64(itpT(hypot(x, y))) * MOD(x, y)
αof(x, y) = Float64(itpF(hypot(x, y)))

# The swirl: a rigid-body rotation that saturates at R_SW and falls off outside it, so the vacuum
# is not spun. u^φ = SWIRL·(r/R_SW) inside, SWIRL·(R_SW/r) outside — continuous, and it carries no
# net radial flow.
function swirl_u(x, y)
    r = hypot(x, y)
    r < 1e-9 && return (0.0, 0.0)
    uφ = SWIRL * (r <= R_SW ? r/R_SW : R_SW/r)
    (-uφ * y/r, uφ * x/r)          # (u^x, u^y) of a counter-clockwise rotation
end

println("\n", "="^94)
println("  11 — swirl showcase: fluctuating + rotating, all transport on")
println("="^94)
@printf("  N = %d (dx = %.3f fm)   τ: %.1f → %.1f   %d sources   swirl = %.2f   seed %d\n",
        N, 2LBOX/N, TAU0, TAUF, NSRC, SWIRL, SEED)

g = H.make_grid2d(N, N; xmax = LBOX, ymax = LBOX)
m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                       enable_shear = true, eta_over_s  = 0.10, tauShear_coeff = 0.2,
                       enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff    = 15.0,
                       enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff   = 1.0,
                       consistent_fm = true, consistent_m2 = true,
                       pi_clip_factor = 1.0,            # ⚠ REQUIRED on a lumpy event, see header
                       terms = (m2_vorticity = true,))  # passive at c_M = 0 — read the π_Q panel
H.show_equations(m)

U = H.allocate_state(g, m)
for ix in 1:g.Nxtot, iy in 1:g.Nytot
    x, y = g.xC[ix], g.yC[iy]
    ux, uy = swirl_u(x, y)
    H.set_cell!(U, H.lin(g, ix, iy), Tof(x, y), αof(x, y), ux, uy, TAU0, m)
end
H.finalize_ic!(U, g, m; τ0 = TAU0)

# ⚠ ω IS NOT ∂_x u^y − ∂_y u^x. The transverse vorticity of a boost-invariant flow carries the
# acceleration terms too, ω^{xy} = ½(∂_xu^y − ∂_yu^x + u^x a^y − u^y a^x), and a^i needs ∂_τu^i.
# So it is built from CONSECUTIVE dumps, exactly as example 08 does — this is that file's helper,
# copied rather than re-derived so the two examples cannot drift apart.
function kinematic_norms(fa, fb, τ, dτ, dx, dy)
    nx, ny = size(fa.T)
    ωn = zeros(nx, ny); σn = zeros(nx, ny)
    for i in 2:nx-1, j in 2:ny-1
        ux = fa.ux[i,j]; uy = fa.uy[i,j]; ut = sqrt(1 + ux^2 + uy^2)
        dtux = (fb.ux[i,j] - ux)/dτ;  dtuy = (fb.uy[i,j] - uy)/dτ
        dxux = (fa.ux[i+1,j] - fa.ux[i-1,j])/(2dx); dxuy = (fa.uy[i+1,j] - fa.uy[i-1,j])/(2dx)
        dyux = (fa.ux[i,j+1] - fa.ux[i,j-1])/(2dy); dyuy = (fa.uy[i,j+1] - fa.uy[i,j-1])/(2dy)
        ax = ut*dtux + ux*dxux + uy*dyux            # a^i = u^μ ∂_μ u^i
        ay = ut*dtuy + ux*dxuy + uy*dyuy
        ωxy = 0.5*(dxuy - dyux + ux*ay - uy*ax)
        θ   = dxux + dyuy + ut/τ
        sxx = dxux - θ/3; syy = dyuy - θ/3; sxy = 0.5*(dxuy + dyux)
        ωn[i,j] = abs(ωxy)
        σn[i,j] = sqrt(max(sxx^2 + syy^2 + 2sxy^2, 0.0))
    end
    ωn, σn
end

# ── the solve, collecting frames through the dump callback (not a re-solve per frame) ────────────
τs = Float64[]; frames = Any[]
function grab(τ, Uc, work)
    push!(τs, τ); push!(frames, H.fields_2d(g, Uc, work, m)); return nothing
end

dump_dt = (TAUF - TAU0)/(NFRAME - 1)
t0 = time()
res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAUF, CFL = 0.15, CFLτ = 0.05,
                    dump_dt = dump_dt, on_dump = grab)
wall = time() - t0
@printf("\n  solve: ok=%s  τ_end = %.3f  steps = %d  primfail = %d  max|u| = %.3f  dQ = %.2e  (%.0f s)\n",
        res.ok, res.τ, res.nsteps, res.nprimfail, res.maxu, res.dQ, wall)
res.ok || @warn "the solve did not complete — the animation below stops where it stopped"
# ⚠ THE SOLVER CAN EMIT A DUPLICATE FINAL DUMP (the last scheduled one and the end-of-run one
# coincide). ω and σ come from CONSECUTIVE dumps, so a zero interval divides by zero and NaNs the
# whole frame — example 08 hit exactly this. Drop the duplicate explicitly.
keep = [k for k in 1:length(τs)-1 if τs[k+1] - τs[k] > 1e-12]
dx = frames[1].x[2] - frames[1].x[1]
Ts    = [copy(frames[k].T) for k in keep]
ntau  = [frames[k].n .* τs[k] for k in keep]
ratio = Matrix{Float64}[]; pQ = Matrix{Float64}[]
for k in keep
    ωn, σn = kinematic_norms(frames[k], frames[k+1], τs[k], τs[k+1] - τs[k], dx, dx)
    push!(ratio, [σn[i] > 0 ? ωn[i]/σn[i] : 0.0 for i in eachindex(ωn)] |> v -> reshape(v, size(ωn)))
    push!(pQ, sqrt.(frames[k].pQxx.^2 .+ 2 .* frames[k].pQxy.^2 .+ frames[k].pQyy.^2))
end
τs = τs[keep]
nF = length(τs)
@printf("  %d frames collected, τ = %.2f … %.2f\n", nF, first(τs), last(τs))

# ── rendering ───────────────────────────────────────────────────────────────────────────────────
# ⚠ the colour scale is FIXED across frames and taken from cells that HOLD FLUID. Per-frame
# rescaling makes a cooling fireball look static; including the vacuum floor flattens every frame
# to one colour. Both learned in Projects/FiVoFluidumComparison/animate_fluctuating_fields.jl.
const BG = RGB(1.0, 1.0, 1.0)
function faded(Z, T, cmap, lo, hi)
    gr_ = cgrad(cmap)
    img = Matrix{RGB{Float64}}(undef, size(Z, 2), size(Z, 1))
    for j in axes(Z, 2), i in axes(Z, 1)
        w = clamp((T[i,j] - 0.6T_VAC)/(0.4T_VAC), 0.0, 1.0)
        c = RGB(get(gr_, clamp((Z[i,j] - lo)/(hi - lo), 0.0, 1.0)))
        img[j, i] = RGB(w*c.r + (1-w)*BG.r, w*c.g + (1-w)*BG.g, w*c.b + (1-w)*BG.b)
    end
    img
end

# ⚠ the quantile is NOT always p99.5. Smooth fields (T, n·τ) frame well at p99.5; |ω|/|σ| and |π_Q|
# span orders of magnitude with a filamentary tail that carries the top percentile, and at p99.5
# those panels render BLACK — the fireball crushed into the bottom 1 % of the map. Example 10's
# header measures this.
function scale(zs, q)
    v = filter(isfinite, reduce(vcat, [vec(zs[k][Ts[k] .> T_VAC]) for k in 1:nF]))
    isempty(v) ? 1.0 : quantile(v, q)
end

xs = g.xC[(g.nghost+1):(g.nghost+N)]
function panel(Z, T, cmap, hi, label, title; lo = 0.0)
    p = plot(xs, xs, faded(Z, T, cmap, lo, hi); aspect_ratio = 1,
             xlims = (-LBOX, LBOX), ylims = (-LBOX, LBOX),
             xlabel = "x  [fm]", ylabel = "y  [fm]",
             title = isempty(label) ? title : "$title\n$label",
             titlefontsize = 10, framestyle = :box, legend = false)
    # ⚠ NO `colorbar_title`: GR puts it on top of the colorbar's own tick labels when those are
    # long. The quantity goes in the panel title instead (measured on example 10).
    scatter!(p, [NaN], [NaN]; zcolor = [lo], c = cmap, clims = (lo, hi), ms = 0,
             colorbar = true, label = "")
    contour!(p, xs, xs, permutedims(T); levels = [T_FO], c = :white, lw = 1, alpha = 0.7,
             colorbar_entry = false)
    p
end

hi_T  = scale(Ts,    0.995)
hi_nt = scale(ntau,  0.995)
hi_r  = scale(ratio, 0.98)     # filamentary — see above
hi_pQ = scale(pQ,    0.90)     # filamentary
@printf("\n  colour scales (fixed across frames, over T > %.0f MeV):\n", 1e3T_VAC)
@printf("    T %.4f   n·τ %.3e   |ω|/|σ| %.3e (p98)   |π_Q| %.3e (p90)\n", hi_T, hi_nt, hi_r, hi_pQ)

function animate(zs, tag, label, cmap, hi)
    anim = @animate for k in 1:nF
        panel(zs[k], Ts[k], cmap, hi, label,
              @sprintf("τ = %5.2f fm/c   (N = %d, swirl = %.2f)", τs[k], N, SWIRL))
        plot!(; size = (ASIZE, round(Int, 0.86ASIZE)),
                xlims = (-LPLOT, LPLOT), ylims = (-LPLOT, LPLOT))
    end
    path = joinpath(ANIM, "ex11_$(tag)$(SFX).gif")
    gif(anim, path; fps = FPS, show_msg = false)
    println("    ", path)
    path
end

println("\n  animations:")
animate(Ts,    "T",          "T  [GeV]",   :inferno, hi_T)
animate(ntau,  "charm",      "n · τ",      :turbo,   hi_nt)
animate(ratio, "vorticity",  "|ω| / |σ|",  :magma,   hi_r)
animate(pQ,    "charmstress","|π_Q|",      :viridis, hi_pQ)

# ── the static figure: four fields at three times ───────────────────────────────────────────────
ks = [1, max(1, nF ÷ 3), max(1, 2nF ÷ 3), nF]
row(zs, cmap, hi, lab) = [plot!(panel(zs[k], Ts[k], cmap, hi, "", @sprintf("%s  τ = %.1f", lab, τs[k]));
                                xlims = (-LPLOT, LPLOT), ylims = (-LPLOT, LPLOT)) for k in ks]
plt = plot(row(Ts, :inferno, hi_T, "T")..., row(ntau, :turbo, hi_nt, "n·τ")...,
           row(ratio, :magma, hi_r, "|ω|/|σ|")..., row(pQ, :viridis, hi_pQ, "|π_Q|")...;
           layout = (4, 4), size = (1800, 1800), left_margin = 2Plots.mm,
           bottom_margin = 2Plots.mm, right_margin = 2Plots.mm, top_margin = 2Plots.mm)
let f = joinpath(FIG, "ex11_swirl$(SFX).png")
    savefig(plt, f); println("\n  ", f)
end

@printf("""

  READING IT
    T        the medium. The swirl is an INITIAL CONDITION, not a conserved spin — Milne with boost
             invariance has no transverse angular-momentum law — so watch it decay while the lumps
             it stirred keep turning.
    n·τ      the charm density with Bjorken dilution divided out, so what moves is REDISTRIBUTION
             rather than expansion. The lumps survive here longest.
    |ω|/|σ|  where the flow swirls rather than shears. ⚠ the OUTER EDGE is the most vortical part
             of the box on these events, and that is measured (example 08, EX08_DIAG=1), not a
             rendering artefact.
    |π_Q|    the charm shear stress, and the ONLY panel the vorticity coupling moves: at c_M = 0
             the second moment is passive, so T, u^i and ν^i are bit-identical with it off.

  ⚠ This is a picture. The IC is synthetic and tuned for drama; no number here belongs in a paper.
  The validated runs are examples 04, 07 and 10, and the gates are ../test/run2d_gates.jl.
""")
