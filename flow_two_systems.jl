#!/usr/bin/env julia
# ==============================================================================
# flow_two_systems.jl — Pb+Pb vs O+O: does the small system have time to flow?
#
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/flow_two_systems.jl
#
# Extends `physics_scans.jl`'s centrality scan to BOTH systems. It became possible
# on 2026-09-05, when ALICE_IC_Creation's 2-D builder stopped needing an external
# per-system alpha0 table (the charm A(T) is now the closed-form EOS) — that table
# was the only thing blocking an O+O 2-D initial condition.
#
# THE OBSERVABLE. eps_p = <T^xx - T^yy>/<T^xx + T^yy> over the WHOLE transverse
# plane, and R = eps_p/eps2 is the hydrodynamic RESPONSE: how much of the initial
# spatial deformation the medium managed to convert. The fireball AREA (cells above
# T_fo) is tracked separately, and only to define a lifetime.
#
# ⚠ eps_p IS NOT v2. There is no 2-D Cooper-Frye yet, so nothing here is a particle
# spectrum and none of it is comparable to an ALICE measurement. It is the fluid's
# own anisotropy. The 2-D normalisation is inherited from the 1-D calibration and
# has never been checked against a measured yield either (BuildIC2D.jl header).
#
# WHEN TO READ IT. Each class is read at its OWN lifetime: tau_life is where the
# freeze-out area falls back through 90% of its initial value, found by LINEAR
# INTERPOLATION between samples, and eps_p is interpolated to the same instant.
# Later samples are a small, highly anisotropic dying remnant on which eps_p jumps
# by a factor of two — that is the T > T_fo cut, not physics.
#
# Two earlier versions of this got it wrong, and both failures were in the SAMPLING
# rather than the solver:
#   * reading every class at a fixed tau reported eps_p = 0 for four of twelve,
#     because those fireballs were already gone;
#   * reading at the last GRID POINT still healthy, on a grid that stepped
#     3.5 -> 4.0 -> 5.0, put O+O 0-5% at tau=5 and its neighbours at tau=4 while
#     eps_p was still climbing steeply. That produced a 50% spike in central O+O
#     that looked like physics and was arithmetic. Hence the uniform fine grid and
#     the interpolation.
#
# CENSORING. A class whose fireball outlives TAU_MAX has no measurable lifetime
# here; it is reported with `censored = true` and drawn as an open marker, never as
# a data point at TAU_MAX. Pb+Pb 0-5% was censored at 10 fm/c in the first pass and
# plotted as though 10.0 were its lifetime.
# ==============================================================================
using Printf, Statistics, Plots, LaTeXStrings, Colors

const _ROOT = @__DIR__
include(joinpath(_ROOT, "main2D.jl")); using .hydro2d; const H = hydro2d

const IC    = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation"))
const OUT   = joinpath(_ROOT, "plots2d")
const T_FO  = 0.1565
const TAU0  = 0.4
const TAUS  = collect(0.4:0.25:16.0)      # uniform and fine: the read-out is interpolated on it
const CACHE = joinpath(@__DIR__, "plots2d", "flow_two_systems_scan.csv")
const CENT  = ["00-05","05-10","10-20","20-30","30-40","40-50"]
const XC    = [2.5, 7.5, 15.0, 25.0, 35.0, 45.0]      # class centres, for the x axis

# Pb+Pb firebrick is kept; O+O navy is REPLACED by a lighter blue. The pair
# (#B22222, #000080) fails a lightness-band check — navy sits at L = 0.271, below
# the 0.43 floor, so it reads as near-black on a projector and in print. #1F5FBF
# keeps the same identity and passes every check, with CVD separation dE 24.7
# (protan) against firebrick.
const COL = Dict("PbPb" => colorant"#B22222", "OO" => colorant"#1F5FBF")
const LBL = Dict("PbPb" => "Pb+Pb 5.02 TeV", "OO" => "O+O 5.36 TeV")
const MRK = Dict("PbPb" => :circle, "OO" => :diamond)   # identity is never colour alone

read_meta(p) = (d = Dict{String,String}();
    for ln in eachline(p); f = split(ln, "="; limit=2)
        length(f) == 2 && (d[strip(f[1])] = strip(split(f[2], "#")[1])); end; d)

"""Evolve one IC on `TAUS`; return eps_p, <u_T> and the freeze-out area at each."""
function run_one(sys, tag; N, box)
    g = H.make_grid2d(N, N; xmax = box, ymax = box)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0, pi_clip_factor = 1.0)
    L = m.layout; U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_from_grid_csv!(U, g, m, TAU0, joinpath(IC, sys, "data", "ic2d_$(tag).csv"))
    ng = g.nghost
    idx = [H.lin(g, ix, iy) for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    dA = g.dx*g.dy
    epsp = Float64[]; uT = Float64[]; area = Float64[]
    function sample(τ)
        H.update_primitives_2d!(U, g, τ, m, wk)
        pxx = 0.0; pyy = 0.0; su = 0.0; sw = 0.0; a = 0.0
        # eps_p over the WHOLE grid, with NO freeze-out cut.
        #
        # ⚠ Cutting on T > T_fo — which an earlier version of this file did — makes eps_p blow up as
        # the fireball dies: the cut removes the cool, nearly isotropic outside first and leaves the
        # most anisotropic cells behind, so eps_p turns sharply upward in the last fm/c of every
        # class. Measured, O+O 0-5% went 0.24 -> 0.44 over its final step, and reading the
        # "saturated" value anywhere near there reports the artefact rather than the flow. The whole
        # -grid T^{ij} has no such bias; the vacuum contributes nothing because it carries no energy.
        #
        # The T_fo cut is still used for the AREA, which is what defines a fireball and hence a
        # lifetime — that is the one thing it is the right tool for.
        for i in idx
            T = exp(wk.yT[i])
            e = wk.e[i]; P = wk.P[i]; ux = wk.ux[i]; uy = wk.uy[i]
            Pi = L.hasPi ? H.phys_from_stored(U[L.iPi,i]) : 0.0
            pxx += (e+P+Pi)*ux*ux + (P+Pi) + (L.hasShear ? H.phys_from_stored(U[L.iPixx,i]) : 0.0)
            pyy += (e+P+Pi)*uy*uy + (P+Pi) + (L.hasShear ? H.phys_from_stored(U[L.iPiyy,i]) : 0.0)
            su += e*hypot(ux, uy); sw += e
            T > T_FO && (a += dA)
        end
        push!(epsp, (pxx+pyy) > 0 ? (pxx-pyy)/(pxx+pyy) : 0.0)
        push!(uT, sw > 0 ? su/sw : 0.0); push!(area, a)
    end
    sample(TAU0)
    for k in 2:length(TAUS)
        res = H.run_sim_2d!(U, g, m; τ0 = TAUS[k-1], τfinal = TAUS[k], CFL = 0.15, CFLτ = 0.05, work = wk)
        sample(res.τ)
    end
    return (; epsp, uT, area)
end

"""
    lifetime(taus, area, epsp) -> (tau_life, eps_p, censored)

The instant the freeze-out area falls back through 90% of its initial value, by linear
interpolation, with eps_p interpolated to the same instant. `censored = true` means the fireball
outlived the grid, so `tau_life` is a LOWER BOUND and must not be plotted as a measurement.
"""
function lifetime(taus, area, epsp)
    thr = 0.9*area[1]
    # ⚠ THE LAST crossing, not the first. The freeze-out area DIPS EARLY — A/A0 falls to ~0.89
    # around tau = 1 fm/c as the hot core cools through T_fo while the edge is still expanding —
    # and only then grows and finally collapses. A `findfirst(area < thr)` catches that birth
    # transient and reports a "lifetime" of 0.6-1.1 fm/c with eps_p ~ 0, which is what the first
    # version of this function did to all twelve classes.
    healthy = findall(k -> area[k] >= thr, eachindex(area))
    isempty(healthy) && return (taus[1], epsp[1], false)
    k = maximum(healthy)
    k == length(area) && return (taus[end], epsp[end], true)   # outlived the grid
    f = (area[k] - thr) / (area[k] - area[k+1])                # in [0,1]
    τ = taus[k] + f*(taus[k+1] - taus[k])
    e = epsp[k] + f*(epsp[k+1] - epsp[k])
    return (τ, e, false)
end

"Write the scan to `CACHE` so a re-plot does not re-run twelve hydro simulations."
function save_cache(res)
    open(CACHE, "w") do io
        println(io, "system,cent,eps2,eps3,tau_life,eps_p,R,censored,ep_tau3,R_tau3,uT,series_tau,series_ep,series_area")
        for sys in ("PbPb","OO"), r in res[sys]
            println(io, join((sys, r.c, r.e2, r.e3, r.τlife, r.epsp, r.R, r.censored,
                              r.ep3, r.R3, r.uT, join(r.series_tau,";"), join(r.series_ep,";"),
                              join(r.series_area,";")), ","))
        end
    end
    println("  cached the scan -> ", CACHE)
end

function load_cache()
    isfile(CACHE) || return nothing
    res = Dict{String,Any}("PbPb" => NamedTuple[], "OO" => NamedTuple[])
    for ln in Iterators.drop(eachline(CACHE), 1)
        f = split(ln, ",")
        # The cache stores the RAW SERIES and the derived scalars are recomputed here. A cache
        # written while the read-out was wrong would otherwise survive the fix and keep serving
        # the wrong numbers from a file that looks like data.
        st = parse.(Float64, split(f[12],";")); se = parse.(Float64, split(f[13],";"))
        sa = parse.(Float64, split(f[14],";"))
        e2 = parse(Float64,f[3])
        τl, epl, cens = lifetime(st, sa, se)
        i3 = argmin(abs.(st .- 3.0))
        push!(res[String(f[1])], (; c = String(f[2]), e2, e3 = parse(Float64,f[4]),
            τlife = τl, epsp = epl, R = epl/e2, censored = cens,
            ep3 = se[i3], R3 = se[i3]/e2, uT = parse(Float64,f[11]),
            series_tau = st, series_ep = se, series_area = sa))
    end
    println("  reusing the cached scan (", CACHE, ") — delete it to re-run the hydro")
    return res
end

function main(; refresh = "--refresh" in ARGS)
    mkpath(OUT)
    cached = refresh ? nothing : load_cache()
    if cached !== nothing
        plot_all(cached); return
    end
    res = Dict{String,Any}()
    for (sys, N, box) in (("PbPb", 250, 20.0), ("OO", 200, 9.0))
        rows = NamedTuple[]
        for c in CENT
            f = joinpath(IC, sys, "data", "ic2d_$(c).csv")
            isfile(f) || (@warn "absent, skipping" f; continue)
            md = read_meta(replace(f, ".csv" => "_meta.txt"))
            e2 = parse(Float64, md["eps2_ncoll"]); e3 = parse(Float64, md["eps3_ncoll"])
            r  = run_one(sys, c; N, box)
            τl, epl, cens = lifetime(TAUS, r.area, r.epsp)
            i3 = argmin(abs.(TAUS .- 3.0))
            push!(rows, (; c, e2, e3, τlife = τl, epsp = epl, R = epl/e2, censored = cens,
                           ep3 = r.epsp[i3], R3 = r.epsp[i3]/e2, uT = r.uT[i3],
                           series_tau = copy(TAUS), series_ep = copy(r.epsp),
                           series_area = copy(r.area)))
            @printf("%-5s %-6s eps2=%.4f  tau_life=%5.2f%s  eps_p=%.4f  R=%.3f  (at tau=3: R=%.3f)\n",
                    sys, c, e2, τl, cens ? ">" : " ", epl, epl/e2, r.epsp[i3]/e2); flush(stdout)
        end
        res[sys] = rows
    end
    save_cache(res)
    plot_all(res)
end

function plot_all(res)
    gr()
    default(fontfamily = "Computer Modern", framestyle = :box, grid = true,
            gridlinewidth = 0.4, gridcolor = :gray92, dpi = 200, legendfontsize = 7,
            legend_background_color = RGBA(1,1,1,0.85), legend_foreground_color = RGBA(0,0,0,0.15))

    mk(sys, f) = (getfield.(res[sys], f))
    """A panel. `censor_aware` draws points whose fireball outlived the grid as OPEN markers with
    an up-arrow: those are lower bounds, and plotting them as filled points would assert a lifetime
    the run never measured."""
    function panel(ylab, title, f; leg = false, censor_aware = false)
        p = plot(; xlabel = "centrality [%]", ylabel = ylab, title = title, titlefontsize = 9,
                   legend = leg ? :topleft : false)
        for sys in ("PbPb", "OO")
            y = mk(sys, f); x = XC[1:length(y)]; cen = mk(sys, :censored)
            keep = censor_aware ? .!cen : trues(length(y))
            plot!(p, x[keep], y[keep]; color = COL[sys], lw = 2.0,
                  marker = MRK[sys], markersize = 5, markerstrokewidth = 0.8,
                  markerstrokecolor = :white, label = LBL[sys])
            if censor_aware && any(cen)
                scatter!(p, x[cen], y[cen]; color = :white, markerstrokecolor = COL[sys],
                         markerstrokewidth = 1.6, marker = MRK[sys], markersize = 5,
                         label = "$(LBL[sys]): lower bound")
            end
        end
        p
    end

    p1 = panel(L"\varepsilon_2", "(a) initial eccentricity", :e2; leg = true)
    # (b) carries a legend ONLY if something is censored — that legend exists to explain the open
    # markers, and with nothing to explain it just sits on top of the Pb+Pb 0-5% point.
    p2 = panel(L"\tau_{\rm life}\ \mathrm{[fm/}c]", "(b) fireball lifetime", :τlife;
               leg = any(r -> r.censored, vcat(res["PbPb"], res["OO"])), censor_aware = true)
    p3 = panel(L"\varepsilon_p", "(c) momentum anisotropy, saturated", :epsp)
    p4 = panel(L"\varepsilon_p/\varepsilon_2", "(d) response", :R)
    fig = plot(p1, p2, p3, p4; layout = (2,2), size = (860, 620),
               left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
    savefig(fig, joinpath(OUT, "flow_two_systems.pdf"))
    savefig(fig, joinpath(OUT, "flow_two_systems.png"))
    println("\n  wrote ", joinpath(OUT, "flow_two_systems.pdf"))

    # eps_p(tau): the reason the two orderings differ
    pt = plot(; xlabel = L"\tau\ \mathrm{[fm/}c]", ylabel = L"\varepsilon_p",
                title = "momentum anisotropy builds; the small system runs out of time",
                # :bottomright, not :topleft — the curves climb into the top-left corner and the
                # legend box was sitting on top of the O+O 20-30% curve, hiding its peak entirely.
                titlefontsize = 9, legend = :bottomright, size = (620, 430),
                left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
    for sys in ("PbPb", "OO"), row in res[sys]
        row.c in ("00-05", "20-30") || continue
        n = something(findlast(<=(row.τlife), row.series_tau), length(row.series_tau))
        plot!(pt, row.series_tau[1:n], row.series_ep[1:n]; color = COL[sys],
              ls = row.c == "00-05" ? :solid : :dash, lw = 2.0,
              marker = MRK[sys], markersize = 3, markerstrokewidth = 0.5,
              markerstrokecolor = :white, label = "$(LBL[sys]) $(row.c)%")
    end
    savefig(pt, joinpath(OUT, "flow_two_systems_tau.pdf"))
    savefig(pt, joinpath(OUT, "flow_two_systems_tau.png"))
    println("  wrote ", joinpath(OUT, "flow_two_systems_tau.pdf"))
end
main()
