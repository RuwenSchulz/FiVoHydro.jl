# ==============================================================================
# exotic_ic2d.jl — deliberately nasty initial conditions, to find out where the
# solver stops working rather than to produce physics.
#
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/exotic_ic2d.jl
#
# Every validated case so far is smooth or nearly so: Bjorken, Gubser, a sound
# wave, a Glauber ensemble, or a single collision event. These are none of those:
#
#   smooth          the reference, an ordinary Woods-Saxon-ish blob
#   noise_10/30/60  multiplicative white noise on T at 10 / 30 / 60%, correlated
#                   over ~0.5 fm -- local fluctuations at the sub-nucleon scale
#   noise_cell      the same at 30% but correlated over ONE CELL, i.e. grid-scale
#                   structure the scheme cannot resolve by construction
#   hotspots        five sharp Gaussians (sigma = 0.4 fm) at 3x the ambient T
#   ring            a hollow donut: cold centre, hot shell at r = 5 fm
#   binary          two well-separated blobs, 8 fm apart
#   filament        a thin hot ridge, 0.5 fm wide and 12 fm long
#
# Reported per case: does it finish, primitive-recovery failures, max|u| above
# freeze-out, charge conservation, min(P+Pi), and max|pi|/P. A case that finishes
# with admissible fields is a pass; a case that finishes with max|u| ~ 20 is the
# silent failure mode that G9 exists to catch.
# ==============================================================================
using Printf, Random, Statistics, Plots
const _ROOT = @__DIR__
include(joinpath(_ROOT, "main2D.jl")); using .hydro2d; const H = hydro2d
const OUT = joinpath(_ROOT, "plots2d"); mkpath(OUT)
const T_FO = 0.1565
gr(); default(fontfamily = "sans-serif", grid = false, framestyle = :box)

"""Box-smoothed white noise with correlation length ~ell."""
function noise_field(nx, ny, dx, ell, seed)
    rng = MersenneTwister(seed)
    w = randn(rng, nx, ny)
    w .-= mean(w); w ./= std(w)
    ell <= dx && return w
    h = max(round(Int, ell/(2dx)), 1)
    s = similar(w)
    for a in 1:nx, b in 1:ny
        acc = 0.0; c = 0
        for p in max(1,a-h):min(nx,a+h), q in max(1,b-h):min(ny,b+h)
            acc += w[p,q]; c += 1
        end
        s[a,b] = acc/c
    end
    s ./= std(s)
    return s
end

"""T(x,y) for each exotic case."""
function tfield(case, xs, ys, dx)
    nx, ny = length(xs), length(ys)
    T = zeros(nx, ny)
    base(r) = 0.45*exp(-r^2/(2*3.2^2)) + 0.02
    for a in 1:nx, b in 1:ny
        T[a,b] = base(hypot(xs[a], ys[b]))
    end
    if startswith(case, "noise")
        amp, ell = case == "noise_10" ? (0.10, 0.5) :
                   case == "noise_30" ? (0.30, 0.5) :
                   case == "noise_60" ? (0.60, 0.5) : (0.30, dx)
        w = noise_field(nx, ny, dx, ell, 20260902)
        @inbounds for a in 1:nx, b in 1:ny
            T[a,b] *= (1 + amp*w[a,b])
        end
    elseif case == "hotspots"
        spots = ((0.0,0.0),(3.0,2.0),(-2.5,3.5),(2.0,-3.5),(-3.5,-2.0))
        for a in 1:nx, b in 1:ny
            for (sx, sy) in spots
                T[a,b] += 0.9*exp(-((xs[a]-sx)^2+(ys[b]-sy)^2)/(2*0.4^2))
            end
        end
    elseif case == "ring"
        for a in 1:nx, b in 1:ny
            r = hypot(xs[a], ys[b])
            T[a,b] = 0.45*exp(-(r-5.0)^2/(2*1.0^2)) + 0.02
        end
    elseif case == "binary"
        for a in 1:nx, b in 1:ny
            T[a,b] = 0.02 +
                0.45*exp(-((xs[a]-4.0)^2+ys[b]^2)/(2*1.6^2)) +
                0.45*exp(-((xs[a]+4.0)^2+ys[b]^2)/(2*1.6^2))
        end
    elseif case == "filament"
        for a in 1:nx, b in 1:ny
            T[a,b] = 0.02 + 0.45*exp(-ys[b]^2/(2*0.5^2))*exp(-xs[a]^2/(2*6.0^2))
        end
    end
    return max.(T, 0.012)
end

function run_case(case; N = 220, box = 16.0, τf = 8.0, clip = 1.0, Piclip = -1.0)
    g = H.make_grid2d(N, N; xmax = box, ymax = box)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0,
        pi_clip_factor = clip, Pi_clip_factor = Piclip)
    L = m.layout; U = H.allocate_state(g, m); wk = H.make_work(g, m)
    ng = g.nghost
    xa = [g.xC[ix] for ix in 1:g.Nxtot]; ya = [g.yC[iy] for iy in 1:g.Nytot]
    Tf = tfield(case, xa, ya, g.dx)
    nbad = 0
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        H.set_cell!(U, H.lin(g,ix,iy), Tf[ix,iy], -4.2, 0.0, 0.0, 0.4, m) || (nbad += 1)
    end
    H.finalize_ic!(U, g, m; τ0 = 0.4)
    idx = [H.lin(g,ix,iy) for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    Q0 = sum(U[L.iDtau,i] for i in idx)
    snap(τ) = begin
        A = zeros(g.Nx, g.Ny)
        for a in 1:g.Nx, b in 1:g.Ny; A[a,b] = exp(wk.yT[H.lin(g,ng+a,ng+b)]); end
        A
    end
    H.rhs_2d!(wk.k, U, g, 0.4, m, wk); T0 = snap(0.4)
    # Sample along the way and keep the WORST value. Reading only the final state
    # is vacuous for these cases: most have no cell above T_fo left at tau = 8, so
    # max|u| comes back 0.00 and min(P+Pi) comes back +Inf -- a "pass" with nothing
    # in it. The stress happens early, when the sharp structure is still there.
    t0 = time(); ok = true; res = nothing; τ = 0.4
    maxu = 0.0; minP = Inf; mpi = 0.0; pf = 0; nhot_peak = 0
    probe!() = begin
        nh = 0
        for i in idx
            exp(wk.yT[i]) > T_FO || continue
            nh += 1
            maxu = max(maxu, hypot(wk.ux[i], wk.uy[i]))
            minP = min(minP, wk.P[i] + H.phys_from_stored(U[L.iPi,i]))
            wk.P[i] > 0 && (mpi = max(mpi, max(abs(H.phys_from_stored(U[L.iPixx,i])),
                                               abs(H.phys_from_stored(U[L.iPiyy,i])))/wk.P[i]))
        end
        nhot_peak = max(nhot_peak, nh)
    end
    probe!()
    try
        for τt in (0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, τf)
            τt > τf && break
            res = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = τt, CFL = 0.15, CFLτ = 0.05, work = wk)
            ok = res.ok; ok || break
            τ = res.τ; pf += res.nprimfail
            H.update_primitives_2d!(U, g, τ, m, wk)
            probe!()
        end
    catch err
        ok = false
        @printf("  %-11s CRASHED: %s\n", case, sprint(showerror, err))
    end
    el = time() - t0
    if !ok
        return (; case, ok, T0, T1 = snap(τ), maxu, dQ = NaN, minP, mpi, pf, el, nbad)
    end
    dQ = abs(sum(U[L.iDtau,i] for i in idx) - Q0)/abs(Q0)
    verdict = (maxu < 3 && dQ < 1e-2 && minP > 0 && nhot_peak > 100) ? "OK" :
              nhot_peak <= 100 ? "(no fireball to test)" : "*** INADMISSIBLE ***"
    @printf("  %-11s %5.1fs | T0 max %.3f | hot cells %6d | pf %7d | WORST over the run: max|u| %6.2f  min(P+Pi) %+9.1e  max|pi|/P %7.3f | dQ %8.1e | %s\n",
            case, el, maximum(T0), nhot_peak, pf, maxu, minP, mpi, dQ, verdict)
    return (; case, ok, T0, T1 = snap(τ), maxu, dQ, minP, mpi, pf, el, nbad)
end

function main()
    cases = ["smooth","noise_10","noise_30","noise_60","noise_cell",
             "hotspots","ring","binary","filament"]
    println("### exotic initial conditions — does it run, and does it stay admissible?")
    rs = [run_case(c) for c in cases]
    # The two that came back inadmissible do so through min(P+Pi) < 0, i.e. the
    # BULK pressure -- and `Pi_clip_factor` is off by default, exactly as
    # `pi_clip_factor` was before G9. Does turning it on rescue them?
    println("\n### the two inadmissible cases, with the BULK regulator on")
    for c in ("noise_60", "hotspots")
        run_case(c; Piclip = 1.0)
    end
    # montage: T at tau0 (top) and tau=8 (bottom)
    g = H.make_grid2d(220, 220; xmax = 16.0, ymax = 16.0)
    ng = g.nghost
    xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]; ys = [g.yC[iy] for iy in (ng+1):(ng+g.Ny)]
    sel = ["noise_30","noise_cell","hotspots","ring","binary","filament"]
    ps = Any[]
    for nm in sel
        r = rs[findfirst(x -> x.case == nm, rs)]
        for (A, lab) in ((r.T0, "τ = 0.4"), (r.T1, "τ = 8.0"))
            push!(ps, heatmap(xs, ys, permutedims(A); c = :inferno, aspect_ratio = 1,
                  xlims = extrema(xs), ylims = extrema(ys), colorbar = false,
                  title = "$nm  $lab", titlefontsize = 8,
                  tickfontsize = 5, xlabel = "", ylabel = ""))
        end
    end
    f = plot(ps...; layout = (3, 4), size = (1250, 900),
             plot_title = "Exotic initial conditions — T(x,y), initial and at τ = 8 fm/c",
             plot_titlefontsize = 12)
    savefig(f, joinpath(OUT, "exotic_ics.png")); savefig(f, joinpath(OUT, "exotic_ics.pdf"))
    println("  wrote ", joinpath(OUT, "exotic_ics.png"))
end
main()
