# ==============================================================================
# physics2d.jl — physical results from the 2+1D solver, with the correctness
# checks that make them believable.
#
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/physics2d.jl
#
# Runs the three centrality classes plus two single events and extracts, as
# functions of proper time:
#
#   v2, v3      momentum anisotropy above freeze-out (harmonics of T^{tau i})
#   <u_T>       radial flow, energy-weighted
#   V_fo        transverse area above T_fo  ->  fireball lifetime
#   S(tau)      total entropy, tau * int s u^tau dx dy, s = (e+P)/T
#   N_c(tau)    charm number, tau * int D_tau
#
# THREE OF THESE ARE CORRECTNESS TESTS, not just outputs:
#
#   * S(tau) must be NON-DECREASING. The second law is not imposed anywhere in
#     the scheme — Israel-Stewart guarantees it in the continuum, a discretisation
#     need not. Entropy going DOWN would be a hard failure.
#   * the IDEAL run must conserve S exactly (up to outflow), so the ideal-vs-
#     viscous pair separates numerical entropy production from physical.
#   * N_c must be conserved: charm is a tracer with no source.
#
# and one is a physics prediction with a known sign:
#
#   * v2 must INCREASE from central to peripheral, following eps2.
#
# Writes plots2d/physics_*.png|pdf and a table to stdout.
# ==============================================================================

using Printf, Statistics
using Plots

const _ROOT = @__DIR__
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const DATA = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation", "PbPb", "data"))
const OUT  = joinpath(_ROOT, "plots2d")
const T_FO = 0.1565
const TAUS = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

gr(); default(fontfamily = "sans-serif", grid = false, framestyle = :box)

"""One run; returns the observables sampled on TAUS."""
function evolve(tag; N = 300, box = 18.0, ideal = false, clip = 1.0)
    g = H.make_grid2d(N, N; xmax = box, ymax = box)
    m = ideal ?
        H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_shear = false,
                           enable_bulk = false, enable_diff = false) :
        H.build_model_2d(; eos = H.LatticeHRGEOS(),
            enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
            enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
            enable_diff  = true, kappa_coeff = 0.1163, tauN_coeff = 1.0,
            pi_clip_factor = clip)
    L = m.layout; U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_from_grid_csv!(U, g, m, TAUS[1], joinpath(DATA, "ic2d_$(tag).csv"))
    ng = g.nghost
    idx = [H.lin(g, ix, iy) for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    xs  = [g.xC[ix] for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    ys  = [g.yC[iy] for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    dA  = g.dx*g.dy

    v2 = Float64[]; v3 = Float64[]; uT = Float64[]; Afo = Float64[]
    Sτ = Float64[]; Nc = Float64[]; T0 = Float64[]; eps2 = Float64[]; epsp = Float64[]

    function sample(τ)
        sc2=0.0; ss2=0.0; sc3=0.0; ss3=0.0; sw=0.0
        su=0.0; sE=0.0; area=0.0; Stot=0.0; Ntot=0.0
        ec=0.0; es=0.0; er=0.0
        # eps_p, the standard momentum anisotropy from the stress tensor over the
        # WHOLE fireball. The T_fo-restricted v2 below stops being a measurement
        # once the fireball is nearly gone: at tau = 8 the 30-40% class has 15 fm^2
        # above T_fo, a handful of cells, and v2 there read 0.95.
        pxx=0.0; pyy=0.0; pxy=0.0
        for (k, i) in enumerate(idx)
            T = exp(wk.yT[i]); e = wk.e[i]; P = wk.P[i]
            ux = wk.ux[i]; uy = wk.uy[i]; uτ = sqrt(1+ux*ux+uy*uy)
            Ntot += U[L.iDtau, i]      # D_tau = tau J^tau ALREADY carries the tau
            # entropy: s = (e+P)/T for the light sector (charm is a tracer and
            # must not enter; the -mu n/T piece belongs to the HQ sector alone).
            T > 1e-3 && (Stot += (e + P)/T * uτ)
            # spatial eccentricity of the energy density, for reference
            r = hypot(xs[k], ys[k])
            if r > 1e-9
                w = U[L.iE, i]*r^2; φ = atan(ys[k], xs[k])
                ec += w*cos(2φ); es += w*sin(2φ); er += w
            end
            Pi = L.hasPi ? H.phys_from_stored(U[L.iPi,i]) : 0.0
            pixx = L.hasShear ? H.phys_from_stored(U[L.iPixx,i]) : 0.0
            piyy = L.hasShear ? H.phys_from_stored(U[L.iPiyy,i]) : 0.0
            pixy = L.hasShear ? H.phys_from_stored(U[L.iPixy,i]) : 0.0
            pxx += (e+P+Pi)*ux*ux + (P+Pi) + pixx
            pyy += (e+P+Pi)*uy*uy + (P+Pi) + piyy
            pxy += (e+P+Pi)*ux*uy + pixy

            T > T_FO || continue
            area += 1.0
            sE += U[L.iE, i]
            su += U[L.iE, i]*hypot(ux, uy)
            sx = U[L.iSx,i]; sy = U[L.iSy,i]; w = hypot(sx, sy)
            w > 0 || continue
            φp = atan(sy, sx)
            sc2 += w*cos(2φp); ss2 += w*sin(2φp)
            sc3 += w*cos(3φp); ss3 += w*sin(3φp); sw += w
        end
        push!(epsp, (pxx+pyy) > 0 ? hypot(pxx-pyy, 2pxy)/(pxx+pyy) : 0.0)
        push!(v2, sw>0 ? hypot(sc2,ss2)/sw : 0.0)
        push!(v3, sw>0 ? hypot(sc3,ss3)/sw : 0.0)
        push!(uT, sE>0 ? su/sE : 0.0)
        push!(Afo, area*dA)
        push!(Sτ, τ*Stot*dA)
        # NOT tau*Ntot: `iDtau` IS tau*J^tau, so an extra factor makes the "drift"
        # come out at exactly tau_f/tau_0 - 1 = 24.0, which is what the first
        # version of this script reported as a charm-conservation failure.
        push!(Nc, Ntot*dA)
        push!(T0, exp(wk.yT[H.lin(g, ng+g.Nx÷2, ng+g.Ny÷2)]))
        push!(eps2, er>0 ? hypot(ec,es)/er : 0.0)
    end

    τ = TAUS[1]
    H.rhs_2d!(wk.k, U, g, τ, m, wk); sample(τ)
    t0 = time()
    for τt in TAUS[2:end]
        r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = τt, CFL = 0.15, CFLτ = 0.05, work = wk)
        @assert r.ok
        τ = r.τ; H.update_primitives_2d!(U, g, τ, m, wk)
        sample(τ)
    end
    kmono = findfirst(k -> Sτ[k+1] < Sτ[k] - 1e-12*maximum(Sτ), 1:length(Sτ)-1)
    @printf("  %-14s %-7s N=%d  %5.1fs | eps_p(8)=%.4f uT(8)=%.4f | dS/S=%+.4f dNc/Nc=%+.2e | S falls first at tau=%s\n",
            tag, ideal ? "ideal" : "viscous", N, time()-t0, epsp[13], uT[13],
            (Sτ[end]-Sτ[1])/Sτ[1], (Nc[end]-Nc[1])/Nc[1],
            kmono === nothing ? "never" : @sprintf("%.1f", TAUS[kmono+1]))
    return (; tag, ideal, v2, v3, uT, Afo, S = Sτ, Nc, T0, eps2, epsp, N)
end

function main()
    mkpath(OUT)
    println("### centrality scan, viscous (eta/s = zeta/s = 0.10)")
    runs = [evolve(t) for t in ("00-05", "20-30", "30-40")]
    println("### single events at 20-30% (fluctuating)")
    evs  = [evolve(t) for t in ("20-30_ev01", "20-30_ev02")]
    println("### ideal controls (entropy must be conserved, not produced)")
    ideals = [evolve(t; ideal = true) for t in ("00-05", "30-40")]
    println("### is the IDEAL entropy loss numerical or outflow? resolution scan")
    idres = [evolve("00-05"; ideal = true, N = n) for n in (150, 225)]

    lbl(r) = r.tag == "00-05" ? "0-5%" : r.tag == "20-30" ? "20-30%" :
             r.tag == "30-40" ? "30-40%" : replace(r.tag, "20-30_ev" => "20-30% ev")
    cols = [:steelblue, :darkorange, :firebrick, :seagreen, :purple]

    # ---- 1. anisotropic flow build-up ----
    p1 = plot(; xlabel = "τ [fm/c]", ylabel = "ε_p  (momentum anisotropy)",
              legend = :bottomright, title = "Elliptic flow builds from spatial eccentricity",
              titlefontsize = 10, size = (700, 470))
    for (k, r) in enumerate(vcat(runs, evs))
        plot!(p1, TAUS, r.epsp; lw = 2, c = cols[k], label = lbl(r),
              ls = occursin("ev", r.tag) ? :dash : :solid)
    end
    plot!(p1, TAUS, ideals[1].epsp; lw = 1.5, c = cols[1], ls = :dot, label = "0-5% ideal")

    # ---- 2. radial flow ----
    p2 = plot(; xlabel = "τ [fm/c]", ylabel = "⟨u_T⟩  (energy-weighted, above T_fo)",
              legend = :bottomright, title = "Radial flow", titlefontsize = 10,
              size = (700, 470))
    for (k, r) in enumerate(runs)
        plot!(p2, TAUS, r.uT; lw = 2, c = cols[k], label = lbl(r))
    end
    for (k, r) in enumerate(ideals)
        plot!(p2, TAUS, r.uT; lw = 1.5, c = cols[k == 1 ? 1 : 3], ls = :dot,
              label = lbl(r)*" ideal")
    end

    # ---- 3. fireball size and lifetime ----
    p3 = plot(; xlabel = "τ [fm/c]", ylabel = "area with T > T_fo  [fm²]",
              legend = :topright, title = "Fireball lifetime (T_fo = 0.1565 GeV)",
              titlefontsize = 10, size = (700, 470))
    for (k, r) in enumerate(runs)
        plot!(p3, TAUS, r.Afo; lw = 2, c = cols[k], label = lbl(r))
    end

    # ---- 4. entropy: the second law, and ideal conservation ----
    p4 = plot(; xlabel = "τ [fm/c]", ylabel = "S(τ) / S(τ₀)",
              legend = :topleft, title = "Entropy — viscous produces, ideal conserves",
              titlefontsize = 10, size = (700, 470))
    for (k, r) in enumerate(runs)
        plot!(p4, TAUS, r.S ./ r.S[1]; lw = 2, c = cols[k], label = lbl(r)*" viscous")
    end
    for (k, r) in enumerate(ideals)
        plot!(p4, TAUS, r.S ./ r.S[1]; lw = 1.5, c = cols[k == 1 ? 1 : 3], ls = :dot,
              label = lbl(r)*" ideal")
    end
    hline!(p4, [1.0]; c = :black, lw = 0.8, ls = :dash, label = "")

    for (nm, f) in (("physics_v2", p1), ("physics_radialflow", p2),
                    ("physics_lifetime", p3), ("physics_entropy", p4))
        savefig(f, joinpath(OUT, nm*".png")); savefig(f, joinpath(OUT, nm*".pdf"))
        println("  wrote ", joinpath(OUT, nm*".png"))
    end

    # ---- the table ----
    println("\n", "="^100)
    @printf("%-16s %8s %9s %9s %9s %9s %9s %10s\n",
            "IC", "eps2(0)", "eps_p(4)", "eps_p(8)", "epsp/eps2", "<u_T>(8)", "A_fo(4)", "dNc/Nc")
    println("="^100)
    for r in vcat(runs, evs, ideals)
        @printf("%-16s %8.4f %9.4f %9.4f %9.3f %9.4f %9.1f %10.2e\n",
                lbl(r)*(r.ideal ? " [ideal]" : ""), r.eps2[1], r.epsp[9], r.epsp[13],
                r.epsp[13]/max(r.eps2[1],1e-9), r.uT[13], r.Afo[9],
                (r.Nc[end]-r.Nc[1])/r.Nc[1])
    end
    println("\nfireball lifetime — last tau at which T(0) > T_fo:")
    for r in vcat(runs, evs)
        k = findlast(>(T_FO), r.T0)
        @printf("  %-16s %s\n", lbl(r),
                k === nothing ? "already below at tau0" :
                k == length(TAUS) ? @sprintf("still above at tau = %.1f (run ended)", TAUS[end]) :
                @sprintf("crosses between tau = %.1f and %.1f fm/c", TAUS[k], TAUS[k+1]))
    end
    println("\nENTROPY — the second law, and what the ideal run does:")
    for r in vcat(runs, ideals)
        kmono = findfirst(k -> r.S[k+1] < r.S[k] - 1e-12*maximum(r.S), 1:length(r.S)-1)
        @printf("  %-16s %-7s dS/S = %+.4f | first decrease at tau = %s\n",
                lbl(r), r.ideal ? "ideal" : "viscous", (r.S[end]-r.S[1])/r.S[1],
                kmono === nothing ? "never" : @sprintf("%.1f", TAUS[kmono+1]))
    end
    println("  ideal 0-5%% entropy loss vs resolution (numerical would SHRINK with N):")
    for r in vcat(idres, [ideals[1]])
        @printf("    N=%3d  dS/S = %+.5f\n", r.N, (r.S[end]-r.S[1])/r.S[1])
    end
end

main()
