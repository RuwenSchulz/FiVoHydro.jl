# ==============================================================================
# plot_benchmarks2d.jl — the four benchmarks with a known answer, in one place.
#
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/plot_benchmarks2d.jl
#
#   1. BJORKEN            analytic          (tau*s and tau*n conserved)
#   2. GUBSER, ideal      analytic          (Gubser PRD82 085027)
#   3. GUBSER, viscous    semi-analytic     (the de Sitter ODE, gate G1v)
#   4. SOUND              semi-analytic     (the Israel-Stewart dispersion)
#
# Each panel shows the solver against the reference; the last panel collects the
# convergence of all of them. Writes plots2d/bench_*.png|pdf.
# ==============================================================================
using Printf, LinearAlgebra, Plots
const _ROOT = @__DIR__
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d
const OUT = joinpath(_ROOT, "plots2d"); mkpath(OUT)
const INVFMGEV = 1/0.1973269804
gr(); default(fontfamily = "sans-serif", grid = false, framestyle = :box)

# ---------------------------------------------------------------- 1. BJORKEN
s_of_T(T, mu, eos) = ((P, n, e) = H.eos_Pne(T, mu, eos); (e + P)/T)
function bjorken_exact(eos, T0, mu0, τ0, τ)
    P0, n0, e0 = H.eos_Pne(T0, mu0, eos)
    star = (e0 + P0)/T0 * τ0/τ; ntar = n0*τ0/τ
    lo, hi = 1e-4, 5.0
    for _ in 1:200
        mid = 0.5*(lo+hi)
        (s_of_T(mid, mu0, eos) < star) ? (lo = mid) : (hi = mid)
    end
    T = 0.5*(lo+hi)
    lo2, hi2 = -50.0, 50.0
    for _ in 1:200
        mid = 0.5*(lo2+hi2)
        (_, nn, _) = H.eos_Pne(T, mid*T, eos)
        (nn < ntar) ? (lo2 = mid) : (hi2 = mid)
    end
    return T
end

function bench_bjorken(; N = 48, τf = 12.0)
    eos = H.LatticeHRGEOS(); T0 = 0.45; α0 = -4.2
    g = H.make_grid2d(N, N; xmax = 4.0, ymax = 4.0)
    m = H.build_model_2d(; eos = eos, enable_shear = false, enable_bulk = false,
                           enable_diff = false)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_uniform!(U, g, m, 0.4; T0 = T0, alpha0 = α0)
    ng = g.nghost; i0 = H.lin(g, ng+N÷2, ng+N÷2)
    τs = Float64[0.4]; Tn = Float64[]; Ta = Float64[]
    H.rhs_2d!(wk.k, U, g, 0.4, m, wk)
    push!(Tn, exp(wk.yT[i0])); push!(Ta, T0)
    τ = 0.4
    for τt in 0.5:0.25:τf
        r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = τt, CFL = 0.15, CFLτ = 0.05, work = wk)
        @assert r.ok; τ = r.τ; H.update_primitives_2d!(U, g, τ, m, wk)
        push!(τs, τ); push!(Tn, exp(wk.yT[i0]))
        push!(Ta, bjorken_exact(eos, T0, α0*T0, 0.4, τ))
    end
    err = maximum(abs.(Tn .- Ta)./Ta)
    @printf("  Bjorken       : max relative error in T over tau=0.4-%.0f = %.3e\n", τf, err)
    p = plot(τs, Ta; lw = 3, c = :black, ls = :dash, label = "analytic",
             xlabel = "τ [fm/c]", ylabel = "T [GeV]", legend = :topright,
             title = @sprintf("Bjorken — exact (max rel. err %.1e)", err),
             titlefontsize = 10, size = (620, 430))
    scatter!(p, τs[1:2:end], Tn[1:2:end]; c = :firebrick, ms = 4, msw = 0, label = "solver")
    return p, err
end

# ------------------------------------------------------- 2/3. GUBSER, ideal + viscous
const QG = 1.0; const TAU0G = 1.0; const TH0 = 0.6; const ALPHAG = -20.0
const BOXG = 10.0; const RCMP = 3.0
ρ_of(τ, r) = asinh(-(1 - QG^2*τ^2 + QG^2*r^2)/(2QG*τ))
urana(τ, r) = hydro.gubser_ur(τ, r, QG)

function gubser_ode(ηs; ρmin = -6.0, ρmax = 2.0, n = 400_000)
    h = (ρmax-ρmin)/n; ρs = collect(range(ρmin, ρmax; length = n+1))
    Th = zeros(n+1); pb = zeros(n+1)
    f(ρ, T, p) = (th = tanh(ρ); τπ = ηs > 0 ? 5*ηs/(max(T,1e-12)*INVFMGEV) : 0.0;
                  (-(2/3)*T*th + (1/3)*T*p*th,
                   ηs > 0 ? -(4/3)*p*p*th - p/τπ + (4/15)*th : 0.0))
    Th[1] = TH0*cosh(ρmin)^(-2/3)
    pb[1] = ηs > 0 ? (4/3)*ηs*tanh(ρmin)/(Th[1]*INVFMGEV) : 0.0
    for i in 1:n
        T = Th[i]; p = pb[i]; ρ = ρs[i]
        k1 = f(ρ,T,p); k2 = f(ρ+h/2,T+h/2*k1[1],p+h/2*k1[2])
        k3 = f(ρ+h/2,T+h/2*k2[1],p+h/2*k2[2]); k4 = f(ρ+h,T+h*k3[1],p+h*k3[2])
        Th[i+1] = T + h/6*(k1[1]+2k2[1]+2k3[1]+k4[1])
        pb[i+1] = p + h/6*(k1[2]+2k2[2]+2k3[2]+k4[2])
    end
    itp(v) = (lo = ρs[1]; hh = ρs[2]-ρs[1]; nn = length(ρs);
              ρ -> (t = (ρ-lo)/hh; j = clamp(floor(Int,t)+1,1,nn-1); w = t-(j-1);
                    (1-w)*v[j] + w*v[j+1]))
    return itp(Th), itp(pb)
end

function run_gubser(N, ηs, τf, That, pibar)
    g = H.make_grid2d(N, N; xmax = BOXG, ymax = BOXG)
    m = H.build_model_2d(; eos = H.ConformalHQEOS(), enable_shear = (ηs > 0),
            eta_over_s = ηs, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
            enable_bulk = false, enable_diff = false)
    L = m.layout; U = H.allocate_state(g, m); wk = H.make_work(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x,y); ρ = ρ_of(TAU0G, r)
        T = That(ρ)/TAU0G; ur = urana(TAU0G, r); γ = sqrt(1+ur^2)
        P, _, e = H.eos_Pne(T, ALPHAG*T, m.eos); Πη = pibar(ρ)*(e+P)
        if r > 1e-12
            c = x/r; sf = y/r
            pxx = Πη*(-(1/2)*(γ^2*c^2 + sf^2)); pyy = Πη*(-(1/2)*(γ^2*sf^2 + c^2))
            pxy = Πη*(-(1/2)*c*sf*(γ^2-1))
        else
            pxx = -Πη/2; pyy = -Πη/2; pxy = 0.0
        end
        H.set_cell!(U, H.lin(g,ix,iy), T, ALPHAG, r>0 ? ur*x/r : 0.0, r>0 ? ur*y/r : 0.0,
                    TAU0G, m; pixx = ηs>0 ? pxx : 0.0, pixy = ηs>0 ? pxy : 0.0,
                    piyy = ηs>0 ? pyy : 0.0)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0G)
    r = H.run_sim_2d!(U, g, m; τ0 = TAU0G, τfinal = τf, CFL = 0.15, CFLτ = 0.05, work = wk)
    @assert r.ok; τ = r.τ; H.update_primitives_2d!(U, g, τ, m, wk)
    ng = g.nghost; iy0 = ng + g.Ny÷2 + 1
    xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]
    Ts = [exp(wk.yT[H.lin(g,ix,iy0)]) for ix in (ng+1):(ng+g.Nx)]
    # the ideal run has no shear slots at all, so iPieta is 0 there
    pb = L.hasShear ?
        [H.phys_from_stored(U[L.iPieta,H.lin(g,ix,iy0)])/
         (wk.e[H.lin(g,ix,iy0)]+wk.P[H.lin(g,ix,iy0)]) for ix in (ng+1):(ng+g.Nx)] :
        zeros(g.Nx)
    sT = 0.0; sTa = 0.0
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        rr = hypot(g.xC[ix], g.yC[iy]); rr <= RCMP || continue
        Tn = exp(wk.yT[H.lin(g,ix,iy)]); Tref = That(ρ_of(τ,rr))/τ
        sT += (Tn-Tref)^2; sTa += Tref^2
    end
    return xs, Ts, pb, τ, sqrt(sT/sTa)
end

function bench_gubser(ηs, label)
    That, pibar = gubser_ode(ηs)
    τs = [1.0, 1.5, 2.0, 3.0]
    cols = palette(:viridis, length(τs))
    pT = plot(; xlabel = "x [fm]  (y = 0)", ylabel = "T [GeV]", legend = :topright,
              xlims = (-6, 6), size = (620, 430), titlefontsize = 10)
    pP = plot(; xlabel = "x [fm]  (y = 0)", ylabel = "π̄ = π^η_η/(e+P)",
              legend = :topright, xlims = (-6, 6), size = (620, 430), titlefontsize = 10)
    errs = Float64[]
    for (k, τf) in enumerate(τs)
        xs, Ts, pb, τ, L2 = run_gubser(200, ηs, τf, That, pibar)
        push!(errs, L2)
        rr = range(-6, 6; length = 500)
        plot!(pT, rr, [That(ρ_of(τ, abs(x)))/τ for x in rr]; c = cols[k], lw = 2,
              label = @sprintf("τ = %.1f", τ))
        sel = 1:8:length(xs)
        scatter!(pT, xs[sel], Ts[sel]; c = cols[k], ms = 2.6, msw = 0, label = "")
        plot!(pP, rr, [pibar(ρ_of(τ, abs(x))) for x in rr]; c = cols[k], lw = 2,
              label = @sprintf("τ = %.1f", τ))
        scatter!(pP, xs[sel], pb[sel]; c = cols[k], ms = 2.6, msw = 0, label = "")
    end
    # convergence at tau = 2
    Ns = [100, 200, 400]
    L2s = [run_gubser(N, ηs, 2.0, That, pibar)[5] for N in Ns]
    ord = log2(L2s[2]/L2s[3])
    @printf("  %-14s: L2(T) = %s ; order %.2f\n", label,
            join((@sprintf("%.2e", e) for e in L2s), " "), ord)
    title!(pT, @sprintf("%s — lines exact, points solver (order %.2f)", label, ord))
    title!(pP, "$label — shear π̄, lines semi-analytic")
    return pT, pP, Ns, L2s, ord
end

# ---------------------------------------------------------------- 4. SOUND
function sound_root(k, cs2, D, τR)
    τR <= 1e-12 && return (sqrt(max(cs2*k^2-(D*k^2/2)^2,0.0)) - im*(D*k^2/2))
    c = [(-im*cs2*k^2), (-(τR*cs2*k^2+D*k^2)), im] ./ τR
    C = [0 0 -c[1]; 1 0 -c[2]; 0 1 -c[3]]
    best = nothing
    for w in eigvals(C)
        real(w) > 1e-9 || continue
        (best === nothing || imag(w) > imag(best)) && (best = w)
    end
    best
end

function bench_sound(; T0 = 0.35, α = -6.0, L = 6.0, N = 128, τ0 = 320.0,
                       τrun = 12.0, dτ = 0.25, AMP = 1e-4)
    ηlist = [0.0, 0.01, 0.02, 0.04]
    cols = palette(:plasma, length(ηlist))
    p = plot(; xlabel = "τ − τ₀ [fm/c]", ylabel = "ln |δT_k| / |δT_k(0)|",
             legend = :bottomleft, size = (620, 430), titlefontsize = 10,
             title = "Sound — lines: Israel-Stewart dispersion, points: solver (ideal run subtracted)")
    rats = Float64[]; base_ts = Float64[]; base_la = Float64[]
    for (j, ηs) in enumerate(ηlist)
        g = H.make_grid2d(N, N; xmax = L/2, ymax = L/2)
        m = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_shear = (ηs>0),
                eta_over_s = ηs, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
                enable_bulk = false, enable_diff = false)
        U = H.allocate_state(g, m); wk = H.make_work(g, m); k = 2π/L
        P0, n0, e0 = H.eos_Pne(T0, α*T0, m.eos); cs2 = H.eos_cs2(T0, α*T0, m.eos)
        hT = 1e-5*T0
        _,_,ep = H.eos_Pne(T0+hT, α*(T0+hT), m.eos); _,_,em = H.eos_Pne(T0-hT, α*(T0-hT), m.eos)
        dedT = (ep-em)/(2hT); s0 = (e0+P0-α*T0*n0)/T0; w = (e0+P0)*INVFMGEV
        D = (4/3)*(ηs*s0)/w; τR = ηs>0 ? 5*ηs/(T0*INVFMGEV) : 0.0
        ω = sound_root(k, cs2, D, τR)
        δe = dedT*AMP*T0*INVFMGEV; δu = ω*δe/(w*k)
        δπ = ηs>0 ? -(4/3)*(ηs*s0)*(im*k*δu)/(1-im*ω*τR) : 0.0+0.0im
        for ix in 1:g.Nxtot, iy in 1:g.Nytot
            ph = cis(k*g.xC[ix]); pxx = real(δπ*ph)/INVFMGEV
            H.set_cell!(U, H.lin(g,ix,iy), T0+AMP*T0*real(ph), α, real(δu*ph), 0.0, τ0, m;
                        pixx = pxx, piyy = -pxx/2)
        end
        H.finalize_ic!(U, g, m; τ0 = τ0)
        ng = g.nghost; iy0 = ng+g.Ny÷2+1
        amp() = (c = 0.0+0.0im; Tb = 0.0;
                 for ix in (ng+1):(ng+g.Nx)
                     T = exp(wk.yT[H.lin(g,ix,iy0)]); Tb += T; c += T*cis(-k*g.xC[ix])
                 end; (2c/g.Nx, Tb/g.Nx))
        τ = τ0; H.rhs_2d!(wk.k, U, g, τ, m, wk)
        ts = Float64[0.0]; la = Float64[]
        c0, Tb = amp(); a0 = abs(c0)/Tb; push!(la, 0.0)
        while τ < τ0+τrun-1e-9
            r = H.run_sim_2d!(U, g, m; τ0=τ, τfinal=min(τ+dτ,τ0+τrun), CFL=0.15,
                              CFLτ=0.05, work=wk, bc=:periodic)
            @assert r.ok; τ = r.τ; H.update_primitives_2d!(U, g, τ, m, wk; bc=:periodic)
            c, Tb = amp(); push!(ts, τ-τ0); push!(la, log(abs(c)/Tb/a0))
        end
        # Subtract the ideal run, exactly as gate Gs does: the eta/s = 0 curve is
        # not flat (the perturbation grows adiabatically as the background cools),
        # and that offset is not viscous damping.
        if ηs == 0
            base_ts = copy(ts); base_la = copy(la)
        end
        lav = la .- base_la
        sel = 1:4:length(ts)
        scatter!(p, ts[sel], lav[sel]; c = cols[j], ms = 3, msw = 0,
                 label = @sprintf("η/s = %.2f", ηs))
        plot!(p, ts, imag(ω).*ts; c = cols[j], lw = 2, ls = :dash, label = "")
        if ηs > 0
            fit(x,y)=(n=length(x); mx=sum(x)/n; my=sum(y)/n;
                      sum((x.-mx).*(y.-my))/sum((x.-mx).^2))
            push!(rats, (-fit(ts,lav))/(-imag(ω)))
        end
    end
    @printf("  Sound         : Γ_measured/Γ_IS = %s\n",
            join((@sprintf("%.5f", r) for r in rats), " "))
    return p, rats
end

function main()
    println("### 2-D benchmarks with a known answer")
    pB, eB = bench_bjorken()
    pGi, pPi, Ns, L2i, oi = bench_gubser(0.0,  "Gubser ideal")
    pGv, pPv, _,  L2v, ov = bench_gubser(0.02, "Gubser viscous η/s=0.02")
    pS, rats = bench_sound()

    pC = plot(Ns, L2i; xscale=:log10, yscale=:log10, marker=:circle, lw=2, c=:steelblue,
              label = @sprintf("Gubser ideal (order %.2f)", oi),
              xlabel = "N (cells per side)", ylabel = "relative L2 error in T at τ = 2",
              legend = :topright, size = (620, 430), titlefontsize = 10,
              title = "Convergence to the exact solutions")
    plot!(pC, Ns, L2v; marker=:square, lw=2, c=:firebrick,
          label = @sprintf("Gubser viscous (order %.2f)", ov))
    plot!(pC, Ns, L2i[1].*(Ns[1]./Ns).^2; ls=:dash, c=:black, lw=1.5, label="2nd order")

    for (nm, f) in (("bench_bjorken", pB), ("bench_gubser_ideal_T", pGi),
                    ("bench_gubser_viscous_T", pGv), ("bench_gubser_viscous_pi", pPv),
                    ("bench_sound", pS), ("bench_convergence", pC))
        savefig(f, joinpath(OUT, nm*".png")); savefig(f, joinpath(OUT, nm*".pdf"))
        println("  wrote ", joinpath(OUT, nm*".png"))
    end
end
main()
