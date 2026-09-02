# Gubser flow: the 2-D solver against the exact solution, and the convergence of
# the error. This is the figure behind gate G1.
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/plot_gubser2d.jl
using Printf, Plots
const _ROOT = @__DIR__
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d

const QG=1.0; const TAU0=1.0; const TC0=0.6; const ALPHA=-20.0; const BOX=10.0; const RCMP=3.0
const EOS = H.ConformalHQEOS()
const TSCALE = hydro.gubser_Tscale_from_center_T(TAU0, QG, TC0)
Tana(τ,r) = hydro.gubser_temperature(τ, r, QG, TSCALE)
urana(τ,r) = hydro.gubser_ur(τ, r, QG)
gr(); default(fontfamily="sans-serif", grid=false, framestyle=:box)

function run_to(N, τlist)
    g = H.make_grid2d(N, N; xmax=BOX, ymax=BOX)
    m = H.build_model_2d(; eos=EOS, enable_shear=false, enable_bulk=false, enable_diff=false)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x=g.xC[ix]; y=g.yC[iy]; r=hypot(x,y); ur=urana(TAU0,r)
        H.set_cell!(U, H.lin(g,ix,iy), Tana(TAU0,r), ALPHA,
                    r>0 ? ur*x/r : 0.0, r>0 ? ur*y/r : 0.0, TAU0, m)
    end
    ng=g.nghost; τ=TAU0; out=Dict{Float64,Any}()
    H.rhs_2d!(wk.k, U, g, τ, m, wk)
    grab(τ) = begin
        iy0 = ng + g.Ny÷2 + 1
        xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]
        Ts = [exp(wk.yT[H.lin(g,ix,iy0)]) for ix in (ng+1):(ng+g.Nx)]
        us = [wk.ux[H.lin(g,ix,iy0)] for ix in (ng+1):(ng+g.Nx)]
        # L2 error inside RCMP
        sT=0.0; sTa=0.0
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            r=hypot(g.xC[ix],g.yC[iy]); r<=RCMP || continue
            Tn=exp(wk.yT[H.lin(g,ix,iy)]); Ta=Tana(τ,r)
            sT+=(Tn-Ta)^2; sTa+=Ta^2
        end
        out[τ] = (xs=xs, T=Ts, ux=us, L2=sqrt(sT/sTa))
    end
    grab(τ)
    for τt in τlist
        r = H.run_sim_2d!(U, g, m; τ0=τ, τfinal=τt, CFL=0.15, CFLτ=0.05, work=wk)
        @assert r.ok; τ=r.τ; H.update_primitives_2d!(U, g, τ, m, wk); grab(τ)
    end
    return out
end

function main()
    τlist=[1.5, 2.0, 3.0]
    res = run_to(200, τlist)
    τs = vcat([TAU0], τlist)
    cols = palette(:viridis, length(τs))

    p1 = plot(; xlabel="x [fm]  (slice through y = 0)", ylabel="T [GeV]",
              title="Gubser flow — solver (points) vs exact (lines)", titlefontsize=10,
              legend=:topright, size=(700,470), xlims=(-6,6))
    for (k,τ) in enumerate(τs)
        d = res[τ]
        rr = range(-6, 6; length=400)
        plot!(p1, rr, [Tana(τ, abs(x)) for x in rr]; c=cols[k], lw=2,
              label=@sprintf("τ = %.1f fm/c", τ))
        sel = 1:6:length(d.xs)
        scatter!(p1, d.xs[sel], d.T[sel]; c=cols[k], ms=2.6, msw=0, label="")
    end

    p2 = plot(; xlabel="x [fm]  (slice through y = 0)", ylabel="u^x",
              title="Radial flow — solver (points) vs exact (lines)", titlefontsize=10,
              legend=:topleft, size=(700,470), xlims=(-6,6))
    for (k,τ) in enumerate(τs)
        d = res[τ]
        rr = range(-6, 6; length=400)
        plot!(p2, rr, [sign(x)*urana(τ, abs(x)) for x in rr]; c=cols[k], lw=2,
              label=@sprintf("τ = %.1f fm/c", τ))
        sel = 1:6:length(d.xs)
        scatter!(p2, d.xs[sel], d.ux[sel]; c=cols[k], ms=2.6, msw=0, label="")
    end

    # convergence panel
    Ns = [100, 200, 400]
    L2 = Float64[]
    for N in Ns
        r = run_to(N, [2.0]); push!(L2, r[2.0].L2)
    end
    p3 = plot(Ns, L2; xscale=:log10, yscale=:log10, marker=:circle, lw=2, c=:steelblue,
              xlabel="N (cells per side)", ylabel="relative L2 error in T at τ = 2 fm/c",
              label="measured", legend=:topright, size=(700,470),
              title="Second-order convergence to the exact solution", titlefontsize=10)
    plot!(p3, Ns, L2[1] .* (Ns[1]./Ns).^2; ls=:dash, c=:black, lw=1.5, label="ideal 2nd order")
    for (N,e) in zip(Ns,L2); @printf("  N=%3d  L2 = %.3e\n", N, e); end
    @printf("  measured order: %.2f, %.2f\n", log2(L2[1]/L2[2]), log2(L2[2]/L2[3]))

    OUT = joinpath(_ROOT, "plots2d"); mkpath(OUT)
    for (nm,f) in (("gubser_T",p1), ("gubser_ux",p2), ("gubser_convergence",p3))
        savefig(f, joinpath(OUT, nm*".png")); savefig(f, joinpath(OUT, nm*".pdf"))
        println("  wrote ", joinpath(OUT, nm*".png"))
    end
end
main()
