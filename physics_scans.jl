# ==============================================================================
# physics_scans.jl — three physics scans with the 2+1D solver.
#   julia -t auto --project=Julia Julia/FiVoHydro.jl/physics_scans.jl
#
#   A. CENTRALITY:   eps_p vs centrality over six classes, ideal and viscous.
#   B. VISCOSITY:    eps_p and <u_T> vs eta/s -- the classic QGP result that
#                    shear viscosity suppresses elliptic flow and enhances radial.
#   C. CHARM:        the tracer's spatial spread <r^2> vs D_sT.
#
# Writes plots2d/scan_*.png|pdf.
# ==============================================================================
using Printf, Statistics, Plots
const _ROOT = @__DIR__
include(joinpath(_ROOT, "main2D.jl")); using .hydro2d; const H = hydro2d
const DATA = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation", "PbPb", "data"))
const OUT  = joinpath(_ROOT, "plots2d")
const T_FO = 0.1565
gr(); default(fontfamily="sans-serif", grid=false, framestyle=:box)

"""Run and return eps_p(tau), <u_T>(tau), charm <r^2>(tau), entropy, on `taus`."""
function run1(tag; N=250, box=20.0, ηs=0.10, ζs=0.10, DsT=0.1163, ideal=false,
              taus=[0.4,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
    g = H.make_grid2d(N, N; xmax=box, ymax=box)
    m = ideal ?
        H.build_model_2d(; eos=H.LatticeHRGEOS(), enable_shear=false, enable_bulk=false,
                           enable_diff=true, kappa_coeff=DsT, tauN_coeff=1.0) :
        H.build_model_2d(; eos=H.LatticeHRGEOS(),
            enable_shear=(ηs>0), eta_over_s=ηs, tauShear_coeff=0.2, deltaShear_factor=4/3,
            enable_bulk=(ζs>0), zeta_over_s=ζs, tauPi_coeff=15.0,
            enable_diff=true, kappa_coeff=DsT, tauN_coeff=1.0, pi_clip_factor=1.0)
    L=m.layout; U=H.allocate_state(g,m); wk=H.make_work(g,m)
    H.initialize_from_grid_csv!(U, g, m, taus[1], joinpath(DATA, "ic2d_$(tag).csv"))
    ng=g.nghost; dA=g.dx*g.dy
    idx=[(H.lin(g,ix,iy), g.xC[ix], g.yC[iy]) for ix in (ng+1):(ng+g.Nx) for iy in (ng+1):(ng+g.Ny)]
    epsp=Float64[]; uT=Float64[]; r2=Float64[]; eps2=Float64[]; S=Float64[]
    function sample(τ)
        pxx=0.0;pyy=0.0;pxy=0.0; su=0.0;sE=0.0; sn=0.0;snr=0.0
        ec=0.0;es=0.0;er=0.0; St=0.0
        for (i,x,y) in idx
            T=exp(wk.yT[i]); e=wk.e[i]; P=wk.P[i]; ux=wk.ux[i]; uy=wk.uy[i]
            uτ=sqrt(1+ux*ux+uy*uy)
            Pi = L.hasPi ? H.phys_from_stored(U[L.iPi,i]) : 0.0
            pxx += (e+P+Pi)*ux*ux + (P+Pi) + (L.hasShear ? H.phys_from_stored(U[L.iPixx,i]) : 0.0)
            pyy += (e+P+Pi)*uy*uy + (P+Pi) + (L.hasShear ? H.phys_from_stored(U[L.iPiyy,i]) : 0.0)
            pxy += (e+P+Pi)*ux*uy + (L.hasShear ? H.phys_from_stored(U[L.iPixy,i]) : 0.0)
            T > 1e-3 && (St += (e+P)/T*uτ)
            nq = U[L.iDtau,i]                     # tau*J^tau, the conserved charm
            sn += nq; snr += nq*(x*x+y*y)
            r = hypot(x,y)
            if r > 1e-9
                w=U[L.iE,i]*r^2; φ=atan(y,x); ec+=w*cos(2φ); es+=w*sin(2φ); er+=w
            end
            T > T_FO || continue
            sE += U[L.iE,i]; su += U[L.iE,i]*hypot(ux,uy)
        end
        push!(epsp, (pxx+pyy)>0 ? hypot(pxx-pyy,2pxy)/(pxx+pyy) : 0.0)
        push!(uT, sE>0 ? su/sE : 0.0)
        push!(r2, sn>0 ? snr/sn : 0.0)
        push!(eps2, er>0 ? hypot(ec,es)/er : 0.0)
        push!(S, τ*St*dA)
    end
    τ=taus[1]; H.rhs_2d!(wk.k,U,g,τ,m,wk); sample(τ)
    for τt in taus[2:end]
        r=H.run_sim_2d!(U,g,m; τ0=τ, τfinal=τt, CFL=0.15, CFLτ=0.05, work=wk)
        @assert r.ok; τ=r.τ; H.update_primitives_2d!(U,g,τ,m,wk); sample(τ)
    end
    return (; tag, epsp, uT, r2, eps2, S, taus)
end

const CENTS = ["00-05","05-10","10-20","20-30","30-40","40-50"]
const CMID  = [2.5, 7.5, 15.0, 25.0, 35.0, 45.0]

function main()
    mkpath(OUT)
    println("### A. centrality scan (viscous eta/s=zeta/s=0.10, and ideal)")
    vis = [run1(c) for c in CENTS]
    idl = [run1(c; ideal=true) for c in CENTS]
    for (c,a,b) in zip(CENTS, vis, idl)
        @printf("  %-6s eps2=%.4f | eps_p(8) viscous %.4f  ideal %.4f  (suppression %5.1f%%) | <u_T>(8) %.4f\n",
                c, a.eps2[1], a.epsp[end], b.epsp[end],
                100*(1 - a.epsp[end]/b.epsp[end]), a.uT[end]); flush(stdout)
    end

    println("\n### B. viscosity scan at 20-30%")
    ηlist = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.24]
    vsc = [run1("20-30"; ηs=η, ζs=0.0) for η in ηlist]
    for (η,r) in zip(ηlist, vsc)
        @printf("  eta/s=%.2f | eps_p(8)=%.4f  <u_T>(8)=%.4f  dS/S=%+.4f\n",
                η, r.epsp[end], r.uT[end], (r.S[end]-r.S[1])/r.S[1]); flush(stdout)
    end

    println("\n### C. charm diffusion scan at 20-30% (2piTD_s equivalent in brackets)")
    Dlist = [0.0, 0.0582, 0.1163, 0.2326, 0.4652]
    chm = [run1("20-30"; DsT=D) for D in Dlist]
    for (D,r) in zip(Dlist, chm)
        @printf("  D_sT=%.4f [2piTD_s=%.2f] | <r^2>(0.4)=%.3f -> <r^2>(8)=%.3f fm^2  (growth %+.2f%%)\n",
                D, 2π*D, r.r2[1], r.r2[end], 100*(r.r2[end]/r.r2[1]-1)); flush(stdout)
    end

    # ---- plots ----
    p1 = plot(; xlabel="centrality [%]", ylabel="ε_p at τ = 8 fm/c", legend=:topleft,
              title="Elliptic flow vs centrality", titlefontsize=10, size=(700,470))
    plot!(p1, CMID, [r.epsp[end] for r in vis]; lw=2, marker=:circle, c=:firebrick,
          label="viscous  η/s = ζ/s = 0.10")
    plot!(p1, CMID, [r.epsp[end] for r in idl]; lw=2, marker=:square, c=:steelblue,
          ls=:dash, label="ideal")
    p1b = plot(; xlabel="ε₂ of the initial energy density", ylabel="ε_p at τ = 8 fm/c",
               legend=:topleft, title="Linear response: ε_p ∝ ε₂", titlefontsize=10,
               size=(700,470))
    scatter!(p1b, [r.eps2[1] for r in vis], [r.epsp[end] for r in vis]; c=:firebrick,
             ms=6, label="viscous")
    scatter!(p1b, [r.eps2[1] for r in idl], [r.epsp[end] for r in idl]; c=:steelblue,
             ms=6, m=:square, label="ideal")
    let xs=[r.eps2[1] for r in vis], ys=[r.epsp[end] for r in vis]
        sl = sum(xs.*ys)/sum(xs.^2)
        plot!(p1b, [0, maximum(xs)*1.05], [0, sl*maximum(xs)*1.05]; c=:firebrick, ls=:dash,
              label=@sprintf("slope %.3f", sl))
    end

    p2 = plot(; xlabel="η/s", ylabel="ε_p at τ = 8 fm/c  (20-30%)", legend=:topright,
              title="Shear viscosity suppresses elliptic flow", titlefontsize=10, size=(700,470))
    plot!(p2, ηlist, [r.epsp[end] for r in vsc]; lw=2, marker=:circle, c=:firebrick, label="ε_p")
    p2b = twinx(p2)
    plot!(p2b, ηlist, [r.uT[end] for r in vsc]; lw=2, marker=:diamond, c=:seagreen,
          ls=:dash, ylabel="⟨u_T⟩ at τ = 8", label="⟨u_T⟩", legend=:bottomright)

    p3 = plot(; xlabel="τ [fm/c]", ylabel="⟨r²⟩ of the charm density [fm²]",
              legend=:topleft, title="Charm diffusion broadens the tracer", titlefontsize=10,
              size=(700,470))
    cols = palette(:viridis, length(Dlist))
    for (j,(D,r)) in enumerate(zip(Dlist, chm))
        plot!(p3, r.taus, r.r2; lw=2, c=cols[j], label=@sprintf("D_sT = %.4f", D))
    end

    for (nm,f) in (("scan_centrality",p1), ("scan_response",p1b),
                   ("scan_viscosity",p2), ("scan_charm",p3))
        savefig(f, joinpath(OUT, nm*".png")); savefig(f, joinpath(OUT, nm*".pdf"))
        println("  wrote ", joinpath(OUT, nm*".png"))
    end
end
main()
