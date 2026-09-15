#!/usr/bin/env julia
#=
03 — the 1+1D solvers against analytically known results, in pictures.

The same referees the validation ladder uses (test/analytic_referees.jl; gates A1, A2, A4, X1 in
test/run1d_gates.jl), plotted instead of asserted:

  (a) ideal Bjorken: the error in T against the step, for both integrators — slopes 2 and 3;
  (b) viscous Bjorken with shear AND bulk AND the λ couplings: the shear-pressure difference
      φ = −π^η_η and the bulk pressure Π against the 0+1D DNMR ODEs;
  (c) viscous Gubser flow: T and π̄ = π^η_η/(e+P) against the semi-analytic solution, at three
      resolutions;
  (d) the charm diffusion mode δn = A J₀(kr), ν^r = B J₁(kr): A(τ) and B(τ) from the bulk solver
      and the IS2 solver against the closed ODE, with the consistent first moment on and off.

Run:  julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples1d/03_analytic_benchmarks.jl
=#
ENV["GKSwstype"] = "100"
using Printf, Plots
using SpecialFunctions: besselj0, besselj1
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
include(joinpath(_ROOT, "main.jl"));     using .hydro;             const H = hydro
include(joinpath(_ROOT, "main2IS2.jl")); using .hydro_current_IS2; const HI = hydro_current_IS2
include(joinpath(_ROOT, "test", "analytic_referees.jl"))
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

uniform(m; T0, α0, τ0, τ1, CFLτ, integ = :ssprk2, dump = Inf, on_dump = nothing) = begin
    g = H.make_grid_1d(40; rmax = 10.0); U = H.allocate_state(g, m)
    H.initialize_uniform!(U, g, m, τ0; T0, alpha0 = α0)
    res = H.run_sim_1d!(U, g, m; τ0, τfinal = τ1, CFL = 10.0, CFLτ, integrator = integ, dump_dt = dump, on_dump)
    (g, U, res)
end

function panel_a()
    m = H.build_model_1d(; eos = H.ConformalHQEOS(m_hq = 0.0, g_hq = 0.0))
    Tex = 0.4*(1/5)^(1/3); cs = [0.08, 0.04, 0.02, 0.01, 0.005]
    p = plot(; xscale = :log10, yscale = :log10, xlabel = "CFLτ (Δτ/τ)", ylabel = "|T/T_exact − 1| at τ = 5",
             title = "(a) ideal Bjorken", legend = :bottomright)
    for (integ, c) in ((:ssprk2, 1), (:ssprk3, 2))
        e = map(cs) do cf
            g, U, res = uniform(m; T0 = 0.4, α0 = 0.0, τ0 = 1.0, τ1 = 5.0, CFLτ = cf, integ)
            f = H.fields_1d(g, U, m; τ = res.τ, work = res.work); abs(f.T[20]/Tex - 1)
        end
        scatter!(p, cs, e; label = string(integ), c)
        plot!(p, cs, e[end] .* (cs ./ cs[end]).^(integ === :ssprk2 ? 2 : 3); c, ls = :dash,
              label = integ === :ssprk2 ? "slope 2" : "slope 3")
    end
    p
end

function panel_b()
    lat = H.LatticeHRGEOS(); α0 = -20.0
    m = H.build_model_1d(; eos = lat, enable_shear = true, eta_over_s = 0.2, enable_bulk = true, zeta_over_s = 0.1,
                           lambda_pi_Pi_factor = 6/5, lambda_Pi_pi_factor = 1.2)
    th(T) = (P = H.eos_Pne(T, α0*T, lat); H.local_thermo(T, α0*T, P[2], P[3], P[1], lat))
    eos_eP(T) = (P = H.eos_Pne(T, α0*T, lat); (P[3], P[1]))
    Tof_e(e) = (lo = 0.01; hi = 2.0; for _ in 1:100; mid = (lo+hi)/2; eos_eP(mid)[1] < e ? (lo = mid) : (hi = mid); end; (lo+hi)/2)
    _, eref, φref, Πref = bjorken_dnmr(; eos_eP, Tof_e, η = T -> H.viscosity(T, th(T), m.shear),
        τπ = T -> H.τ_shear(T, th(T), m.shear), ζ = T -> H.bulk_viscosity(T, th(T), m.bulk),
        τΠ = T -> H.τ_bulk(T, th(T), m.bulk), δππ = 4/3, λπΠ = 6/5, λΠπ = 1.2,
        e0 = eos_eP(0.45)[1], τ0 = 0.6, τ1 = 3.0)
    τs = Float64[]; φs = Float64[]; Πs = Float64[]
    uniform(m; T0 = 0.45, α0, τ0 = 0.6, τ1 = 3.0, CFLτ = 0.005, dump = 0.05,
            on_dump = (τ, U, wk) -> (push!(τs, τ); push!(φs, -H.phys_from_stored(U[m.layout.iPiEta, 20]));
                                     push!(Πs, H.phys_from_stored(U[m.layout.iPi, 20]))))
    tt = range(0.6, 3.0; length = 200)
    p = plot(; xlabel = "τ [fm]", ylabel = "GeV/fm³", title = "(b) viscous Bjorken + λ couplings", legend = :topright)
    plot!(p, tt, φref.(tt); c = 1, label = "φ = −π^η_η, DNMR ODE"); scatter!(p, τs[1:4:end], φs[1:4:end]; c = 1, ms = 3, label = "FiVo")
    plot!(p, tt, Πref.(tt); c = 2, label = "Π, DNMR ODE");          scatter!(p, τs[1:4:end], Πs[1:4:end]; c = 2, ms = 3, label = "FiVo")
    p
end

function panel_c()
    ηs = 0.02; eos = H.ConformalHQEOS(m_hq = 0.0, g_hq = 0.0)
    _, Th, pb = gubser_viscous(ηs; TH0 = 0.6)
    pT = plot(; xlabel = "r [fm]", ylabel = "T [GeV]", title = "(c) viscous Gubser, τ = 2 fm, η/s = 0.02")
    pp = plot(; xlabel = "r [fm]", ylabel = "π̄ = π^η_η/(e+P)", title = "(c′) the shear, same run", legend = :bottomleft)
    rr = range(0, 4; length = 200)
    plot!(pT, rr, [Th(gubser_rho(2.0, r))/2 for r in rr]; c = :black, label = "semi-analytic")
    plot!(pp, rr, [pb(gubser_rho(2.0, r)) for r in rr]; c = :black, label = "semi-analytic")
    # ⚠ READ THE INSET, NOT THE FIRST DOT. Cell 1 sits at r = dr/2, so it is a DIFFERENT PHYSICAL
    # POINT at every resolution — it chases the axis as the grid refines, and no sequence of such
    # points measures convergence. The inset does the honest test at a FIXED radius.
    # 🔑 The first cell used to sit visibly off the curve here, and that WAS a real defect, repaired
    # 2026-09-15 (README §8, EQUATIONS1D §10b): three places treated the cell as if it were AT r = 0
    # — apply_bc! zeroed its conserved S_r, rhs! zeroed its u^r, and the axis face ran first order —
    # while u^r(dr/2) = dr/2 exactly on Gubser. Repairing all three took L2(T) at Nr = 800 from
    # 2.614e-04 to 2.780e-05 and the order in T from 1.77 to 2.12. `FIVO_AXIS_CELL_EXACT=0` puts the
    # old arithmetic back, bit for bit, if you want to see what this panel looked like before.
    conv_r = 0.3; errT = Float64[]; errP = Float64[]; Ns = (100, 200, 400)
    for (k, Nr) in enumerate(Ns)
        g = H.make_grid_1d(Nr; rmax = 10.0)
        m = H.build_model_1d(; eos, enable_shear = true, eta_over_s = ηs)
        U = H.allocate_state(g, m)
        for i in (g.nghost+1):(size(U,2)-g.nghost)
            r = g.rC[i]; ρ = gubser_rho(1.0, r); T = Th(ρ)
            P, _, e = H.eos_Pne(T, 0.0, eos); Πη = pb(ρ)*(e + P)
            H.set_cell!(U, i, T, 0.0, gubser_ur(1.0, r), 1.0, m, g; piR = -Πη/2, piEta = Πη)
        end
        H.finalize_ic!(U, g, m; τ0 = 1.0)
        res = H.run_sim_1d!(U, g, m; τ0 = 1.0, τfinal = 2.0, CFL = 0.15)
        f = H.fields_1d(g, U, m; τ = res.τ, work = res.work)
        pib = f.piEta ./ (f.e .+ f.P)
        sel = f.r .<= 4
        scatter!(pT, f.r[sel][1:max(1, Nr ÷ 50):end], f.T[sel][1:max(1, Nr ÷ 50):end]; ms = 2.5, c = k + 1, label = "FiVo Nr = $Nr")
        plot!(pp, f.r[sel], pib[sel]; c = k + 1, label = "FiVo Nr = $Nr")
        j = findfirst(x -> x >= conv_r, f.r)
        ρj = gubser_rho(res.τ, f.r[j])
        push!(errT, abs(f.T[j]/(Th(ρj)/res.τ) - 1))
        push!(errP, abs(pib[j] - pb(ρj)))
    end
    # the inset says what the eye cannot read off the first cell: at a fixed radius this converges.
    plot!(pT; legend = :topright)
    # ⚠ the inset goes BOTTOM-LEFT: top-right is the legend, and an inset placed there overlaps it
    # and clips its own title (measured 2026-09-15). bbox() is (x, y, w, h) from the TOP-left of the
    # panel, so y = 0.60 puts it in the lower half.
    plot!(pT, collect(Ns), 100 .* errT; inset = (1, bbox(0.13, 0.60, 0.33, 0.30)), subplot = 2,
          xscale = :log10, yscale = :log10, m = :circle, ms = 3, lw = 1.6, c = :black, label = "",
          title = "|ΔT|/T [%] at r = $(conv_r) fm, vs Nr", titlefontsize = 7, xlabel = "", ylabel = "",
          guidefontsize = 6, tickfontsize = 6, framestyle = :box, grid = false,
          xticks = ([100, 200, 400], ["100", "200", "400"]),
          background_color_subplot = RGBA(1, 1, 1, 0.92))
    pT, pp
end

function panel_d()
    eos1 = H.LatticeHRGEOS(canon_factor = 1.0); eosI = HI.LatticeHRGEOS(canon_factor = 1.0)
    T0, α0, τ0, τ1, DsT = 0.35, -2.0, 1.0, 2.5, 0.1163; k = 2.404825557695773/1.5; ε = 1e-3
    _, Tbar, _ = bjorken_background((T, μ) -> H.eos_Pne(T, μ, eos1), T0, α0, τ0, τ1 + 0.1)
    n0 = H.eos_Pne(T0, α0*T0, eos1)[2]
    J0(r) = besselj0(k*r); J1(r) = besselj1(k*r)
    pA = plot(; xlabel = "τ [fm]", ylabel = "τ·A(τ) / (τ₀A₀)", title = "(d) the charm diffusion mode — δn")
    pB = plot(; xlabel = "τ [fm]", ylabel = "B(τ)  [fm⁻³]", title = "(d′) — the current ν^r", legend = :bottomright)
    for (c, cfm, lbl) in ((1, false, "shipped"), (2, true, "consistent fm"))
        _, Ar, Br = diffusion_mode(; k, DsT, m = H.hq_mass(eos1), Tbar, τ0, τ1, A0 = ε*n0,
                                     expansion = cfm, dlnh = cfm)
        tt = range(τ0, τ1; length = 200)
        plot!(pA, tt, tt .* Ar.(tt) ./ (τ0*ε*n0); c, label = "referee, $lbl")
        plot!(pB, tt, Br.(tt); c, label = "referee, $lbl")
        # Each sample is its own pair of solves ending EXACTLY at τ: perturbed and unperturbed
        # runs take slightly different steps (the diffusion step cap depends on n), so their
        # dumps would not land at the same τ — and one step of background evolution is far
        # larger than an ε = 1e-3 mode. (Gate X1 compares at the final time for the same reason.)
        ts = [1.5, 2.0, 2.5]
        function bulk_at(τe, eps)
            g = H.make_grid_1d(300; rmax = 15.0)
            m = H.build_model_1d(; eos = eos1, enable_diff = true, kappa_coeff = DsT, consistent_fm = cfm)
            U = H.allocate_state(g, m)
            H.initialize_from_radial!(U, g, m, τ0, r -> T0, r -> α0 + log1p(eps*J0(r)))
            res = H.run_sim_1d!(U, g, m; τ0, τfinal = τe, CFLτ = 0.01)
            H.fields_1d(g, U, m; τ = res.τ, work = res.work)
        end
        Ab = Float64[]; Bb = Float64[]
        for τe in ts
            fp = bulk_at(τe, ε); f0 = bulk_at(τe, 0.0); sel = fp.r .<= 4
            push!(Ab, project_mode(fp.r[sel], (fp.n .- f0.n)[sel], J0))
            push!(Bb, project_mode(fp.r[sel], (fp.nur .- f0.nur)[sel], J1))
        end
        scatter!(pA, ts, ts .* Ab ./ (τ0*ε*n0); c, ms = 5, label = "bulk 1D, $lbl")
        scatter!(pB, ts, Bb; c, ms = 5, label = "bulk 1D, $lbl")
        # the IS2 solver on the analytic background
        bg = HI.analytic_background(; T = (τ, r) -> Tbar(τ), ur = (τ, r) -> 0.0,
                                      r_grid = collect(0.0:0.05:20.0), t_grid = collect(0.5:0.01:3.0))
        is2_at(τe, eps) = HI.run_static_IS2_test(; background = bg, DsT, τ0, τfinal = τe, Nr = 300, rmax = 15.0,
                                          init_mode = :n_profile, n_profile = r -> n0*(1 + eps*J0(r)),
                                          dump_dt = τe - τ0, eos = eosI, consistent_fm = cfm, log_every = 10^9)
        AI = Float64[]; BI = Float64[]; tI = Float64[]
        for τe in ts
            rp = is2_at(τe, ε); r0 = is2_at(τe, 0.0); rI = rp["r_grid"]; sI = rI .<= 4
            push!(tI, rp["t_grid"][end])
            push!(AI, project_mode(rI[sI], (rp["n"][:, end] .- r0["n"][:, end])[sI], J0))
            push!(BI, project_mode(rI[sI], (rp["nur"][:, end] .- r0["nur"][:, end])[sI], J1))
        end
        scatter!(pA, tI, tI .* AI ./ (τ0*ε*n0); c, m = :diamond, ms = 5, label = "IS2, $lbl")
        scatter!(pB, tI, BI; c, m = :diamond, ms = 5, label = "IS2, $lbl")
    end
    pA, pB
end

function main()
    t0 = time()
    pa = panel_a(); pb = panel_b(); pT, pp = panel_c(); pA, pB = panel_d()
    fig = plot(pa, pb, pT, pp, pA, pB; layout = (3, 2), size = (1200, 1250), left_margin = 6Plots.mm,
               bottom_margin = 4Plots.mm, plot_title = "FiVo 1+1D against analytically known results")
    savefig(fig, joinpath(FIG, "ex03_analytic_benchmarks.png"))
    @printf("wrote %s  (%.0f s)\n", joinpath(FIG, "ex03_analytic_benchmarks.png"), time() - t0)
    return true
end
main() || exit(1)
