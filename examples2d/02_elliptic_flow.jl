#!/usr/bin/env julia
#=
02 — elliptic flow: the reason to run 2+1D at all.

An almond-shaped fireball converts its SPATIAL eccentricity ε₂ into a MOMENTUM anisotropy, because
the pressure gradient is steeper along the short axis. In 1+1D this number does not exist. Here it
is built from an analytic elliptic initial condition and measured as a function of the dissipative
sectors, which is the comparison the coefficients are actually tuned against.

WHAT TO COPY FROM THIS FILE
  * a hand-built (non-axisymmetric) IC, and the `finalize_ic!` that makes it admissible;
  * the energy-weighted momentum anisotropy, restricted to cells ABOVE freeze-out;
  * the sector ladder — ideal / +shear / +shear+bulk — run from ONE initial condition.

⚠ THE REPO'S OWN INITIAL CONDITIONS ARE AZIMUTHALLY SYMMETRIC BY CONSTRUCTION. `data/initial_
profiles_physical.csv` is r,T,α on 1002 radial points because the IC builder φ-averages every binary
collision upstream of FiVo (TWOD_PROGRAM.md §0). Running those in 2+1D is a VALIDATION exercise —
the 2-D answer must reproduce the 1-D one — and produces no elliptic flow at all. Any ε₂ here comes
from the analytic deformation below, not from the data.

⚠ MEASURE THE RESPONSE, NOT THE ANISOTROPY. ε₂ → anisotropy is the transfer function; quoting the
output alone hides whether a change moved the medium or moved the initial condition. The ε₂ = 0
control run is printed first for exactly that reason: it must return identically zero.
=#
ENV["GKSwstype"] = "100"
using Printf, Plots
gr(); default(; fontfamily = "sans-serif", framestyle = :box, grid = false, dpi = 150, lw = 2)

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl")); include(joinpath(_ROOT, "main2D.jl"))
using .hydro; using .hydro2d; const H = hydro2d
const FIG = joinpath(@__DIR__, "figures"); isdir(FIG) || mkpath(FIG)

const IC_CSV = joinpath(_ROOT, "data", "initial_profiles_physical.csv")
const TAU0, TAUF, RMAX = 0.4, 8.0, 20.0
const T_FO = 0.1565            # freeze-out temperature: below it nothing reaches a detector

"""Run one sector setting on an ε₂-deformed production profile.

The deformation is AREA PRESERVING (`a·b = 1`), so the three ε₂ values are the same fireball
squeezed, not three fireballs of different size — otherwise the comparison would be confounded by
the total entropy."""
function run_case(N, eps2; shear, bulk, diff, tauf = TAUF)
    itpT, itpF, _, _ = hydro.load_initial_interpolants(IC_CSV;
        fugacity_kind = :alpha, taper_width = 1.0, interp_kind = :linear)
    g = H.make_grid2d(N, N; xmax = RMAX, ymax = RMAX)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
                           enable_shear = shear, eta_over_s = 0.10, tauShear_coeff = 0.2,
                                                 deltaShear_factor = 4/3,
                           enable_bulk  = bulk,  zeta_over_s = 0.10, tauPi_coeff = 15.0,
                           enable_diff  = diff,  kappa_coeff = 0.1163, tauN_coeff = 1.0)
    U = H.allocate_state(g, m)
    a = sqrt(1 - eps2); b = sqrt(1 + eps2)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix]/a, g.yC[iy]/b)
        H.set_cell!(U, H.lin(g, ix, iy), Float64(itpT(r)), Float64(itpF(r)), 0.0, 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)          # ⚠ never skip this on a hand-built IC

    τs = Float64[]; an = Float64[]
    anisotropy(wk, Uu) = begin
        sx = 0.0; sy = 0.0; ng = g.nghost
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H.lin(g, ix, iy)
            exp(wk.yT[i]) < T_FO && continue     # only the region that produces observables
            w = Uu[m.layout.iE, i]               # energy weighted, as v₂ is
            sx += w*wk.ux[i]^2; sy += w*wk.uy[i]^2
        end
        (sx + sy) > 0 ? (sx - sy)/(sx + sy) : 0.0
    end
    res = H.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = tauf, CFL = 0.15, CFLτ = 0.05, dump_dt = 0.25,
        on_dump = (τ, Uu, wk) -> (push!(τs, τ); push!(an, anisotropy(wk, Uu))))
    @assert res.ok
    (τs, an, res)
end

const N = 150
println("\n  ε₂ = 0 CONTROL — an axisymmetric IC must give identically zero anisotropy:")
_, a0, _ = run_case(N, 0.0; shear = true, bulk = true, diff = true, tauf = 4.0)
@printf("    anisotropy = %+.3e   (%s)\n", a0[end], abs(a0[end]) < 1e-12 ? "OK" : "SUSPECT")

cases = (("ideal",             false, false, true),
         ("+ shear",           true,  false, true),
         ("+ shear + bulk",    true,  true,  true))
plt = plot(size = (760, 470), xlabel = "τ  [fm/c]", ylabel = "momentum anisotropy  (⟨u_x²⟩−⟨u_y²⟩)/(⟨u_x²⟩+⟨u_y²⟩)",
           title = "ε₂ = 0.25, N = $N, energy-weighted above T_fo", legend = :bottomright)
println("\n  ε₂ = 0.25:")
@printf("    %-18s %10s %10s %8s %8s %8s\n", "sector", "anisotropy", "response", "max|u|", "steps", "wall")
for (lab, sh, bu, df) in cases
    t = @elapsed global τs, an, res = run_case(N, 0.25; shear = sh, bulk = bu, diff = df)
    @printf("    %-18s %+10.5f %10.3f %8.3f %8d %7.1fs\n", lab, an[end], an[end]/0.25, res.maxu, res.nsteps, t)
    plot!(plt, τs, an; label = lab)
end
savefig(plt, joinpath(FIG, "ex02_elliptic_flow.png"))
println("\n  -> ", joinpath(FIG, "ex02_elliptic_flow.png"))
println("""
  READING IT
    Shear viscosity REDUCES the anisotropy — it resists the differential expansion that builds it,
    which is the whole reason v₂ constrains η/s. Bulk viscosity reduces it further and more weakly.
    The build-up is steepest in the first ~3 fm/c and then saturates: by the time the medium is
    dilute the pressure gradients that drive it are gone. Nothing here is a v₂: turning this into
    one needs a freeze-out surface and Cooper-Frye (`freezeout2d.jl`), which this file stops short of.""")
