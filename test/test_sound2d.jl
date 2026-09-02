# ==============================================================================
# test/test_sound2d.jl — GATE Gs: SOUND SPEED AND ATTENUATION vs THEORY.
#
# `bc2d.jl`'s `:periodic` carries the comment "provided for the sound-wave
# benchmark (gate G3)". This is that benchmark. It is the only quantitative test
# of the DISSIPATIVE coefficients against theory rather than against another
# implementation — and the only test of the BULK sector against theory at all,
# since Gubser is conformal and cannot constrain zeta.
#
# Uniform background at rest, periodic box, a 1e-3 travelling sound wave. The
# complex Fourier amplitude of the k-mode gives c_s from its phase and Gamma from
# its envelope.
#
#   c_s      = sqrt(eos_cs2)
#   Gamma    = (1/2) k^2 [(4/3) eta + zeta] / (e+P)     (Navier-Stokes)
#
# ---------------------------------------------------------------------------
# THINGS THAT MUST BE RIGHT FOR THIS TO MEAN ANYTHING, all learned the hard way:
#
#  * TRAVELLING, not standing. cos(kx) with u = 0 is two counter-propagating
#    modes: the Fourier coefficient oscillates through zero and the phase never
#    advances. That measured c_s = 0.13 against 0.54.
#  * NATURAL UNITS. eta = (eta/s) s [fm^-3], (e+P) [fm^-4] = (e+P)[GeV/fm^3] *
#    invfmGeV. Mixing them inflates the predicted Gamma by invfmGeV^2 = 26.
#    Correctly normalised, tau_pi = 5 (eta/s)/T_nat = 0.06 fm, so omega tau_pi =
#    0.03 and NAVIER-STOKES is the right limit, not the IS cubic.
#  * zeta IS A LORENTZIAN peaked at T = 0.175 GeV with width 0.024 GeV, not
#    (zeta/s) s. At T = 0.35 it is 1.8% of peak, i.e. bulk is effectively off
#    there. The bulk test therefore runs ON the peak.
#  * and the background COOLS during the run, which moves it along that
#    Lorentzian; the prediction uses the run-averaged factor. Using the initial
#    value instead made the ratio drift 1.02 -> 1.40 with T0, which is the drift
#    and not the solver.
#
# The residual ~7% excess is the Bjorken expansion (theta = 1/tau) that the static
# dispersion relation omits; it is the same for shear and for bulk, and constant
# across a factor of 8 in zeta, which is what identifies it as a background effect
# rather than a coefficient error.
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const INVFMGEV = 1/0.1973269804
const ALPHA = -6.0
const TAU0  = 40.0        # large, so theta = 1/tau is small against omega
const AMP   = 1e-3
const LBOX  = 6.0

lorentz(T) = 1/(1 + ((T - 0.175)/0.024)^2)

"""One sound run; returns c_s, Gamma and the run-averaged Lorentzian factor."""
function sound(T0, ηs, ζs; N = 128, τrun = 12.0, dτ = 0.5)
    g = H.make_grid2d(N, N; xmax = LBOX/2, ymax = LBOX/2)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
            enable_shear = (ηs > 0), eta_over_s = ηs, tauShear_coeff = 0.2,
            deltaShear_factor = 4/3,
            enable_bulk = (ζs > 0), zeta_over_s = ζs, tauPi_coeff = 15.0,
            enable_diff = false)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    k = 2π/LBOX
    P0, n0, e0 = H.eos_Pne(T0, ALPHA*T0, m.eos)
    cs2 = H.eos_cs2(T0, ALPHA*T0, m.eos)
    hT = 1e-5*T0
    _, _, ep = H.eos_Pne(T0+hT, ALPHA*(T0+hT), m.eos)
    _, _, em = H.eos_Pne(T0-hT, ALPHA*(T0-hT), m.eos)
    dedT = (ep-em)/(2hT)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        δT = AMP*T0*cos(k*g.xC[ix])
        H.set_cell!(U, H.lin(g,ix,iy), T0 + δT, ALPHA,
                    sqrt(cs2)*(dedT*δT)/(e0+P0), 0.0, TAU0, m)
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0)
    ng = g.nghost; iy0 = ng + g.Ny÷2 + 1
    amp() = begin
        c = 0.0 + 0.0im; Tb = 0.0
        for ix in (ng+1):(ng+g.Nx)
            T = exp(wk.yT[H.lin(g,ix,iy0)]); Tb += T; c += T*cis(-k*g.xC[ix])
        end
        2c/g.Nx, Tb/g.Nx
    end
    τ = TAU0; H.rhs_2d!(wk.k, U, g, τ, m, wk)
    ts = Float64[0.0]; la = Float64[]; ph = Float64[]; lor = Float64[]
    c0, Tb = amp(); push!(la, log(abs(c0)/Tb)); push!(ph, angle(c0)); push!(lor, lorentz(Tb))
    while τ < TAU0 + τrun - 1e-9
        r = H.run_sim_2d!(U, g, m; τ0=τ, τfinal=min(τ+dτ, TAU0+τrun),
                          CFL=0.15, CFLτ=0.05, work=wk, bc=:periodic)
        @assert r.ok; τ = r.τ; H.update_primitives_2d!(U, g, τ, m, wk; bc=:periodic)
        c, Tb = amp(); push!(ts, τ-TAU0); push!(la, log(abs(c)/Tb)); push!(ph, angle(c))
        push!(lor, lorentz(Tb))
    end
    for j in 2:length(ph)
        while ph[j]-ph[j-1] >  π; ph[j] -= 2π; end
        while ph[j]-ph[j-1] < -π; ph[j] += 2π; end
    end
    fit(x,y) = (n=length(x); mx=sum(x)/n; my=sum(y)/n;
                sum((x.-mx).*(y.-my))/sum((x.-mx).^2))
    s0 = (e0+P0-ALPHA*T0*n0)/T0
    return (; Γ = -fit(ts,la), cs = -fit(ts,ph)/k, cs_exact = sqrt(cs2),
              k, eP_nat = (e0+P0)*INVFMGEV, s0, lorbar = sum(lor)/length(lor))
end

@testset "Gs — sound speed and attenuation" begin
    # ---------- the sound speed, ideal, two wavelengths ----------
    # No viscosity involved: this is a pure test of the ideal fluxes and the EOS.
    T0 = 0.35
    base = sound(T0, 0.0, 0.0)      # doubles as the shear baseline below
    @printf("  ideal T=%.2f: c_s measured %.5f vs exact %.5f (%.2f%%)\n",
            T0, base.cs, base.cs_exact, 100*abs(base.cs-base.cs_exact)/base.cs_exact)
    @test abs(base.cs - base.cs_exact)/base.cs_exact < 0.02
    @test abs(base.Γ) < 5e-3        # numerical damping floor must be small

    # ---------- SHEAR: Gamma must be linear in eta/s with the NS coefficient ----------
    rats = Float64[]
    @printf("  shear at T=%.2f (floor Gamma = %.3e):\n", T0, base.Γ)
    for ηs in (0.01, 0.02, 0.04)
        r = sound(T0, ηs, 0.0)
        Γp = 0.5*r.k^2*(4/3)*(ηs*r.s0)/r.eP_nat
        push!(rats, (r.Γ - base.Γ)/Γp)
        @printf("    eta/s=%.3f | Gamma-floor %.5f | NS %.5f | ratio %.3f\n",
                ηs, r.Γ-base.Γ, Γp, rats[end])
    end
    for x in rats; @test 0.85 < x < 1.25; end
    # the OFFSET must be constant: that is what makes it a background effect
    @test (maximum(rats) - minimum(rats))/minimum(rats) < 0.05

    # ---------- BULK: on the Lorentzian peak, and following its shape ----------
    # The only quantitative check the bulk sector has: Gubser is conformal.
    bl = Float64[]
    println("  bulk (zeta/s = 0.05), across the Lorentzian:")
    for T in (0.175, 0.210)
        b = sound(T, 0.0, 0.0)
        r = sound(T, 0.0, 0.05)
        Γp = 0.5*r.k^2*(0.05*r.lorbar*r.s0)/r.eP_nat
        push!(bl, (r.Γ - b.Γ)/Γp)
        @printf("    T=%.3f <lorentz>=%.4f | Gamma-floor %.5f | NS %.5f | ratio %.3f\n",
                T, r.lorbar, r.Γ-b.Γ, Γp, bl[end])
    end
    for x in bl; @test 0.85 < x < 1.25; end
    # the temperature dependence: the Lorentzian must account for the change, so
    # the ratio must be the SAME at both temperatures even though zeta differs 2.3x
    @test abs(bl[1] - bl[2])/bl[1] < 0.08
end
