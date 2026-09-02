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
# ---------------------------------------------------------------------------
# THE MODE MUST BE THE EXACT VISCOUS EIGENMODE, and this is the whole ballgame.
#
# Initialising with the IDEAL eigenvector -- delta u in phase with delta T, and
# pi = 0 -- is wrong at O(Gamma/omega) ~ 1.5%, and that residue excites the
# BACKWARD-propagating mode. |c(tau)| then BEATS, and a straight-line fit to
# log|c| returns a window average rather than Gamma. Measured in sub-windows the
# apparent Gamma swung 1.28 / 0.48 / 0.92 while the full-window fit averaged to
# 1.035.
#
# The trap is that the eigenvector error scales with eta and Gamma scales with eta
# too, so the RATIO is eta-independent: the excess sat at a stubborn 1.0345 for
# eta/s = 0.01, 0.02 and 0.04, and survived scans in resolution, timestep,
# amplitude, tau_pi, tau0 and the estimator. It looked exactly like a coefficient
# error, and for a while I reported it as one -- "the realized eta is 4% above
# (4/3)(eta/s)s". That was WRONG.
#
# What settled it was measuring pi DIRECTLY against its target rather than through
# the dispersion relation: |pi^xx| / |-2 eta sigma^xx| = 0.99978 / 0.99947 /
# 0.99856, with a phase offset of exactly omega*tau_pi. The constitutive relation
# was never off by 4%. With the exact eigenmode (delta u from omega/(w k), delta
# pi from the IS relation) the damping matches theory to 0.2-0.34% and the
# half-window split collapses from 0.09 to 0.002.
#
# Other things that had to be right, learned earlier:
#  * TRAVELLING, not standing: cos(kx) with u = 0 is two counter-propagating
#    modes and the phase never advances. That measured c_s = 0.13 against 0.54.
#  * NATURAL UNITS: eta = (eta/s) s [fm^-3], (e+P)[fm^-4] = (e+P)[GeV/fm^3] *
#    invfmGeV. Mixing them inflates the prediction by invfmGeV^2 = 26.
#  * zeta IS A LORENTZIAN peaked at T = 0.175 GeV, width 0.024 -- at T = 0.35 it
#    is 1.8% of peak, so the bulk test runs ON the peak, and the background cools
#    along it during the run, so the prediction uses the run-averaged factor.
# ==============================================================================

using Printf
using Test
using LinearAlgebra

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

const INVFMGEV = 1/0.1973269804
const ALPHA = -6.0
# tau0 = 320: the Bjorken background contributes an excess that falls like 1/tau0
# (measured 11.07 / 7.15 / 5.05 / 3.96 / 3.41 % at tau0 = 20 / 40 / 80 / 160 /
# 320), so pushing it out is what makes sub-percent reachable at all.
const TAU0  = 320.0
const AMP   = 1e-3
const LBOX  = 6.0

lorentz(T) = 1/(1 + ((T - 0.175)/0.024)^2)

"""Sound root of the IS dispersion (omega^2 - cs^2 k^2)(1 - i w tau) + i w D k^2 = 0."""
function sound_root(k, cs2, D, τR)
    if τR <= 1e-12
        disc = cs2*k^2 - (D*k^2/2)^2
        return sqrt(max(disc, 0.0)) - im*(D*k^2/2)
    end
    c = [(-im*cs2*k^2), (-(τR*cs2*k^2 + D*k^2)), im] ./ τR
    C = [0 0 -c[1]; 1 0 -c[2]; 0 1 -c[3]]
    best = nothing
    for w in eigvals(C)
        real(w) > 1e-9 || continue
        (best === nothing || imag(w) > imag(best)) && (best = w)
    end
    return best
end

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
    # THE EXACT VISCOUS EIGENMODE (see the header). delta u carries the complex
    # phase of omega, and delta pi is the IS response; both vanish smoothly as the
    # viscosity goes to zero, so the ideal run is unaffected.
    s0 = (e0+P0-ALPHA*T0*n0)/T0
    w_nat = (e0+P0)*INVFMGEV
    ηv = ηs*s0
    ζv = ζs*lorentz(T0)*s0
    D  = ((4/3)*ηv + ζv)/w_nat
    τR = ηs > 0 ? 5*ηs/(T0*INVFMGEV) : 0.0
    ω  = sound_root(k, cs2, D, τR)
    δe_c = dedT*AMP*T0*INVFMGEV
    δu_c = ω*δe_c/(w_nat*k)
    δπ_c = ηv > 0 ? -(4/3)*ηv*(im*k*δu_c)/(1 - im*ω*τR) : 0.0+0.0im
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        ph = cis(k*g.xC[ix])
        δT = AMP*T0*real(ph)
        pxx = real(δπ_c*ph)/INVFMGEV
        H.set_cell!(U, H.lin(g,ix,iy), T0 + δT, ALPHA, real(δu_c*ph), 0.0, TAU0, m;
                    pixx = pxx, piyy = -pxx/2, pixy = 0.0)
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
    nn = length(ts)
    Γ1 = -fit(ts[1:nn÷2], la[1:nn÷2]); Γ2 = -fit(ts[nn÷2:nn], la[nn÷2:nn])
    return (; Γ = -fit(ts,la), Γ1, Γ2, cs = -fit(ts,ph)/k, cs_exact = sqrt(cs2),
              k, eP_nat = w_nat, s0, lorbar = sum(lor)/length(lor))
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
    rats = Float64[]; splits = Float64[]
    @printf("  shear at T=%.2f (floor Gamma = %.3e):\n", T0, base.Γ)
    for ηs in (0.01, 0.02, 0.04)
        r = sound(T0, ηs, 0.0)
        Γp = 0.5*r.k^2*(4/3)*(ηs*r.s0)/r.eP_nat
        g = r.Γ - base.Γ
        push!(rats, g/Γp)
        push!(splits, abs((r.Γ1-base.Γ1) - (r.Γ2-base.Γ2))/g)
        @printf("    eta/s=%.3f | Gamma-floor %.6f | NS %.6f | ratio %.5f | half-split %.4f\n",
                ηs, g, Γp, rats[end], splits[end])
    end
    # SUB-PERCENT, now that the mode is the exact eigenmode.
    for x in rats; @test 0.98 < x < 1.02; end
    # a clean exponential: the two half-windows must agree. This is the beat
    # diagnostic -- it read 0.09 with the ideal eigenvector and 0.002 with the
    # exact one, and it is what distinguishes a real coefficient error (which
    # would leave the decay exponential) from mode contamination (which does not).
    @test maximum(splits) < 0.01

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
