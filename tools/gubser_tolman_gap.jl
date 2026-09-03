# =================================================================================================
# gubser_tolman_gap.jl — AttractorHydro's gap coefficient, on the minimal transversally expanding
# background where it is EXACTLY computable.
#
# AH's closed form (project_ah_tolman_gap_0831) is
#
#       nu_NS - nu_kin  =  n tau_n ( Delta^{mu nu} grad_nu ln T  +  a^mu )
#
# and the bracket is the TOLMAN-EHRENFEST combination. For ANY ideal fluid at mu_B = 0 the Euler
# equation (e+P) a^mu = -grad^mu P together with grad^mu P = s grad^mu T and e+P = Ts forces
# a^mu = -grad^mu ln T IDENTICALLY, so the bracket vanishes at every gradient order -- and it
# vanishes on any Bjorken flow trivially, which is why the whole boost-invariant attractor
# literature is blind to it. AH measured |grad^r lnT|/a^r = 0.87-0.95 on its Pb+Pb background:
# two channels 5-20x the gap, cancelling to a tenth.
#
# GUBSER IS THE MINIMAL TEST OF THAT STATEMENT. It is an exact, transversally expanding, conformal
# solution, so:
#   * ideal Gubser is an ideal conformal mu_B = 0 fluid  =>  the bracket must be EXACTLY ZERO,
#     not small. That is a null with no tolerance to argue about.
#   * viscous Gubser adds pi^{mu nu} to the Euler equation, so whatever survives IS the viscous
#     force and nothing else -- with eta/s as a dial that turns the gap on continuously.
#
# This is the analytic toy AH does not have. Nothing here runs the solver: T and u are closed form
# and pi comes from the de Sitter ODE that gate G1v already validates three ways.
#
# 🔑 The ideal check uses the CLOSED FORM T_hat = T_hat0/(cosh rho)^(2/3), never the ODE
# interpolant, so "exactly zero" is not limited by a lookup table.
#
# Run: julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/tools/gubser_tolman_gap.jl
# =================================================================================================

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
using .hydro
using Printf

const QG   = 1.0
const TH0  = 0.6
const INVFMGEV = 1/0.1973269804

ρ_of(τ, r) = asinh(-(1 - QG^2*τ^2 + QG^2*r^2)/(2QG*τ))
urana(τ, r) = hydro.gubser_ur(τ, r, QG)

"""Ideal Gubser temperature, CLOSED FORM (no ODE, no interpolation)."""
T_ideal(τ, r) = (TH0 / cosh(ρ_of(τ, r))^(2/3)) / τ

# ------------------------------------------------------------------------------------------------
# The two channels and their sum. Cylindrical Milne (τ, r, φ, η), u = (u^τ, u^r, 0, 0):
#   a^r                    = u^τ ∂_τ u^r + u^r ∂_r u^r        (Γ^r_{ττ}=Γ^r_{τr}=Γ^r_{rr}=0)
#   Δ^{rν} ∂_ν lnT         = ∂_r lnT + u^r (u^τ ∂_τ + u^r ∂_r) lnT
# and the Tolman combination is their SUM.
# ------------------------------------------------------------------------------------------------
function channels(Tfun, τ, r; h = 1e-4)
    lnT(a, b) = log(Tfun(a, b))
    dτ_lnT = (lnT(τ+h, r) - lnT(τ-h, r)) / (2h)
    dr_lnT = (lnT(τ, r+h) - lnT(τ, r-h)) / (2h)
    ur     = urana(τ, r); uτ = sqrt(1 + ur^2)
    dτ_ur  = (urana(τ+h, r) - urana(τ-h, r)) / (2h)
    dr_ur  = (urana(τ, r+h) - urana(τ, r-h)) / (2h)

    a_r    = uτ*dτ_ur + ur*dr_ur
    grad_r = dr_lnT + ur*(uτ*dτ_lnT + ur*dr_lnT)
    return grad_r, a_r, grad_r + a_r
end

# ------------------------------------------------------------------------------------------------
println("="^96)
println("IDEAL GUBSER — the Tolman combination must be EXACTLY zero (ideal conformal, mu_B = 0)")
println("="^96)
@printf("  %6s %6s %14s %14s %14s %12s\n", "tau", "r", "Δ^{rν}∂_ν lnT", "a^r", "SUM", "|sum|/|a^r|")
worst = 0.0
function ideal_scan()
    w = 0.0
    for τ in (1.0, 1.5, 2.5), r in (0.5, 1.5, 3.0)
        g, a, s = channels(T_ideal, τ, r)
        rel = abs(s)/max(abs(a), 1e-300)
        w = max(w, rel)
        @printf("  %6.2f %6.2f %14.6e %14.6e %14.3e %12.2e\n", τ, r, g, a, s, rel)
    end
    return w
end
worst = ideal_scan()
@printf("\n  ⇒ worst |Tolman| / |a^r| over the scan: %.3e\n", worst)
# Is that zero, or just small?  A central difference truncates at O(h^2): if the
# residual falls like h^2 it IS the FD floor and the true value is zero. If it
# plateaus, the combination is genuinely non-zero and small.
println("\n  h-scaling of the residual at (tau,r) = (1.5,1.5) -- O(h^2) proves it is the FD floor:")
@printf("  %10s %16s %10s\n", "h", "|sum|", "ratio")
let prev = 0.0
    for h in (4e-4, 2e-4, 1e-4, 5e-5)
        _, _, s2 = channels(T_ideal, 1.5, 1.5; h = h)
        @printf("  %10.1e %16.6e %10s\n", h, abs(s2),
                prev == 0.0 ? "-" : @sprintf("%.2f", prev/abs(s2)))
        prev = abs(s2)
    end
end
println("  (ratio ~4 per halving = O(h^2) = the combination is EXACTLY zero)\n")

# ------------------------------------------------------------------------------------------------
# Viscous: reuse G1v's ODE. dT_hat/drho = -(2/3)T_hat tanh - (1/3)T_hat pibar tanh
#          dpibar/drho = (4/3)pibar^2 tanh - pibar/tau_pi_hat - (4/15) tanh
# tau_pi_hat = 5 (eta/s)/T_hat_nat, T_hat_nat = T_hat*INVFMGEV.
# ------------------------------------------------------------------------------------------------
function solve_ode(ηs; ρmin = -6.0, ρmax = 3.0, n = 200_000)
    # eta/s = 0  =>  tau_pi = 5(eta/s)/T_hat_nat = 0, i.e. relaxation is INFINITELY
    # FAST and pibar is pinned to its NS value 0 -- not "no relaxation". Writing
    # tau_pi = Inf there leaves the -(4/15)tanh source undamped and integrates up a
    # spurious pibar = 0.45 on a run that is supposed to BE the ideal solution.
    if ηs <= 0
        return (ρ -> TH0 / cosh(ρ)^(2/3)), (ρ -> 0.0)
    end
    ρs = range(ρmin, ρmax; length = n); dρ = step(ρs)
    Th = zeros(n); pb = zeros(n)
    Th[1] = TH0 / cosh(ρmin)^(2/3)
    pb[1] = ηs > 0 ? (4/3)*ηs*tanh(ρmin)/(Th[1]*INVFMGEV) : 0.0
    f(T, p, ρ) = begin
        th = tanh(ρ); τπ = ηs > 0 ? 5*ηs/(T*INVFMGEV) : Inf
        (-(2/3)*T*th - (1/3)*T*p*th,
          (4/3)*p^2*th - (ηs > 0 ? p/τπ : 0.0) - (4/15)*th)
    end
    for i in 1:(n-1)
        ρ = ρs[i]
        k1 = f(Th[i], pb[i], ρ)
        k2 = f(Th[i]+dρ/2*k1[1], pb[i]+dρ/2*k1[2], ρ+dρ/2)
        k3 = f(Th[i]+dρ/2*k2[1], pb[i]+dρ/2*k2[2], ρ+dρ/2)
        k4 = f(Th[i]+dρ*k3[1],   pb[i]+dρ*k3[2],   ρ+dρ)
        Th[i+1] = Th[i] + dρ/6*(k1[1]+2k2[1]+2k3[1]+k4[1])
        pb[i+1] = pb[i] + dρ/6*(k1[2]+2k2[2]+2k3[2]+k4[2])
    end
    itp(v) = ρ -> begin
        x = (ρ - ρmin)/dρ + 1
        i = clamp(floor(Int, x), 1, n-1); f0 = x - i
        v[i]*(1-f0) + v[i+1]*f0
    end
    return itp(Th), itp(pb)
end

println("="^96)
println("VISCOUS GUBSER — whatever survives IS the viscous force, and eta/s dials it")
println("="^96)
@printf("  %8s %6s %6s %14s %14s %14s %10s %10s\n",
        "eta/s", "tau", "r", "Δ^{rν}∂_ν lnT", "a^r", "TOLMAN SUM", "|g|/|a|", "pibar")
function visc_scan()
    # G1v's validated seeding range. At eta/s >= 0.08 the NS seed at rho = -6 is
    # already |pibar| = 1.2, i.e. off the attractor before the first step.
    for ηs in (0.0, 0.002, 0.005, 0.02)
        That, pibar = solve_ode(ηs)
        Tv(τ, r) = That(ρ_of(τ, r))/τ
        for (τ, r) in ((1.5, 1.5), (2.5, 3.0))
            g, a, s = channels(Tv, τ, r; h = 1e-3)
            @printf("  %8.3f %6.2f %6.2f %14.6e %14.6e %14.4e %10.2e %10.4f\n",
                    ηs, τ, r, g, a, s, abs(s)/max(abs(a),1e-300), pibar(ρ_of(τ, r)))
        end
    end
end
visc_scan()

# ------------------------------------------------------------------------------------------------
# WHY the viscosity does not move it -- and this is the real result.
#
# On Gubser the KINEMATICS are viscosity-independent (conformal fixed point), so a^mu is exactly the
# ideal one and the whole viscous effect enters through T. But the viscous correction to T is a
# function of RHO ALONE:  d/drho [ln T_hat_visc - ln T_hat_ideal] = -(1/3) pibar tanh(rho).
# So it can only reach the Tolman combination through the TRANSVERSE gradient of rho -- and the
# fluid is static in de Sitter, i.e. u is along d/drho, so that gradient vanishes identically.
# ------------------------------------------------------------------------------------------------
println("\n" * "="^96)
println("WHY: the transverse gradient of the de Sitter time rho")
println("="^96)
@printf("  %6s %6s %18s %18s %14s\n", "tau", "r", "Δ^{rν}∂_ν rho", "Δ^{rν}∂_ν ln(tau)", "ratio")
function rho_scan(h = 1e-4)
    for τ in (1.0, 1.5, 2.5), r in (0.5, 1.5, 3.0)
        ur = urana(τ, r); uτ = sqrt(1 + ur^2)
        dτρ = (ρ_of(τ+h, r) - ρ_of(τ-h, r))/(2h)
        drρ = (ρ_of(τ, r+h) - ρ_of(τ, r-h))/(2h)
        Δρ  = drρ + ur*(uτ*dτρ + ur*drρ)
        Δlt = ur*uτ*(1/τ)          # Δ^{rν}∂_ν ln(tau), analytic
        @printf("  %6.2f %6.2f %18.4e %18.6f %14.2e\n", τ, r, Δρ, Δlt, abs(Δρ)/max(abs(Δlt),1e-300))
    end
end
rho_scan()
println("""
  ⇒ Δ^{rν}∂_ν rho = 0 to the finite-difference floor. The fluid is STATIC in de Sitter, so rho is
    the proper time along the flow and has no transverse gradient. Every viscous correction on
    Gubser is a function of rho alone, therefore it CANNOT enter the Tolman combination.

  ⭐ GUBSER IS BLIND TO THE AH GAP, at every eta/s -- for the same structural reason Bjorken is.
    Both are symmetry orbits: the flow is generated by a Killing-type direction, the dissipative
    correction depends only on the coordinate along it, and the transverse gradient that the gap
    is built from does not exist. AH's claim that the boost-invariant literature cannot see this
    is therefore STRONGER than stated -- adding transverse expansion is not enough if the
    transverse expansion is itself a symmetry. It needs a background with no such symmetry, i.e.
    a real fireball, which is what AH measures on.

  ⛔ So Gubser is NOT the analytic toy for the AH gap. It is a sharp NULL test: any implementation
    that reports a non-zero gap on Gubser has a bug. That is worth having, and it is what this
    file should be used for.""")
