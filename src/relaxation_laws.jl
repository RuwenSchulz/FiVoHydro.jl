# ==============================================================================
# src/relaxation_laws.jl
#
# EDIT THIS FILE to change the MIS relaxation equations.
#
# The solver calls these hooks from `relax_dissipative!`.
# Keep these functions pure (no allocations) for performance.
# ==============================================================================

abstract type RelaxationLaw end

"""Default MIS-like relaxation law (exact exponential relaxation to Navier–Stokes targets).

This matches the current solver behavior:
- diffusion: ν relaxes to ν_NS = -κ uτ^2 ∂r α with optional δ term
- bulk: Π relaxes to Π_NS = -ζ θ
- shear: evolve independent mixed (πφ, πη), reconstruct πr by tracelessness
"""
struct DefaultRelaxationLaw <: RelaxationLaw end

# ------------------------------------------------------------
# Diffusion
# ------------------------------------------------------------
@inline function relaxation_update_nur_phys(::DefaultRelaxationLaw;
    νold_phys::Float64,
    νNS_phys::Float64,
    adv_src::Float64,
    Δ::Float64,
    τn::Float64,
    δ::Float64,
    θ::Float64,
    uτ::Float64,
)
    # Exact exponential relaxation for: τn uτ dν + ν + δ ν θ = ν_NS (+ transport)
    # Hold (ν_NS, τn, δ, θ, uτ) fixed during the substep.
    τeff = posden(τn * uτ)
    g = 1 + δ * θ
    g = max(g, 1e-12)  # enforce damping even if coefficients get noisy

    νpre = νold_phys + adv_src
    νeq  = νNS_phys / g
    x    = (g * Δ) / τeff
    return νeq + (νpre - νeq) * exp(-x)
end

# ------------------------------------------------------------
# Bulk
# ------------------------------------------------------------
@inline function relaxation_update_Pi_phys(::DefaultRelaxationLaw;
    Πold_phys::Float64,
    ΠNS_phys::Float64,
    adv_src::Float64,
    Δ::Float64,
    τΠ::Float64,
    θ::Float64,
    uτ::Float64,
)
    # Exact exponential relaxation for Eq. (6.8):
    #   τΠ u^μ ∂_μ Π + Π + ζ θ = 0  ⇔  τΠ uτ dΠ + Π = Π_NS, with Π_NS = -ζ θ
    # During this (operator-split) substep, hold (Π_NS, τΠ, uτ) fixed.
    _ = θ  # kept for API symmetry with other hooks
    τeff = posden(τΠ * uτ)

    Πpre = Πold_phys + adv_src
    Πeq  = ΠNS_phys
    x    = Δ / τeff
    return Πeq + (Πpre - Πeq) * exp(-x)
end

# ------------------------------------------------------------
# Shear (mixed diagonal components)
# ------------------------------------------------------------
@inline function relaxation_update_pi_phi_eta_phys(::DefaultRelaxationLaw;
    πφ_old_phys::Float64,
    πη_old_phys::Float64,
    πφNS_phys::Float64,
    πηNS_phys::Float64,
    adv_πφ::Float64,
    adv_πη::Float64,
    Δ::Float64,
    τπ::Float64,
    δπ::Float64,
    θ::Float64,
    uτ::Float64,
)
    # Exact exponential relaxation for: τπ uτ dπ + π + δπ π θ = π_NS (+ transport)
    τeff = posden(τπ * uτ)
    g = 1 + δπ * θ
    g = max(g, 1e-12)
    x = (g * Δ) / τeff
    f = exp(-x)

    πφ_pre = πφ_old_phys + adv_πφ
    πη_pre = πη_old_phys + adv_πη
    πφ_eq  = πφNS_phys / g
    πη_eq  = πηNS_phys / g

    πφ_new = πφ_eq + (πφ_pre - πφ_eq) * f
    πη_new = πη_eq + (πη_pre - πη_eq) * f
    return πφ_new, πη_new
end
