# ==============================================================================
# src/shear_tensor.jl
#
# Single source of truth for the *contravariant* shear-stress tensor Π^{μν}
# in Milne+cylindrical coordinates (τ,r,φ,η), for the 1D (r) + boost-invariant
# + azimuthally symmetric setup used by this code.
#
# Edit `shear_tensor_contravariant` to paste your Π^{μν} “in full glory”.
# The rest of the solver (fluxes, sources, primitive recovery) will then use
# the exact same tensor, avoiding inconsistencies.
# ==============================================================================

"""Return the contravariant shear tensor Π^{μν}.

Conventions:
- Coordinates: (τ, r, φ, η)
- Input `ur` is the contravariant 4-velocity component u^r.
- Input `uτ` is u^τ = sqrt(1 + ur^2).
- `r` and `τ` are the coordinate values at the cell center.

The returned NamedTuple must contain these fields:
- `tt`   = Π^{ττ}
- `tr`   = Π^{τr} (equal to Π^{rτ} by symmetry)
- `rr`   = Π^{rr}
- `phph` = Π^{φφ}
- `etaeta` = Π^{ηη}

Default implementation matches the current model variables:
- stored shear dofs are `piR_phys` and `piEta_phys`
- tracelessness in mixed diagonal components is enforced via
  piPhi_phys = -(piR_phys + piEta_phys)
- Π^{φφ} = piPhi_phys / r^2 and Π^{ηη} = piEta_phys / τ^2
- Boosted (τ,r) block is consistent with orthogonality.

Paste/replace the body if you want a different Π^{μν}.
"""
function shear_tensor_contravariant(
    ur::Float64,
    uτ::Float64,
    r::Float64,
    τ::Float64,
    piR_phys::Float64,
    piEta_phys::Float64,
)
    # -----------------------------
    # PASTE YOUR Π^{μν} HERE
    # -----------------------------
    # Example template:
    # Πtt   = ...
    # Πtr   = ...
    # Πrr   = ...
    # Πphph = ...
    # Πetaeta = ...

    piPhi_phys = -(piR_phys + piEta_phys)

    invr2  = safe_inv(r * r)
    invτ2  = safe_inv(τ * τ)

    #Πtt     = (ur*ur)     * piR_phys
    #Πtr     = (uτ*ur)     * piR_phys
    #Πrr     = (uτ*uτ)     * piR_phys
    #Πphph   = piPhi_phys  * invr2
    #Πetaeta = piEta_phys  * invτ2

    s = piPhi_phys + piEta_phys
    ur2 = ur * ur
    uτ2 = uτ * uτ
    Πtt = -ur2 * s
    Πtr = -(ur * uτ) * s
    Πrr = -uτ2 * s
    Πphph = piPhi_phys * invr2
    Πetaeta = piEta_phys * invτ2


    return (tt=Πtt, tr=Πtr, rr=Πrr, phph=Πphph, etaeta=Πetaeta)
end

  """Extract the solver's shear dofs from a contravariant shear tensor Π^{μν}.

  This defines how the code maps a full Π^{μν} back to its stored variables.

  Default mapping (matches current solver conventions):
  - `piEta_phys` is the mixed-diagonal π^η_η, so Π^{ηη} = piEta_phys / τ^2
  - `piPhi_phys` is the mixed-diagonal π^φ_φ, so Π^{φφ} = piPhi_phys / r^2
  - Tracelessness in mixed diagonal components: piR_phys = -(piPhi_phys + piEta_phys)

  If you change `shear_tensor_contravariant`, you should generally also update
  this inverse mapping to remain consistent.
  """
  function shear_dofs_from_contravariant(
    ur::Float64,
    uτ::Float64,
    r::Float64,
    τ::Float64,
    Π::NamedTuple,
  )
    piEta_phys = (τ*τ) * Π.etaeta
    piPhi_phys = (r*r) * Π.phph
    piR_phys   = -(piPhi_phys + piEta_phys)
    return (piR_phys, piEta_phys)
  end

  """Compute the Navier–Stokes shear target as a contravariant tensor Π_NS^{μν}.

  Inputs:
  - `ur`, `uτ`: flow 4-velocity components
  - `r`, `τ`: coordinates
  - `η`: shear viscosity (physical)
  - `θ`: expansion scalar θ = ∇_μ u^μ (your MIS version)
  - `durdr`: radial derivative ∂_r u^r

  This uses the same sign convention as the existing code: π_NS = -2η σ.
  """
  function shear_NS_target_contravariant(
    ur::Float64,
    uτ::Float64,
    r::Float64,
    τ::Float64,
    η::Float64,
    θ::Float64,
    durdr::Float64,
  )
    invτ = safe_inv(τ)

    uτ_over_τ = uτ * invτ
    ur_over_r = θ - uτ_over_τ - durdr

    σr = durdr      - θ/3
    # σφ corresponds to the mixed diagonal φφ component (with the 1/r geometry)
    σφ = ur_over_r  - θ/3
    ση = uτ_over_τ  - θ/3

    piR_NS_phys   = -2η * σr
    piEta_NS_phys = -2η * ση

    return shear_tensor_contravariant(ur, uτ, r, τ, piR_NS_phys, piEta_NS_phys)
  end

