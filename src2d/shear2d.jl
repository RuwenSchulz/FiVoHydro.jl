# ==============================================================================
# src2d/shear2d.jl
#
# Single source of truth for the contravariant shear-stress tensor π^{μν} in
# 2+1D Milne (τ, x, y, η): transverse Cartesian, boost-invariant.
#
# Counterpart of src/shear_tensor.jl, which does the same job for the 1-D radial
# case with the azimuthally symmetric parametrisation (piR, piEta).
#
# ------------------------------------------------------------------------------
# CONVENTIONS
# ------------------------------------------------------------------------------
# Metric g_{μν} = diag(-1, 1, 1, τ²);  u^μ = (u^τ, u^x, u^y, 0);  ∂_η = 0.
# Lowered velocity: u_τ = -u^τ, u_x = u^x, u_y = u^y.
#
# Boost invariance forces π^{τη} = π^{xη} = π^{yη} = 0 (odd under η → -η), so the
# potentially non-zero components are ττ, τx, τy, xx, xy, yy, ηη: seven, cut by
# three orthogonality relations and one trace condition to THREE independent dofs.
# Fluidum's 2+1D state (Matrix/2d_viscous_fugacity.jl) uses the same count with
# the choice (π^{yy}, π^{zz}, π^{xy}).
#
# STORED DOFS (four, one redundant — see TWOD_PROGRAM.md §2 for why):
#     pixx  = π^{xx}
#     pixy  = π^{xy}
#     piyy  = π^{yy}
#     pieta = π^η_η = τ² π^{ηη}     (MIXED component, τ-independent in dimension —
#                                    the same storage convention as the 1-D code's
#                                    piEta_phys, src/shear_tensor.jl)
#
# ORTHOGONALITY (u_μ π^{μν} = 0) is enforced EXACTLY by construction: the τ-row is
# always *derived* from the transverse block, never stored. The ν=η relation is
# satisfied identically because every η-mixed component vanishes.
#
# TRACELESSNESS is the one constraint carried numerically. It is monitored by
# `shear_constraint_residual_2d` and restored by `project_shear_traceless_2d`,
# which corrects `pieta` alone — leaving the transverse block untouched and hence
# preserving x↔y symmetry exactly. That matters: gate G4 measures x↔y asymmetry on
# an azimuthally symmetric IC, so a closure that singled out one transverse axis
# (as eliminating π^{xx} would) could not be used to run that test honestly.
# ==============================================================================

"""
    shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)

Return the full contravariant `π^{μν}` as a NamedTuple with fields
`tt, tx, ty, xx, xy, yy, etaeta`.

The τ-row is derived from orthogonality `u_μ π^{μν} = 0`:

    π^{τx} = (u^x π^{xx} + u^y π^{xy}) / u^τ
    π^{τy} = (u^x π^{xy} + u^y π^{yy}) / u^τ
    π^{ττ} = (u^x π^{τx} + u^y π^{τy}) / u^τ
           = [ (u^x)² π^{xx} + 2 u^x u^y π^{xy} + (u^y)² π^{yy} ] / (u^τ)²

`π^{ηη} = pieta / τ²` follows the stored mixed-component convention.

Note `π^{ττ}` is taken from ORTHOGONALITY, not from the trace. The two agree only
when the traceless constraint holds; their difference is exactly what
`shear_constraint_residual_2d` reports.
"""
@inline function shear_tensor_contravariant_2d(
    ux::Float64, uy::Float64, uτ::Float64, τ::Float64,
    pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
)
    invuτ = safe_inv(uτ)

    Πtx = (ux*pixx + uy*pixy) * invuτ
    Πty = (ux*pixy + uy*piyy) * invuτ
    Πtt = (ux*Πtx + uy*Πty) * invuτ

    Πetaeta = pieta * safe_inv(τ*τ)

    return (tt=Πtt, tx=Πtx, ty=Πty, xx=pixx, xy=pixy, yy=piyy, etaeta=Πetaeta)
end

"""
    shear_constraint_residual_2d(ux, uy, uτ, pixx, pixy, piyy, pieta) -> (absres, relres)

Tracelessness residual `g_{μν} π^{μν} = -π^{ττ} + π^{xx} + π^{yy} + τ²π^{ηη}`,
evaluated with `π^{ττ}` from orthogonality and `τ²π^{ηη} = pieta`:

    R = -π^{ττ}_orth + pixx + piyy + pieta

Returns the absolute residual and the residual scaled by the tensor magnitude.
`relres` should sit at round-off in every gate; a drifting value is an early
warning that nothing else in the scheme provides.
"""
@inline function shear_constraint_residual_2d(
    ux::Float64, uy::Float64, uτ::Float64,
    pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
)
    invuτ = safe_inv(uτ)
    Πtx = (ux*pixx + uy*pixy) * invuτ
    Πty = (ux*pixy + uy*piyy) * invuτ
    Πtt = (ux*Πtx + uy*Πty) * invuτ

    R = -Πtt + pixx + piyy + pieta

    scale = max(abs(Πtt), abs(pixx), abs(piyy), abs(pieta), abs(pixy))
    return R, (scale > TINY ? abs(R)/scale : 0.0)
end

"""
    project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, pieta) -> (pieta_new, R)

Restore tracelessness by correcting `pieta` alone:

    pieta_new = π^{ττ}_orth - pixx - piyy

Returns the corrected value and the size of the correction (= the residual `R`
that was present). The transverse block is untouched, so x↔y symmetry is exact.

`pieta_new` is bounded by `max|π^{ij}|`: `π^{ττ}_orth` is a convex-weighted
combination of the transverse components with weights `(u^i u^j)/(u^τ)² < 1`, so
this projection cannot amplify. Contrast with eliminating `π^{xx}` instead, which
carries a `1/(1 + (u^y)²)` denominator — safe, but x↔y asymmetric.
"""
@inline function project_shear_traceless_2d(
    ux::Float64, uy::Float64, uτ::Float64,
    pixx::Float64, pixy::Float64, piyy::Float64, pieta::Float64,
)
    invuτ = safe_inv(uτ)
    Πtx = (ux*pixx + uy*pixy) * invuτ
    Πty = (ux*pixy + uy*piyy) * invuτ
    Πtt = (ux*Πtx + uy*Πty) * invuτ

    pieta_new = Πtt - pixx - piyy
    return pieta_new, (pieta_new - pieta)
end

"""
    pixx_from_closure_2d(ux, uy, uτ, pixy, piyy, pieta)

The alternative 3-dof closure, eliminating `π^{xx}` from trace + orthogonality:

    π^{xx} = [ 2u^xu^y π^{xy} + (u^y)² π^{yy} - (u^τ)²(π^{yy} + pieta) ] / (1 + (u^y)²)

using `(u^x)² - (u^τ)² = -(1 + (u^y)²)`. This is Fluidum's parametrisation (which
stores yy, zz, xy). Not used by the solver — kept as an INDEPENDENT check: applied
to a constraint-satisfying state it must return the stored `π^{xx}` to round-off.
Exercised by `test_shear2d_algebra.jl`.

No τ argument: `pieta` is already the mixed component τ²π^{ηη}, so the closure is
τ-free.
"""
@inline function pixx_from_closure_2d(
    ux::Float64, uy::Float64, uτ::Float64,
    pixy::Float64, piyy::Float64, pieta::Float64,
)
    num = 2*ux*uy*pixy + (uy*uy)*piyy - (uτ*uτ)*(piyy + pieta)
    den = 1.0 + uy*uy          # ≥ 1, never singular
    return num / den
end

# ------------------------------------------------------------------------------
# Orthogonality of the diffusion current — same structure, one index.
# ------------------------------------------------------------------------------

"""
    nu_tau_2d(ux, uy, uτ, nux, nuy)

`ν^τ` from `u_μ ν^μ = 0`:  `ν^τ = (u^x ν^x + u^y ν^y) / u^τ`.

The 1-D code's `D = n u^τ + v ν^r` (src/primrec.jl:773) is the `u^y = 0` case of
`J^τ = n u^τ + ν^τ`, since there `v = u^r/u^τ`.
"""
@inline nu_tau_2d(ux::Float64, uy::Float64, uτ::Float64, nux::Float64, nuy::Float64) =
    (ux*nux + uy*nuy) * safe_inv(uτ)
