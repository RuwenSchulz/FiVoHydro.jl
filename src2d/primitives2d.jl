# ==============================================================================
# src2d/primitives2d.jl
#
# 2+1D primitive state + model struct. The transport-coefficient models
# (QGPViscosity, SimpleBulkViscosity, LocalThermo, viscosity, τ_shear, τ_bulk)
# are dimension-agnostic and are reused verbatim from src/primitives.jl.
# ==============================================================================

"""Primitive state of one cell, dissipatives in PHYSICAL (not stored) units."""
struct PrimIdealVisc2D
    T::Float64
    mu::Float64
    ux::Float64
    uy::Float64
    n::Float64
    e::Float64
    P::Float64
    nux::Float64
    nuy::Float64
    Pi::Float64
    pixx::Float64
    pixy::Float64
    piyy::Float64
    pieta::Float64
    ok::Bool
end

@inline utau_of(p::PrimIdealVisc2D) = sqrt(1 + p.ux^2 + p.uy^2)

# ------------------------------------------------------------------------------
# Model
#
# Keyword-constructed, unlike the 1-D IdealDiffViscModel's 40 positional
# arguments. The 1-D struct grew that way historically and needs a
# backward-compatible varargs constructor to stay callable; there is no such
# legacy here, so the 2-D model is built with @kwdef and defaults that reproduce
# the BARE scheme (every stabilizer off), matching the 1-D kwarg defaults
# documented in README.md "Stabilizers (bulk solver) — all OFF by default".
# ------------------------------------------------------------------------------

Base.@kwdef struct IdealDiffVisc2DModel{EOS,PR,SH,BU}
    eos::EOS
    layout::StateLayout2D
    primrec::PR

    # ---- charge diffusion ----
    enable_diff::Bool           = false
    diffusion_drive::Symbol     = :alpha      # :alpha | :n
    kappa_coeff::Float64        = 0.0         # D_s·T [GeV·fm]
    tauN_coeff::Float64         = 1.0
    deltaN_factor::Float64      = 0.0
    diff_dt_coeff::Float64      = 0.0
    # TRANSPORT CONVENTION — matches main.jl's production defaults
    # (advect_* = false, relax_advect_* = true, main.jl:540-564). The dissipative
    # dofs are transported by UPWINDING INSIDE the relaxation substep, not as
    # passive scalars through the finite-volume flux. Getting this backwards is
    # not cosmetic: with `advect_nu = true` the charge sector goes unstable on the
    # production IC (x<->y asymmetry 1.0, 11% charge drift) while the 1-D solver
    # on the same IC conserves charge to 5e-16.
    advect_nu::Bool             = false
    relax_advect_nu::Bool       = true
    # Causality bound |nu| <= f * n * u^tau (the LRF bound |nu| <= f*n expressed
    # in the lab frame). OFF by default, matching the 1-D `nur_clip_factor`, so
    # that comparisons against the production solver stay like-for-like.
    #
    # ⚠ MEASURED: on the production IC the outer region reaches |nu|/n = 418 —
    # a diffusion current 400x the charge density, i.e. a real causality
    # violation. Turning the clip on at the physical value f = 1 does NOT fix the
    # open charge-sector defect (TWOD_PROGRAM.md D6), so it is not enabled by
    # default on that basis; it is available and should be revisited once D6 is
    # understood.
    nu_clip_factor::Float64     = -1.0

    # Density-gated vacuum ramp for the charge sector, the same device
    # main2IS2.jl's `_vacuum_weight` applies (FIVO_VACUUM_N_LO / _N_HI), with the
    # Pb+Pb production values. Below `vacuum_n_lo` the current is held at zero;
    # between lo and hi its Navier-Stokes drive is ramped in linearly.
    #
    # WHY the charge sector needs this and shear/bulk do not: their targets are
    # built from gradients of u and T, which stay smooth into the dilute tail.
    # The charge target is built from grad(alpha) with alpha = mu/T, and in the
    # tail the charge row `n u^tau + nu^tau = D` is nearly DEGENERATE — n depends
    # exponentially on phi, so a tiny error in D swings phi wildly, alpha = mu/T
    # with it, and nu_NS = -kappa grad(alpha) feeds straight back into that same
    # row. Measured on the production IC without the ramp: |grad alpha| = 119/fm
    # and |nu|/n = 418 in the tail.
    vacuum_n_lo::Float64        = 1e-6
    vacuum_n_hi::Float64        = 2e-3
    # D9 (2026-09-02): ramp the charge RELAXATION TIME by the same weight that
    # ramps the drive, τ_eff = wv·τ_n. Ramping only the drive was sufficient while
    # τ_n was 6x too short (D8); at the corrected τ_n the current made in the fluid
    # is frozen into the tail instead of decaying, |ν|/n diverges and the charge row
    # goes degenerate. Off reproduces the pre-D9 behaviour exactly, for A/B.
    vacuum_ramp_relax::Bool     = true

    # ---- shear ----
    enable_shear::Bool          = false
    shear::SH                   = ZeroViscosity()
    deltaShear_factor::Float64  = 4/3
    taupi_pi_factor::Float64    = 0.0
    shear_dt_coeff::Float64     = 0.0
    advect_pi::Bool             = false
    relax_advect_pi::Bool       = true
    pi_clip_factor::Float64     = -1.0

    # ---- bulk ----
    enable_bulk::Bool           = false
    bulk::BU                    = ZeroBulkViscosity()
    deltaPi_factor::Float64     = 0.0
    lambda_Pi_pi_factor::Float64 = 0.0
    lambda_pi_Pi_factor::Float64 = 0.0
    bulk_dt_coeff::Float64      = 0.0
    advect_Pi::Bool             = false
    relax_advect_Pi::Bool       = true
    Pi_clip_factor::Float64     = -1.0

    # ---- domain radius ----
    # Evolve only the DISC r <= r_domain; cells outside are held at vacuum every
    # stage. This is what makes the 2-D problem the same problem as the 1-D one:
    # `main.jl` solves on r ∈ [0, rmax], whereas a square box of half-width rmax
    # also contains corners out to sqrt(2)*rmax. Those corners are where the
    # charge sector goes unstable (D6) and they have no 1-D counterpart at all.
    # Inf = evolve the whole box.
    r_domain::Float64           = Inf

    # ---- vacuum cut ----
    # Cells colder than `T_vac_cut` are declared vacuum by `enforce_floors_2d!`.
    # Stated as a TEMPERATURE so it is auditable: the default 0.02 GeV is 8x below
    # the freeze-out temperature (0.1565), where e ~ 2e-7 fm^-4 — ten orders below
    # the fireball centre and far outside anything the lattice-HRG fit is meant to
    # describe. Without it the outer corners of a square box (r > rmax, a region
    # the 1-D radial grid does not have at all) sit at T ~ 16 MeV where e(T) falls
    # five orders per 10 MeV, and the recovery cannot converge there.
    # `E_vac_cut` is derived from it in build_model_2d; set > 0 to override.
    #
    # MEASURED sensitivity on gate G4 (N=200 against the 1-D production run):
    #     T_vac_cut   L2 vs 1-D    residual primfails
    #        0.02      4.670e-4          3760
    #        0.05      4.640e-4           564
    #        0.09      4.702e-4            24
    # The answer moves by 1.3% relative across a 4.5x change in the cut, i.e. it
    # is INSENSITIVE — which is the evidence that the cut only touches cells that
    # do not matter. 0.05 is the operating point: 3x below freeze-out, and it
    # removes 85% of the residual failures at no cost in accuracy.
    T_vac_cut::Float64          = 0.05
    E_vac_cut::Float64          = -1.0

    # Carry the projected-comoving-derivative correction tau_pi*(u^i c^j + u^j c^i)?
    # Diagnostic switch: the term is derived and correct, but it is EXPLICIT and
    # carries a factor tau_pi that is large compared to the timestep, so it is a
    # candidate stiff-explicit instability.
    shear_projected_deriv::Bool = true
    diff_projected_deriv::Bool  = true

    # ---- shear constraint handling ----
    # :project  restore tracelessness by correcting pieta after each relaxation
    #           substep (default; see shear2d.jl / TWOD_PROGRAM.md §2)
    # :monitor  measure the residual but do not correct — for gate work only
    shear_constraint::Symbol    = :project
end

@inline layout2d(m::IdealDiffVisc2DModel) = m.layout
