# ==============================================================================
# main2D.jl — FiVo 2+1D (transverse Cartesian, boost-invariant Milne).
#
# Sibling of main.jl. Carries bulk + charge:
#   (T, u^x, u^y, Π, π^{xx}, π^{xy}, π^{yy}, π^{ηη}, n, ν^x, ν^y)
#
# main.jl and src/ are NOT touched by this module — the 1-D production path stays
# bit-identical so every published number remains reproducible. Shared, genuinely
# dimension-agnostic files are included from src/; everything that knows about
# dimensionality lives in src2d/.
#
# Design, derivation and the validation ladder: TWOD_PROGRAM.md
#
# Usage:
#   include("Julia/FiVoHydro.jl/main2D.jl"); using .hydro2d
#   g     = hydro2d.make_grid2d(200, 200; xmax=20.0, ymax=20.0)
#   model = hydro2d.build_model_2d(; eos=hydro2d.LatticeHRGEOS())
#   U     = hydro2d.allocate_state(g, model)
#   hydro2d.initialize_uniform!(U, g, model, 0.4; T0=0.5, alpha0=-4.2)
#   hydro2d.run_sim_2d!(U, g, model; τ0=0.4, τfinal=13.0)
# ==============================================================================

module hydro2d

using Printf
using SpecialFunctions
using Base.Threads

const _SRC   = joinpath(@__DIR__, "src")
const _SRC2D = joinpath(@__DIR__, "src2d")

# ---- shared, dimension-agnostic ----
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "primitives.jl"))       # transport-coefficient models
include(joinpath(_SRC, "relaxation_laws.jl"))

# ---- 2-D specific ----
include(joinpath(_SRC2D, "grid2d.jl"))
include(joinpath(_SRC2D, "shear2d.jl"))
include(joinpath(_SRC2D, "state_layout2d.jl"))
include(joinpath(_SRC2D, "primitives2d.jl"))
include(joinpath(_SRC2D, "primrec2d.jl"))
include(joinpath(_SRC2D, "work2d.jl"))
include(joinpath(_SRC2D, "fluxes2d.jl"))
include(joinpath(_SRC2D, "reconstruction2d.jl"))
include(joinpath(_SRC2D, "bc2d.jl"))
include(joinpath(_SRC2D, "rhs2d.jl"))
include(joinpath(_SRC2D, "transport2d.jl"))
include(joinpath(_SRC2D, "dissipation2d.jl"))
include(joinpath(_SRC2D, "floors2d.jl"))
include(joinpath(_SRC2D, "timestepper2d.jl"))

export make_grid2d, make_layout2d, build_model_2d, allocate_state,
       initialize_uniform!, run_sim_2d!, LatticeHRGEOS, ConformalHQEOS

# ------------------------------------------------------------------------------
# Model / state construction
# ------------------------------------------------------------------------------

"""
    build_model_2d(; eos, enable_shear=false, enable_bulk=false, enable_diff=false, kwargs...)

Assemble the model with a layout matching the enabled sectors. Defaults reproduce
the BARE scheme (ideal fluid, no stabilizers), mirroring main.jl's kwarg defaults.
"""
function build_model_2d(; eos = LatticeHRGEOS(),
                          r_domain::Float64 = Inf,
                          enable_shear::Bool = false,
                          enable_bulk::Bool  = false,
                          enable_diff::Bool  = false,
                          with_charge::Bool  = true,
                          eta_over_s::Float64 = 0.1, tauShear_coeff::Float64 = 0.2,
                          zeta_over_s::Float64 = 0.1, tauPi_coeff::Float64 = 15.0,
                          kwargs...)
    L = make_layout2d(; with_charge = with_charge,
                        with_bulk   = enable_bulk,
                        with_shear  = enable_shear)
    sh = enable_shear ? QGPViscosity(eta_over_s, tauShear_coeff) : ZeroViscosity()
    bu = enable_bulk  ? SimpleBulkViscosity(zeta_over_s, tauPi_coeff) : ZeroBulkViscosity()

    # Derive the vacuum energy cut from its temperature statement, unless given.
    Evac = get(kwargs, :E_vac_cut, -1.0)
    if Evac <= 0.0
        Tvc = get(kwargs, :T_vac_cut, 0.05)
        _, _, evc = eos_Pne(Tvc, 0.0, eos)
        Evac = evc
    end

    return IdealDiffVisc2DModel(; eos = eos, layout = L, primrec = IdealPrimRec2D(),
                                  enable_shear = enable_shear, shear = sh,
                                  enable_bulk  = enable_bulk,  bulk  = bu,
                                  enable_diff  = enable_diff,
                                  E_vac_cut = Evac, r_domain = r_domain,
                                  filter(p -> p.first !== :E_vac_cut, kwargs)...)
end

allocate_state(g::Grid2D, model::IdealDiffVisc2DModel) =
    zeros(Float64, nvars(model.layout), g.Ntot)

make_work(g::Grid2D, model::IdealDiffVisc2DModel) =
    Work2D(nvars(model.layout), g.Ntot)

# ------------------------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------------------------

"""
    set_cell!(U, i, T, alpha, ux, uy, τ, model; Pi=0, pixx=0, pixy=0, piyy=0)

Write one cell from primitives. `alpha = μ/T`. `pieta` is not an argument: it is
fixed by tracelessness (`project_shear_traceless_2d`), which is what makes the
stored four-dof shear consistent by construction at t = 0.
"""
function set_cell!(U::AbstractMatrix, i::Int, T::Float64, alpha::Float64,
                   ux::Float64, uy::Float64, τ::Float64,
                   model::IdealDiffVisc2DModel;
                   Pi::Float64 = 0.0, pixx::Float64 = 0.0,
                   pixy::Float64 = 0.0, piyy::Float64 = 0.0,
                   nux::Float64 = 0.0, nuy::Float64 = 0.0)
    L = model.layout
    uτ = sqrt(1 + ux*ux + uy*uy)
    pieta, _ = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, 0.0)
    ok, _ = prim_to_cons_2d!(U, i, T, alpha*T, ux, uy, nux, nuy, Pi,
                             pixx, pixy, piyy, pieta, τ, model.eos, L)
    return ok
end

"""
    finalize_ic!(U, g, model; bc=:outflow)

Floors + admissibility + BCs on a freshly built IC, exactly as the 1-D
`initialize!` ends (`enforce_floors!` then `enforce_Sr_energy_constraint!`).
Required for any IC with a vacuum tail — the production profile tapers T to
`T_MIN`, so its outer cells start below the energy floor.
"""
function finalize_ic!(U::AbstractMatrix, g::Grid2D, model::IdealDiffVisc2DModel;
                      bc::Symbol = :outflow, Emin::Float64 = E_FLOOR, τ0::Float64 = 1.0)
    enforce_floors_2d!(U, g, model; Emin = Emin)
    enforce_S_energy_constraint_2d!(U, g, model; χ = χ_SrE, P = nothing)
    enforce_nu_charge_constraint_2d!(U, g, model, τ0)
    apply_bc_2d!(U, g, model.layout; bc = bc)
    return nothing
end

"""Transversely uniform state — the Bjorken gate (G0)."""
function initialize_uniform!(U::AbstractMatrix, g::Grid2D, model::IdealDiffVisc2DModel,
                             τ0::Float64; T0::Float64 = 0.5, alpha0::Float64 = -4.2)
    nbad = 0
    for i in 1:g.Ntot
        set_cell!(U, i, T0, alpha0, 0.0, 0.0, τ0, model) || (nbad += 1)
    end
    finalize_ic!(U, g, model; τ0 = τ0)
    return nbad
end

"""
    initialize_from_radial!(U, g, model, τ0, Tof, alphaof)

Seed the 2-D grid from radial profile functions `T(r)`, `α(r)`. This is how the
production IC enters: `data/initial_profiles_physical.csv` is `r, T0, alpha0`,
because the IC builder φ-averages every binary collision
(`Julia/Projects/ALICE_IC_Creation/MCGCollisionDensity.jl:34`). The resulting 2-D
state is azimuthally symmetric, so this is the REPRODUCTION gate (G4), not a
physics run — see TWOD_PROGRAM.md §0.
"""
function initialize_from_radial!(U::AbstractMatrix, g::Grid2D, model::IdealDiffVisc2DModel,
                                 τ0::Float64, Tof, alphaof)
    nbad = 0
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        r = hypot(g.xC[ix], g.yC[iy])
        set_cell!(U, lin(g, ix, iy), Tof(r), alphaof(r), 0.0, 0.0, τ0, model) || (nbad += 1)
    end
    finalize_ic!(U, g, model; τ0 = τ0)
    return nbad
end

# ------------------------------------------------------------------------------
# Shear-constraint projection (gate G1)
# ------------------------------------------------------------------------------

"""
    project_shear!(U, g, work, model) -> (max_rel_residual, mean_rel_residual)

Restore tracelessness by correcting `pieta`, and report the residual that was
present. With `model.shear_constraint === :monitor` the residual is measured but
not corrected — for gate work only.
"""
function project_shear!(U::AbstractMatrix, g::Grid2D, work::Work2D,
                        model::IdealDiffVisc2DModel)
    L = model.layout
    L.hasShear || return (0.0, 0.0)
    correct = model.shear_constraint === :project

    maxr = 0.0; sumr = 0.0; cnt = 0
    ng = g.nghost
    @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = lin(g, ix, iy)
        ux = work.ux[i]; uy = work.uy[i]
        uτ = sqrt(1 + ux*ux + uy*uy)
        pixx  = phys_from_stored(U[L.iPixx,i])
        pixy  = phys_from_stored(U[L.iPixy,i])
        piyy  = phys_from_stored(U[L.iPiyy,i])
        pieta = phys_from_stored(U[L.iPieta,i])

        _, rel = shear_constraint_residual_2d(ux, uy, uτ, pixx, pixy, piyy, pieta)
        maxr = max(maxr, rel); sumr += rel; cnt += 1

        if correct
            new, _ = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, pieta)
            U[L.iPieta,i] = stored_from_phys(new)
        end
    end
    return maxr, (cnt > 0 ? sumr/cnt : 0.0)
end

# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------

"""
    run_sim_2d!(U, g, model; τ0, τfinal, CFL, CFLτ, integrator, bc, on_dump, dump_dt)

Advance `U` from `τ0` to `τfinal`. `on_dump(τ, U, work)` is called at most every
`dump_dt` in proper time (and always at the final time). Returns a NamedTuple of
run diagnostics.
"""
function run_sim_2d!(U::AbstractMatrix, g::Grid2D, model::IdealDiffVisc2DModel;
                     τ0::Float64 = 0.4, τfinal::Float64 = 13.0,
                     CFL::Float64 = 0.2, CFLτ::Float64 = 0.05,
                     integrator::Symbol = :ssprk2, bc::Symbol = :outflow,
                     work::Union{Nothing,Work2D} = nothing,
                     on_dump = nothing, dump_dt::Float64 = 0.5,
                     max_steps::Int = 2_000_000, verbose::Bool = false)

    wk   = work === nothing ? make_work(g, model) : work
    step = integrator === :ssprk3 ? step_ssprk3_2d! : step_ssprk2_2d!

    fill!(wk.x0_yT, NaN); fill!(wk.x0_phi, NaN)
    fill!(wk.x0_ux, NaN); fill!(wk.x0_uy, NaN)
    # NaN marks "no previous step": kinematics_2d then drops the ∂_τ u^i pieces on
    # the first relaxation substep rather than differencing against zero.
    fill!(wk.ux_prev, NaN); fill!(wk.uy_prev, NaN); fill!(wk.alpha_prev, NaN)

    # one RHS pass to populate primitives and signal speeds before the first dt
    rhs_2d!(wk.k, U, g, τ0, model, wk; bc = bc)

    τ = τ0
    nsteps = 0
    nprimfail = 0
    nvacuum = 0
    max_shear_res = 0.0
    next_dump = τ0
    on_dump !== nothing && (on_dump(τ, U, wk); next_dump = τ0 + dump_dt)

    while τ < τfinal && nsteps < max_steps
        Δ = compute_dt_2d(wk, g, τ; CFL = CFL, CFLτ = CFLτ)
        Δ = min(Δ, τfinal - τ)
        Δ <= 0 && break

        ok, Δused = step(U, g, τ, Δ, model, wk; bc = bc)
        if !ok
            @warn "2-D step failed after dt halving" τ = τ Δ = Δused
            return (ok = false, τ = τ, nsteps = nsteps, nprimfail = nprimfail,
                    nvacuum = nvacuum, max_shear_res = max_shear_res, work = wk)
        end

        τ += Δused
        nsteps += 1

        # ---- operator-split relaxation, once per accepted step ----
        # Same splitting as the 1-D solver (src/timestepper.jl:139). The primitive
        # refresh first is a deliberate difference: the shear NS target depends on
        # velocity GRADIENTS, so relaxing on the last RK stage's stale primitives
        # would be a needless error inside an already first-order splitting.
        if model.enable_shear || model.enable_bulk || model.enable_diff
            update_primitives_2d!(U, g, τ, model, wk; bc = bc)
            relax_dissipative_2d!(U, g, τ, Δused, model, wk)
        end
        nprimfail += sum(wk.primfail_tls)
        nvacuum   += sum(wk.vacuum_tls)

        if model.layout.hasShear
            r, _ = project_shear!(U, g, wk, model)
            max_shear_res = max(max_shear_res, r)
        end

        if on_dump !== nothing && τ >= next_dump - 1e-12
            on_dump(τ, U, wk)
            next_dump += dump_dt
        end
        verbose && nsteps % 200 == 0 &&
            @printf("  step %6d  τ = %8.4f  Δτ = %.3e  primfail %d\n", nsteps, τ, Δused, nprimfail)
    end

    on_dump !== nothing && on_dump(τ, U, wk)

    return (ok = true, τ = τ, nsteps = nsteps, nprimfail = nprimfail,
            nvacuum = nvacuum, max_shear_res = max_shear_res, work = wk)
end

end # module
