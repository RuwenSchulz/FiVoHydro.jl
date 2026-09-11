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
using DelimitedFiles
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
include(joinpath(_SRC2D, "terms2d.jl"))          # Terms2D — before the model that carries it
include(joinpath(_SRC2D, "primitives2d.jl"))
include(joinpath(_SRC2D, "bessel2d.jl"))       # fast K₂ for the 2-D EOS (before primrec2d)
include(joinpath(_SRC2D, "primrec2d.jl"))
include(joinpath(_SRC2D, "work2d.jl"))
include(joinpath(_SRC2D, "fluxes2d.jl"))
include(joinpath(_SRC2D, "reconstruction2d.jl"))
include(joinpath(_SRC2D, "bc2d.jl"))
include(joinpath(_SRC2D, "rhs2d.jl"))
include(joinpath(_SRC2D, "transport2d.jl"))
include(joinpath(_SRC2D, "hq_consistent_firstmoment2d.jl"))
include(joinpath(_SRC2D, "hq_consistent_m2_2d.jl"))
include(joinpath(_SRC2D, "dissipation2d.jl"))
include(joinpath(_SRC2D, "floors2d.jl"))
include(joinpath(_SRC2D, "timestepper2d.jl"))

export make_grid2d, make_layout2d, build_model_2d, allocate_state,
       initialize_uniform!, run_sim_2d!, LatticeHRGEOS, ConformalHQEOS,
       Terms2D, show_equations, fields_2d

# ------------------------------------------------------------------------------
# Model / state construction
# ------------------------------------------------------------------------------

"""
    build_model_2d(; eos, enable_shear=false, enable_bulk=false, enable_diff=false,
                     terms=Terms2D(), kwargs...)

Assemble the model with a layout matching the enabled sectors. Defaults reproduce
the BARE scheme (ideal fluid, no stabilizers), mirroring main.jl's kwarg defaults.

Every other field of `IdealDiffVisc2DModel` can be passed as a keyword (README2D.md
lists them). `terms` switches individual terms of the charm sector and takes a
`Terms2D` or a `NamedTuple`, e.g. `terms = (fm_inertial = false,)`; see
`src2d/terms2d.jl`. `show_equations(model)` prints what the result integrates.
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
                        with_shear  = enable_shear,
                        with_m2     = get(kwargs, :consistent_m2, false))
    sh = enable_shear ? QGPViscosity(eta_over_s, tauShear_coeff) : ZeroViscosity()
    bu = enable_bulk  ? SimpleBulkViscosity(zeta_over_s, tauPi_coeff) : ZeroBulkViscosity()

    # Derive the vacuum energy cut from its temperature statement, unless given.
    Evac = get(kwargs, :E_vac_cut, -1.0)
    if Evac <= 0.0
        Tvc = get(kwargs, :T_vac_cut, 0.05)
        _, _, evc = eos_Pne(Tvc, 0.0, eos)
        Evac = evc
    end

    m = IdealDiffVisc2DModel(; eos = eos, layout = L, primrec = IdealPrimRec2D(),
                               enable_shear = enable_shear, shear = sh,
                               enable_bulk  = enable_bulk,  bulk  = bu,
                               enable_diff  = enable_diff,
                               E_vac_cut = Evac, r_domain = r_domain,
                               terms = terms2d(get(kwargs, :terms, Terms2D())),
                               filter(p -> !(p.first in (:E_vac_cut, :terms)), kwargs)...)
    reject_unwired_knobs_2d(m)   # see primitives2d.jl — eight 1-D knobs the 2-D solver never reads
    return m
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

"""
    initialize_from_grid_csv!(U, g, model, τ0, csv; taper_width=1.0, smooth_fm=0.0)

Seed from a genuinely 2-D initial condition: a CSV of `x,y,T0,alpha0` on a uniform
grid, as written by `Julia/Projects/ALICE_IC_Creation/BuildIC2D.jl`.

This is the IC that makes the solver worth having. Everything else in the ladder
runs the production profile, which is azimuthally symmetric because the IC builder
φ-averages every binary collision; `BuildIC2D.jl` keeps the collision midpoint
vectors instead, so this file carries real ε₂.

Bilinear interpolation onto the solver grid; outside the file's extent the state
is vacuum, tapered over `taper_width` so the edge is not a step.

`smooth_fm > 0` blurs the FILE's `T` and `α` grids with a Gaussian of that width in
fm before interpolating (separable, normalised, edge-clamped; default 0 = off, and
then this function is byte-for-byte what it was). A real MC-Glauber event carries
sub-nucleon structure — `BuildIC2D.jl` deposits `W = 0.5 fm` sources — that a grid
of dx ≳ 0.2 fm does not resolve, and an unresolved hot spot is what drives the
shear past |π| ~ P and puts the run on the regulators (TWOD_PROGRAM.md §6j; gate G9
sets `pi_clip_factor` for exactly this reason). A mild blur, σ ≈ 0.3-0.6 fm, is the
cheaper half of that fix.

⚠ It smooths T and α AS GIVEN, which is not the same as smoothing the entropy or
energy density the event was built from: T ↦ ⟨T⟩ lowers the peak less than
e ↦ ⟨e⟩ would (e ~ T⁴ here). At these widths it is a regularisation of structure
the grid cannot carry, not a physics model of the initial state — and example 07
measures what it costs in ε₂, ε₃ and the flow response.

Returns `(; nbad, nvacuum)` — cells that had matter and still failed, versus cells
that were legitimately vacuum. Keeping those separate matters: a ±14 fm IC in a
±20 fm box makes ~65% of the grid vacuum.
"""
function initialize_from_grid_csv!(U::AbstractMatrix, g::Grid2D,
                                   model::IdealDiffVisc2DModel, τ0::Float64,
                                   csv::AbstractString; taper_width::Float64 = 1.0,
                                   smooth_fm::Float64 = 0.0)
    raw = readdlm(csv, ','; skipstart = 1)
    xs = sort(unique(Float64.(raw[:,1])))
    ys = sort(unique(Float64.(raw[:,2])))
    nx = length(xs); ny = length(ys)
    @assert size(raw,1) == nx*ny "grid CSV is not a complete uniform grid"
    Tg = zeros(nx, ny); Ag = zeros(nx, ny)
    dx = xs[2]-xs[1]; dy = ys[2]-ys[1]
    for row in 1:size(raw,1)
        ix = round(Int, (Float64(raw[row,1]) - xs[1])/dx) + 1
        iy = round(Int, (Float64(raw[row,2]) - ys[1])/dy) + 1
        Tg[ix,iy] = Float64(raw[row,3]); Ag[ix,iy] = Float64(raw[row,4])
    end

    # optional Gaussian blur of the FILE grid (see the docstring); separable, so two
    # 1-D passes, and the kernel is normalised over the samples that exist, which
    # clamps at the edges rather than pulling vacuum in.
    if smooth_fm > 0.0
        σx = smooth_fm/dx; radx = max(1, ceil(Int, 3σx))
        σy = smooth_fm/dy; rady = max(1, ceil(Int, 3σy))
        wx = [exp(-0.5*(k/σx)^2) for k in -radx:radx]
        wy = [exp(-0.5*(k/σy)^2) for k in -rady:rady]
        for A in (Tg, Ag)
            B = similar(A)
            for j in 1:ny, i in 1:nx                    # x pass, kernel in units of dx
                acc = 0.0; wsum = 0.0
                for (kk, k) in enumerate(-radx:radx)
                    ii = i + k
                    (1 <= ii <= nx) || continue
                    acc += wx[kk]*A[ii,j]; wsum += wx[kk]
                end
                B[i,j] = acc/wsum
            end
            for j in 1:ny, i in 1:nx                    # y pass, kernel in units of dy
                acc = 0.0; wsum = 0.0
                for (kk, k) in enumerate(-rady:rady)
                    jj = j + k
                    (1 <= jj <= ny) || continue
                    acc += wy[kk]*B[i,jj]; wsum += wy[kk]
                end
                A[i,j] = acc/wsum
            end
        end
    end

    xlo, xhi = xs[1], xs[end]; ylo, yhi = ys[1], ys[end]
    rmax_data = min(xhi, yhi)
    tw = max(taper_width, 0.0)
    rt0 = max(rmax_data - tw, 0.0)

    @inline function bilin(A, x, y)
        (x <= xlo || x >= xhi || y <= ylo || y >= yhi) && return 0.0
        i = clamp(Int(floor((x - xlo)/dx)) + 1, 1, nx-1)
        j = clamp(Int(floor((y - ylo)/dy)) + 1, 1, ny-1)
        tx = (x - xs[i])/dx; ty = (y - ys[j])/dy
        return (1-tx)*(1-ty)*A[i,j] + tx*(1-ty)*A[i+1,j] +
               (1-tx)*ty*A[i,j+1]   + tx*ty*A[i+1,j+1]
    end

    # Count only cells that had MATTER and still failed. Cells beyond the file's
    # extent are vacuum by construction — with a ±14 fm IC in a ±20 fm box that is
    # ~65% of the grid, and reporting them as failures is the same vacuum/failure
    # conflation that made an early production run look broken (TWOD_PROGRAM.md §6d).
    nbad = 0; nvac = 0
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        x = g.xC[ix]; y = g.yC[iy]; r = hypot(x, y)
        T = bilin(Tg, x, y); a = bilin(Ag, x, y)
        if r > rt0 && tw > 0
            w = 1.0 - smoothstep01_5((r - rt0)/tw)
            T = max(w*T, T_MIN); a = w*a
        end
        T = max(T, T_MIN)
        ok = set_cell!(U, lin(g, ix, iy), T, a, 0.0, 0.0, τ0, model)
        ok && continue
        T <= 10*T_MIN ? (nvac += 1) : (nbad += 1)
    end
    finalize_ic!(U, g, model; τ0 = τ0)
    return (nbad = nbad, nvacuum = nvac)
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
                     reset_history::Bool = (work === nothing),
                     on_dump = nothing, dump_dt::Float64 = 0.5,
                     max_steps::Int = 2_000_000, verbose::Bool = false)

    reject_unwired_knobs_2d(model)   # direct struct construction bypasses build_model_2d
    wk   = work === nothing ? make_work(g, model) : work
    step = integrator === :ssprk3 ? step_ssprk3_2d! : step_ssprk2_2d!

    # ---- ∂_τ HISTORY ACROSS A RESTART ----------------------------------------
    # This used to reset unconditionally, and that made `run_sim_2d!` NOT
    # RESTART-NEUTRAL: `kinematics_2d` reads `ux_prev`/`uy_prev`/`alpha_prev` for
    # the ∂_τ pieces of the covariant NS drives, so the FIRST relaxation substep
    # of every call dropped them. Chunked drivers therefore ran with a different
    # shear target than a single continuous call.
    #
    # MEASURED on a tau = 0.4 -> 3.0 elliptic run, over cells above T_fo, this is
    # what one restart step throws away:
    #     sigma^{ij}   median 0.5 %   p90 1.9 %   max 3.1 %
    #     Dalpha       median 2 %     p90 10 %
    # (theta is nearly unaffected, 0.2 %, because its ∂_τ piece is u^i∂_τu^i/u^τ.)
    # Harmless when a chunk is many steps; `freezeout2d.jl` chunks at dtau = 0.02
    # against a CFL step of the SAME SIZE, i.e. about one step per call, so the
    # whole freeze-out evolution ran with those terms off.
    #
    # The history now belongs to the work array (Work2D seeds it with NaN), so the
    # default is: reset when we allocated the work ourselves, keep it when the
    # caller handed us one. Every existing caller allocates its `wk` immediately
    # before its own evolution and reuses it for nothing else (verified by grep,
    # 2026-09-08), so this is exactly "one work array, one evolution". Pass
    # `reset_history = true` explicitly to reuse a work array across UNRELATED
    # runs, and `false` to keep it across a join.
    if reset_history
        fill!(wk.x0_yT, NaN); fill!(wk.x0_phi, NaN)
        fill!(wk.x0_ux, NaN); fill!(wk.x0_uy, NaN)
        # NaN marks "no previous step": kinematics_2d then drops the ∂_τ u^i pieces
        # on the first relaxation substep rather than differencing against zero.
        fill!(wk.ux_prev, NaN); fill!(wk.uy_prev, NaN); fill!(wk.alpha_prev, NaN)
        fill!(wk.T_prev, NaN)     # ∂_τT, read by the consistent first/second moments
        fill!(wk.nux_prev, NaN); fill!(wk.nuy_prev, NaN)   # ∂_τν, charm second moment
    end

    # one RHS pass to populate primitives and signal speeds before the first dt
    rhs_2d!(wk.k, U, g, τ0, model, wk; bc = bc)

    # Charge at entry, for the drift diagnostic below.
    Qin = 0.0
    let L = model.layout, ng = g.nghost
        @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            Qin += U[L.iDtau, lin(g, ix, iy)]
        end
    end

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

        # 🔴 2026-09-09 (second pass). The history snapshot belongs HERE, before the
        # state is advanced. Taken after `tau += Delta` (where it sat until now) the
        # work arrays hold primitives recovered at the OLD tau from a state the flux
        # update has ALREADY moved, so `T - T_prev` came out with the WRONG SIGN:
        # measured +1.57e-05 on a cooling medium whose true d_tau T is -0.159. The
        # trace row carries eta_bar*(A/B * DlnT + 5/3 theta) with A/B ~ 7, so a
        # sign-flipped DlnT is a leading-order error, and the traceless rows carry
        # DlnT through `geo` as well.
        # ⚠ `isfinite(...)` and NOT `nsteps > 0`. The history is valid whenever the
        # work arrays carry it, which is the case on a JOIN -- `run_sim_2d!` called
        # again on the same `work` with `reset_history = false`, as a snapshot walk
        # does. Gating on `nsteps > 0` re-skipped the snapshot at EVERY segment
        # boundary, so the answer depended on how many snapshots the caller took:
        # measured, Pi_Q at the centre came out -3.6e-03, +1.5e-02 or exactly 0.0
        # for the same physics at NSNAP = 2, 5, 13. The m2 block then also skipped
        # its first substep after every join (it tests the same history), which is
        # how a harness detail became a physics-looking sign flip.
        if isfinite(wk.alpha_prev[1]) || nsteps > 0
            copyto!(wk.alpha_prev, wk.alpha)
            @inbounds for j in eachindex(wk.T_prev)
                wk.T_prev[j] = exp(wk.yT[j])
            end
        end

        ok, Δused = step(U, g, τ, Δ, model, wk; bc = bc)
        if !ok
            @warn "2-D step failed after dt halving" τ = τ Δ = Δused
            # same NamedTuple shape as the success path, so a caller can read
            # res.maxu / res.dQ without first testing res.ok
            return (ok = false, τ = τ, nsteps = nsteps, nprimfail = nprimfail,
                    nvacuum = nvacuum, max_shear_res = max_shear_res, work = wk,
                    maxu = NaN, dQ = NaN, Q0 = Qin, Q1 = NaN, minPtot = NaN)
        end

        τ += Δused
        nsteps += 1

        # ---- operator-split relaxation, once per accepted step ----
        # Same splitting as the 1-D solver (src/timestepper.jl:139). The primitive
        # refresh first is a deliberate difference: the shear NS target depends on
        # velocity GRADIENTS, so relaxing on the last RK stage's stale primitives
        # would be a needless error inside an already first-order splitting.
        if model.enable_shear || model.enable_bulk || model.enable_diff
            # 🔴 2026-09-09. alpha_prev/T_prev must be captured HERE, from the
            # PREVIOUS step's primitives, not at the end of the relaxation.
            # `relax_dissipative_2d!` only READS work.yT and work.alpha -- it never
            # writes them -- so storing the history there set X_prev to the value the
            # next step then differences against ITSELF: measured, alpha_prev == alpha
            # and T_prev == T EXACTLY (max|diff| = 0.0 over every hot cell), making
            # D alpha and D ln T IDENTICALLY ZERO. That is the failure
            # src/dissipation.jl warns about, reintroduced by the 2-D port in the
            # mirror-image way, and it is invisible to every RHS gate because those
            # take these derivatives as INPUTS. u^i and nu^i are NOT captured here:
            # nu is written by the relaxation itself, so its history belongs at the
            # end of it, where it already is.
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

    # ---- END-OF-RUN DIAGNOSTICS ----
    # `ok` is NOT a statement that the run is physically trustworthy. A
    # single-event IC has satisfied it while |pi|/P ran to 1e12, max|u| reached 58
    # and 60% of the charge left the grid (TWOD_PROGRAM.md §6j) — the run
    # "succeeded" by every criterion the solver had.
    #
    # Part of that WAS the criterion: until 2026-09-08 `state_ok_2d` tested only
    # the invariants the sanitiser had just imposed, so it could not fail and the
    # MOOD/dt-halving escalation never ran. It now inverts the state
    # (`cell_admissible_2d`), which is a real test — but `ok` still says nothing
    # about |pi|/P, max|u| or charge loss, so the diagnostics below stay.
    #
    # So return the two numbers a caller needs to disbelieve it. Both are one pass
    # over the interior and are computed once, at the end. Neither changes the
    # evolution; `ok` keeps its meaning, it just no longer has to carry a job it
    # was never doing.
    # ⚠ `maxu` is over ALL interior cells, INCLUDING the dilute tail, where the
    # fluid legitimately free-streams outward. It is therefore LARGER than the
    # above-freeze-out figure the gates assert on: measured on ev02 at tau = 8,
    # 3.46 here against 1.36 restricted to T > T_fo. Do not compare the two.
    # `minPtot` exists because a negative total pressure is the failure mode that
    # hides: the bulk regulator caps Pi against the pressure AT RELAXATION TIME,
    # and if P then falls ~10% before the next step P + Pi can go negative with
    # nothing reporting it. Measured on 60% multiplicative noise and on 1.3 GeV
    # hot spots, min(P+Pi) reached -4.0e-2 and -2.9e-2 while `ok` stayed true.
    minPtot = Inf
    maxu = 0.0; Qout = 0.0
    let L = model.layout, ng = g.nghost
        @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            Qout += U[L.iDtau, i]
            u = hypot(wk.ux[i], wk.uy[i])
            u > maxu && (maxu = u)
            if wk.P[i] > 0
                pt = wk.P[i] + (L.hasPi ? phys_from_stored(U[L.iPi, i]) : 0.0)
                pt < minPtot && (minPtot = pt)
            end
        end
    end
    dQ = abs(Qin) > 0 ? abs(Qout - Qin)/abs(Qin) : 0.0

    return (ok = true, τ = τ, nsteps = nsteps, nprimfail = nprimfail,
            nvacuum = nvacuum, max_shear_res = max_shear_res, work = wk,
            maxu = maxu, dQ = dQ, Q0 = Qin, Q1 = Qout, minPtot = minPtot)
end

# ------------------------------------------------------------------------------
# Readout
# ------------------------------------------------------------------------------

"""
    fields_2d(g, U, work, model) -> NamedTuple

The state on the interior cells as ordinary `Nx × Ny` matrices indexed `[ix, iy]`
(x first), plus the cell centres `x`, `y`:

    T, mu, alpha, n, e, P, ux, uy          primitives   (GeV, fm⁻³, GeV fm⁻³)
    nux, nuy                               charge current        — if `with_charge`
    Pi                                     bulk                  — if `enable_bulk`
    pixx, pixy, piyy, pieta                shear (pieta = π^η_η) — if `enable_shear`
    pQxx, pQxy, pQyy, pQeta, PiQ           charm second moment   — if `consistent_m2`

Dissipative fields are in PHYSICAL units. `work` is the one the primitives live
in: `res.work` after `run_sim_2d!`, or the third argument of an `on_dump`
callback. The primitives are those of the last recovery, taken before the final
relaxation substep — the same values every diagnostic in the ladder reads.

    res = run_sim_2d!(U, g, m; τ0 = 0.4, τfinal = 2.0)
    f = fields_2d(g, U, res.work, m)
    heatmap(f.x, f.y, permutedims(f.T))      # heatmap wants [iy, ix]
"""
function fields_2d(g::Grid2D, U::AbstractMatrix, work::Work2D, model::IdealDiffVisc2DModel)
    L = model.layout; ng = g.nghost
    ixs = (ng+1):(ng+g.Nx); iys = (ng+1):(ng+g.Ny)
    grab(f) = [f(lin(g, ix, iy)) for ix in ixs, iy in iys]
    prim = (x = g.xC[ixs], y = g.yC[iys],
            T  = grab(i -> exp(work.yT[i])), mu = grab(i -> work.mu[i]),
            alpha = grab(i -> work.alpha[i]), n = grab(i -> work.n[i]),
            e  = grab(i -> work.e[i]),  P  = grab(i -> work.P[i]),
            ux = grab(i -> work.ux[i]), uy = grab(i -> work.uy[i]))
    # every stored dissipative field, by its layout name (the conserved four come first)
    diss = [s => grab(i -> phys_from_stored(U[L.idx[s], i])) for s in L.names[5:end]]
    return merge(prim, NamedTuple(diss))
end

end # module
