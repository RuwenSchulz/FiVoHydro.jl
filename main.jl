#!/usr/bin/env julia
# ==============================================================================
# hydro_ideal_axisface_trustworthy_stable_phi_diff_visc_clean.jl
#
# Cleaned + fixes applied:
#   - Introduced TINY (=1e-300) and removed all Float64-underflow "1e-500" guards
#   - HLLE / wavespeeds / fits now use TINY-guarded denominators
#   - diff_tauN: K2 protected against near-zero to avoid Inf ratios
#   - compute_dt_from_work now honors CFL, CFLτ, diff_dt_coeff and is called with run-time values
#   - Added prime_work_from_U!() to initialize work caches before the first dt-from-work call
#   - (Recommended physics) ν_NS uses the conventional sign: ν_NS = -κ uτ^2 ∂r α
#
# NOTE ON SIGN CONVENTIONS (IMPORTANT):
#   stored = DISS_SIGN * physical   (DISS_SIGN = ±1)
# ==============================================================================
module hydro

using LinearAlgebra
using Printf
using CSV
using Logging
using Base.Threads
using Interpolations
using SpecialFunctions
using Tables

# ------------------------------------------------------------
# Load split files
# ------------------------------------------------------------
const _SRC = joinpath(@__DIR__, "src")
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "grid.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "logging_setup.jl"))
include(joinpath(_SRC, "primitives.jl"))
include(joinpath(_SRC, "state_layout.jl"))
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "gubser.jl"))
include(joinpath(_SRC, "runtime_flags.jl"))
include(joinpath(_SRC, "diagnostics.jl"))
include(joinpath(_SRC, "floors.jl"))
include(joinpath(_SRC, "primrec.jl"))
include(joinpath(_SRC, "work.jl"))
include(joinpath(_SRC, "ic_diagnostics.jl"))
include(joinpath(_SRC, "shear_tensor.jl"))
include(joinpath(_SRC, "relaxation_laws.jl"))
include(joinpath(_SRC, "dissipation.jl"))
include(joinpath(_SRC, "io.jl"))
include(joinpath(_SRC, "mood.jl"))




# ------------------------------------------------------------
# Debugging + failure dumps
# ------------------------------------------------------------
include(joinpath(_SRC, "debugging.jl"))

# ------------------------------------------------------------

# ------------------------------------------------------------
# Core numerics (factored out of main)
# ------------------------------------------------------------
include(joinpath(_SRC, "boundary_conditions.jl"))
include(joinpath(_SRC, "fluxes.jl"))
include(joinpath(_SRC, "reconstruction.jl"))
include(joinpath(_SRC, "rhs.jl"))
include(joinpath(_SRC, "timestepper.jl"))



# ------------------------------------------------------------
# dt (CFL + diffusion cap for charge diffusion only)
# ------------------------------------------------------------

function compute_dt_from_work(work::Work1D, grid, τ, model; CFL::Float64=0.2, CFLτ::Float64=0.05)
    ng = grid.nghost
    amax = 1e-30
    kmax = 0.0

    τπ_min = Inf
    τΠ_min = Inf

    @inbounds for i in (ng+1):(length(work.yT)-ng)
        work.ok[i] || continue

        T  = exp(work.yT[i])
        μ  = work.mu[i]
        ur = sinh(work.y[i])

        λm, λp = wavespeeds_from_prim(T, μ, ur, model.eos)
        a = max(abs(λm), abs(λp))

        if model.enable_diff
            uτ = sqrt(1 + ur^2)
            if model.diffusion_drive === :alpha
                κ  = diff_kappa(T, μ, work.n[i], model)
                kmax = max(kmax, κ * (uτ^2))
            elseif model.diffusion_drive === :n
                DsT = model.kappa_coeff
                D = safe_div(DsT, T) / fmGeV
                kmax = max(kmax, D * (uτ^2))
            else
                throw(ArgumentError("Unknown diffusion_drive=$(model.diffusion_drive). Use :alpha or :n"))
            end
            a = max(a, 0.999999)
        end
        amax = max(amax, a)

        # --- NEW: viscous timescale caps ---
        if model.enable_shear
            τπ = visc_tauShear(T, μ, work.n[i], work.e[i], work.P[i], model)
            if isfinite(τπ) && τπ > 0
                τπ_min = min(τπ_min, τπ)
            end
        end
        if model.enable_bulk
            τΠ = visc_tauPi(T, μ, work.n[i], work.e[i], work.P[i], model)
            if isfinite(τΠ) && τΠ > 0
                τΠ_min = min(τΠ_min, τΠ)
            end
        end
    end

    dt = min(CFL * grid.dr/(amax + TINY), CFLτ * τ)

    if model.enable_diff && kmax > 0
        dt = min(dt, model.diff_dt_coeff * grid.dr^2/(kmax + TINY))
    end

    # Viscous relaxation timescale caps (MIS stiffness control)
    if model.enable_shear && isfinite(τπ_min) && model.shear_dt_coeff > 0
        dt = min(dt, model.shear_dt_coeff * τπ_min)
    end
    if model.enable_bulk && isfinite(τΠ_min) && model.bulk_dt_coeff > 0
        dt = min(dt, model.bulk_dt_coeff * τΠ_min)
    end

    return dt
end


@inline function _velocity_stats(work::Work1D, grid; vnear::Float64=0.999)
    ng = grid.nghost
    vmax = 0.0
    nnear = 0
    @inbounds for i in (ng+1):(length(work.vC)-ng)
        work.ok[i] || continue
        v = abs(work.vC[i])
        if isfinite(v)
            vmax = max(vmax, v)
            if v >= vnear
                nnear += 1
            end
        end
    end
    return vmax, nnear
end



# ------------------------------------------------------------
# Dynamic grid expansion (uniform dr, grow rmax by adding cells)
# ------------------------------------------------------------
@inline function _tail_expand_trigger(U, grid, model;
                                     tail_cells::Int,
                                     tail_frac::Float64,
                                     tail_abs_E::Float64)
    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U, 2) - ng

    maxE = 0.0
    @inbounds for i in i0:iL
        maxE = max(maxE, U[L.iE, i])
    end
    if !(isfinite(maxE)) || maxE <= 0.0
        return false, maxE, 0.0
    end

    istart = max(i0, iL - max(tail_cells, 1) + 1)
    maxTail = 0.0
    @inbounds for i in istart:iL
        maxTail = max(maxTail, U[L.iE, i])
    end

    trigger = (maxTail > max(tail_abs_E, tail_frac * maxE))
    return trigger, maxE, maxTail
end

function _resize_grid_uniform!(U, grid::Grid1D, model::IdealDiffViscModel, τ::Float64;
                               Nr_new::Int)
    ng = grid.nghost
    Nvars = size(U, 1)
    Ntot_new = Nr_new + 2*ng
    Unew = zeros(Nvars, Ntot_new)

    rmax_new = grid.dr * Nr_new
    grid_new = make_grid(Nr_new; rmax=rmax_new, nghost=ng)

    _conservative_remap_U!(Unew, grid_new, U, grid, model)

    apply_bc!(Unew, grid_new, τ, model)
    work_new = make_work(Unew)
    prime_work_from_U!(work_new, Unew, grid_new, τ, model)

    return Unew, grid_new, work_new
end

@inline function _expand_grid_uniform!(U, grid::Grid1D, model::IdealDiffViscModel, τ::Float64; Nr_new::Int)
    return _resize_grid_uniform!(U, grid, model, τ; Nr_new=Nr_new)
end

@inline function _find_last_good_cell(U, grid, τ, model::IdealDiffViscModel, work;
                                      Emin::Float64=E_FLOOR, χ::Float64=χ_SrE)
    mark_bad!(work.bad, U, grid, τ, model, work; Emin=Emin, χ=χ)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U, 2) - ng
    last_good = iL
    @inbounds for i in i0:iL
        if work.bad[i]
            last_good = i - 1
            break
        end
    end
    return last_good
end

function _conservative_remap_U!(Unew, grid_new::Grid1D, Uold, grid_old::Grid1D, model::IdealDiffViscModel)
    L = layout(model)
    ng_new = grid_new.nghost
    ng_old = grid_old.nghost

    i0_new = ng_new + 1
    iL_new = size(Unew, 2) - ng_new
    i0_old = ng_old + 1
    iL_old = size(Uold, 2) - ng_old

    # Sweep remap using cylindrical shell areas.
    io = i0_old
    rF_old = grid_old.rF
    rF_new = grid_new.rF

    @inbounds for inew in i0_new:iL_new
        rL_new = max(rF_new[inew], 0.0)
        rR_new = max(rF_new[inew + 1], 0.0)
        A_new = π * (rR_new*rR_new - rL_new*rL_new)
        if !(isfinite(A_new)) || A_new <= 0.0
            continue
        end

        sum_w = 0.0

        # Advance old index to first cell that can overlap.
        while io <= iL_old && max(rF_old[io + 1], 0.0) <= rL_new
            io += 1
        end

        j = io
        while j <= iL_old
            rL_old = max(rF_old[j], 0.0)
            rR_old = max(rF_old[j + 1], 0.0)
            if rL_old >= rR_new
                break
            end
            rL = max(rL_new, rL_old)
            rR = min(rR_new, rR_old)
            if rR > rL
                A_ov = π * (rR*rR - rL*rL)
                w = A_ov / A_new
                sum_w += w
                for a in 1:size(Unew, 1)
                    Unew[a, inew] += Uold[a, j] * w
                end
            end
            j += 1
        end

        # Fill uncovered fraction with vacuum values for all conserved vars.
        if sum_w < 1.0
            vac_w = 1.0 - sum_w
            @inbounds begin
                Unew[L.iDtau, inew] += vac_w * D_VAC
                Unew[L.iSr,   inew] += vac_w * 0.0
                Unew[L.iE,    inew] += vac_w * E_VAC
                if L.hasNur
                    Unew[L.iNur, inew] += vac_w * 0.0
                end
                if L.hasPi
                    Unew[L.iPi, inew] += vac_w * 0.0
                end
                if L.hasPiR
                    Unew[L.iPiR, inew] += vac_w * 0.0
                end
                if L.hasPiEta
                    Unew[L.iPiEta, inew] += vac_w * 0.0
                end
            end
        end
    end

    return nothing
end



# ------------------------------------------------------------
# Conserved charge integral
# ------------------------------------------------------------
function charge_integral_Dtau(U, grid, model::IdealDiffViscModel)
    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng
    Q = 0.0
    @inbounds for i in i0:iL
        rL = max(grid.rF[i],   0.0)
        rR = max(grid.rF[i+1], 0.0)
        Ai = π * (rR*rR - rL*rL)
        Q += U[L.iDtau, i] * Ai
    end
    return Q
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
function initialize!(U, grid, τ0, model::IdealDiffViscModel;
                     init_csv::Union{Nothing,String}=nothing,
                     fugacity_kind::Symbol = :alpha,
                     taper_width::Float64 = 1.0,
                     interp_kind::Symbol = :linear,
                     interp_dr::Union{Nothing,Float64} = nothing,
                     Emin::Float64=E_FLOOR,
                     χ::Float64=χ_SrE)

    eos  = model.eos
    L    = layout(model)
    ng   = grid.nghost
    Ntot = size(U,2)
    fill!(U, 0.0)

    if init_csv === nothing
        @inbounds for i in (ng+1):(Ntot-ng)
            r = grid.rC[i]

            T  = 0.35 + 0.25*exp(-(r/3.0)^2)
            μ  = 0.10*exp(-(r/5.0)^2)

            ur = 0.0
            uτ = 1.0
            v  = 0.0

            nur0_phys   = 0.0
            Pi0_phys    = 0.0
            piR0_phys   = 0.0
            piEta0_phys = 0.0

            P, n, e = eos_Pne(T, μ, eos)

            if !(isfinite(T) && isfinite(μ) && isfinite(P) && isfinite(n) && isfinite(e)) || e < Emin
                T = max(T_MIN, T)
                μ = 0.0
                P, n, e = eos_Pne(T, μ, eos)
                n = 0.0
                e = max(Emin, e)
            end

            w    = e + P
            Ptot = P + Pi0_phys
            weff = w + Pi0_phys

            D  = n*uτ + v*nur0_phys

            U[L.iDtau, i] = τ0 * D
            U[L.iSr,  i]  = weff * (uτ*ur)
            U[L.iE,   i]  = weff * (uτ^2) - Ptot

            if L.hasNur
                U[L.iNur, i] = stored_from_phys(nur0_phys)
            end
            if L.hasPi
                U[L.iPi, i] = stored_from_phys(Pi0_phys)
            end
            if L.hasPiR
                U[L.iPiR, i] = stored_from_phys(piR0_phys)
            end
            if L.hasPiEta
                U[L.iPiEta, i] = stored_from_phys(piEta0_phys)
            end
        end
    else
        itpT, itpF, itpNurStored, rmax_data = load_initial_interpolants(init_csv;
            fugacity_kind=fugacity_kind,
            taper_width=taper_width,
            interp_kind=interp_kind,
            interp_dr=interp_dr,
        )
        @info "Loaded CSV IC (vacuum extrap + taper)" init_csv=init_csv fugacity_kind=fugacity_kind rmax_data=rmax_data grid_rmax=grid.rmax taper_width=taper_width

        @inbounds for i in (ng+1):(Ntot-ng)
            r = grid.rC[i]

            T  = max(itpT(r), T_MIN)
            f0 = itpF(r)
            if fugacity_kind == :alpha
                μ = f0 * T
            else
                λ = max(f0, TINY)
                μ = log(λ) * T
            end

        ur = 0.0
        uτ = 1.0
        v  = 0.0

        nur0_stored = (L.hasNur ? itpNurStored(r) : 0.0)
        nur0_phys   = phys_from_stored(nur0_stored)
        Pi0_phys    = 0.0
        piR0_phys   = 0.0
        piEta0_phys = 0.0

        P, n, e = eos_Pne(T, μ, eos)

        if !(isfinite(T) && isfinite(μ) && isfinite(P) && isfinite(n) && isfinite(e)) || e < Emin
            T = max(T_MIN, T)
            μ = 0.0
            P, n, e = eos_Pne(T, μ, eos)
            n = 0.0
            e = max(Emin, e)
        end

        w    = e + P
        Ptot = P + Pi0_phys
        weff = w + Pi0_phys

        D  = n*uτ + v*nur0_phys

        U[L.iDtau, i] = τ0 * D
        U[L.iSr,  i]  = weff * (uτ*ur)
        U[L.iE,   i]  = weff * (uτ^2) - Ptot

        if L.hasNur
            U[L.iNur, i] = nur0_stored
        end
        if L.hasPi
            U[L.iPi, i] = stored_from_phys(Pi0_phys)
        end
        if L.hasPiR
            U[L.iPiR, i] = stored_from_phys(piR0_phys)
        end
        if L.hasPiEta
            U[L.iPiEta, i] = stored_from_phys(piEta0_phys)
        end
        end
    end

    apply_bc!(U, grid, τ0, model)
    enforce_floors!(U, grid, τ0, model; Emin=Emin, diag=nothing)
    enforce_Sr_energy_constraint!(U, grid, model; χ=χ, mask=nothing, diag=nothing)
    return nothing
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
function run_sim_ideal_diff_visc(; outdir::String,
                                Nr::Int=300, rmax::Float64=20.0,
                                nghost::Int=3,
                                τ0::Float64=0.4, τfinal::Float64=5.0,
                                CFL::Float64=0.2, CFLτ::Float64=0.05,
                                auto_cfl::Bool=false,
                                cfl_safety::Float64=0.9,
                                cfl_max::Float64=0.5,
                                cfltau_max::Float64=0.2,
                                time_integrator::Symbol = :ssprk2,
                                dump_dt::Float64=0.1,
                                Emin::Float64=E_FLOOR,
                                χ::Float64=χ_SrE,
                                init_csv::Union{Nothing,String}=nothing,
                                fugacity_kind::Symbol = :alpha,
                                taper_width::Float64 = 0.0,
                                interp_kind::Symbol = :linear,
                                interp_dr::Union{Nothing,Float64} = nothing,
                                log_every::Int=50,
                                log_corrections_every::Int=50,
                                # dynamic grid expansion
                                expand_grid::Bool=false,
                                expand_factor::Float64=1.5,
                                expand_tail_cells::Int=8,
                                expand_tail_frac::Float64=1e-6,
                                expand_tail_abs_E::Float64=1e-18,
                                expand_min_cells::Int=64,
                                expand_max_nr::Int=200_000,
                                expand_max_rmax::Float64=Inf,
                                expand_cooldown_steps::Int=25,
                                # initial shrink-to-good range
                                init_good_range::Bool=true,
                                init_good_min_cells::Int=32,
                                init_good_buffer_cells::Int=4,
                                # charge diffusion
                                enable_diff::Bool=true,
                                DsT::Float64=5.24,
                                kappa_coeff::Union{Nothing,Float64}=nothing,
                                diffusion_drive::Symbol = :alpha,
                                tauN_coeff::Float64=1.0, #CHECK THAT!
                                deltaN_factor::Float64=0.0,
                                diff_dt_coeff::Float64=0.02,
                                shear_dt_coeff::Float64=0.3,
                                bulk_dt_coeff::Float64=0.3,
                                nur_clip_factor::Float64=-1.0,
                                alpha_filter_eps::Float64=0.0,
                                nur_filter_eps::Float64=0.0,
                                alpha_smooth_len::Float64=0.0,
                                nur_smooth_len::Float64=0.0,
                                do_soft_project_nur::Bool=false,
                                do_axis_project_nur::Bool=false,
                                axis_project_nfit::Int=2,
                                # transport choice for dissipatives (per variable)
                                # - advect_*       => FV advect as extra conserved scalar
                                # - relax_advect_* => include u^r ∂r term inside MIS PDE relaxation update
                                # NOTE: you must not enable BOTH for the same variable.
                                advect_nur::Bool=false,
                                relax_advect_nur::Bool=true,
                                # viscosity
                                enable_shear::Bool=true,
                                enable_bulk::Bool=true,
                                eta_over_s::Float64=0.1,
                                zeta_over_s::Float64=0.1,
                                tauPi_coeff::Float64=1.0,
                                tauShear_coeff::Float64=1.0,
                                deltaPi_factor::Float64=0.0,
                                deltaShear_factor::Float64=0.0,

                                # full (2nd-order) relaxation couplings
                                taupi_pi_factor::Float64=0.0,
                                lambda_Pi_pi_factor::Float64=0.0,
                                lambda_pi_Pi_factor::Float64=0.0,
                                lambda_NN_factor::Float64=0.0,
                                visc_filter_eps::Float64=0.0,
                                visc_smooth_len::Float64=0.0,
                                Pi_clip_factor::Float64=-1.0,
                                pi_clip_factor::Float64=-1.0,
                                advect_Pi::Bool=false,
                                relax_advect_Pi::Bool=true,
                                advect_pi::Bool=false,
                                relax_advect_pi::Bool=true,
                                # postprocessing
                                postprocess::Bool=true,
                                # EOS choice
                                eos = ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0))

    grid = make_grid(Nr; rmax=rmax, nghost=nghost)

    layout = StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    shear_model = QGPViscosity(eta_over_s, tauShear_coeff)
    bulk_model  = (zeta_over_s == 0.0) ? ZeroBulkViscosity() : SimpleBulkViscosity(zeta_over_s, tauPi_coeff)


    _check_transport_flags!(;
        advect_nur=advect_nur, relax_advect_nur=relax_advect_nur,
        advect_Pi=advect_Pi,   relax_advect_Pi=relax_advect_Pi,
        advect_pi=advect_pi,   relax_advect_pi=relax_advect_pi
    )

    # Convention: the diffusion input is the *dimensionless* constant DsT.
    # The code then uses κ(T,μ,n) = (DsT/T) * n (up to unit conversion).
    DsT_val = isnothing(kappa_coeff) ? DsT : kappa_coeff

    model  = IdealDiffViscModel(eos, layout, IdealPrimRec(),
        # charge diffusion
        enable_diff, diffusion_drive, DsT_val, tauN_coeff, deltaN_factor,
        diff_dt_coeff, shear_dt_coeff, bulk_dt_coeff, nur_clip_factor,
        alpha_filter_eps, nur_filter_eps, alpha_smooth_len, nur_smooth_len,
        do_soft_project_nur,
        do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur,
        # viscosity
        enable_shear, enable_bulk, shear_model, bulk_model,
        deltaPi_factor, deltaShear_factor,
        taupi_pi_factor, lambda_Pi_pi_factor, lambda_pi_Pi_factor, lambda_NN_factor,
        visc_filter_eps, visc_smooth_len,
        Pi_clip_factor, pi_clip_factor,
        advect_Pi, advect_pi,
        # NEW
        relax_advect_Pi, relax_advect_pi
    )

    U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
    initialize!(U, grid, τ0, model;
        init_csv=init_csv,
        fugacity_kind=fugacity_kind,
        taper_width=taper_width,
        interp_kind=interp_kind,
        interp_dr=interp_dr,
        Emin=Emin,
        χ=χ,
    )
    work = make_work(U)
    diag = DiagCounters()

    τ = τ0
    it = 0
    next_dump = τ0
    last_expand_it = -10^9

    # Seed work caches once so dt-from-work is valid at the first step
    prime_work_from_U!(work, U, grid, τ, model)

    if init_good_range
        last_good = _find_last_good_cell(U, grid, τ, model, work; Emin=Emin, χ=χ)
        ng = grid.nghost
        min_good = ng + max(init_good_min_cells, 1)
        iL = size(U, 2) - ng
        if last_good >= min_good && last_good < iL
            Nr_new = last_good - ng - max(init_good_buffer_cells, 0)
            Nr_new = max(Nr_new, max(init_good_min_cells, 1))
            U, grid, work = _resize_grid_uniform!(U, grid, model, τ; Nr_new=Nr_new)
            @info "Init shrink to good cells" τ=τ Nr_new=grid.Nr rmax_new=grid.rmax last_good=last_good buffer=init_good_buffer_cells
        end
    end

    CFL_run = CFL
    CFLτ_run = CFLτ
    if auto_cfl
        # Use the same dt constraint logic as IC diagnostics to pick a CFL that
        # is not more restrictive than diffusion/viscosity constraints at τ0.
        dtstats = ic_dt_stats(work, grid, τ, model; CFL=1.0, CFLτ=1.0)

        dt_space_1 = grid.dr / (dtstats.amax + TINY)
        dt_tau_1 = τ
        dt_non_cfl = Inf
        if isfinite(dtstats.dt_diff)
            dt_non_cfl = min(dt_non_cfl, dtstats.dt_diff)
        end
        if isfinite(dtstats.dt_shear)
            dt_non_cfl = min(dt_non_cfl, dtstats.dt_shear)
        end
        if isfinite(dtstats.dt_bulk)
            dt_non_cfl = min(dt_non_cfl, dtstats.dt_bulk)
        end

        if isfinite(dt_non_cfl)
            dt_target = cfl_safety * dt_non_cfl
            CFL_run = clamp(dt_target / dt_space_1, 1e-6, cfl_max)
            CFLτ_run = clamp(dt_target / dt_tau_1, 1e-6, cfltau_max)
            @info "Auto CFL/CFLτ from IC" τ=τ dr=grid.dr amax=dtstats.amax dt_space_1=dt_space_1 dt_tau_1=dt_tau_1 dt_non_cfl=dt_non_cfl cfl_safety=cfl_safety CFL=CFL_run CFLτ=CFLτ_run
        end
    end

    @info "Start IDEAL+DIFF+VISC" threads=Threads.nthreads() outdir=outdir Nr=grid.Nr dr=grid.dr τ0=τ0 τfinal=τfinal rmax=grid.rmax CFL=CFL_run CFLτ=CFLτ_run time_integrator=time_integrator enable_diff=enable_diff diffusion_drive=diffusion_drive enable_shear=enable_shear enable_bulk=enable_bulk eta_over_s=eta_over_s zeta_over_s=zeta_over_s DISS_SIGN=DISS_SIGN

    nur_transport = advect_nur ? "FV" : (relax_advect_nur ? "relax" : "off")
    Pi_transport  = advect_Pi  ? "FV" : (relax_advect_Pi  ? "relax" : "off")
    pi_transport  = advect_pi  ? "FV" : (relax_advect_pi  ? "relax" : "off")

    # Operator splitting only applies if you transport a dissipative variable via FV fluxes
    # while the relaxation update itself is local (i.e. no u^r ∂r term).
    nur_split = advect_nur && !relax_advect_nur
    Pi_split  = advect_Pi  && !relax_advect_Pi
    pi_split  = advect_pi  && !relax_advect_pi

    @info "Dissipative transport config"  nur_transport=nur_transport Pi_transport=Pi_transport pi_transport=pi_transport operator_split=(nur=nur_split, Pi=Pi_split, pi=pi_split)

    mkpath(outdir)
    clear_dir!(outdir)
    write_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)), U, grid, τ, model)

    while τ < τfinal - 1e-12
        diag_reset_last!(diag)

        if expand_grid && (it - last_expand_it >= expand_cooldown_steps)
            trigger, maxE, maxTail = _tail_expand_trigger(U, grid, model;
                                                         tail_cells=expand_tail_cells,
                                                         tail_frac=expand_tail_frac,
                                                         tail_abs_E=expand_tail_abs_E)
            if trigger
                Nr_old = grid.Nr
                rmax_old = grid.rmax
                Nr_target = max(Int(ceil(Nr_old * max(expand_factor, 1.0))), Nr_old + max(expand_min_cells, 1))
                Nr_new = min(Nr_target, expand_max_nr)

                if isfinite(expand_max_rmax)
                    Nr_cap = Int(floor(expand_max_rmax / grid.dr))
                    Nr_new = min(Nr_new, Nr_cap)
                end

                if Nr_new > Nr_old
                    U, grid, work = _resize_grid_uniform!(U, grid, model, τ; Nr_new=Nr_new)
                    last_expand_it = it
                    @info "Expanded grid" it=it τ=τ Nr_old=Nr_old Nr_new=grid.Nr rmax_old=rmax_old rmax_new=grid.rmax maxE=maxE maxTail=maxTail tail_frac=expand_tail_frac tail_abs_E=expand_tail_abs_E
                end
            end
        end

        # IMPORTANT: honor run-time CFL values
        Δτ = compute_dt_from_work(work, grid, τ, model; CFL=CFL_run, CFLτ=CFLτ_run)
        if τ + Δτ > τfinal
            Δτ = τfinal - τ
        end

        if time_integrator == :ssprk2
            Δused = step_ssprk2!(U, grid, τ, Δτ, model, work; Emin=Emin, χ=χ, diag=diag)
        elseif time_integrator == :ssprk3
            Δused = step_ssprk3!(U, grid, τ, Δτ, model, work; Emin=Emin, χ=χ, diag=diag)
        else
            error("Unknown time_integrator=$(time_integrator). Use :ssprk2 or :ssprk3")
        end
        τ += Δused
        it += 1

        # Late-time debugging: track approach to causality limit.
        vmax, vnear = _velocity_stats(work, grid; vnear=0.999)
        diag_add!(diag; vmax=vmax, vnear=vnear)

        _maybe_log_sr_clamp!(diag, U, grid, τ, model, work, it; Emin=Emin, χ=χ)
        _abort_on_first_srscale!(diag, U, grid, τ, model, work, it; Emin=Emin, χ=χ)
        _abort_on_first_primfail!(diag, U, grid, τ, model, work, it; Emin=Emin, χ=χ)

        if diag.last_prim_fail_cells > 0 && diag.last_prim_fail_i != 0
            i = diag.last_prim_fail_i
            r = (1 <= i <= length(grid.rC)) ? grid.rC[i] : NaN
            @warn "primitive recovery failed" it=it τ=τ i=i r=r Dtau=diag.last_prim_fail_Dtau Sr=diag.last_prim_fail_Sr E=diag.last_prim_fail_E nur_stored=diag.last_prim_fail_nur_stored Pi_stored=diag.last_prim_fail_Pi_stored piR_stored=diag.last_prim_fail_piR_stored piEta_stored=diag.last_prim_fail_piEta_stored
        end

        if (it % log_corrections_every) == 0
            @debug "corrections (last window)" it=it τ=τ Δτ=Δused primfail=diag.last_prim_fail_cells floorE=diag.last_floor_E_cells floorD=diag.last_floor_D_cells nanfix=diag.last_nanfix_cells SrScaled=diag.last_Sr_scaled_cells SrScaledMax=diag.last_Sr_scaled_maxratio SrScaledMaxI=diag.last_Sr_scaled_max_i vMax=diag.last_v_max vNear=diag.last_v_near_cells PiClip=diag.last_Pi_clipped_cells PiClipMax=diag.last_Pi_clip_maxratio PiClipMaxI=diag.last_Pi_clip_max_i piClip=diag.last_pi_clipped_cells piClipMax=diag.last_pi_clip_maxratio piClipMaxI=diag.last_pi_clip_max_i mood1=diag.last_mood_stage1_bad mood2=diag.last_mood_stage2_bad dtHalvings=diag.last_stage_dt_halvings
        end

        if τ >= next_dump - 1e-12
            fname = joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ))
            write_snapshot_csv(fname, U, grid, τ, model)

            @info "dump" τ=τ file=fname it=it Q_Dtau=charge_integral_Dtau(U, grid, model) vMax=diag.last_v_max vNear=diag.last_v_near_cells primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells floorD_total=diag.floor_D_cells SrScaled_total=diag.Sr_scaled_cells mood1_total=diag.mood_stage1_bad mood2_total=diag.mood_stage2_bad
            next_dump += dump_dt
        end

        if (it % log_every) == 0
            @info "progress" τ=τ Δτ=Δused it=it Q_Dtau=charge_integral_Dtau(U, grid, model) vMax=diag.last_v_max vNear=diag.last_v_near_cells primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells floorD_total=diag.floor_D_cells SrScaled_total=diag.Sr_scaled_cells
        end
    end

    @info "Done IDEAL+DIFF+VISC" τ=τ it=it outdir=outdir Q_Dtau=charge_integral_Dtau(U, grid, model) primfail_total=diag.prim_fail_cells floorE_total=diag.floor_E_cells floorD_total=diag.floor_D_cells nanfix_total=diag.nanfix_cells SrScaled_total=diag.Sr_scaled_cells mood1_total=diag.mood_stage1_bad mood2_total=diag.mood_stage2_bad dt_halvings_total=diag.stage_dt_halvings

    if postprocess
        # Postprocess: write Langevin-style current snapshots + splines (JLD2)
        currents_path = write_hydro_currents_jld2(outdir; outdir=outdir)
        plot_splines_dir = normpath(joinpath(@__DIR__, "..", "..", "Julia/Plot", "splines"))
        splines_path  = write_hydro_currents_splines_jld2(
            outdir;
            outdir=plot_splines_dir,
            filename="FiVo.jld2",
            tag="FiVo",
            overwrite=true,
        )
        @info "Wrote hydro currents JLD2" currents_path splines_path
    end
    return nothing
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
function main()
    setup_logger!(level=Logging.Info)

    env_float(name::String, default::Float64) = haskey(ENV, name) ? parse(Float64, ENV[name]) : default
    env_int(name::String, default::Int) = haskey(ENV, name) ? parse(Int, ENV[name]) : default
    env_str(name::String, default::String) = haskey(ENV, name) ? String(ENV[name]) : default
    env_bool(name::String, default::Bool) = haskey(ENV, name) ? (parse(Int, ENV[name]) != 0) : default
    env_maybe_float(name::String, default::Union{Nothing,Float64}=nothing) = haskey(ENV, name) ? ((s = strip(String(ENV[name]))); isempty(s) ? default : parse(Float64, s)) : default

    function load_kv_file(path::String)
        d = Dict{String,String}()
        isfile(path) || return d
        for raw in eachline(path)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            occursin("=", line) || continue
            k, v = split(line, "=", limit=2)
            d[strip(k)] = strip(v)
        end
        return d
    end

    use_diag_settings = env_int("USE_DIAG_SETTINGS", 1) != 0
    diag_env_path = env_str("DIAG_ENV_FILE", "last_ic_diagnostics.env")
    diag = use_diag_settings ? load_kv_file(diag_env_path) : Dict{String,String}()
    if use_diag_settings && !isempty(diag)
        @info "Loaded diagnostics settings" file=diag_env_path
    end

    diag_str(name::String, default::String) = haskey(diag, name) ? diag[name] : default
    diag_int(name::String, default::Int) = haskey(diag, name) ? parse(Int, diag[name]) : default
    diag_float(name::String, default::Float64) = (haskey(diag, name) && !isempty(strip(diag[name]))) ? parse(Float64, diag[name]) : default
    diag_bool(name::String, default::Bool) = haskey(diag, name) ? (parse(Int, diag[name]) != 0) : default
    diag_maybe_float(name::String, default::Union{Nothing,Float64}=nothing) = (haskey(diag, name) && !isempty(strip(diag[name]))) ? parse(Float64, diag[name]) : default

    init_csv = haskey(ENV, "INIT_CSV") ? env_str("INIT_CSV", "") : diag_str("INIT_CSV", "data/initial_profiles_physical.csv")
    τ0 = haskey(ENV, "TAU0") ? env_float("TAU0", 0.4) : diag_float("TAU0", 0.4)
    τfinal = haskey(ENV, "TAUFINAL") ? env_float("TAUFINAL", 5.0) : diag_float("TAUFINAL", 5.0)
    Emin = haskey(ENV, "EMIN") ? env_float("EMIN", E_FLOOR) : diag_float("EMIN", E_FLOOR)
    χ = haskey(ENV, "CHI") ? env_float("CHI", χ_SrE) : diag_float("CHI", χ_SrE)

    # Match IC CSV grid by default.
    # CSV is assumed to contain cell-center r values including ghost cells.
    Nr = haskey(ENV, "NR") ? env_int("NR", -1) : diag_int("NR", -1)
    rmax = haskey(ENV, "RMAX") ? env_float("RMAX", NaN) : diag_float("RMAX", NaN)
    nghost = haskey(ENV, "NGHOST") ? env_int("NGHOST", -1) : diag_int("NGHOST", -1)
    if (Nr < 0 || !isfinite(rmax) || nghost < 0) && !isempty(init_csv) && isfile(init_csv)
        rs = Float64[]
        for row in CSV.File(init_csv)
            push!(rs, Float64(row.r))
        end
        if length(rs) >= 2
            N = length(rs)
            rmin = rs[1]
            rmax_data = rs[end]
            dr = rs[2] - rs[1]
            # Infer nghost: rmin = -(nghost - 0.5)*dr  =>  nghost = -rmin/dr + 0.5
            ng_est = round(Int, (-rmin / dr) + 0.5)
            Nr_est = N - 2 * ng_est
            rmax_est = rmax_data - (ng_est - 0.5) * dr
            @info "Auto grid from CSV" init_csv=init_csv N=N rmin=rmin rmax_data=rmax_data dr=dr Nr=Nr_est nghost=ng_est rmax=rmax_est
            if Nr < 0
                Nr = Nr_est
            end
            if !isfinite(rmax)
                rmax = rmax_est
            end
            if nghost < 0
                nghost = ng_est
            end
        end
    end

    if Nr < 0
        Nr = 1000
    end
    if !isfinite(rmax)
        rmax = 28.0
    end
    if nghost < 0
        nghost = 1
    end

    # CFL defaults (can override via ENV)
    CFL = haskey(ENV, "CFL") ? env_float("CFL", 0.2) : diag_float("CFL", 0.2)
    CFLτ = if haskey(ENV, "CFLTAU")
        parse(Float64, ENV["CFLTAU"])
    elseif haskey(ENV, "CFLtau")
        parse(Float64, ENV["CFLtau"])
    else
        diag_float("CFLTAU", 0.05)
    end
    auto_cfl = env_int("AUTO_CFL", 1) != 0
    cfl_safety = env_float("CFL_SAFETY", 0.9)
    cfl_max = env_float("CFL_MAX", 0.5)
    cfltau_max = env_float("CFLTAU_MAX", 0.2)
    dump_dt = haskey(ENV, "DUMP_DT") ? env_float("DUMP_DT", 0.1) : diag_float("DUMP_DT", 0.1)
    log_every = haskey(ENV, "LOG_EVERY") ? env_int("LOG_EVERY", 50) : diag_int("LOG_EVERY", 50)
    log_corrections_every = haskey(ENV, "LOG_CORRECTIONS_EVERY") ? env_int("LOG_CORRECTIONS_EVERY", 50) : diag_int("LOG_CORRECTIONS_EVERY", 50)
    time_integrator_str = lowercase(haskey(ENV, "TIME_INTEGRATOR") ? env_str("TIME_INTEGRATOR", "ssprk3") : diag_str("TIME_INTEGRATOR", "ssprk3"))
    time_integrator = time_integrator_str == "ssprk2" ? :ssprk2 : :ssprk3

    # No taper + linear interp by default (matches requested setting)
    taper_width = haskey(ENV, "TAPER_WIDTH") ? env_float("TAPER_WIDTH", 0.0) : diag_float("TAPER_WIDTH", 0.0)
    interp_kind = begin
        interp_s = haskey(ENV, "INTERP") ? env_str("INTERP", "linear") : diag_str("INTERP", "linear")
        lowercase(interp_s) == "cubic" ? :cubic : :linear
    end
    interp_dr_val = haskey(ENV, "INTERP_DR") ? env_float("INTERP_DR", NaN) : diag_float("INTERP_DR", NaN)
    interp_dr = isfinite(interp_dr_val) ? interp_dr_val : nothing
    fug_str = lowercase(haskey(ENV, "FUGACITY") ? env_str("FUGACITY", "alpha") : diag_str("FUGACITY", "alpha"))
    fugacity_kind = fug_str == "lambda" ? :lambda : :alpha

    outdir = haskey(ENV, "HYDRO_OUTDIR") ? env_str("HYDRO_OUTDIR", "") : diag_str("HYDRO_OUTDIR", normpath(joinpath(@__DIR__, "..", "..", "Julia/Plot", "snapshots", "FiVo")))
    outdir = normpath(outdir)

    # Dynamic grid expansion settings
    expand_grid = env_int("EXPAND_GRID", 0) != 0
    expand_factor = env_float("EXPAND_FACTOR", 1.5)
    expand_tail_cells = env_int("EXPAND_TAIL_CELLS", 8)
    expand_tail_frac = env_float("EXPAND_TAIL_FRAC", 1e-6)
    expand_tail_abs_E = env_float("EXPAND_TAIL_ABS_E", 1e-18)
    expand_min_cells = env_int("EXPAND_MIN_CELLS", 64)
    expand_max_nr = env_int("EXPAND_MAX_NR", 200_000)
    expand_max_rmax = env_float("EXPAND_MAX_RMAX", Inf)
    expand_cooldown_steps = env_int("EXPAND_COOLDOWN", 25)
    init_good_range = env_int("INIT_GOOD_RANGE", 1) != 0
    init_good_min_cells = env_int("INIT_GOOD_MIN_CELLS", 32)
    init_good_buffer_cells = env_int("INIT_GOOD_BUFFER", 4)

    # Physics toggles (allow env override; otherwise use last diagnostics settings; default off)
    enable_diff  = haskey(ENV, "ENABLE_DIFF")  ? (env_int("ENABLE_DIFF", 0) != 0)  : (diag_int("ENABLE_DIFF", 0) != 0)
    enable_shear = haskey(ENV, "ENABLE_SHEAR") ? (env_int("ENABLE_SHEAR", 0) != 0) : (diag_int("ENABLE_SHEAR", 0) != 0)
    enable_bulk  = haskey(ENV, "ENABLE_BULK")  ? (env_int("ENABLE_BULK", 0) != 0)  : (diag_int("ENABLE_BULK", 0) != 0)
    DsT = haskey(ENV, "DS_T") ? env_float("DS_T", 0.24) : diag_float("DS_T", 0.24)
    kappa_coeff = haskey(ENV, "KAPPA_COEFF") ? env_maybe_float("KAPPA_COEFF", nothing) : diag_maybe_float("KAPPA_COEFF", nothing)
    diffusion_drive_str = lowercase(haskey(ENV, "DIFFUSION_DRIVE") ? env_str("DIFFUSION_DRIVE", "alpha") : diag_str("DIFFUSION_DRIVE", "alpha"))
    diffusion_drive = diffusion_drive_str == "n" ? :n : :alpha
    tauN_coeff = haskey(ENV, "TAU_N_COEFF") ? env_float("TAU_N_COEFF", 1.0) : diag_float("TAU_N_COEFF", 1.0)
    deltaN_factor = haskey(ENV, "DELTA_N_FACTOR") ? env_float("DELTA_N_FACTOR", 0.0) : diag_float("DELTA_N_FACTOR", 0.0)
    diff_dt_coeff = haskey(ENV, "DIFF_DT_COEFF") ? env_float("DIFF_DT_COEFF", 0.01) : diag_float("DIFF_DT_COEFF", 0.01)
    shear_dt_coeff = haskey(ENV, "SHEAR_DT_COEFF") ? env_float("SHEAR_DT_COEFF", 0.01) : diag_float("SHEAR_DT_COEFF", 0.01)
    bulk_dt_coeff = haskey(ENV, "BULK_DT_COEFF") ? env_float("BULK_DT_COEFF", 0.01) : diag_float("BULK_DT_COEFF", 0.01)
    nur_clip_factor = haskey(ENV, "NUR_CLIP_FACTOR") ? env_float("NUR_CLIP_FACTOR", -1.0) : diag_float("NUR_CLIP_FACTOR", -1.0)
    alpha_filter_eps = haskey(ENV, "ALPHA_FILTER_EPS") ? env_float("ALPHA_FILTER_EPS", 0.0) : diag_float("ALPHA_FILTER_EPS", 0.0)
    nur_filter_eps = haskey(ENV, "NUR_FILTER_EPS") ? env_float("NUR_FILTER_EPS", 0.0) : diag_float("NUR_FILTER_EPS", 0.0)
    alpha_smooth_len = haskey(ENV, "ALPHA_SMOOTH_LEN") ? env_float("ALPHA_SMOOTH_LEN", 0.0) : diag_float("ALPHA_SMOOTH_LEN", 0.0)
    nur_smooth_len = haskey(ENV, "NUR_SMOOTH_LEN") ? env_float("NUR_SMOOTH_LEN", 0.0) : diag_float("NUR_SMOOTH_LEN", 0.0)
    do_soft_project_nur = haskey(ENV, "DO_SOFT_PROJECT_NUR") ? env_bool("DO_SOFT_PROJECT_NUR", true) : diag_bool("DO_SOFT_PROJECT_NUR", true)
    do_axis_project_nur = haskey(ENV, "DO_AXIS_PROJECT_NUR") ? env_bool("DO_AXIS_PROJECT_NUR", false) : diag_bool("DO_AXIS_PROJECT_NUR", false)
    axis_project_nfit = haskey(ENV, "AXIS_PROJECT_NFIT") ? env_int("AXIS_PROJECT_NFIT", 16) : diag_int("AXIS_PROJECT_NFIT", 16)
    advect_nur = haskey(ENV, "ADVECT_NUR") ? env_bool("ADVECT_NUR", false) : diag_bool("ADVECT_NUR", false)
    relax_advect_nur = haskey(ENV, "RELAX_ADVECT_NUR") ? env_bool("RELAX_ADVECT_NUR", true) : diag_bool("RELAX_ADVECT_NUR", true)

    eta_over_s = haskey(ENV, "ETA_OVER_S") ? env_float("ETA_OVER_S", 0.1) : diag_float("ETA_OVER_S", 0.1)
    zeta_over_s = haskey(ENV, "ZETA_OVER_S") ? env_float("ZETA_OVER_S", 0.1) : diag_float("ZETA_OVER_S", 0.1)
    tauPi_coeff = haskey(ENV, "TAU_PI_COEFF") ? env_float("TAU_PI_COEFF", 15.0) : diag_float("TAU_PI_COEFF", 15.0)
    tauShear_coeff = haskey(ENV, "TAU_SHEAR_COEFF") ? env_float("TAU_SHEAR_COEFF", 0.2) : diag_float("TAU_SHEAR_COEFF", 0.2)
    deltaPi_factor = haskey(ENV, "DELTA_BULK_FACTOR") ? env_float("DELTA_BULK_FACTOR", 0.0) : diag_float("DELTA_BULK_FACTOR", 0.0)
    deltaShear_factor = haskey(ENV, "DELTA_SHEAR_FACTOR") ? env_float("DELTA_SHEAR_FACTOR", 0.0) : diag_float("DELTA_SHEAR_FACTOR", 0.0)
    taupi_pi_factor = haskey(ENV, "TAUPI_PI_FACTOR") ? env_float("TAUPI_PI_FACTOR", 0.0) : diag_float("TAUPI_PI_FACTOR", 0.0)
    lambda_Pi_pi_factor = haskey(ENV, "LAMBDA_BULK_SHEAR_FACTOR") ? env_float("LAMBDA_BULK_SHEAR_FACTOR", 0.0) : diag_float("LAMBDA_BULK_SHEAR_FACTOR", 0.0)
    lambda_pi_Pi_factor = haskey(ENV, "LAMBDA_SHEAR_BULK_FACTOR") ? env_float("LAMBDA_SHEAR_BULK_FACTOR", 0.0) : diag_float("LAMBDA_SHEAR_BULK_FACTOR", 0.0)
    lambda_NN_factor = haskey(ENV, "LAMBDA_NN_FACTOR") ? env_float("LAMBDA_NN_FACTOR", 0.0) : diag_float("LAMBDA_NN_FACTOR", 0.0)
    visc_filter_eps = haskey(ENV, "VISC_FILTER_EPS") ? env_float("VISC_FILTER_EPS", 0.0) : diag_float("VISC_FILTER_EPS", 0.0)
    visc_smooth_len = haskey(ENV, "VISC_SMOOTH_LEN") ? env_float("VISC_SMOOTH_LEN", 0.0) : diag_float("VISC_SMOOTH_LEN", 0.0)
    Pi_clip_factor = haskey(ENV, "PI_CLIP_FACTOR") ? env_float("PI_CLIP_FACTOR", -1.0) : diag_float("PI_CLIP_FACTOR", -1.0)
    pi_clip_factor = haskey(ENV, "SHEAR_CLIP_FACTOR") ? env_float("SHEAR_CLIP_FACTOR", -1.0) : diag_float("SHEAR_CLIP_FACTOR", -1.0)
    advect_Pi = haskey(ENV, "ADVECT_BULK_PI") ? env_bool("ADVECT_BULK_PI", false) : diag_bool("ADVECT_BULK_PI", false)
    relax_advect_Pi = haskey(ENV, "RELAX_ADVECT_BULK_PI") ? env_bool("RELAX_ADVECT_BULK_PI", true) : diag_bool("RELAX_ADVECT_BULK_PI", true)
    advect_pi = haskey(ENV, "ADVECT_SHEAR_PI") ? env_bool("ADVECT_SHEAR_PI", false) : diag_bool("ADVECT_SHEAR_PI", false)
    relax_advect_pi = haskey(ENV, "RELAX_ADVECT_SHEAR_PI") ? env_bool("RELAX_ADVECT_SHEAR_PI", true) : diag_bool("RELAX_ADVECT_SHEAR_PI", true)

    function build_eos(kind::String)
        k = lowercase(kind)
        if k in ("lattice", "latticehrg", "lhrg")
            return LatticeHRGEOS()
        elseif k in ("conformal", "conformalhq", "chq")
            return ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        elseif k in ("running", "runningconformal", "rconformal")
            return RunningConformalHQEOS()
        else
            @warn "Unknown EOS in settings; falling back to LatticeHRGEOS" kind=kind
            return LatticeHRGEOS()
        end
    end
    eos_kind = haskey(ENV, "EOS") ? env_str("EOS", "latticehrg") : diag_str("EOS", "latticehrg")

    eos = build_eos(eos_kind)
    run_sim_ideal_diff_visc(outdir=outdir,
                            Nr=Nr, rmax=rmax, nghost=nghost, τ0=τ0, τfinal=τfinal,
                            CFL=CFL,
                            CFLτ=CFLτ,
                            auto_cfl=auto_cfl,
                            cfl_safety=cfl_safety,
                            cfl_max=cfl_max,
                            cfltau_max=cfltau_max,
                            time_integrator=time_integrator,
                            dump_dt=dump_dt,
                            Emin=Emin,
                            χ=χ,

                            init_csv=init_csv,
                            fugacity_kind=fugacity_kind,
                            taper_width=taper_width,
                            interp_kind=interp_kind,
                            interp_dr=interp_dr,
                            log_every=log_every,
                            log_corrections_every=log_corrections_every,

                            expand_grid=expand_grid,
                            expand_factor=expand_factor,
                            expand_tail_cells=expand_tail_cells,
                            expand_tail_frac=expand_tail_frac,
                            expand_tail_abs_E=expand_tail_abs_E,
                            expand_min_cells=expand_min_cells,
                            expand_max_nr=expand_max_nr,
                            expand_max_rmax=expand_max_rmax,
                            expand_cooldown_steps=expand_cooldown_steps,

                            init_good_range=init_good_range,
                            init_good_min_cells=init_good_min_cells,
                            init_good_buffer_cells=init_good_buffer_cells,

                            enable_diff=enable_diff,
                            DsT=DsT,
                            kappa_coeff=kappa_coeff,
                            diffusion_drive=diffusion_drive,
                            tauN_coeff=tauN_coeff,
                            deltaN_factor=deltaN_factor,
                            diff_dt_coeff=diff_dt_coeff,
                            nur_clip_factor=nur_clip_factor,
                            alpha_filter_eps=alpha_filter_eps,
                            nur_filter_eps=nur_filter_eps,
                            alpha_smooth_len=alpha_smooth_len,
                            nur_smooth_len=nur_smooth_len,
                            do_soft_project_nur=do_soft_project_nur,
                            do_axis_project_nur=do_axis_project_nur,
                            axis_project_nfit=axis_project_nfit,
                            advect_nur=advect_nur,
                            relax_advect_nur=relax_advect_nur,

                            enable_shear=enable_shear,
                            shear_dt_coeff=shear_dt_coeff,
                            eta_over_s=eta_over_s,
                            tauShear_coeff=tauShear_coeff,
                            deltaShear_factor=deltaShear_factor,

                            enable_bulk=enable_bulk,
                            bulk_dt_coeff=bulk_dt_coeff,
                            zeta_over_s=zeta_over_s,
                            tauPi_coeff=tauPi_coeff,
                            deltaPi_factor=deltaPi_factor,

                            taupi_pi_factor=taupi_pi_factor,
                            lambda_Pi_pi_factor=lambda_Pi_pi_factor,
                            lambda_pi_Pi_factor=lambda_pi_Pi_factor,
                            lambda_NN_factor=lambda_NN_factor,
                            visc_filter_eps=visc_filter_eps,
                            visc_smooth_len=visc_smooth_len,
                            Pi_clip_factor=Pi_clip_factor,
                            pi_clip_factor=pi_clip_factor,
                            advect_Pi=advect_Pi,
                            relax_advect_Pi=relax_advect_Pi,
                            advect_pi=advect_pi,
                            relax_advect_pi=relax_advect_pi,

                            eos = eos)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module hydro