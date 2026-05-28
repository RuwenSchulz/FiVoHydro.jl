#!/usr/bin/env julia
# ==============================================================================
# mainBDNK.jl — BDNK causal first-order diffusion with FiVoHydro infrastructure
#
# Instead of Israel-Stewart relaxation (τ_n Du ν^r + ν^r = ν_NS),
# uses BDNK constitutive relation:
#   ν^μ = -κ Δ^{μν} ∇_ν α  +  ε_ν u^μ (u^ν ∇_ν α)
#
# Approach: operator-split BDNK diffusion.
#   1. The FV step evolves {E, Sr, Dtau, nur} with HLLE.
#      nur stores the Landau-frame projection of the BDNK diffusion
#      current, so the Landau-frame assumptions in primitive recovery
#      and HLLE flux are satisfied.  The total charge current J^μ is
#      frame-independent: n_eff u^r + ν^r_Landau = n u^r + ν^r_BDNK.
#   2. After the SSPRK step, instead of IS relaxation, we set
#      ν^r_Landau = BDNK constitutive value projected to Landau frame.
#
# The model is constructed with enable_diff=false so that
# relax_dissipative! skips the IS diffusion update; we apply our own.
#
# CFL: must respect v_sig = √(κ/ε_ν) ≤ c.
# ==============================================================================

include(joinpath(@__DIR__, "main.jl"))

using Printf
using Logging
using CSV
using SpecialFunctions

# ============================================================================
# BDNK diffusion step (replaces IS relaxation for charge sector)
# ============================================================================
"""
    bdnk_diffusion_step!(U, grid, τ, Δ, model, work; ε_ν, alpha_prev)

Compute the BDNK diffusion current ν^r and store it in U[iNur,:].
The FV flux is F_Dtau = τ(n u^r + ν^r), and the conserved variable
is D_tau = τ(n u^τ + v ν^r), both assuming Landau frame.

For BDNK, the charge flux J^r simplifies (ε_ν cancels) to:

    J^r = v (J^τ − κ ∂_τ α) − κ ∂_r α

The spatial diffusion part is ν^r_NS = −κ ∂_r α.  The temporal
correction −v κ ∂_τ α enters through the conserved D_tau.
We store ν^r = −κ ∂_r α − κ u^r ∂_τ α to match J^r exactly.
"""
function bdnk_diffusion_step!(U, grid, τ, Δ, model, work;
                              ε_ν::Float64,
                              alpha_prev::Vector{Float64})
    L = hydro.layout(model)
    eos = model.eos
    Ntot = size(U, 2)
    ng = grid.nghost

    # ---- 1. Recover primitives and compute α = μ/T ----
    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D  = hydro.safe_div(U[L.iDtau, i], τ)
        Sr = U[L.iSr, i]
        E  = U[L.iE, i]

        nur_phys   = L.hasNur   ? hydro.phys_from_stored(U[L.iNur, i])   : 0.0
        Pi_phys    = L.hasPi    ? hydro.phys_from_stored(U[L.iPi, i])    : 0.0
        piR_phys   = L.hasPiR   ? hydro.phys_from_stored(U[L.iPiR, i])   : 0.0
        piEta_phys = L.hasPiEta ? hydro.phys_from_stored(U[L.iPiEta, i]) : 0.0

        T, μ, ur, n, e, P, ok = hydro.cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, piEta_phys,
            grid.rC[i], τ, eos;
            yT0=work.yT[i], φ0=work.phi[i], y0=work.y[i], maxit=40
        )

        if ok
            T = max(T, hydro.T_MIN)
            work.yT[i]    = log(T)
            work.mu[i]    = μ
            work.phi[i]   = (μ - hydro.hq_mass(eos)) / T
            work.y[i]     = asinh(ur)
            work.n[i]     = n
            work.P[i]     = P
            work.e[i]     = e
            work.ok[i]    = true
            uτ = sqrt(1 + ur^2)
            work.vC[i]    = hydro.safe_div(ur, uτ)
            work.alpha[i] = μ / T
        else
            work.ok[i]    = false
            work.vC[i]    = 0.0
            work.alpha[i] = 0.0
            work.n[i]     = 0.0
            work.P[i]     = 0.0
            work.e[i]     = 0.0
            work.mu[i]    = 0.0
            work.yT[i]    = log(hydro.T_MIN)
            work.y[i]     = 0.0
            work.phi[i]   = 0.0
        end
    end

    # Zero dissipatives at bad cells
    @inbounds for i in (ng+1):(Ntot-ng)
        if !work.ok[i] && L.hasNur
            U[L.iNur, i] = hydro.stored_from_phys(0.0)
        end
    end

    # ---- 2. Smooth α and compute ∂_r α ----
    @inbounds for g in 1:ng
        il = ng + 1 - g
        ir = ng + g
        work.alpha[il] = work.alpha[ir]
    end
    hydro.compute_field_grad_fv!(work.alphaF, work.gradAlpha, work.alpha, grid;
                                  limiter=hydro.mc_limiter)

    # ---- 3. Compute ν^r_BDNK at each cell ----
    DsT = model.kappa_coeff

    @inbounds for i in (ng+1):(Ntot-ng)
        work.ok[i] || continue

        Tl = exp(work.yT[i])
        n  = work.n[i]
        ur = sinh(work.y[i])
        uτ = sqrt(1 + ur^2)

        # Diffusion coefficient κ = DsT * n / T  (in fm units)
        κ = hydro.diff_kappa(Tl, work.mu[i], n, model)

        # ---- BDNK constitutive (ε_ν-independent after Landau projection) ----
        # From main2BDNK.jl: J^r = v(J^τ − κ ∂_τ α) − κ ∂_r α
        # With J^τ = n u^τ + v ν^r and J^r = n u^r + ν^r, solving for ν^r:
        #   ν^r = -κ u^τ² ∂_r α  −  κ u^r u^τ ∂_τ α
        dalpha_dt = Δ > hydro.TINY ? (work.alpha[i] - alpha_prev[i]) / Δ : 0.0
        νr_bdnk = -κ * uτ * (uτ * work.gradAlpha[i] + ur * dalpha_dt)

        # Soft projection: keep |ν^r| < n u^τ (physically admissible)
        νbound = max(n * uτ, 0.0)
        if νbound > 0.0
            νr_bdnk = νbound * tanh(νr_bdnk / (νbound + hydro.TINY))
        else
            νr_bdnk = 0.0
        end

        U[L.iNur, i] = hydro.stored_from_phys(νr_bdnk)
    end

    # ---- 4. Update α_prev for next step ----
    copyto!(alpha_prev, work.alpha)

    # ---- 5. Apply BCs ----
    hydro.apply_bc!(U, grid, τ, model)

    return nothing
end


# ============================================================================
# CFL with BDNK signal speed
# ============================================================================
function compute_dt_bdnk(work, grid, τ, model;
                         CFL::Float64=0.2, CFLτ::Float64=0.05,
                         ε_ν::Float64=0.0,
                         diff_dt_coeff::Float64=0.02)
    ng = grid.nghost
    amax = 1e-30
    kmax = 0.0
    DsT = model.kappa_coeff

    @inbounds for i in (ng+1):(length(work.yT)-ng)
        work.ok[i] || continue
        T  = exp(work.yT[i])
        μ  = work.mu[i]
        ur = sinh(work.y[i])
        uτ = sqrt(1 + ur^2)

        # Ideal + viscous wavespeeds (energy-momentum sector)
        λm, λp = hydro.wavespeeds_from_prim(T, μ, ur, model.eos)
        a = max(abs(λm), abs(λp))

        # BDNK signal speed: v_sig = √(κ / ε_ν)
        # The charge sector wave has maximum phase velocity v_sig.
        # For ε_ν = κ, v_sig = c = 1.
        # Boost formula for diffusion wavespeed in lab frame:
        v = hydro.safe_div(ur, uτ)
        if ε_ν > hydro.TINY
            v_sig = sqrt(hydro.diff_kappa(T, μ, work.n[i], model) / ε_ν)
            v_sig = min(v_sig, 0.999999)
            λp_bdnk = (v + v_sig) / (1 + v * v_sig + hydro.TINY)
            λm_bdnk = (v - v_sig) / (1 - v * v_sig + hydro.TINY)
            a = max(a, abs(λp_bdnk), abs(λm_bdnk))
        else
            # ε_ν ≈ 0 means pure parabolic → CFL limited by diffusion coefficient
            a = max(a, 0.999999)
        end

        # Diffusion CFL: dt < diff_dt_coeff * dr² / (D u^τ²)
        κ = hydro.diff_kappa(T, μ, work.n[i], model)
        kmax = max(kmax, κ * uτ^2)

        amax = max(amax, a)
    end

    dt = min(CFL * grid.dr / (amax + hydro.TINY), CFLτ * τ)

    if kmax > 0
        dt = min(dt, diff_dt_coeff * grid.dr^2 / (kmax + hydro.TINY))
    end

    return dt
end


# ============================================================================
# Main BDNK simulation driver
# ============================================================================
function run_sim_bdnk(;
    outdir::String,
    Nr::Int=300, rmax::Float64=25.0, nghost::Int=3,
    τ0::Float64=0.4, τfinal::Float64=15.0,
    CFL::Float64=0.2, CFLτ::Float64=0.05,
    time_integrator::Symbol=:ssprk2,
    dump_dt::Float64=0.1,
    log_every::Int=50,
    Emin::Float64=hydro.E_FLOOR,
    χ::Float64=hydro.χ_SrE,
    init_csv::Union{Nothing,String}=nothing,
    fugacity_kind::Symbol=:alpha,
    # BDNK parameters
    DsT::Float64=0.24,
    ε_ν_factor::Symbol=:kappa,   # :kappa (ε_ν = κ, v_sig = c) or :is_match (ε_ν = χ τ_D)
    ε_ν_value::Float64=NaN,      # explicit override (if not NaN, uses this)
    diff_dt_coeff::Float64=0.02,
    # viscosity (energy-momentum sector, same as IS)
    enable_shear::Bool=false,
    enable_bulk::Bool=false,
    eta_over_s::Float64=0.0,
    zeta_over_s::Float64=0.0,
    tauShear_coeff::Float64=0.2,
    tauPi_coeff::Float64=1.0,
    # EOS
    eos=hydro.LatticeHRGEOS())

    hydro.setup_logger!(level=Logging.Info)

    grid = hydro.make_grid(Nr; rmax=rmax, nghost=nghost)

    # Layout: keep nur slot (stores BDNK ν^r) but disable IS relaxation
    layout_bdnk = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta];
                                     odd_syms=[:Sr,:nur])

    shear_model = enable_shear ? hydro.QGPViscosity(eta_over_s, tauShear_coeff) : hydro.ZeroViscosity()
    bulk_model  = enable_bulk  ? hydro.SimpleBulkViscosity(zeta_over_s, tauPi_coeff) : hydro.ZeroBulkViscosity()

    # IMPORTANT: enable_diff=false disables IS relaxation in relax_dissipative!
    # We handle diffusion ourselves via bdnk_diffusion_step!
    model = hydro.IdealDiffViscModel(
        eos, layout_bdnk, hydro.IdealPrimRec(),
        # charge diffusion (disabled for IS, but kappa_coeff used for BDNK)
        false,        # enable_diff = false
        :alpha,       # diffusion_drive
        DsT,          # kappa_coeff (used by diff_kappa)
        1.0,          # tauN_coeff (unused since enable_diff=false)
        0.0,          # deltaN_factor
        diff_dt_coeff,# diff_dt_coeff
        0.3,          # shear_dt_coeff
        0.3,          # bulk_dt_coeff
        -1.0,         # nur_clip_factor
        0.0,          # alpha_filter_eps
        0.0,          # nur_filter_eps
        0.0,          # alpha_smooth_len
        0.0,          # nur_smooth_len
        false,        # do_soft_project_nur (we do it in bdnk_diffusion_step!)
        false,        # do_axis_project_nur
        2,            # axis_project_nfit
        false,        # advect_nur = false (don't FV-advect ν^r)
        false,        # relax_advect_nur
        # viscosity
        enable_shear, enable_bulk, shear_model, bulk_model,
        0.0,          # deltaPi_factor
        0.0,          # deltaShear_factor
        # second-order couplings (all off)
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,    # visc_filter_eps, visc_smooth_len
        -1.0, -1.0,  # Pi_clip_factor, pi_clip_factor
        false, false, # advect_Pi, advect_pi
        true, true    # relax_advect_Pi, relax_advect_pi
    )

    Nvars = length(layout_bdnk.names)
    Ntot  = Nr + 2*nghost
    U = zeros(Nvars, Ntot)
    hydro.initialize!(U, grid, τ0, model;
                       init_csv=init_csv,
                       fugacity_kind=fugacity_kind,
                       Emin=Emin, χ=χ)
    work = hydro.make_work(U)
    diag = hydro.DiagCounters()

    # α storage for temporal correction (from previous step)
    alpha_prev = zeros(Ntot)

    # Seed work caches
    hydro.prime_work_from_U!(work, U, grid, τ0, model)

    # Initialize α_prev from the initial state
    @inbounds for i in 1:Ntot
        alpha_prev[i] = work.alpha[i]
    end

    # Determine ε_ν value
    local ε_ν_use::Float64
    if isfinite(ε_ν_value)
        ε_ν_use = ε_ν_value
    else
        # Compute a representative κ for the ε_ν factor
        i_ref = nghost + div(Nr, 4)  # representative interior cell
        T_ref = exp(work.yT[i_ref])
        n_ref = work.n[i_ref]
        μ_ref = work.mu[i_ref]
        κ_ref = hydro.diff_kappa(T_ref, μ_ref, n_ref, model)

        if ε_ν_factor === :kappa
            # Minimal causal: ε_ν = κ → v_sig = c
            # κ varies in space, so we set ε_ν dynamically in the step
            ε_ν_use = κ_ref
            @info "BDNK ε_ν = κ (minimal causal, v_sig = c)" κ_ref=κ_ref
        elseif ε_ν_factor === :is_match
            # Match IS gap: ε_ν = χ τ_D
            τ_D = hydro.diff_tauN(T_ref, μ_ref, model)
            _, n_tmp, _ = hydro.eos_Pne(T_ref, μ_ref, eos)
            χ_ref = hydro.safe_div(n_tmp, T_ref)
            ε_ν_use = χ_ref * τ_D
            @info "BDNK ε_ν = χ τ_D (IS-matched)" ε_ν=ε_ν_use τ_D=τ_D χ_ref=χ_ref
        else
            error("Unknown ε_ν_factor=$ε_ν_factor. Use :kappa or :is_match")
        end
    end

    # Set initial ν^r_BDNK from the constitutive relation
    bdnk_diffusion_step!(U, grid, τ0, 1e-6, model, work;
                          ε_ν=ε_ν_use, alpha_prev=alpha_prev)

    τ = τ0
    it = 0
    next_dump = τ0

    mkpath(outdir)
    hydro.clear_dir!(outdir)
    hydro.write_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)),
                              U, grid, τ, model)

    @info "Start BDNK" threads=Threads.nthreads() outdir=outdir Nr=Nr dr=grid.dr τ0=τ0 τfinal=τfinal DsT=DsT ε_ν=ε_ν_use time_integrator=time_integrator enable_shear=enable_shear enable_bulk=enable_bulk

    while τ < τfinal - 1e-12
        hydro.diag_reset_last!(diag)

        # CFL including BDNK signal speed
        Δτ = compute_dt_bdnk(work, grid, τ, model;
                              CFL=CFL, CFLτ=CFLτ,
                              ε_ν=ε_ν_use,
                              diff_dt_coeff=diff_dt_coeff)
        if τ + Δτ > τfinal
            Δτ = τfinal - τ
        end

        # SSPRK2/3 step (ideal + viscous, NO IS diffusion relaxation)
        if time_integrator == :ssprk2
            Δused = hydro.step_ssprk2!(U, grid, τ, Δτ, model, work;
                                        Emin=Emin, χ=χ, diag=diag)
        elseif time_integrator == :ssprk3
            Δused = hydro.step_ssprk3!(U, grid, τ, Δτ, model, work;
                                        Emin=Emin, χ=χ, diag=diag)
        else
            error("Unknown time_integrator=$time_integrator")
        end

        τ += Δused
        it += 1

        # ---- BDNK diffusion step (replaces IS relaxation) ----
        bdnk_diffusion_step!(U, grid, τ, Δused, model, work;
                              ε_ν=ε_ν_use, alpha_prev=alpha_prev)

        # Post-step enforcement
        hydro.enforce_floors!(U, grid, τ, model; Emin=Emin, diag=diag)
        hydro.enforce_Sr_energy_constraint!(U, grid, model; χ=χ, P=work.P, mask=nothing, diag=diag)
        hydro.apply_bc!(U, grid, τ, model)

        if τ >= next_dump - 1e-12
            fname = joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ))
            hydro.write_snapshot_csv(fname, U, grid, τ, model)
            Q = hydro.charge_integral_Dtau(U, grid, model)
            @info "dump" τ=τ file=fname it=it Q_Dtau=Q
            next_dump += dump_dt
        end

        if (it % log_every) == 0
            Q = hydro.charge_integral_Dtau(U, grid, model)
            @info "progress" τ=τ Δτ=Δused it=it Q_Dtau=Q
        end
    end

    Q = hydro.charge_integral_Dtau(U, grid, model)
    @info "Done BDNK" τ=τ it=it outdir=outdir Q_Dtau=Q

    # ---- Serialization (same format as main2.jl) ----
    _fname_float(x; digits=3) = replace(replace(@sprintf("%.*f", digits, Float64(x)), "." => "p"), "-" => "m")
    param_stamp = "tau0_$(_fname_float(τ0))_tauf_$(_fname_float(τfinal))_rmax_$(_fname_float(rmax))_nr_$(Nr)_dst_$(_fname_float(DsT))"
    label = "BDNK_full_FV"
    tag = "BDNK_full_FV_$(param_stamp)"

    currents_path = hydro.write_hydro_currents_jld2(outdir; outdir=outdir, tag=tag)

    plot_splines_dir = normpath(joinpath(@__DIR__, "..", "Plot", "splines"))
    splines_path = try
        hydro.write_hydro_currents_splines_jld2(
            outdir;
            outdir=plot_splines_dir,
            filename="BDNK_full_FV_$(param_stamp).jld2",
            tag=label,
            overwrite=true,
            kx=1, ky=3,
            store_phi_splines=false,
        )
    catch e
        @warn "Spline serialization failed (non-finite values in snapshots?)" exception=e
        nothing
    end
    @info "Wrote BDNK full-FV outputs" currents_path splines_path

    return nothing
end


# ============================================================================
# Entry point
# ============================================================================
function main_bdnk()
    outdir = get(ENV, "HYDRO_OUTDIR", joinpath(@__DIR__, "snapshots", "BDNK"))

    τ0     = parse(Float64, get(ENV, "TAU0",     "0.4"))
    τfinal = parse(Float64, get(ENV, "TAUFINAL", "15.0"))
    Nr     = parse(Int,     get(ENV, "NR",       "300"))
    rmax   = parse(Float64, get(ENV, "RMAX",     "25.0"))
    DsT    = parse(Float64, get(ENV, "DS_T",     "0.24"))
    CFL    = parse(Float64, get(ENV, "CFL",      "0.2"))
    CFLτ   = parse(Float64, get(ENV, "CFLTAU",   "0.05"))

    dump_dt = parse(Float64, get(ENV, "DUMP_DT", "0.1"))

    init_csv = get(ENV, "INIT_CSV", nothing)
    if init_csv !== nothing && !isfile(init_csv)
        init_csv = nothing
    end

    eps_nu_str = lowercase(get(ENV, "EPS_NU", "kappa"))
    ε_ν_factor = eps_nu_str == "is_match" ? :is_match : :kappa
    ε_ν_value  = let s = get(ENV, "EPS_NU_VALUE", "")
        isempty(s) ? NaN : parse(Float64, s)
    end

    enable_shear = parse(Int, get(ENV, "ENABLE_SHEAR", "0")) != 0
    enable_bulk  = parse(Int, get(ENV, "ENABLE_BULK",  "0")) != 0
    eta_over_s   = parse(Float64, get(ENV, "ETA_OVER_S",  "0.0"))
    zeta_over_s  = parse(Float64, get(ENV, "ZETA_OVER_S", "0.0"))

    time_int_str = lowercase(get(ENV, "TIME_INTEGRATOR", "ssprk2"))
    time_integrator = time_int_str == "ssprk3" ? :ssprk3 : :ssprk2

    run_sim_bdnk(;
        outdir=outdir,
        Nr=Nr, rmax=rmax, nghost=3,
        τ0=τ0, τfinal=τfinal,
        CFL=CFL, CFLτ=CFLτ,
        time_integrator=time_integrator,
        dump_dt=dump_dt,
        init_csv=init_csv,
        DsT=DsT,
        ε_ν_factor=ε_ν_factor,
        ε_ν_value=ε_ν_value,
        diff_dt_coeff=0.02,
        enable_shear=enable_shear,
        enable_bulk=enable_bulk,
        eta_over_s=eta_over_s,
        zeta_over_s=zeta_over_s,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main_bdnk()
end
