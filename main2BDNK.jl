#!/usr/bin/env julia
# ==============================================================================
# main2BDNK.jl — BDNK current-only solver on fixed background fields
#
# Analogue of main2.jl (IS current-only) but with BDNK constitutive relation
# instead of Israel-Stewart relaxation for ν^r.
#
# FP-matched BDNK constitutive relation (FP_Hydro_matching eq:BDNK_nu_matched):
#   ν^μ = σ_α ∇^μ α  +  σ_T ∇^μ ln T  +  σ_a u̇^μ
#
# with matched coefficients (eq:BDNK_coeffs):
#   σ_α = κ_n = D_s n₀
#   σ_T = κ_n z K₃(z)/K₂(z)       (Soret / thermo-diffusion)
#   σ_a = −κ_n / T                 (inertial / acceleration coupling)
#
# In Milne 1+1D (mostly-plus metric, boost-invariance):
#   ν^r = −κ [u^r u^τ ∂_τα + (u^τ)² ∂_rα]
#         −σ_T [u^r u^τ ∂_τ(ln T) + (u^τ)² ∂_r(ln T)]
#         +σ_a [u^τ ∂_τu^r + u^r ∂_ru^r]
#
# Note: in 1+1D boost-invariant Milne, ν^τ = (u^r/u^τ) ν^r exactly,
# so J^r = v J^τ + ν^r/(u^τ)² as in IS.
#
# Legacy modes (:kappa, :is_match) retain the old 2-parameter ansatz
#   ν^μ = −κ Δ^{μν} ∂_ν α  +  ε_ν u^μ (u^ν ∂_ν α)
#
# The charge q = r J^τ is evolved with the same FV transport as main2.jl.
# ν^r is set algebraically at each step (no ODE, no relaxation).
# ==============================================================================

include("main2.jl")

module hydro_current_bdnk

import ..hydro_current
import ..hydro_current:
    CurrentGrid1D, CurrentWorkspace1D, BackgroundFields, Grid1D,
    load_background, T_bg, ur_bg, v_bg, alpha_bg,
    dt_alpha_bg, dr_alpha_bg,
    dt_T_bg, dr_T_bg,
    dt_ur_bg, dr_ur_bg,
    _u_from_ur, _eval_spline_clamped,
    diff_kappa_bg, diff_tauN_bg, diff_sigmaT_bg, diff_sigmaa_bg,
    transport_update_q!, enforce_regularity_bc!,
    initial_state_from_background,
    reconstruct_snapshot, write_current_snapshot_csv,
    build_eos,
    make_grid,
    clear_dir!, write_hydro_currents_jld2, write_hydro_currents_splines_jld2,
    setup_logger!,
    LatticeHRGEOS, ConformalHQEOS, eos_Pne,
    T_MIN, TINY, fmGeV, hq_mass,
    stored_from_phys, phys_from_stored

using Printf
using Logging


# ============================================================================
# BDNK flux builder  (replaces build_q_flux! from main2.jl)
#
# In 1+1D boost-invariant Milne, ν^τ = (u^r/u^τ) ν^r for ALL three BDNK
# terms (σ_α, σ_T, σ_a).  Therefore:
#   J^r = v J^τ + ν^r/(u^τ)²
#
# Substituting the constitutive relation and dividing by (u^τ)²:
#   ν^r/(u^τ)² = −κ (v ∂_τα + ∂_rα)
#                −σ_T (v ∂_τ ln T + ∂_r ln T)
#                +(σ_a/u^τ)(∂_τu^r + v ∂_ru^r)
#
# The advective part (v J^τ) is upwinded; all constitutive terms are
# centered at faces from background splines.
# ============================================================================
function build_q_flux_bdnk!(
    flux_q::Vector{Float64},
    q::Vector{Float64},
    τ::Float64,
    grid::CurrentGrid1D,
    bg::BackgroundFields;
    DsT::Float64,
    T_floor::Float64,
    eos,
    ε_ν_mode::Symbol,
)
    Nr = length(grid.r)
    dr = grid.dr[1]   # uniform grid
    flux_q[1] = 0.0   # axis BC: zero flux

    @inbounds for i in 2:Nr
        rface = grid.rF[i]
        vface = v_bg(bg, τ, rface)

        # ---- advection: upwind J^τ ----
        Jtau_up = vface >= 0.0 ?
            q[i - 1] / max(grid.r[i - 1], 1e-12) :
            q[i]     / max(grid.r[i],     1e-12)

        # ---- face thermodynamics ----
        T_face = max(T_bg(bg, τ, rface), T_floor)
        α_face = alpha_bg(bg, τ, rface)
        μ_face = α_face * T_face
        n_face = bg.n_spline === nothing ?
            eos_Pne(T_face, μ_face, eos)[2] :
            _eval_spline_clamped(bg.n_spline, rface, τ,
                                 bg.r_grid, bg.t_grid)
        κ_face = diff_kappa_bg(T_face, α_face, n_face, DsT, eos)

        # ∂_r α at face via centered difference of cell-centre background α
        α_L = alpha_bg(bg, τ, grid.r[i - 1])
        α_R = alpha_bg(bg, τ, grid.r[i])
        drα_face = (α_R - α_L) / dr

        # ∂_τ α at face from background spline
        dtα_face = dt_alpha_bg(bg, τ, rface)

        if ε_ν_mode === :fp_matched
            # Full FP-matched BDNK flux:
            # J^r = v J^τ − κ(v ∂_τα + ∂_rα)
            #             − σ_T(v ∂_τ ln T + ∂_r ln T)
            #             + (σ_a/u^τ)(∂_τu^r + v ∂_ru^r)
            σ_T = diff_sigmaT_bg(T_face, α_face, n_face, DsT, eos)
            σ_a = diff_sigmaa_bg(T_face, α_face, n_face, DsT, eos)

            urface = ur_bg(bg, τ, rface)
            uτface = sqrt(1.0 + urface^2)

            # ∂ ln T at face
            T_L = max(T_bg(bg, τ, grid.r[i - 1]), T_floor)
            T_R = max(T_bg(bg, τ, grid.r[i]), T_floor)
            dr_lnT_face = (log(T_R) - log(T_L)) / dr
            dt_lnT_face = dt_T_bg(bg, τ, rface) / T_face

            # ∂ u^r at face
            dtur_face = dt_ur_bg(bg, τ, rface)
            drur_face = dr_ur_bg(bg, τ, rface)

            Jr_face = vface * Jtau_up -
                      κ_face * (vface * dtα_face + drα_face) -
                      σ_T * (vface * dt_lnT_face + dr_lnT_face) +
                      (σ_a / uτface) * (dtur_face + vface * drur_face)
        elseif ε_ν_mode === :density_frame
            # Density frame: μ is fixed from the on-slice density, so the comoving
            # ∂_τα term is eliminated and only the spatial parabolic flux survives.
            # Using J^r = v J^τ + ν^r/(u^τ)^2 with ν^r = -κ (u^τ)^2 ∂_rα this is
            #   J^r = v J^τ − κ ∂_rα ,
            # i.e. the legacy alpha-only flux with the ∂_τα piece dropped.
            # See Tex/DensityFrame/df_fp_derivation.tex.
            Jr_face = vface * Jtau_up - κ_face * drα_face
        else
            # Legacy 2-parameter mode: J^r = v(J^τ − κ ∂_τα) − κ ∂_rα
            Jr_face = vface * (Jtau_up - κ_face * dtα_face) - κ_face * drα_face
        end

        flux_q[i] = rface * Jr_face
    end

    # ---- outer boundary: advection only, zero-gradient diffusion ----
    rface = grid.rF[Nr + 1]
    vface = v_bg(bg, τ, rface)
    Jtau_up = q[Nr] / max(grid.r[Nr], 1e-12)
    flux_q[Nr + 1] = rface * vface * Jtau_up
    return nothing
end

# ============================================================================
# BDNK algebraic step for ν^r  (replaces step_nur_expanded! from main2.jl)
# ============================================================================
"""
    step_nur_bdnk!(nu_r, τ, grid, bg; DsT, T_floor, eos, ε_ν_mode)

Set ν^r from the BDNK constitutive relation (algebraic, no time derivative).

**:fp_matched** (default) — full FP-matched BDNK (eq:BDNK_nu_matched):

    ν^r = −κ [u^r u^τ ∂_τα + (u^τ)² ∂_rα]
          −σ_T [u^r u^τ ∂_τ(ln T) + (u^τ)² ∂_r(ln T)]
          +σ_a [u^τ ∂_τu^r + u^r ∂_ru^r]

with σ_α = κ_n, σ_T = κ_n z K₃/K₂, σ_a = −κ_n/T.

**:kappa** — legacy 2-parameter, ε_ν = κ  (v_sig = c):
    ν^r = −κ (u^τ)² ∂_r α

**:is_match** — legacy 2-parameter, ε_ν = χ τ_D  (IS gap match)
"""
function step_nur_bdnk!(nu_r::Vector{Float64}, τ::Float64,
                        grid::CurrentGrid1D, bg::BackgroundFields;
                        DsT::Float64, T_floor::Float64, eos,
                        ε_ν_mode::Symbol = :fp_matched)
    Nr = length(grid.r)

    @inbounds for i in 1:Nr
        r = grid.r[i]
        T = max(T_bg(bg, τ, r), T_floor)
        α = alpha_bg(bg, τ, r)
        μ = α * T
        nbg = bg.n_spline === nothing ?
              eos_Pne(T, μ, eos)[2] :
              _eval_spline_clamped(bg.n_spline, r, τ, bg.r_grid, bg.t_grid)
        κ = diff_kappa_bg(T, α, nbg, DsT, eos)

        uτ, ur, _ = _u_from_ur(ur_bg(bg, τ, r))
        drα = dr_alpha_bg(bg, τ, r)
        dtα = dt_alpha_bg(bg, τ, r)

        νr = if ε_ν_mode === :fp_matched
            # Full FP-matched BDNK (Milne mostly-plus sign convention)
            σ_T = diff_sigmaT_bg(T, α, nbg, DsT, eos)
            σ_a = diff_sigmaa_bg(T, α, nbg, DsT, eos)

            dt_lnT = dt_T_bg(bg, τ, r) / T
            dr_lnT = dr_T_bg(bg, τ, r) / T
            dtur = dt_ur_bg(bg, τ, r)
            drur = dr_ur_bg(bg, τ, r)

            -κ   * (ur * uτ * dtα    + uτ^2 * drα) -
             σ_T * (ur * uτ * dt_lnT + uτ^2 * dr_lnT) +
             σ_a * (uτ * dtur + ur * drur)
        elseif ε_ν_mode === :density_frame
            # Density frame: spatial gradient only (no comoving ∂_τα), no σ_T/σ_a.
            #   ν^r = -κ (u^τ)^2 ∂_rα      (Tex/DensityFrame/df_fp_derivation.tex)
            -κ * uτ^2 * drα
        else
            # Legacy 2-parameter ansatz:
            #   ν^r = (ε_ν − κ) u^r u^τ ∂_τα + (ε_ν u^r² − κ u^τ²) ∂_rα
            ε_ν = if ε_ν_mode === :kappa
                κ
            elseif ε_ν_mode === :is_match
                τn = diff_tauN_bg(T, α, DsT, eos)
                χ = nbg / max(T, T_MIN)
                χ * τn
            else
                κ
            end
            (ε_ν - κ) * ur * uτ * dtα + (ε_ν * ur^2 - κ * uτ^2) * drα
        end

        # Soft bound: |ν^r| < n u^τ  (physical admissibility)
        νbound = max(nbg * uτ, 0.0)
        if νbound > 0.0
            νr = νbound * tanh(νr / (νbound + TINY))
        else
            νr = 0.0
        end

        nu_r[i] = νr
    end

    # Regularity at axis
    if Nr >= 2
        nu_r[1] = nu_r[2] * (grid.r[1] / grid.r[2])
    elseif Nr == 1
        nu_r[1] = 0.0
    end

    return nothing
end


# ============================================================================
# Single time step (transport + BDNK ν^r)
#
# Order:  ν^r(τ) → flux(τ) → transport q → regularity BC
# Computing ν^r BEFORE the flux ensures the diffusion current is evaluated
# at the correct time level (unlike IS, where backward Euler is implicit).
# ============================================================================
function step_system_current_only_bdnk!(
    q::Vector{Float64},
    nu_r::Vector{Float64},
    τ::Float64,
    dt::Float64,
    grid::CurrentGrid1D,
    ws::CurrentWorkspace1D,
    bg::BackgroundFields;
    DsT::Float64,
    T_floor::Float64,
    eos,
    ε_ν_mode::Symbol,
)
    # 1. Build BDNK flux with centered diffusion at current τ
    build_q_flux_bdnk!(ws.flux_q, q, τ, grid, bg;
                       DsT=DsT, T_floor=T_floor, eos=eos, ε_ν_mode=ε_ν_mode)
    # 2. Forward-Euler transport of q
    transport_update_q!(q, τ, dt, grid, ws)
    # 3. Set ν^r from BDNK constitutive at new τ (for snapshots / next step)
    step_nur_bdnk!(nu_r, τ, grid, bg;
                   DsT=DsT, T_floor=T_floor, eos=eos, ε_ν_mode=ε_ν_mode)
    # 4. Regularity at axis
    enforce_regularity_bc!(q, nu_r, grid)
    return nothing
end


# ============================================================================
# Solver loop
# ============================================================================
function solve_current_bdnk(
    grid::CurrentGrid1D,
    q0::Vector{Float64},
    nu0::Vector{Float64},
    τ0::Float64,
    τf::Float64,
    bg::BackgroundFields;
    CFL::Float64,
    CFLτ::Float64,
    CFL_diff::Float64 = 0.4,
    save_dt::Float64,
    log_every::Int = 50,
    DsT::Float64,
    T_floor::Float64,
    eos,
    ε_ν_mode::Symbol = :fp_matched,
)
    Nr = length(grid.r)
    length(q0) == Nr || error("q0 must have length $Nr")
    length(nu0) == Nr || error("nu0 must have length $Nr")

    q    = copy(q0)
    nu_r = copy(nu0)
    τ    = τ0
    it   = 0
    next_dump = τ0
    ws   = CurrentWorkspace1D(grid)

    τs  = Float64[τ]
    qs  = Vector{Float64}[copy(q)]
    nus = Vector{Float64}[copy(nu_r)]

    # Initialize ν^r from BDNK constitutive relation
    step_nur_bdnk!(nu_r, τ, grid, bg;
                   DsT=DsT, T_floor=T_floor, eos=eos, ε_ν_mode=ε_ν_mode)

    while τ < τf - 1e-12
        dr = minimum(grid.dr)

        # ---- Advection CFL (background flow speed) ----
        vmax = maximum(abs.(v_bg.(Ref(bg), Ref(τ), grid.rF[2:end])))

        # ---- BDNK characteristic speed ----
        # For :fp_matched and :kappa, v_sig = 1 (maximally causal).
        v_sig = ε_ν_mode === :is_match ? 0.99 : 1.0
        λmax = max(vmax, v_sig, 1e-8)
        dt_adv = CFL * dr / λmax

        # ---- Parabolic (diffusion) CFL:  dt < C dr²/(2 D_max) ----
        # Conservative: use max(κ, σ_T) as effective diffusion coefficient.
        # σ_T = κ z K₃/K₂ ≥ κ for z > 0.
        D_max = 0.0
        @inbounds for i in 1:Nr
            ri = grid.r[i]
            Ti = max(T_bg(bg, τ, ri), T_floor)
            αi = alpha_bg(bg, τ, ri)
            μi = αi * Ti
            ni = bg.n_spline === nothing ?
                eos_Pne(Ti, μi, eos)[2] :
                _eval_spline_clamped(bg.n_spline, ri, τ,
                                     bg.r_grid, bg.t_grid)
            κi = diff_kappa_bg(Ti, αi, ni, DsT, eos)
            Di = ε_ν_mode === :fp_matched ? diff_sigmaT_bg(Ti, αi, ni, DsT, eos) : κi
            D_max = max(D_max, Di)
        end
        dt_diff = D_max > TINY ? CFL_diff * dr^2 / (2.0 * D_max) : Inf

        # ---- Proper-time CFL ----
        dt_tau = CFLτ * τ

        Δτ = min(dt_adv, dt_diff, dt_tau)
        if τ + Δτ > τf
            Δτ = τf - τ
        end

        τ_eval = τ + Δτ
        step_system_current_only_bdnk!(q, nu_r, τ_eval, Δτ, grid, ws, bg;
                                       DsT=DsT, T_floor=T_floor, eos=eos,
                                       ε_ν_mode=ε_ν_mode)
        τ = τ_eval
        it += 1

        if τ >= next_dump - 1e-12
            push!(τs, τ)
            push!(qs, copy(q))
            push!(nus, copy(nu_r))
            next_dump += save_dt
        end

        if (it % log_every) == 0
            @info "BDNK current-only progress" τ=τ Δτ=Δτ it=it vmax=vmax
        end
    end

    if τs[end] != τ
        push!(τs, τ)
        push!(qs, copy(q))
        push!(nus, copy(nu_r))
    end

    return τs, qs, nus
end


# ============================================================================
# Driver
# ============================================================================
function run_current_background_bdnk(;
    outdir::String,
    Nr::Int = 300,
    rmax::Float64 = 25.0,
    nghost::Int = 1,
    τ0::Float64 = 0.4,
    τfinal::Float64 = 15.0,
    CFL::Float64 = 0.3,
    CFLτ::Float64 = 0.05,
    dump_dt::Float64 = 0.1,
    log_every::Int = 50,
    background_file::String,
    DsT::Float64 = 0.24,
    T_floor::Float64 = 1e-6,
    eos = LatticeHRGEOS(),
    ε_ν_mode::Symbol = :fp_matched,
    run_label::String = "",
)
    _fname_float(x::Real; digits::Int=3) = replace(replace(@sprintf("%.*f", digits, Float64(x)), "." => "p"), "-" => "m")
    function _sanitize_filename_base(s::AbstractString)
        t = strip(String(s))
        endswith(lowercase(t), ".jld2") && (t = t[1:(end - 5)])
        isempty(t) && return ""
        t = replace(t, r"\s+" => "_")
        t = replace(t, r"[^A-Za-z0-9._-]+" => "_")
        t = replace(t, r"_+" => "_")
        t = strip(t, '_')
        return t
    end

    bg = load_background(background_file)
    rmax_use = min(rmax, last(bg.r_grid))
    grid_full = make_grid(Nr; rmax=rmax_use, nghost=nghost)
    grid = CurrentGrid1D(grid_full)

    bg.α_spline === nothing && error("Background file needs α_spline for BDNK constitutive relation")

    # ---- BDNK-consistent initial condition ----
    # Use n from background, compute ν^r from BDNK constitutive at τ₀,
    # then set q₀ = r (n u^τ + ν^τ_BDNK).
    q0  = Vector{Float64}(undef, length(grid.r))
    nu0 = Vector{Float64}(undef, length(grid.r))

    # First pass: set ν^r from BDNK constitutive
    step_nur_bdnk!(nu0, τ0, grid, bg;
                   DsT=DsT, T_floor=T_floor, eos=eos, ε_ν_mode=ε_ν_mode)

    # Second pass: build q₀ = r J^τ with BDNK ν^τ
    @inbounds for i in eachindex(grid.r)
        r = grid.r[i]
        T0 = T_bg(bg, τ0, r)
        α0 = alpha_bg(bg, τ0, r)
        μ0 = α0 * T0
        n0 = bg.n_spline === nothing ?
            eos_Pne(T0, μ0, eos)[2] :
            _eval_spline_clamped(bg.n_spline, r, τ0, bg.r_grid, bg.t_grid)
        ur0 = ur_bg(bg, τ0, r)
        uτ0 = sqrt(1.0 + ur0^2)
        # ν^τ = (u^r / u^τ) ν^r  (exact in 1+1D boost-inv. Milne for all BDNK terms)
        ντ0 = uτ0 > 0.0 ? (ur0 / uτ0) * nu0[i] : 0.0
        Jτ0 = n0 * uτ0 + ντ0
        q0[i] = r * Jτ0
    end

    mkpath(outdir)
    clear_dir!(outdir)
    write_current_snapshot_csv(
        joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ0)),
        q0, nu0, τ0, grid, bg; DsT=DsT, T_floor=T_floor, eos=eos)

    @info "Start BDNK current-only background evolution" outdir=outdir Nr=Nr rmax=rmax_use τ0=τ0 τfinal=τfinal CFL=CFL CFLτ=CFLτ DsT=DsT ε_ν_mode=ε_ν_mode background_file=background_file

    τs, qs, nus = solve_current_bdnk(grid, q0, nu0, τ0, τfinal, bg;
        CFL=CFL, CFLτ=CFLτ, save_dt=dump_dt, log_every=log_every,
        DsT=DsT, T_floor=T_floor, eos=eos, ε_ν_mode=ε_ν_mode)

    for (τ, q, nu_r) in zip(τs[2:end], qs[2:end], nus[2:end])
        write_current_snapshot_csv(
            joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)),
            q, nu_r, τ, grid, bg; DsT=DsT, T_floor=T_floor, eos=eos)
    end

    # ---- Serialization (same format as main2.jl) ----
    param_stamp = "tau0_$(_fname_float(τ0))_tauf_$(_fname_float(τfinal))_rmax_$(_fname_float(rmax_use))_nr_$(Nr)_dst_$(_fname_float(DsT))"

    label_clean = strip(run_label)
    label_for_plots = isempty(label_clean) ? "BDNK current-only" : label_clean
    file_base = isempty(label_clean) ? "BDNKCurrentOnly" : _sanitize_filename_base(label_clean)
    isempty(file_base) && (file_base = "BDNKCurrentOnly")

    currents_tag = _sanitize_filename_base(label_for_plots)
    isempty(currents_tag) && (currents_tag = "BDNK_current_only")
    currents_tag *= "_" * param_stamp
    currents_path = write_hydro_currents_jld2(outdir; outdir=outdir, tag=currents_tag)

    plot_splines_dir = normpath(joinpath(@__DIR__, "..", "Plot", "splines"))
    splines_path = write_hydro_currents_splines_jld2(
        outdir;
        outdir=plot_splines_dir,
        filename="$(file_base)_$(param_stamp).jld2",
        tag=label_for_plots,
        overwrite=true,
        kx=1, ky=3,
        store_phi_splines=false,
    )
    @info "Wrote BDNK current-only outputs" currents_path splines_path

    return nothing
end


# ============================================================================
# Entry point
# ============================================================================
function main()
    setup_logger!(level=Logging.Info)

    env_float(name::String, default::Float64) = haskey(ENV, name) ? parse(Float64, ENV[name]) : default
    env_int(name::String, default::Int) = haskey(ENV, name) ? parse(Int, ENV[name]) : default
    env_str(name::String, default::String) = haskey(ENV, name) ? String(ENV[name]) : default

    background_file = env_str("BACKGROUND_JLD2",
        normpath(joinpath(@__DIR__, "..", "LangevInMedium.jl", "src", "data", "Fluidum_MIS_HQ.jld2")))
    outdir = normpath(env_str("HYDRO_OUTDIR",
        joinpath(@__DIR__, "snapshots", "current_only_bdnk")))

    eos = build_eos(env_str("EOS", "latticehrg"))

    τ0     = env_float("TAU0", 0.4)
    τfinal = env_float("TAUFINAL", 15.0)
    Nr     = env_int("NR", 300)
    rmax   = env_float("RMAX", 25.0)
    nghost = env_int("NGHOST", 1)
    CFL    = env_float("CFL", 0.3)
    CFLτ   = haskey(ENV, "CFLTAU") ? parse(Float64, ENV["CFLTAU"]) : 0.05
    dump_dt   = env_float("DUMP_DT", 0.1)
    log_every = env_int("LOG_EVERY", 50)
    DsT       = env_float("DS_T", 0.24)
    T_floor   = env_float("T_FLOOR", 1e-6)
    run_label  = env_str("RUN_LABEL", "")

    eps_nu_str = lowercase(env_str("EPS_NU", "kappa"))
    ε_ν_mode =
        eps_nu_str == "is_match"                       ? :is_match :
        eps_nu_str == "fp_matched"                     ? :fp_matched :
        eps_nu_str in ("density_frame", "df")          ? :density_frame :
                                                         :kappa

    run_current_background_bdnk(
        outdir=outdir,
        Nr=Nr, rmax=rmax, nghost=nghost,
        τ0=τ0, τfinal=τfinal,
        CFL=CFL, CFLτ=CFLτ,
        dump_dt=dump_dt, log_every=log_every,
        background_file=background_file,
        DsT=DsT, T_floor=T_floor,
        eos=eos, ε_ν_mode=ε_ν_mode,
        run_label=run_label,
    )
end

end # module hydro_current_bdnk


if abspath(PROGRAM_FILE) == @__FILE__
    hydro_current_bdnk.main()
end
