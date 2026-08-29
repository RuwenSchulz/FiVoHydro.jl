#!/usr/bin/env julia
module hydro_current

using LinearAlgebra
using Printf
using Logging
using CSV
using Tables
using JLD2
using Interpolations
using SpecialFunctions

const _SRC = joinpath(@__DIR__, "src")
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "grid.jl"))
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "io.jl"))
include(joinpath(_SRC, "logging_setup.jl"))

const TWO_PI = 2π

struct CurrentGrid1D
    r::Vector{Float64}
    rF::Vector{Float64}
    dr::Vector{Float64}
end

function CurrentGrid1D(grid::Grid1D)
    ng = grid.nghost
    i0 = ng + 1
    iL = ng + grid.Nr
    r = Float64.(grid.rC[i0:iL])
    rF = Float64.(grid.rF[i0:(iL + 1)])
    dr = fill(grid.dr, grid.Nr)
    return CurrentGrid1D(r, rF, dr)
end

struct CurrentWorkspace1D
    flux_q::Vector{Float64}
    q_old::Vector{Float64}
    nu_old::Vector{Float64}
    nu_slope::Vector{Float64}
    # Self-consistent fugacity field (coupled α–ν mode): α recovered from the
    # evolved charm density each step, its r-slope, and its previous-step value
    # for ∂τα.  Unused when couple_alpha=false (background-α mode).
    alpha::Vector{Float64}
    alpha_old::Vector{Float64}
    alpha_slope::Vector{Float64}
end

function CurrentWorkspace1D(grid::CurrentGrid1D)
    Nr = length(grid.r)
    return CurrentWorkspace1D(
        zeros(Float64, Nr + 1),
        zeros(Float64, Nr),
        zeros(Float64, Nr),
        zeros(Float64, Nr),
        zeros(Float64, Nr),
        zeros(Float64, Nr),
        zeros(Float64, Nr),
    )
end

struct BackgroundFields
    r_grid::Vector{Float64}
    t_grid::Vector{Float64}
    T_spline::Any
    ur_spline::Any
    v_spline::Any
    α_spline::Any
    n_spline::Any
    nur_spline::Any
end

@inline function _u_from_v(v::Real)
    vC = clamp(Float64(v), -0.999999, 0.999999)
    uτ = 1.0 / sqrt(1.0 - vC^2 + 1e-12)
    ur = uτ * vC
    return uτ, ur, vC
end

@inline function _u_from_ur(ur::Real)
    urf = Float64(ur)
    uτ = sqrt(1.0 + urf^2)
    v = urf / uτ
    return uτ, urf, v
end

@inline function _eval_spline_clamped(spl, r::Real, τ::Real, r_grid::AbstractVector, t_grid::AbstractVector)
    rr = clamp(Float64(r), Float64(first(r_grid)), Float64(last(r_grid)))
    tt = clamp(Float64(τ), Float64(first(t_grid)), Float64(last(t_grid)))
    return Float64(spl(rr, tt))
end

function load_background(path::AbstractString)
    isfile(path) || error("Background file not found: $path")
    bg = jldopen(path, "r") do f
        # Support both old keys ("T_spline") and new keys ("T_spline1")
        _key(k) = haskey(f, k) ? k : haskey(f, k*"1") ? k*"1" : k
        BackgroundFields(
            Float64.(f["r_grid"]),
            Float64.(f["t_grid"]),
            f[_key("T_spline")],
            haskey(f, "ur_spline") || haskey(f, "ur_spline1") ? f[_key("ur_spline")] : nothing,
            haskey(f, "v_spline") || haskey(f, "v_spline1") ? f[_key("v_spline")] : nothing,
            haskey(f, "α_spline") || haskey(f, "α_spline1") ? f[_key("α_spline")] : nothing,
            haskey(f, "n_spline") || haskey(f, "n_spline1") ? f[_key("n_spline")] : nothing,
            haskey(f, "nur_spline") || haskey(f, "nur_spline1") ? f[_key("nur_spline")] : nothing,
        )
    end
    bg.ur_spline === nothing && bg.v_spline === nothing && error("Background file needs either ur_spline or v_spline")
    return bg
end

@inline function T_bg(bg::BackgroundFields, τ::Real, r::Real)
    return _eval_spline_clamped(bg.T_spline, r, τ, bg.r_grid, bg.t_grid)
end

@inline function ur_bg(bg::BackgroundFields, τ::Real, r::Real)
    if bg.ur_spline !== nothing
        return _eval_spline_clamped(bg.ur_spline, r, τ, bg.r_grid, bg.t_grid)
    end
    v = _eval_spline_clamped(bg.v_spline, r, τ, bg.r_grid, bg.t_grid)
    _, ur, _ = _u_from_v(v)
    return ur
end

@inline function v_bg(bg::BackgroundFields, τ::Real, r::Real)
    if bg.v_spline !== nothing
        return clamp(_eval_spline_clamped(bg.v_spline, r, τ, bg.r_grid, bg.t_grid), -0.999999, 0.999999)
    end
    _, _, v = _u_from_ur(ur_bg(bg, τ, r))
    return v
end

@inline function alpha_bg(bg::BackgroundFields, τ::Real, r::Real)
    bg.α_spline === nothing && error("Background file needs α_spline for the expanded MIS ν equation")
    return _eval_spline_clamped(bg.α_spline, r, τ, bg.r_grid, bg.t_grid)
end

@inline function _local_spacing(x::Float64, grid::AbstractVector{<:Real})
    N = length(grid)
    N <= 1 && return 1e-3
    idx = searchsortedfirst(grid, x)
    hL = idx > 1 ? abs(x - Float64(grid[idx - 1])) : Inf
    hR = idx <= N ? abs(Float64(grid[idx]) - x) : Inf
    h = min(hL, hR)
    if !isfinite(h) || h <= 0.0
        h = minimum(abs.(diff(Float64.(grid))))
    end
    return max(h, 1e-6)
end

@inline function _eval_spline_partial_clamped(spl, r::Real, τ::Real, r_grid::AbstractVector, t_grid::AbstractVector; wrt::Symbol)
    rr = clamp(Float64(r), Float64(first(r_grid)), Float64(last(r_grid)))
    tt = clamp(Float64(τ), Float64(first(t_grid)), Float64(last(t_grid)))
    if wrt === :r
        h = _local_spacing(rr, r_grid)
        rp = clamp(rr + h, Float64(first(r_grid)), Float64(last(r_grid)))
        rm = clamp(rr - h, Float64(first(r_grid)), Float64(last(r_grid)))
        return (Float64(spl(rp, tt)) - Float64(spl(rm, tt))) / max(rp - rm, 1e-12)
    elseif wrt === :t
        h = _local_spacing(tt, t_grid)
        tp = clamp(tt + h, Float64(first(t_grid)), Float64(last(t_grid)))
        tm = clamp(tt - h, Float64(first(t_grid)), Float64(last(t_grid)))
        return (Float64(spl(rr, tp)) - Float64(spl(rr, tm))) / max(tp - tm, 1e-12)
    else
        error("Unknown derivative direction: $wrt")
    end
end

@inline dt_alpha_bg(bg::BackgroundFields, τ::Real, r::Real) = _eval_spline_partial_clamped(bg.α_spline, r, τ, bg.r_grid, bg.t_grid; wrt=:t)
@inline dr_alpha_bg(bg::BackgroundFields, τ::Real, r::Real) = _eval_spline_partial_clamped(bg.α_spline, r, τ, bg.r_grid, bg.t_grid; wrt=:r)

@inline function dt_ur_bg(bg::BackgroundFields, τ::Real, r::Real)
    spl = bg.ur_spline === nothing ? bg.v_spline : bg.ur_spline
    bg.ur_spline === nothing && error("Background file needs ur_spline for the expanded MIS ν equation")
    return _eval_spline_partial_clamped(spl, r, τ, bg.r_grid, bg.t_grid; wrt=:t)
end

@inline function dr_ur_bg(bg::BackgroundFields, τ::Real, r::Real)
    spl = bg.ur_spline === nothing ? bg.v_spline : bg.ur_spline
    bg.ur_spline === nothing && error("Background file needs ur_spline for the expanded MIS ν equation")
    return _eval_spline_partial_clamped(spl, r, τ, bg.r_grid, bg.t_grid; wrt=:r)
end

@inline dt_T_bg(bg::BackgroundFields, τ::Real, r::Real) = _eval_spline_partial_clamped(bg.T_spline, r, τ, bg.r_grid, bg.t_grid; wrt=:t)
@inline dr_T_bg(bg::BackgroundFields, τ::Real, r::Real) = _eval_spline_partial_clamped(bg.T_spline, r, τ, bg.r_grid, bg.t_grid; wrt=:r)

@inline function _fluidum_single_hadron_normalization(T::Float64, α::Float64, eos::LatticeHRGEOS)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    b2 = SpecialFunctions.besselkx(2, z)
    ex = exp(clamp(α - z, -700.0, 700.0))
    return eos.g_hq * (Tm / (2π^2)) * m^2 * ex * b2 * fmGeV3
end

@inline function diff_kappa_bg(T::Float64, α::Float64, nbg::Float64, DsT::Float64, eos::LatticeHRGEOS)
    # Match Fluidum's `diffusion_hadron(T, α, x::Heavy_Quark, y::HQdiffusion)`.
    return DsT / max(T, T_MIN) * _fluidum_single_hadron_normalization(T, α, eos) / fmGeV
end

@inline function diff_kappa_bg(T::Float64, α::Float64, nbg::Float64, DsT::Float64, eos)
    return DsT / max(T, T_MIN) * max(nbg, 0.0) / fmGeV
end

@inline function diff_tauN_bg(T::Float64, α::Float64, DsT::Float64, eos::LatticeHRGEOS)
    # Match Fluidum's `τ_diffusion_hadron(T, α, x::Heavy_Quark, y::HQdiffusion)`
    # with `tauD = 1` for the single-species Boltzmann approximation used here.
    #
    # τ_n = D_s I_31/(T P_0) is a RATIO of equilibrium moments, so the degeneracy must
    # CANCEL: τ_n = D_s z K₃(z)/K₂(z), independent of g_hq.  Fluidum's hadron-sum form
    # gets this for free because its numerator and its `normalization` both carry the
    # per-species factors (and its pseudoscalar D mesons have Degeneracy=1).  The
    # single-species port below divides by `_fluidum_single_hadron_normalization`, which
    # DOES carry `eos.g_hq` (= 6) — so `tauq` must carry it too, otherwise τ_n comes out
    # a factor g_hq too SMALL.  (2026-07-16: it did; τ_n was 6× short of the matched
    # `LangevInMedium.tau_n_main3` at every T.  See DERIVATION_CHECK.md finding F2.)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0.0 && return 0.0

    b1 = SpecialFunctions.besselkx(1, z)
    b2 = SpecialFunctions.besselkx(2, z)
    b3 = b1 + 4.0 / z * b2
    b4 = b2 + 6.0 / z * b3
    b5 = b3 + 8.0 / z * b4
    ex = exp(clamp(α - z, -700.0, 700.0))

    tauq = eos.g_hq * (DsT / (96.0 * π^2 * Tm^3)) * m^5 * ex * (2.0 * b1 - 3.0 * b3 + b5)
    norm = _fluidum_single_hadron_normalization(Tm, α, eos)
    abs(norm) <= TINY && return 0.0

    τn = tauq / norm * (fmGeV^2)
    if !isfinite(τn)
        return 0.0
    end
    return max(τn, 0.0)
end

@inline function diff_tauN_bg(T::Float64, α::Float64, DsT::Float64, eos)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0.0 && return 0.0

    if z > 50.0
        τ_GeVinv = (DsT / 48.0) * (m^2 / (Tm^2 + 1e-10))
        return min(τ_GeVinv / fmGeV, 1e20)
    end

    K1x = SpecialFunctions.besselkx(1, z)
    K2x = SpecialFunctions.besselkx(2, z)
    K3x = SpecialFunctions.besselkx(3, z)
    K5x = SpecialFunctions.besselkx(5, z)

    numerator = 2*K1x - 3*K3x + K5x
    denominator = max(abs(K2x), TINY)
    ratio = numerator / denominator * sign(K2x == 0 ? 1.0 : K2x)
    z3_over_Tm = z^3 / Tm
    if !isfinite(z3_over_Tm) || z3_over_Tm > 1e50
        z3_over_Tm = 1e50
    end

    τ_GeVinv = (DsT / 48.0) * z3_over_Tm * ratio
    if !isfinite(τ_GeVinv) || abs(τ_GeVinv) > 1e50
        return 1e50 * sign(τ_GeVinv)
    end
    return τ_GeVinv / fmGeV
end

@inline function diff_sigmaT_bg(T::Float64, α::Float64, nbg::Float64, DsT::Float64, eos)
    # σ_T = κ_n z K₃(z)/K₂(z) — Soret coefficient (FP_Hydro_matching eq:BDNK_coeffs)
    Tm = max(T, T_MIN)
    m = hq_mass(eos)
    z = m / Tm
    z <= 0.0 && return 0.0
    κ = diff_kappa_bg(T, α, nbg, DsT, eos)
    K2x = SpecialFunctions.besselkx(2, z)
    K3x = SpecialFunctions.besselkx(3, z)
    return κ * z * K3x / max(abs(K2x), TINY)
end

@inline function diff_sigmaa_bg(T::Float64, α::Float64, nbg::Float64, DsT::Float64, eos)
    # σ_a = -σ_T = -κ_n z K₃(z)/K₂(z) = -n τ_n — the acceleration (inertial-lag) coefficient.
    #
    # 🔴 CORRECTED 2026-08-29. This returned `-κ/T` until today, transcribed from
    # FP_Hydro_matching eq:BDNK_coeffs, whose eq:CE_LHS drops a factor E from the
    # acceleration term of the Chapman–Enskog vector source. Recomputing the streaming
    # term gives   k^<μ>( ∇_μα + (E/T)[∇_μ lnT ∓ u̇_μ] ),  so u̇ carries the SAME moment
    # weight E/T as ∇lnT and the two coefficients are tied: |σ_a| = |σ_T| = n τ_n.
    # Three independent confirmations:
    #   (i)   -κ/T is not dimensionally commensurate with σ_α = κ_n;
    #   (ii)  Tolman–Ehrenfest: ∇lnT ∓ u̇ is the combination that vanishes in global
    #         equilibrium, so a hydrostatic medium must carry no diffusion current.
    #         With -κ/T it carries one, ~53% of either term on a u^r ≲ 1.2 background;
    #   (iii) AttractorHydro App. app:overdamped derives the inertial channel
    #         independently, from the Smoluchowski limit, and gets exactly -n τ_n a^r.
    # Symbolic proof + three negative gates: Julia/tools/derive_bdnk_frame_coeffs.wl (19/19).
    # Numerical gate: test/test_bdnk_frame_coeffs.jl. It FAILS on the old form, and it needs
    # a background with u^r ≠ 0 AND ∂_rT ≠ 0 — which is exactly why nothing caught this:
    # both shipped BDNK backgrounds have neither (FiVoBenchmark/CAUSAL_HYDRO_AUDIT.md, B-4),
    # and in the probe limit σ_T and σ_a do not enter ModePaper1's eigenproblem either.
    return -diff_sigmaT_bg(T, α, nbg, DsT, eos)
end

# Charm number density n(T,α) consistent with the diffusion coefficients above
# (for the lattice HRG, n(T,α)=n_th(T)·e^α with n_th=_fluidum_single_hadron_normalization).
@inline _charm_n_density(T::Float64, α::Float64, eos::LatticeHRGEOS) =
    _fluidum_single_hadron_normalization(T, α, eos)
@inline _charm_n_density(T::Float64, α::Float64, eos) = eos_Pne(T, α * T, eos)[2]

# Invert n → α at fixed T:  α = log(n / n_th(T)).  Self-consistent with κ and τ_n.
@inline function _alpha_from_n(n::Float64, T::Float64, eos)
    n_th = _charm_n_density(T, 0.0, eos)
    return log(max(n, 1e-12 * max(n_th, TINY)) / max(n_th, TINY))
end

function build_limited_slopes!(slopes::Vector{Float64}, u::Vector{Float64}, r::Vector{Float64})
    Nr = length(u)
    Nr == length(slopes) || error("slopes must have length $Nr")
    Nr == length(r) || error("r must have length $Nr")

    fill!(slopes, 0.0)
    if Nr == 0
        return nothing
    end

    if Nr >= 2
        slopes[1] = (u[2] - u[1]) / max(r[2] - r[1], 1e-12)
        slopes[Nr] = (u[Nr] - u[Nr - 1]) / max(r[Nr] - r[Nr - 1], 1e-12)
    end

    if Nr <= 2
        return nothing
    end

    @inbounds for i in 2:(Nr - 1)
        ΔC = max(0.5 * (r[i + 1] - r[i - 1]), 1e-12)
        slopes[i] = mc_limiter(u[i] - u[i - 1], u[i + 1] - u[i]) / ΔC
    end
    return nothing
end

@inline function limited_upwind_gradient(u::Vector{Float64}, slopes::Vector{Float64}, r::Vector{Float64}, i::Int, vel::Float64)
    return slopes[i]
end

# Coupled α–ν mode: recover the fugacity field α from the evolved charm density
# (n from q=r·Jτ and ν_r) on the fixed (T,u^r) background, and its limited r-slope.
# ∂τα is taken as (α - α_old)/dt by the caller.  This closes the n↔α↔ν loop so the
# diffusion current is driven by the self-consistent ∇α, not a frozen background α.
function compute_alpha_field!(ws::CurrentWorkspace1D, q::Vector{Float64}, nu_r::Vector{Float64},
                              τ::Float64, grid::CurrentGrid1D, bg::BackgroundFields;
                              T_floor::Float64, T_freeze::Float64, eos)
    Nr = length(grid.r)
    @inbounds for i in 1:Nr
        r = grid.r[i]
        Tbg = T_bg(bg, τ, r)
        if Tbg < T_freeze
            # Frozen-out tail (no medium below T_fo): carry the last active α with
            # zero gradient so it does not drive a spurious diffusion current.
            ws.alpha[i] = i > 1 ? ws.alpha[i - 1] : 0.0
            continue
        end
        T = max(Tbg, T_floor)
        uτ, ur, _ = _u_from_ur(ur_bg(bg, τ, r))
        Jτ = q[i] / max(r, 1e-12)
        ντ = (uτ <= 0.0) ? 0.0 : (ur / uτ) * nu_r[i]
        n  = (uτ <= 0.0) ? 0.0 : (Jτ - ντ) / uτ
        ws.alpha[i] = _alpha_from_n(n, T, eos)
    end
    build_limited_slopes!(ws.alpha_slope, ws.alpha, grid.r)
    return nothing
end

@inline upwind_flux(v::Real, uL::Real, uR::Real) = (v >= 0) ? v * uL : v * uR

@inline function _diffusion_face_coeffs(Dface::Float64, Δr_face::Float64, rface::Float64)
    invΔ = 1.0 / Δr_face
    inv2r = 0.5 / rface
    α = -Dface * (invΔ - inv2r)
    β = -Dface * (-invΔ - inv2r)
    return α, β
end

function build_adv_flux!(flux_adv::Vector{Float64}, q::Vector{Float64}, τ::Float64, grid::CurrentGrid1D, v_face_fn)
    Nr = length(grid.r)
    flux_adv[1] = 0.0

    @inbounds for i in 2:Nr
        rface = grid.rF[i]
        flux_adv[i] = upwind_flux(v_face_fn(τ, rface), q[i - 1], q[i])
    end

    flux_adv[Nr + 1] = upwind_flux(v_face_fn(τ, grid.rF[Nr + 1]), q[Nr], q[Nr])
    return nothing
end

function build_q_flux!(flux_q::Vector{Float64}, q::Vector{Float64}, nu_r::Vector{Float64}, τ::Float64, grid::CurrentGrid1D, bg::BackgroundFields)
    Nr = length(grid.r)
    flux_q[1] = 0.0

    @inbounds for i in 2:Nr
        rface = grid.rF[i]
        vface = v_bg(bg, τ, rface)
        urface = ur_bg(bg, τ, rface)
        uτface = sqrt(1.0 + urface^2)
        if vface >= 0.0
            Jtau_up = q[i - 1] / max(grid.r[i - 1], 1e-12)
            nu_up = nu_r[i - 1]
        else
            Jtau_up = q[i] / max(grid.r[i], 1e-12)
            nu_up = nu_r[i]
        end
        Jr_face = vface * Jtau_up + nu_up / (uτface^2)
        flux_q[i] = rface * Jr_face
    end

    rface = grid.rF[Nr + 1]
    vface = v_bg(bg, τ, rface)
    urface = ur_bg(bg, τ, rface)
    uτface = sqrt(1.0 + urface^2)
    Jtau_up = q[Nr] / max(grid.r[Nr], 1e-12)
    nu_up = nu_r[Nr]
    Jr_face = vface * Jtau_up + nu_up / (uτface^2)
    flux_q[Nr + 1] = rface * Jr_face
    return nothing
end

function transport_update_q!(
    q::Vector{Float64},
    τ::Float64,
    dt::Float64,
    grid::CurrentGrid1D,
    ws::CurrentWorkspace1D,
)
    Nr = length(grid.r)
    length(q) == Nr || error("q must have length $Nr")

    ws.q_old .= q

    τ_safe = max(τ, 1e-12)
    @inbounds for i in 1:Nr
        q[i] = ws.q_old[i] - dt * (((ws.flux_q[i + 1] - ws.flux_q[i]) / grid.dr[i]) + ws.q_old[i] / τ_safe)
    end
    return nothing
end

function enforce_regularity_bc!(q::Vector{Float64}, nu_r::Vector{Float64}, grid::CurrentGrid1D)
    Nr = length(grid.r)
    if Nr >= 2
        r1 = grid.r[1]
        r2 = grid.r[2]
        fac = r1 / max(r2, 1e-12)
        q[1] = q[2] * fac
        nu_r[1] = nu_r[2] * fac
    elseif Nr == 1
        q[1] = 0.0
        nu_r[1] = 0.0
    end
    # Outer boundary: keep last cell unchanged.
    return nothing
end

function step_system_current_only!(
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
    couple_alpha::Bool=false,
    T_freeze::Float64=0.0,
)
    build_q_flux!(ws.flux_q, q, nu_r, τ, grid, bg)
    transport_update_q!(q, τ, dt, grid, ws)
    # Coupled mode: refresh α from the just-updated charm density before the ν step,
    # then advance α_old for the next step's ∂τα.
    couple_alpha && compute_alpha_field!(ws, q, nu_r, τ, grid, bg; T_floor=T_floor, T_freeze=T_freeze, eos=eos)
    step_nur_expanded!(nu_r, τ, dt, grid, ws, bg; DsT=DsT, T_floor=T_floor, eos=eos, couple_alpha=couple_alpha, T_freeze=T_freeze)
    couple_alpha && (ws.alpha_old .= ws.alpha)
    enforce_regularity_bc!(q, nu_r, grid)
    return nothing
end

function step_nur_expanded!(nu_r::Vector{Float64}, τ::Float64, dt::Float64, grid::CurrentGrid1D, ws::CurrentWorkspace1D, bg::BackgroundFields; DsT::Float64, T_floor::Float64, eos, couple_alpha::Bool=false, T_freeze::Float64=0.0)
    Nr = length(grid.r)
    ws.nu_old .= nu_r
    build_limited_slopes!(ws.nu_slope, ws.nu_old, grid.r)

    @inbounds for i in 1:Nr
        r = grid.r[i]
        # Below freeze-out the charm has decoupled: no thermal diffusion current.
        if couple_alpha && T_bg(bg, τ, r) < T_freeze
            nu_r[i] = 0.0
            continue
        end
        T = max(T_bg(bg, τ, r), T_floor)
        # Coupled mode: α (and ∂rα, ∂τα) from the self-consistent evolved field;
        # background mode: α frozen to the input background spline.
        α = couple_alpha ? ws.alpha[i] : alpha_bg(bg, τ, r)
        μ = α * T
        nbg = bg.n_spline === nothing ? eos_Pne(T, μ, eos)[2] : _eval_spline_clamped(bg.n_spline, r, τ, bg.r_grid, bg.t_grid)
        κ = diff_kappa_bg(T, α, nbg, DsT, eos)
        τn = diff_tauN_bg(T, α, DsT, eos)

        uτ, ur, _ = _u_from_ur(ur_bg(bg, τ, r))
        dtα = couple_alpha ? (ws.alpha[i] - ws.alpha_old[i]) / max(dt, 1e-12) : dt_alpha_bg(bg, τ, r)
        drα = couple_alpha ? ws.alpha_slope[i] : dr_alpha_bg(bg, τ, r)
        dtur = dt_ur_bg(bg, τ, r)
        drur = dr_ur_bg(bg, τ, r)

        source_alpha = κ * (ur * uτ * dtα + (uτ^2) * drα)

        if !isfinite(τn) || τn <= 0.0
            nu_r[i] = -source_alpha
            continue
        end

        dν_dr = 0.0
        if Nr >= 2
            dν_dr = limited_upwind_gradient(ws.nu_old, ws.nu_slope, grid.r, i, ur)
        end

        # Expanded fixed-background MIS equation for νr:
        #   τn[uτ ∂τ νr + ur ∂r νr - (ur/uτ^2) νr ∂τ ur - (ur²/uτ²) νr ∂r ur]
        #   + νr + κ[ur uτ ∂τ α + uτ² ∂r α] = 0.
        coeff = 1.0 - τn * ((ur / (uτ^2)) * dtur + (ur^2 / (uτ^2)) * drur)
        A = τn * uτ / dt
        denom = max(A + coeff, 1e-12)
        nu_r[i] = (A * ws.nu_old[i] - τn * ur * dν_dr - source_alpha) / denom
    end

    if Nr >= 2
        nu_r[1] = nu_r[2] * (grid.r[1] / grid.r[2])
    elseif Nr == 1
        nu_r[1] = 0.0
    end
    return nothing
end

function solve_current_only(
    grid::CurrentGrid1D,
    q0::Vector{Float64},
    nu0::Vector{Float64},
    τ0::Float64,
    τf::Float64,
    bg::BackgroundFields;
    CFL::Float64,
    CFLτ::Float64,
    save_dt::Float64,
    log_every::Int = 50,
    DsT::Float64,
    T_floor::Float64,
    eos,
    couple_alpha::Bool=false,
    T_freeze::Float64=0.0,
)
    Nr = length(grid.r)
    length(q0) == Nr || error("q0 must have length $Nr")
    length(nu0) == Nr || error("nu0 must have length $Nr")

    q = copy(q0)
    nu_r = copy(nu0)
    τ = τ0
    it = 0
    next_dump = τ0
    ws = CurrentWorkspace1D(grid)

    # Seed the coupled fugacity field from the IC so the first step's ∂τα is well defined.
    if couple_alpha
        compute_alpha_field!(ws, q, nu_r, τ0, grid, bg; T_floor=T_floor, T_freeze=T_freeze, eos=eos)
        ws.alpha_old .= ws.alpha
    end

    τs = Float64[τ]
    qs = Vector{Float64}[copy(q)]
    nus = Vector{Float64}[copy(nu_r)]

    while τ < τf - 1e-12
        vmax = maximum(abs.(v_bg.(Ref(bg), Ref(τ), grid.rF[2:end])))
        λmax = max(vmax, 1e-8)
        dt_adv = CFL * minimum(grid.dr) / λmax
        dt_tau = CFLτ * τ
        Δτ = min(dt_adv, dt_tau)
        if τ + Δτ > τf
            Δτ = τf - τ
        end

        τ_eval = τ + Δτ
        step_system_current_only!(q, nu_r, τ_eval, Δτ, grid, ws, bg; DsT=DsT, T_floor=T_floor, eos=eos, couple_alpha=couple_alpha, T_freeze=T_freeze)
        τ = τ_eval
        it += 1

        if τ >= next_dump - 1e-12
            push!(τs, τ)
            push!(qs, copy(q))
            push!(nus, copy(nu_r))
            next_dump += save_dt
        end

        if (it % log_every) == 0
            @info "current-only progress" τ=τ Δτ=Δτ it=it vmax=vmax
        end
    end

    if τs[end] != τ
        push!(τs, τ)
        push!(qs, copy(q))
        push!(nus, copy(nu_r))
    end

    return τs, qs, nus
end

function build_eos(kind::String)
    k = lowercase(kind)
    if k in ("lattice", "latticehrg", "lhrg")
        return LatticeHRGEOS()
    elseif k in ("conformal", "conformalhq", "chq")
        return ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    elseif k in ("running", "runningconformal", "rconformal")
        return RunningConformalHQEOS()
    else
        @warn "Unknown EOS, falling back to LatticeHRGEOS" kind=kind
        return LatticeHRGEOS()
    end
end

function initial_state_from_background(bg::BackgroundFields, τ0::Float64, r::Float64, eos)
    T0 = T_bg(bg, τ0, r)
    α0 = bg.α_spline === nothing ? 0.0 : alpha_bg(bg, τ0, r)
    μ0 = α0 * T0
    n0 = bg.n_spline === nothing ? eos_Pne(T0, μ0, eos)[2] : _eval_spline_clamped(bg.n_spline, r, τ0, bg.r_grid, bg.t_grid)
    ur0 = ur_bg(bg, τ0, r)
    uτ0, _, _ = _u_from_ur(ur0)
    νr0 = bg.nur_spline === nothing ? 0.0 : _eval_spline_clamped(bg.nur_spline, r, τ0, bg.r_grid, bg.t_grid)
    ντ0 = (uτ0 <= 0.0) ? 0.0 : (ur0 / uτ0) * νr0
    return n0 * uτ0 + ντ0, νr0
end

function initial_current_from_csv(init_csv::AbstractString, grid::CurrentGrid1D, τ0::Float64, bg::BackgroundFields, eos; fugacity_kind::Symbol = :alpha, taper_width::Float64 = 0.0, interp_kind::Symbol = :linear, interp_dr::Union{Nothing,Float64} = nothing)
    itpT, itpF, itpNurStored, _ = load_initial_interpolants(init_csv;
        fugacity_kind=fugacity_kind,
        taper_width=taper_width,
        interp_kind=interp_kind,
        interp_dr=interp_dr,
    )

    q0 = Vector{Float64}(undef, length(grid.r))
    nu0 = Vector{Float64}(undef, length(grid.r))
    @inbounds for i in eachindex(grid.r)
        r = grid.r[i]
        T0 = max(Float64(itpT(r)), T_MIN)
        f0 = Float64(itpF(r))
        μ0 = fugacity_kind === :alpha ? f0 * T0 : log(max(f0, TINY)) * T0
        _, n0, _ = eos_Pne(T0, μ0, eos)
        νr0 = phys_from_stored(Float64(itpNurStored(r)))
        ur0 = ur_bg(bg, τ0, r)
        uτ0, _, _ = _u_from_ur(ur0)
        ντ0 = (uτ0 <= 0.0) ? 0.0 : (ur0 / uτ0) * νr0
        Jτ0 = n0 * uτ0 + ντ0
        q0[i] = r * Jτ0
        nu0[i] = νr0
    end
    return q0, nu0
end

function reconstruct_snapshot(q::Vector{Float64}, nu_r::Vector{Float64}, τ::Float64, grid::CurrentGrid1D, bg::BackgroundFields; DsT::Float64, T_floor::Float64, eos)
    Nr = length(grid.r)
    length(q) == Nr || error("q must have length $Nr")
    length(nu_r) == Nr || error("nu_r must have length $Nr")

    Jtau = similar(q)
    Jr = similar(q)
    ur = similar(q)
    u_tau = similar(q)
    v = similar(q)
    T = similar(q)
    mu = similar(q)
    alpha = similar(q)
    kappa = similar(q)
    tau_n = similar(q)
    n_bg = similar(q)

    @inbounds for i in 1:Nr
        r = grid.r[i]
        r_safe = max(r, 1e-12)
        Jtau[i] = q[i] / r_safe

        T[i] = max(T_bg(bg, τ, r), T_floor)
        α = alpha_bg(bg, τ, r)
        μ = α * T[i]
        alpha[i] = α
        mu[i] = μ
        n_bg[i] = bg.n_spline === nothing ? eos_Pne(T[i], μ, eos)[2] : _eval_spline_clamped(bg.n_spline, r, τ, bg.r_grid, bg.t_grid)
        kappa[i] = diff_kappa_bg(T[i], α, n_bg[i], DsT, eos)
        tau_n[i] = diff_tauN_bg(T[i], α, DsT, eos)
        u_tau[i], ur[i], v[i] = _u_from_ur(ur_bg(bg, τ, r))
        Jr[i] = v[i] * Jtau[i] + nu_r[i] / (u_tau[i]^2)
    end

    nu_tau = ifelse.(u_tau .<= 0.0, 0.0, (ur ./ u_tau) .* nu_r)
    n = @. (Jtau - nu_tau) / u_tau
    nur = stored_from_phys.(nu_r)

    # Column order follows main.jl's write_snapshot_csv, dropping hydro-only fields
    # (Dtau, Sr, E, Pi, piR, piEta, piPhi, D, phi, e, P, u_tau, nu_tau)
    return (; r=grid.r, tau=fill(τ, Nr),
             nur, nu_r, Jtau, Jr,
             T, mu, alpha, ur, v, n, ok=trues(Nr),
             kappa, tau_n)
end

function write_current_snapshot_csv(fname::AbstractString, q::Vector{Float64}, nu_r::Vector{Float64}, τ::Float64, grid::CurrentGrid1D, bg::BackgroundFields; DsT::Float64, T_floor::Float64, eos)
    dir = dirname(fname)
    !isempty(dir) && mkpath(dir)

    snap = reconstruct_snapshot(q, nu_r, τ, grid, bg; DsT=DsT, T_floor=T_floor, eos=eos)
    CSV.write(fname, snap)

    meta = (; tau=[τ], Q=[TWO_PI * τ * sum(q .* grid.dr)])
    CSV.write(replace(fname, ".csv" => "_meta.csv"), meta)
    return nothing
end

function run_current_background(; outdir::String,
    Nr::Int = 300,
    rmax::Float64 = 25.0,
    nghost::Int = 1,
    τ0::Float64 = 0.4,
    τfinal::Float64 = 10.0,
    CFL::Float64 = 0.3,
    CFLτ::Float64 = 0.05,
    dump_dt::Float64 = 0.1,
    log_every::Int = 50,
    background_file::String,
    init_csv::Union{Nothing,String} = nothing,
    init_mode::Symbol = :auto,
    fugacity_kind::Symbol = :alpha,
    taper_width::Float64 = 0.0,
    interp_kind::Symbol = :linear,
    interp_dr::Union{Nothing,Float64} = nothing,
    DsT::Float64 = 0.24,
    T_floor::Float64 = 1e-6,
    eos = LatticeHRGEOS(),
    run_label::String = "",
    splines_outdir::Union{Nothing,String} = nothing,
    couple_alpha::Bool = false,
    T_freeze::Float64 = 0.0,
)
    _fname_float(x::Real; digits::Int=3) = replace(replace(@sprintf("%.*f", digits, Float64(x)), "." => "p"), "-" => "m")
    _fname_symbol(s::Symbol) = replace(String(s), ":" => "")
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

    init_choice = init_mode
    if init_choice === :auto
        init_choice = (bg.n_spline !== nothing) ? :background : :csv
    end

    q0, nu0 = if init_choice === :background
        Jτ0 = Vector{Float64}(undef, length(grid.r))
        νr0 = Vector{Float64}(undef, length(grid.r))
        @inbounds for i in eachindex(grid.r)
            Jτ0[i], νr0[i] = initial_state_from_background(bg, τ0, grid.r[i], eos)
        end
        grid.r .* Jτ0, νr0
    elseif init_choice === :csv
        init_csv === nothing && error("INIT_CSV is required when INIT_MODE=csv")
        initial_current_from_csv(init_csv, grid, τ0, bg, eos;
            fugacity_kind=fugacity_kind,
            taper_width=taper_width,
            interp_kind=interp_kind,
            interp_dr=interp_dr,
        )
    else
        error("Unknown init_mode=$init_choice. Use :auto, :background, or :csv")
    end

    bg.α_spline === nothing && error("Background file needs α_spline to evolve the expanded MIS ν equation")
    bg.ur_spline === nothing && error("Background file needs ur_spline to evolve the expanded MIS ν equation")

    mkpath(outdir)
    clear_dir!(outdir)
    write_current_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ0)), q0, nu0, τ0, grid, bg; DsT=DsT, T_floor=T_floor, eos=eos)

    @info "Start current-only background evolution" outdir=outdir Nr=Nr rmax=rmax_use τ0=τ0 τfinal=τfinal CFL=CFL CFLτ=CFLτ DsT=DsT init_mode=init_choice background_file=background_file
    τs, qs, nus = solve_current_only(grid, q0, nu0, τ0, τfinal, bg;
        CFL=CFL,
        CFLτ=CFLτ,
        save_dt=dump_dt,
        log_every=log_every,
        DsT=DsT,
        T_floor=T_floor,
        eos=eos,
        couple_alpha=couple_alpha,
        T_freeze=T_freeze,
    )

    for (τ, q, nu_r) in zip(τs[2:end], qs[2:end], nus[2:end])
        write_current_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)), q, nu_r, τ, grid, bg; DsT=DsT, T_floor=T_floor, eos=eos)
    end

    param_stamp = "tau0_$(_fname_float(τ0; digits=3))_tauf_$(_fname_float(τfinal; digits=3))_rmax_$(_fname_float(rmax_use; digits=3))_nr_$(Nr)_dst_$(_fname_float(DsT; digits=3))"

    label_clean = strip(run_label)
    label_for_plots = isempty(label_clean) ? "FiVo current-only" : label_clean
    file_base = isempty(label_clean) ? "FiVoCurrentOnly" : _sanitize_filename_base(label_clean)
    isempty(file_base) && (file_base = "FiVoCurrentOnly")

    currents_tag = _sanitize_filename_base(label_for_plots)
    isempty(currents_tag) && (currents_tag = "FiVo_current_only")
    currents_tag *= "_" * param_stamp
    currents_path = write_hydro_currents_jld2(outdir; outdir=outdir, tag=currents_tag)
    plot_splines_dir = splines_outdir !== nothing ? splines_outdir : normpath(joinpath(@__DIR__, "..", "Plot", "splines"))
    mkpath(plot_splines_dir)
    splines_path = write_hydro_currents_splines_jld2(
        outdir;
        outdir=plot_splines_dir,
        filename="$(file_base)_$(param_stamp).jld2",
        tag=label_for_plots,
        overwrite=true,
        kx=1,
        ky=3,
        store_phi_splines=false,
    )
    @info "Wrote current-only outputs" currents_path splines_path
    return nothing
end

function main()
    setup_logger!(level=Logging.Info)

    env_float(name::String, default::Float64) = haskey(ENV, name) ? parse(Float64, ENV[name]) : default
    env_int(name::String, default::Int) = haskey(ENV, name) ? parse(Int, ENV[name]) : default
    env_str(name::String, default::String) = haskey(ENV, name) ? String(ENV[name]) : default
    env_bool(name::String, default::Bool) = haskey(ENV, name) ? (lowercase(strip(String(ENV[name]))) in ("1", "true", "yes", "on")) : default

    function env_maybe_float(name::String, default::Union{Nothing,Float64}=nothing)
        if !haskey(ENV, name)
            return default
        end
        s = strip(String(ENV[name]))
        isempty(s) && return default
        return parse(Float64, s)
    end

    background_file = env_str("BACKGROUND_JLD2", normpath(joinpath(@__DIR__, "..", "LangevInMedium.jl", "src", "data", "Fluidum_MIS_HQ.jld2")))
    outdir = normpath(env_str("HYDRO_OUTDIR", joinpath(@__DIR__, "snapshots", "current_only")))
    init_csv_default = joinpath(@__DIR__, "data", "initial_profiles_physical.csv")
    init_csv_env = env_str("INIT_CSV", init_csv_default)
    init_csv = (isempty(strip(init_csv_env)) || !isfile(init_csv_env)) ? nothing : init_csv_env

    init_mode_str = lowercase(env_str("INIT_MODE", "auto"))
    init_mode = init_mode_str == "background" ? :background : (init_mode_str == "csv" ? :csv : :auto)
    fugacity_kind = lowercase(env_str("FUGACITY", "alpha")) == "lambda" ? :lambda : :alpha
    interp_kind = lowercase(env_str("INTERP", "linear")) == "cubic" ? :cubic : :linear
    interp_dr = env_maybe_float("INTERP_DR", nothing)
    eos = build_eos(env_str("EOS", "latticehrg"))

    τ0 = env_float("TAU0", 0.4)
    τfinal = env_float("TAUFINAL", 5.0)
    Nr = env_int("NR", 200)
    rmax = env_float("RMAX", 20.0)
    nghost = env_int("NGHOST", 1)
    CFL = env_float("CFL", 0.3)
    CFLτ = haskey(ENV, "CFLTAU") ? parse(Float64, ENV["CFLTAU"]) : 0.05
    dump_dt = env_float("DUMP_DT", 0.1)
    log_every = env_int("LOG_EVERY", 50)
    DsT = env_float("DS_T", 0.24)
    taper_width = env_float("TAPER_WIDTH", 0.0)
    T_floor = env_float("T_FLOOR", 1e-6)
    run_label = env_str("RUN_LABEL", "")
    splines_outdir_env = env_str("HYDRO_SPLINES_OUTDIR", "")
    splines_outdir = isempty(strip(splines_outdir_env)) ? nothing : normpath(splines_outdir_env)
    # Coupled α–ν solve (default). Set COUPLE_ALPHA=0 to recover the legacy
    # background-α (ν-only) mode where α is read frozen from the input.
    couple_alpha = env_bool("COUPLE_ALPHA", true)
    # Freeze-out temperature: below it the charm decouples (no diffusion current).
    T_freeze = env_float("T_FREEZE", 0.156)

    run_current_background(
        outdir=outdir,
        Nr=Nr,
        rmax=rmax,
        nghost=nghost,
        τ0=τ0,
        τfinal=τfinal,
        CFL=CFL,
        CFLτ=CFLτ,
        dump_dt=dump_dt,
        log_every=log_every,
        background_file=background_file,
        init_csv=init_csv,
        init_mode=init_mode,
        fugacity_kind=fugacity_kind,
        taper_width=taper_width,
        interp_kind=interp_kind,
        interp_dr=interp_dr,
        DsT=DsT,
        T_floor=T_floor,
        eos=eos,
        run_label=run_label,
        splines_outdir=splines_outdir,
        couple_alpha=couple_alpha,
        T_freeze=T_freeze,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module hydro_current
