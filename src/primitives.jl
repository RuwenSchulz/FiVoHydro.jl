# ==============================================================================
# src/primitives.jl
#
# Primitive state + transport models + main model struct.
# ==============================================================================

# ------------------------------------------------------------
# Primitives (PHYSICAL dissipatives)
# ------------------------------------------------------------
struct PrimIdealVisc
    T::Float64
    mu::Float64
    ur::Float64
    n::Float64
    e::Float64
    P::Float64
    nur::Float64     # ν^r (physical)
    Pi::Float64      # Π (physical)
    piR::Float64     # πR (physical)
    piEta::Float64   # πEta (physical)
    ok::Bool
end

# ------------------------------------------------------------
# Transport models (shear + bulk)
# ------------------------------------------------------------
abstract type ShearViscosity end
abstract type BulkViscosity  end

struct LocalThermo{T}
    T::T
    mu::T
    n::T
    e::T
    P::T
    s::T
    cs2::T
end

@inline function local_thermo(T::Float64, μ::Float64, n::Float64, e::Float64, P::Float64, eos)
    s   = eos_entropy(T, μ, n, e, P)
    cs2 = eos_cs2(T, μ, eos)
    return LocalThermo(T, μ, n, e, P, s, cs2)
end

# ---- Shear ----
struct QGPViscosity{T} <: ShearViscosity
    ηs::T   # eta/s
    Cs::T   # relaxation coefficient
end
struct ZeroViscosity <: ShearViscosity end

@inline viscosity(T, th::LocalThermo, ::ZeroViscosity) = 0.0
@inline τ_shear(T, th::LocalThermo, ::ZeroViscosity)   = 0.0

@inline function viscosity(T, th::LocalThermo, sh::QGPViscosity)
    s = max(th.s, 0.0)
    return max(sh.ηs, 0.0) * s * invfmGeV
end

@inline function τ_shear(T, th::LocalThermo, sh::QGPViscosity)
    η  = viscosity(T, th, sh)
    Ts = posden(T) * posden(th.s)
    return safe_div(η, (Ts * posden(sh.Cs)))
end

"""
    warn_if_acausal_shear(; enable_shear, eta_over_s, tauShear_coeff, solver)

Warn when the Israel–Stewart shear sector is acausal. With τ_π = η/(C_s T s) the
coupling is η/(τ_π(e+P)) = C_s (at μ = 0), and for a conformal fluid causality of the
longitudinal mode needs c_s² + (4/3) C_s ≤ 1, i.e. **C_s ≤ 1/2** (the bound loosens as
c_s² drops below 1/3). An acausal IS theory is unstable in a moving frame (Pu, Koide &
Rischke, PRD 81 (2010) 114039). MEASURED in the 1+1D solver on viscous Gubser flow
(η/s = 0.1): stable at C_s ≤ 0.6, a runaway that grows with resolution at 0.8, a crash
at 1.0 (2026-09-11; FiVoBenchmark's viscous Gubser ran at 1.0 only while ∂_τu^r was
~4 % of its value — EQUATIONS1D.md §8). Production uses C_s = 0.2.
"""
function warn_if_acausal_shear(; enable_shear::Bool, eta_over_s::Real, tauShear_coeff::Real,
                               solver::AbstractString = "FiVo")
    if enable_shear && eta_over_s > 0 && tauShear_coeff > 0.5
        @warn "$solver: tauShear_coeff = C_s = $tauShear_coeff > 1/2 — the shear sector is ACAUSAL " *
              "(conformal IS needs η/(τ_π(e+P)) = C_s ≤ 1/2) and unstable in a moving frame. Production uses 0.2." maxlog = 1
    end
    return nothing
end

# ---- Bulk ----
struct ZeroBulkViscosity <: BulkViscosity end
@inline bulk_viscosity(T, th::LocalThermo, ::ZeroBulkViscosity) = 0.0
@inline τ_bulk(T, th::LocalThermo, ::ZeroBulkViscosity) = 0.0

struct SimpleBulkViscosity{T} <: BulkViscosity
    ζs::T   # zeta/s peak height
    Cζ::T   # relaxation coefficient
end

@inline function bulk_viscosity(T, th::LocalThermo, b::SimpleBulkViscosity)
    peak = b.ζs / (1.0 + ((T - 0.175)/0.024)^2)
    return peak * invfmGeV * max(th.s, 0.0)
end

@inline function τ_bulk(T, th::LocalThermo, b::SimpleBulkViscosity)
    ζ  = bulk_viscosity(T, th, b)
    s  = posden(th.s)
    Δ  = max(abs(1/3 - th.cs2), 1e-8)
    return safe_div(ζ, (posden(T) * s * posden(b.Cζ))) * (1/Δ^2) + 0.1
end

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
# The per-term switches (`Terms`, src/terms.jl) are a field of the model. Every
# driver includes the register first; the guard covers the scripts that include
# this file on its own (test_primrec2d*.jl, test_charge2d.jl).
@isdefined(Terms) || include(joinpath(@__DIR__, "terms.jl"))

struct IdealDiffViscModel{EOS,LAY,PR,SH<:ShearViscosity,BU<:BulkViscosity}
    eos::EOS
    layout::LAY
    primrec::PR

    # charge diffusion
    enable_diff::Bool
    diffusion_drive::Symbol      # :alpha (ν_NS ∝ ∂r(μ/T)) or :n (density-based form with thermal piece subtracted)
    kappa_coeff::Float64
    tauN_coeff::Float64
    deltaN_factor::Float64
    diff_dt_coeff::Float64
    shear_dt_coeff::Float64
    bulk_dt_coeff::Float64
    nur_clip_factor::Float64

    alpha_filter_eps::Float64
    nur_filter_eps::Float64
    alpha_smooth_len::Float64
    nur_smooth_len::Float64
    do_soft_project_nur::Bool

    do_axis_project_nur::Bool
    axis_project_nfit::Int
    advect_nur::Bool            # flux advection of stored ν
    relax_advect_nur::Bool      # snapshot transport term in relaxation

    # viscosity (explicit models)
    enable_shear::Bool
    enable_bulk::Bool
    shear::SH
    bulk::BU

    deltaPi_factor::Float64
    deltaShear_factor::Float64

    # Second-order relaxation couplings (scaled by relaxation times).
    # These are expressed as dimensionless factors multiplying τΠ, τπ, τn.
    # Set factors to 0.0 to recover minimal MIS relaxation.
    taupi_pi_factor::Float64      # τππ = factor * τπ
    lambda_Pi_pi_factor::Float64  # λΠπ = factor * τΠ
    lambda_pi_Pi_factor::Float64  # λπΠ = factor * τπ
    lambda_NN_factor::Float64     # λNN = factor * τn

    visc_filter_eps::Float64
    visc_smooth_len::Float64

    Pi_clip_factor::Float64
    pi_clip_factor::Float64

    advect_Pi::Bool
    advect_pi::Bool

    # NEW: snapshot transport term in relaxation for Π and π
    relax_advect_Pi::Bool
    relax_advect_pi::Bool

    # Charge-sector closure:
    #   :mis           -> Israel-Stewart relaxation of ν^r toward ν_NS (default)
    #   :density_frame -> canonical density frame: μ fixed by the on-slice charge
    #                     density (no auxiliary ν field; realized by a ν-less layout),
    #                     diffusion added as a first-order-in-time parabolic flux
    #                     J^r_D = -κ (u^τ)^2 ∂_r α in rhs!.  See
    #                     Tex/DensityFrame/df_fp_derivation.tex.
    charge_mode::Symbol

    # THE THERMODYNAMICALLY CONSISTENT FIRST MOMENT in the bulk solver (2026-09-11):
    # the five full-∇P sources of src/hq_consistent_firstmoment.jl (the O+O charm IS2
    # production closure) subtracted from the ν^r relaxation target — the 1+1D twin of
    # the 2-D `consistent_fm`. Default false = the shipped ∇α-only drive, bit-identical.
    consistent_fm::Bool
    # Per-term switches (src/terms.jl). `Terms()` = the equations as they stand.
    terms::Terms
end


# Backward-compatible constructors. Callers that predate the charge_mode field
# (e.g. mainBDNK.jl, the benches, test/runtests.jl) pass eos, layout, primrec + the 36 fields
# up to relax_advect_pi (39 positional args); callers that predate consistent_fm/terms pass 40
# (… charge_mode). Both get the shipped defaults: charge_mode = :mis, consistent_fm = false,
# terms = Terms(). The full 42-argument inner constructor is what main.jl uses.
IdealDiffViscModel(eos, layout, primrec, rest::Vararg{Any,36}) =
    IdealDiffViscModel(eos, layout, primrec, rest..., :mis, false, Terms())
IdealDiffViscModel(eos, layout, primrec, rest::Vararg{Any,37}) =
    IdealDiffViscModel(eos, layout, primrec, rest..., false, Terms())

@inline layout(model::IdealDiffViscModel) = model.layout
