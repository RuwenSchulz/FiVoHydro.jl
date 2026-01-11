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
    Ts = max(T, 1e-50) * max(th.s, 1e-50)
    return η / (Ts * max(sh.Cs, 1e-50))
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
    s  = max(th.s, 1e-50)
    Δ  = max(abs(1/3 - th.cs2), 1e-6)
    return (ζ / (max(T,1e-50) * s * max(b.Cζ,1e-50))) * (1/Δ^2) + 0.1
end

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
struct IdealDiffViscModel{EOS,LAY,PR,SH<:ShearViscosity,BU<:BulkViscosity}
    eos::EOS
    layout::LAY
    primrec::PR

    # charge diffusion
    enable_diff::Bool
    kappa_coeff::Float64
    tauN_coeff::Float64
    deltaN_factor::Float64
    diff_dt_coeff::Float64
    nur_clip_factor::Float64

    alpha_filter_eps::Float64
    nur_filter_eps::Float64
    alpha_smooth_len::Float64
    nur_smooth_len::Float64

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

    visc_filter_eps::Float64
    visc_smooth_len::Float64

    Pi_clip_factor::Float64
    pi_clip_factor::Float64

    advect_Pi::Bool
    advect_pi::Bool

    # NEW: snapshot transport term in relaxation for Π and π
    relax_advect_Pi::Bool
    relax_advect_pi::Bool
end


@inline layout(model::IdealDiffViscModel) = model.layout
