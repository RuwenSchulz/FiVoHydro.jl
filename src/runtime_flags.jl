# ==============================================================================
# src/runtime_flags.jl
#
# Centralized runtime flag / debugging / logging controls.
#
# Policy:
# - Only this file reads ENV for FiVoHydro runtime toggles.
# - Downstream code calls `hydro_flags()` (cached) or helper accessors.
# - Defaults are conservative: debugging/logging/abort features are OFF unless enabled.
#
# Boolean ENV parsing:
# - true  values: "1", "true", "yes", "on" (case-insensitive)
# - false values: "0", "false", "no", "off" (case-insensitive)
# - anything else falls back to the default.
# ==============================================================================

@inline function _env_bool(name::AbstractString, default::Bool)
    v = get(ENV, name, nothing)
    v === nothing && return default
    s = lowercase(String(v))
    s in ("1", "true", "yes", "on")  && return true
    s in ("0", "false", "no", "off") && return false
    return default
end

@inline function _env_int(name::AbstractString, default::Int)
    v = get(ENV, name, nothing)
    v === nothing && return default
    x = tryparse(Int, String(v))
    return x === nothing ? default : x
end

@inline function _env_float(name::AbstractString, default::Float64)
    v = get(ENV, name, nothing)
    v === nothing && return default
    x = tryparse(Float64, String(v))
    return x === nothing ? default : x
end

"""Runtime/debug flags for FiVoHydro.

This struct is intentionally the single source of truth for ENV-driven toggles.
Use `hydro_flags()` to obtain a cached instance.

See README.md §"Environment flags" and the generated ENV_FLAGS.md (tools/list_env_flags.jl) for the user-facing list.
"""
Base.@kwdef struct HydroFlags
    # Admissibility / repair knobs
    repair_theta::Bool = false                   # HYDRO_REPAIR_THETA
    srscale_margin::Float64 = 0.999              # HYDRO_SRSCALE_MARGIN

    # Debug checks
    check_tcons::Bool = false                    # HYDRO_CHECK_TCONS
    check_tcons_every::Int = 50                  # HYDRO_CHECK_TCONS_EVERY
    check_tcons_i::Int = -1                      # HYDRO_CHECK_TCONS_I
    check_tcons_tol_rel::Float64 = 1e-6          # HYDRO_CHECK_TCONS_TOL_REL
    check_tcons_tol_abs::Float64 = 1e-10         # HYDRO_CHECK_TCONS_TOL_ABS

    # Sr scaling/clamp logging
    log_srscale::Bool = false                    # HYDRO_LOG_SRSCALE
    log_srscale_eps::Float64 = 0.01              # HYDRO_LOG_SRSCALE_EPS
    log_srscale_every::Int = 50                  # HYDRO_LOG_SRSCALE_EVERY
    dump_srscale::Bool = false                   # HYDRO_DUMP_SRSCALE

    # Primitive recovery failure logging
    log_primfail::Bool = false                   # HYDRO_LOG_PRIMFAIL
    log_primfail_every::Int = 20                 # HYDRO_LOG_PRIMFAIL_EVERY
    dump_primfail::Bool = false                  # HYDRO_DUMP_PRIMFAIL

    # Abort-on-first-event debugging
    abort_on_first_primfail::Bool = false        # HYDRO_ABORT_ON_FIRST_PRIMFAIL
    abort_on_first_srscale::Bool = false         # HYDRO_ABORT_ON_FIRST_SRSCALE
    abort_srscale_min_tau::Float64 = 0.0         # HYDRO_ABORT_SRSCALE_MIN_TAU
    abort_srscale_eps::Float64 = 0.0             # HYDRO_ABORT_SRSCALE_EPS
    abort_srscale_post_eps::Float64 = 1e-6       # HYDRO_ABORT_SRSCALE_POST_EPS

    # Debug artifact output
    write_badmask::Bool = false                  # HYDRO_WRITE_BADMASK
end

const _HYDRO_FLAGS_CACHE = Ref{Union{Nothing,HydroFlags}}(nothing)

function hydro_flags(; refresh::Bool=false)::HydroFlags
    if refresh || (_HYDRO_FLAGS_CACHE[] === nothing)
        flags = HydroFlags(
            repair_theta = _env_bool("HYDRO_REPAIR_THETA", false),
            srscale_margin = clamp(_env_float("HYDRO_SRSCALE_MARGIN", 0.999), 0.0, 1.0),

            check_tcons = _env_bool("HYDRO_CHECK_TCONS", false),
            check_tcons_every = max(_env_int("HYDRO_CHECK_TCONS_EVERY", 50), 1),
            check_tcons_i = _env_int("HYDRO_CHECK_TCONS_I", -1),
            check_tcons_tol_rel = max(_env_float("HYDRO_CHECK_TCONS_TOL_REL", 1e-6), 0.0),
            check_tcons_tol_abs = max(_env_float("HYDRO_CHECK_TCONS_TOL_ABS", 1e-10), 0.0),

            log_srscale = _env_bool("HYDRO_LOG_SRSCALE", false),
            log_srscale_eps = max(_env_float("HYDRO_LOG_SRSCALE_EPS", 0.01), 0.0),
            log_srscale_every = max(_env_int("HYDRO_LOG_SRSCALE_EVERY", 50), 1),
            dump_srscale = _env_bool("HYDRO_DUMP_SRSCALE", false),

            log_primfail = _env_bool("HYDRO_LOG_PRIMFAIL", false),
            log_primfail_every = max(_env_int("HYDRO_LOG_PRIMFAIL_EVERY", 20), 1),
            dump_primfail = _env_bool("HYDRO_DUMP_PRIMFAIL", false),

            abort_on_first_primfail = _env_bool("HYDRO_ABORT_ON_FIRST_PRIMFAIL", false),
            abort_on_first_srscale = _env_bool("HYDRO_ABORT_ON_FIRST_SRSCALE", false),
            abort_srscale_min_tau = max(_env_float("HYDRO_ABORT_SRSCALE_MIN_TAU", 0.0), 0.0),
            abort_srscale_eps = max(_env_float("HYDRO_ABORT_SRSCALE_EPS", 0.0), 0.0),
            abort_srscale_post_eps = max(_env_float("HYDRO_ABORT_SRSCALE_POST_EPS", 1e-6), 0.0),

            write_badmask = _env_bool("HYDRO_WRITE_BADMASK", false),
        )
        _HYDRO_FLAGS_CACHE[] = flags
    end
    return _HYDRO_FLAGS_CACHE[]::HydroFlags
end

@inline sr_margin() = hydro_flags().srscale_margin

# ── the axis cell ────────────────────────────────────────────────────────────────────────────────
# The first physical cell sits at r = dr/2, NOT at r = 0. Until 2026-09-15 three places treated it
# as if it were on the axis: `apply_bc!` zeroed its conserved radial momentum S_r, `rhs!` zeroed
# its u^r, and `reconstruct_muscl_prims!` left its face first order. u^r(dr/2) = dr/2
# on Gubser — an O(dr) quantity being discarded, which is exactly the first-order error cell 1 used
# to show. Setting this to "0" restores the pre-2026-09-15 arithmetic bit for bit.
# Measured (viscous Gubser, eta/s = 0.02, tau = 1 -> 2): RHS error in cell 1 -18.7 % -> +4.5e-5,
# L2(T) at Nr = 800  2.614e-04 -> 2.780e-05, order in T 1.77 -> 2.12. Ladder 10/10 either way.
# ⚠ The three act TOGETHER. Reconstruction alone makes cell 1 WORSE (-18.7 % -> -38 %).
const AXIS_CELL_EXACT = get(ENV, "FIVO_AXIS_CELL_EXACT", "1") == "1"
