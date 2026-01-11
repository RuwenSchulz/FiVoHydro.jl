# ==============================================================================
# src/constants.jl
#
# Global constants / knobs + sign convention helpers.
# Keep this file dependency-free (only Base).
# ==============================================================================



# --- physical / unit constants used across modules ---
const ħc     = 0.1973269804
const fmGeV  = 1 / ħc
const invfmGeV = ħc
const fmGeV3 = (1 / ħc)^3


const T_EOS_MIN = 1e-15  # choose for numerical safety (GeV)

# --- numeric safety knobs ---
const T_MIN  = 1e-15
#const LOG_MAX = 3000.0
#const LOG_MIN = -3000.0


const T_SOLVE_MAX = 10.0
const Y_CAP       = 10000.0
const PHI_CAP     = 10000.0     # keeps μ = m + T*φ from exploding


# Vacuum / floors
const E_VAC   = 1e-15
const D_VAC   = 0.0
const E_FLOOR = E_VAC

# Controls
const DO_MOOD         = true
const SANITIZE_SCOPE  = :local
const χ_SrE           = 0.99999999999

const TINY = 1e-300  # safely > 0 in Float64
# or: const TINY = floatmin(Float64)


# Debug output toggle (env-controlled)
const WRITE_BADMASK = get(ENV, "HYDRO_WRITE_BADMASK", "0") == "1"

# ------------------------------------------------------------
# SIGN CONVENTION KNOB
# stored = DISS_SIGN * physical   (DISS_SIGN = ±1)
# ------------------------------------------------------------
const DISS_SIGN = 1.0

@inline phys_from_stored(x::Float64) = DISS_SIGN * x
@inline stored_from_phys(x::Float64) = DISS_SIGN * x
@inline phys_from_stored(x) = DISS_SIGN * x
@inline stored_from_phys(x) = DISS_SIGN * x
