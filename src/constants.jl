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


const T_EOS_MIN = 1e-20  # choose for numerical safety (GeV)

# --- numeric safety knobs ---
const T_MIN  = 1e-20

const T_SOLVE_MAX = 10.0
const Y_CAP       = 10000.0
const PHI_CAP     = 10000.0     # keeps μ = m + T*φ from exploding


# Vacuum / floors
const E_VAC   = 1e-20
const D_VAC   = 0.0
const E_FLOOR = E_VAC

# Controls
const DO_MOOD         = true
const SANITIZE_SCOPE  = :local
const χ_SrE           = 0.99999999

const TINY = 1e-300  # safely > 0 in Float64

# Common “positive denominator” guard (used for τ, r, uτ, T, etc.).
const EPS_POS = 1e-50

@inline posden(x::Float64) = x > EPS_POS ? x : EPS_POS
@inline safe_inv(x::Float64) = 1.0 / posden(x)
@inline safe_div(a::Float64, b::Float64) = a / posden(b)


# ------------------------------------------------------------
# Primitive recovery status (used by primrec + diagnostics)
# ------------------------------------------------------------

@enum PrimRecReason::UInt8 begin
	PRR_UNSET = 0
	PRR_VACUUM = 1
	PRR_CONVERGED = 2
	PRR_NONFINITE_INITIAL = 3
	PRR_NONFINITE_JACOBIAN = 4
	PRR_LINSOLVE_FAILED = 5
	PRR_LINESEARCH_FAILED = 6
	PRR_EOS_INVALID = 7
	PRR_RESIDUAL_TOO_LARGE = 8
	PRR_MAXIT = 9
end


# ------------------------------------------------------------
# SIGN CONVENTION KNOB
# stored = DISS_SIGN * physical   (DISS_SIGN = ±1)
# ------------------------------------------------------------
const DISS_SIGN = 1.0

@inline phys_from_stored(x::Float64) = DISS_SIGN * x
@inline stored_from_phys(x::Float64) = DISS_SIGN * x
@inline phys_from_stored(x) = DISS_SIGN * x
@inline stored_from_phys(x) = DISS_SIGN * x
