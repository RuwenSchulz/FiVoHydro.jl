#!/usr/bin/env julia
# ==============================================================================
# mainDensityFrame.jl — canonical density-frame charge diffusion in FiVoHydro
#
# In the density frame the chemical potential is fixed directly from the
# on-slice charge density (μ = J^0/(χ u^0)); there is NO auxiliary diffusion
# field and NO relaxation time.  This is realized inside the self-consistent
# FiVoHydro solver by:
#
#   1. building the state with a ν-less layout  [Dtau, Sr, E, Pi, piR, piEta]
#      so primitive recovery solves  n u^τ = J^τ  (the density-frame definition);
#   2. adding the first-order-in-time parabolic diffusion flux
#          J^r_D = -κ (u^τ)^2 ∂_r α ,   κ = DsT · n / T
#      to the (advective) HLLE charge flux inside rhs! (charge_mode=:density_frame);
#   3. relying on the existing parabolic dt cap  Δτ ≲ diff_dt_coeff · Δr²/(D u^τ²).
#
# Unlike the MIS sector there is no τ_N relaxation (no non-hydrodynamic mode),
# and unlike mainBDNK.jl there are no σ_T / σ_a / ∂_τα regulator terms.
#
# Derivation of the transport coefficient from the heavy-quark Fokker–Planck
# equation: Tex/DensityFrame/df_fp_derivation.tex
# (verified numerically by Projects/DensityFrame/Code/verify_df_transport.jl).
#
# Usage (all the usual FiVo ENV knobs apply, e.g. INIT_CSV, NR, TAU0, TAUFINAL,
# DS_T, EOS, HYDRO_OUTDIR, ETA_OVER_S, ...):
#
#   ENABLE_DIFF=1 DS_T=0.2 INIT_CSV=path/to/ic.csv \
#       julia --project mainDensityFrame.jl
#
# This driver simply forces CHARGE_MODE=density_frame (and turns charge diffusion
# on by default) and then defers entirely to hydro.main().
# ==============================================================================

include(joinpath(@__DIR__, "main.jl"))

# Force the density-frame charge closure for this driver.
ENV["CHARGE_MODE"] = "density_frame"

# Charge diffusion is the whole point of this driver: enable it unless the user
# has explicitly disabled it.
get!(ENV, "ENABLE_DIFF", "1")

# The density-frame current is α-driven (J^r_D = -κ (u^τ)^2 ∂_r α).
get!(ENV, "DIFFUSION_DRIVE", "alpha")

hydro.main()
