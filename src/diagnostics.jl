# ==============================================================================
# src/diagnostics.jl
#
# Diagnostic counters + helpers.
# Dependency-free (only Base).
# ==============================================================================

Base.@kwdef mutable struct DiagCounters
    prim_fail_cells::Int = 0
    floor_E_cells::Int   = 0
    floor_D_cells::Int   = 0
    nanfix_cells::Int    = 0
    Sr_scaled_cells::Int = 0
    Sr_scaled_maxratio::Float64 = 0.0
    Sr_scaled_max_i::Int = 0

    v_near_cells::Int = 0
    v_max::Float64 = 0.0

    # ∂_τ u^r gone grid-scale: the short-wavelength instability of the operator-split
    # NS target (2026-09-13, see relax_dissipative!).  `q_max` is the running maximum of
    # ‖∇²(∂_τu^r)‖/‖∂_τu^r‖ (0 = smooth, 4 = two-cell zig-zag); `steps` counts relaxation
    # substeps past the threshold.  A producer that ends with steps > 0 has ringed.
    dtau_ur_gridscale_steps::Int = 0
    dtau_ur_q_max::Float64 = 0.0

    Pi_clipped_cells::Int = 0
    Pi_clip_maxratio::Float64 = 0.0
    Pi_clip_max_i::Int = 0

    pi_clipped_cells::Int = 0
    pi_clip_maxratio::Float64 = 0.0
    pi_clip_max_i::Int = 0
    mood_stage1_bad::Int = 0
    mood_stage2_bad::Int = 0
    stage_dt_halvings::Int = 0

    last_prim_fail_cells::Int = 0
    last_floor_E_cells::Int   = 0
    last_floor_D_cells::Int   = 0
    last_nanfix_cells::Int    = 0
    last_Sr_scaled_cells::Int = 0
    last_Sr_scaled_maxratio::Float64 = 0.0
    last_Sr_scaled_max_i::Int = 0

    last_v_near_cells::Int = 0
    last_v_max::Float64 = 0.0

    last_Pi_clipped_cells::Int = 0
    last_Pi_clip_maxratio::Float64 = 0.0
    last_Pi_clip_max_i::Int = 0

    last_pi_clipped_cells::Int = 0
    last_pi_clip_maxratio::Float64 = 0.0
    last_pi_clip_max_i::Int = 0
    last_mood_stage1_bad::Int = 0
    last_mood_stage2_bad::Int = 0
    last_stage_dt_halvings::Int = 0

    # last primitive-recovery failure context (best-effort)
    last_prim_fail_i::Int = 0
    last_prim_fail_tau::Float64 = 0.0
    last_prim_fail_Dtau::Float64 = 0.0
    last_prim_fail_Sr::Float64 = 0.0
    last_prim_fail_E::Float64 = 0.0
    last_prim_fail_nur_stored::Float64 = 0.0
    last_prim_fail_Pi_stored::Float64 = 0.0
    last_prim_fail_piR_stored::Float64 = 0.0
    last_prim_fail_piEta_stored::Float64 = 0.0

    last_prim_fail_reason::PrimRecReason = PRR_UNSET
    last_prim_fail_iters::Int = 0
    last_prim_fail_resnorm::Float64 = NaN
end

function diag_reset_last!(d::DiagCounters)
    d.last_prim_fail_cells = 0
    d.last_floor_E_cells   = 0
    d.last_floor_D_cells   = 0
    d.last_nanfix_cells    = 0
    d.last_Sr_scaled_cells = 0
    d.last_Sr_scaled_maxratio = 0.0
    d.last_Sr_scaled_max_i = 0

    d.last_v_near_cells = 0
    d.last_v_max = 0.0

    d.last_Pi_clipped_cells = 0
    d.last_Pi_clip_maxratio = 0.0
    d.last_Pi_clip_max_i = 0

    d.last_pi_clipped_cells = 0
    d.last_pi_clip_maxratio = 0.0
    d.last_pi_clip_max_i = 0
    d.last_mood_stage1_bad = 0
    d.last_mood_stage2_bad = 0
    d.last_stage_dt_halvings = 0

    d.last_prim_fail_i = 0
    d.last_prim_fail_tau = 0.0
    d.last_prim_fail_Dtau = 0.0
    d.last_prim_fail_Sr = 0.0
    d.last_prim_fail_E = 0.0
    d.last_prim_fail_nur_stored = 0.0
    d.last_prim_fail_Pi_stored = 0.0
    d.last_prim_fail_piR_stored = 0.0
    d.last_prim_fail_piEta_stored = 0.0

    d.last_prim_fail_reason = PRR_UNSET
    d.last_prim_fail_iters = 0
    d.last_prim_fail_resnorm = NaN
    return nothing
end

@inline function diag_add!(d::DiagCounters;
                           primfail::Int=0, floorE::Int=0, floorD::Int=0,
                           nanfix::Int=0, srscaled::Int=0,
                           srscaled_maxratio::Float64=0.0,
                           srscaled_max_i::Int=0,
                           vnear::Int=0,
                           vmax::Float64=0.0,
                           PiClip::Int=0,
                           PiClipMax::Float64=0.0,
                           PiClipMaxI::Int=0,
                           piClip::Int=0,
                           piClipMax::Float64=0.0,
                           piClipMaxI::Int=0,
                           mood1::Int=0, mood2::Int=0,
                           halvings::Int=0)
    d.prim_fail_cells += primfail
    d.floor_E_cells   += floorE
    d.floor_D_cells   += floorD
    d.nanfix_cells    += nanfix
    d.Sr_scaled_cells += srscaled
    if srscaled_maxratio > d.Sr_scaled_maxratio
        d.Sr_scaled_maxratio = srscaled_maxratio
        d.Sr_scaled_max_i    = srscaled_max_i
    end

    d.v_near_cells += vnear
    d.v_max = max(d.v_max, vmax)

    d.Pi_clipped_cells += PiClip
    if PiClipMax > d.Pi_clip_maxratio
        d.Pi_clip_maxratio = PiClipMax
        d.Pi_clip_max_i    = PiClipMaxI
    end

    d.pi_clipped_cells += piClip
    if piClipMax > d.pi_clip_maxratio
        d.pi_clip_maxratio = piClipMax
        d.pi_clip_max_i    = piClipMaxI
    end
    d.mood_stage1_bad += mood1
    d.mood_stage2_bad += mood2
    d.stage_dt_halvings += halvings

    d.last_prim_fail_cells += primfail
    d.last_floor_E_cells   += floorE
    d.last_floor_D_cells   += floorD
    d.last_nanfix_cells    += nanfix
    d.last_Sr_scaled_cells += srscaled
    d.last_Sr_scaled_maxratio = max(d.last_Sr_scaled_maxratio, srscaled_maxratio)
    if srscaled_maxratio >= d.last_Sr_scaled_maxratio
        d.last_Sr_scaled_max_i = srscaled_max_i
    end

    d.last_v_near_cells += vnear
    d.last_v_max = max(d.last_v_max, vmax)

    d.last_Pi_clipped_cells += PiClip
    d.last_Pi_clip_maxratio = max(d.last_Pi_clip_maxratio, PiClipMax)
    if PiClipMax >= d.last_Pi_clip_maxratio
        d.last_Pi_clip_max_i = PiClipMaxI
    end

    d.last_pi_clipped_cells += piClip
    d.last_pi_clip_maxratio = max(d.last_pi_clip_maxratio, piClipMax)
    if piClipMax >= d.last_pi_clip_maxratio
        d.last_pi_clip_max_i = piClipMaxI
    end
    d.last_mood_stage1_bad += mood1
    d.last_mood_stage2_bad += mood2
    d.last_stage_dt_halvings += halvings
    return nothing
end
