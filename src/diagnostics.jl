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
    mood_stage1_bad::Int = 0
    mood_stage2_bad::Int = 0
    stage_dt_halvings::Int = 0

    last_prim_fail_cells::Int = 0
    last_floor_E_cells::Int   = 0
    last_floor_D_cells::Int   = 0
    last_nanfix_cells::Int    = 0
    last_Sr_scaled_cells::Int = 0
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
end

function diag_reset_last!(d::DiagCounters)
    d.last_prim_fail_cells = 0
    d.last_floor_E_cells   = 0
    d.last_floor_D_cells   = 0
    d.last_nanfix_cells    = 0
    d.last_Sr_scaled_cells = 0
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
    return nothing
end

@inline function diag_add!(d::DiagCounters;
                           primfail::Int=0, floorE::Int=0, floorD::Int=0,
                           nanfix::Int=0, srscaled::Int=0,
                           mood1::Int=0, mood2::Int=0,
                           halvings::Int=0)
    d.prim_fail_cells += primfail
    d.floor_E_cells   += floorE
    d.floor_D_cells   += floorD
    d.nanfix_cells    += nanfix
    d.Sr_scaled_cells += srscaled
    d.mood_stage1_bad += mood1
    d.mood_stage2_bad += mood2
    d.stage_dt_halvings += halvings

    d.last_prim_fail_cells += primfail
    d.last_floor_E_cells   += floorE
    d.last_floor_D_cells   += floorD
    d.last_nanfix_cells    += nanfix
    d.last_Sr_scaled_cells += srscaled
    d.last_mood_stage1_bad += mood1
    d.last_mood_stage2_bad += mood2
    d.last_stage_dt_halvings += halvings
    return nothing
end
