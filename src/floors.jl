# ==============================================================================
# src/floors.jl
#
# Floors / simple constraints. Designed to be side-effect safe:
# - only touches U (and optionally diag)
# - no hidden globals besides constants in constants.jl
# ==============================================================================

# ------------------------------------------------------------
# Floors / constraint enforcement
# ------------------------------------------------------------
function enforce_floors!(U, grid, τ, model::IdealDiffViscModel;
                         Emin::Float64=E_FLOOR,
                         diag::Union{Nothing,DiagCounters}=nothing)
    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng

    floorE = 0
    floorD = 0
    nanfix = 0

    @inbounds for i in i0:iL
        Ei = U[L.iE,i]

        # NaN/Inf energy: hard reset this cell
        if !isfinite(Ei)
            nanfix += 1
            U[L.iE,i]    = Emin
            U[L.iSr,i]   = 0.0
            U[L.iDtau,i] = 0.0
            if L.hasNur;   U[L.iNur,i]   = 0.0; end
            if L.hasPi;    U[L.iPi,i]    = 0.0; end
            if L.hasPiR;   U[L.iPiR,i]   = 0.0; end
            if L.hasPiEta; U[L.iPiEta,i] = 0.0; end
            continue
        end

        # Energy floor: hard reset momenta/densities/dissipatives
        if Ei < Emin
            floorE += 1
            U[L.iE,i]    = Emin
            U[L.iSr,i]   = 0.0
            U[L.iDtau,i] = 0.0
            if L.hasNur;   U[L.iNur,i]   = 0.0; end
            if L.hasPi;    U[L.iPi,i]    = 0.0; end
            if L.hasPiR;   U[L.iPiR,i]   = 0.0; end
            if L.hasPiEta; U[L.iPiEta,i] = 0.0; end
            continue
        end

        # Near-vacuum: treat as (effectively) empty matter.
        # This avoids ill-conditioned primitive recovery / relaxation updates when
        # E is tiny but still above the hard floor.
        if Ei <= 1e6 * Emin
            U[L.iSr,i]   = 0.0
            U[L.iDtau,i] = 0.0
            if L.hasNur;   U[L.iNur,i]   = 0.0; end
            if L.hasPi;    U[L.iPi,i]    = 0.0; end
            if L.hasPiR;   U[L.iPiR,i]   = 0.0; end
            if L.hasPiEta; U[L.iPiEta,i] = 0.0; end
        end

        # Dtau floor (must be nonnegative and finite)
        Di = U[L.iDtau,i]
        if !isfinite(Di) || Di < 0.0
            floorD += 1
            U[L.iDtau,i] = 0.0
        end

        # Sr must be finite
        Sr = U[L.iSr,i]
        if !isfinite(Sr)
            nanfix += 1
            U[L.iSr,i] = 0.0
        end

        # Dissipatives must be finite
        if L.hasNur && !isfinite(U[L.iNur,i]);     nanfix += 1; U[L.iNur,i]   = 0.0; end
        if L.hasPi  && !isfinite(U[L.iPi,i]);      nanfix += 1; U[L.iPi,i]    = 0.0; end
        if L.hasPiR && !isfinite(U[L.iPiR,i]);     nanfix += 1; U[L.iPiR,i]   = 0.0; end
        if L.hasPiEta && !isfinite(U[L.iPiEta,i]); nanfix += 1; U[L.iPiEta,i] = 0.0; end
    end

    diag === nothing || diag_add!(diag; floorE=floorE, floorD=floorD, nanfix=nanfix)
    return nothing
end

function enforce_Sr_energy_constraint!(U, grid, model::IdealDiffViscModel;
                                      χ::Float64=χ_SrE,
                                      P::Union{Nothing,AbstractVector{Float64}}=nothing,
                                      mask::Union{Nothing,BitVector}=nothing,
                                      diag::Union{Nothing,DiagCounters}=nothing)
    L  = layout(model)
    ng = grid.nghost
    i0 = ng + 1
    iL = size(U,2) - ng

    scaled = 0
    maxratio = 0.0
    maxi = 0
    @inbounds for i in i0:iL
        if mask !== nothing && !mask[i]
            continue
        end

        Ei = U[L.iE,i]
        Sr = U[L.iSr,i]
        if !(isfinite(Ei) && isfinite(Sr)) || Ei <= 0
            continue
        end

        # Physical-ish momentum bound.
        # For ideal hydro (and in many coordinate choices), a robust admissibility limit is
        #   |S| < (E + P)
        # rather than |S| < E.
        # Here we use a best-effort pressure guess (typically from the last successful primrec)
        # and add conservative viscous terms as absolute magnitudes.
        Pi_phys   = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys  = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)
        piEta_phys = (L.hasPiEta ? phys_from_stored(U[L.iPiEta,i]) : 0.0)

        Pguess = (P === nothing ? 0.0 : P[i])
        if !isfinite(Pguess)
            Pguess = 0.0
        end
        Pguess = max(Pguess, 0.0)

        Peff = Pguess
        if isfinite(Pi_phys)
            Peff += abs(Pi_phys)
        end
        if isfinite(piR_phys)
            Peff += abs(piR_phys)
        end
        if isfinite(piEta_phys)
            Peff += abs(piEta_phys)
        end

        Srmax = χ * (Ei + Peff)
        aSr   = abs(Sr)
        if aSr > Srmax
            scaled += 1
            ratio = aSr / (Srmax + TINY)
            if ratio > maxratio
                maxratio = ratio
                maxi = i
            end
            U[L.iSr,i] *= (Srmax / (aSr + TINY))
        end
    end

    diag === nothing || diag_add!(diag; srscaled=scaled, srscaled_maxratio=maxratio, srscaled_max_i=maxi)
    return nothing
end

# ------------------------------------------------------------
# Optional sanitation hook (currently no-op, but kept as a stable API)
# ------------------------------------------------------------
function sanitize_state!(U, grid, τ, model::IdealDiffViscModel;
                         Emin::Float64=E_FLOOR,
                         mask::Union{Nothing,BitVector}=nothing,
                         diag::Union{Nothing,DiagCounters}=nothing)
    return nothing
end
