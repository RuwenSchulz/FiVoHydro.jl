# ==============================================================================
# src/state_layout.jl
#
# State layout (variable ordering / indices) + parity metadata for axis BCs.
# ==============================================================================

struct StateLayout
    names::Vector{Symbol}
    idx::Dict{Symbol,Int}

    iDtau::Int
    iSr::Int
    iE::Int

    hasNur::Bool
    iNur::Int

    hasPi::Bool
    iPi::Int

    hasPiR::Bool
    iPiR::Int

    hasPiEta::Bool
    iPiEta::Int

    # odd parity flags (e.g. Sr, nur are odd across r=0)
    odd::BitVector
end

function StateLayout(names::Vector{Symbol};
                     Dtausym::Symbol=:Dtau, Srsym::Symbol=:Sr, Esym::Symbol=:E,
                     Nursym::Symbol=:nur, Pisym::Symbol=:Pi, PiRsym::Symbol=:piR, PiEtasym::Symbol=:piEta,
                     odd_syms::Vector{Symbol} = [:Sr])

    idx = Dict(s => i for (i,s) in pairs(names))
    @assert haskey(idx, Dtausym) && haskey(idx, Srsym) && haskey(idx, Esym)

    hasNur   = haskey(idx, Nursym)
    iNur     = hasNur ? idx[Nursym] : 0

    hasPi    = haskey(idx, Pisym)
    iPi      = hasPi ? idx[Pisym] : 0

    hasPiR   = haskey(idx, PiRsym)
    iPiR     = hasPiR ? idx[PiRsym] : 0

    hasPiEta = haskey(idx, PiEtasym)
    iPiEta   = hasPiEta ? idx[PiEtasym] : 0

    odd = falses(length(names))
    for s in odd_syms
        @assert haskey(idx, s)
        odd[idx[s]] = true
    end

    return StateLayout(names, idx,
                       idx[Dtausym], idx[Srsym], idx[Esym],
                       hasNur, iNur,
                       hasPi, iPi,
                       hasPiR, iPiR,
                       hasPiEta, iPiEta,
                       odd)
end
