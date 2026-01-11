# =========================
# src/io.jl   (STEP 4)
# =========================

using CSV
using Tables
using Interpolations
using Printf

# Depends on: T_MIN, smoothstep01_5

"""
Load initial profiles from CSV with columns:
  r, T0, and either alpha0 (if fugacity_kind=:alpha) or lambda0 (if :lambda).
Returns:
  itpT(rr)::Float64, itpF(rr)::Float64, rmax_data::Float64
with a smooth taper to vacuum over taper_width near rmax_data.
"""
function load_initial_interpolants(csvfile::AbstractString;
                                  fugacity_kind::Symbol = :alpha,
                                  taper_width::Float64 = 1.0)

    df = CSV.read(csvfile, Tables.columntable)
    @assert haskey(df, :r)
    @assert haskey(df, :T0)

    r  = Vector{Float64}(df.r)
    T0 = Vector{Float64}(df.T0)

    f0 = if fugacity_kind == :alpha
        @assert haskey(df, :alpha0)
        Vector{Float64}(df.alpha0)
    elseif fugacity_kind == :lambda
        @assert haskey(df, :lambda0)
        Vector{Float64}(df.lambda0)
    else
        error("fugacity_kind must be :alpha or :lambda")
    end

    p = sortperm(r)
    r, T0, f0 = r[p], T0[p], f0[p]

    # drop duplicate r entries (keep first)
    keep = trues(length(r))
    for i in 2:length(r)
        if r[i] == r[i-1]
            keep[i] = false
        end
    end
    r, T0, f0 = r[keep], T0[keep], f0[keep]

    rmin_data = r[1]
    rmax_data = r[end]

    itpT0 = interpolate((r,), T0, Gridded(Linear()))
    itpF0 = interpolate((r,), f0, Gridded(Linear()))

    T_floor = T_MIN
    f_vac   = (fugacity_kind == :alpha) ? 0.0 : 1.0

    tw   = max(taper_width, 0.0)
    r_t0 = rmax_data - tw

    itpT = function (rr::Float64)
        if rr <= rmin_data
            return max(itpT0(rmin_data), T_floor)
        elseif rr < r_t0 || tw == 0.0
            return max(itpT0(rr), T_floor)
        elseif rr <= rmax_data
            w = 1.0 - smoothstep01_5((rr - r_t0) / max(tw, 1e-50))
            return max(w*itpT0(rr) + (1.0-w)*T_floor, T_floor)
        else
            return T_floor
        end
    end

    itpF = function (rr::Float64)
        if rr <= rmin_data
            return itpF0(rmin_data)
        elseif rr < r_t0 || tw == 0.0
            return itpF0(rr)
        elseif rr <= rmax_data
            w = 1.0 - smoothstep01_5((rr - r_t0) / max(tw, 1e-50))
            return w*itpF0(rr) + (1.0-w)*f_vac
        else
            return f_vac
        end
    end

    return itpT, itpF, rmax_data
end

function clear_dir!(dir::AbstractString; also_meta::Bool = true)
    isdir(dir) || (mkpath(dir); return nothing)
    for f in readdir(dir; join=true)
        bn = basename(f)
        if startswith(bn, "snapshot_tau_") && endswith(bn, ".csv")
            rm(f; force=true)
        end
        if also_meta && endswith(bn, "_meta.csv")
            rm(f; force=true)
        end
    end
    return nothing
end

function write_badmask_csv(fname::AbstractString, bad::BitVector, grid::Grid1D)
    mkpath(dirname(fname))
    ng = grid.nghost
    Ntot = length(bad)
    i0 = ng + 1
    iL = Ntot - ng
    r = Float64[]
    b = Int[]
    for i in i0:iL
        push!(r, grid.rC[i])
        push!(b, bad[i] ? 1 : 0)
    end
    CSV.write(fname, (; r, bad=b))
    return nothing
end

# The two routines below are kept “as-is” to avoid solver coupling changes.
# They assume `layout(model)`, `cons_to_prim(Uv, τ, model)`, and `charge_integral_Dtau(U, grid, model)`
# exist in the same module (they can be defined later).
function write_snapshot_csv(fname, U, grid, τ, model)
    dir = dirname(fname)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    ng = grid.nghost
    L  = layout(model)
    Np = size(U,2) - 2ng

    r   = Vector{Float64}(undef, Np)
    tau = fill(τ, Np)

    Dtau = Vector{Float64}(undef, Np)
    Sr   = Vector{Float64}(undef, Np)
    EE   = Vector{Float64}(undef, Np)
    nur  = Vector{Float64}(undef, Np)

    Pi    = Vector{Float64}(undef, Np)
    piR   = Vector{Float64}(undef, Np)
    piEta = Vector{Float64}(undef, Np)
    piPhi = Vector{Float64}(undef, Np)

    D     = Vector{Float64}(undef, Np)
    T     = Vector{Float64}(undef, Np)
    mu    = Vector{Float64}(undef, Np)
    alpha = Vector{Float64}(undef, Np)
    phi   = Vector{Float64}(undef, Np)
    ur    = Vector{Float64}(undef, Np)
    v     = Vector{Float64}(undef, Np)
    n     = Vector{Float64}(undef, Np)
    e     = Vector{Float64}(undef, Np)
    P     = Vector{Float64}(undef, Np)
    ok    = Vector{Bool}(undef, Np)

    mhq = hq_mass(model.eos)

    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        j = i - ng

        # NEW: no @view, use your no-view helper
        prim = cons_to_prim_col(U, i, τ, model)

        uτ = sqrt(1 + prim.ur^2)
        vv = prim.ur / max(uτ, 1e-50)

        @inbounds begin
            r[j] = grid.rC[i]

            Dtau[j] = U[L.iDtau, i]
            Sr[j]   = U[L.iSr,   i]
            EE[j]   = U[L.iE,    i]
            nur[j]  = (L.hasNur ? U[L.iNur, i] : 0.0)

            Pi[j]    = (L.hasPi    ? U[L.iPi,    i] : 0.0)
            piR[j]   = (L.hasPiR   ? U[L.iPiR,   i] : 0.0)
            piEta[j] = (L.hasPiEta ? U[L.iPiEta, i] : 0.0)
            piPhi[j] = -piR[j] - piEta[j]

            D[j]     = U[L.iDtau, i] / max(τ, 1e-50)
            T[j]     = prim.T
            mu[j]    = prim.mu
            alpha[j] = prim.mu / max(prim.T, 1e-50)
            phi[j]   = (prim.mu - mhq) / max(prim.T, 1e-50)

            ur[j]    = prim.ur
            v[j]     = vv
            n[j]     = prim.n
            e[j]     = prim.e
            P[j]     = prim.P
            ok[j]    = prim.ok
        end
    end

    tbl = (; r, tau, Dtau, Sr, E=EE, nur, Pi, piR, piEta, piPhi,
           D, T, mu, alpha, phi, ur, v, n, e, P, ok)
    CSV.write(fname, tbl)

    meta = (; tau=[τ], Q_Dtau=[charge_integral_Dtau(U, grid, model)])
    CSV.write(replace(fname, ".csv" => "_meta.csv"), meta)

    return nothing
end
