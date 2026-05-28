#!/usr/bin/env julia
# ==============================================================================
# plot_compare_fluidum_vs_FV_minimal_scaled_n.jl
#
# Compare TWO snapshot folders (same naming: snapshot_tau_*.csv) and overlay BOTH
# for selected timesteps.
#
# IMPORTANT (Fluidum export conventions):
# - Fluidum's exported "n" is already weighted by (r*τ):  n_export = n_phys * r * τ
# - Fluidum's exported "Sr" and "E" are NOT FV conserved momentum/energy. They are
#   the heavy-quark current components already weighted by (r*τ):
#      Sr_export = J^r * r * τ,   E_export = J^τ * r * τ
#
# To compare apples-to-apples, this script:
#   1) Scales FV density by (r*τ) when plotting :n.
#   2) Adds derived fields :Jr_w and :Jtau_w (= J^r*r*τ and J^τ*r*τ).
#      For Fluidum (dataset A) these are taken from columns Sr/E.
#      For FV (dataset B) these are computed from (n, ur, v, nur).
#
# Keeps overlays for:
#   :T, :n, :alpha, :mu, :ur, :v, :P, :nur,
#   and visc if present (:Pi, :piEta, :piPhi)
# ==============================================================================

using CSV
using DataFrames
using Glob
using Plots
using LaTeXStrings
using Printf
using Logging

# ----------------------------------------------------------------------------
# USER CONFIG
# ----------------------------------------------------------------------------
const DIR_A   = "snapshots/snapshots_ideal_fluidum"
const DIR_B   = "snapshots/snapshots_ideal_diff_visc_phi"
const LABEL_A = "fluidum"
const LABEL_B = "FV"

const PAT_A = joinpath(DIR_A, "snapshot_tau_*.csv")
const PAT_B = joinpath(DIR_B, "snapshot_tau_*.csv")

const OUTDIR = "plots_compare_ideal_fluidum_vs_FV"

# Optional: restrict the shown radial range in plots (does not modify data).
# Set to `nothing` to plot the full range, or e.g. `10.0` for r ∈ [0, 10].
const R_PLOT_MAX = 20

# Grid matching for overlays.
# To compare point-by-point, resample BOTH datasets onto a shared r-grid.
# Recommended default: use dataset B's grid (often coarser / smaller rmax).
const MATCH_GRID = :B   # :A | :B

# Timesteps to compare (targets). Nearest existing snapshots will be used.
const TAU_TARGETS = [0.4,15.]
const TAU_TOL = 5e-2  # require |τnear - τtarget| <= tol

# Diagnostics thresholds (printing only)
const DIAG_T_SMALL = 1e-4
const DIAG_N_SMALL = 1e-20

# Fields to plot
const OVERLAY_SYMS = Symbol[
    :T, :n, :alpha, :mu, :ur, :v, :P, :nur,
    :Pi, :piEta, :piPhi
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
hascol(df::DataFrame, sym::Symbol) = sym in propertynames(df)
is_meta_file(f::AbstractString) = endswith(f, "_meta.csv")

function _sorted_xy(r::AbstractVector, y::AbstractVector)
    rr = Float64.(r)
    yy = Float64.(y)
    p = sortperm(rr)
    return rr[p], yy[p]
end

function _interp_linear(rsrc::Vector{Float64}, ysrc::Vector{Float64}, rtar::Vector{Float64})
    # Simple 1D linear interpolation onto rtar; out-of-range -> NaN.
    n = length(rsrc)
    out = Vector{Float64}(undef, length(rtar))
    @inbounds for i in eachindex(rtar)
        x = rtar[i]
        if !(isfinite(x)) || x < rsrc[1] || x > rsrc[end]
            out[i] = NaN
            continue
        end
        j = searchsortedlast(rsrc, x)
        if j <= 0
            out[i] = NaN
        elseif j >= n
            out[i] = ysrc[end]
        elseif rsrc[j] == x
            out[i] = ysrc[j]
        else
            x0 = rsrc[j]; x1 = rsrc[j+1]
            y0 = ysrc[j]; y1 = ysrc[j+1]
            t = (x - x0) / (x1 - x0)
            out[i] = (1 - t) * y0 + t * y1
        end
    end
    return out
end

function _resample_pair(rA::Vector{Float64}, yA::Vector{Float64}, rB::Vector{Float64}, yB::Vector{Float64})
    rA, yA = _sorted_xy(rA, yA)
    rB, yB = _sorted_xy(rB, yB)
    rcommon = (MATCH_GRID == :A) ? rA : rB
    yA_c = _interp_linear(rA, yA, rcommon)
    yB_c = _interp_linear(rB, yB, rcommon)
    return rcommon, yA_c, yB_c
end

function extract_tau_from_filename(path::AbstractString)
    bn = basename(path)
    m = match(r"snapshot_tau_([0-9]+\.[0-9]+)", bn)
    m === nothing && return NaN
    try
        return parse(Float64, m.captures[1])
    catch
        return NaN
    end
end

function normalize_columns!(df::DataFrame)
    # r
    if !hascol(df, :r)
        for c in (:rC, :rc, :R)
            if hascol(df, c)
                df.r = df[!, c]
                break
            end
        end
    end

    # tau
    if !hascol(df, :tau)
        if hascol(df, Symbol("τ"))
            df.tau = df[!, Symbol("τ")]
        elseif hascol(df, :t)
            df.tau = df.t
        end
    end

    # Sr normalization
    if !hascol(df, :Sr)
        for c in (:S_r, :SrTau, :S, :Srtau)
            if hascol(df, c)
                df.Sr = df[!, c]
                break
            end
        end
    end

    # nu_r normalization
    if !hascol(df, :nur)
        for c in (:nu_r, :nur_r, :nuR, :NuR, :nu, :nuRr)
            if hascol(df, c)
                df.nur = df[!, c]
                break
            end
        end
    end

    # visc normalization (aliases -> standard)
    if !hascol(df, :Pi)
        for c in (:PI, :bulk, :Bulk, :bulkPi, :piBulk)
            if hascol(df, c)
                df.Pi = df[!, c]
                break
            end
        end
    end
    if !hascol(df, :piR)
        for c in (:pir, :pi_r, :piRR, :PiR, :shearR, :piRadial)
            if hascol(df, c)
                df.piR = df[!, c]
                break
            end
        end
    end
    if !hascol(df, :piEta)
        for c in (:piη, :pi_eta, :piE, :PiEta, :piLong, :piZ)
            if hascol(df, c)
                df.piEta = df[!, c]
                break
            end
        end
    end
    if !hascol(df, :piPhi)
        for c in (:piφ, :pi_phi, :piP, :PiPhi, :piAzimuth)
            if hascol(df, c)
                df.piPhi = df[!, c]
                break
            end
        end
    end

    # If πϕ missing but πr and πη exist, synthesize traceless πϕ = -πr - πη
    if !hascol(df, :piPhi) && hascol(df, :piR) && hascol(df, :piEta)
        df.piPhi = .-(Float64.(df.piR) .+ Float64.(df.piEta))
    end

    # v_r: derive from ur if missing
    if !hascol(df, :v) && hascol(df, :ur)
        ur = Float64.(df.ur)
        df.v = ur ./ sqrt.(1 .+ ur.^2)
    end

    # ----------------------------------------------------------------
    # Fluidum-style weighted currents:
    #   Jr_w   = J^r * r * τ
    #   Jtau_w = J^τ * r * τ
    #
    # - For Fluidum snapshots, columns Sr/E already store these.
    # - For FV snapshots, compute from primitives if possible.
    #   NOTE: FV snapshot column :nur is stored; for DISS_SIGN=+1, stored==physical.
    # ----------------------------------------------------------------

    # Map Fluidum's Sr/E -> Jr_w/Jtau_w (only if those exist)
    if !hascol(df, :Jr_w) && hascol(df, :Sr)
        df.Jr_w = df[!, :Sr]
    end
    if !hascol(df, :Jtau_w) && hascol(df, :E)
        df.Jtau_w = df[!, :E]
    end

    # FV-derived Jr_w/Jtau_w if we have r,tau,n,ur,v,nur
    if hascol(df, :r) && hascol(df, :tau) && hascol(df, :n) && hascol(df, :ur) && hascol(df, :v) && hascol(df, :nur)
        # Only compute if they were not already set (avoid clobbering Fluidum's meaning).
        if !hascol(df, :Jr_w) || !hascol(df, :Jtau_w)
            r = Float64.(df.r)
            τ = Float64.(df.tau)
            n = Float64.(df.n)
            ur = Float64.(df.ur)
            v  = Float64.(df.v)
            nur = Float64.(df.nur)

            uτ = sqrt.(1 .+ ur.^2)
            # J^r = n*ur + ν^r
            Jr = n .* ur .+ nur
            # ν^τ = v * ν^r ;  J^τ = n*uτ + ν^τ
            Jtau = n .* uτ .+ v .* nur
            w = r .* τ

            if !hascol(df, :Jr_w)
                df.Jr_w = Jr .* w
            end
            if !hascol(df, :Jtau_w)
                df.Jtau_w = Jtau .* w
            end
        end
    end

    return df
end

function extract_tau_from_df(df::DataFrame; fallback_file::Union{Nothing,String}=nothing)
    if hascol(df, :tau)
        return Float64(df.tau[1])
    end
    fallback_file === nothing && error("No :tau and no filename fallback")
    τf = extract_tau_from_filename(fallback_file)
    isfinite(τf) || error("Could not extract τ from file name: $fallback_file")
    return τf
end

function ylabel_for(sym::Symbol)
    sym == :ur    && return L"u_r"
    sym == :v     && return L"v_r"
    sym == :Sr    && return L"S_r"
    sym == :nur   && return L"\nu_r"
    sym == :Jr_w  && return L"J^r\,r\,\tau"
    sym == :Jtau_w && return L"J^\tau\,r\,\tau"
    sym == :Pi    && return L"\Pi"
    sym == :piR   && return L"\pi_r"
    sym == :piEta && return L"\pi_\eta"
    sym == :piPhi && return L"\pi_\phi"
    sym == :T     && return L"T"
    sym == :P     && return L"P"
    sym == :n     && return L"n"
    sym == :mu    && return L"\mu"
    sym == :alpha && return L"\alpha=\mu/T"
    return string(sym)
end

function saveplot_both(p, outdir::AbstractString, bn::AbstractString)
    mkpath(outdir)
    png = joinpath(outdir, bn * ".png")
    pdf = joinpath(outdir, bn * ".pdf")
    savefig(p, png)
    #savefig(p, pdf)
    return (png=png, pdf=pdf)
end

function load_snapshots(pattern::String)
    files_all = sort(glob(pattern))
    files = [f for f in files_all if !is_meta_file(f)]
    isempty(files) && error("No snapshots for pattern: $pattern")

    taus = Float64[]
    for f in files
        # Fast path: most files encode τ in the filename.
        τ = extract_tau_from_filename(f)
        if !isfinite(τ)
            # Fallback: read the file and try to extract τ from columns.
            df = normalize_columns!(CSV.read(f, DataFrame))
            τ = try
                extract_tau_from_df(df; fallback_file=f)
            catch
                NaN
            end
        end
        push!(taus, τ)
    end

    if all(isfinite, taus)
        p = sortperm(taus)
        return files[p], taus[p]
    end
    return files, taus
end

function find_nearest_snapshot(files::Vector{String}, taus::Vector{Float64}, τtarget::Float64; tol::Float64=TAU_TOL)
    isempty(files) && return (nothing, NaN)
    good = findall(isfinite, taus)
    isempty(good) && return (nothing, NaN)

    τg = taus[good]
    idxg = good[argmin(abs.(τg .- τtarget))]
    τnear = taus[idxg]

    if abs(τnear - τtarget) > tol
        @warn "No snapshot within tolerance" τtarget=τtarget τnear=τnear tol=tol
        return (nothing, τnear)
    end
    return (files[idxg], τnear)
end

# Diagnostics (printing only)
function col_minmax(df::DataFrame, sym::Symbol)
    hascol(df, sym) || return (NaN, NaN)
    x = Float64.(df[!, sym])
    x = x[isfinite.(x)]
    isempty(x) && return (NaN, NaN)
    return (minimum(x), maximum(x))
end

function diag_snapshot(df::DataFrame, τ::Float64; scale_n_by_rτ::Bool=false)
    normalize_columns!(df)
    n_all = nrow(df)

    Tmin, Tmax = col_minmax(df, :T)
    amin, amax = col_minmax(df, :alpha)
    Pmin, Pmax = col_minmax(df, :P)
    urmin, urmax = col_minmax(df, :ur)
    vmin, vmax = col_minmax(df, :v)
    nurmin, nurmax = col_minmax(df, :nur)
    Srmin, Srmax = col_minmax(df, :Sr)

    # n stats (optionally scaled by r*tau)
    nmin, nmax = (NaN, NaN)
    n_Tsmall = hascol(df, :T) ? count(t -> isfinite(t) && t < DIAG_T_SMALL, Float64.(df.T)) : 0
    n_nsmall = 0
    if hascol(df, :n)
        nraw = Float64.(df.n)
        if scale_n_by_rτ
            hascol(df, :r) || error("diag_snapshot: need :r to scale n by r*tau")
            r = Float64.(df.r)
            nuse = nraw .* (r .* τ)
        else
            nuse = nraw
        end
        nuse_f = nuse[isfinite.(nuse)]
        if !isempty(nuse_f)
            nmin, nmax = minimum(nuse_f), maximum(nuse_f)
        end
        n_nsmall = count(x -> isfinite(x) && x < DIAG_N_SMALL, nuse)
    end

    Pimin, Pimax       = col_minmax(df, :Pi)
    piRmin, piRmax     = col_minmax(df, :piR)
    piEtamin, piEtamax = col_minmax(df, :piEta)
    piPhimin, piPhimax = col_minmax(df, :piPhi)

    return (; τ, n_all,
             Tmin, Tmax, nmin, nmax, amin, amax, Pmin, Pmax,
             urmin, urmax, vmin, vmax, Srmin, Srmax, nurmin, nurmax,
             Pimin, Pimax, piRmin, piRmax, piEtamin, piEtamax, piPhimin, piPhimax,
             n_Tsmall, n_nsmall)
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
function main()
    Plots.default(
        linewidth  = 2,
        framestyle = :box,
        fontfamily = "Computer Modern",
        #legend     = auto,
        size       = (900, 500)
    )
    mkpath(OUTDIR)

    filesA, tausA = load_snapshots(PAT_A)
    filesB, tausB = load_snapshots(PAT_B)
    println("Found $(length(filesA)) snapshots in $DIR_A")
    println("Found $(length(filesB)) snapshots in $DIR_B")

    # Select nearest pairs for each τ target
    sel = NamedTuple[]
    for τt in TAU_TARGETS
        fA, τA = find_nearest_snapshot(filesA, tausA, τt; tol=TAU_TOL)
        fB, τB = find_nearest_snapshot(filesB, tausB, τt; tol=TAU_TOL)
        if fA === nothing || fB === nothing
            @warn "Skipping τtarget (missing in one dataset within tol)" τtarget=τt fA=fA fB=fB
            continue
        end
        push!(sel, (τtarget=τt, fA=fA, τA=τA, fB=fB, τB=τB))
    end
    isempty(sel) && error("No τ targets matched within TAU_TOL=$(TAU_TOL). Adjust TAU_TOL or TAU_TARGETS.")

    # Cache normalized snapshots for selected pairs to avoid re-reading CSVs for every plotted field.
    pairs = NamedTuple[]
    for s in sel
        dA = normalize_columns!(CSV.read(s.fA, DataFrame))
        dB = normalize_columns!(CSV.read(s.fB, DataFrame))
        push!(pairs, (; s..., dA=dA, dB=dB))
    end

    # Diagnostics
    println("\nDiagnostics (printing only; no masking):")
    @printf("  thresholds: T < %.1e small-count, n < %.1e small-count\n", DIAG_T_SMALL, DIAG_N_SMALL)
    println("  NOTE: Fluidum exports n as n_phys*(r*tau); FV is scaled to match when plotting :n")
    println("  NOTE: Fluidum exports Sr/E as J^r*(r*tau), J^tau*(r*tau); FV derived fields are :Jr_w, :Jtau_w")
    println("  NOTE: overlay plots resample both datasets to MATCH_GRID=$(MATCH_GRID) grid")

    for s in pairs
        dgA = diag_snapshot(s.dA, s.τA; scale_n_by_rτ=false)
        dgB = diag_snapshot(s.dB, s.τB; scale_n_by_rτ=true)

        @printf("  τtarget=%6.3f | %s: τ=%7.3f n=%5d  T[%.3e,%.3e]  n[%.3e,%.3e]  α[%.3e,%.3e]  P[%.3e,%.3e]  ur[%.3e,%.3e]  v[%.3e,%.3e]  Sr[%.3e,%.3e]  νr[%.3e,%.3e]  Π[%.3e,%.3e]  πr[%.3e,%.3e]  πη[%.3e,%.3e]  πφ[%.3e,%.3e]  #Tsmall=%d #nsmall=%d\n",
            s.τtarget, LABEL_A, dgA.τ, dgA.n_all,
            dgA.Tmin, dgA.Tmax, dgA.nmin, dgA.nmax, dgA.amin, dgA.amax, dgA.Pmin, dgA.Pmax,
            dgA.urmin, dgA.urmax, dgA.vmin, dgA.vmax, dgA.Srmin, dgA.Srmax, dgA.nurmin, dgA.nurmax,
            dgA.Pimin, dgA.Pimax, dgA.piRmin, dgA.piRmax, dgA.piEtamin, dgA.piEtamax, dgA.piPhimin, dgA.piPhimax,
            dgA.n_Tsmall, dgA.n_nsmall)

        @printf("              | %s: τ=%7.3f n=%5d  T[%.3e,%.3e]  n_scaled[%.3e,%.3e]  α[%.3e,%.3e]  P[%.3e,%.3e]  ur[%.3e,%.3e]  v[%.3e,%.3e]  Sr[%.3e,%.3e]  νr[%.3e,%.3e]  Π[%.3e,%.3e]  πr[%.3e,%.3e]  πη[%.3e,%.3e]  πφ[%.3e,%.3e]  #Tsmall=%d #nsmall=%d\n",
            LABEL_B, dgB.τ, dgB.n_all,
            dgB.Tmin, dgB.Tmax, dgB.nmin, dgB.nmax, dgB.amin, dgB.amax, dgB.Pmin, dgB.Pmax,
            dgB.urmin, dgB.urmax, dgB.vmin, dgB.vmax, dgB.Srmin, dgB.Srmax, dgB.nurmin, dgB.nurmax,
            dgB.Pimin, dgB.Pimax, dgB.piRmin, dgB.piRmax, dgB.piEtamin, dgB.piEtamax, dgB.piPhimin, dgB.piPhimax,
            dgB.n_Tsmall, dgB.n_nsmall)
    end
    println()

    # Overlays for selected fields
    for sym in OVERLAY_SYMS
        p = plot(
            xlabel = L"r\;[\mathrm{fm}]",
            ylabel = ylabel_for(sym),
            title  = @sprintf("%s(r)  %s vs %s  (τ targets: %s)",
                              string(sym), LABEL_A, LABEL_B,
                              join((@sprintf("%.3g", x) for x in TAU_TARGETS), ", ")),
            #legend = :topright
        )

        if R_PLOT_MAX !== nothing
            xlims!(p, (0.0, Float64(R_PLOT_MAX)))
        end

        any_added = false
        for s in pairs
            dA = s.dA
            dB = s.dB

            if !hascol(dA, :r) || !hascol(dB, :r)
                @warn "Missing :r; skip" sym=sym τtarget=s.τtarget
                continue
            end
            if !hascol(dA, sym) || !hascol(dB, sym)
                @info "Skip sym at τ (missing col in one dataset)" sym=sym τtarget=s.τtarget hasA=hascol(dA,sym) hasB=hascol(dB,sym)
                continue
            end

            rA = Float64.(dA.r); yA = Float64.(dA[!, sym])
            rB = Float64.(dB.r); yB = Float64.(dB[!, sym])

            # Resample both datasets to the same r-grid (same N and rmax).
            rC, yA, yB = _resample_pair(rA, yA, rB, yB)

            # Apply requested scaling ONLY for dataset B density
            if sym == :n
                yB = yB .* (rC .* s.τB)
            end

            #if sym == :nur
            #    yB = -yB /0.1973269804#.* (s.τB) # scale νr by τ only
            #end 
            # fluidum dashed
            plot!(p, rC, yA, label=@sprintf("%s τ=%.3f", LABEL_A, s.τA), linestyle=:dash)
            plot!(p, rC, yB, label=@sprintf("%s τ=%.3f", LABEL_B, s.τB))
            any_added = true
        end

        if any_added
            out = saveplot_both(
                p, OUTDIR,
                "compare_overlay_$(sym)_taus_" * join((@sprintf("%g", x) for x in TAU_TARGETS), "_")
            )
            @info "saved" sym=sym png=out.png pdf=out.pdf
        else
            @info "no curves added; skip saving" sym=sym
        end
    end

    println("\nAll comparison plots saved in: $OUTDIR")
end

main()
