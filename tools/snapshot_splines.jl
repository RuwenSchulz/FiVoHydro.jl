"""tools/snapshot_splines.jl

Helpers to read FiVoHydro `snapshot_tau_*.csv` outputs and build splines.

Main entry point (what you asked for):

        r_grid, tau_grid, spl = load_snapshot_dir_spline2d("snapshots/snapshots_ideal_diff_visc_phi")
        T = spl[:T](r, tau)

This returns one `Dierckx.Spline2D` per numeric field (excluding `:r`, `:tau`, `:ok`).

Notes:
- Assumes files like `snapshot_tau_XX.XXX.csv` (not `*_meta.csv`).
- Requires the snapshots to live on a consistent 1D r-grid across τ, or you can
    enable linear resampling onto the first snapshot's r-grid.
"""

using CSV
using Tables
using Dierckx: Spline1D, Spline2D

const _DEFAULT_EXCLUDE = Set{Symbol}((:r, :tau, :ok))

is_snapshot_csv(path::AbstractString) = startswith(basename(path), "snapshot_tau_") && endswith(path, ".csv") && !endswith(path, "_meta.csv")

function _sorted_perm(r_raw::AbstractVector{<:Real})
    r = Float64.(r_raw)
    p = sortperm(r)
    return r, p
end

function _interp_linear(xsrc::Vector{Float64}, ysrc::Vector{Float64}, xtar::Vector{Float64})
    # Linear interpolation; out-of-range -> NaN.
    n = length(xsrc)
    out = Vector{Float64}(undef, length(xtar))
    @inbounds for i in eachindex(xtar)
        x = xtar[i]
        if !(isfinite(x)) || x < xsrc[1] || x > xsrc[end]
            out[i] = NaN
            continue
        end
        j = searchsortedlast(xsrc, x)
        if j <= 0
            out[i] = NaN
        elseif j >= n
            out[i] = ysrc[end]
        elseif xsrc[j] == x
            out[i] = ysrc[j]
        else
            x0 = xsrc[j]; x1 = xsrc[j+1]
            y0 = ysrc[j]; y1 = ysrc[j+1]
            t = (x - x0) / (x1 - x0)
            out[i] = (1 - t) * y0 + t * y1
        end
    end
    return out
end

"""
    load_snapshot_splines(snapshot_csv; k=1, exclude=Set([:r,:tau,:ok])) -> (tau, r_sorted, splines)

Reads a single FiVoHydro `snapshot_tau_*.csv` and returns:
- `tau::Float64`: snapshot time
- `r_sorted::Vector{Float64}`: sorted radial coordinates
- `splines::Dict{Symbol,Spline1D}`: one spline per *numeric* column (excluding `exclude`)

Columns with fewer than 2 finite points are skipped.
"""
function load_snapshot_splines(snapshot_csv::AbstractString; k::Int=1, exclude::AbstractSet{Symbol}=_DEFAULT_EXCLUDE)
    tbl = Tables.columntable(CSV.File(snapshot_csv))

    haskey(tbl, :r)   || error("Missing column :r in $snapshot_csv")
    haskey(tbl, :tau) || error("Missing column :tau in $snapshot_csv")

    r_raw, p = _sorted_perm(tbl.r)
    τ = Float64(first(tbl.tau))
    r = r_raw[p]

    spl = Dict{Symbol,Spline1D}()
    for sym in propertynames(tbl)
        sym in exclude && continue

        col = getproperty(tbl, sym)
        eltype(col) <: Real || continue

        y = Float64.(col)[p]
        mask = isfinite.(r) .& isfinite.(y)
        if count(mask) >= 2
            spl[sym] = Spline1D(r[mask], y[mask]; k=k)
        end
    end

    return τ, r, spl
end

"""
    load_snapshot_dir_splines(dir; k=1, exclude=Set([:r,:tau,:ok])) -> (taus, spl_by_tau)

Loads all `snapshot_tau_*.csv` files in `dir` (excluding `*_meta.csv`), builds
splines for all numeric fields, and returns:
- `taus::Vector{Float64}` sorted
- `spl_by_tau::Dict{Float64,Dict{Symbol,Spline1D}}`

If multiple files have the same `tau` value (unlikely), later ones overwrite.
"""
function load_snapshot_dir_splines(dir::AbstractString; k::Int=1, exclude::AbstractSet{Symbol}=_DEFAULT_EXCLUDE)
    isdir(dir) || error("Not a directory: $dir")

    files = filter(is_snapshot_csv, readdir(dir; join=true))
    isempty(files) && error("No snapshot_tau_*.csv found in $dir")

    taus = Float64[]
    out = Dict{Float64,Dict{Symbol,Spline1D}}()

    # Sort by filename for determinism
    sort!(files)
    for f in files
        τ, _, spl = load_snapshot_splines(f; k=k, exclude=exclude)
        out[τ] = spl
        push!(taus, τ)
    end

    sort!(unique!(taus))
    return taus, out
end

"""
    load_snapshot_dir_spline2d(dir; kx=1, ky=1, exclude=Set([:r,:tau,:ok]),
                               rgrid_mode=:strict, rtol=1e-12, atol=0.0)
        -> (r_grid, tau_grid, splines)

Builds `Dierckx.Spline2D` in `(r, τ)` for every numeric field column in the snapshots.

`rgrid_mode`:
- `:strict` (default): require identical r-grid in every snapshot.
- `:resample`: linearly resample each snapshot onto the first snapshot's r-grid.

Returns:
- `r_grid::Vector{Float64}`
- `tau_grid::Vector{Float64}` (sorted)
- `splines::Dict{Symbol,Spline2D}` mapping field -> spline
"""
function load_snapshot_dir_spline2d(
    dir::AbstractString;
    kx::Int=1,
    ky::Int=1,
    exclude::AbstractSet{Symbol}=_DEFAULT_EXCLUDE,
    rgrid_mode::Symbol=:strict,
    rtol::Float64=1e-12,
    atol::Float64=0.0,
)
    isdir(dir) || error("Not a directory: $dir")
    files = filter(is_snapshot_csv, readdir(dir; join=true))
    isempty(files) && error("No snapshot_tau_*.csv found in $dir")
    sort!(files)

    # Read first snapshot: define r-grid and field list
    tbl0 = Tables.columntable(CSV.File(files[1]))
    haskey(tbl0, :r)   || error("Missing column :r in $(files[1])")
    haskey(tbl0, :tau) || error("Missing column :tau in $(files[1])")
    r0_raw, p0 = _sorted_perm(tbl0.r)
    r0 = r0_raw[p0]

    # Determine numeric fields to spline (from first snapshot)
    fields = Symbol[]
    for sym in propertynames(tbl0)
        sym in exclude && continue
        col = getproperty(tbl0, sym)
        (eltype(col) <: Real) || continue
        push!(fields, sym)
    end
    isempty(fields) && error("No numeric fields found in $(files[1]) after excluding $(collect(exclude)).")

    taus = Vector{Float64}(undef, length(files))
    # data[field] is Nr x Nt (r as first dimension)
    data = Dict{Symbol,Matrix{Float64}}(sym => Matrix{Float64}(undef, length(r0), length(files)) for sym in fields)

    for (it, f) in pairs(files)
        tbl = Tables.columntable(CSV.File(f))
        haskey(tbl, :r)   || error("Missing column :r in $f")
        haskey(tbl, :tau) || error("Missing column :tau in $f")

        r_raw, p = _sorted_perm(tbl.r)
        r = r_raw[p]
        τ = Float64(first(tbl.tau))
        taus[it] = τ

        same_grid = (length(r) == length(r0)) && all(isapprox.(r, r0; rtol=rtol, atol=atol))
        if !same_grid
            if rgrid_mode == :strict
                error("r-grid mismatch in $f. Set rgrid_mode=:resample to interpolate onto the first snapshot grid.")
            elseif rgrid_mode != :resample
                error("rgrid_mode must be :strict or :resample; got $(rgrid_mode)")
            end
        end

        for sym in fields
            haskey(tbl, sym) || error("Missing field $(sym) in $f")
            y = Float64.(getproperty(tbl, sym))[p]
            if same_grid
                data[sym][:, it] = y
            else
                data[sym][:, it] = _interp_linear(r, y, r0)
            end
        end
    end

    # Sort by τ and permute all data accordingly
    pt = sortperm(taus)
    tau_grid = taus[pt]

    spl = Dict{Symbol,Spline2D}()
    for sym in fields
        A = data[sym][:, pt]
        if any(!isfinite, A)
            error("Non-finite values encountered for field $(sym); cannot build Spline2D. (Maybe rgrid_mode=:strict with missing values, or resampling produced NaNs out of range.)")
        end
        spl[sym] = Spline2D(r0, tau_grid, A; kx=kx, ky=ky)
    end

    return r0, tau_grid, spl
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) >= 1 || error("usage: julia tools/snapshot_splines.jl <snapshot_dir>  (a directory of snapshot_tau_*.csv written by run_sim_ideal_diff_visc)")
    isdir(ARGS[1]) || error("snapshot directory not found: $(ARGS[1])")
    r_grid, tau_grid, spl = load_snapshot_dir_spline2d(ARGS[1]; kx=1, ky=1)
    @show tau_grid[1]
    @show spl[:T](r_grid[10], tau_grid[1])
end
