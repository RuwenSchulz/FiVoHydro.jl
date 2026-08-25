# =========================
# src/io.jl — CSV/JLD2 snapshot writing, initial-profile interpolants, spline export
# =========================

using CSV
using Tables
using Interpolations
using Printf
using Statistics
using Dates
using JLD2
using Dierckx: Spline2D

const SPLINE_EXPORT_EXCLUDED_FIELDS = Set([
    :ddp,
    :D,
    :ddp_conformal,
    :dp,
    :dp_conformal,
    :Dtau,
    :e,
    :E,
    :mu,
    :nu_r,
    :ok,
    :p,
    :P,
    :p_conformal,
    :Phi,
    :piEta,
    :piPhi,
    :piR,
    :Sr,
])

# Depends on: T_MIN, smoothstep01_5

"""
Load initial profiles from CSV with columns:
    - Required: r, T0, and either alpha0 (if fugacity_kind=:alpha) or lambda0 (if :lambda).
    - Optional: one of nu_r0 / nu_r (physical) or nur0 / nur (stored sign convention).

Returns:
    itpT(rr)::Float64, itpF(rr)::Float64, itpNurStored(rr)::Float64, rmax_data::Float64
with a smooth taper to vacuum over taper_width near rmax_data.

Notes:
    - If the CSV provides physical nu_r (nu_r0/nu_r), it is converted to stored via stored_from_phys.
    - If no diffusion-current column is present, itpNurStored(rr) returns 0.0.
"""
function load_initial_interpolants(csvfile::AbstractString;
                                  fugacity_kind::Symbol = :alpha,
                                  taper_width::Float64 = 1.0,
                                  interp_kind::Symbol = :linear,
                                  interp_dr::Union{Nothing,Float64} = nothing,
                                  dup_r_tol::Float64 = 0.0,
                                  strict_csv::Bool = true)

    # Parse strictly for consistency (fail fast on malformed / non-numeric entries)
    types = Dict{Symbol,Type}(:r => Float64, :T0 => Float64)
    if fugacity_kind == :alpha
        types[:alpha0] = Float64
    elseif fugacity_kind == :lambda
        types[:lambda0] = Float64
    end
    # Optional diffusion-current columns (ignored if not present in the file).
    types[:nu_r0] = Float64
    types[:nu_r]  = Float64
    types[:nur0]  = Float64
    types[:nur]   = Float64

    # `validate=false` lets us provide type hints for optional columns
    # without erroring when they are absent from the CSV.
    df = Tables.columntable(CSV.File(csvfile; types=types, strict=strict_csv, validate=false))
    @assert haskey(df, :r)
    @assert haskey(df, :T0)

    r  = collect(df.r)
    T0 = collect(df.T0)

    f0 = if fugacity_kind == :alpha
        @assert haskey(df, :alpha0)
        collect(df.alpha0)
    elseif fugacity_kind == :lambda
        @assert haskey(df, :lambda0)
        collect(df.lambda0)
    else
        error("fugacity_kind must be :alpha or :lambda")
    end

    # Optional: diffusion current profile.
    # Convention: return an interpolant for the stored value (matches state storage).
    nur_stored = zeros(Float64, length(r))
    have_nur = false
    if haskey(df, :nu_r0)
        nur_stored .= stored_from_phys.(collect(df.nu_r0))
        have_nur = true
    elseif haskey(df, :nu_r)
        nur_stored .= stored_from_phys.(collect(df.nu_r))
        have_nur = true
    elseif haskey(df, :nur0)
        nur_stored .= collect(df.nur0)
        have_nur = true
    elseif haskey(df, :nur)
        nur_stored .= collect(df.nur)
        have_nur = true
    end

    if length(r) < 2
        error("Initial profile CSV must contain at least 2 rows; got $(length(r)).")
    end
    if any(!isfinite, r) || any(!isfinite, T0) || any(!isfinite, f0)
        error("Initial profile CSV contains NaN/Inf values in r/T0/(alpha0|lambda0).")
    end
    if have_nur && any(!isfinite, nur_stored)
        error("Initial profile CSV contains NaN/Inf values in (nu_r0|nu_r|nur0|nur).")
    end

    p = sortperm(r)
    r, T0, f0 = r[p], T0[p], f0[p]
    if have_nur
        nur_stored = nur_stored[p]
    end

    # Drop duplicate r entries (keep first). Optionally tolerate near-equal duplicates.
    # NOTE: duplicates make the profile non-single-valued in r and will otherwise break gridded interpolation.
    keep = trues(length(r))
    ndups = 0
    tol = max(dup_r_tol, 0.0)
    for i in 2:length(r)
        if abs(r[i] - r[i-1]) <= tol
            keep[i] = false
            ndups += 1
        end
    end
    if ndups > 0
        @warn "Initial profile CSV has duplicate/near-duplicate r values; keeping first occurrence" csvfile=csvfile ndups=ndups dup_r_tol=tol
    end
    r, T0, f0 = r[keep], T0[keep], f0[keep]
    if have_nur
        nur_stored = nur_stored[keep]
    end

    if length(r) < 2
        error("After removing duplicate r entries, fewer than 2 points remain; cannot build interpolants.")
    end

    rmin_data = r[1]
    rmax_data = r[end]

    if fugacity_kind == :lambda
        nbad = count(x -> x <= 0.0, f0)
        if nbad > 0
            @warn "lambda0 contains non-positive values; clamping to TINY to avoid log(λ) issues" csvfile=csvfile nbad=nbad
            @inbounds for i in eachindex(f0)
                f0[i] = max(f0[i], TINY)
            end
        end
    end

    # Base gridded interpolation over the raw (possibly non-uniform) CSV abscissae.
    itpT_lin = interpolate((r,), T0, Gridded(Linear()))
    itpF_lin = interpolate((r,), f0, Gridded(Linear()))
    itpNur_lin = interpolate((r,), nur_stored, Gridded(Linear()))

    itpT0 = itpT_lin
    itpF0 = itpF_lin
    itpNur0 = itpNur_lin

    # Optional: resample onto a uniform grid and build a cubic B-spline interpolant.
    # This makes first/second derivatives far more meaningful than with piecewise-linear data,
    # at the cost of introducing a smooth approximation.
    if interp_kind == :cubic
        dr = if interp_dr === nothing
            # Robust default spacing.
            dif = diff(r)
            isempty(dif) ? (rmax_data - rmin_data) : median(dif)
        else
            interp_dr
        end
        dr = max(Float64(dr), (rmax_data - rmin_data) / 10_000)

        N = max(2, Int(floor((rmax_data - rmin_data) / dr + 0.5)) + 1)
        ru = range(rmin_data, rmax_data; length=N)

        Tu = [Float64(itpT_lin(x)) for x in ru]
        Fu = [Float64(itpF_lin(x)) for x in ru]
        Nu = [Float64(itpNur_lin(x)) for x in ru]

        # Build cubic B-splines on a uniform grid.
        itpT_bs = scale(interpolate(Tu, BSpline(Cubic(Line(OnGrid())))), ru)
        itpF_bs = scale(interpolate(Fu, BSpline(Cubic(Line(OnGrid())))), ru)
        itpNur_bs = scale(interpolate(Nu, BSpline(Cubic(Line(OnGrid())))), ru)

        itpT0 = itpT_bs
        itpF0 = itpF_bs
        itpNur0 = itpNur_bs
    elseif interp_kind == :linear
        # keep defaults
    else
        error("interp_kind must be :linear or :cubic; got $(interp_kind)")
    end

    T_floor = T_MIN
    f_vac   = (fugacity_kind == :alpha) ? 0.0 : 1.0

    tw   = max(taper_width, 0.0)
    # Ensure taper does not start before available data range.
    r_t0 = max(rmax_data - tw, rmin_data)
    inv_tw = safe_inv(tw)

    itpT = function (rr::Float64)
        if rr <= rmin_data
            return max(itpT0(rmin_data), T_floor)
        elseif rr <= rmax_data
            if tw == 0.0 || rr < r_t0
                return max(itpT0(rr), T_floor)
            else
                w = 1.0 - smoothstep01_5((rr - r_t0) * inv_tw)
                return max(w*itpT0(rr) + (1.0-w)*T_floor, T_floor)
            end
        else
            return T_floor
        end
    end

    itpF = function (rr::Float64)
        if rr <= rmin_data
            return itpF0(rmin_data)
        elseif rr <= rmax_data
            if tw == 0.0 || rr < r_t0
                return itpF0(rr)
            else
                w = 1.0 - smoothstep01_5((rr - r_t0) * inv_tw)
                return w*itpF0(rr) + (1.0-w)*f_vac
            end
        else
            return f_vac
        end
    end

    itpNurStored = function (rr::Float64)
        # Vacuum value is always 0.0.
        if rr <= rmin_data
            return itpNur0(rmin_data)
        elseif rr <= rmax_data
            if tw == 0.0 || rr < r_t0
                return itpNur0(rr)
            else
                w = 1.0 - smoothstep01_5((rr - r_t0) * inv_tw)
                return w*itpNur0(rr)
            end
        else
            return 0.0
        end
    end

    return itpT, itpF, itpNurStored, rmax_data
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
    nur  = Vector{Float64}(undef, Np)   # stored (sign convention)
    nu_r = Vector{Float64}(undef, Np)   # physical

    uτv  = Vector{Float64}(undef, Np)
    nu_τ = Vector{Float64}(undef, Np)
    Jtau = Vector{Float64}(undef, Np)
    Jr   = Vector{Float64}(undef, Np)

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
    kappa = Vector{Float64}(undef, Np)
    tau_n = Vector{Float64}(undef, Np)

    mhq = hq_mass(model.eos)
    no_charge_snapshot = !eos_has_charge(model.eos) && !L.hasNur

    Threads.@threads for i in (ng+1):(size(U,2)-ng)
        j = i - ng

        # NEW: no @view, use your no-view helper
        prim = cons_to_prim_col(U, i, grid.rC[i], τ, model)

        uτ = sqrt(1 + prim.ur^2)
        vv = safe_div(prim.ur, uτ)

        @inbounds begin
            r[j] = grid.rC[i]

            Dtau[j] = U[L.iDtau, i]
            Sr[j]   = U[L.iSr,   i]
            EE[j]   = U[L.iE,    i]
            nur_stored = (L.hasNur ? U[L.iNur, i] : 0.0)
            nur[j]  = nur_stored

            # Physical diffusion current component ν^r
            nur_phys = phys_from_stored(nur_stored)
            nu_r[j] = nur_phys

            Pi[j]    = (L.hasPi    ? U[L.iPi,    i] : 0.0)
            piR[j]   = (L.hasPiR   ? U[L.iPiR,   i] : 0.0)
            piEta[j] = (L.hasPiEta ? U[L.iPiEta, i] : 0.0)
            piPhi[j] = -piR[j] - piEta[j]

            D[j]     = safe_div(U[L.iDtau, i], τ)
            T[j]     = prim.T
            mu[j]    = prim.mu
            alpha[j] = safe_div(prim.mu, prim.T)
            phi[j]   = safe_div((prim.mu - mhq), prim.T)

            ur[j]    = prim.ur
            v[j]     = vv
            n[j]     = prim.n
            e[j]     = prim.e
            P[j]     = prim.P
            ok[j]    = prim.ok

                 # Charge current components (Milne coordinates, 1+1D radial):
                 # u^μ = (u^τ, u^r), ν^μ orthogonal to u^μ -> ν^τ = (u^r/u^τ) ν^r.
                 uτv[j] = uτ
                 nuτ = (uτ <= 0) ? 0.0 : (prim.ur / uτ) * nur_phys
                 nu_τ[j] = nuτ
                 Jtau[j] = prim.n * uτ + nuτ
                 Jr[j]   = prim.n * prim.ur + nur_phys

                 kappa[j] = diff_kappa(prim.T, prim.mu, prim.n, model)
                 tau_n[j] = diff_tauN(prim.T, prim.mu, model)
        end
    end

            tbl = if no_charge_snapshot
                (; r, tau,
                 Sr, E=EE,
                 Pi, piR, piEta, piPhi,
                 T, ur, v, e, P, ok)
            else
                (; r, tau, Dtau, Sr, E=EE,
                 nur, nu_r,
                 Jtau, Jr,
                 Pi, piR, piEta, piPhi,
                 D, T, mu, alpha, phi, ur, v, n, e, P, ok,
                 kappa, tau_n)
            end
    CSV.write(fname, tbl)

    meta = no_charge_snapshot ? (; tau=[τ]) : (; tau=[τ], Q_Dtau=[charge_integral_Dtau(U, grid, model)])
    CSV.write(replace(fname, ".csv" => "_meta.csv"), meta)

    return nothing
end


# -----------------------------------------------------------------------------
# Langevin-style JLD2 outputs from snapshot_tau_*.csv
# -----------------------------------------------------------------------------

_is_snapshot_csv(path::AbstractString) = startswith(basename(path), "snapshot_tau_") && endswith(path, ".csv") && !endswith(path, "_meta.csv")

function _sorted_by_r(r_raw::AbstractVector{<:Real})
    r = Float64.(r_raw)
    p = sortperm(r)
    return r[p], p
end

function _interp_linear(xsrc::Vector{Float64}, ysrc::Vector{Float64}, xtar::Vector{Float64})
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

function _load_snapshot_cols(path::AbstractString)
    tbl = Tables.columntable(CSV.File(path))
    haskey(tbl, :r) || error("Missing column :r in $path")
    haskey(tbl, :tau) || error("Missing column :tau in $path")
    r_sorted, p = _sorted_by_r(tbl.r)
    τ = Float64(first(tbl.tau))

    # Prefer physical columns if present; otherwise reconstruct from stored nur + n, ur.
    ncol  = haskey(tbl, :n)  ? Float64.(tbl.n)[p]  : fill(0.0, length(r_sorted))
    urcol = haskey(tbl, :ur) ? Float64.(tbl.ur)[p] : error("Missing column :ur in $path")

    nur_phys = if haskey(tbl, :nu_r)
        Float64.(tbl.nu_r)[p]
    elseif haskey(tbl, :nur)
        phys_from_stored.(Float64.(tbl.nur)[p])
    else
        fill(0.0, length(r_sorted))
    end

    uτcol = sqrt.(1 .+ urcol.^2)
    nuτcol = ifelse.(uτcol .<= 0, 0.0, (urcol ./ uτcol) .* nur_phys)

    Jtaucol = if haskey(tbl, :Jtau)
        Float64.(tbl.Jtau)[p]
    else
        @. ncol * uτcol + nuτcol
    end
    Jrcol = if haskey(tbl, :Jr)
        Float64.(tbl.Jr)[p]
    else
        @. ncol * urcol + nur_phys
    end

    kappa_col = haskey(tbl, :kappa) ? Float64.(tbl.kappa)[p] : (haskey(tbl, :D_s) ? Float64.(tbl.D_s)[p] : fill(NaN, length(r_sorted)))
    tau_n_col = haskey(tbl, :tau_n) ? Float64.(tbl.tau_n)[p] : fill(NaN, length(r_sorted))
    Phi_col = if haskey(tbl, :Phi)
        Float64.(tbl.Phi)[p]
    elseif haskey(tbl, :phiS)
        Float64.(tbl.phiS)[p]
    else
        fill(0.0, length(r_sorted))
    end

    numeric_cols = Dict{Symbol,Vector{Float64}}()
    for (name, col) in pairs(tbl)
        name in (:r, :tau) && continue
        if Base.nonmissingtype(eltype(col)) <: Number
            numeric_cols[name] = Float64.(col)[p]
        end
    end

    numeric_cols[:Jtau] = Jtaucol
    numeric_cols[:Jr] = Jrcol
    numeric_cols[:n] = ncol
    numeric_cols[:nur] = haskey(tbl, :nur) ? Float64.(tbl.nur)[p] : stored_from_phys.(nur_phys)
    numeric_cols[:nu_r] = nur_phys
    numeric_cols[:kappa] = kappa_col
    numeric_cols[:tau_n] = tau_n_col
    numeric_cols[:Phi] = Phi_col

    # Canonicalize FiVo viscous-field names to the names consumed by plot_splines.jl.
    if haskey(numeric_cols, :piEta)
        numeric_cols[:pietaeta] = numeric_cols[:piEta]
    end
    if haskey(numeric_cols, :piPhi)
        numeric_cols[:piphiphi] = numeric_cols[:piPhi]
    end

    # Backfill only non-excluded fields that the plotter can consume.
    npts = length(r_sorted)
    for name in (:alpha, :phi, :tau_diff)
        haskey(numeric_cols, name) || (numeric_cols[name] = fill(0.0, npts))
    end
    if !haskey(numeric_cols, :kappa) || any(!isfinite, numeric_cols[:kappa])
        numeric_cols[:kappa] = fill(0.0, npts)
    end
    if !haskey(numeric_cols, :tau_n) || any(!isfinite, numeric_cols[:tau_n])
        numeric_cols[:tau_n] = fill(0.0, npts)
    end

    return (; τ=τ, r=r_sorted, Jtau=Jtaucol, Jr=Jrcol, n=ncol, nu_r=nur_phys, kappa=kappa_col, tau_n=tau_n_col, Phi=Phi_col, numeric_cols=numeric_cols)
end

"""
    write_hydro_currents_jld2(snapshot_dir; outdir=snapshot_dir, tag="") -> path

Builds Langevin-style current arrays from FiVoHydro `snapshot_tau_*.csv` files and
        writes a JLD2 file with keys:
    r, tau, Jau, Jr, n, nu_r, kappa, tau_n, Phi
"""
function write_hydro_currents_jld2(
    snapshot_dir::AbstractString;
    outdir::AbstractString = snapshot_dir,
    tag::AbstractString = "",
    rgrid_mode::Symbol = :resample,
)
    isdir(snapshot_dir) || error("Not a directory: $snapshot_dir")
    files = filter(_is_snapshot_csv, readdir(snapshot_dir; join=true))
    isempty(files) && error("No snapshot_tau_*.csv found in $snapshot_dir")
    sort!(files)

    snap0 = _load_snapshot_cols(files[1])
    r0 = snap0.r

    Nt = length(files)
    Nr = length(r0)
    τs = Vector{Float64}(undef, Nt)
    Jtau = Matrix{Float64}(undef, Nr, Nt)
    Jr   = Matrix{Float64}(undef, Nr, Nt)
    n    = Matrix{Float64}(undef, Nr, Nt)
    nur  = Matrix{Float64}(undef, Nr, Nt)
    kappa  = Matrix{Float64}(undef, Nr, Nt)
    tau_n = Matrix{Float64}(undef, Nr, Nt)
    Phi   = Matrix{Float64}(undef, Nr, Nt)

    for (it, f) in pairs(files)
        s = _load_snapshot_cols(f)
        τs[it] = s.τ
        same = (length(s.r) == Nr) && all(isapprox.(s.r, r0; rtol=1e-12, atol=0.0))
        if same
            Jtau[:, it] = s.Jtau
            Jr[:, it]   = s.Jr
            n[:, it]    = s.n
            nur[:, it]  = s.nu_r
            kappa[:, it]  = s.kappa
            tau_n[:, it] = s.tau_n
            Phi[:, it]   = s.Phi
        else
            rgrid_mode == :resample || error("r-grid mismatch in $f (set rgrid_mode=:resample to interpolate onto first snapshot grid)")
            Jtau[:, it] = _interp_linear(s.r, s.Jtau, r0)
            Jr[:, it]   = _interp_linear(s.r, s.Jr, r0)
            n[:, it]    = _interp_linear(s.r, s.n, r0)
            nur[:, it]  = _interp_linear(s.r, s.nu_r, r0)
            kappa[:, it]  = _interp_linear(s.r, s.kappa, r0)
            tau_n[:, it] = _interp_linear(s.r, s.tau_n, r0)
            Phi[:, it]   = _interp_linear(s.r, s.Phi, r0)
        end
    end

    pt = sortperm(τs)
    τgrid = τs[pt]

    mkpath(outdir)
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    tag_part = isempty(tag) ? "" : "_" * replace(tag, r"\s+" => "_")
    fpath = joinpath(outdir, "hydro_currents$(tag_part)_$(ts).jld2")

    # Match Langevin naming: Jau == J^\tau
    jldsave(
        fpath;
        r = r0,
        tau = τgrid,
        Jau = Jtau[:, pt],
        Jr  = Jr[:, pt],
        n   = n[:, pt],
        nu_r = nur[:, pt],
        kappa = kappa[:, pt],
        tau_n = tau_n[:, pt],
        Phi = Phi[:, pt],
    )

    return fpath
end

"""
    write_hydro_currents_splines_jld2(snapshot_dir; outdir=snapshot_dir, filename="hydro_currents_splines.jld2") -> path

Writes a single JLD2 containing both current grids and Spline2D objects.
In addition to the historical current fields, this exports all numeric columns present in
the FiVo `snapshot_tau_*.csv` files as `<name>_grid` and, when possible, `<name>_spline`.
This includes fields such as `T`, `ur`, `alpha`, `Pi`, `piR`, `piEta`, `piPhi`, and others.
"""
function write_hydro_currents_splines_jld2(
    snapshot_dir::AbstractString;
    outdir::AbstractString = snapshot_dir,
    filename::AbstractString = "hydro_currents_splines.jld2",
    tag::AbstractString = "",
    kx::Int = 3,
    ky::Int = 3,
    s::Real = 0.0,
    rgrid_mode::Symbol = :resample,
    overwrite::Bool = true,
    store_phi_splines::Bool = true,
)
    isdir(snapshot_dir) || error("Not a directory: $snapshot_dir")
    files = filter(_is_snapshot_csv, readdir(snapshot_dir; join=true))
    isempty(files) && error("No snapshot_tau_*.csv found in $snapshot_dir")
    sort!(files)

    snap0 = _load_snapshot_cols(files[1])
    r0 = Float64.(snap0.r)

    Nt = length(files)
    Nr = length(r0)
    τs = Vector{Float64}(undef, Nt)
    field_names = sort!([name for name in keys(snap0.numeric_cols) if !(name in SPLINE_EXPORT_EXCLUDED_FIELDS)]; by=String)
    field_mats = Dict{Symbol,Matrix{Float64}}(name => Matrix{Float64}(undef, Nr, Nt) for name in field_names)

    for (it, f) in pairs(files)
        s1 = _load_snapshot_cols(f)
        τs[it] = s1.τ
        same = (length(s1.r) == Nr) && all(isapprox.(s1.r, r0; rtol=1e-12, atol=0.0))
        missing_fields = setdiff(Set(field_names), Set(keys(s1.numeric_cols)))
        isempty(missing_fields) || error("Snapshot $f is missing numeric fields required by the reference snapshot: $(sort!(collect(missing_fields); by=String))")

        for name in field_names
            vals = s1.numeric_cols[name]
            if same
                field_mats[name][:, it] = vals
            else
                rgrid_mode == :resample || error("r-grid mismatch in $f (set rgrid_mode=:resample to interpolate onto first snapshot grid)")
                field_mats[name][:, it] = _interp_linear(Float64.(s1.r), Float64.(vals), r0)
            end
        end
    end

    pt = sortperm(τs)
    τgrid = Float64.(τs[pt])
    field_grids = Dict{Symbol,Matrix{Float64}}(name => field_mats[name][:, pt] for name in field_names)

    # Dierckx requires: length(x) > kx and length(y) > ky.
    # For short test runs (few snapshots) we still want this routine to be safe.
    Nr_eff = length(r0)
    Nt_eff = length(τgrid)
    kx_eff = min(kx, Nr_eff - 1)
    ky_eff = min(ky, Nt_eff - 1)

    field_splines = Dict{Symbol,Any}()

    if kx_eff >= 1 && ky_eff >= 1
        for name in field_names
            grid = field_grids[name]
            if all(isfinite, grid)
                if name == :Phi && !store_phi_splines
                    field_splines[name] = nothing
                else
                    field_splines[name] = Spline2D(r0, τgrid, grid; kx=kx_eff, ky=ky_eff, s=float(s))
                end
            else
                field_splines[name] = nothing
            end
        end
    else
        @warn "Not enough snapshots/points to build Spline2D; writing grids only" Nr=Nr_eff Nt=Nt_eff requested=(kx=kx,ky=ky) used=(kx=kx_eff,ky=ky_eff)
    end

    mkpath(outdir)
    fpath = joinpath(outdir, filename)
    if overwrite && isfile(fpath)
        rm(fpath; force=true)
    end

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    label = isempty(tag) ? "FiVoHydro_currents" : tag

    payload = Dict{String,Any}(
        "created_at" => ts,
        "label" => label,
        "r_grid" => r0,
        "t_grid" => τgrid,
    )

    for name in field_names
        grid = field_grids[name]
        payload["$(name)_grid"] = grid
        payload["$(name)_spline"] = get(field_splines, name, nothing)
    end

    # Match the plotter contract: canonical field names plus the alpha unicode alias.
    if haskey(field_grids, :alpha)
        payload["α_grid"] = field_grids[:alpha]
        payload["α_spline"] = get(field_splines, :alpha, nothing)
    end

    jldopen(fpath, "w") do io
        for (key, value) in payload
            io[key] = value
        end
    end

    return fpath
end
