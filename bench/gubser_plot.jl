#!/usr/bin/env julia
# bench/gubser_plot.jl
#
# Post-processing plots for Gubser validation output.
#
# Example:
#   julia --project=. --threads=auto bench/gubser_plot.jl --indir=snapshots/gubser --tau=1.0
#
# Outputs (PNG) go to --outdir (default: --indir/plots).

include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro
using CSV
using Tables
using Printf
using Plots

# ----------------------------
# Minimal CLI parsing (no deps)
# ----------------------------
hasflag(name::String) = any(==(name), ARGS)

function getkw_str(name::String, default::AbstractString)
    pref = name * "="
    for a in ARGS
        startswith(a, pref) || continue
        return String(split(a, "=", limit=2)[2])
    end
    return String(default)
end

function getkw_float(name::String, default::Float64)
    pref = name * "="
    for a in ARGS
        startswith(a, pref) || continue
        return parse(Float64, split(a, "=", limit=2)[2])
    end
    return default
end

# ----------------------------
# Helpers
# ----------------------------
function _read_config(indir::String)
    cfg_path = joinpath(indir, "gubser_config.csv")
    if isfile(cfg_path)
        tbl = Tables.columntable(CSV.File(cfg_path))
        return (
            tau0 = Float64(tbl.tau0[1]),
            q    = Float64(tbl.q[1]),
            Tc0  = Float64(tbl.Tc0[1]),
        )
    end
    return nothing
end

function _parse_tau_from_snapshot_name(fname::String)
    m = match(r"snapshot_tau_([0-9]+\.[0-9]+)\.csv$", fname)
    return m === nothing ? nothing : parse(Float64, m.captures[1])
end

function _pick_snapshot(indir::String, τwant::Float64)
    files = filter(f -> endswith(f, ".csv") && startswith(f, "snapshot_tau_") && !endswith(f, "_meta.csv"), readdir(indir))
    if isempty(files)
        error("No snapshot_tau_*.csv found in indir=$(indir)")
    end

    best = files[1]
    bestΔ = Inf
    bestτ = NaN
    for f in files
        τ = _parse_tau_from_snapshot_name(f)
        τ === nothing && continue
        Δ = abs(τ - τwant)
        if Δ < bestΔ
            bestΔ = Δ
            best = f
            bestτ = τ
        end
    end
    return (joinpath(indir, best), bestτ, bestΔ)
end

# ----------------------------
# Main plotting
# ----------------------------
function run()
    indir  = getkw_str("--indir", "snapshots/snapshots_gubser_validation")
    outdir = getkw_str("--outdir", joinpath(indir, "plots"))
    mkpath(outdir)

    errs_path = joinpath(indir, "gubser_errors.csv")
    isfile(errs_path) || error("Missing gubser_errors.csv in indir=$(indir)")

    df = Tables.columntable(CSV.File(errs_path))
    τs   = collect(Float64, df.tau)
    errT = collect(Float64, df.err_T)
    erre = collect(Float64, df.err_e)
    erru = collect(Float64, df.err_ur)

    has_cons = hasproperty(df, :err_E) && hasproperty(df, :err_Sr) && hasproperty(df, :err_Dtau)
    errE = has_cons ? collect(Float64, getproperty(df, :err_E)) : Float64[]
    errS = has_cons ? collect(Float64, getproperty(df, :err_Sr)) : Float64[]
    errD = has_cons ? collect(Float64, getproperty(df, :err_Dtau)) : Float64[]

    # Error-vs-τ plot
    p1 = plot(τs, errT; yscale=:log10, lw=2, label="prim: err_T", xlabel="τ", ylabel="relative L2 error")
    plot!(p1, τs, erre; yscale=:log10, lw=2, label="prim: err_e")
    plot!(p1, τs, erru; yscale=:log10, lw=2, label="prim: err_ur")
    if has_cons
        plot!(p1, τs, errE; yscale=:log10, lw=2, ls=:dot, label="cons: err_E")
        plot!(p1, τs, errS; yscale=:log10, lw=2, ls=:dot, label="cons: err_Sr")
        # err_Dtau can be identically zero for baryonless runs; keep log-scale happy.
        errDplot = max.(errD, 1e-30)
        plot!(p1, τs, errDplot; yscale=:log10, lw=2, ls=:dot, label="cons: err_Dtau")
    end
    savefig(p1, joinpath(outdir, "errors_vs_tau.png"))

    # Pick τ for profile plot
    τarg = getkw_float("--tau", NaN)
    τplot = if isfinite(τarg)
        τarg
    else
        τs[end]
    end

    cfg = _read_config(indir)
    q   = getkw_float("--q",  cfg === nothing ? 1.0 : cfg.q)
    Tc0 = getkw_float("--Tc0", cfg === nothing ? 0.5 : cfg.Tc0)
    τ0  = getkw_float("--tau0", cfg === nothing ? τs[1] : cfg.tau0)

    snap_path, τsnap, Δ = _pick_snapshot(indir, τplot)
    @info "Using snapshot" file=snap_path tau=τsnap dtau=Δ

    snap = Tables.columntable(CSV.File(snap_path))
    r   = collect(Float64, snap.r)
    Tn  = collect(Float64, snap.T)
    en  = collect(Float64, snap.e)
    urn = collect(Float64, snap.ur)

    # Build analytic profiles at the snapshot τ
    Tscale = hydro.gubser_Tscale_from_center_T(τ0, q, Tc0)
    Ta = similar(Tn)
    ea = similar(en)
    ura = similar(urn)

    eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=0.0, g_hq=0.0)
    for i in eachindex(r)
        Ta[i]  = hydro.gubser_temperature(τsnap, r[i], q, Tscale)
        ura[i] = hydro.gubser_ur(τsnap, r[i], q)
        P, _, e = hydro.eos_Pne(Ta[i], 0.0, eos)
        ea[i] = e
    end

    # Profiles: T, e, ur
    ttl = @sprintf("Gubser validation profiles at τ=%.3f", τsnap)

    pT = plot(r, Tn; lw=2, label="numeric", xlabel="r", ylabel="T", title=ttl)
    plot!(pT, r, Ta; lw=2, ls=:dash, label="analytic")
    savefig(pT, joinpath(outdir, @sprintf("profile_T_tau_%06.3f.png", τsnap)))

    pe = plot(r, en; lw=2, label="numeric", xlabel="r", ylabel="e", title=ttl)
    plot!(pe, r, ea; lw=2, ls=:dash, label="analytic")
    savefig(pe, joinpath(outdir, @sprintf("profile_e_tau_%06.3f.png", τsnap)))

    pu = plot(r, urn; lw=2, label="numeric", xlabel="r", ylabel="u^r", title=ttl)
    plot!(pu, r, ura; lw=2, ls=:dash, label="analytic")
    savefig(pu, joinpath(outdir, @sprintf("profile_ur_tau_%06.3f.png", τsnap)))

    @info "Wrote plots" outdir=outdir
end

run()
