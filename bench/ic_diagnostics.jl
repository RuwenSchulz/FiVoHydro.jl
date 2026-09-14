#!/usr/bin/env julia
# bench/ic_diagnostics.jl
#
# IC diagnostics runner (separate from the solver).
#
# Writes per-cell IC fields + 1st/2nd derivatives to CSV and produces a compact
# summary across a suite of configurations.
#
# Usage examples:
#   julia --threads=auto bench/ic_diagnostics.jl --suite
#   julia --threads=auto bench/ic_diagnostics.jl --outdir=icdiag --analytic
#   julia --threads=auto bench/ic_diagnostics.jl --outdir=icdiag --init_csv=data/initial_profiles_physical.csv --Nr=600 --nghost=3 --rmax=22 --tau0=0.4
#   julia --threads=auto bench/ic_diagnostics.jl --suite --diff --shear --bulk
#
# Notes on CLI parsing:
# - Keyword arguments must be passed as --key=value (no spaces).
#
# Notes:
# - This does NOT time-step.
# - It uses the same initialize!/primitive-recovery path as the solver.

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

function getkw_int(name::String, default::Int)
    pref = name * "="
    for a in ARGS
        startswith(a, pref) || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _sanitize_label(s::AbstractString)
    t = replace(String(s), r"[^A-Za-z0-9_.-]+" => "_")
    return isempty(t) ? "case" : t
end

# ----------------------------
# Terminal output helpers
# ----------------------------
const _SPARK = ['▁','▂','▃','▄','▅','▆','▇','█']

function _fmt(x)
    if x === nothing
        return ""
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.6g", x) : "NaN"
    else
        return string(x)
    end
end

function _sparkline(y::AbstractVector{<:Real}; width::Int=80)
    width = max(width, 8)
    n = length(y)
    n == 0 && return ""

    # Downsample to at most `width` points.
    m = min(n, width)
    idx(i) = clamp(1 + Int(floor((i-1) * (n-1) / max(m-1, 1))), 1, n)
    ys = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        ys[i] = Float64(y[idx(i)])
    end

    ymin = Inf
    ymax = -Inf
    @inbounds for v in ys
        if isfinite(v)
            ymin = min(ymin, v)
            ymax = max(ymax, v)
        end
    end
    if !(isfinite(ymin) && isfinite(ymax))
        return repeat("·", m)
    end

    if ymin == ymax
        return repeat(string(_SPARK[4]), m)
    end

    s = IOBuffer()
    @inbounds for v in ys
        if !isfinite(v)
            print(s, '·')
            continue
        end
        t = (v - ymin) / (ymax - ymin)
        k = clamp(1 + Int(floor(t * (length(_SPARK)-1) + 1e-12)), 1, length(_SPARK))
        print(s, _SPARK[k])
    end
    return String(take!(s))
end

function _print_table(rows::AbstractVector{<:NamedTuple}; cols::Vector{Symbol}, headers::Vector{String})
    @assert length(cols) == length(headers)
    # Convert to strings first.
    data = [ [ _fmt(getproperty(r, c)) for c in cols ] for r in rows ]
    widths = [ length(h) for h in headers ]
    for i in eachindex(cols)
        for r in data
            widths[i] = max(widths[i], length(r[i]))
        end
    end

    function _line(parts)
        io = IOBuffer()
        for (i, p) in enumerate(parts)
            i > 1 && print(io, " | ")
            print(io, rpad(p, widths[i]))
        end
        return String(take!(io))
    end

    println(_line(headers))
    println(join([repeat("-", w) for w in widths], "-+-"))
    for r in data
        println(_line(r))
    end
    return nothing
end

function _read_single_csv_row(path::AbstractString)
    rows = Tables.rowtable(CSV.File(path))
    isempty(rows) && error("Empty CSV: $path")
    return rows[1]
end

function _read_csv_rows(path::AbstractString)
    return Tables.rowtable(CSV.File(path))
end

function _col_as_f64(path::AbstractString, col::Symbol)
    tbl = CSV.File(path)
    c = Tables.getcolumn(tbl, col)
    out = Vector{Float64}(undef, length(c))
    @inbounds for (i, x) in enumerate(c)
        out[i] = x === missing ? NaN : Float64(x)
    end
    return out
end

function _csv_colset(path::AbstractString)
    sch = Tables.schema(CSV.File(path))
    # schema names are Symbols in Tables.jl
    return Set(Symbol.(sch.names))
end

function _finite_minmax(v::Vector{Float64})
    vmin = Inf
    vmax = -Inf
    @inbounds for x in v
        if isfinite(x)
            vmin = min(vmin, x)
            vmax = max(vmax, x)
        end
    end
    if !(isfinite(vmin) && isfinite(vmax))
        return (NaN, NaN)
    end
    return (vmin, vmax)
end

function _print_case_run_summary(case_outdir::AbstractString)
    s = _read_single_csv_row(joinpath(case_outdir, "ic_run_summary.csv"))
    rows = [s]
    cols = [:label, :ok_cells, :bad_cells, :dt, :dt_cfl, :dt_diff, :dt_shear, :dt_bulk, :amax_hyp, :amax_used, :kmax, :vabs_max]
    headers = ["case", "ok", "bad", "dt", "dt_cfl", "dt_diff", "dt_shear", "dt_bulk", "amax_hyp", "amax_used", "kmax", "|v|max"]
    _print_table(rows; cols=cols, headers=headers)
    return nothing
end

function _print_case_field_summary(case_outdir::AbstractString)
    rows = _read_csv_rows(joinpath(case_outdir, "ic_field_summary.csv"))
    cols = [:name, :min, :max, :tv, :max_abs_d1, :max_abs_d2, :max_dimless_curv, :d1_sign_changes, :n_at_floor]
    headers = ["field", "min", "max", "TV", "max|d1|", "max|d2|", "max κ", "signflips", "at_floor"]
    _print_table(rows; cols=cols, headers=headers)
    return nothing
end

function _print_case_plots(case_outdir::AbstractString; width::Int=90, kind::Symbol=:both, fields::Vector{Symbol}=[:T,:mu,:alpha,:phi])
    fields_csv = joinpath(case_outdir, "ic_fields.csv")
    cols = _csv_colset(fields_csv)

    for f in fields
        d1 = Symbol("d", String(f))
        d2 = Symbol("d2", String(f))
        if kind in (:d1, :both)
            if d1 in cols
                y = _col_as_f64(fields_csv, d1)
                vmin, vmax = _finite_minmax(y)
                println("\n  ", String(d1), "  (sparkline)")
                println("  ", _sparkline(y; width=width))
                println("  min=", _fmt(vmin), "  max=", _fmt(vmax))
            else
                println("\n  ", String(d1), "  (missing)")
            end
        end
        if kind in (:d2, :both)
            if d2 in cols
                y2 = _col_as_f64(fields_csv, d2)
                vmin2, vmax2 = _finite_minmax(y2)
                println("\n  ", String(d2), "  (sparkline)")
                println("  ", _sparkline(y2; width=width))
                println("  min=", _fmt(vmin2), "  max=", _fmt(vmax2))
            else
                println("\n  ", String(d2), "  (missing)")
            end
        end
    end
    return nothing
end

function _save_case_png_plots(case_outdir::AbstractString;
                              fields::Vector{Symbol}=[:T,:mu,:alpha,:phi],
                              kind::Symbol=:both,
                              size_px::Tuple{Int,Int}=(1100, 900))
    plots_dir = joinpath(case_outdir, "plots")
    mkpath(plots_dir)

    fields_csv = joinpath(case_outdir, "ic_fields.csv")
    cols = _csv_colset(fields_csv)
    r = _col_as_f64(fields_csv, :r)

    for f in fields
        if !(f in cols)
            @warn "PNG plot field not found in ic_fields.csv; skipping" field=String(f) case_outdir=String(case_outdir)
            continue
        end
        # base field
        y  = _col_as_f64(fields_csv, f)
        d1 = Symbol("d", String(f))
        d2 = Symbol("d2", String(f))

        p1 = Plots.plot(r, y; lw=2, label=String(f), xlabel="r", ylabel=String(f))
        if kind in (:d1, :both)
            if d1 in cols
                y1 = _col_as_f64(fields_csv, d1)
                p2 = Plots.plot(r, y1; lw=2, label=String(d1), xlabel="r", ylabel=String(d1))
            else
                p2 = Plots.plot(r, zeros(length(r)); lw=1, label="(missing)", xlabel="r", ylabel=String(d1))
            end
        else
            p2 = Plots.plot(r, zeros(length(r)); lw=1, label="(disabled)", xlabel="r", ylabel="d1")
        end
        if kind in (:d2, :both)
            if d2 in cols
                y2 = _col_as_f64(fields_csv, d2)
                p3 = Plots.plot(r, y2; lw=2, label=String(d2), xlabel="r", ylabel=String(d2))
            else
                p3 = Plots.plot(r, zeros(length(r)); lw=1, label="(missing)", xlabel="r", ylabel=String(d2))
            end
        else
            p3 = Plots.plot(r, zeros(length(r)); lw=1, label="(disabled)", xlabel="r", ylabel="d2")
        end

        fig = Plots.plot(p1, p2, p3; layout=(3,1), size=size_px)
        out = joinpath(plots_dir, "$(String(f))_profiles.png")
        Plots.savefig(fig, out)
    end

    return nothing
end

# ----------------------------
# EOS selection
# ----------------------------
function build_eos(kind::String)
    k = lowercase(kind)
    if k in ("lattice", "latticehrg", "lhrg")
        return hydro.LatticeHRGEOS()
    elseif k in ("conformal", "conformalhq", "chq")
        return hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    elseif k in ("running", "runningconformal", "rconformal")
        return hydro.RunningConformalHQEOS()
    else
        error("Unknown --eos=$kind. Use latticehrg|conformalhq|runningconformal")
    end
end

# ----------------------------
# Build state for IC analysis
# ----------------------------
function build_ic_state(; Nr::Int, rmax::Float64, nghost::Int, τ0::Float64,
                        init_csv::Union{Nothing,String},
                        fugacity_kind::Symbol, taper_width::Float64,
                        interp_kind::Symbol, interp_dr::Union{Nothing,Float64},
                        CFL::Float64, CFLτ::Float64,
                        enable_diff::Bool, enable_shear::Bool, enable_bulk::Bool,
                        eos_kind::String)

    grid = hydro.make_grid(Nr; rmax=rmax, nghost=nghost)

    layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    eos = build_eos(eos_kind)
    shear_model = hydro.QGPViscosity(0.1, 1.0)
    bulk_model  = hydro.SimpleBulkViscosity(0.083, 1.0)

    model = hydro.IdealDiffViscModel(
        eos, layout, hydro.IdealPrimRec(),
        # diffusion
        enable_diff, :alpha, 0.1, 1.0, 0.0,
        0.02, 0.3, 0.3, -1.0,
        0.0, 0.0, 0.0, 0.0,
        true, false, 16, false, true,   # do_soft_project_nur, do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur
        # viscosity
        enable_shear, enable_bulk, shear_model, bulk_model,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        -1.0, -1.0,
        false, false,
        # relax_advect_Pi, relax_advect_pi
        true, true,
    )

    U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
    hydro.initialize!(U, grid, τ0, model;
        init_csv=init_csv,
        fugacity_kind=fugacity_kind,
        taper_width=taper_width,
        interp_kind=interp_kind,
        interp_dr=interp_dr,
        Emin=hydro.E_FLOOR,
        χ=hydro.χ_SrE,
    )

    # The initializer already applies BC + floors + Sr constraint.
    # Still, we prime work once so dt stats match the solver path.
    work = hydro.make_work(U)
    hydro.prime_work_from_U!(work, U, grid, τ0, model)

    return (; U, grid, model)
end

# ----------------------------
# Suite definition
# ----------------------------
function default_suite(; init_csv_default::String)
    return [
        (label="physical_csv", init_csv=init_csv_default, taper_width=0.5, fugacity_kind=:alpha),
        (label="physical_csv_no_taper", init_csv=init_csv_default, taper_width=0.0, fugacity_kind=:alpha),
        (label="analytic_default", init_csv=nothing, taper_width=0.0, fugacity_kind=:alpha),
    ]
end

# ----------------------------
# Runner
# ----------------------------
function run_one!(rows::Vector{Any}; outdir::String, case, common)
    label = case.label
    case_outdir = joinpath(outdir, _sanitize_label(label))

    st = build_ic_state(;
        Nr=common.Nr,
        rmax=common.rmax,
        nghost=common.nghost,
        τ0=common.τ0,
        init_csv=case.init_csv,
        fugacity_kind=case.fugacity_kind,
        taper_width=case.taper_width,
        interp_kind=common.interp_kind,
        interp_dr=common.interp_dr,
        CFL=common.CFL,
        CFLτ=common.CFLτ,
        enable_diff=common.enable_diff,
        enable_shear=common.enable_shear,
        enable_bulk=common.enable_bulk,
        eos_kind=common.eos_kind,
    )

    res = hydro.write_ic_diagnostics(case_outdir, st.U, st.grid, common.τ0, st.model;
        label=label,
        CFL=common.CFL,
        CFLτ=common.CFLτ,
    )

    if common.print_table
        println("\n=== IC diagnostics: ", label, " ===")
        _print_case_run_summary(case_outdir)
        if common.print_field_table
            println("\n-- Field summary (", label, ") --")
            _print_case_field_summary(case_outdir)
        end
        if common.print_plots
            println("\n-- Derivative plots (", label, ") --")
            _print_case_plots(case_outdir;
                width=common.plot_width,
                kind=common.plot_kind,
                fields=common.plot_fields,
            )
        end
        println("")
    end

    if common.save_png
        _save_case_png_plots(case_outdir;
            fields=common.png_fields,
            kind=common.png_kind,
            size_px=common.png_size_px,
        )
        if common.print_table
            println("Saved PNG plots to: ", joinpath(case_outdir, "plots"))
        end
    end

    push!(rows, (
        label=label,
        outdir=String(case_outdir),
        init_csv=(case.init_csv === nothing ? "" : String(case.init_csv)),
        fugacity_kind=String(case.fugacity_kind),
        taper_width=Float64(case.taper_width),
        interp_kind=String(common.interp_kind),
        interp_dr=(common.interp_dr === nothing ? NaN : Float64(common.interp_dr)),
        Nr=Int(common.Nr),
        nghost=Int(common.nghost),
        rmax=Float64(common.rmax),
        tau0=Float64(common.τ0),
        eos=String(common.eos_kind),
        enable_diff=Bool(common.enable_diff),
        enable_shear=Bool(common.enable_shear),
        enable_bulk=Bool(common.enable_bulk),
        ok_cells=Int(res.ok_cells),
        bad_cells=Int(res.bad_cells),
        dt=Float64(res.dt),
    ))

    return nothing
end

function main()
    outdir = getkw_str("--outdir", "ic_diagnostics")

    Nr   = getkw_int("--Nr", 600)
    nghost = getkw_int("--nghost", 3)
    rmax = getkw_float("--rmax", 22.0)
    τ0   = getkw_float("--tau0", 0.4)

    CFL  = getkw_float("--CFL", 0.2)
    CFLτ = getkw_float("--CFLtau", 0.05)

    eos_kind = getkw_str("--eos", "latticehrg")

    enable_diff  = hasflag("--diff")
    enable_shear = hasflag("--shear")
    enable_bulk  = hasflag("--bulk")

    init_csv_default = getkw_str("--init_csv", "data/initial_profiles_physical.csv")

    do_suite = hasflag("--suite")
    do_analytic = hasflag("--analytic")

    print_table = !hasflag("--no-table")
    print_field_table = hasflag("--field-table")
    print_plots = hasflag("--plots")
    plot_width = getkw_int("--plot_width", 90)
    plot_kind_str = lowercase(getkw_str("--plot_kind", "both"))
    plot_kind = plot_kind_str == "d1" ? :d1 : (plot_kind_str == "d2" ? :d2 : :both)
    plot_fields_str = getkw_str("--plot_fields", "T,mu,alpha,phi")
    plot_fields = Symbol[]
    for t in split(plot_fields_str, ",")
        s = strip(t)
        isempty(s) && continue
        push!(plot_fields, Symbol(s))
    end

    save_png = hasflag("--png")
    png_kind_str = lowercase(getkw_str("--png_kind", "both"))
    png_kind = png_kind_str == "d1" ? :d1 : (png_kind_str == "d2" ? :d2 : :both)
    png_fields_str = getkw_str("--png_fields", plot_fields_str)
    png_fields = Symbol[]
    for t in split(png_fields_str, ",")
        s = strip(t)
        isempty(s) && continue
        push!(png_fields, Symbol(s))
    end
    png_w = getkw_int("--png_w", 1100)
    png_h = getkw_int("--png_h", 900)

    fug_kind_str = getkw_str("--fugacity", "alpha")
    fugacity_kind = lowercase(fug_kind_str) == "lambda" ? :lambda : :alpha

    interp_str = lowercase(getkw_str("--interp", "linear"))
    interp_kind = interp_str == "cubic" ? :cubic : :linear
    interp_dr_val = getkw_float("--interp_dr", NaN)
    interp_dr = isfinite(interp_dr_val) ? interp_dr_val : nothing

    taper_width = getkw_float("--taper_width", 0.0)

    mkpath(outdir)

    common = (;
        Nr=Nr,
        nghost=nghost,
        rmax=rmax,
        τ0=τ0,
        CFL=CFL,
        CFLτ=CFLτ,
        enable_diff=enable_diff,
        enable_shear=enable_shear,
        enable_bulk=enable_bulk,
        eos_kind=eos_kind,
        print_table=print_table,
        print_field_table=print_field_table,
        print_plots=print_plots,
        plot_width=plot_width,
        plot_kind=plot_kind,
        plot_fields=plot_fields,
        save_png=save_png,
        png_kind=png_kind,
        png_fields=png_fields,
        png_size_px=(png_w, png_h),
        interp_kind=interp_kind,
        interp_dr=interp_dr,
    )

    rows = Any[]

    if do_suite
        for c in default_suite(; init_csv_default=init_csv_default)
            run_one!(rows; outdir=outdir, case=c, common=common)
        end
    else
        init_csv = do_analytic ? nothing : init_csv_default
        c = (;
            label = do_analytic ? "analytic" : "single",
            init_csv = init_csv,
            taper_width = taper_width,
            fugacity_kind = fugacity_kind,
        )
        run_one!(rows; outdir=outdir, case=c, common=common)
    end

    if do_suite
        CSV.write(joinpath(outdir, "ic_suite_summary.csv"), rows)
        println("Wrote suite summary: ", joinpath(outdir, "ic_suite_summary.csv"))

        if print_table
            println("\n=== IC suite summary (", outdir, ") ===")
            # Re-read per-case run summaries to show full dt breakdown.
            suite_rows = NamedTuple[]
            for r in rows
                case_outdir = r.outdir
                push!(suite_rows, _read_single_csv_row(joinpath(case_outdir, "ic_run_summary.csv")))
            end
            cols = [:label, :ok_cells, :bad_cells, :dt, :dt_cfl, :dt_diff, :dt_shear, :dt_bulk, :amax_hyp, :amax_used, :kmax, :vabs_max]
            headers = ["case", "ok", "bad", "dt", "dt_cfl", "dt_diff", "dt_shear", "dt_bulk", "amax_hyp", "amax_used", "kmax", "|v|max"]
            _print_table(suite_rows; cols=cols, headers=headers)
            println("")
        end
    end
    return nothing
end

main()