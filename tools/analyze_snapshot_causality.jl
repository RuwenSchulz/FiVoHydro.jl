#!/usr/bin/env julia

using CSV
using Printf

include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

const TAU_RE = r"snapshot_tau_(\d+\.\d+)\.csv$"

@inline function parse_tau(fname::AbstractString)
    m = match(TAU_RE, fname)
    m === nothing && error("Could not parse tau from filename: $fname")
    return parse(Float64, m.captures[1])
end

@inline clamp01(x::Float64) = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)

@inline function lambda_plus(v::Float64, cs::Float64)
    denom = 1.0 + v*cs
    return denom == 0.0 ? sign(v + cs) * Inf : (v + cs)/denom
end

function analyze_dir(snapshot_dir::AbstractString; eos=hydro.LatticeHRGEOS())
    files = sort(filter(f -> occursin(TAU_RE, f), readdir(snapshot_dir; join=true)); by=f->parse_tau(basename(f)))
    isempty(files) && error("No snapshot_tau_*.csv found in $snapshot_dir")

    out_csv = joinpath(snapshot_dir, "snapshot_causality_summary.csv")
    open(out_csv, "w") do io
        println(io, "tau,max_abs_v,r_at_max_abs_v,max_lambda_plus,cs2_at_max_lambda_plus")

        global_max_v = -Inf
        global_max_lam = -Inf
        global_bad_v = false
        global_bad_lam = false

        for path in files
            tau = parse_tau(basename(path))
            max_abs_v = -Inf
            r_at_max_abs_v = NaN
            max_lam = -Inf
            cs2_at_max_lam = NaN

            for row in CSV.File(path; normalizenames=true)
                ok = getproperty(row, :ok)
                ok isa Bool || (ok = String(ok) == "true")
                ok || continue

                r = Float64(getproperty(row, :r))
                r < 0.0 && continue

                v = Float64(getproperty(row, :v))
                T = Float64(getproperty(row, :T))
                μ = Float64(getproperty(row, :mu))

                av = abs(v)
                if av > max_abs_v
                    max_abs_v = av
                    r_at_max_abs_v = r
                end

                cs2 = hydro.eos_cs2(T, μ, eos)
                # guard against tiny negatives from numerics; do NOT clamp >1 here
                cs = sqrt(max(cs2, 0.0))
                lam = lambda_plus(v, cs)
                if lam > max_lam
                    max_lam = lam
                    cs2_at_max_lam = cs2
                end
            end

            println(io, @sprintf("%.12g,%.12g,%.12g,%.12g,%.12g", tau, max_abs_v, r_at_max_abs_v, max_lam, cs2_at_max_lam))

            global_max_v = max(global_max_v, max_abs_v)
            global_max_lam = max(global_max_lam, max_lam)
            global_bad_v |= (max_abs_v > 1.0 + 1e-12)
            global_bad_lam |= (max_lam > 1.0 + 1e-12)
        end

        @printf("Wrote %s\n", out_csv)
        @printf("Global max |v| = %.6f\n", global_max_v)
        @printf("Global max lambda+ = %.6f\n", global_max_lam)
        global_bad_v && @printf("WARNING: found |v| > 1\n")
        global_bad_lam && @printf("WARNING: found lambda+ > 1\n")
    end

    return out_csv
end

function main()
    snapshot_dir = length(ARGS) >= 1 ? ARGS[1] : "snapshots/snapshots_ideal_fluidum"
    analyze_dir(snapshot_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
