#!/usr/bin/env julia
# bench/bench.jl
#
# Usage examples:
#   julia --threads=auto bench/bench.jl --bench
#   julia --threads=auto bench/bench.jl --allocs
#   julia --threads=auto bench/bench.jl --bench --tabulated --Nr=600 --rmax=20.0 --tau=0.6
#   julia --threads=auto bench/bench.jl --bench --diff --visc

include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

# Optional: BenchmarkTools is nice, but keep this script runnable in minimal
# environments (e.g. offline, restricted registries).
const _HAVE_BENCHMARKTOOLS = let ok = false
    try
        @eval using BenchmarkTools
        ok = true
    catch
        ok = false
    end
    ok
end

using Profile
const _HAVE_PROFILE_ALLOCS = isdefined(Profile, :Allocs)
_HAVE_PROFILE_ALLOCS && @eval using Profile.Allocs

using Statistics
using Printf

# ----------------------------
# Minimal CLI parsing (no deps)
# ----------------------------
hasflag(name::String) = any(==(name), ARGS)

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

# ----------------------------
# Build a benchmark case
# ----------------------------
function build_case(; Nr::Int=600, rmax::Float64=20.0, tau::Float64=0.6,
                    enable_diff::Bool=false, enable_visc::Bool=false,
                    tabulated::Bool=false,
                    use_lhrg::Bool=false)

    grid = hydro.make_grid(Nr; rmax=rmax, nghost=3)

    eos = if use_lhrg
        hydro.LatticeHRGEOS()
    elseif !tabulated
        hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    else
        base = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        hydro.TabulatedHQEOS(base; Tmin=hydro.T_MIN, Tmax=1.5, αmin=0.0, αmax=10.0, NT=256, Nα=256)
    end

    layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    model = hydro.IdealDiffViscModel(
        eos, layout, hydro.IdealPrimRec(),
        # diffusion
        enable_diff, :alpha, 0.1, 1.0, 4/3, 0.02, 0.3, 0.3, 0.5,
        0.0, 0.0, 0.3, 0.3,
        true, false, 16, false, false,   # do_soft_project_nur, do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur
        # viscosity
        enable_visc, false, hydro.QGPViscosity(0.08, 1.0), hydro.ZeroBulkViscosity(),
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.3,
        0.5, 0.5,
        false, false,
        # NEW
        false, false
    )

    U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
    hydro.initialize!(U, grid, tau, model; init_csv=nothing)
    work = hydro.make_work(U)
    dU = similar(U)

    return (; U, dU, work, grid, model, tau)
end

# ----------------------------
# Helpers: reset warm-start per sample
# ----------------------------
const _yT_init = log(0.25)

@inline function reset_work_warmstart!(work)
    @inbounds begin
        work.x0_yT  .= _yT_init
        fill!(work.x0_phi, 0.0)
        fill!(work.x0_y,   0.0)
    end
    return nothing
end

# ----------------------------
# Bench suite
# ----------------------------
function run_bench(case)

    U    = case.U
    dU   = case.dU
    work = case.work
    grid = case.grid
    model= case.model
    tau  = case.tau

    # representative dt (from pristine state)
    hydro.prime_work_from_U!(work, U, grid, tau, model)
    Δtau = hydro.compute_dt_from_work(work, grid, tau, model; CFL=0.2, CFLτ=0.05)

    # pristine copies used for resets
    U0   = copy(U)
    Utmp = similar(U)

    println("Threads: ", Threads.nthreads())
    println("U size: ", size(U), "   Nr=", grid.Nr, " dr=", grid.dr)
    println("Representative Δtau = ", Δtau)

    # warm compile
    reset_work_warmstart!(work)
    hydro.rhs!(dU, U, grid, tau, model, work)

    @inbounds Utmp .= U0
    reset_work_warmstart!(work)
    hydro.step_ssprk2!(Utmp, grid, tau, Δtau, model, work; Emin=hydro.E_FLOOR, diag=nothing)

    #hydro.any_bad(U, grid, tau, model; Emin=hydro.E_FLOOR)
    hydro.any_bad(U, grid, tau, model, work; Emin=hydro.E_FLOOR)

    base_only = hasflag("--basebench") || !_HAVE_BENCHMARKTOOLS
    samples = getkw_int("--samples", 20)

    function _print_stats(label::String, times::Vector{Float64}, bytes::Vector{Int}, gctimes::Vector{Float64})
        tmin = minimum(times)
        tmed = median(times)
        tmean = mean(times)
        bmed = median(bytes)
        gcmed = median(gctimes)
        println(label)
        println("  time:  min=", @sprintf("%.6g", tmin), " s  med=", @sprintf("%.6g", tmed), " s  mean=", @sprintf("%.6g", tmean), " s")
        println("  alloc: med_bytes=", bmed, "  med_gctime=", @sprintf("%.6g", gcmed), " s")
        return nothing
    end

    function basebench(label::String, f; samples::Int=20)
        # Warmup once (compilation already warmed above, but keep stable)
        f()
        times = Vector{Float64}(undef, samples)
        bytes = Vector{Int}(undef, samples)
        gctimes = Vector{Float64}(undef, samples)
        for k in 1:samples
            t = Base.@timed f()
            times[k] = Float64(t.time)
            bytes[k] = Int(t.bytes)
            gctimes[k] = Float64(get(t, :gctime, 0.0))
        end
        _print_stats(label, times, bytes, gctimes)
        return nothing
    end

    if base_only
        println("\n=== rhs! (Base.@timed, samples=$(samples)) ===")
        basebench("rhs!", () -> hydro.rhs!(dU, U, grid, tau, model, work); samples=samples)

        println("\n=== any_bad (MOOD scan) ===")
        basebench("any_bad", () -> hydro.any_bad(U, grid, tau, model, work; Emin=hydro.E_FLOOR); samples=samples)

        println("\n=== step_ssprk2! (RESET U EACH SAMPLE) ===")
        basebench("step_ssprk2!", () -> begin
            @inbounds Utmp .= U0
            reset_work_warmstart!(work)
            hydro.step_ssprk2!(Utmp, grid, tau, Δtau, model, work; Emin=hydro.E_FLOOR, diag=nothing)
        end; samples=samples)

        println("\n=== apply_bc! ===")
        basebench("apply_bc!", () -> hydro.apply_bc!(U, grid, tau, model); samples=samples)

        println("\n=== enforce_floors! ===")
        basebench("enforce_floors!", () -> hydro.enforce_floors!(U, grid, tau, model; Emin=hydro.E_FLOOR, diag=nothing); samples=samples)

        println("\n=== enforce_Sr_energy_constraint! ===")
        basebench("enforce_Sr_energy_constraint!", () -> hydro.enforce_Sr_energy_constraint!(U, grid, model; χ=hydro.χ_SrE, mask=nothing, diag=nothing); samples=samples)

        println("\n=== _repair_theta_admissibility! ===")
        basebench("_repair_theta_admissibility!", () -> begin
            @inbounds Utmp .= U
            hydro._repair_theta_admissibility!(Utmp, U, grid, tau, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, mask=nothing, diag=nothing)
        end; samples=samples)

        println("\n=== relax_dissipative! ===")
        basebench("relax_dissipative!", () -> hydro.relax_dissipative!(U, grid, tau, Δtau, model, work); samples=samples)
    else
        println("\n=== rhs! (BenchmarkTools) ===")
        @btime hydro.rhs!($dU, $U, $grid, $tau, $model, $work);

        println("\n=== any_bad (MOOD scan) ===")
        @btime hydro.any_bad($U, $grid, $tau, $model, $work; Emin=hydro.E_FLOOR);

        println("\n=== step_ssprk2! (RESET U EACH SAMPLE) ===")
        @btime begin
            @inbounds $Utmp .= $U0
            reset_work_warmstart!($work)
            hydro.step_ssprk2!($Utmp, $grid, $tau, $Δtau, $model, $work; Emin=hydro.E_FLOOR, diag=nothing)
        end evals=1;

        println("\n=== apply_bc! ===")
        @btime hydro.apply_bc!($U, $grid, $tau, $model);

        println("\n=== enforce_floors! ===")
        @btime hydro.enforce_floors!($U, $grid, $tau, $model; Emin=hydro.E_FLOOR, diag=nothing);

        println("\n=== enforce_Sr_energy_constraint! ===")
        @btime hydro.enforce_Sr_energy_constraint!($U, $grid, $model; χ=hydro.χ_SrE, mask=nothing, diag=nothing);

        println("\n=== relax_dissipative! ===")
        @btime hydro.relax_dissipative!($U, $grid, $tau, $Δtau, $model, $work);
    end
end

# ----------------------------
# Allocation profiling
# ----------------------------
function run_allocs(case)

    U    = case.U
    dU   = case.dU
    work = case.work
    grid = case.grid
    model= case.model
    tau  = case.tau

    # warm compile
    reset_work_warmstart!(work)
    hydro.rhs!(dU, U, grid, tau, model, work)

    # representative dt (from pristine state)
    hydro.prime_work_from_U!(work, U, grid, tau, model)
    Δtau = hydro.compute_dt_from_work(work, grid, tau, model; CFL=0.2, CFLτ=0.05)

    # pristine copies used for resets (for step profiling)
    U0   = copy(U)
    Utmp = similar(U)

    if !_HAVE_PROFILE_ALLOCS
        error("Profile.Allocs not available in this Julia build")
    end

    mode = "rhs"
    for a in ARGS
        if startswith(a, "--allocs=")
            mode = split(a, "=", limit=2)[2]
        end
    end

    Profile.Allocs.clear()
    if mode == "rhs"
        Profile.Allocs.@profile hydro.rhs!(dU, U, grid, tau, model, work)
    elseif mode == "step"
        @inbounds Utmp .= U0
        reset_work_warmstart!(work)
        Profile.Allocs.@profile hydro.step_ssprk2!(Utmp, grid, tau, Δtau, model, work; Emin=hydro.E_FLOOR, diag=nothing)
    elseif mode == "any_bad"
        Profile.Allocs.@profile hydro.any_bad(U, grid, tau, model, work; Emin=hydro.E_FLOOR)
    elseif mode == "relax"
        Profile.Allocs.@profile hydro.relax_dissipative!(U, grid, tau, Δtau, model, work)
    else
        error("Unknown --allocs mode: $(mode). Use --allocs=rhs|step|any_bad|relax")
    end
    Profile.Allocs.print()
end

# ----------------------------
# Allocation breakdown: SSPRK2 internals
# ----------------------------
function run_step_breakdown(case; reps::Int=7)

    U    = case.U
    work = case.work
    grid = case.grid
    model= case.model
    tau  = case.tau

    hydro.prime_work_from_U!(work, U, grid, tau, model)
    Δ = hydro.compute_dt_from_work(work, grid, tau, model; CFL=0.2, CFLτ=0.05)

    U0   = copy(U)
    Utmp = similar(U)

    # Warm compilation + caches
    reset_work_warmstart!(work)
    @inbounds Utmp .= U0
    hydro.step_ssprk2!(Utmp, grid, tau, Δ, model, work; Emin=hydro.E_FLOOR, diag=nothing)

    labels = String[
        "rhs!(k, U)",
        "U1 = U + Δ*k",
        "apply_bc!(U1)",
        "enforce_floors!(U1)",
        "repair_theta!(U1)",
        "enforce_Sr_constraint!(U1)",
        "apply_bc!(U1) [2]",
        "relax_dissipative!(U1)",
        "apply_bc!(U1) [3]",
        "enforce_floors!(U1) [2]",
        "repair_theta!(U1) [2]",
        "enforce_Sr_constraint!(U1) [2]",
        "apply_bc!(U1) [4]",
        "any_bad(U1) (MOOD scan)",
        "rhs!(k, U1)",
        "U2 = 0.5*U + 0.5*(U1 + Δ*k)",
        "apply_bc!(U2)",
        "enforce_floors!(U2)",
        "repair_theta!(U2)",
        "enforce_Sr_constraint!(U2)",
        "apply_bc!(U2) [2]",
        "relax_dissipative!(U2)",
        "any_bad(U2) (MOOD scan)",
    ]

    nsteps = length(labels)
    bytes = zeros(Int, nsteps, reps)

    for rep in 1:reps
        GC.gc()
        @inbounds Utmp .= U0
        reset_work_warmstart!(work)

        j = 0

        j += 1
        bytes[j, rep] = @allocated hydro.rhs!(work.k, Utmp, grid, tau, model, work; Emin=hydro.E_FLOOR, force_first_order=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated (@. work.U1 = Utmp + Δ*work.k)

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U1, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_floors!(work.U1, grid, tau+Δ, model; Emin=hydro.E_FLOOR, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro._repair_theta_admissibility!(work.U1, Utmp, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_Sr_energy_constraint!(work.U1, grid, model; χ=hydro.χ_SrE, P=work.P, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U1, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.relax_dissipative!(work.U1, grid, tau+Δ, Δ, model, work; diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U1, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_floors!(work.U1, grid, tau+Δ, model; Emin=hydro.E_FLOOR, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro._repair_theta_admissibility!(work.U1, Utmp, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_Sr_energy_constraint!(work.U1, grid, model; χ=hydro.χ_SrE, P=work.P, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U1, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.any_bad(work.U1, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE)

        # Stage 2 (mirrors step_ssprk2! when the first stage is admissible)
        j += 1
        bytes[j, rep] = @allocated hydro.rhs!(work.k, work.U1, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, force_first_order=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated (@. work.U2 = 0.5*Utmp + 0.5*(work.U1 + Δ*work.k))

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U2, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_floors!(work.U2, grid, tau+Δ, model; Emin=hydro.E_FLOOR, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro._repair_theta_admissibility!(work.U2, work.U1, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.enforce_Sr_energy_constraint!(work.U2, grid, model; χ=hydro.χ_SrE, P=work.P, mask=nothing, diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.apply_bc!(work.U2, grid, tau+Δ, model)

        j += 1
        bytes[j, rep] = @allocated hydro.relax_dissipative!(work.U2, grid, tau+Δ, Δ, model, work; diag=nothing)

        j += 1
        bytes[j, rep] = @allocated hydro.any_bad(work.U2, grid, tau+Δ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE)

        @assert j == nsteps
    end

    function med_int(v::AbstractVector{Int})
        w = sort!(collect(v))
        return w[clamp(1 + length(w) ÷ 2, 1, length(w))]
    end

    med = Vector{Int}(undef, nsteps)
    for j in 1:nsteps
        med[j] = med_int(view(bytes, j, :))
    end

    total = sum(med)
    println("\n=== step_ssprk2! allocation breakdown (median over reps=$(reps)) ===")
    println("Nr=", grid.Nr, "  threads=", Threads.nthreads(), "  Δ=", Δ)
    println("Total (sum of medians): ", @sprintf("%.3f", total/1024^2), " MiB  (", total, " bytes)")
    println("\n  label                               MiB      bytes      share")
    println("  ---------------------------------------------------------------")
    for j in 1:nsteps
        b = med[j]
        share = total > 0 ? (100*b/total) : 0.0
        println("  ", rpad(labels[j], 35), " ", lpad(@sprintf("%.4f", b/1024^2), 7), "  ", lpad(string(b), 10), "  ", @sprintf("%5.1f%%", share))
    end

    return nothing
end

# ----------------------------
# Main
# ----------------------------
function main()
    do_bench = hasflag("--bench")
    do_allocs = hasflag("--allocs")
    do_breakdown = hasflag("--step-breakdown")
    tabulated = hasflag("--tabulated")
    use_lhrg = hasflag("--lhrg") || hasflag("--latticehrg")
    enable_diff = hasflag("--diff")
    enable_visc = hasflag("--visc")
    # If BenchmarkTools is not available, --bench still works via Base.@timed.

    Nr   = getkw_int("--Nr", 600)
    rmax = getkw_float("--rmax", 20.0)
    tau  = getkw_float("--tau", 0.6)

    case = build_case(; Nr=Nr, rmax=rmax, tau=tau,
                      enable_diff=enable_diff, enable_visc=enable_visc,
                      tabulated=tabulated,
                      use_lhrg=use_lhrg)

    if do_allocs
        run_allocs(case)
        return
    end
    if do_breakdown
        reps = getkw_int("--reps", 7)
        run_step_breakdown(case; reps=reps)
        return
    end
    if do_bench
        run_bench(case)
        return
    end

    # default: just do a quick smoke run of rhs!
    println("No mode given. Use --bench or --allocs.")
    reset_work_warmstart!(case.work)
    hydro.rhs!(case.dU, case.U, case.grid, case.tau, case.model, case.work)
    println("rhs! ok")
end

main()
