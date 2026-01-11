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
using BenchmarkTools
using Profile, Profile.Allocs

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
                    tabulated::Bool=false)

    grid = hydro.make_grid(Nr; rmax=rmax, nghost=3)

    eos = if !tabulated
        hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    else
        base = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        hydro.TabulatedHQEOS(base; Tmin=hydro.T_MIN, Tmax=1.5, αmin=0.0, αmax=10.0, NT=256, Nα=256)
    end

    layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    model = hydro.IdealDiffViscModel(
        eos, layout, hydro.IdealPrimRec(),
        # diffusion
        enable_diff, 0.1, 1.0, 4/3, 0.02, 0.5,
        0.0, 0.0, 0.3, 0.3,
        true, 16, false,
        # viscosity
        enable_visc, 0.08, 0.0,
        1.0, 1.0, 1.0, 4/3,
        0.0, 0.3,
        0.5, 0.5,
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
    #Δtau = hydro.compute_dt(U, grid, tau, model; CFL=0.2, CFLτ=0.05)
    Δtau = hydro.compute_dt(U, grid, tau, model, work; CFL=0.2, CFLτ=0.05)

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

    println("\n=== rhs! ===")
    @btime hydro.rhs!($dU, $U, $grid, $tau, $model, $work);

    println("\n=== any_bad (MOOD scan) ===")
    #@btime hydro.any_bad($U, $grid, $tau, $model; Emin=hydro.E_FLOOR);
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

    Profile.Allocs.clear()
    Profile.Allocs.@profile hydro.rhs!(dU, U, grid, tau, model, work)
    Profile.Allocs.print()
end

# ----------------------------
# Main
# ----------------------------
function main()
    do_bench = hasflag("--bench")
    do_allocs = hasflag("--allocs")
    tabulated = hasflag("--tabulated")
    enable_diff = hasflag("--diff")
    enable_visc = hasflag("--visc")

    Nr   = getkw_int("--Nr", 600)
    rmax = getkw_float("--rmax", 20.0)
    tau  = getkw_float("--tau", 0.6)

    case = build_case(; Nr=Nr, rmax=rmax, tau=tau,
                      enable_diff=enable_diff, enable_visc=enable_visc,
                      tabulated=tabulated)

    if do_allocs
        run_allocs(case)
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
