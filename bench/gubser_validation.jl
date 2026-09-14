#!/usr/bin/env julia
# bench/gubser_validation.jl
#
# End-to-end validation run against ideal Gubser flow.
#
# Example:
#   julia --project=. --threads=auto bench/gubser_validation.jl --outdir=snapshots/gubser --Nr=400 --rmax=15 --tau0=0.6 --taufinal=4.0 --q=1.0 --Tc0=0.5

include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro
using CSV
using Tables
using Printf

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

# ----------------------------
# Run validation
# ----------------------------
function run()
    outdir   = getkw_str("--outdir", "snapshots/snapshots_gubser_validation")
    Nr       = getkw_int("--Nr", 400)
    rmax     = getkw_float("--rmax", 15.0)
    τ0       = getkw_float("--tau0", 0.6)
    τfinal   = getkw_float("--taufinal", 4.0)
    dump_dt  = getkw_float("--dump_dt", 0.1)
    CFL      = getkw_float("--CFL", 0.2)
    CFLτ     = getkw_float("--CFLτ", 0.05)

    q        = getkw_float("--q", 1.0)
    Tc0      = getkw_float("--Tc0", 0.5)

    ti = getkw_str("--time_integrator", "ssprk3")
    time_integrator = (ti == "ssprk2") ? :ssprk2 : :ssprk3

    # Recommended: conformal, baryonless EOS (no HQ sector)
    eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=0.0, g_hq=0.0)

    grid = hydro.make_grid(Nr; rmax=rmax, nghost=3)
    layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])

    hydro._check_transport_flags!(;
        advect_nur=false, relax_advect_nur=true,
        advect_Pi=false,  relax_advect_Pi=true,
        advect_pi=false,  relax_advect_pi=true
    )

    model = hydro.IdealDiffViscModel(
        eos, layout, hydro.IdealPrimRec(),
        # diffusion
        false, :alpha, 0.0, 0.0, 0.0, 0.02, 0.3, 0.3, -1.0,
        0.0, 0.0, 0.0, 0.0,
        true, false, 16, false, true,   # do_soft_project_nur, do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur
        # viscosity
        false, false, hydro.ZeroViscosity(), hydro.ZeroBulkViscosity(),
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        -1.0, -1.0,
        false, false,
        # NEW
        true, true
    )

    U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
    hydro.initialize_gubser!(U, grid, τ0, model; q=q, Tc0=Tc0, μ0=0.0)

    work = hydro.make_work(U)
    diag = hydro.DiagCounters()

    τ = τ0
    it = 0
    next_dump = τ0

    hydro.prime_work_from_U!(work, U, grid, τ, model)

    mkpath(outdir)
    hydro.clear_dir!(outdir)

        # Write config so post-processing can be one-command.
        cfg_csv = joinpath(outdir, "gubser_config.csv")
        cfg = (; tau0=τ0, taufinal=τfinal, dump_dt=dump_dt, Nr=Nr, rmax=rmax,
            q=q, Tc0=Tc0, CFL=CFL, CFLτ=CFLτ,
            time_integrator=String(time_integrator),
            eos_kind="ConformalHQEOS", eos_g_eff=40.0, eos_m_hq=0.0, eos_g_hq=0.0)
        CSV.write(cfg_csv, Tables.rowtable([cfg]))

    # Errors CSV
    err_csv = joinpath(outdir, "gubser_errors.csv")
    τs = Float64[]
    eT = Float64[]; ee = Float64[]; eur = Float64[]
    eE = Float64[]; eSr = Float64[]; eDtau = Float64[]

    # initial dump
    hydro.write_snapshot_csv(joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ)), U, grid, τ, model)
    errs0p = hydro.gubser_relL2_errors(U, grid, τ, model; q=q, Tc0=Tc0, τref=τ0, μ0=0.0)
    errs0c = hydro.gubser_relL2_errors_cons(U, grid, τ, model; q=q, Tc0=Tc0, τref=τ0, μ0=0.0)
    push!(τs, τ)
    push!(eT, errs0p.err_T); push!(ee, errs0p.err_e); push!(eur, errs0p.err_ur)
    push!(eE, errs0c.err_E); push!(eSr, errs0c.err_Sr); push!(eDtau, errs0c.err_Dtau)
    CSV.write(err_csv, (; tau=τs, err_T=eT, err_e=ee, err_ur=eur, err_E=eE, err_Sr=eSr, err_Dtau=eDtau))

    @info "Start Gubser validation" outdir=outdir Nr=Nr rmax=rmax τ0=τ0 τfinal=τfinal q=q Tc0=Tc0 time_integrator=time_integrator

    while τ < τfinal - 1e-12
        hydro.diag_reset_last!(diag)

        Δτ = hydro.compute_dt_from_work(work, grid, τ, model; CFL=CFL, CFLτ=CFLτ)
        if τ + Δτ > τfinal
            Δτ = τfinal - τ
        end

        if time_integrator == :ssprk2
            Δused = hydro.step_ssprk2!(U, grid, τ, Δτ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, diag=diag)
        else
            Δused = hydro.step_ssprk3!(U, grid, τ, Δτ, model, work; Emin=hydro.E_FLOOR, χ=hydro.χ_SrE, diag=diag)
        end

        τ += Δused
        it += 1

        if τ >= next_dump - 1e-12
            fname = joinpath(outdir, @sprintf("snapshot_tau_%06.3f.csv", τ))
            hydro.write_snapshot_csv(fname, U, grid, τ, model)

            errsp = hydro.gubser_relL2_errors(U, grid, τ, model; q=q, Tc0=Tc0, τref=τ0, μ0=0.0)
            errsc = hydro.gubser_relL2_errors_cons(U, grid, τ, model; q=q, Tc0=Tc0, τref=τ0, μ0=0.0)
            push!(τs, τ)
            push!(eT, errsp.err_T); push!(ee, errsp.err_e); push!(eur, errsp.err_ur)
            push!(eE, errsc.err_E); push!(eSr, errsc.err_Sr); push!(eDtau, errsc.err_Dtau)
            CSV.write(err_csv, (; tau=τs, err_T=eT, err_e=ee, err_ur=eur, err_E=eE, err_Sr=eSr, err_Dtau=eDtau))

            @info "dump" τ=τ it=it file=fname err_E=errsc.err_E err_Sr=errsc.err_Sr err_Dtau=errsc.err_Dtau err_T=errsp.err_T err_e=errsp.err_e err_ur=errsp.err_ur
            next_dump += dump_dt
        end
    end

    @info "Done Gubser validation" outdir=outdir it=it τ=τ err_csv=err_csv
end

run()
