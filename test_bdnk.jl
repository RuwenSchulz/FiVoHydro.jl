#!/usr/bin/env julia
# Quick test for mainBDNK.jl
include(joinpath(@__DIR__, "mainBDNK.jl"))

run_sim_bdnk(;
    outdir=joinpath(@__DIR__, "snapshots", "BDNK_test"),
    Nr=200, rmax=20.0, nghost=3,
    τ0=0.4, τfinal=3.0,
    CFL=0.2, CFLτ=0.05,
    dump_dt=0.5,
    DsT=0.24,
    ε_ν_factor=:kappa,
    log_every=100,
)
println("\nBDNK test run completed!")
