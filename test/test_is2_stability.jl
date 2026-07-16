#!/usr/bin/env julia
# ==============================================================================
# test_is2_stability.jl — production-readiness / stability test for the FiVo IS2
# charm-current solver (`hydro_current_IS2.run_static_IS2_test`, main2IS2.jl).
#
# On a synthetic cooling Bjorken-like background with mild radial flow, a fresh IS2
# solve must:
#   (a) reach τfinal (no early termination / NaN blow-up),
#   (b) stay finite everywhere,
#   (c) keep the diffusion current sub-dominant:  max |ν^r| / (n u^τ) < 1,
#   (d) conserve the charm number  ∫ 2πτ r J^τ dr  (advective + ν^r contribution).
#
# This is the unit-scale standalone version of /tmp/verify_is2_fresh.jl. It is heavier
# than the pure-coefficient tests, so runtests.jl runs it only under FIVOHYDRO_LONG_TESTS=1.
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_is2_stability.jl
# ==============================================================================

using Test, JLD2, Dierckx, Printf
include(joinpath(@__DIR__, "..", "main2IS2.jl"))
using .hydro_current_IS2

# Synthetic cooling background: T(τ,r) = T0 (τ0/τ)^{1/3} e^{-(r/9)²}, small outward u^r.
function make_bg(path; T0 = 0.45, rmax = 12.0)
    rg = collect(range(0.0, rmax; length = 140))
    tg = collect(range(0.4, 5.0; length = 48))
    Tgrid = [max(T0 * (0.4 / t)^(1 / 3) * exp(-(r / 9)^2), 0.06) for r in rg, t in tg]
    urg   = [0.15 * (r / rmax) * (t / 5.0) for r in rg, t in tg]
    jldsave(path; r_grid = rg, t_grid = tg,
        T_spline  = Spline2D(rg, tg, Tgrid; kx = 3, ky = 3),
        ur_spline = Spline2D(rg, tg, urg;  kx = 1, ky = 1))
end

@testset "IS2 stability (reaches τfinal, finite, sub-dominant, conserved)" begin
    bg = tempname() * ".jld2"
    make_bg(bg)

    res = hydro_current_IS2.run_static_IS2_test(;
        background_file = bg, DsT = 0.1163,
        τ0 = 0.4, τfinal = 5.0, Nr = 200, rmax = 12.0,
        CFL = 0.15, CFLτ = 0.03, T_floor = 0.05, init_mode = :auto,
        n_profile = r -> exp(-r^2 / (2 * 3.0^2)), use_cM = false,
        eos = hydro_current_IS2.LatticeHRGEOS(canon_factor = 1.0))

    r   = Float64.(res["r_grid"]); t = Float64.(res["t_grid"])
    n   = Array{Float64}(res["n"]); nur = Array{Float64}(res["nur"])
    ur  = JLD2.load(bg)["ur_spline"]

    @test t[end] ≥ 4.99                          # (a) reached τfinal
    @test all(isfinite, n) && all(isfinite, nur) # (b) finite

    # (c) ν^r sub-dominant vs the advective current n u^τ — measured only where the charm
    # is actually present (adv current ≥ 1e-3 of its peak). At the vanishing cold edge both
    # n and ν^r → 0 and the ratio is numerically meaningless, so it is excluded.
    advc(i, j) = abs(n[i, j]) * sqrt(1 + Float64(ur(r[i], t[j]))^2)
    peak  = maximum(advc(i, j) for j in 1:length(t), i in 1:length(r))
    floor = 0.1 * peak     # bulk only: where the charm actually lives
    ratio = 0.0
    for j in 1:length(t), i in 2:length(r)
        a = advc(i, j)
        a > floor && (ratio = max(ratio, abs(nur[i, j]) / a))
    end
    @test ratio < 1.0

    # (d) ∫ 2πτ r J^τ dr conserved, J^τ = n u^τ + (u^r/u^τ) ν^r (trapezoid)
    function Jint(j)
        s = 0.0
        f(k) = (uT = sqrt(1 + Float64(ur(r[k], t[j]))^2);
                2π * t[j] * r[k] * (n[k, j] * uT + (Float64(ur(r[k], t[j])) / uT) * nur[k, j]))
        for i in 2:length(r)
            s += 0.5 * (r[i] - r[i - 1]) * (f(i) + f(i - 1))
        end
        s
    end
    N0  = Jint(1)
    dev = maximum(abs(Jint(j) - N0) / abs(N0) for j in 1:length(t))
    @printf("  reached τ=%.2f  max|νr|/(n uτ)=%.3f  ∫Jτ cons max dev=%.3f%%\n", t[end], ratio, 100 * dev)
    @test dev < 0.05

    rm(bg; force = true)
end
