#!/usr/bin/env julia
# ==============================================================================
# test_eos_consistency.jl — thermodynamic-consistency unit tests for LatticeHRGEOS
# (the production lattice-HRG + heavy-quark Boltzmann EOS in src/eos.jl), plus the
# canonical charm-suppression factor f_can(N) = I₁(N/2)/I₀(N/2).
#
# Guards refactors of the analytic P(T)/dP/dT/d²P/dT² fit: the energy density must
# stay e = -P + T dP/dT, the sound speed cs² = P'/(T P'') must be causal/physical,
# and all of P, e, n must stay positive over the physical T range. Also pins the
# Pb+Pb (N≈21.55 ⟹ ≈0.953) and O+O (N≈0.275 ⟹ ≈0.0685) canonical factors.
#
# Runnable standalone or included from runtests.jl (reuses the in-process `hydro`).
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_eos_consistency.jl
# ==============================================================================

using Test
@isdefined(hydro) || include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

@testset "LatticeHRGEOS thermodynamic consistency" begin
    eos = hydro.LatticeHRGEOS()
    Ts  = collect(0.12:0.02:0.60)

    @testset "T = $(round(T; digits=3)) GeV" for T in Ts
        P   = hydro.light_P(T, eos)
        dP  = hydro.light_dP_dT(T, eos)
        e   = hydro.light_e(T, eos)
        cs2 = hydro.eos_cs2(T, 0.0, eos)

        # positivity (P, entropy s = dP/dT, energy density)
        @test P  > 0.0
        @test dP > 0.0
        @test e  > 0.0

        # identity e ≡ -P + T dP/dT (definition of light_e — guards independent refactors)
        @test isapprox(e, -P + T * dP; rtol = 1e-12, atol = 1e-14)

        # causal, physical sound speed cs² = P'/(T P'') ∈ (0, 1); below conformal 1/3 + headroom
        @test isfinite(cs2)
        @test 0.0 < cs2 < 1.0
        @test cs2 < 0.40
    end

    @testset "total interface eos_Pne positivity" begin
        for (T, μ) in ((0.16, 0.0), (0.25, 0.0), (0.40, 0.1), (0.55, 0.2))
            P, n, e = hydro.eos_Pne(T, μ, eos)
            @test isfinite(P) && isfinite(n) && isfinite(e)
            @test P > 0.0
            @test e > 0.0
            @test n ≥ 0.0
        end
        # charm density rises with T at fixed μ (T·e^{-m/T}·K₂ envelope is increasing)
        n_lo = hydro.eos_Pne(0.20, 0.0, eos)[2]
        n_hi = hydro.eos_Pne(0.45, 0.0, eos)[2]
        @test n_hi > n_lo
    end
end

@testset "Canonical charm-suppression factor I₁(N/2)/I₀(N/2)" begin
    # Pb+Pb: near grand-canonical (the EOS default N=21.55 ⟹ f_can≈0.95240).
    # (Note: audit_coefficients.jl's 0.9526511 corresponds to a slightly different N=21.654.)
    @test isapprox(hydro.canonical_factor(21.55), 0.9524014587885938; rtol = 1e-6)
    @test 0.95 < hydro.canonical_factor(21.55) < 0.96
    # O+O: strongly suppressed (N≈0.2747 ⟹ f_can≈0.0685)
    @test isapprox(hydro.canonical_factor(0.2747), 0.0685; atol = 1e-3)
    # monotone increasing in N, asymptotes to 1
    @test hydro.canonical_factor(0.5) < hydro.canonical_factor(5.0) < hydro.canonical_factor(50.0)
    @test hydro.canonical_factor(200.0) > 0.99
    # default EOS canon_factor equals canonical_factor(21.55)
    @test isapprox(hydro.LatticeHRGEOS().canon_factor, hydro.canonical_factor(21.55); rtol = 1e-12)
end
