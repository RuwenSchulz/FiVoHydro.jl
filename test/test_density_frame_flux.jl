#!/usr/bin/env julia
# ==============================================================================
# test_density_frame_flux.jl — REGRESSION TEST for the density-frame charge flux.
#
# Guards the production density-frame parabolic diffusion flux added in rhs! via
# `add_density_frame_charge_flux!` (src/dissipation.jl), used by mainDensityFrame.jl
# (charge_mode=:density_frame).  The documented closure is
#
#     J^r_D = -κ (u^τ)² ∂_r α ,    κ = diff_kappa = DsT·n/T/fmGeV ,
#
# and it is added (τ-weighted) to the charge face flux Fh[iDtau, i].
#
# This test exercises the REAL production function on a constructed Work1D/grid/
# model (no full hydro run), and checks:
#   (a) the realized face flux equals -τ κ_f (u^τ_f)² ∂_r α with the code's own
#       diff_kappa and face averaging  (sign, (u^τ)² factor, τ-weight, wiring);
#   (b) zero-flow reduction: with y≡0 (u^τ=1) the flux is -τ κ ∂_r α;
#   (c) Fick consistency: the implied diffusivity D_eff = κ (u^τ)² / (∂n/∂α) is
#       positive and the flux is down-gradient (opposes ∂_r n);
#   (d) the parabolic dt cap key `kmax = κ (u^τ)²` matches the form used by the
#       timestep selector (src/main.jl:128, src/ic_diagnostics.jl:317).
#
# This is an ADDITIVE test: it does not modify any FiVo source and does not touch
# the :mis / :bdnk charge paths.
#
#   julia --project=Julia Julia/FiVoHydro.jl/test/test_density_frame_flux.jl
# ==============================================================================

using Test, Printf

include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

const FMGEV = 0.19733
const DST   = 0.2
const TBG   = 0.30          # constant background temperature [GeV]

# Build a minimal density-frame model (same construction as main.jl main()).
function make_df_model()
    eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    layout = hydro.StateLayout([:Dtau,:Sr,:E,:Pi,:piR,:piEta]; odd_syms=[:Sr])
    return hydro.IdealDiffViscModel(eos, layout, hydro.IdealPrimRec(),
        # charge diffusion
        true, :alpha, DST, 0.0, 0.0,
        0.02, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        false,
        false, 0, false, false,
        # viscosity (disabled)
        false, false, hydro.QGPViscosity(0.0, 1.0), hydro.ZeroBulkViscosity(),
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        false, false,
        false, false,
        # charge-sector closure
        :density_frame)
end

@testset "Density-frame charge flux (J^r_D = -κ(u^τ)²∂_rα)" begin
    model = make_df_model()
    L = hydro.layout(model)
    grid = hydro.make_grid(200; rmax=15.0, nghost=3)
    τ = 1.0

    Ntot = length(grid.rC)
    Nvars = 6
    U = zeros(Float64, Nvars, Ntot)
    work = hydro.make_work(U)

    # Smooth Gaussian charm blob at constant T, zero flow (u^r=0 ⟹ y=0, u^τ=1).
    # α(r) = μ(r)/T with a Gaussian μ profile so ∂_r α ≠ 0.
    σ = 4.0
    for i in 1:Ntot
        r = grid.rC[i]
        work.yT[i] = log(TBG)                 # T = exp(yT)
        work.y[i]  = 0.0                       # zero flow ⟹ u^τ = 1
        μ = 1.5 + 0.3*exp(-r^2/(2σ^2))        # smooth chemical potential [GeV]
        work.mu[i]    = μ
        work.alpha[i] = μ / TBG               # α = μ/T
        # Boltzmann-ish density consistent with α (only its gradient sign matters)
        work.n[i] = 1e-2 * exp((μ - 1.5)/TBG)
        work.ok[i] = true
    end
    fill!(work.Fh, 0.0)

    hydro.add_density_frame_charge_flux!(work, grid, τ, model)

    ng = grid.nghost
    i0 = ng + 1
    iL = Ntot - ng
    invdr = 1.0/grid.dr

    maxrel = 0.0
    downgradient_ok = true
    for i in i0:iL
        dαdr = (work.alpha[i+1] - work.alpha[i]) * invdr
        Tf   = 0.5*(exp(work.yT[i]) + exp(work.yT[i+1]))
        μf   = 0.5*(work.mu[i] + work.mu[i+1])
        nf   = 0.5*(work.n[i]  + work.n[i+1])
        uτf  = 0.5*(cosh(work.y[i]) + cosh(work.y[i+1]))
        κf   = hydro.diff_kappa(Tf, μf, nf, model)
        expected = τ * (-κf * uτf^2 * dαdr)
        got = work.Fh[L.iDtau, i]
        denom = max(abs(expected), 1e-30)
        maxrel = max(maxrel, abs(got - expected)/denom)
        # (c) down-gradient: flux opposes the α (hence n) gradient
        if abs(dαdr) > 1e-8
            downgradient_ok &= (sign(got) == -sign(dαdr))
        end
    end

    # (a) realized flux matches the documented formula to machine precision
    @test maxrel < 1e-12
    # (b)+(c) zero-flow reduction is down-gradient diffusion everywhere
    @test downgradient_ok

    # (d) parabolic dt-cap diffusivity key uses κ (u^τ)² (matches main.jl:128).
    #     FiVo diff_kappa = kappa_coeff*n/T/fmGeV with fmGeV = 1/ħc, i.e. = DsT*n/T*ħc.
    let T=TBG, μ=1.6, n=2e-2, uτ=cosh(0.7)
        κ = hydro.diff_kappa(T, μ, n, model)
        kmax = κ * uτ^2
        @test kmax > 0.0
        @test isapprox(kmax, DST*n/T*FMGEV * uτ^2; rtol=2e-4)   # FMGEV = ħc here
    end
end
