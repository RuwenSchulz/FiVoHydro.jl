# ==============================================================================
# test/test_gubser_viscous1d.jl — gate A4: VISCOUS Gubser flow, 1+1D bulk solver.
#
# The 1-D twin of G1v (test_gubser_viscous2d.jl), against the same semi-analytic
# referee (analytic_referees.jl `gubser_viscous`). Gubser flow carries strong radial
# ACCELERATION, so this is the gate that sees ∂_τu^r: θ = ∇·u contains ∂_τu^τ =
# v ∂_τu^r, and the NS targets of π are built from θ.
#
# 🔴 WHY IT EXISTS (2026-09-11). The 1-D relaxation built ∂_τu^r from a y_prev that
# was copied at the START of `relax_dissipative!`, where work.y holds the last RK
# STAGE, not the previous step — ∂_τu^r came out at ~4 % of its value. No 1-D gate
# could see it: FiVoBenchmark's viscous Gubser was self-convergence only, and every
# Bjorken check has u^r = 0. Part (c) below runs the OLD arithmetic
# (HYDRO_LEGACY_DTAU_UR) beside the fixed one and records both.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_gubser_viscous1d.jl
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(@__DIR__, "analytic_referees.jl"))
using .hydro
const H = hydro

const TAU0 = 1.0
const TH0  = 0.6          # T̂ at ρ = 0 (= T(τ0, r = 0) since τ0 = 1), GeV
const ALPHA = 0.0         # irrelevant: the fluid carries no charge (EOS below)
const RMAX = 10.0
const RCMP = 3.0
# A charge-free conformal fluid, which is what the referee describes. (G1v in 2-D
# uses ConformalHQEOS at α = −20 instead; the 1-D cold-start recovery does not
# converge there — README.md, known limitations — and a charge-free EOS is the
# cleaner statement of the problem anyway.)
const EOS  = H.ConformalHQEOS(m_hq = 0.0, g_hq = 0.0)
const INVFMGEV = 1/HBARC_REF

function seed!(U, g, m, τ, That, pibar)
    fill!(U, 0.0)
    for i in (g.nghost+1):(size(U,2)-g.nghost)
        r = g.rC[i]
        ρ = gubser_rho(τ, r)
        T = That(ρ)/τ
        P, _, e = H.eos_Pne(T, ALPHA*T, EOS)
        Πη = pibar(ρ)*(e + P)
        # π = Π_η[−½(l l + φ̂ φ̂) + η̂ η̂]: the de Sitter frame is isotropic in (l, φ̂)
        H.set_cell!(U, i, T, ALPHA, gubser_ur(τ, r), τ, m, g; piR = -Πη/2, piEta = Πη)
    end
    H.finalize_ic!(U, g, m; τ0 = τ)
end

function run_visc(Nr, ηs, τf, That, pibar; legacy::Bool = false)
    g = H.make_grid_1d(Nr; rmax = RMAX)
    m = H.build_model_1d(; eos = EOS, enable_shear = true, eta_over_s = ηs,
                           tauShear_coeff = 0.2, deltaShear_factor = 4/3)
    U = H.allocate_state(g, m)
    seed!(U, g, m, TAU0, That, pibar)
    H.HYDRO_LEGACY_DTAU_UR[] = legacy
    res = try
        H.run_sim_1d!(U, g, m; τ0 = TAU0, τfinal = τf, CFL = 0.15, CFLτ = 0.05)
    finally
        H.HYDRO_LEGACY_DTAU_UR[] = false
    end
    @test res.ok
    f = H.fields_1d(g, U, m; τ = res.τ, work = res.work)
    @test all(f.ok)
    sT = 0.0; sTa = 0.0; sp = 0.0; spa = 0.0
    for k in eachindex(f.r)
        f.r[k] <= RCMP || continue
        ρ = gubser_rho(res.τ, f.r[k])
        Ta = That(ρ)/res.τ
        sT += (f.T[k] - Ta)^2; sTa += Ta^2
        pbn = f.piEta[k]/(f.e[k] + f.P[k]); pba = pibar(ρ)
        sp += (pbn - pba)^2; spa += pba^2
    end
    return (L2T = sqrt(sT/sTa), L2p = sqrt(sp/spa))
end

function main()
    @testset "A4 — viscous Gubser (1+1D)" begin
        # (a) the referee: η/s → 0 is the code's own analytic ideal Gubser
        ρs0, Th0, _ = gubser_viscous(0.0; TH0)
        Tscale = H.gubser_Tscale_from_center_T(TAU0, 1.0, Th0(0.0))
        worst = maximum(abs(Th0(gubser_rho(τ, r))/τ / H.gubser_temperature(τ, r, 1.0, Tscale) - 1)
                        for τ in (1.0, 1.5, 2.0), r in 0.0:0.25:RCMP)
        wu = maximum(abs(gubser_ur(τ, r) - H.gubser_ur(τ, r, 1.0)) for τ in (1.0, 2.0), r in 0.0:0.5:6.0)
        @printf("  (a) referee at η/s = 0 vs the code's ideal Gubser: T %.2e, u^r %.2e\n", worst, wu)
        @test worst < 1e-8 && wu < 1e-14

        # (b) the solver, three resolutions, the fixed code
        for ηs in (0.005, 0.02)
            _, Th, pb = gubser_viscous(ηs; TH0)
            rs = [run_visc(N, ηs, 2.0, Th, pb) for N in (200, 400, 800)]
            for (N, r) in zip((200, 400, 800), rs)
                @printf("  (b) η/s=%.3f Nr=%4d  L2(T)=%.3e  L2(π̄)=%.3e\n", ηs, N, r.L2T, r.L2p)
            end
            oT = log2(rs[2].L2T/rs[3].L2T); op = log2(rs[2].L2p/rs[3].L2p)
            @printf("      order  T: %.2f   π̄: %.2f\n", oT, op)
            @test rs[end].L2T < 2e-3
            @test rs[end].L2p < 5e-2
            @test oT > 1.0
            @test op > 0.8
            # (c) the pre-2026-09-11 arithmetic on the finest grid, for the record
            if ηs == 0.02
                old = run_visc(800, ηs, 2.0, Th, pb; legacy = true)
                @printf("  (c) OLD ∂_τu^r (HYDRO_LEGACY_DTAU_UR), Nr=800: L2(T)=%.3e  L2(π̄)=%.3e   (fixed: %.3e, %.3e)\n",
                        old.L2T, old.L2p, rs[end].L2T, rs[end].L2p)
                @test old.L2p > 2*rs[end].L2p       # the fix is what the gate measures
            end
        end
    end
end
main()
