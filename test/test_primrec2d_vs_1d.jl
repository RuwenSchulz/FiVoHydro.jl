# ==============================================================================
# test/test_primrec2d_vs_1d.jl
#
# Cross-validation of the 2+1D primitive recovery against the 1-D PRODUCTION
# recovery, on the (T, φ) locus that the production initial condition actually
# occupies (data/initial_profiles_physical.csv: T ∈ [0.128, 0.563],
# φ = (μ-m)/T ∈ [-9.84, -4.23]).
#
# The 2-D state is built to be the SAME physical state as the 1-D one: u^y = 0,
# π^{xy} = 0, and the shear mapping established in test_shear2d_algebra.jl,
#
#     pixx = (u^τ)² piR       piyy = π^φ_φ = -(piR + piEta)
#
# (`piR` in src/shear_tensor.jl is the LOCAL-REST-FRAME amplitude, not π^r_r:
#  Π^{rr} = (u^τ)² piR. See TWOD_PROGRAM.md §6a.)
#
# ------------------------------------------------------------------------------
# THIS FILE ALSO RECORDS THE D3 MEASUREMENT.
#
# Question: the 1-D recovery accepts on one GLOBAL residual norm,
# `resnorm < tol_res*(1 + |D| + |Sr| + |E|)` (src/primrec.jl). With the charm EOS
# the charge row is O(n) ~ 1e-4 while the energy row is O(e) ~ 1e3, so the charge
# equation is bounded far more loosely than the energy one. Does that cost the
# charm sector accuracy in production?
#
# Answer: the mechanism is REAL but the magnitude is irrelevant. Measured worst
# relative charge error along the production locus:
#
#       T     1-D dn/n     2-D dn/n (row-equilibrated)
#     0.130   3.3e-12      1.5e-14
#     0.200   4.7e-13      1.4e-14
#     0.300   3.5e-14      1.6e-14
#     0.563   7.9e-15      1.1e-14
#
# The 1-D error grows toward the dilute edge (n = 2.3e-4 at T = 0.13) exactly as
# the loose bound predicts — 220x worse than a properly scaled solve — while the
# 2-D version is flat. But 3.3e-12 is ~10 orders of magnitude below any quoted
# number, so NO published result is affected and NO production file was changed.
#
# What saves the 1-D solver is accidental: its loop test is `nrm < tol` with
# tol = 1e-15 ABSOLUTE, which is unreachable once E ~ 1e3, so the iteration runs
# on until the line search stagnates — i.e. to full floating-point convergence.
# The loose gate only says what it would ACCEPT, not what it produces.
#
# The bound below (1e-9) is a REGRESSION guard, not the achieved accuracy.
# ==============================================================================

using Printf
using Random
using Test
using SpecialFunctions

const _HERE = @__DIR__
const _ROOT = normpath(joinpath(_HERE, ".."))

include(joinpath(_ROOT, "src", "constants.jl"))
include(joinpath(_ROOT, "src", "utils.jl"))
include(joinpath(_ROOT, "src", "eos.jl"))
include(joinpath(_ROOT, "src", "primitives.jl"))
include(joinpath(_ROOT, "src", "state_layout.jl"))
include(joinpath(_ROOT, "src", "shear_tensor.jl"))
include(joinpath(_ROOT, "src", "primrec.jl"))          # 1-D production recovery
include(joinpath(_ROOT, "src2d", "shear2d.jl"))
include(joinpath(_ROOT, "src2d", "state_layout2d.jl"))
include(joinpath(_ROOT, "src2d", "primitives2d.jl"))
include(joinpath(_ROOT, "src2d", "primrec2d.jl"))      # 2-D recovery under test

const LAY1 = StateLayout([:Dtau, :Sr, :E, :nur, :Pi, :piR, :piEta])
const LAY2 = make_layout2d()

# Production locus: T and the corresponding φ. φ is deeply negative because
# m_hq/T is large — charm is a dilute tracer everywhere in the fireball.
const LOCUS = ((0.130, -9.94), (0.200, -9.10), (0.300, -6.60),
               (0.400, -5.35), (0.500, -4.60), (0.5633, -4.26))

"""One matched pair of recoveries. Guesses are perturbed by `pert` — seeding with
the exact solution would give iters = 0 and measure nothing."""
function matched_pair(eos, w1, w2, T, φ, ur, τ, diss, rng; pert = 1e-2)
    μ = hq_mass(eos) + T*φ
    α = μ/T
    P, n, e = eos_Pne(T, μ, eos)
    (!isfinite(P) || e <= 0) && return nothing

    Pi    = diss*P*(2rand(rng) - 1)
    piR   = diss*P*(2rand(rng) - 1)
    piEta = diss*P*(2rand(rng) - 1)
    nur   = diss*n*(2rand(rng) - 1)

    # ---------------- 1-D ----------------
    U1 = zeros(length(LAY1.names), 1)
    ok, _ = prim_to_cons_col_ideal_phi_diff_visc!(U1, 1, log(T), φ, asinh(ur),
                                                  nur, Pi, piR, piEta, 0.0, τ, eos, LAY1)
    ok || return nothing
    D1 = U1[LAY1.iDtau,1]/τ; Sr = U1[LAY1.iSr,1]; E1 = U1[LAY1.iE,1]

    T1, μ1, ur1, nA, _, _, o1 = cons_to_prim_ideal_phi_diff_visc!(
        w1, D1, Sr, E1, nur, Pi, piR, piEta, 0.0, τ, eos;
        yT0 = log(T) + pert*(2rand(rng)-1),
        φ0  = φ      + 10*pert*(2rand(rng)-1),
        y0  = asinh(ur) + pert*(2rand(rng)-1))

    # ---------------- 2-D, same physical state ----------------
    uτ    = sqrt(1 + ur^2)
    pixx  = uτ^2 * piR
    piyy  = -(piR + piEta)
    pieta, _ = project_shear_traceless_2d(ur, 0.0, uτ, pixx, 0.0, piyy, 0.0)

    U2 = zeros(nvars(LAY2), 1)
    ok2, _ = prim_to_cons_2d!(U2, 1, T, μ, ur, 0.0, nur, 0.0, Pi,
                              pixx, 0.0, piyy, pieta, τ, eos, LAY2)
    ok2 || return nothing

    T2, μ2, ux2, uy2, nB, _, _, o2 = cons_to_prim_2d!(
        w2, U2[LAY2.iDtau,1]/τ, U2[LAY2.iSx,1], U2[LAY2.iSy,1], U2[LAY2.iE,1],
        nur, 0.0, Pi, pixx, 0.0, piyy, pieta, τ, eos;
        yT0 = log(T) + pert*(2rand(rng)-1),
        φ0  = φ      + 10*pert*(2rand(rng)-1),
        ux0 = ur     + pert*(2rand(rng)-1),
        uy0 = 0.0)

    return (o1 = o1, o2 = o2, E1 = E1, E2 = U2[LAY2.iE,1], n = n,
            # the two solvers must agree that this is the same state
            dE      = abs(U2[LAY2.iE,1] - E1)/max(abs(E1), 1e-300),
            dS      = abs(U2[LAY2.iSx,1] - Sr)/max(abs(Sr), 1e-300),
            dD      = abs(U2[LAY2.iDtau,1] - U1[LAY1.iDtau,1])/max(abs(U1[LAY1.iDtau,1]), 1e-300),
            dalpha1 = abs(μ1/T1 - α), dn1 = abs(nA - n)/max(n, 1e-300), dur1 = abs(ur1 - ur),
            dalpha2 = abs(μ2/T2 - α), dn2 = abs(nB - n)/max(n, 1e-300),
            dux2    = abs(ux2 - ur), duy2 = abs(uy2))
end

@testset "2-D vs 1-D primitive recovery on the production locus" begin
    eos = LatticeHRGEOS()
    w1  = PrimRecWork()
    w2  = PrimRecWork2D()

    @printf("%7s %7s %11s %11s | %-21s | %-21s\n",
            "T", "phi", "E", "n", "1-D  d(alpha)  dn/n", "2-D  d(alpha)  dn/n")

    worst_cons = 0.0
    for (T, φ) in LOCUS
        rng = MersenneTwister(2024)
        a1 = 0.0; n1 = 0.0; a2 = 0.0; n2 = 0.0
        Em = 0.0; nn = 0.0; f1 = 0; f2 = 0; wuy = 0.0; nsamp = 0
        for _ in 1:400
            r = matched_pair(eos, w1, w2, T, φ, 2.0*rand(rng), 0.4 + 12*rand(rng), 0.05, rng)
            r === nothing && continue
            nsamp += 1
            r.o1 || (f1 += 1)
            r.o2 || (f2 += 1)
            a1 = max(a1, r.dalpha1); n1 = max(n1, r.dn1)
            a2 = max(a2, r.dalpha2); n2 = max(n2, r.dn2)
            wuy = max(wuy, r.duy2); Em = max(Em, r.E1); nn = r.n
            worst_cons = max(worst_cons, r.dE, r.dS, r.dD)
        end
        @printf("%7.3f %7.2f %11.3e %11.3e | %10.2e %10.2e | %10.2e %10.2e\n",
                T, φ, Em, nn, a1, n1, a2, n2)

        @test nsamp > 0
        @test f1 == 0            # 1-D must not fail on its own production locus
        @test f2 == 0            # nor must the 2-D solver
        @test a1 < 1e-9          # D3 regression guard (achieved: <= 3.3e-12)
        @test a2 < 1e-9
        @test n1 < 1e-9
        @test n2 < 1e-9
        @test wuy == 0.0         # 2-D must keep u^y identically zero on a 1-D state
    end

    # The 2-D forward map must reproduce the 1-D conserved variables exactly:
    # if it did not, the two solvers would not be inverting the same problem and
    # the accuracy comparison above would be meaningless.
    @printf("  worst |ΔU|/|U| between the 1-D and 2-D forward maps = %.3e\n", worst_cons)
    @test worst_cons < 1e-13
end
