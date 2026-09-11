# ==============================================================================
# test/test_terms1d.jl — gate T: the per-term switches in the 1+1D solvers.
#
# The 1-D twin of test_terms2d.jl (Gt). The switches are the shared `Terms`
# (src/terms.jl); here they reach the 1-D bulk solver (main.jl, `build_model_1d`)
# and the 1-D charm IS2 solver (main2IS2.jl, `run_static_IS2_test`).
#
# T1  the register: every `Terms` field listed once, in order; defaults.
# T2  the switched arithmetic IS the shipped arithmetic. `hq_consistent_extras` and
#     `hq_consistent_m2_rhs` keep the shipped expression as their all-on fast path
#     and a term-by-term path otherwise; the one-term-at-a-time pieces must sum to
#     the all-on value (1e-13) at random states, each piece must be nonzero, and
#     m2_vorticity / m2_projector must be inert (they vanish in radial symmetry).
# T3  on a SOLVE: `terms = :homogeneous` with consistent_fm = true reproduces
#     consistent_fm = false — bit for bit in the bulk solver, to round-off in IS2 —
#     and in IS2 `nu_gradalpha = false` removes the drive (the current never builds).
# T4  the physics of the consistent first moment: on an IDEAL fluid, Euler makes the
#     pressure-gradient and inertial terms cancel, (τ_n n/T)(∇⊥T + T a) = 0 — the
#     hidden justification of the shipped ∇α-only drive (src/hq_consistent_firstmoment.jl
#     header). Carried TOGETHER they must barely move ν; either ALONE must move it a lot.
# T5  refusals: a named term in a disabled sector, an unknown name, consistent_m2 in
#     the bulk solver, tauN_coeff ≠ 1 with consistent_fm.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_terms1d.jl
# ==============================================================================

using Printf
using Test

# A small seeded generator instead of `using Random`: under `Pkg.test()` this file runs
# as a subprocess whose load path holds only the package's declared dependencies, and
# Random is not one (the 2-D gates, run by run2d_gates.jl, never meet that restriction).
mutable struct LCG; s::UInt64; end
unif(g::LCG) = (g.s = g.s*0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(g.s >> 11) * 2.0^-53)
gauss(g::LCG) = sqrt(-2log(max(unif(g), 1e-300)))*cos(2π*unif(g))

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2IS2.jl"))
using .hydro, .hydro_current_IS2
const H1 = hydro; const HI = hydro_current_IS2

only_on(names, keep) = H1.Terms(; (n => (n === keep) for n in names)...)
const FM = (:fm_gradT, :fm_inertial, :fm_nu_gradu, :fm_expansion, :fm_dlnh)
const M2 = (:m2_nu_gradient, :m2_bg_gradu, :m2_bg_DlnT, :m2_bg_Dalpha, :m2_expansion,
            :m2_pi_sigma, :m2_PiQ_sigma, :m2_accel_nu, :m2_nu_gradTh)

function gate_T1()
    println("\nT1 — the register")
    names = [t[1] for t in H1.TERM_REGISTER]
    @test names == collect(fieldnames(H1.Terms))
    @test H1.TERM_REGISTER == HI.TERM_REGISTER            # one register, two modules
    d = H1.Terms()
    @test all(getfield(d, f) == !(f in (:m2_vorticity, :shear_vorticity)) for f in fieldnames(H1.Terms))
    @printf("  %d terms, shared by main.jl and main2IS2.jl; the two vorticity couplings default off\n", length(names))
end

function gate_T2()
    println("\nT2 — the switched arithmetic reproduces the shipped expressions")
    rng = LCG(3)
    eos = H1.LatticeHRGEOS()
    worst_fm = 0.0; worst_m2 = 0.0; minpiece_fm = Inf; minpiece_m2 = Inf; inert = 0.0
    for _ in 1:200
        τ = 0.5 + 3*unif(rng); r = 0.2 + 6*unif(rng); ur = 0.9*gauss(rng); T = 0.16 + 0.3*unif(rng)
        dtT = -0.1*unif(rng); drT = 0.05*gauss(rng); drur = 0.2*gauss(rng); dtur = 0.2*gauss(rng)
        n = 0.05*unif(rng); ν = 0.01*gauss(rng)
        h, hp = H1.hq_h_hprime(T, eos); dn = H1.hq_dn_dT(T, n, eos)
        Ds = 0.1163/T/H1.fmGeV; τn = Ds*h/T
        f(t) = H1.hq_consistent_extras(τ, r, ur, T, dtT, drT, drur, dtur, n, dn, ν, τn, Ds, h, hp; terms = t)
        all_on = f(H1.Terms())
        pieces = [f(only_on(FM, k)) for k in FM]
        worst_fm = max(worst_fm, abs(sum(pieces) - all_on)/max(abs(all_on), 1e-300))
        minpiece_fm = min(minpiece_fm, minimum(abs, pieces)/abs(all_on))
        # the second moment: rate(all) − rate(none) = Σ_k [rate(only k) − rate(none)]
        pl, pφ, PiQ = 0.01*gauss(rng), 0.01*gauss(rng), 0.01*gauss(rng)
        g(t) = collect(HI.hq_consistent_m2_rhs(τ, r, ur, T, dtT, drT, drur, dtur, pl, pφ, PiQ, ν,
                        0.3*gauss(LCG(1)), 0.2, 0.01, 0.01, (0.001, -0.002, 0.003),
                        n, τn, Ds, h, hp, 0.55τn, T*τn/2, 1.5; terms = t))
        tall = HI.Terms(); tnone = HI.Terms(; (k => false for k in M2)...)
        rall = g(tall); rnone = g(tnone)
        dsum = sum(g(HI.Terms(; (k2 => (k2 === k) for k2 in M2)...)) .- rnone for k in M2)
        worst_m2 = max(worst_m2, maximum(abs, dsum .- (rall .- rnone))/maximum(abs, rall .- rnone))
        minpiece_m2 = min(minpiece_m2, minimum(maximum(abs, g(HI.Terms(; (k2 => (k2 === k) for k2 in M2)...)) .- rnone)
                                              for k in M2)/maximum(abs, rall .- rnone))
        # the radial-zero terms: flipping them changes nothing
        inert = max(inert, maximum(abs, g(HI.Terms(; m2_vorticity = true, m2_projector = false)) .- rall))
    end
    @printf("  first moment:  Σ one-term pieces vs the shipped expression: worst rel %.1e (smallest piece %.1e)\n",
            worst_fm, minpiece_fm)
    @printf("  second moment: Σ one-term pieces vs all-on:                 worst rel %.1e (smallest piece %.1e)\n",
            worst_m2, minpiece_m2)
    @printf("  m2_vorticity on + m2_projector off in 1+1D: max change %.1e (≡ 0 in radial symmetry)\n", inert)
    @test worst_fm < 1e-12
    @test worst_m2 < 1e-12
    @test minpiece_fm > 0 && minpiece_m2 > 0
    @test inert == 0.0
end

# a small flowing fireball with charm: T and α profiles at rest, τ 0.6 → 1.8
function fireball_1d(; kw...)
    g = H1.make_grid_1d(120; rmax = 10.0)
    m = H1.build_model_1d(; eos = H1.LatticeHRGEOS(), enable_diff = true, kappa_coeff = 0.1163, kw...)
    U = H1.allocate_state(g, m)
    H1.initialize_from_radial!(U, g, m, 0.6, r -> 0.08 + 0.35*exp(-r^2/8), r -> -3.0 + 0.6*exp(-r^2/6))
    res = H1.run_sim_1d!(U, g, m; τ0 = 0.6, τfinal = 1.8)
    @test res.ok
    return U, H1.fields_1d(g, U, m; τ = res.τ, work = res.work)
end

function gate_T3()
    println("\nT3 — :homogeneous is the shipped row, on a solve; nu_gradalpha acts in IS2")
    Ush, fsh = fireball_1d()
    Uho, _   = fireball_1d(; consistent_fm = true, terms = :homogeneous)
    Uco, fco = fireball_1d(; consistent_fm = true)
    same = Ush == Uho
    @printf("  bulk 1D: consistent_fm = true, terms = :homogeneous == consistent_fm = false: %s\n", same)
    @printf("           (and the full consistent first moment moves ν by %.1f %% of max|ν|)\n",
            100*maximum(abs, fco.nur .- fsh.nur)/maximum(abs, fsh.nur))
    @test same
    @test maximum(abs, fco.nur .- fsh.nur) > 0.05*maximum(abs, fsh.nur)
    # IS2 on an analytic expanding, cooling, flowing background
    Tb(τ, r) = (0.08 + 0.35*exp(-r^2/(8 + 2τ)))*(0.6/τ)^(1/3)
    urb(τ, r) = 0.12*r*(τ - 0.5)/(1 + 0.05r^2)
    bg = HI.analytic_background(; T = Tb, ur = urb, r_grid = collect(0.0:0.05:12.0),
                                  t_grid = collect(0.5:0.01:2.5))
    run(; kw...) = HI.run_static_IS2_test(; background = bg, DsT = 0.1163, τ0 = 0.6, τfinal = 1.6, Nr = 200,
                                          rmax = 10.0, init_mode = :n_profile,
                                          n_profile = r -> 0.05*exp(-r^2/10), dump_dt = 1.0,
                                          log_every = 10^9, kw...)
    rsh = run(); rho = run(; consistent_fm = true, terms = :homogeneous); rde = run(; terms = HI.Terms())
    dh = maximum(abs, rho["nur"] .- rsh["nur"])/maximum(abs, rsh["nur"])
    @printf("  IS2: :homogeneous vs shipped max|Δν|/max|ν| = %.1e;  terms = Terms() vs no terms: %s\n",
            dh, rde["nur"] == rsh["nur"])
    @test dh < 1e-12
    @test rde["nur"] == rsh["nur"] && rde["n"] == rsh["n"]
    @test rho["terms"] == HI.resolve_terms(:homogeneous; sector_on = s -> s in (:enable_diff, :consistent_fm))
    rno = run(; terms = (nu_gradalpha = false,))
    @printf("  IS2: nu_gradalpha = false keeps ν at %.1e of the shipped max|ν| (IC ν = 0)\n",
            maximum(abs, rno["nur"])/maximum(abs, rsh["nur"]))
    @test maximum(abs, rno["nur"]) < 1e-6*maximum(abs, rsh["nur"])
    # the Refs are restored after a solve
    @test HI.IS2_CONSISTENT_FM[] == false && HI.IS2_TERMS[] == HI.Terms()
end

function gate_T4()
    println("\nT4 — Euler cancellation: fm_gradT and fm_inertial cancel on an ideal fluid")
    base = (consistent_fm = true, terms = (fm_nu_gradu = false, fm_expansion = false, fm_dlnh = false,
                                           fm_gradT = false, fm_inertial = false))
    both = (consistent_fm = true, terms = (fm_nu_gradu = false, fm_expansion = false, fm_dlnh = false))
    gT   = (consistent_fm = true, terms = (fm_nu_gradu = false, fm_expansion = false, fm_dlnh = false,
                                           fm_inertial = false))
    acc  = (consistent_fm = true, terms = (fm_nu_gradu = false, fm_expansion = false, fm_dlnh = false,
                                           fm_gradT = false))
    _, f0 = fireball_1d(; base...)
    _, fb = fireball_1d(; both...)
    _, fg = fireball_1d(; gT...)
    _, fa = fireball_1d(; acc...)
    sc = maximum(abs, f0.nur)
    # Away from the axis. The AXIS CELL (r = dr/2) carries an O(dr) mismatch between the
    # solver's acceleration (its momentum update at the reflecting axis) and the ∇T
    # difference: measured 1.43 / 0.69 / 0.34 of max|ν| at Nr = 120/240/480 — a value
    # that vanishes with dr, not a failure of the cancellation. Off the axis the pair
    # cancels to 7e-2 / 1.3e-2 / 5.7e-3 against 16 for either term alone (99.97 % at 480).
    sel = f0.r .> 0.5
    dboth = maximum(abs, (fb.nur .- f0.nur)[sel])/sc
    daxis = abs(fb.nur[1] - f0.nur[1])/sc
    dg = maximum(abs, fg.nur .- f0.nur)/sc; da = maximum(abs, fa.nur .- f0.nur)/sc
    @printf("  ν moved (rel. to max|ν| of the ∇α drive alone) by: ∇T alone %.2f, a alone %.2f, BOTH %.2e (r > 0.5 fm)\n",
            dg, da, dboth)
    @printf("  (the axis cell r = dr/2: %.2f — an O(dr) value, halves with each refinement)\n", daxis)
    @test dg > 0.3 && da > 0.3
    @test dboth < 0.01*min(dg, da)
end

function gate_T5()
    println("\nT5 — refusals")
    @test_throws ErrorException H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1, terms = (fm_inertial = false,))
    @test_throws ErrorException H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1, terms = (fm_inertia = false,))
    @test_throws ErrorException H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1, consistent_m2 = true)
    @test_throws ErrorException H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1, consistent_fm = true,
                                                   tauN_coeff = 0.5)
    @test_throws ErrorException H1.build_model_1d(; consistent_fm = true)       # no charge sector
    @test_throws ErrorException HI.show_equations_IS2(; terms = (shear_vorticity = true,))
    m = H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1, terms = :homogeneous)   # inert: fine
    @test m.terms == H1.Terms()
    io = IOBuffer(); H1.show_equations(io, H1.build_model_1d(; enable_diff = true, kappa_coeff = 0.1,
                                        consistent_fm = true, terms = H1.without(:acceleration)))
    txt = String(take!(io))
    @test occursin("[ ] −τ_n n a^i", txt) && occursin("[x] −κ ∇^⟨i⟩α", txt)
    println("  named-in-disabled-sector, typo, consistent_m2 in bulk, tauN_coeff≠1, no charge: all refused")
end

function main()
    @testset "T — term switches (1+1D)" begin
        gate_T1(); gate_T2(); gate_T3(); gate_T4(); gate_T5()
    end
end
main()
