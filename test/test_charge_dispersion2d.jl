# ==============================================================================
# test/test_charge_dispersion2d.jl — GATE Gk: THE CHARGE DISPERSION RELATION,
# and the wavenumber at which the hydrodynamic branch CEASES TO EXIST.
#
# Gs measures the SOUND dispersion; this is its charge-sector twin, and it is the
# first measurement in this repo of the thing the programme is named after: the
# gap between the hydrodynamic and the transient mode.
#
# ---------------------------------------------------------------------------
# WHAT IS BEING MEASURED
#
# Uniform background at rest, periodic box, no shear, no bulk. The charm is a
# tracer: `eos_Pne` returns e and P INDEPENDENT of alpha (verified below to
# round-off) and `dP/dmu = 0`, so a fugacity perturbation does not push the
# fluid back. The linearised system is therefore CLOSED in two variables --
# phi = delta n / n and nu^x -- with no eigenvector subtlety in the T/u sector
# at all. That is what makes this test sharp where Gs had to fight.
#
#   d_tau phi + (1/n) d_x nu = 0                    (charge conservation)
#   tau_n d_tau nu + nu = -D_s n d_x phi            (the IS relaxation as coded)
#
# with D_s = kappa/n = (D_sT/T)/fmGeV, i.e. exactly `diff_coeffs_2d`'s kappa
# divided by n. Note d ln n / d alpha = 1 EXACTLY for the Boltzmann tracer, so
# seeding alpha = alpha0 + A cos(kx) seeds phi = A cos(kx) with no conversion.
#
# One Fourier mode gives
#
#       tau_n w^2 + i w - D_s k^2 = 0
#       w = [ -i +/- sqrt(4 tau_n D_s k^2 - 1) ] / (2 tau_n)
#
# and the whole point is the sign of that discriminant:
#
#   * 4 tau_n D_s k^2 < 1  -- OVERDAMPED. Two purely decaying modes: a
#     hydrodynamic one at rate -> D_s k^2 (Fick) and a transient one at rate
#     -> 1/tau_n. This is the regime in which a gradient expansion means
#     something.
#
#   * 4 tau_n D_s k^2 > 1  -- PROPAGATING. The two roots COLLIDE and move off
#     the imaginary axis as a complex pair, w = +/- Omega - i/(2 tau_n). There
#     is no diffusive branch any more: charge perturbations propagate and are
#     damped at 1/(2 tau_n) REGARDLESS of k. The spectral gap has closed.
#
# The collision sits at
#
#       k_* = 1 / (2 sqrt(D_s tau_n))
#
# and locating it is gate Gk-c. This is a statement about the SOLVER's own
# coefficients: every reference number below is built from `diff_coeffs_2d`, as
# G0b builds its reference from `bulk_coeffs_2d`, so what is under test is the
# DYNAMICS -- the implicit relaxation, the flux coupling, the splitting -- and
# not the coefficient formulas, which G3a already pins to round-off.
#
# ---------------------------------------------------------------------------
# THINGS THAT MUST BE RIGHT, and why
#
#  * SEED THE EXACT EIGENMODE, including nu. Gs's header records what a wrong
#    eigenvector costs: the residue excites the OTHER root, |c| beats, and a
#    straight-line fit returns a window average that looks exactly like a
#    coefficient error. Here the eigenvector is closed form,
#        nu_c = -i D_s n k A / (1 - i w tau_n),
#    so there is no excuse for seeding nu = 0. The half-window split is reported
#    for every run and is the diagnostic that this worked.
#
#  * TRAVELLING, not standing, on the propagating branch -- the same trap Gs
#    hit. Seeding a real cos(kx) with nu = 0 excites w = +Omega and w = -Omega
#    equally; the Fourier coefficient then oscillates THROUGH ZERO and its phase
#    never advances. Seeding the complex eigenvector for ONE root makes arg(c)
#    advance linearly, and Omega comes off the phase slope exactly as c_s does
#    in Gs.
#
#  * THE BACKGROUND MUST NOT DRIFT. n dilutes as 1/tau and T cools, which moves
#    both D_s (~1/T) and tau_n. Gs solved this by pushing tau0 out; the same
#    device works here and costs nothing, because CFLtau = 0.05 x 3200 is not
#    the binding timestep constraint. At tau0 = 3200 over a 20 fm/c run the
#    background moves ~0.6% in n and ~0.2% in T. Both are reported.
#
#  * THE VACUUM RAMP MUST BE INERT. `vacuum_weight_2d` multiplies the NS drive
#    by a density gate with production thresholds n_lo = 1e-6, n_hi = 2e-3, and
#    a throttled drive is a SMALLER realized D_s -- the measured 0.21x shortfall
#    recorded in TWOD_PROGRAM.md 6l. alpha0 = -2 gives n = 4.9e-2, 25x n_hi, and
#    the gate asserts it rather than trusting it.
#
#  * deltaN_factor = 0 (the model default), so the delta_N theta term is off and
#    the reference above is the equation the code actually integrates. With it
#    on, the `1` in the relaxation becomes `1 + delta_N theta` and every number
#    here would need the Bjorken theta = 1/tau folded in.
#
# ---------------------------------------------------------------------------
# THE CONTROL (Gk-d) IS THE POINT
#
# An oscillation appearing in a diffusion solver is exactly what a dispersive
# numerical artifact looks like. The control is tau_n -> 0 at the SAME k, where
# the theory says the oscillation must vanish and the mode must return to Fick:
# if the ringing were the scheme's it would not care about tau_n. Nothing else
# in the ladder distinguishes those two explanations.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl \
#          Julia/FiVoHydro.jl/test/test_charge_dispersion2d.jl
# ==============================================================================

using Printf
using Test
using SpecialFunctions

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d

# Background. T0 = 0.35 to sit where Gs runs; alpha0 = -2 to clear the vacuum
# ramp by 25x; tau0 = 3200 to freeze the Bjorken drift (see the header).
const T0   = 0.35
const A0   = -2.0
const TAU0 = 3200.0
const AMP  = 1e-3
const LBOX = 12.0
const NX   = 192
const NY   = 16

# ------------------------------------------------------------------------------
# The reference: roots of  tau_n w^2 + i w - D k^2 = 0.
# ------------------------------------------------------------------------------

"""
    modes(k, D, τn) -> (w_slow, w_fast)

Both roots, ordered so that `w_slow` is the longer-lived one. On the propagating
branch the two decay rates are equal, so the tie is broken by `Re w >= 0`, which
makes `w_slow` the mode this file always seeds.
"""
function modes(k::Float64, D::Float64, τn::Float64)
    τn <= 1e-14 && return (-im*D*k^2, complex(Inf, -Inf))
    disc = sqrt(complex(4*τn*D*k^2 - 1.0))
    w1 = (-im + disc)/(2τn)
    w2 = (-im - disc)/(2τn)
    if abs(imag(w1) - imag(w2)) < 1e-12*max(abs(imag(w1)), 1.0)
        return real(w1) >= real(w2) ? (w1, w2) : (w2, w1)
    end
    return imag(w1) > imag(w2) ? (w1, w2) : (w2, w1)   # less negative = slower
end

"""`k_*` -- where the two roots collide and the hydrodynamic branch ends."""
kstar(D::Float64, τn::Float64) = 1/(2*sqrt(D*τn))

# ------------------------------------------------------------------------------
# Choosing tau_n: by the REGIME we want, never by a bare number.
#
# Every regime in this file is defined by k/k_*, and `k_* = 1/(2 sqrt(D_s tau_n))`
# -- so pinning `tauN_coeff` to a literal pins the regimes to whatever tau_n
# HAPPENS to evaluate to. That is not hypothetical: the D8 fix (2026-09-02) moved
# tau_n by a factor 6, which would have slid Gk-a's two harmonics off the
# overdamped branch entirely. It would have failed loudly rather than silently,
# but it would have failed for the wrong reason.
#
# So the coefficient is DERIVED from the target: tau_n = 1/(4 D_s k_target^2).
# ------------------------------------------------------------------------------

const KFUND = 2π/LBOX          # the box's fundamental wavenumber

"""
    tauD_for_kstar(k_target; DsT) -> tauN_coeff placing the collision at `k_target`
"""
function tauD_for_kstar(k_target::Float64; DsT::Float64 = 0.75)
    _, n0, _ = H.eos_Pne(T0, A0*T0, H.LatticeHRGEOS())
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_diff = true,
                           kappa_coeff = DsT, tauN_coeff = 1.0)
    κ, τn1, _ = H.diff_coeffs_2d(T0, A0*T0, n0, m)
    D = κ/n0
    return 1/(4*D*k_target^2)/τn1
end

# Gk-a wants BOTH its harmonics overdamped -> put the collision above them.
# The rest want m=1 overdamped and m>=2 propagating, so the collision sits just
# above the fundamental.
const TAUD_OVER = tauD_for_kstar(2.5*KFUND)
const TAUD_PROP = tauD_for_kstar(1.4*KFUND)

# ------------------------------------------------------------------------------
# One mode run.
# ------------------------------------------------------------------------------

"""
    run_mode(; harmonic, DsT, tauD, τrun, dτs, CFL, N)

Seed the exact eigenmode of the slow root at `k = 2π·harmonic/LBOX` on a uniform
background at rest and follow the complex Fourier amplitude of `tau*J^tau`.

At rest `u = 0`, so `nu^tau = 0` by orthogonality and the conserved row IS
`tau*n` -- the mode variable is read straight off the state with no primitive
recovery in the way.

Returns the measured `w` (from the envelope and the phase, exactly as Gs
measures Gamma and c_s), the reference roots, and the background drift.
"""
function run_mode(; harmonic::Int = 1, DsT::Float64 = 0.75, tauD::Float64 = 1.0,
                    τrun::Float64 = 20.0, dτs::Float64 = 0.25,
                    CFL::Float64 = 0.15, N::Int = NX, vacuum::Bool = true)
    dx = LBOX/N
    g = H.make_grid2d(N, NY; xmax = LBOX/2, ymax = NY*dx/2)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_diff = true,
                           kappa_coeff = DsT, tauN_coeff = tauD,
                           deltaN_factor = 0.0,
                           vacuum_n_lo = vacuum ? 1e-6 : 0.0,
                           vacuum_n_hi = vacuum ? 2e-3 : 0.0)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)

    # --- background and the solver's OWN coefficients -------------------------
    _, n0, _  = H.eos_Pne(T0, A0*T0, m.eos)
    κ, τn, _  = H.diff_coeffs_2d(T0, A0*T0, n0, m)
    D  = κ/n0                       # = (D_sT/T)/fmGeV, n cancels
    k  = 2π*harmonic/LBOX
    w  = modes(k, D, τn)[1]

    # --- the exact eigenvector: nu_c = -i D n k A / (1 - i w tau_n) ----------
    νc = -im*D*n0*k*AMP/(1 - im*w*τn)
    for ix in 1:g.Nxtot, iy in 1:g.Nytot
        ph = cis(k*g.xC[ix])
        α  = A0 + AMP*real(ph)      # dln n/dalpha = 1 exactly ⇒ phi = A cos(kx)
        H.set_cell!(U, H.lin(g, ix, iy), T0, α, 0.0, 0.0, TAU0, m;
                    nux = real(νc*ph), nuy = 0.0) || error("set_cell! rejected")
    end
    H.finalize_ic!(U, g, m; τ0 = TAU0, bc = :periodic)

    L = m.layout; ng = g.nghost; iy0 = ng + NY÷2 + 1
    # complex Fourier amplitude of the conserved charge row, normalised by its
    # own mean so that the 1/tau dilution divides out
    amp() = begin
        c = 0.0 + 0.0im; b = 0.0
        for ix in (ng+1):(ng+g.Nx)
            d = U[L.iDtau, H.lin(g, ix, iy0)]
            b += d; c += d*cis(-k*g.xC[ix])
        end
        2c/(b/g.Nx)/g.Nx, b/g.Nx
    end

    τ = TAU0
    H.update_primitives_2d!(U, g, τ, m, wk; bc = :periodic)
    c0, b0 = amp()
    ts = [0.0]; la = [log(abs(c0))]; ph = [angle(c0)]
    Ts = [exp(wk.yT[H.lin(g, ng+1, iy0)])]
    while τ < TAU0 + τrun - 1e-9
        r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = min(τ+dτs, TAU0+τrun),
                          CFL = CFL, CFLτ = 0.05, work = wk, bc = :periodic)
        r.ok || error("run_sim_2d! failed at τ = $(r.τ)")
        τ = r.τ
        H.update_primitives_2d!(U, g, τ, m, wk; bc = :periodic)
        c, _ = amp()
        push!(ts, τ-TAU0); push!(la, log(abs(c))); push!(ph, angle(c))
        push!(Ts, exp(wk.yT[H.lin(g, ng+1, iy0)]))
    end
    _, b1 = amp()

    # The mode is y-independent by construction and periodic BCs preserve that, so
    # `amp()` samples a single row. That is an ASSUMPTION the entire measurement
    # rests on, so it is checked rather than trusted: max spread of the conserved
    # charge row across y, relative to its own mean.
    ysprd = 0.0
    let
        for ix in (ng+1):(ng+g.Nx)
            v0 = U[L.iDtau, H.lin(g, ix, ng+1)]
            for iy in (ng+1):(ng+g.Ny)
                ysprd = max(ysprd, abs(U[L.iDtau, H.lin(g, ix, iy)] - v0)/max(abs(v0), 1e-300))
            end
        end
    end

    # unwrap the phase (Gs does the same; a mode with Omega*dτs > π would alias,
    # which is why dτs is a keyword and is checked against the period below)
    for j in 2:length(ph)
        while ph[j]-ph[j-1] >  π; ph[j] -= 2π; end
        while ph[j]-ph[j-1] < -π; ph[j] += 2π; end
    end

    fit(x, y) = (n = length(x); mx = sum(x)/n; my = sum(y)/n;
                 sum((x .- mx).*(y .- my))/sum((x .- mx).^2))
    nn = length(ts); h = nn÷2
    wI  = fit(ts, la)                        # Im w  (negative = decaying)
    wR  = -fit(ts, ph)                       # Re w
    wI1 = fit(ts[1:h], la[1:h]); wI2 = fit(ts[h:nn], la[h:nn])
    split = abs(wI1 - wI2)/max(abs(wI), 1e-30)

    return (; k, D, τn, n0, dx, w_meas = complex(wR, wI), w_ref = w,
              w_fast = modes(k, D, τn)[2], kstar = kstar(D, τn), split,
              Tdrift = (Ts[end]-Ts[1])/Ts[1], ndrift = (b1-b0)/b0,
              nsamp = nn, ysprd, τrun, dτs, ts, la, ph)
end

_relerr(a, b) = abs(a - b)/max(abs(b), 1e-30)

"""
    fick_excess(x)

`gamma_slow / (D_s k^2)` for the overdamped branch, with `x = (k/k_*)^2`. The IS
slow root decays FASTER than Fick, and this is the whole k-dependence of that
excess -- so matching it is a test of the RELAXATION structure and not merely of
a diffusion constant. `-> 1` as `x -> 0`.
"""
fick_excess(x::Float64) = x >= 1 ? NaN : 2*(1 - sqrt(1 - x))/x

# The scheme's numerical-diffusion floor on the charge row, MEASURED in Gk-e:
# D_num = C_NUM * dx with C_NUM ~ 0.13, first order in dx. It is additive and
# positive, so on the propagating branch it can only push the damping ABOVE
# 1/(2 tau_n). This constant is the bound used there, with margin.
const C_NUM = 0.20

# ==============================================================================
@testset "Gk — the charge dispersion relation" begin

# ------------------------------------------------------------------------------
@testset "Gk-0 — the setup is what the derivation assumes" begin
    eos = H.LatticeHRGEOS()
    # (i) the tracer does not back-react: e and P must not move with alpha, or
    #     the two-variable system above is not closed.
    P1, n1, e1 = H.eos_Pne(T0, -6.0*T0, eos)
    P2, n2, e2 = H.eos_Pne(T0, +1.0*T0, eos)
    @printf("  d e/d alpha = %.3e   d P/d alpha = %.3e   (over alpha = -6 .. +1)\n",
            abs(e2-e1)/e1, abs(P2-P1)/P1)
    @test abs(e2-e1)/e1 < 1e-14
    @test abs(P2-P1)/P1 < 1e-14
    # (ii) dln n/dalpha = 1 exactly, so seeding alpha seeds phi one-for-one
    @printf("  dln n/d alpha = %.12f\n", log(n2/n1)/7.0)
    @test abs(log(n2/n1)/7.0 - 1) < 1e-12
    # (iii) the vacuum ramp is inert at the background used here
    m = H.build_model_2d(; eos = eos, enable_diff = true, kappa_coeff = 0.75)
    _, n0, _ = H.eos_Pne(T0, A0*T0, eos)
    @printf("  background n = %.4e   vacuum ramp n_hi = %.1e   weight = %.4f\n",
            n0, m.vacuum_n_hi, H.vacuum_weight_2d(n0, m))
    @test H.vacuum_weight_2d(n0, m) == 1.0
    @test n0 > 10*m.vacuum_n_hi
    # (iv) delta_N is off, so the coded relaxation is the one in the header
    _, _, δN = H.diff_coeffs_2d(T0, A0*T0, n0, m)
    @test δN == 0.0

    # (v) 🔴 tau_n against an INDEPENDENT closed form, not against a copy.
    #
    # G3a checks `transport2d.jl`'s tau_n against `src/dissipation.jl`, which is
    # the file it was copied FROM -- so it pins the duplication and can never see
    # an error the two share. This is that check done the other way: tau_n is a
    # RATIO of equilibrium moments, tau_n = D_s z K3/K2, in which the degeneracy
    # g_hq must cancel. Written out here from Bessel functions with nothing from
    # the solver but D_sT, m and T.
    #
    # It comes out a factor g_hq = 6 SHORT, at every temperature. That is not a
    # new number: README.md ("tau_n is bare") records `39da649` (2026-07-16,
    # `main2.jl::diff_tauN_bg`) and `c4fe4a0` (`main2IS2.jl`) removing "a spurious
    # 1/g_hq that made tau_n 6x too small and the diffusion signal speed
    # superluminal above T = 0.48 GeV". `main2.jl:233` carries the fix as the
    # single token `eos.g_hq *` on its `tauq` line; `src/dissipation.jl:105` is
    # otherwise the same function WITHOUT it, and `transport2d.jl` copied it
    # verbatim. The GENERIC method of that same function computes the ratio bare,
    # so the two methods of `_diff_tauN_impl` disagree by 6x depending on the EOS
    # type passed in.
    #
    # FIXED 2026-09-02 (operator instruction), in `src/dissipation.jl` and its
    # 2-D copy together, matching the convention main2.jl/main2IS2.jl and Fluidum
    # already carry. This assertion was `@test_broken` for exactly one session; it
    # is a live `@test` now and is what stops the 6x coming back. Every reference
    # in this file is built from `diff_coeffs_2d`, so Gk measures the solver
    # against its OWN tau_n and none of its other verdicts moved.
    bare(T, mm, DsT) = ((DsT/T)*(mm/T)*(SpecialFunctions.besselkx(3, mm/T) /
                                        SpecialFunctions.besselkx(2, mm/T)))/H.fmGeV
    worst = 0.0
    for T in (0.500, 0.300, 0.200, 0.156)
        a = H._diff_tauN_impl_2d(T, A0, eos, 0.7458, 1.0)
        b = bare(T, H.hq_mass(eos), 0.7458)
        worst = max(worst, abs(b/a - 1))
        @printf("  tau_n(T=%.3f): solver %.5f fm   bare D_s z K3/K2 %.5f fm   ratio %.4f\n", T, a, b, b/a)
    end
    @printf("  ⇒ worst departure from the bare moment ratio: %.4f  (g_hq = %.1f)\n",
            worst, eos.g_hq)
    @test worst < 1e-12
end

# ------------------------------------------------------------------------------
@testset "Gk-a — the hydrodynamic branch: rate -> D_s k^2" begin
    # Well below k_*, where a gradient expansion is supposed to work. Two
    # harmonics, so that the k^2 scaling is measured and not assumed.
    println("  overdamped branch (tau_n from the solver's own diff_coeffs_2d):")
    rats = Float64[]; fick = Float64[]
    for harm in (1, 2)
        r = run_mode(; harmonic = harm, DsT = 0.75, tauD = TAUD_OVER, τrun = 18.0)
        γ  = -imag(r.w_meas); γr = -imag(r.w_ref)
        xk = (r.k/r.kstar)^2
        @printf("    m=%d k=%.4f (k/k_*=%.3f) | gamma meas %.6f  ref %.6f  ratio %.5f | vs Fick meas %.4f pred %.4f (%.2f%%) | split %.4f | Re w %.2e\n",
                harm, r.k, r.k/r.kstar, γ, γr, γ/γr,
                γ/(r.D*r.k^2), fick_excess(xk),
                100*(γ/(r.D*r.k^2)/fick_excess(xk) - 1), r.split, real(r.w_meas))
        push!(rats, γ/γr); push!(fick, γ/(r.D*r.k^2)/fick_excess(xk))
        # an overdamped mode must not oscillate at all
        @test abs(real(r.w_meas)) < 0.02*γ
        @test r.split < 0.05
        @test r.Tdrift < 5e-3
        @test r.ysprd < 1e-12          # the single-row sampling is legitimate
    end
    # 2% and not tighter ON PURPOSE: gamma_slow has a SQUARE-ROOT BRANCH POINT at
    # k_*, so d gamma/gamma per d D/D is x/(2 sqrt(1-x)[1-sqrt(1-x)]) -- 1.0 at
    # k/k_* = 0.48 but 2.4 at 0.97. The measurement is 0.02% at m=1 and 0.8% at
    # m=2 for that reason, not because the second is worse.
    for x in rats; @test 0.98 < x < 1.02; end
    # The IS slow root decays FASTER than Fick (2[1-sqrt(1-x)]/x >= 1), by a
    # factor fixed entirely by k/k_*. Reproducing that k-dependence is what
    # distinguishes an IS relaxation from a rescaled diffusion constant.
    for x in fick; @test 0.98 < x < 1.02; end
end

# ------------------------------------------------------------------------------
@testset "Gk-b — the propagating branch: Omega, and damping independent of k" begin
    # Above k_* the roots have left the imaginary axis. Two signatures, and the
    # second is the sharp one: the decay rate must be 1/(2 tau_n) at EVERY k,
    # where a diffusive mode's rate grows like k^2.
    #
    # The damping is asserted as a BAND, not a value, and the band is physics:
    # the scheme's numerical diffusion on the charge row is additive and positive
    # (Gk-e measures D_num = 0.13 dx, first order), so gamma must sit between the
    # exact floor 1/(2 tau_n) and floor + C_NUM dx k^2. Asserting a bare ratio
    # would be asserting that a first-order scheme has no first-order error.
    println("  propagating branch:")
    rs = []
    for harm in (2, 3)
        r = run_mode(; harmonic = harm, DsT = 0.75, tauD = TAUD_PROP, τrun = 18.0,
                       dτs = 0.2)
        Ω  = real(r.w_meas); Ωr = real(r.w_ref)
        γ  = -imag(r.w_meas); γ0 = 1/(2r.τn)
        @printf("    m=%d k=%.4f (k/k_*=%.3f) | Omega meas %.5f ref %.5f (%.2f%%) | gamma %.5f  floor 1/(2 tau_n) %.5f  excess %.5f = %.4f dx k^2\n",
                harm, r.k, r.k/r.kstar, Ω, Ωr, 100*(Ω/Ωr-1), γ, γ0, γ-γ0,
                (γ-γ0)/(r.dx*r.k^2))
        @test r.k > r.kstar
        @test abs(imag(r.w_ref) + 1/(2r.τn)) < 1e-10   # the reference IS -i/2tau_n
        @test 0.97 < Ω/Ωr < 1.03
        @test γ0 <= γ <= γ0 + C_NUM*r.dx*r.k^2
        push!(rs, (Ω, γ, r))
    end
    # THE SIGNATURE: Omega grows with k while gamma does not.
    Ω1, γ1, r1 = rs[1]; Ω2, γ2, r2 = rs[2]
    @printf("    k %.3f -> %.3f : Omega x%.3f   gamma x%.3f   (a diffusive mode would give x%.3f)\n",
            r1.k, r2.k, Ω2/Ω1, γ2/γ1, (r2.k/r1.k)^2)
    @test Ω2/Ω1 > 1.2
    @test abs(γ2/γ1 - 1) < 0.10
end

# ------------------------------------------------------------------------------
@testset "Gk-c — the collision: locating k_*" begin
    # Scan k across the predicted collision. Below it Re w = 0 identically (the
    # roots are on the imaginary axis); above it Re w != 0. The transition
    # wavenumber is the measurement.
    DsT, tauD = 0.75, TAUD_PROP
    _, n0, _ = H.eos_Pne(T0, A0*T0, H.LatticeHRGEOS())
    mref = H.build_model_2d(; eos = H.LatticeHRGEOS(), enable_diff = true,
                              kappa_coeff = DsT, tauN_coeff = tauD)
    κ, τn, _ = H.diff_coeffs_2d(T0, A0*T0, n0, mref)
    ks = kstar(κ/n0, τn)
    @printf("  predicted k_* = %.4f 1/fm  (lambda_* = %.3f fm)   D_s = %.4f fm  tau_n = %.4f fm\n",
            ks, 2π/ks, κ/n0, τn)
    lastflat = 0.0; firstosc = Inf
    for harm in 1:6
        r = run_mode(; harmonic = harm, DsT = DsT, tauD = tauD, τrun = 10.0,
                       dτs = 0.2)
        osc = abs(real(r.w_meas)) > 0.05*abs(imag(r.w_meas))
        @printf("    m=%d k=%.4f k/k_*=%.3f | Re w %.5f  Im w %.5f | %s\n",
                harm, r.k, r.k/ks, real(r.w_meas), imag(r.w_meas),
                osc ? "PROPAGATING" : "overdamped")
        osc ? (firstosc = min(firstosc, r.k)) : (lastflat = max(lastflat, r.k))
    end
    @printf("  measured transition in (%.4f, %.4f); predicted k_* = %.4f\n",
            lastflat, firstosc, ks)
    @test lastflat < ks < firstosc
end

# ------------------------------------------------------------------------------
@testset "Gk-d — CONTROL: the oscillation is the transient mode, not the scheme" begin
    # Same k, same D_s, same grid, same scheme -- only tau_n changes. A dispersive
    # numerical artifact cannot know about tau_n; the transient mode is DEFINED by
    # it. Nothing else in the ladder separates those two explanations.
    #
    # The window has to follow the mode: at tau_n -> 0 the rate is D_s k^2 = 4.5
    # /fm, an e-fold in 0.22 fm, and a 20 fm window measures nothing but the
    # noise floor. Rate first, window second.
    println("  control, k = pi/fm fixed, tau_n scanned over 400x:")
    rows = []
    for (fac, τrun, dτs) in ((1.0, 20.0, 0.2), (0.25, 6.0, 0.1), (0.0025, 1.2, 0.04))
        tauD = TAUD_PROP*fac
        r = run_mode(; harmonic = 6, DsT = 0.75, tauD = tauD, τrun = τrun, dτs = dτs)
        γ = -imag(r.w_meas); xk = (r.k/r.kstar)^2
        @printf("    tau_n = %.4f fm  k/k_* = %.3f | Re w %.5f | gamma %.5f | gamma/(D_s k^2) %.4f%s | split %.4f\n",
                r.τn, r.k/r.kstar, real(r.w_meas), γ, γ/(r.D*r.k^2),
                xk < 1 ? @sprintf(" (Fick+IS predicts %.4f)", fick_excess(xk)) : "",
                r.split)
        push!(rows, r)
    end
    # tau_n large: propagating. tau_n -> 0: the branch point moves out past k and
    # the mode is Fick again, with NO oscillation at all.
    @test abs(real(rows[1].w_meas)) > 0.2*abs(imag(rows[1].w_meas))
    @test abs(real(rows[2].w_meas)) > 0.2*abs(imag(rows[2].w_meas))
    @test rows[3].k < rows[3].kstar
    # not `== 0.0`: this is a least-squares slope through a phase that never
    # advances, so it returns ~1e-12, not a hard zero. Gk-c's printed 0.00000 is
    # the format, not the value.
    @test abs(real(rows[3].w_meas)) < 1e-9*abs(imag(rows[3].w_meas))
    xk3 = (rows[3].k/rows[3].kstar)^2
    rat = -imag(rows[3].w_meas)/(rows[3].D*rows[3].k^2)/fick_excess(xk3)
    @printf("    tau_n -> 0 recovers Fick to %.2f%% (upper margin is the scheme, not the physics)\n",
            100*(rat-1))
    @test 0.98 < rat < 1.06
    # NON-MONOTONE, which an artifact would not be: Omega = sqrt(4 tau_n D k^2 - 1)
    # /(2 tau_n) RISES as tau_n falls, and then collapses to zero when the branch
    # point crosses k.
    @test real(rows[2].w_meas) > real(rows[1].w_meas)
end

# ------------------------------------------------------------------------------
@testset "Gk-e — where the error lives: first order in dx, and only above k_*" begin
    # On the propagating branch the exact damping is k-INDEPENDENT, 1/(2 tau_n).
    # Every bit of excess is therefore the scheme's, with an exact answer to
    # subtract -- which makes this the cleanest measurement in the ladder of the
    # charge row's numerical diffusion: one number, D_num = excess/k^2.
    #
    # 🔴 It converges at FIRST order. TWOD_PROGRAM.md 6x has since RETRACTED the
    # two earlier first-order measurements (G3g's 0.85 was a bad reference --
    # ConformalHQEOS back-reacts via P_hq = nT; the lumpy-event 1.13 just matches
    # T's own 1.09, a TVD limiter at sharp extrema). Both retractions stand and
    # neither covers this one -- they are about DIFFERENT FLUXES. G3g runs at
    # kappa = 0, so it never exercises the nu flux at all; the advective flux
    # n u^x is built from RECONSTRUCTED primitives and is second order, while nu
    # is taken cell-centred and is not. See 6aa.
    #
    # This configuration is immune to both retracted explanations by
    # construction: LatticeHRGEOS gives de/dalpha = dP/dalpha = 0 EXACTLY (Gk-0
    # asserts it), so nothing back-reacts, and the mode is one smooth sinusoid at
    # amplitude 1e-3 on a uniform background, not a lumpy event. The
    # discriminator against a limiter explanation is part (ii): the SAME smooth
    # mode on the SAME grid shows the floor above k_* and not below it, and a
    # limiter clipping smooth extrema would clip both branches identically.
    #
    # So the mechanism: what differs between the branches is whether nu CARRIES
    # the mode or is slaved to -D n grad(alpha), and `rhs2d.jl:269` states that
    # the dissipative dofs -- nu among them -- are taken CELL-CENTRED at the
    # faces rather than reconstructed, so the flux they carry is first-order
    # accurate where the ideal ones are second.
    #
    # A SUSPECT, not a verdict -- but 6x sharpens it rather than weakening it:
    # with the advective charge flux now shown second order, nu is the one charge
    # flux left that is not, and it is the one the code says is unreconstructed.
    # The decisive test is to reconstruct nu and see part (i) go second order
    # while part (ii) barely moves. Not done here.
    println("  (i) propagating branch (m=4), dx scanned:")
    dn = Float64[]; dxs = Float64[]
    for N in (96, 192, 384)
        r = run_mode(; harmonic = 4, DsT = 0.75, tauD = TAUD_PROP, τrun = 12.0,
                       dτs = 0.2, N = N)
        γ = -imag(r.w_meas); γ0 = 1/(2r.τn)
        push!(dn, (γ-γ0)/r.k^2); push!(dxs, r.dx)
        @printf("      N=%4d dx=%.5f k/k_*=%.3f | gamma %.6f  excess %.6f | D_num %.4e = %.4f dx  (%.2f%% of D_s)\n",
                N, r.dx, r.k/r.kstar, γ, γ-γ0, dn[end], dn[end]/r.dx, 100*dn[end]/r.D)
    end
    ord = [log2(dn[i]/dn[i+1]) for i in 1:length(dn)-1]
    @printf("      observed order in dx: %s\n", join(map(o -> @sprintf("%.2f", o), ord), ", "))
    @test all(dn .> 0)                      # additive and positive, as dissipation must be
    @test dn[2] < dn[1] && dn[3] < dn[2]
    for o in ord; @test 0.80 < o < 1.35; end   # FIRST order. See the note above.

    println("  (ii) overdamped branch (m=1), same scan — NO floor:")
    devs = Float64[]
    for N in (96, 192, 384)
        r = run_mode(; harmonic = 1, DsT = 0.75, tauD = TAUD_OVER, τrun = 12.0, N = N)
        γ = -imag(r.w_meas); γr = -imag(r.w_ref)
        push!(devs, γ/γr - 1)
        @printf("      N=%4d dx=%.5f k/k_*=%.3f | gamma %.7f ref %.7f | dev %+.4f%%\n",
                N, r.dx, r.k/r.kstar, γ, γr, 100*devs[end])
    end
    # 30x smaller than (i) at the same dx, and it CHANGES SIGN -- so it is not a
    # diffusion floor at all, it is two small errors of opposite sign.
    @test maximum(abs.(devs)) < 3e-3
    @test minimum(devs) < 0 < maximum(devs)

    println("  (iii) the same error is NOT the timestep:")
    gs = Float64[]
    for CFL in (0.30, 0.15)
        r = run_mode(; harmonic = 4, DsT = 0.75, tauD = TAUD_PROP, τrun = 12.0,
                       dτs = 0.2, CFL = CFL)
        push!(gs, -imag(r.w_meas))
        @printf("      CFL %.3f | gamma %.6f\n", CFL, gs[end])
    end
    # halving dtau at fixed dx must move gamma by far less than halving dx did
    dtau_effect = abs(gs[2]-gs[1])
    dx_effect   = abs(dn[2]-dn[1])*(2π*4/LBOX)^2
    @printf("      halving dtau moves gamma by %.3e; halving dx moved it by %.3e (%.0fx)\n",
            dtau_effect, dx_effect, dx_effect/max(dtau_effect, 1e-30))
    @test dtau_effect < 0.25*dx_effect
end

end # Gk
