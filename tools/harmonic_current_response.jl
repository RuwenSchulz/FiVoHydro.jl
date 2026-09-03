# =================================================================================================
# harmonic_current_response.jl — the charm diffusion current's response to its OWN constitutive
# target, resolved in azimuthal harmonics. The vector generalisation of the programme's scalar R.
#
# ── WHY THIS NEEDS 2+1D ──────────────────────────────────────────────────────────────────────────
# In 1+1D radial everything is collinear: grad(alpha), u and nu all point along r-hat, so a closure
# can only fail in MAGNITUDE. That is what R = nu_hydro/nu_NS measures, and it is the number the
# whole programme quotes (LangevinPaperOO: R = 0.06-0.32 on Sigma). In 2+1D the current has two more
# ways to be wrong:
#   * by DIRECTION  -- nu need not be parallel to -kappa grad(alpha);
#   * by HARMONIC   -- each azimuthal m has its own target with its own history.
# Neither is expressible on an azimuthally symmetric background.
#
# ── WHAT IS MEASURED ─────────────────────────────────────────────────────────────────────────────
# Charm-weighted azimuthal decomposition, over cells above freeze-out, of the RADIAL current and of
# the solver's own Navier-Stokes target built from the same primitives:
#
#     c_m[X] = sum_cells (n * X * e^{-i m phi}) / sum_cells n
#
# for X = nu^r and X = nu_NS^r, giving per harmonic
#     R_m      = |c_m[nu]| / |c_m[nu_NS]|          the lag in MAGNITUDE (the usual R, per m)
#     dpsi_m   = arg(c_m[nu]) - arg(c_m[nu_NS])    the lag in ANGLE -- the event plane of the
#                                                  current against the event plane of its target
# plus a pointwise ALIGNMENT cosine between the vectors nu^i and nu_NS^i, which is the direction
# failure with no reference to any plane at all.
#
# 🔑 R_m and the cosine are RATIOS of like quantities and dpsi_m is a PHASE, so all three are immune
# to the normalisation traps that have repeatedly bitten this programme ("compare nu-hat, never nu").
#
# ── TWO COMPETING PREDICTIONS, stated before looking ─────────────────────────────────────────────
# (a) HIGHER m LAGS MORE. nu relaxes toward nu_NS at rate 1/tau_n, which is m-INDEPENDENT, while the
#     target itself decays at gamma_m growing with m. Quasi-steady gives R_m = 1/(1 + gamma_m tau_n),
#     decreasing in m.
# (b) HIGHER m LAGS LESS. The MODE damping is 1/tau_n + D_s k_m^2, so short-wavelength structure
#     reaches its attractor sooner.
# These differ because (a) is about the response to a moving target and (b) about the decay of a
# free mode. I stated (b) first in this session and now think (a) is the right frame -- but the
# measurement is what settles it, which is the point of writing both down.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl \
#          Julia/FiVoHydro.jl/tools/harmonic_current_response.jl
# =================================================================================================

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d
using Printf

const DATA  = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation", "PbPb", "data"))
const T_FO  = 0.1565
const MMAX  = 4

"""One run; returns (grid, model, state, work) at each requested time."""
function run_case(ic; N = 200, DsT = 0.1163, τf = 8.0, snaps = (1.0, 2.0, 4.0, 6.0, 8.0))
    g = H.make_grid2d(N, N; xmax = 16.0, ymax = 16.0)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = 0.10, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = 0.10, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = DsT, tauN_coeff = 1.0,
        pi_clip_factor = 1.0)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_from_grid_csv!(U, g, m, 0.4, joinpath(DATA, ic))
    out = []
    τ = 0.4; δ = 0.05
    for ts in snaps
        # advance to ts - delta, record alpha, then step delta and analyse THERE with a
        # backward difference. nu_NS = -kappa(d_i alpha + u^i D alpha) and D carries
        # d_tau alpha, which README.md records as "~2x on a flowing background" -- setting
        # it to zero does not approximate the target, it changes which vector it is.
        r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = ts-δ, CFL = 0.2, CFLτ = 0.02, work = wk)
        r.ok || break
        τ = r.τ
        H.update_primitives_2d!(U, g, τ, m, wk)
        αprev = copy(wk.alpha)
        r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = τ+δ, CFL = 0.2, CFLτ = 0.02, work = wk)
        r.ok || break
        Δ = r.τ - τ; τ = r.τ
        H.update_primitives_2d!(U, g, τ, m, wk)
        push!(out, (τ = τ, snap = analyse(U, g, m, wk, αprev, Δ)))
    end
    return out
end

"""Charm-weighted harmonics of nu^r and nu_NS^r, plus the pointwise alignment."""
function analyse(U, g, m, wk, αprev, Δ)
    L = m.layout; ng = g.nghost; dx = g.xC[2]-g.xC[1]
    cν  = zeros(ComplexF64, MMAX+1); cNS = zeros(ComplexF64, MMAX+1)
    wsum = 0.0; cosum = 0.0; cow = 0.0; Rpt = 0.0; Rw = 0.0; ncell = 0
    for ix in (ng+2):(ng+g.Nx-1), iy in (ng+2):(ng+g.Ny-1)
        i = H.lin(g, ix, iy)
        T = exp(wk.yT[i]); n = wk.n[i]
        (T <= T_FO || n <= 1e-12) && continue
        x = g.xC[ix]; y = g.yC[iy]; rr = hypot(x, y); rr < 1e-9 && continue
        ux = wk.ux[i]; uy = wk.uy[i]; uτ = sqrt(1+ux*ux+uy*uy)

        # the solver's OWN NS target, from the same primitives it evolves on
        κ, _, _ = H.diff_coeffs_2d(T, wk.mu[i], n, m)
        dxa = (wk.alpha[i+g.Nytot] - wk.alpha[i-g.Nytot])/(2dx)
        dya = (wk.alpha[i+1]       - wk.alpha[i-1])/(2dx)
        dta = (wk.alpha[i] - αprev[i])/Δ
        nsx, nsy = H.ns_diffusion_target_2d(ux, uy, uτ, dxa, dya, dta, κ)

        νx = H.phys_from_stored(U[L.iNux,i]); νy = H.phys_from_stored(U[L.iNuy,i])
        c = x/rr; s = y/rr
        νr   = νx*c + νy*s
        nsr  = nsx*c + nsy*s
        φ    = atan(y, x)
        for mm in 0:MMAX
            e = cis(-mm*φ)
            cν[mm+1]  += n*νr*e
            cNS[mm+1] += n*nsr*e
        end
        wsum += n; ncell += 1
        # pointwise direction failure, weighted by charm and by the target's size so
        # that near-zero cells do not dominate an angle
        a = hypot(νx, νy); b = hypot(nsx, nsy)
        if a > 1e-30 && b > 1e-30
            w = n*b
            cosum += w*(νx*nsx + νy*nsy)/(a*b); cow += w
            Rpt += w*(a/b); Rw += w
        end
    end
    return (cν = cν./max(wsum,1e-300), cNS = cNS./max(wsum,1e-300),
            cosθ = cosum/max(cow,1e-300), Rmag = Rpt/max(Rw,1e-300), ncell = ncell)
end

function report(label, ic)
    println("\n" * "="^100)
    println(label, "   (", ic, ")")
    println("="^100)
    # 🔑 R_m = |c_m[nu]|/|c_m[nu_NS]| is a ratio whose DENOMINATOR CROSSES ZERO -- the O+O
    # analysis records the same pole ("R has a pole at the nu_NS zero"). So the magnitudes are
    # printed SEPARATELY and R is only shown where the target carries real weight
    # (|c_m[nu_NS]| >= 5% of the m=0 target). Elsewhere it is a dash, not a number.
    @printf("  %5s %6s %8s %9s | per m:  |c_m[nu]|  |c_m[nu_NS]|  R_m  dpsi_m[rad]\n",
            "tau", "cells", "cos", "<|nu|/|nu_NS|>")
    for rec in run_case(ic)
        s = rec.snap
        ref = abs(s.cNS[1])
        @printf("  %5.2f %6d %8.4f %9.3f |", rec.τ, s.ncell, s.cosθ, s.Rmag)
        for mm in 0:MMAX
            a = abs(s.cν[mm+1]); b = abs(s.cNS[mm+1])
            if b < 0.05*ref
                @printf("  m=%d %8.2e %8.2e   --      --", mm, a, b)
            else
                dψ = mod(angle(s.cν[mm+1]) - angle(s.cNS[mm+1]) + π, 2π) - π
                @printf("  m=%d %8.2e %8.2e %6.3f %+6.3f", mm, a, b, a/b, dψ)
            end
        end
        println()
    end
end

report("ENSEMBLE 20-30% — eps2 by construction, eps3 ~ 0 (the CONTROL)", "ic2d_20-30.csv")
report("SINGLE EVENT ev02 — eps2 = 0.204, eps3 = 0.227", "ic2d_20-30_ev02.csv")

# =================================================================================================
# MEASURED 2026-09-03  (N = 200, D_sT = 0.1163, all sectors, corrected tau_n, pi_clip = 1)
#
# ── TWO INTERNAL CONSISTENCY CHECKS, both pass ───────────────────────────────────────────────────
# * The ENSEMBLE IC has eps2 by construction and eps3 ~ 0, and the decomposition returns weight in
#   m = 0, 2, 4 only -- m = 1 and 3 fall below the 5% cut at nearly every time. An ellipse has
#   2-fold symmetry, so that is what it must do.
# * The SINGLE EVENT carries m = 1 and m = 3 at comparable weight to m = 2, as a fluctuating IC must.
# Neither was arranged; they are what tells us the harmonics are the fireball's and not the grid's.
#
# ── 1. THE DIRECTION FAILURE IS REAL, AND IT IS A FLUCTUATION EFFECT ─────────────────────────────
#   pointwise cos<nu, nu_NS>, charm- and target-weighted:
#     ensemble      0.940  0.943  0.979  0.936  0.980      (tau = 1,2,4,6,8)
#     single event  0.969  0.837  0.848  0.787  0.684
# On a smooth elliptic background the current stays aligned with its target to ~2-6%. On a single
# fluctuating event the alignment DEGRADES MONOTONICALLY to 0.68 -- a ~47 degree spread. This is a
# way for the closure to be wrong that 1+1D cannot represent at all, since there grad(alpha), u and
# nu are collinear by construction.
#
# ── 2. THE LAG ANGLE IS SMALL -- my proposed observable is REAL BUT WEAK ─────────────────────────
# dpsi_m, the phase of c_m[nu] against c_m[nu_NS] (the current's event plane against its target's),
# stays |dpsi| <~ 0.3 rad on every harmonic that carries weight, at every time, in both cases. So
# the current's PLANE tracks its target's plane closely; what fails is the magnitude and the
# POINTWISE direction. => the alignment cosine is the better observable, and the event-plane lag
# angle I proposed earlier in this session is not where the effect lives.
#
# ── 3. NEITHER PREDICTION (a) NOR (b) IS RIGHT ───────────────────────────────────────────────────
# R_m at fixed tau INCREASES with m (single event, tau = 4): m=0 1.36, m=2 1.40, m=3 1.92, m=4 2.59.
# But R > 1 is not "closer to the attractor" -- it is OVERSHOOT: nu exceeds its own constitutive
# value. The m = 0 history says what is happening:
#     |c_0[nu]|     1.05e-4  6.72e-4  9.67e-4  6.26e-4  3.07e-4     (ensemble, tau = 1..8)
#     |c_0[nu_NS]|  4.69e-4  8.97e-4  7.61e-4  2.30e-4  6.12e-5
# The TARGET collapses ~8x from tau = 2 to 8 while the current falls ~2x. Early the current lags
# (R_0 = 0.225 at tau = 1, the familiar R < 1); late it is a RELIC and the ratio inverts. Higher m
# overshoot more because their targets die faster.
# 🔑 So the robust statement is the MAGNITUDES, not R: nu decays far more slowly than the constitutive
# value it is supposed to track. R_m is a ratio with a collapsing denominator and is quoted here only
# where the target still carries >= 5% of the m = 0 weight.
#
# ── WHAT THIS IS NOT ─────────────────────────────────────────────────────────────────────────────
# This compares the IS current against ITS OWN NS target -- it is the programme's R generalised to
# vectors and harmonics, NOT a hydro-vs-transport comparison. Doing that needs a Langevin twin on a
# 2-D background, which does not exist (kernels_gpu.jl takes a radial background with an ad-hoc v2
# modulation). That remains the blocking item for anything observable-level in 2+1D.
# =================================================================================================
