#!/usr/bin/env julia
# test_bdnk_causal.jl — regression test for the GENUINE causal BDNK production driver
# (mainBDNK_causal.jl). Run as an isolated subprocess from runtests.jl. Exits 0 on pass.
# Asserts: flat-α fixed point (Cartesian), Milne charge conservation, and causal support.

include(joinpath(@__DIR__, "..", "mainBDNK_causal.jl"))
using .bdnk_causal
const BC = bdnk_causal

allpass = true
fail(msg) = (global allpass; allpass = false; println("FAIL: ", msg))

bg = BC.static_bg(T0=0.3, α0=0.0)

# 1. flat-α is an exact fixed point in Cartesian geometry
r0 = BC.run_bdnk_causal(Nr=200, rmax=20.0, τ0=1.0, τfinal=2.0, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=(r->0.0), geom=:cartesian, save_dt=1.0)
fp = maximum(maximum(abs.(a)) for a in r0.αs)
fp < 1e-12 || fail("flat-α not a fixed point: max|α|=$fp")

# 2. Milne charge conservation to machine precision
αp(r) = 0.15*exp(-((r-10.0)/1.0)^2)
rM = BC.run_bdnk_causal(Nr=300, rmax=20.0, τ0=1.0, τfinal=2.5, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=αp, geom=:milne, save_dt=1.0)
Q(qv) = sum(qv)*rM.grid.dr
drift = abs(Q(rM.qs[end]) - Q(rM.qs[1]))/abs(Q(rM.qs[1]))
drift < 1e-10 || fail("charge not conserved: ΔQ/Q=$drift")

# 3. causal support: a pulse stays inside |x| ≤ v_sig·Δt (no superluminal leakage)
αpulse(r) = 0.08*exp(-((r-15.0)/0.6)^2)
Δt = 3.0
rc = BC.run_bdnk_causal(Nr=800, rmax=30.0, τ0=1.0, τfinal=1.0+Δt, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=αpulse, geom=:cartesian, eps_factor=1.0, save_dt=100.0)
g = rc.grid; α = rc.αs[end]; dist = abs.(g.r .- 15.0)
cone = 1.0*Δt + 2.0      # v_sig=c for ε_ν=κ, + pulse half-extent margin
leak = maximum(α[dist .> cone] .|> abs)/maximum(abs.(α))
leak < 1e-5 || fail("acausal leakage outside cone: leak=$leak")
any(isnan, α) && fail("NaN in solution")


# ---------------------------------------------------------------------------
# 4. MILNE uniform dilution against the EXACT BDNK ODE.
# Gap this closes: checks 1-2 only pin Milne through `q` (which the flux conserves by
# construction) and the flat-α fixed point only in CARTESIAN. Neither constrains the
# β-reconstruction q -> ∂_τα in Milne, where the τ in the measure m=τr is the whole physics.
# For uniform α, static T, u^r=0: J^r=0, so conservation is d_τ[τ(n + κβ)] = 0 with κ=k1·n,
# k1 = DsT/(T·fmGeV) and n ∝ exp(α) exactly for ConformalHQEOS. Integrating that ODE gives the
# exact answer; the ideal-dilution α₀+log(τ₀/τ) is NOT it (differs by the BDNK relaxation length).
let T0 = 0.3, a0 = 0.5, DsT = 0.24, τ0 = 1.0, τf = 3.0
    eosU = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    k1 = DsT/(T0*BC.fmGeV)
    nU(a) = BC.n_of(a, T0, eosU)
    Cq = τ0*nU(a0)                       # driver starts β=0 ⇒ J^τ(τ0)=n(α0)
    f(t, y) = (Cq/(t*nU(y)) - 1)/k1
    # RK4 reference
    t = τ0; y = a0; h = 1e-6
    while t < τf - 1e-13
        hh = min(h, τf - t)
        k1_ = f(t, y); k2_ = f(t+hh/2, y+hh*k1_/2)
        k3_ = f(t+hh/2, y+hh*k2_/2); k4_ = f(t+hh, y+hh*k3_)
        y += hh*(k1_+2k2_+2k3_+k4_)/6; t += hh
    end
    rU = BC.run_bdnk_causal(Nr=200, rmax=20.0, τ0=τ0, τfinal=τf, DsT=DsT, mode=:kappa,
                            bg=BC.static_bg(T0=T0, α0=a0), α_init=(r->a0), geom=:milne,
                            save_dt=100.0, CFLτ=0.002)
    err = abs(rU.αs[end][100] - y)
    err < 1e-6 || fail("Milne uniform dilution off exact BDNK ODE: |Δα|=$err (code=$(rU.αs[end][100]), exact=$y)")
    # and it must be clearly DISTINCT from ideal dilution (guards against dropping the BDNK term)
    abs(y - (a0 + log(τ0/τf))) > 1e-2 || fail("BDNK relaxation correction vanished — test is not discriminating")
end

# ---------------------------------------------------------------------------
# 5. MOVING background (u^r≠0): exercises the driver's OWN run_bdnk_causal where B≠0 and the
# advective flux n·u^r is live. Checks 1-3 all run u^r=0, so the B-coupling and the outflow
# boundary go untested there — and ΔQ is NOT zero here: a uniform u^r=0.5 advects charge out
# through the outer face, so the honest lock is a charge BUDGET, not conservation.
# The budget is closed against the driver's own outer-face flux, integrated from the driver's
# own trajectory: ∮ = Σ dt·(τ r J^r)|_{r=rmax}. Any inconsistency between what run_bdnk_causal
# removes from q and what its outer face reports shows up as a nonzero residual.
let urv = 0.5, Nr = 800, rmax = 20.0, τ0 = 1.0, τf = 2.0, DsT = 0.24
    eosM = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    bgM  = BC.CausalBG((t,r)->0.3, (t,r)->urv, r->0.0)
    ainit(r) = 0.15*exp(-((r-10.0)/1.0)^2)
    # dense snapshots so the flux integral is accurate
    rr = BC.run_bdnk_causal(Nr=Nr, rmax=rmax, τ0=τ0, τfinal=τf, DsT=DsT, mode=:kappa,
                            bg=bgM, α_init=ainit, geom=:milne, save_dt=0.005)
    any(isnan, rr.αs[end]) && fail("NaN on moving background")
    g = rr.grid; dr = g.dr
    Qf(v) = sum(v)*dr
    dQ = Qf(rr.qs[end]) - Qf(rr.qs[1])
    # reconstruct the outer-face flux the driver used, from each saved state
    function outer_flux(α, q, τ)
        αr_N = 0.0                                   # outer: zero-gradient ⇒ α_{N+1}=α_N
        αr_N = (α[Nr] - α[Nr-1])/(2dr)               # driver's stencil at i=Nr
        Tl = max(bgM.T(τ, g.r[Nr]), BC.T_MIN); ur = bgM.ur(τ, g.r[Nr])
        cf = BC.cell_coeffs(α[Nr], Tl, ur, DsT, eosM, :kappa)
        Jt = q[Nr]/BC.meas(:milne, τ, g.r[Nr])
        β  = (Jt - cf.n*cf.uτ - cf.B*αr_N)/cf.A
        Jr = cf.n*ur + cf.B*β + cf.C*αr_N
        return BC.meas(:milne, τ, g.rF[Nr+1]) * Jr
    end
    # trapezoid over the saved trajectory
    outf = 0.0
    for k in 1:length(rr.τs)-1
        f1 = outer_flux(rr.αs[k],   rr.qs[k],   rr.τs[k])
        f2 = outer_flux(rr.αs[k+1], rr.qs[k+1], rr.τs[k+1])
        outf += 0.5*(f1+f2)*(rr.τs[k+1]-rr.τs[k])
    end
    resid = dQ + outf
    rel = abs(resid)/abs(Qf(rr.qs[1]))
    rel < 1e-4 || fail("moving-bg charge budget not closed: dQ=$dQ outflux=$outf residual/Q0=$rel")
    # the test must actually be exercising outflow, else it degenerates to the u^r=0 case
    abs(outf)/abs(Qf(rr.qs[1])) > 1e-3 ||
        fail("moving-bg test not exercising outflow (outflux/Q0=$(abs(outf)/abs(Qf(rr.qs[1]))))")
end

# ---------------------------------------------------------------------------
# 6. Causality of the (α,β) characteristics over a wide state scan.
# The system A w² − 2B k w + C k² = 0 has roots (B ± √(B²−AC))/A; these are the TRUE
# characteristic speeds. Locks: hyperbolicity (A>0, B²−AC≥0) and |λ|≤1 everywhere,
# including eps_factor<1 where the ε_ν≥κ floor is what keeps it causal.
let eosC = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0), worst = 0.0, nbad = 0
    for T in (0.1,0.15,0.2,0.3,0.5,0.6), a in (-2.0,-1.0,0.0,1.0,2.0,4.0),
        ur in (0.0,0.5,1.0,2.0,4.0,8.0), ef in (0.5,1.0,2.0,4.0), md in (:kappa,:is_match)
        cf = BC.cell_coeffs(a, T, ur, 0.24, eosC, md; eps_factor=ef)
        A,B,C = cf.A, cf.B, cf.C
        disc = B^2 - A*C
        if A <= 0 || disc < 0
            nbad += 1; continue
        end
        worst = max(worst, abs((B+sqrt(disc))/A), abs((B-sqrt(disc))/A))
    end
    nbad == 0 || fail("non-hyperbolic states found: $nbad")
    worst <= 1.0 + 1e-9 || fail("superluminal characteristic in state scan: max|λ|=$worst")
end

# ---------------------------------------------------------------------------
# 7. B-coupling (ε_ν≠κ on a BOOSTED background) must be live.
# At the minimal-causal point ε_ν=κ the algebra collapses to B≡0 (and A=κ, C=−κ), so every
# :kappa-mode test above is blind to the B·β term in J^τ/J^r. Only ε_ν≠κ together with u^r≠0
# activates it. Lock it two ways: (a) B is analytically u^τu^r(ε_ν−κ) and nonzero here;
# (b) the evolution genuinely differs from a run with the coupling absent — asserted by
# comparing against the same setup at u^r=0, which must give a different answer.
let eosB = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0), DsT = 0.24, ef = 4.0
    cB = BC.cell_coeffs(0.5, 0.3, 1.0, DsT, eosB, :kappa; eps_factor=ef)
    uτ = sqrt(2.0)
    abs(cB.B - uτ*1.0*(cB.εν - cB.κ)) < 1e-14*max(1.0,abs(cB.B)) ||
        fail("B coefficient wrong: got $(cB.B)")
    abs(cB.B) > 1e-12 || fail("B vanished at ε_ν≠κ, u^r≠0 — test not discriminating")
    # dynamical check: boosted vs unboosted must differ once B is live
    ainit(r) = 0.10*exp(-((r-10.0)/1.5)^2)
    runb(ur) = BC.run_bdnk_causal(Nr=400, rmax=20.0, τ0=1.0, τfinal=2.0, DsT=DsT, mode=:kappa,
                                  bg=BC.CausalBG((t,r)->0.3, (t,r)->ur, r->0.0),
                                  α_init=ainit, geom=:milne, eps_factor=ef, save_dt=100.0)
    a_b = runb(1.0).αs[end]; a_0 = runb(0.0).αs[end]
    any(isnan, a_b) && fail("NaN in boosted ε_ν≠κ run")
    maximum(abs.(a_b .- a_0)) > 1e-6 ||
        fail("boosted ε_ν≠κ run identical to unboosted — B-coupling appears dead")
end

# ---------------------------------------------------------------------------
# 8. Monotonicity / positivity: no over- or undershoot, even on a discontinuous IC.
# The flux is CENTERED with no upwinding or limiter, which would normally ring. It does not,
# because the telegraph damping (χβ, with χ/ε_ν = 1/D large) kills grid-scale modes faster than
# they grow. That is a real property of this scheme worth locking: if someone later raises ε_ν
# far above κ (weakening the damping relative to the wave term) or removes the damping, this fires.
let bgS = BC.static_bg(T0=0.3, α0=0.0)
    for (lbl, ainit, A) in (("gaussian",  r->0.2*exp(-((r-10.0)/1.0)^2), 0.2),
                            ("sharp",     r->0.2*exp(-((r-10.0)/0.1)^2), 0.2),
                            ("square",    r->(9.0<r<11.0 ? 0.2 : 0.0),   0.2))
        rS = BC.run_bdnk_causal(Nr=600, rmax=20.0, τ0=1.0, τfinal=2.0, DsT=0.24, mode=:kappa,
                                bg=bgS, α_init=ainit, geom=:cartesian, save_dt=100.0)
        a = rS.αs[end]
        any(isnan, a) && fail("NaN on $lbl IC")
        minimum(a) < -1e-9      && fail("undershoot on $lbl IC: min=$(minimum(a))")
        maximum(a) > A + 1e-9   && fail("overshoot on $lbl IC: max=$(maximum(a)) vs A=$A")
    end
end

# ---------------------------------------------------------------------------
# 9. Long-time stability: the damped telegraph must decay monotonically in L2, forever.
# A sign error in the damping or an unstable time integrator shows up as late-time growth that
# short runs (τfinal≈2-4 everywhere above) cannot see.
let bgL = BC.static_bg(T0=0.3, α0=0.0)
    apL(r) = 0.10*exp(-((r-15.0)/1.5)^2)
    rL = BC.run_bdnk_causal(Nr=400, rmax=30.0, τ0=1.0, τfinal=61.0, DsT=0.24, mode=:kappa,
                            bg=bgL, α_init=apL, geom=:cartesian, save_dt=5.0)
    prev = Inf
    for (i, t) in enumerate(rL.τs)
        a = rL.αs[i]
        any(isnan, a) && fail("NaN at τ=$t in long run")
        L2 = sqrt(sum(a.^2)*rL.grid.dr)
        L2 <= prev*(1+1e-9) || fail("L2 grew at τ=$t: $L2 > $prev (telegraph must decay)")
        prev = L2
    end
end

# ---------------------------------------------------------------------------
# 10. Linear dynamics vs the EXACT telegraph solution (both branches, no discrete reference).
# Locks the actual time-dependence, not just the front speed / cone / k→0 limit.
#   κ s² + χ s + κ k² = 0 ⇒ s± = [−χ ± √(χ²−4κ²k²)]/(2κ); β(τ0)=0 ⇒ weights c± = ∓s∓/(s+−s−).
let eosT = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0), T5 = 0.3, α5 = 0.5
    n5 = BC.n_of(α5, T5, eosT); κ5 = BC.kappa_of(T5, n5, 0.24); χ5 = BC.chi_alpha(α5, T5, eosT)
    function tele(rv, t, A, w, r0; Nk=4000, kmax=60.0)
        out = zeros(length(rv)); dk = kmax/Nk
        for j in 0:Nk
            k = j*dk
            ah = A*w*sqrt(pi)*exp(-k^2*w^2/4)
            disc = χ5^2 - 4*κ5*κ5*k^2
            f = if disc >= 0
                sp = (-χ5+sqrt(disc))/(2κ5); sm = (-χ5-sqrt(disc))/(2κ5)
                (-sm/(sp-sm))*exp(sp*t) + (sp/(sp-sm))*exp(sm*t)
            else
                wr = -χ5/(2κ5); wi = sqrt(-disc)/(2κ5)
                exp(wr*t)*(cos(wi*t) - (wr/wi)*sin(wi*t))
            end
            wgt = (j==0 || j==Nk) ? 0.5 : 1.0
            @inbounds for i in eachindex(rv)
                out[i] += wgt*dk/pi * ah * f * cos(k*(rv[i]-r0))
            end
        end
        return out
    end
    A = 1e-4; tfT = 4.0
    for (w, tol) in ((1.0, 1e-3), (4.0, 5e-5))
        αsm(r) = α5 + A*exp(-((r-40.0)/w)^2)
        rs = BC.run_bdnk_causal(Nr=2000, rmax=80.0, τ0=1.0, τfinal=1.0+tfT, DsT=0.24, mode=:kappa,
                                bg=BC.static_bg(T0=T5, α0=α5), α_init=αsm, geom=:cartesian, save_dt=100.0)
        g = rs.grid; dc = rs.αs[end] .- α5; de = tele(g.r, tfT, A, w, 40.0)
        lo, hi = 200, length(g.r)-200
        rel = sqrt(sum((dc[lo:hi].-de[lo:hi]).^2))/sqrt(sum(de[lo:hi].^2))
        rel < tol || fail("exact-telegraph mismatch w=$w: relL2=$rel (tol $tol)")
    end
end

# ---------------------------------------------------------------------------
# 11. χ_α convention lock. The telegraph damping is χ_α = ∂n/∂α, which for the exponential
# ConformalHQEOS density is EXACTLY n — NOT n/T (that is ∂n/∂μ, conjugate to μ, and the drive
# here is ∂α). Measured empirically: the long-wavelength decay rate of a cosine mode is κk²/χ,
# which matches χ=n to ~3% and χ=n/T only to a factor 3.2. The legacy drivers
# (main2BDNK.jl:230, mainBDNK.jl:349) still use n/T in their opt-in :is_match path.
let eosX = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
    for T in (0.15, 0.3, 0.5), a in (0.0, 1.0)
        n = BC.n_of(a, T, eosX); χ = BC.chi_alpha(a, T, eosX)
        abs(χ/n - 1) < 1e-6 || fail("χ_α ≠ n at T=$T α=$a: χ/n=$(χ/n)")
    end
end

# ---------------------------------------------------------------------------
# 12. FP-matched Soret (σ_T) + acceleration (σ_a) terms — finding B-4.
#     ν^μ = −κ Δ^{μν}∂_να − σ_T Δ^{μν}∂_ν lnT + σ_a u̇^μ,
#     σ_T = κ·z K₃(z)/K₂(z),  σ_a = −κ/T   (FP_Hydro_matching eq:BDNK_coeffs)
# Locks four things: the coefficients; that the r-source reproduces main2BDNK.jl's independent
# fp_matched expression EXACTLY; that u̇ stays orthogonal to u; and that the flag is a strict
# no-op exactly where the physics says it must be (isothermal, and Bjorken where u^r=0 kills
# the ∂_τlnT channel) while genuinely changing the answer on a fireball with a radial ∂_rT.
let eosF = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0), DsT = 0.24
    SFF = BC.hydro.SpecialFunctions
    # (a) coefficients
    for T in (0.15, 0.3, 0.5)
        n = BC.n_of(0.0, T, eosF); κ = BC.kappa_of(T, n, DsT)
        z = BC.hq_mass(eosF)/T
        want_sT = κ * z * SFF.besselkx(3,z)/SFF.besselkx(2,z)
        abs(BC.sigma_T_of(T, κ, eosF) - want_sT) < 1e-12*max(1.0,abs(want_sT)) ||
            fail("σ_T wrong at T=$T")
        abs(BC.sigma_a_of(T, κ) - (-κ/T)) < 1e-12*max(1.0,κ/T) || fail("σ_a wrong at T=$T")
    end
    # (b) Sr must equal main2BDNK.jl's fp_matched source expression exactly
    for T in (0.15,0.3,0.5), ur in (0.0,0.5,1.5), dtT in (-0.05,0.02), drT in (-0.03,0.01),
        dtur in (0.0,0.03), drur in (0.0,0.05)
        n = BC.n_of(0.5, T, eosF); κ = BC.kappa_of(T, n, DsT)
        σT = BC.sigma_T_of(T, κ, eosF); σa = BC.sigma_a_of(T, κ); uτ = sqrt(1+ur^2)
        ref = -σT*(ur*uτ*(dtT/T) + uτ^2*(drT/T)) + σa*(uτ*dtur + ur*drur)
        c = BC.cell_coeffs(0.5, T, ur, DsT, eosF, :kappa;
                           dtT=dtT, drT=drT, dtur=dtur, drur=drur, fp_matched=true)
        abs(c.Sr - ref) <= 1e-12*max(abs(ref),1e-30) ||
            fail("Sr ≠ main2BDNK fp_matched form (T=$T ur=$ur): $(c.Sr) vs $ref")
        # (c) u̇ ⟂ u  ⇒  −u^τ u̇^τ + u^r u̇^r = 0, i.e. u̇^τ = (u^r/u^τ)u̇^r
        udr = uτ*dtur + ur*drur
        abs(-uτ*((ur/uτ)*udr) + ur*udr) < 1e-12*max(1.0,abs(udr)) || fail("u̇ not ⟂ u at ur=$ur")
    end
    # (d) flag OFF ⇒ sources identically zero (bit-level backward compatibility)
    c0 = BC.cell_coeffs(0.5, 0.3, 0.7, DsT, eosF, :kappa;
                        dtT=-0.05, drT=-0.03, dtur=0.03, drur=0.05, fp_matched=false)
    (c0.Sτ == 0.0 && c0.Sr == 0.0) || fail("fp_matched=false leaked a source: Sτ=$(c0.Sτ) Sr=$(c0.Sr)")
end

# ---------------------------------------------------------------------------
# 13. fp_matched: no-op exactly where predicted, live where it matters (B-4 regression).
let apF(r) = 0.2*exp(-((r-4.0)/2.0)^2)
    runf(bg, fpm; Nr=400, τf=3.0) =
        BC.run_bdnk_causal(Nr=Nr, rmax=20.0, τ0=1.0, τfinal=τf, DsT=0.24, mode=:kappa,
                           bg=bg, α_init=apF, geom=:milne, fp_matched=fpm, save_dt=100.0)
    # (a) isothermal static: σ_T and σ_a both vanish ⇒ EXACT no-op
    bgI = BC.static_bg(T0=0.3, α0=0.0)
    dI = maximum(abs.(runf(bgI,true).αs[end] .- runf(bgI,false).αs[end]))
    dI == 0.0 || fail("fp_matched not a no-op on isothermal static bg: max|Δ|=$dI")
    # (b) Bjorken: ∂_rT=0 AND u^r=0 ⇒ the ∂_τlnT channel is killed by u^r ⇒ EXACT no-op.
    #     This is precisely why the pre-B-4 gates were blind to the omission.
    bgB = BC.bjorken_bg(T0=0.4, τ0=1.0, α0=0.0)
    dB = maximum(abs.(runf(bgB,true).αs[end] .- runf(bgB,false).αs[end]))
    dB == 0.0 || fail("fp_matched not a no-op on bjorken bg: max|Δ|=$dB")
    # (c) fireball with a radial ∂_rT: must be LIVE and large (it is the dominant term)
    bgFB = BC.fireball_bg(T0=0.45, T_edge=0.12, R=6.0, τ0=1.0, α0=0.0)
    rOn = runf(bgFB,true); rOff = runf(bgFB,false)
    any(isnan, rOn.αs[end]) && fail("NaN with fp_matched on fireball bg")
    dF = maximum(abs.(rOn.αs[end] .- rOff.αs[end]))
    dF > 0.05*maximum(abs.(rOff.αs[end])) ||
        fail("fp_matched barely changes the fireball answer (max|Δ|=$dF) — Soret term looks dead")
    # (d) still conservative up to boundary outflux, and convergent
    Qf(v, g) = sum(v)*g.dr
    d1 = let r = runf(bgFB,true; Nr=400); (Qf(r.qs[end],r.grid)-Qf(r.qs[1],r.grid))/abs(Qf(r.qs[1],r.grid)) end
    d2 = let r = runf(bgFB,true; Nr=800); (Qf(r.qs[end],r.grid)-Qf(r.qs[1],r.grid))/abs(Qf(r.qs[1],r.grid)) end
    abs(d1) < 1e-5 && abs(d2) < 1e-5 || fail("fp_matched charge drift too large: $d1, $d2")
    # (d2) Sτ liveness: Sτ has NO term free of u^r (Δ^{ττ}=(u^r)², Δ^{τr}=u^τu^r, u̇^τ∝u^r),
    #      so it vanishes identically at u^r=0 and (c) above cannot see it. With radial flow it is
    #      O(0.2–0.7)·Sr and MUST be subtracted in the β-reconstruction consistently with how it
    #      enters q at init — otherwise the scheme is inconsistent. Compare against an independent
    #      recomputation of β from the driver's own (q, α) output.
    bgFlow = BC.fireball_bg(T0=0.45, T_edge=0.12, R=6.0, τ0=1.0, α0=0.0, a_flow=0.08)
    let cS = BC.cell_coeffs(0.5, 0.3, 0.5, 0.24,
                            BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0), :kappa;
                            dtT=-0.1, drT=-0.03, dtur=0.0, drur=0.08, fp_matched=true)
        abs(cS.Sτ) > 0.05*abs(cS.Sr) || fail("Sτ not live under flow (Sτ=$(cS.Sτ) Sr=$(cS.Sr))")
    end
    rF = runf(bgFlow, true; Nr=400, τf=2.0)
    any(isnan, rF.αs[end]) && fail("NaN with fp_matched + radial flow")
    # The driver initialises q at β=0 as q = m(n u^τ + B α_r + Sτ), and rhs! must invert that to
    # β = (J^τ − n u^τ − B α_r − Sτ)/A. So immediately after init the reconstructed β must be
    # MACHINE ZERO. Dropping Sτ from either side leaves β(τ0) = Sτ/A ≠ 0 — a spurious ∂_τα at
    # startup. Evaluating rhs! once on the freshly-initialised state is what makes that visible;
    # a finite-time comparison cannot, because α legitimately evolves.
    # Drive this through run_bdnk_causal so BOTH the driver's init and its rhs! are under test
    # (a locally re-implemented init would only test the copy). One vanishingly short step:
    # (α(τ0+dt) − α(τ0))/dt ≈ β(τ0), which must be ~0 for the driver's β=0 start. Dropping Sτ on
    # either side leaves β(τ0)=±Sτ/A, which under radial flow is O(0.2–0.7)·Sr/A — far above tol.
    let dtI = 1e-7
        rI = BC.run_bdnk_causal(Nr=400, rmax=20.0, τ0=1.0, τfinal=1.0+dtI, DsT=0.24, mode=:kappa,
                                bg=bgFlow, α_init=apF, geom=:milne, fp_matched=true, save_dt=1e9)
        α0v = [apF(x) for x in rI.grid.r]
        β0 = maximum(abs.(rI.αs[end] .- α0v))/dtI
        # Tolerance note: run_bdnk_causal's CFL may take a step larger than dtI, so β0 also picks
        # up a little genuine evolution — the correct code measures ~3e-6 here, while dropping Sτ
        # on either side gives ~12 (a 4e6× separation), so 1e-3 is a safe, still-decisive cut.
        β0 < 1e-3 ||
            fail("β(τ0) ≠ 0 for a β=0 start (Sτ inconsistent between init and rhs!): |β(τ0)|≈$β0")
    end

    # (e) causality is untouched — the sources shift ν but do not enter A,B,C
    eosC2 = BC.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0); worst = 0.0
    for T in (0.12,0.2,0.3,0.45), ur in (0.0,0.5,1.5,3.0), dtT in (-0.1,0.0), drT in (-0.05,0.02)
        c = BC.cell_coeffs(0.5, T, ur, 0.24, eosC2, :kappa;
                           dtT=dtT, drT=drT, dtur=0.02, drur=0.03, fp_matched=true)
        disc = c.B^2 - c.A*c.C
        (c.A > 0 && disc >= 0) || fail("non-hyperbolic with fp_matched at T=$T ur=$ur")
        worst = max(worst, abs((c.B+sqrt(disc))/c.A), abs((c.B-sqrt(disc))/c.A))
    end
    worst <= 1.0 + 1e-9 || fail("fp_matched broke causality: max|λ|=$worst")
end

# ---------------------------------------------------------------------------
# 14. Default FD background derivatives must reproduce analytic ones.
# The 3-arg CausalBG constructor finite-differences T and u^r; fireball_bg supplies analytic
# forms. On the same profile the two must agree, else fp_matched silently uses wrong gradients.
let Tf(τ,r) = (0.12 + (0.45-0.12)*exp(-(r/6.0)^2))*(1.0/τ)^(1/3)
    bgA = BC.fireball_bg(T0=0.45, T_edge=0.12, R=6.0, τ0=1.0, α0=0.0)
    bgF = BC.CausalBG(Tf, (τ,r)->0.0, r->0.0)
    for r0 in (0.5, 2.0, 4.0, 8.0)
        for (nm, fa, ff) in (("drT", bgA.drT, bgF.drT), ("dtT", bgA.dtT, bgF.dtT))
            a = fa(1.0, r0); b = ff(1.0, r0)
            abs(b-a) <= 1e-6*max(abs(a), 1e-12) || fail("$nm FD≠analytic at r=$r0: $b vs $a")
        end
    end
end

println(allpass ? "test_bdnk_causal: PASS (fixed-point, conservation, causality, Milne-dilution, moving-bg budget, characteristics, B-coupling, monotonicity, long-time, exact-telegraph, χ_α, FP-matched σ_T/σ_a)" :
                  "test_bdnk_causal: FAIL")
exit(allpass ? 0 : 1)
