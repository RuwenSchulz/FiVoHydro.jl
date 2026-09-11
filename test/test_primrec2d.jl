# ==============================================================================
# test/test_primrec2d.jl
#
# Gate: the 2+1D primitive recovery (src2d/primrec2d.jl) inverts the
# conserved-variable map. prim -> cons -> prim must return the original state.
#
# This is the piece flagged in TWOD_PROGRAM.md §4 as the schedule risk: in 1-D the
# stored shear makes π^{τr} collinear with u, so the flow direction is fixed by
# S_r alone; in 2-D with π^{xy} ≠ 0 the momentum is tilted away from the flow and
# the direction is a genuine unknown. The `shear tilts the flow` testset below
# MEASURES that tilt, so the hard case is provably exercised rather than assumed.
#
# Run: julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_primrec2d.jl
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
include(joinpath(_ROOT, "src", "primitives.jl"))   # dimension-agnostic transport models
include(joinpath(_ROOT, "src2d", "shear2d.jl"))
include(joinpath(_ROOT, "src2d", "state_layout2d.jl"))
include(joinpath(_ROOT, "src2d", "terms2d.jl"))       # Terms2D, carried by the model
include(joinpath(_ROOT, "src2d", "primitives2d.jl"))
include(joinpath(_ROOT, "src2d", "bessel2d.jl"))      # fast K₂, needed by primrec2d.jl
include(joinpath(_ROOT, "src2d", "primrec2d.jl"))

const RNG = MersenneTwister(20260901)

"""Draw a physically sensible cell state. `diss` scales the dissipative fields
relative to the pressure; 0 gives ideal fluid."""
function random_state(rng, eos; umax = 1.5, diss = 0.0, Tlo = 0.12, Thi = 0.55)
    T  = Tlo + (Thi - Tlo)*rand(rng)
    φ  = 4.0*(2rand(rng) - 1)
    μ  = hq_mass(eos) + T*φ
    ux = umax*(2rand(rng) - 1)
    uy = umax*(2rand(rng) - 1)
    uτ = sqrt(1 + ux^2 + uy^2)

    P, n, _ = eos_Pne(T, μ, eos)

    Pi   = diss * P * (2rand(rng) - 1)
    pixx = diss * P * (2rand(rng) - 1)
    pixy = diss * P * (2rand(rng) - 1)
    piyy = diss * P * (2rand(rng) - 1)
    pieta, _ = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, 0.0)

    nux = diss * n * (2rand(rng) - 1)
    nuy = diss * n * (2rand(rng) - 1)

    return (T=T, μ=μ, ux=ux, uy=uy, nux=nux, nuy=nuy, Pi=Pi,
            pixx=pixx, pixy=pixy, piyy=piyy, pieta=pieta)
end

function roundtrip(eos, L, w, st, τ)
    Umat = zeros(nvars(L), 1)
    ok1, _ = prim_to_cons_2d!(Umat, 1, st.T, st.μ, st.ux, st.uy,
                              st.nux, st.nuy, st.Pi,
                              st.pixx, st.pixy, st.piyy, st.pieta, τ, eos, L)
    ok1 || return (false, st.T, st.ux, st.uy, st.μ, Umat)

    D  = Umat[L.iDtau,1] / τ
    Sx = Umat[L.iSx,1]; Sy = Umat[L.iSy,1]; E = Umat[L.iE,1]

    T2, μ2, ux2, uy2, _, _, _, ok2 = cons_to_prim_2d!(
        w, D, Sx, Sy, E, st.nux, st.nuy, st.Pi,
        st.pixx, st.pixy, st.piyy, st.pieta, τ, eos)

    return (ok2, T2, ux2, uy2, μ2, Umat)
end

function sweep(eos, name; diss, umax, ntrial = 1500, tol = 1e-9)
    L = make_layout2d()
    w = PrimRecWork2D()
    worstT = 0.0; worstU = 0.0; worstMu = 0.0; nfail = 0
    for _ in 1:ntrial
        τ  = 0.4 + 12*rand(RNG)
        st = random_state(RNG, eos; umax = umax, diss = diss)
        ok, T2, ux2, uy2, μ2, _ = roundtrip(eos, L, w, st, τ)
        if !ok
            nfail += 1
            continue
        end
        worstT  = max(worstT,  abs(T2 - st.T)/st.T)
        worstU  = max(worstU,  max(abs(ux2 - st.ux), abs(uy2 - st.uy))/max(1.0, abs(st.ux), abs(st.uy)))
        worstMu = max(worstMu, abs(μ2 - st.μ)/max(abs(st.μ), 1e-3))
    end
    @printf("  %-34s  T %.2e   u %.2e   mu %.2e   fail %d/%d\n",
            name, worstT, worstU, worstMu, nfail, ntrial)
    @test nfail == 0
    @test worstT < tol
    @test worstU < tol
    @test worstMu < tol
    return worstT, worstU, worstMu
end

@testset "2+1D primitive recovery" begin

    @testset "round trip — ConformalHQEOS" begin
        eos = ConformalHQEOS()
        sweep(eos, "ideal (no dissipatives)";        diss = 0.0,  umax = 1.5)
        sweep(eos, "viscous + diffusive (10% of P)"; diss = 0.10, umax = 1.5)
        sweep(eos, "viscous + diffusive, fast flow"; diss = 0.10, umax = 4.0)
    end

    @testset "round trip — LatticeHRGEOS (production)" begin
        eos = LatticeHRGEOS()
        sweep(eos, "ideal (no dissipatives)";        diss = 0.0,  umax = 1.5)
        sweep(eos, "viscous + diffusive (10% of P)"; diss = 0.10, umax = 1.5)
        sweep(eos, "viscous + diffusive, fast flow"; diss = 0.10, umax = 4.0)
    end

    @testset "shear tilts the flow away from the momentum (the 2-D-only hard case)" begin
        # If S^i stayed parallel to u^i the recovery would be a 1-D problem in
        # disguise and this suite would prove nothing. Measure the angle between
        # (S^x,S^y) and (u^x,u^y): with π^{xy} ≠ 0 it must be resolvably non-zero.
        eos = LatticeHRGEOS()
        L = make_layout2d()
        maxtilt = 0.0
        for _ in 1:2000
            τ  = 0.4 + 12*rand(RNG)
            st = random_state(RNG, eos; umax = 1.5, diss = 0.10)
            Umat = zeros(nvars(L), 1)
            ok, _ = prim_to_cons_2d!(Umat, 1, st.T, st.μ, st.ux, st.uy,
                                     st.nux, st.nuy, st.Pi,
                                     st.pixx, st.pixy, st.piyy, st.pieta, τ, eos, L)
            ok || continue
            Sx = Umat[L.iSx,1]; Sy = Umat[L.iSy,1]
            ns = hypot(Sx, Sy); nu = hypot(st.ux, st.uy)
            (ns < 1e-12 || nu < 1e-12) && continue
            cosang = clamp((Sx*st.ux + Sy*st.uy)/(ns*nu), -1.0, 1.0)
            maxtilt = max(maxtilt, acos(cosang))
        end
        @printf("  max |angle(S, u)| = %.4f rad (%.2f deg)\n", maxtilt, rad2deg(maxtilt))
        @test maxtilt > 1e-3        # the hard case is genuinely exercised
    end

    @testset "1-D limit: u^y = 0, π^{xy} = 0 keeps the flow on the x-axis" begin
        # A state with no y-structure must recover u^y = 0 exactly, not merely
        # to tolerance — otherwise the 2-D solver would break the symmetry of a
        # symmetric IC through the recovery alone, poisoning gate G4.
        eos = LatticeHRGEOS()
        L = make_layout2d()
        w = PrimRecWork2D()
        worst_uy = 0.0
        for _ in 1:1500
            τ  = 0.4 + 12*rand(RNG)
            st0 = random_state(RNG, eos; umax = 1.5, diss = 0.10)
            uτ = sqrt(1 + st0.ux^2)
            pieta, _ = project_shear_traceless_2d(st0.ux, 0.0, uτ, st0.pixx, 0.0, st0.piyy, 0.0)
            st = (T=st0.T, μ=st0.μ, ux=st0.ux, uy=0.0, nux=st0.nux, nuy=0.0,
                  Pi=st0.Pi, pixx=st0.pixx, pixy=0.0, piyy=st0.piyy, pieta=pieta)
            ok, _, _, uy2, _, _ = roundtrip(eos, L, w, st, τ)
            ok || continue
            worst_uy = max(worst_uy, abs(uy2))
        end
        @printf("  max |recovered u^y| on a y-symmetric state = %.3e\n", worst_uy)
        @test worst_uy < 1e-12
    end

    @testset "vacuum and degenerate cells are handled, not crashed" begin
        eos = LatticeHRGEOS()
        w = PrimRecWork2D()
        # exact vacuum
        T, μ, ux, uy, n, e, P, ok = cons_to_prim_2d!(w, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                     1.0, eos)
        @test !ok
        @test w.last_reason == PRR_VACUUM
        @test isfinite(T) && isfinite(ux) && isfinite(uy)
        # zero flow, finite energy: must converge with u = 0 exactly
        L = make_layout2d()
        Umat = zeros(nvars(L), 1)
        okf, _ = prim_to_cons_2d!(Umat, 1, 0.30, hq_mass(eos), 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, eos, L)
        @test okf
        T2, μ2, ux2, uy2, _, _, _, ok2 = cons_to_prim_2d!(
            w, Umat[L.iDtau,1], Umat[L.iSx,1], Umat[L.iSy,1], Umat[L.iE,1],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, eos)
        @test ok2
        @printf("  zero-flow cell: T err %.3e, |u| = %.3e\n", abs(T2-0.30)/0.30, hypot(ux2,uy2))
        @test abs(T2 - 0.30)/0.30 < 1e-10
        @test hypot(ux2, uy2) < 1e-12
    end

    # ── the 2-D equation of state (2026-09-11) ───────────────────────────────────
    # `eos_Pne_2d` is a copy of `eos_Pne`'s arithmetic (src/eos.jl, shared with the
    # 1-D production solver, is not touched) with a fast K₂ and a T-keyed memo. This
    # is what keeps the copy honest: if src/eos.jl changes, these fail.
    @testset "2-D EOS: fast K₂, the copy of eos_Pne, and the memo" begin
        eos = LatticeHRGEOS()
        # (a) the vendored Bessel against SpecialFunctions (Amos)
        worst = 0.0
        for x in exp.(range(log(0.1), log(BESSELKX_ASYM_X*0.999); length = 20_000))
            worst = max(worst, abs(safe_besselk2x_2d(x)/SpecialFunctions.besselkx(2, x) - 1))
        end
        @printf("  K₂: max rel diff vs SpecialFunctions over x ∈ [0.1, 1e4] = %.2e\n", worst)
        @test worst < 5e-15
        # the asymptotic regime (vacuum, floors) is the SAME code path, bit for bit
        @test all(safe_besselk2x_2d(x) == safe_besselkx(2, x) for x in (1e4, 3.7e5, 1.5e20))
        @test safe_besselk2x_2d(0.0) == 0.0 && safe_besselk2x_2d(NaN) == 0.0

        # (b) eos_Pne_2d vs eos_Pne: P and e bit for bit, n to round-off
        wn = 0.0; nbitP = 0; nbitE = 0; ntot = 0
        for T in vcat(exp.(range(log(1e-3), log(5.0); length = 400)), [T_MIN, 1e-12, 0.1565]),
            α in (-40.0, -4.2, 0.0, 3.0)
            P1, n1, e1 = eos_Pne(T, hq_mass(eos) + T*α, eos)
            P2, n2, e2 = eos_Pne_2d(T, hq_mass(eos) + T*α, eos)
            ntot += 1
            nbitP += (P1 === P2); nbitE += (e1 === e2)
            n1 > 0 && (wn = max(wn, abs(n2/n1 - 1)))
            n1 == 0 && @test n2 == 0
        end
        @printf("  EOS: P bitwise %d/%d, e bitwise %d/%d, n max rel diff %.2e\n",
                nbitP, ntot, nbitE, ntot, wn)
        @test nbitP == ntot && nbitE == ntot
        @test wn < 1e-14
        @test all(eos_Pe_2d(T, 1.0, eos) === (eos_Pne(T, 1.0, eos)[1], eos_Pne(T, 1.0, eos)[3])
                  for T in (T_MIN, 0.01, 0.2, 0.6, 3.0))

        # (c) the memo returns exactly what the uncached call returns, hits and misses
        c = EOSCache2D(); nsame = 0; nq = 0
        rng = MersenneTwister(5)
        Ts = [0.12 + 0.5rand(rng) for _ in 1:7]
        for rep in 1:3, T in Ts, α in (-5.0, -1.0, 2.0)      # 7 distinct T > 4 slots: evictions too
            nq += 1
            nsame += (eos_Pne_2d(T, hq_mass(eos) + T*α, eos, c) ===
                      eos_Pne_2d(T, hq_mass(eos) + T*α, eos))
        end
        @test nsame == nq
        reset_eos_cache_2d!(c); @test all(isnan, c.T)
        # a different EOS type never touches the memo
        ceos = ConformalHQEOS()
        @test eos_Pne_2d(0.3, 1.2, ceos, c) === eos_Pne(0.3, 1.2, ceos)
        @printf("  memo: %d/%d cached evaluations bit-identical to uncached\n", nsame, nq)
    end
end
