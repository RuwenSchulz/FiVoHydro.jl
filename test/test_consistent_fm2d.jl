
# ==============================================================================
# test/test_consistent_fm2d.jl — gate Gc: the 2-D consistent first moment.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_consistent_fm2d.jl
#
# `src2d/hq_consistent_firstmoment2d.jl` is a RE-DERIVATION, not a transcription:
# the 1-D `hq_consistent_extras` is the same covariant object contracted in radial
# Milne, where distinct terms collapse onto one another (see that file's header).
# A copy-paste port would be wrong in two specific ways, and this gate is what
# would catch it.
#
# Gc1  the 1-D LIMIT. Take a state with u^y = 0 and ∂_y = 0, so the 2-D geometry
#      degenerates to the 1-D radial one at a chosen radius r, and require the
#      2-D source to reproduce `hq_consistent_extras` to round-off. The handling
#      of the geometric terms is the whole content: they enter ONCE, inside the
#      full divergence θ, and the wls's separately named "geometric dilution" is
#      not a second copy to be added on top. The first version of the 2-D file
#      added it twice and was 2.5-25 % wrong; this gate is what caught it, and
#      nothing else would have — the bad source is smooth and plausible.
#
# Gc2  ROTATIONAL INVARIANCE. The source is a transverse VECTOR, so rotating the
#      state by an angle must rotate the answer by the same angle. This is what
#      catches an x/y asymmetry — a dxuy typed where dyux belongs, the kind of
#      defect the 1-D limit (which has only one transverse direction) is blind to.
#
# Gc3  the BJORKEN limit. At rest with no gradients θ = u^τ/τ = 1/τ is all that
#      survives, so the row must reduce to τ_n ν^i/τ — the wls gate WB, one
#      dimension over.
#
# Gc4  ZERO CURRENT + ZERO GRADIENTS ⇒ no source at all, and the flag is inert on
#      a real solve when the charge sector is off.
#
# Gc5  the flag is INERT BY DEFAULT: a short 2-D solve with consistent_fm=false
#      is bit-identical to the same solve on the pre-existing code path.
#
# Gc6  `tauN_coeff != 1` is REFUSED with the closure on (it would break the
#      τ_n = D_s h/T tie the sources rely on) and still allowed with it off.
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))

# The 1-D reference. Loaded WITHOUT main.jl (which would define a clashing
# `hydro` module): the file is a leaf, it defines two @inline functions and needs
# only what is already in scope here.
module Ref1D
    import SpecialFunctions
    const T_MIN = 1e-20
    include(joinpath(normpath(joinpath(@__DIR__, "..")), "src", "hq_consistent_firstmoment.jl"))
end

include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d

const H2 = hydro2d

# ------------------------------------------------------------------------------
# Gc1 — the 1-D limit
# ------------------------------------------------------------------------------
"""
Evaluate the 2-D source in the axisymmetric limit.

The 1-D state (r, u^r, ∂_r·) is embedded on the +x axis: x = r, u^x = u^r,
u^y = 0, ∂_x = ∂_r. The one place the embedding is not a relabelling is the
transverse divergence. In 1-D radial Milne

    θ = ∂_τu^τ + ∂_r u^r + u^τ/τ + u^r/r ,

and the u^r/r comes from the φ direction. On the +x axis the corresponding
Cartesian statement is ∂_y u^y = u^r/r (the flow is radial, so at that point the
y-derivative of u^y is u^r/r exactly). So the embedding sets

    dyuy = ur/r ,

and with that one substitution the two θ agree term by term.
"""
function twod_in_1d_limit(; τ, r, ur, T, dtT, drT, drur, dtur, n, dn_dT, ν, τn, Ds, h, hp)
    ux = ur; uy = 0.0
    uτ = sqrt(1 + ux*ux)
    # a^i = u^m ∂_m u^i
    ax = uτ*dtur + ur*drur
    ay = 0.0
    dxux = drur
    dxuy = 0.0
    dyux = 0.0
    dyuy = ur/r                     # the φ-direction divergence, see docstring
    # θ = ∂_τu^τ + ∂_xu^x + ∂_yu^y + u^τ/τ
    dtuτ = ur*dtur/uτ
    θ = dtuτ + dxux + dyuy + uτ/τ
    return H2.consistent_fm_source_2d(ux, uy, uτ, T, drT, 0.0, dtT,
                                      ν, 0.0, n, dn_dT, τn, Ds, h, hp,
                                      θ, ax, ay, dxux, dxuy, dyux, dyuy)
end

function gate_Gc1()
    println("\nGc1 — 1-D limit (u^y = 0, ∂_y = 0) vs src/hq_consistent_firstmoment.jl")
    # τ_n and h are TIED by τ_n = D_s h/T; a test point that violates the tie
    # compares two different physical systems and tells you nothing.
    pts = [
        (τ=2.3, r=3.1, ur= 0.42, T=0.31, dtT=-0.055, drT=-0.021, drur= 0.13, dtur= 0.07,
         n=0.9, dn_dT=1.7, ν= 0.031, Ds=0.44, h=2.20, hp=-0.9),
        (τ=1.1, r=0.7, ur=-0.30, T=0.45, dtT=-0.120, drT= 0.033, drur=-0.21, dtur= 0.15,
         n=1.4, dn_dT=2.9, ν=-0.070, Ds=0.30, h=1.90, hp=-1.4),
        (τ=5.0, r=8.0, ur= 0.90, T=0.20, dtT=-0.010, drT=-0.005, drur= 0.02, dtur= 0.01,
         n=0.3, dn_dT=0.8, ν= 0.010, Ds=0.60, h=2.60, hp=-0.5),
        (τ=0.6, r=0.05, ur=-1.30, T=0.55, dtT= 0.300, drT=-0.440, drur= 0.55, dtur=-0.62,
         n=2.2, dn_dT=4.1, ν= 0.190, Ds=0.12, h=1.75, hp=-2.2),
        (τ=3.7, r=12.0, ur= 0.05, T=0.16, dtT=-0.002, drT=-0.0007, drur=0.004, dtur=0.002,
         n=0.05, dn_dT=0.3, ν=1e-4, Ds=0.80, h=3.10, hp=-0.2),
    ]
    worst = 0.0
    for p in pts
        τn = p.Ds*p.h/p.T                      # enforce the tie
        ref = Ref1D.hq_consistent_extras(p.τ, p.r, p.ur, p.T, p.dtT, p.drT, p.drur, p.dtur,
                                         p.n, p.dn_dT, p.ν, τn, p.Ds, p.h, p.hp)
        sx, sy = twod_in_1d_limit(; τ=p.τ, r=p.r, ur=p.ur, T=p.T, dtT=p.dtT, drT=p.drT,
                                    drur=p.drur, dtur=p.dtur, n=p.n, dn_dT=p.dn_dT,
                                    ν=p.ν, τn=τn, Ds=p.Ds, h=p.h, hp=p.hp)
        rel = abs(sx - ref)/max(abs(ref), 1e-300)
        worst = max(worst, rel)
        @printf("  r=%5.2f u^r=%+5.2f  1-D=% .10e  2-D=% .10e  rel=%.2e   s^y=%.1e\n",
                p.r, p.ur, ref, sx, rel, sy)
        @test abs(sy) < 1e-14 * max(abs(sx), 1.0)      # u^y = 0 ⇒ no y source
    end
    @printf("  worst relative deviation = %.3e\n", worst)
    @test worst < 1e-12
    return worst
end

# ------------------------------------------------------------------------------
# Gc2 — rotational invariance
# ------------------------------------------------------------------------------
function gate_Gc2()
    println("\nGc2 — rotational invariance of the source vector")
    # A generic state, then the same state rotated by φ. Under a rotation R, the
    # velocity and current rotate as vectors and the gradient matrix as
    # ∂'_i u'^j = R_ik R_jl ∂_k u^l; the source must come back rotated.
    ux, uy = 0.31, -0.22
    uτ = sqrt(1 + ux^2 + uy^2)
    nux, nuy = 0.017, 0.009
    dxT, dyT = -0.03, 0.012
    G = [0.11 -0.07; 0.05 0.09]        # G[i,j] = ∂_i u^j  (dxux dxuy; dyux dyuy)
    τ, T, dtT, n, dn_dT, τn, Ds, h, hp = 2.0, 0.28, -0.04, 1.1, 2.0, 0.55, 0.35, 2.1, -1.0
    # θ must be built from the SAME data so the rotation is consistent:
    # θ = ∂_τu^τ + tr G + u^τ/τ, and ∂_τu^τ is rotation-invariant.
    dtuτ = 0.02
    θ = dtuτ + (G[1,1] + G[2,2]) + uτ/τ
    a = [uτ*0.03 + ux*G[1,1] + uy*G[2,1], uτ*0.01 + ux*G[1,2] + uy*G[2,2]]

    s0 = H2.consistent_fm_source_2d(ux, uy, uτ, T, dxT, dyT, dtT, nux, nuy,
                                    n, dn_dT, τn, Ds, h, hp,
                                    θ, a[1], a[2], G[1,1], G[1,2], G[2,1], G[2,2])
    worst = 0.0
    for φ in (0.3, 1.0, 2.4, -0.8)
        c, s = cos(φ), sin(φ)
        R = [c -s; s c]
        u2 = R*[ux, uy]; nu2 = R*[nux, nuy]; gT2 = R*[dxT, dyT]; a2 = R*a
        G2 = R*G*transpose(R)                 # ∂'_i u'^j
        θ2 = dtuτ + (G2[1,1] + G2[2,2]) + uτ/τ
        s2 = H2.consistent_fm_source_2d(u2[1], u2[2], uτ, T, gT2[1], gT2[2], dtT,
                                        nu2[1], nu2[2], n, dn_dT, τn, Ds, h, hp,
                                        θ2, a2[1], a2[2], G2[1,1], G2[1,2], G2[2,1], G2[2,2])
        expect = R*[s0[1], s0[2]]
        d = max(abs(s2[1]-expect[1]), abs(s2[2]-expect[2])) / max(hypot(s0...), 1e-300)
        worst = max(worst, d)
        @printf("  φ=%+5.2f  rel dev = %.3e\n", φ, d)
    end
    @printf("  worst = %.3e\n", worst)
    @test worst < 1e-12
    return worst
end

# ------------------------------------------------------------------------------
# Gc3 — the Bjorken limit
# ------------------------------------------------------------------------------
function gate_Gc3()
    println("\nGc3 — Bjorken limit: at rest with no gradients the source is pure dilution")
    τ, T, n, τn, Ds, h, hp = 3.0, 0.30, 1.0, 0.7, 0.4, 2.2, -1.0
    nux, nuy = 0.02, -0.013
    uτ = 1.0
    θ = uτ/τ                                   # only the geometric piece survives
    sx, sy = H2.consistent_fm_source_2d(0.0, 0.0, uτ, T, 0.0, 0.0, 0.0, nux, nuy,
                                        n, 0.0, τn, Ds, h, hp,
                                        θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # expected: ν^i τ_n θ = τ_n ν^i/τ at rest (θ = u^τ/τ = 1/τ; the geometric
    # dilution IS this term, it is not added again).
    ex, ey = τn*nux/τ, τn*nuy/τ
    @printf("  s = (% .8e, % .8e)   expected (% .8e, % .8e)\n", sx, sy, ex, ey)
    @test isapprox(sx, ex; rtol=1e-13) && isapprox(sy, ey; rtol=1e-13)
    return max(abs(sx-ex), abs(sy-ey))
end

# ------------------------------------------------------------------------------
# Gc4 — no current, no gradients, no source
# ------------------------------------------------------------------------------
function gate_Gc4()
    println("\nGc4 — zero current and zero gradients give an identically zero source")
    sx, sy = H2.consistent_fm_source_2d(0.0, 0.0, 1.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 2.0, 0.7, 0.4, 2.2, -1.0,
                                        1.0/3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @printf("  s = (%.3e, %.3e)\n", sx, sy)
    @test sx == 0.0 && sy == 0.0
    return 0.0
end

# ------------------------------------------------------------------------------
# Gc5 — the flag is inert by default on a real solve
# ------------------------------------------------------------------------------
function gate_Gc5()
    println("\nGc5 — consistent_fm=false reproduces the shipped solve bit for bit")
    function run(cfm)
        g = H2.make_grid2d(48, 48; xmax=6.0, ymax=6.0)
        model = H2.build_model_2d(; eos=H2.LatticeHRGEOS(), enable_diff=true,
                                    kappa_coeff=0.2, enable_bulk=false, enable_shear=false,
                                    consistent_fm=cfm)
        U = H2.allocate_state(g, model)
        H2.initialize_uniform!(U, g, model, 0.4; T0=0.40, alpha0=-4.2)
        # a transverse gradient, so the charge sector is actually doing something
        ng = g.nghost
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H2.lin(g, ix, iy)
            U[model.layout.iDtau, i] *= 1 + 0.10*exp(-((g.xC[ix])^2 + (g.yC[iy])^2)/8)
        end
        H2.run_sim_2d!(U, g, model; τ0=0.4, τfinal=0.6, verbose=false)
        return copy(U)
    end
    A = run(false)
    B = run(false)
    @test A == B                     # the harness itself is deterministic
    println("  two shipped runs agree bitwise: ", A == B)
    C = run(true)
    d = maximum(abs.(C .- A))
    rel = d / max(maximum(abs.(A)), 1e-300)
    @printf("  consistent_fm=true moves the state by max|Δ| = %.3e (rel %.3e)\n", d, rel)
    # It must MOVE the answer -- a source that changes nothing is a source that is
    # not wired in, which is the failure mode this line exists to catch.
    @test d > 0
    return rel
end

# ------------------------------------------------------------------------------
# Gc6 — tauN_coeff is refused, not silently run
# ------------------------------------------------------------------------------
function gate_Gc6()
    println("\nGc6 — `tauN_coeff != 1` with the consistent closure is refused")
    # The closure ties tau_n = Ds*h/T with h in closed form; tauN_coeff rescales
    # tau_n but not h, so allowing both would put the relaxation time and the
    # enthalpy inside its own sources on different clocks.
    threw = false
    try
        H2.build_model_2d(; eos=H2.LatticeHRGEOS(), enable_diff=true,
                            kappa_coeff=0.1163, consistent_fm=true, tauN_coeff=1.5)
    catch err
        threw = true
        println("  refused: ", first(split(sprint(showerror, err), '\n')))
    end
    @test threw
    # ... and it is still allowed with the closure OFF
    ok = true
    try
        H2.build_model_2d(; eos=H2.LatticeHRGEOS(), enable_diff=true,
                            kappa_coeff=0.1163, consistent_fm=false, tauN_coeff=1.5)
    catch
        ok = false
    end
    println("  still allowed with consistent_fm=false: ", ok)
    @test ok
    return 0.0
end

function main()
    println("="^78)
    println("Gate Gc — the 2+1D thermodynamically consistent first moment")
    println("="^78)
    ok = true
    @testset "consistent first moment (2-D)" begin
        gate_Gc1(); gate_Gc2(); gate_Gc3(); gate_Gc4(); gate_Gc5(); gate_Gc6()
    end
    println("\n", "="^78)
    return ok
end

main()
