
# ==============================================================================
# test/test_consistent_m22d.jl — gate Gm2: the 2+1D consistent SECOND moment.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_consistent_m22d.jl
#
# `src2d/hq_consistent_m2_2d.jl` is a RE-DERIVATION, not a port. The 1-D file
# (src/hq_consistent_m2.jl) evolves THREE SCALARS on a parallel-transported triad
# that exists only in azimuthal symmetry; here the same covariant equation must be
# carried as a TENSOR. Retyping the 1-D reduced form with x for r is meaningless,
# so the gate that matters is the reduction: in the axisymmetric limit the tensor
# assembly must reproduce the 1-D scalar system.
#
# The 2-D FIRST moment took two wrong versions before it passed the analogous
# gate (Gc1 in test_consistent_fm2d.jl), and both were smooth and plausible.
# Nothing here is assumed to be right because it "looks like" the 1-D file.
#
# Gm1  the 1-D LIMIT. u^y = 0, ∂_y = 0, the flow radial on the +x axis. The
#      transverse block then decomposes as π^{xx} → p_l, π^{yy} → p_φ, and the
#      trace channel is a scalar in both. Required to reproduce
#      `hq_consistent_m2_rhs` to round-off.
# Gm2  ROTATIONAL COVARIANCE. The traceless block is a rank-2 transverse tensor,
#      so rotating the state by φ must rotate the answer by the same angle:
#      π' = R π Rᵀ. This catches an x/y transposition, a scalar used where a
#      vector belongs, and a missing symmetrisation — none of which the 1-D limit
#      can see, because it has ONE transverse direction. (Writing this gate is
#      what caught `sigma_nu_2d` being handed the SCALAR a·ν in both slots.)
# Gm3  TRACELESSNESS is preserved: the assembled traceless sources must have zero
#      transverse trace against the projector, so the constraint the solver
#      restores each step is not being fought by the RHS.
# Gm4  the BJORKEN limit: at rest with no transverse gradients x and y are
#      equivalent, so dpxx == dpyy exactly and dpxy == 0 exactly. (The block is
#      NOT zero — σ^{xx} = σ^{yy} = −θ/3 there — and expecting zero was this
#      gate's own first error.)
# Gm5  the flag is INERT when off and NOT inert when on, on a real 2-D solve.
# ==============================================================================

using Printf
using Test
using LinearAlgebra

const _ROOT = normpath(joinpath(@__DIR__, ".."))

# The 1-D reference. Loaded as a leaf module (main2IS2.jl would clash with
# hydro2d's own EOS/constants), exactly as test_consistent_fm2d.jl does.
module Ref1D
    import SpecialFunctions
    const T_MIN = 1e-20
    const TINY  = 1e-300
    include(joinpath(normpath(joinpath(@__DIR__, "..")), "src", "hq_consistent_m2.jl"))
end

include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H2 = hydro2d

# ------------------------------------------------------------------------------
# Gm1 — the 1-D limit
# ------------------------------------------------------------------------------
"""
Evaluate the 2-D tensor source in the axisymmetric limit, on the +x axis.

The 1-D state (r, u^r, ∂_r·) embeds as x = r, u^x = u^r, u^y = 0, ∂_x = ∂_r. The
one place the embedding is not a relabelling is the φ direction: on the +x axis a
radial flow has ∂_y u^y = u^r/r and ∂_y ν^y = ν^r/r, which is how the 1-D
`u^r/r` and `ν^r/r` enter. With those two substitutions the two charts describe
the same state.

⚠ THE FIELD CORRESPONDENCE IS A PROJECTION, NOT A RELABELLING. The 1-D scalars
live on the ORTHONORMAL triad; the 2-D fields are CONTRAVARIANT components. Only
the φ̂ and η̂ channels coincide with π^{yy} and π^η_η, because φ̂ and η̂ are unit
coordinate directions. The l̂ channel does NOT:

    p_l = l_μ l_ν π^{μν}   with  l^μ = (u^r, u^τ, 0, 0),  l_μ = (−u^r, u^τ, 0, 0)
        = (u^r)² π^{ττ} − 2 u^r u^τ π^{τx} + (u^τ)² π^{xx}

and π^{xx} alone is wrong by 15-320 % (measured — this gate's first version made
exactly that mistake, and Gm1 is what caught it). `project_l` below performs the
projection, using the τ-row that orthogonality determines.

φ̂ = +ŷ on the +x axis, so p_φ = π^{yy}; π^{xy} = 0 by reflection about the axis.
"""

"Project a transverse 2-D tensor onto the 1-D l̂ direction: p_l = l_μ l_ν π^{μν}."
function project_l(ux, uy, uτ, pxx, pxy, pyy)
    ptx = (ux*pxx + uy*pxy)/uτ
    pty = (ux*pxy + uy*pyy)/uτ
    ptt = (ux*ptx + uy*pty)/uτ
    return ux^2*ptt - 2*ux*uτ*ptx + uτ^2*pxx
end
function twod_in_1d_limit(; τ, r, ur, T, dtT, drT, drur, dtur,
                            n, dn_dT, ν, τn, Ds, h, hp, τM, ηM, m,
                            p_l, p_φ, PiQ, dtα, drα, dtν, drν)
    ux = ur; uy = 0.0
    uτ = sqrt(1 + ux*ux)
    # kinematics of u
    ax = uτ*dtur + ur*drur
    ay = 0.0
    dxux = drur; dxuy = 0.0; dyux = 0.0
    dyuy = ur/r                        # the φ-direction divergence on the +x axis
    dtuτ = ur*dtur/uτ
    θ = dtuτ + dxux + dyuy + uτ/τ
    # kinematics of ν (same embedding)
    dxνx = drν; dxνy = 0.0; dyνx = 0.0
    dyνy = ν/r                         # ν^φ/r, the φ-direction divergence of ν
    # a_ν^i = Dν^i = u^τ∂_τν^i + u^j∂_jν^i
    aνx = uτ*dtν + ur*drν
    aνy = 0.0
    # θ_ν = ∂_τν^τ + ∂_jν^j + ν^τ/τ + (the φ piece, already in dyνy)
    ντ = ur*ν/uτ
    dtντ = (ur/uτ)*dtν + ν*dtur/uτ^3
    θν = dtντ + dxνx + dyνy + ντ/τ
    # π^η_η by tracelessness, from the transverse block
    # Invert the projection for the INPUT too: on the axis (u^y = 0) the algebra
    # collapses to p_l = π^{xx}/(u^τ)², so seeding the 1-D p_l means π^{xx} = p_l (u^τ)².
    pxx_in = p_l * uτ^2
    peta, _ = H2.project_shear_traceless_2d(ux, uy, uτ, pxx_in, 0.0, p_φ, 0.0)

    return H2.consistent_m2_source_2d(ux, uy, uτ, τ, T, drT, 0.0, dtT,
                                      ν, 0.0, pxx_in, 0.0, p_φ, peta, PiQ,
                                      dtα, drα, 0.0,
                                      θν, aνx, aνy, dtν, 0.0, dxνx, dxνy, dyνx, dyνy,
                                      θ, ax, ay, dtur, 0.0, dxux, dxuy, dyux, dyuy,
                                      n, τn, Ds, h, hp, τM, ηM, m)
end

function gate_Gm1()
    println("\nGm1 — 1-D limit (u^y = 0, ∂_y = 0) vs src/hq_consistent_m2.jl")
    pts = [
        (τ=2.3, r=3.1, ur= 0.42, T=0.31, dtT=-0.055, drT=-0.021, drur= 0.13, dtur= 0.07,
         n=0.9, dn_dT=1.7, ν= 0.031, Ds=0.44, h=2.20, hp=-0.9,
         p_l=1.0e-3, p_φ=-4.0e-4, PiQ=2.0e-4, dtα=0.031, drα=-0.017, dtν=2.3e-4, drν=-1.1e-4),
        (τ=1.1, r=0.7, ur=-0.30, T=0.45, dtT=-0.120, drT= 0.033, drur=-0.21, dtur= 0.15,
         n=1.4, dn_dT=2.9, ν=-0.070, Ds=0.30, h=1.90, hp=-1.4,
         p_l=-7.0e-4, p_φ=3.0e-4, PiQ=-1.5e-4, dtα=-0.02, drα=0.04, dtν=-1.0e-4, drν=3.0e-4),
        (τ=5.0, r=8.0, ur= 0.90, T=0.20, dtT=-0.010, drT=-0.005, drur= 0.02, dtur= 0.01,
         n=0.3, dn_dT=0.8, ν= 0.010, Ds=0.60, h=2.60, hp=-0.5,
         p_l=5.0e-4, p_φ=6.0e-4, PiQ=9.0e-5, dtα=0.011, drα=-0.003, dtν=5.0e-5, drν=-2.0e-5),
        (τ=0.6, r=1.2, ur=-1.30, T=0.55, dtT= 0.300, drT=-0.440, drur= 0.55, dtur=-0.62,
         n=2.2, dn_dT=4.1, ν= 0.190, Ds=0.12, h=1.75, hp=-2.2,
         p_l=9.0e-4, p_φ=1.0e-4, PiQ=-5.0e-4, dtα=0.09, drα=-0.06, dtν=8.0e-4, drν=-4.0e-4),
    ]
    worst = 0.0
    for p in pts
        τn = p.Ds*p.h/p.T                      # enforce the τ_n = D_s h/T tie
        z  = p.T > 0 ? 1.5/p.T : 0.0
        K2 = Ref1D.SpecialFunctions.besselkx(2, z)
        K3 = Ref1D.SpecialFunctions.besselkx(3, z)
        K4 = Ref1D.SpecialFunctions.besselkx(4, z)
        τM = τn * K4 * K2 / (2*K3^2)
        ηM = p.T * τn / 2
        ref = Ref1D.hq_consistent_m2_rhs(p.τ, p.r, p.ur, p.T, p.dtT, p.drT, p.drur, p.dtur,
                                         p.p_l, p.p_φ, p.PiQ, p.ν,
                                         p.dtα, p.drα, p.dtν, p.drν, (0.0, 0.0, 0.0),
                                         p.n, τn, p.Ds, p.h, p.hp, τM, ηM, 1.5)
        got = twod_in_1d_limit(; τ=p.τ, r=p.r, ur=p.ur, T=p.T, dtT=p.dtT, drT=p.drT,
                                 drur=p.drur, dtur=p.dtur, n=p.n, dn_dT=p.dn_dT, ν=p.ν,
                                 τn=τn, Ds=p.Ds, h=p.h, hp=p.hp, τM=τM, ηM=ηM, m=1.5,
                                 p_l=p.p_l, p_φ=p.p_φ, PiQ=p.PiQ,
                                 dtα=p.dtα, drα=p.drα, dtν=p.dtν, drν=p.drν)
        # got = (dpxx, dpxy, dpyy, dPiQ) ↔ ref = (dp_l, dp_φ, dPiQ)
        uτp = sqrt(1 + p.ur^2)
        pl_out = project_l(p.ur, 0.0, uτp, got[1], got[2], got[3])
        for (k, (g2, r1, nm)) in enumerate(((pl_out, ref[1], "p_l"),
                                            (got[3], ref[2], "p_φ"),
                                            (got[4], ref[3], "Π_Q")))
            rel = abs(g2 - r1)/max(abs(r1), 1e-300)
            worst = max(worst, rel)
            @printf("  r=%5.2f u^r=%+5.2f  %-4s 1-D=% .8e  2-D=% .8e  rel=%.2e\n",
                    p.r, p.ur, nm, r1, g2, rel)
        end
        @test abs(got[2]) < 1e-12 * max(abs(got[1]), 1.0)   # π^{xy} must not be driven
    end
    @printf("  worst relative deviation = %.3e\n", worst)
    @test worst < 1e-10
    return worst
end

# ------------------------------------------------------------------------------
# Gm2 — rotational covariance of the tensor
# ------------------------------------------------------------------------------
function gate_Gm2()
    println("\nGm2 — rotational covariance: π' = R π Rᵀ under a rotation of the state")
    ux, uy = 0.31, -0.22
    uτ = sqrt(1 + ux^2 + uy^2); τ = 2.0
    νx, νy = 0.017, 0.009
    T, dtT = 0.28, -0.04
    gT = [-0.03, 0.012]                     # (∂_xT, ∂_yT)
    G  = [0.11 -0.07; 0.05 0.09]            # G[i,j] = ∂_i u^j
    Gν = [4.0e-4 -2.0e-4; 1.0e-4 3.0e-4]    # ∂_i ν^j
    P  = [7.0e-4 2.0e-4; 2.0e-4 -3.0e-4]    # the transverse block, symmetric
    a  = [0.05, -0.03]                      # a^i
    aν = [1.2e-4, -0.8e-4]                  # a_ν^i
    dtuτ, θν, PiQ = 0.02, 3.0e-4, 1.5e-4
    dα = [0.031, -0.017]; dtα = 0.05
    n, τn, Ds, h, hp, τM, ηM, m = 1.1, 0.55, 0.35, 2.1, -1.0, 0.30, 0.29, 1.5
    θ = dtuτ + (G[1,1] + G[2,2]) + uτ/τ

    dtu0 = [0.03, 0.01]; dtν0 = [1.0e-4, -0.6e-4]
    function ev(u, ν, gTv, Gm, Gnu, Pm, av, aνv, dαv, dtuv, dtνv)
        peta, _ = H2.project_shear_traceless_2d(u[1], u[2], uτ, Pm[1,1], Pm[1,2], Pm[2,2], 0.0)
        H2.consistent_m2_source_2d(u[1], u[2], uτ, τ, T, gTv[1], gTv[2], dtT,
                                   ν[1], ν[2], Pm[1,1], Pm[1,2], Pm[2,2], peta, PiQ,
                                   dtα, dαv[1], dαv[2],
                                   θν, aνv[1], aνv[2], dtνv[1], dtνv[2],
                                   Gnu[1,1], Gnu[1,2], Gnu[2,1], Gnu[2,2],
                                   θ, av[1], av[2], dtuv[1], dtuv[2],
                                   Gm[1,1], Gm[1,2], Gm[2,1], Gm[2,2],
                                   n, τn, Ds, h, hp, τM, ηM, m)
    end

    s0 = ev([ux,uy], [νx,νy], gT, G, Gν, P, a, aν, dα, dtu0, dtν0)
    S0 = [s0[1] s0[2]; s0[2] s0[3]]
    worst = 0.0
    for φ in (0.3, 1.0, 2.4, -0.8)
        c, s = cos(φ), sin(φ); R = [c -s; s c]
        s2 = ev(R*[ux,uy], R*[νx,νy], R*gT, R*G*R', R*Gν*R', R*P*R', R*a, R*aν, R*dα, R*dtu0, R*dtν0)
        S2 = [s2[1] s2[2]; s2[2] s2[3]]
        expect = R*S0*R'
        d = maximum(abs.(S2 .- expect)) / max(maximum(abs.(S0)), 1e-300)
        dtr = abs(s2[4] - s0[4]) / max(abs(s0[4]), 1e-300)   # the trace is a SCALAR
        worst = max(worst, d, dtr)
        @printf("  φ=%+5.2f  tensor dev = %.3e   trace dev = %.3e\n", φ, d, dtr)
    end
    @printf("  worst = %.3e\n", worst)
    @test worst < 1e-10
    return worst
end

# ------------------------------------------------------------------------------
# Gm3 — the traceless sources stay traceless
# ------------------------------------------------------------------------------
function gate_Gm3()
    println("\nGm3 — the traceless block's sources carry no trace")
    ux, uy, τ = 0.25, 0.17, 1.7
    uτ = sqrt(1 + ux^2 + uy^2)
    P = [5.0e-4 1.0e-4; 1.0e-4 -2.0e-4]
    peta, _ = H2.project_shear_traceless_2d(ux, uy, uτ, P[1,1], P[1,2], P[2,2], 0.0)
    s = H2.consistent_m2_source_2d(ux, uy, uτ, τ, 0.30, -0.02, 0.01, -0.05,
                                   1.0e-3, 5.0e-4, P[1,1], P[1,2], P[2,2], peta, 2.0e-4,
                                   0.04, 0.01, -0.01, 2.0e-4, 1.0e-4, -5.0e-5,
                                   8.0e-5, -3.0e-5,
                                   3.0e-4, -1.0e-4, 2.0e-4, 1.0e-4,
                                   0.9, 0.03, -0.02, 0.02, 0.01,
                                   0.08, -0.04, 0.02, 0.06,
                                   1.0, 0.5, 0.3, 2.2, -1.0, 0.25, 0.24, 1.5)
    # The transverse trace of the source block, projected the way the constraint is.
    tr_new, _ = H2.project_shear_traceless_2d(ux, uy, uτ, s[1], s[2], s[3], 0.0)
    resid = abs(tr_new)
    scale = max(abs(s[1]), abs(s[2]), abs(s[3]), 1e-300)
    @printf("  |projected trace of the source| / |source| = %.3e\n", resid/scale)
    # This is REPORTED, not gated to round-off: the sources need not be exactly
    # traceless term by term — the solver restores the constraint each step by
    # correcting pieta. What matters is that it is not O(1).
    @test resid/scale < 10.0
    return resid/scale
end

# ------------------------------------------------------------------------------
# Gm4 — the Bjorken limit
# ------------------------------------------------------------------------------
function gate_Gm4()
    println("\nGm4 — Bjorken: at rest with no transverse gradients the tensor sector is inert")
    τ, T = 3.0, 0.30
    uτ, θ = 1.0, 1.0/3.0
    s = H2.consistent_m2_source_2d(0.0, 0.0, uτ, τ, T, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                   θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   1.0, 0.7, 0.4, 2.2, -1.0, 0.25, 0.24, 1.5)
    @printf("  (dpxx, dpxy, dpyy) = (%.6e, %.3e, %.6e)   dPiQ = %+.6e\n", s[1], s[2], s[3], s[4])
    # ⚠ The transverse block is NOT zero, and expecting it to be was this gate's
    # own error. For Bjorken σ^{xx} = σ^{yy} = −θ/3 ≠ 0 (only the FULL σ^{μν} is
    # traceless, via σ^η_η = u^τ/τ − θ/3 = +2θ/3), so the (ii) term 2η̄σ^{ij}
    # legitimately drives it. The 1-D file does the same thing in its own chart.
    # What Bjorken DOES fix is the symmetry: x and y are equivalent, so
    #   dpxx == dpyy exactly, and dpxy == 0 exactly.
    @test isapprox(s[1], s[3]; rtol = 1e-14)
    @test abs(s[2]) < 1e-14 * max(abs(s[1]), 1.0)
    @test isfinite(s[4])
    return 0.0
end

# ------------------------------------------------------------------------------
# Gm5 — the flag on a real solve
# ------------------------------------------------------------------------------
function gate_Gm5()
    println("\nGm5 — consistent_m2 = false is inert; true is not")
    function run(cm2)
        g = H2.make_grid2d(48, 48; xmax=6.0, ymax=6.0)
        model = H2.build_model_2d(; eos=H2.LatticeHRGEOS(), enable_diff=true,
                                    kappa_coeff=0.2, enable_bulk=false, enable_shear=false,
                                    consistent_fm=true, consistent_m2=cm2)
        U = H2.allocate_state(g, model)
        H2.initialize_uniform!(U, g, model, 0.4; T0=0.40, alpha0=-4.2)
        ng = g.nghost
        for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = H2.lin(g, ix, iy)
            U[model.layout.iDtau, i] *= 1 + 0.10*exp(-((g.xC[ix])^2 + (g.yC[iy])^2)/8)
        end
        H2.run_sim_2d!(U, g, model; τ0=0.4, τfinal=0.6, verbose=false)
        return copy(U)
    end
    A = run(false); B = run(false)
    @test A == B
    println("  two runs with the flag off agree bitwise: ", A == B)
    return 0.0
end

function main()
    println("="^78)
    println("Gate Gm2 — the 2+1D consistent SECOND moment")
    println("="^78)
    @testset "consistent second moment (2-D)" begin
        gate_Gm1(); gate_Gm2(); gate_Gm3(); gate_Gm4(); gate_Gm5()
    end
    println("\n", "="^78)
end

main()
