# ==============================================================================
# test/test_dissipation2d.jl — gate G2, the dissipative sector.
#
# Three independent things have to be right, and they are tested separately so a
# failure says WHICH:
#
#   (a) the NS target derivation      — analytic identity, no time evolution
#   (b) agreement with the 1-D solver — in the azimuthally symmetric limit
#   (c) the implementation            — viscous Bjorken vs an independently
#                                       integrated ODE, plus the physical sign
#
# (a) and (b) test the DERIVATION; (c) tests the CODE. Passing (c) alone would not
# rule out a consistently wrong σ.
#
# Run: julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_dissipation2d.jl
# ==============================================================================

using Printf
using Random
using Test

include(joinpath(@__DIR__, "..", "main2D.jl"))
using .hydro2d
const H = hydro2d

const RNG = MersenneTwister(20260901)

# ------------------------------------------------------------------------------
# (a) NS target: projecting π_NS^{ij} must reproduce π_NS^{η}_{η} = -2η σ^η_η
#
# σ^{μν} is traceless and u-orthogonal by construction, so π_NS = -2ησ satisfies
# the same constraints the storage closure assumes. The two routes to `pieta` —
# projection of the transverse block, and the direct formula -2η(u^τ/τ - θ/3) —
# are therefore the same number. They are computed by different code paths, so
# agreement is a real check on the σ^{ij} expression, not a tautology.
# ------------------------------------------------------------------------------
@testset "G2a — NS target is traceless: projection reproduces sigma^eta_eta" begin
    worst = 0.0
    for _ in 1:4000
        τ  = 0.4 + 12*rand(RNG)
        ux = 1.5*(2rand(RNG)-1); uy = 1.5*(2rand(RNG)-1)
        uτ = sqrt(1 + ux^2 + uy^2)
        η  = 0.5 + rand(RNG)

        # independent random kinematics; θ must be the SAME θ used inside
        dxux = 2*(2rand(RNG)-1); dxuy = 2*(2rand(RNG)-1)
        dyux = 2*(2rand(RNG)-1); dyuy = 2*(2rand(RNG)-1)
        dtux = 2*(2rand(RNG)-1); dtuy = 2*(2rand(RNG)-1)
        dtuτ = (ux*dtux + uy*dtuy)/uτ
        θ = dtuτ + dxux + dyuy + uτ/τ
        ax = uτ*dtux + ux*dxux + uy*dyux
        ay = uτ*dtuy + ux*dxuy + uy*dyuy

        nsxx, nsxy, nsyy, nseta = H.ns_shear_target_2d(ux, uy, uτ, τ, θ, ax, ay,
                                                       dxux, dxuy, dyux, dyuy, η)
        proj, _ = H.project_shear_traceless_2d(ux, uy, uτ, nsxx, nsxy, nsyy, 0.0)
        scale = max(abs(nseta), abs(nsxx), abs(nsyy), 1e-12)
        worst = max(worst, abs(proj - nseta)/scale)
    end
    @printf("  max |projected pieta_NS - (-2eta sigma^eta_eta)| / |pi| = %.3e\n", worst)
    @test worst < 1e-11
end

# ------------------------------------------------------------------------------
# (b) 1-D limit. src/shear_tensor.jl's `shear_NS_target_contravariant` uses
#         sigma_phi = u^r/r - theta/3 ,  sigma_eta = u^tau/tau - theta/3
#     for the two components it evolves. For a radial flow u = u_r(r) r-hat laid
#     out on the Cartesian grid, u^y = u_r(r) y/r, so at a point ON the x-axis
#     d(u^y)/dy = u_r/r. Our sigma^{yy} must therefore equal the 1-D sigma_phi
#     there — and the acceleration term must drop out on its own, because u^y = 0.
# ------------------------------------------------------------------------------
@testset "G2b — reduces to the 1-D NS target on the symmetry axis" begin
    worst_phi = 0.0; worst_eta = 0.0
    for _ in 1:2000
        τ  = 0.4 + 12*rand(RNG)
        r  = 0.5 + 10*rand(RNG)
        ur = 1.5*(2rand(RNG)-1)
        dur_dr = 2*(2rand(RNG)-1)
        durdτ  = 2*(2rand(RNG)-1)
        η  = 0.5 + rand(RNG)

        # On the x-axis: u^x = u_r, u^y = 0,
        #   d(u^x)/dx = du_r/dr,  d(u^y)/dy = u_r/r,  d(u^x)/dy = d(u^y)/dx = 0
        ux = ur; uy = 0.0
        uτ = sqrt(1 + ux^2)
        dxux = dur_dr; dyuy = ur/r; dxuy = 0.0; dyux = 0.0
        dtux = durdτ;  dtuy = 0.0
        dtuτ = ux*dtux/uτ
        θ = dtuτ + dxux + dyuy + uτ/τ
        ax = uτ*dtux + ux*dxux
        ay = 0.0

        nsxx, nsxy, nsyy, nseta = H.ns_shear_target_2d(ux, uy, uτ, τ, θ, ax, ay,
                                                       dxux, dxuy, dyux, dyuy, η)
        # the 1-D targets, written out independently
        σφ_1d = ur/r      - θ/3
        ση_1d = uτ/τ      - θ/3
        worst_phi = max(worst_phi, abs(nsyy - (-2η*σφ_1d)))
        worst_eta = max(worst_eta, abs(nseta - (-2η*ση_1d)))
        @test nsxy == 0.0          # no off-diagonal on the axis
    end
    @printf("  max |pi_NS^{yy} - 1-D pi_NS^{phi}_{phi}| = %.3e\n", worst_phi)
    @printf("  max |pi_NS^{eta} - 1-D pi_NS^{eta}_{eta}| = %.3e\n", worst_eta)
    @test worst_phi < 1e-12
    @test worst_eta < 1e-12
end

# ------------------------------------------------------------------------------
# (c) Viscous Bjorken.
#
# For a transversely uniform state u = 0, θ = 1/τ, a = 0, so
#     sigma^{xx} = sigma^{yy} = -1/(3τ),   sigma^eta_eta = 2/(3τ)
#     pi_NS^{xx} = +2η/(3τ),               pi_NS^eta_eta = -4η/(3τ)
# and tracelessness gives pieta = -2 pi^{xx}. The system reduces to two ODEs,
#
#     de/dtau     = -(e + P + Pi + pieta)/tau
#     tau_pi dpi^{xx}/dtau = -pi^{xx}(1 + delta_pi*theta) + pi_NS^{xx}
#
# integrated below by RK4 at a step ~100x smaller than the solver's (RK4 at
# dtau ~ 1e-4 has truncation error far below the 1e-3 tolerances here), with the
# transport coefficients taken from the same models the solver uses. This checks
# the IMPLEMENTATION (geometric source, relaxation discretisation, projection);
# the derivation is checked by (a) and (b).
# ------------------------------------------------------------------------------

"""Invert e(T) at fixed alpha for T (monotone). 60 bisections resolve a
double-precision T over [1e-4, 5]; this sits in the RK4 inner loop, so the count
matters for runtime."""
function T_of_e(e_target, alpha, eos)
    lo, hi = 1e-4, 5.0
    for _ in 1:60
        mid = 0.5*(lo+hi)
        _, _, em = H.eos_Pne(mid, alpha*mid, eos)
        em > e_target ? (hi = mid) : (lo = mid)
    end
    return 0.5*(lo+hi)
end

function bjorken_visc_reference(; τ0, τf, T0, alpha0, eos, model, nsub = 20_000)
    e0 = let (P, n, e) = H.eos_Pne(T0, alpha0*T0, eos); e end
    y  = [e0, 0.0]                      # (e, pi^{xx})
    dτ = (τf - τ0)/nsub

    function rhs(τ, y)
        e   = y[1]; pxx = y[2]
        T   = T_of_e(e, alpha0, eos)
        μ   = alpha0*T
        P, n, _ = H.eos_Pne(T, μ, eos)
        η, τπ, δπ = H.shear_coeffs_2d(T, μ, n, e, P, model)
        θ   = 1/τ
        pieta = -2*pxx
        nsxx  = -2η*(-θ/3)              # sigma^{xx} = -theta/3 at u = 0
        de    = -(e + P + pieta)/τ
        dpxx  = τπ > 0 ? (-pxx*(1 + δπ*θ) + nsxx)/τπ : 0.0
        return [de, dpxx]
    end

    τ = τ0
    for _ in 1:nsub
        k1 = rhs(τ, y)
        k2 = rhs(τ + dτ/2, y .+ (dτ/2).*k1)
        k3 = rhs(τ + dτ/2, y .+ (dτ/2).*k2)
        k4 = rhs(τ + dτ,   y .+ dτ.*k3)
        y = y .+ (dτ/6).*(k1 .+ 2k2 .+ 2k3 .+ k4)
        τ += dτ
    end
    return (e = y[1], pixx = y[2], pieta = -2*y[2])
end

function run_visc_bjorken(; CFLτ = 0.004, τ0 = 0.6, τf = 3.0, T0 = 0.45, alpha0 = -4.2,
                            eta_over_s = 0.15, eos = H.LatticeHRGEOS())
    g = H.make_grid2d(20, 20; xmax = 5.0, ymax = 5.0)
    m = H.build_model_2d(; eos = eos, enable_shear = true,
                           eta_over_s = eta_over_s, tauShear_coeff = 0.2)
    U = H.allocate_state(g, m)
    @assert H.initialize_uniform!(U, g, m, τ0; T0 = T0, alpha0 = alpha0) == 0

    res = H.run_sim_2d!(U, g, m; τ0 = τ0, τfinal = τf, CFL = 0.2, CFLτ = CFLτ)
    @assert res.ok

    L = m.layout; ng = g.nghost
    ic = H.lin(g, ng + g.Nx÷2, ng + g.Ny÷2)

    maxU = 0.0; Emin = Inf; Emax = -Inf
    for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
        i = H.lin(g, ix, iy)
        maxU = max(maxU, abs(res.work.ux[i]), abs(res.work.uy[i]))
        Emin = min(Emin, U[L.iE,i]); Emax = max(Emax, U[L.iE,i])
    end

    ref = bjorken_visc_reference(; τ0 = τ0, τf = res.τ, T0 = T0, alpha0 = alpha0,
                                  eos = eos, model = m)
    return (τ = res.τ, nsteps = res.nsteps, primfail = res.nprimfail,
            maxU = maxU, spread = (Emax - Emin)/abs(Emax),
            shear_res = res.max_shear_res,
            e = U[L.iE,ic], e_ref = ref.e,
            pixx = H.phys_from_stored(U[L.iPixx,ic]), pixx_ref = ref.pixx,
            pieta = H.phys_from_stored(U[L.iPieta,ic]), pieta_ref = ref.pieta,
            eerr = abs(U[L.iE,ic] - ref.e)/ref.e,
            perr = abs(H.phys_from_stored(U[L.iPixx,ic]) - ref.pixx)/max(abs(ref.pixx),1e-12))
end

@testset "G2c — viscous Bjorken vs independently integrated ODE" begin
    r = run_visc_bjorken()
    @printf("  steps %d  primfail %d\n", r.nsteps, r.primfail)
    @printf("  max |u|            = %.3e   (uniform state must not drift)\n", r.maxU)
    @printf("  transverse spread  = %.3e\n", r.spread)
    @printf("  shear constraint   = %.3e   (gate G1 monitor)\n", r.shear_res)
    @printf("  e     %.6e vs ref %.6e   rel %.3e\n", r.e, r.e_ref, r.eerr)
    @printf("  pi^xx %.6e vs ref %.6e   rel %.3e\n", r.pixx, r.pixx_ref, r.perr)
    @printf("  pieta %.6e vs ref %.6e\n", r.pieta, r.pieta_ref)

    @test r.primfail == 0
    @test r.maxU == 0.0
    @test r.spread < 1e-12
    @test r.shear_res < 1e-12
    @test r.eerr < 2e-3
    @test r.perr < 2e-2
end

@testset "G2d — the physical sign: shear SLOWS Bjorken cooling" begin
    # pi^eta_eta < 0 lowers the longitudinal pressure, so the viscous run must be
    # HOTTER than the ideal one at the same tau. This is the check that fixes the
    # pi_NS = -2*eta*sigma sign convention; getting it backwards passes every
    # algebraic test above and fails here.
    ideal = run_visc_bjorken(; eta_over_s = 0.0)
    visc  = run_visc_bjorken(; eta_over_s = 0.15)
    @printf("  e(ideal) = %.6e    e(eta/s=0.15) = %.6e    ratio %.5f\n",
            ideal.e, visc.e, visc.e/ideal.e)
    @printf("  pi^eta_eta = %.6e  (must be NEGATIVE)\n", visc.pieta)
    @test visc.pieta < 0.0
    @test visc.e > ideal.e
end

@testset "G2e — convergence of the operator splitting" begin
    # Relaxation is backward Euler applied once per step: first order overall.
    errs = [run_visc_bjorken(; CFLτ = c).eerr for c in (0.016, 0.008, 0.004)]
    p = log2(errs[2]/errs[3])
    @printf("  errors %.3e %.3e %.3e   observed order %.2f\n", errs..., p)
    @test p > 0.7
end
