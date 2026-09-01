# ==============================================================================
# test/test_shear2d_algebra.jl
#
# Gate G1(algebra): the 2+1D shear closure in src2d/shear2d.jl is a genuine
# symmetric / traceless / u-orthogonal tensor, and it REDUCES TO THE 1-D
# PRODUCTION PARAMETRISATION in the azimuthally symmetric limit.
#
# Every check is against an independently written expression — the metric
# contractions here are spelled out explicitly rather than reusing the closure,
# so a sign error in shear2d.jl cannot cancel itself.
#
# Run:  julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_shear2d_algebra.jl
# ==============================================================================

using Printf
using Random
using Test

const _HERE = @__DIR__
const _ROOT = normpath(joinpath(_HERE, ".."))

include(joinpath(_ROOT, "src", "constants.jl"))
include(joinpath(_ROOT, "src", "utils.jl"))
include(joinpath(_ROOT, "src2d", "shear2d.jl"))

# ------------------------------------------------------------------------------
# Independent metric machinery — deliberately NOT sharing code with shear2d.jl.
# Coordinates (τ, x, y, η); g = diag(-1, 1, 1, τ²).
# ------------------------------------------------------------------------------

metric_lower(τ) = [-1.0 0.0 0.0 0.0;
                    0.0 1.0 0.0 0.0;
                    0.0 0.0 1.0 0.0;
                    0.0 0.0 0.0 τ*τ]

"""Assemble the full 4×4 contravariant π^{μν} from the closure under test."""
function pi_full(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
    Π = shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
    return [Π.tt  Π.tx  Π.ty  0.0;
            Π.tx  Π.xx  Π.xy  0.0;
            Π.ty  Π.xy  Π.yy  0.0;
            0.0   0.0   0.0   Π.etaeta]
end

uvec(ux, uy) = [sqrt(1 + ux^2 + uy^2), ux, uy, 0.0]

"""Random constraint-SATISFYING state: pick the transverse block freely, then let
the projection fix pieta. Returns (ux, uy, uτ, pixx, pixy, piyy, pieta)."""
function random_projected_state(rng; umax = 2.0, pscale = 0.3)
    ux = umax*(2rand(rng) - 1)
    uy = umax*(2rand(rng) - 1)
    uτ = sqrt(1 + ux^2 + uy^2)
    pixx = pscale*(2rand(rng) - 1)
    pixy = pscale*(2rand(rng) - 1)
    piyy = pscale*(2rand(rng) - 1)
    pieta, _ = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, 0.0)
    return ux, uy, uτ, pixx, pixy, piyy, pieta
end

# ------------------------------------------------------------------------------

const RNG = MersenneTwister(20260901)
const NTRIAL = 2000

@testset "2+1D shear closure algebra" begin

    @testset "orthogonality u_μ π^{μν} = 0 (exact by construction)" begin
        worst = 0.0
        for _ in 1:NTRIAL
            τ = 0.4 + 12*rand(RNG)
            ux, uy, uτ, pixx, pixy, piyy, pieta = random_projected_state(RNG)
            Π = pi_full(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
            g = metric_lower(τ)
            u = uvec(ux, uy)
            ul = g*u                      # u_μ
            res = Π'*ul                   # (u_μ π^{μν})_ν
            scale = maximum(abs, Π)
            worst = max(worst, maximum(abs, res)/max(scale, TINY))
        end
        @printf("  orthogonality   max rel residual = %.3e\n", worst)
        @test worst < 1e-12
    end

    @testset "tracelessness g_{μν} π^{μν} = 0 after projection" begin
        worst = 0.0
        for _ in 1:NTRIAL
            τ = 0.4 + 12*rand(RNG)
            ux, uy, uτ, pixx, pixy, piyy, pieta = random_projected_state(RNG)
            Π = pi_full(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
            g = metric_lower(τ)
            tr = sum(g .* Π)              # g_{μν} π^{μν}
            worst = max(worst, abs(tr)/max(maximum(abs, Π), TINY))
        end
        @printf("  tracelessness   max rel residual = %.3e\n", worst)
        @test worst < 1e-12
    end

    @testset "residual monitor detects a deliberately broken constraint" begin
        # Perturb pieta away from its projected value; the monitor must report it
        # at the size of the perturbation, and the projection must undo it.
        worst_detect = 0.0
        worst_restore = 0.0
        for _ in 1:NTRIAL
            ux, uy, uτ, pixx, pixy, piyy, pieta = random_projected_state(RNG)
            δ = 1e-3*(2rand(RNG) - 1)
            R, _ = shear_constraint_residual_2d(ux, uy, uτ, pixx, pixy, piyy, pieta + δ)
            worst_detect = max(worst_detect, abs(R - δ))
            fixed, corr = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, pieta + δ)
            worst_restore = max(worst_restore, abs(fixed - pieta))
            @test isapprox(corr, -δ; atol = 1e-14)
        end
        @printf("  monitor         max |R - δ|      = %.3e\n", worst_detect)
        @printf("  projection      max restore err  = %.3e\n", worst_restore)
        @test worst_detect < 1e-13
        @test worst_restore < 1e-13
    end

    @testset "independent 3-dof closure (Fluidum parametrisation) agrees" begin
        # pixx_from_closure_2d eliminates π^{xx} instead of π^{ηη}. On a
        # constraint-satisfying state the two closures must return the same tensor.
        worst = 0.0
        for _ in 1:NTRIAL
            ux, uy, uτ, pixx, pixy, piyy, pieta = random_projected_state(RNG)
            back = pixx_from_closure_2d(ux, uy, uτ, pixy, piyy, pieta)
            worst = max(worst, abs(back - pixx)/max(abs(pixx), 1e-8))
        end
        @printf("  3-dof closure   max rel disagreement = %.3e\n", worst)
        @test worst < 1e-9
    end

    @testset "x↔y exchange symmetry" begin
        # Swapping the axes must map the tensor onto itself with x and y traded.
        # The 3-dof-eliminate-π^{xx} closure would FAIL this; that is why the
        # solver projects pieta instead. See TWOD_PROGRAM.md §2.
        worst = 0.0
        for _ in 1:NTRIAL
            τ = 0.4 + 12*rand(RNG)
            ux, uy, uτ, pixx, pixy, piyy, pieta = random_projected_state(RNG)

            A = shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
            # swapped state
            petaS, _ = project_shear_traceless_2d(uy, ux, uτ, piyy, pixy, pixx, 0.0)
            B = shear_tensor_contravariant_2d(uy, ux, uτ, τ, piyy, pixy, pixx, petaS)

            scale = max(maximum(abs, (A.tt, A.tx, A.ty, A.xx, A.xy, A.yy)), 1e-12)
            d = max(abs(A.tt - B.tt), abs(A.tx - B.ty), abs(A.ty - B.tx),
                    abs(A.xx - B.yy), abs(A.yy - B.xx), abs(A.xy - B.xy),
                    abs(A.etaeta - B.etaeta))
            worst = max(worst, d/scale)
        end
        @printf("  x<->y symmetry  max rel asymmetry = %.3e\n", worst)
        @test worst < 1e-12
    end

    @testset "reduces to the 1-D production parametrisation (src/shear_tensor.jl)" begin
        # The 1-D code stores piR, piEta as the LOCAL-REST-FRAME diagonal, boosted:
        #     Π^{ττ} = (u^r)² piR,  Π^{τr} = u^τu^r piR,  Π^{rr} = (u^τ)² piR
        #     π^φ_φ  = piPhi = -(piR + piEta),  π^η_η = piEta
        # So in the u^y = 0, π^{xy} = 0 limit the correspondence is
        #     pixx = (u^τ)² piR,   piyy = piPhi,   pieta = piEta
        # and our projection must RETURN piEta on its own.
        worst_tt = 0.0; worst_tx = 0.0; worst_eta = 0.0
        for _ in 1:NTRIAL
            τ  = 0.4 + 12*rand(RNG)
            ur = 2.0*(2rand(RNG) - 1)
            uτ = sqrt(1 + ur^2)
            piR   = 0.3*(2rand(RNG) - 1)
            piEta = 0.3*(2rand(RNG) - 1)
            piPhi = -(piR + piEta)

            pixx = uτ^2 * piR
            piyy = piPhi

            # our projection, told nothing about piEta, must reconstruct it
            pieta, _ = project_shear_traceless_2d(ur, 0.0, uτ, pixx, 0.0, piyy, 0.0)
            worst_eta = max(worst_eta, abs(pieta - piEta))

            A = shear_tensor_contravariant_2d(ur, 0.0, uτ, τ, pixx, 0.0, piyy, pieta)
            worst_tt = max(worst_tt, abs(A.tt - ur^2 * piR))
            worst_tx = max(worst_tx, abs(A.tx - uτ*ur*piR))
        end
        @printf("  1-D limit       max |π^ττ - u_r²piR|   = %.3e\n", worst_tt)
        @printf("  1-D limit       max |π^τx - u^τu^r piR|= %.3e\n", worst_tx)
        @printf("  1-D limit       max |pieta - piEta|    = %.3e\n", worst_eta)
        @test worst_tt  < 1e-12
        @test worst_tx  < 1e-12
        @test worst_eta < 1e-12
    end

    @testset "diffusion current orthogonality u_μ ν^μ = 0" begin
        worst = 0.0
        for _ in 1:NTRIAL
            τ = 0.4 + 12*rand(RNG)
            ux = 2.0*(2rand(RNG) - 1); uy = 2.0*(2rand(RNG) - 1)
            uτ = sqrt(1 + ux^2 + uy^2)
            nux = 0.5*(2rand(RNG) - 1); nuy = 0.5*(2rand(RNG) - 1)
            nut = nu_tau_2d(ux, uy, uτ, nux, nuy)
            ν = [nut, nux, nuy, 0.0]
            u = uvec(ux, uy)
            res = (metric_lower(τ)*u)' * ν
            worst = max(worst, abs(res)/max(maximum(abs, ν), TINY))
        end
        @printf("  ν orthogonality max rel residual = %.3e\n", worst)
        @test worst < 1e-12
    end
end
