# ==============================================================================
# test/test_diffusion_mode.jl — gate X1: the charm current, THREE solvers, ONE referee.
#
# A small charge perturbation δn = A J₀(kr) on a transversely uniform, charm-carrying
# Bjorken background. Its linearised equations close EXACTLY on (A, B) with
# ν^r = B J₁(kr) (analytic_referees.jl `diffusion_mode`; coefficients in closed form,
# independent of every solver):
#
#     dA/dτ = −A/τ − kB ,     τ_n dB/dτ = D_s k A − (1 + b) B ,
#     b = τ_n θ [fm_expansion] + (D_s/T) h′ DT [fm_dlnh]      (θ = 1/τ here)
#
# The same state is run through
#     1D   the 1+1D bulk solver (main.jl, `run_sim_1d!`)      — ν^r relaxation
#     IS2  the 1+1D charm solver (main2IS2.jl) on an analytic background — the 5×5 matrix
#     2D   the 2+1D solver (main2D.jl) — a Cartesian grid carrying the Bessel pattern
# in four configurations of the first moment:
#     shipped            consistent_fm = false                       (b = 0)
#     consistent         every term                                  (b = τ_nθ + (D_s/T)h′DT)
#     no expansion       terms = (fm_expansion = false,)              (b = (D_s/T)h′DT)
#     homogeneous        terms = :homogeneous — must equal shipped   (b = 0)
# On this state ∇T = 0, a = 0 and ν·∇u = 0, so fm_gradT, fm_inertial and fm_nu_gradu
# vanish identically: the mode tests the relaxation, the fugacity drive (the telegraph
# dispersion, i.e. τ_n and D_s), and the two ν-proportional consistent terms — the
# ones whose SIGN the 2-D solver had wrong for two days (EQUATIONS2D.md §10).
#
# The amplitudes are projections of (perturbed − unperturbed) solves, so each solver's
# background discretisation cancels and what is compared is the mode alone.
#
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_diffusion_mode.jl
# ==============================================================================

using Printf
using Test
using SpecialFunctions: besselj0, besselj1

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2IS2.jl"))
include(joinpath(_ROOT, "main2D.jl"))
include(joinpath(@__DIR__, "analytic_referees.jl"))
using .hydro, .hydro_current_IS2, .hydro2d
const H1 = hydro; const HI = hydro_current_IS2; const H2 = hydro2d

# The IS2 production EOS. Each module defines its own LatticeHRGEOS type (the drivers
# `include` src/eos.jl into separate modules), so there is one object per solver,
# built with identical parameters.
const EOS  = H1.LatticeHRGEOS(canon_factor = 1.0)
const EOSI = HI.LatticeHRGEOS(canon_factor = 1.0)
const EOS2 = H2.LatticeHRGEOS(canon_factor = 1.0)
const T0   = 0.35
const A0L  = -2.0          # background fugacity: n̄ ≈ 0.1 fm⁻³, far above every vacuum ramp
const TAU0 = 1.0
const TAU1 = 2.5
const DST  = 0.1163
const K    = 2.404825557695773/1.5      # J₀'s first zero at r = 1.5 fm
const EPS  = 1e-3
const RCMP = 4.0

J0(r) = besselj0(K*r); J1(r) = besselj1(K*r)
αpert(r, ε) = A0L + log1p(ε*J0(r))       # n ∝ e^α at fixed T ⇒ δn/n̄ = ε J₀

const BG = bjorken_background((T, μ) -> H1.eos_Pne(T, μ, EOS), T0, A0L, TAU0, TAU1 + 0.1; n = 40_000)
const TBAR = BG[2]
const NBAR0 = H1.eos_Pne(T0, A0L*T0, EOS)[2]

referee(; expansion, dlnh) = diffusion_mode(; k = K, DsT = DST, m = H1.hq_mass(EOS), Tbar = TBAR,
                                             τ0 = TAU0, τ1 = TAU1, A0 = EPS*NBAR0, expansion, dlnh)

# ---- 1D bulk --------------------------------------------------------------------
function mode_1d(; consistent_fm, terms, Nr = 600)
    function solve(ε)
        g = H1.make_grid_1d(Nr; rmax = 15.0)
        m = H1.build_model_1d(; eos = EOS, enable_diff = true, kappa_coeff = DST, consistent_fm, terms)
        U = H1.allocate_state(g, m)
        H1.initialize_from_radial!(U, g, m, TAU0, r -> T0, r -> αpert(r, ε))
        res = H1.run_sim_1d!(U, g, m; τ0 = TAU0, τfinal = TAU1, CFLτ = 0.01)
        @test res.ok
        return H1.fields_1d(g, U, m; τ = res.τ, work = res.work)
    end
    fp = solve(EPS); f0 = solve(0.0)
    sel = fp.r .<= RCMP
    A = project_mode(fp.r[sel], (fp.n .- f0.n)[sel], J0)
    B = project_mode(fp.r[sel], (fp.nur .- f0.nur)[sel], J1)
    return A, B
end

# ---- IS2 on the analytic background --------------------------------------------
function mode_is2(; consistent_fm, terms, Nr = 600)
    dT(τ) = (TBAR(τ + 1e-5) - TBAR(τ - 1e-5))/2e-5
    bg = HI.analytic_background(; T = (τ, r) -> TBAR(τ), ur = (τ, r) -> 0.0,
                                  dtT = (τ, r) -> dT(τ), drT = (τ, r) -> 0.0,
                                  dtur = (τ, r) -> 0.0, drur = (τ, r) -> 0.0,
                                  r_grid = collect(0.0:0.05:20.0), t_grid = collect(0.5:0.01:3.0))
    function solve(ε)
        HI.run_static_IS2_test(; background = bg, DsT = DST, τ0 = TAU0, τfinal = TAU1, Nr, rmax = 15.0,
                               init_mode = :n_profile, n_profile = r -> NBAR0*(1 + ε*J0(r)),
                               dump_dt = TAU1 - TAU0, eos = EOSI, consistent_fm, terms, log_every = 10^9)
    end
    rp = solve(EPS); r0 = solve(0.0)
    @test rp["diagnostics"]["linear_failures"] == 0 && rp["diagnostics"]["eigen_failures"] == 0
    r = rp["r_grid"]; sel = r .<= RCMP
    A = project_mode(r[sel], (rp["n"][:, end] .- r0["n"][:, end])[sel], J0)
    B = project_mode(r[sel], (rp["nur"][:, end] .- r0["nur"][:, end])[sel], J1)
    return A, B, rp["t_grid"][end]
end

# ---- 2D ---------------------------------------------------------------------------
function mode_2d(; consistent_fm, terms, N = 160, L = 8.0)
    function solve(ε)
        g = H2.make_grid2d(N, N; xmax = L, ymax = L)
        m = H2.build_model_2d(; eos = EOS2, enable_diff = true, kappa_coeff = DST, consistent_fm, terms)
        U = H2.allocate_state(g, m)
        for ix in 1:g.Nxtot, iy in 1:g.Nytot
            r = hypot(g.xC[ix], g.yC[iy])
            H2.set_cell!(U, H2.lin(g, ix, iy), T0, αpert(r, ε), 0.0, 0.0, TAU0, m)
        end
        H2.finalize_ic!(U, g, m; τ0 = TAU0)
        res = H2.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAU1, CFLτ = 0.01)
        @test res.ok
        return H2.fields_2d(g, U, res.work, m)
    end
    fp = solve(EPS); f0 = solve(0.0)
    rs = Float64[]; dn = Float64[]; nr = Float64[]
    for ix in axes(fp.T, 1), iy in axes(fp.T, 2)
        x = fp.x[ix]; y = fp.y[iy]; r = hypot(x, y)
        (r <= RCMP && r > 0) || continue
        push!(rs, r); push!(dn, fp.n[ix, iy] - f0.n[ix, iy])
        push!(nr, (x*(fp.nux[ix, iy] - f0.nux[ix, iy]) + y*(fp.nuy[ix, iy] - f0.nuy[ix, iy]))/r)
    end
    # a Cartesian sum is already area-weighted: project with weight 1, not r
    pr(v, b) = sum(v .* b.(rs)) / sum(b.(rs).^2)
    return pr(dn, J0), pr(nr, J1)
end

function main()
    configs = (
        ("shipped",      false, :default,               (expansion = false, dlnh = false)),
        ("consistent",   true,  :default,               (expansion = true,  dlnh = true)),
        ("no expansion", true,  (fm_expansion = false,), (expansion = false, dlnh = true)),
        ("homogeneous",  true,  :homogeneous,            (expansion = false, dlnh = false)),
    )
    @testset "X1 — the diffusion mode, three solvers" begin
        @printf("  background: T0 = %.2f GeV, α0 = %.1f, n̄0 = %.3f fm⁻³, k = %.3f fm⁻¹, τ %.1f → %.1f\n",
                T0, A0L, NBAR0, K, TAU0, TAU1)
        results = Dict{String,Any}()
        for (label, cfm, terms, rkw) in configs
            _, Aref, Bref = referee(; rkw...)
            A1, B1 = mode_1d(; consistent_fm = cfm, terms)
            AI, BI, τI = mode_is2(; consistent_fm = cfm, terms)
            A2, B2 = mode_2d(; consistent_fm = cfm, terms)
            a = Aref(TAU1); b = Bref(TAU1)
            @printf("  %-13s referee A=%.5e B=%+.5e | rel.err A: 1D %+.1e IS2 %+.1e 2D %+.1e | B: 1D %+.1e IS2 %+.1e 2D %+.1e\n",
                    label, a, b, A1/a - 1, AI/Aref(τI) - 1, A2/a - 1, B1/b - 1, BI/Bref(τI) - 1, B2/b - 1)
            results[label] = (A1, B1, AI, BI, A2, B2)
            # 1D and IS2 at dr = 0.025 fm; 2D at dx = 0.1 fm, i.e. 4× coarser. The error is
            # first order in the cell size (the MC-limited ∇α clips at the J₀ extrema), which
            # is what (c) below checks — the bounds are that resolution, not a tolerance.
            @test abs(A1/a - 1) < 5e-3;  @test abs(AI/Aref(τI) - 1) < 5e-3;  @test abs(A2/a - 1) < 2e-2
            @test abs(B1/b - 1) < 2e-2;  @test abs(BI/Bref(τI) - 1) < 2e-2;  @test abs(B2/b - 1) < 6e-2
        end
        # (c) the 2D error is resolution: halving dx must shrink it
        _, Aref, Bref = referee(; expansion = true, dlnh = true)
        A2f, B2f = mode_2d(; consistent_fm = true, terms = :default, N = 320)
        co = results["consistent"]
        eA = (abs(co[5]/Aref(TAU1) - 1), abs(A2f/Aref(TAU1) - 1))
        eB = (abs(co[6]/Bref(TAU1) - 1), abs(B2f/Bref(TAU1) - 1))
        @printf("  2D, consistent, dx 0.1 → 0.05 fm:  |err A| %.1e → %.1e,  |err B| %.1e → %.1e\n", eA..., eB...)
        # B's error is spatial and halves; A also carries the operator split's first-order
        # TIME error (≈2e-3 here, the same floor the 1D solver shows at 4× finer dr; IS2,
        # which is unsplit RK4, sits at 1e-4), so it shrinks less
        @test eB[2] < 0.6*eB[1]
        @test eA[2] < 0.8*eA[1]
        # the consistent terms are not a small correction on this state: they must move B
        # by far more than the solvers disagree with the referee
        sh = results["shipped"]; co = results["consistent"]
        @printf("  consistent − shipped moves B by %.1f %% (1D) — the terms act, with the derived sign\n",
                100*(co[2]/sh[2] - 1))
        @test abs(co[2]/sh[2] - 1) > 0.1
        # :homogeneous is the shipped row, in every solver
        ho = results["homogeneous"]
        @test ho[1] == sh[1] && ho[2] == sh[2]          # 1D: bit for bit
        @test ho[5] == sh[5] && ho[6] == sh[6]          # 2D: bit for bit
        @test isapprox(ho[3], sh[3]; rtol = 1e-12) && isapprox(ho[4], sh[4]; rtol = 1e-12)   # IS2
    end
end
main()
