using Test

# Load the solver module as currently defined by the repo entrypoint.
include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

"""Every snapshot_tau_*.csv in `outdir` parses and its T, e, ur columns are finite (T > 0)."""
function _snapshots_finite(outdir)
    files = filter(f -> startswith(f, "snapshot_tau_") && endswith(f, ".csv") && !endswith(f, "_meta.csv"), readdir(outdir))
    isempty(files) && return false
    for f in files
        lines = readlines(joinpath(outdir, f)); length(lines) >= 2 || return false
        hdr = split(strip(lines[1]), ','); iT = findfirst(==("T"), hdr); ie = findfirst(==("e"), hdr)
        (iT === nothing || ie === nothing) && return false
        for ln in lines[2:end]
            v = split(ln, ','); T = parse(Float64, v[iT]); e = parse(Float64, v[ie])
            (isfinite(T) && T > 0 && isfinite(e) && e >= 0) || return false
        end
    end
    return true
end

@testset "FiVoHydro smoke" begin
    @testset "EOS finite" begin
        eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        for (T, μ) in ((0.2, 0.0), (0.35, 0.1), (0.6, 0.2))
            P, n, e = hydro.eos_Pne(T, μ, eos)
            @test isfinite(P)
            @test isfinite(n)
            @test isfinite(e)
            @test P >= 0
            @test e >= 0
        end
    end

    @testset "Primitive recovery basic" begin
        eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])
        model  = hydro.IdealDiffViscModel(
            eos, layout, hydro.IdealPrimRec(),
            # diffusion
            false, :alpha, 0.1, 1.0, 0.0, 0.02, 0.3, 0.3, -1.0,
            0.0, 0.0, 0.0, 0.0,
            true, false, 16, false, true,
            # viscosity
            false, false, hydro.ZeroViscosity(), hydro.ZeroBulkViscosity(),
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            -1.0, -1.0,
            false, false,
            # relax advect
            true, true
        )

        grid = hydro.make_grid(64; rmax=5.0, nghost=3)
        U = zeros(length(layout.names), grid.Nr + 2*grid.nghost)
        hydro.initialize!(U, grid, 0.4, model; init_csv=nothing)
        work = hydro.make_work(U)

        # Prime cache and ensure it doesn't throw.
        hydro.prime_work_from_U!(work, U, grid, 0.4, model)

        # Spot-check one representative cell.
        i = grid.nghost + 10
        D  = U[layout.iDtau, i] / 0.4
        Sr = U[layout.iSr, i]
        E  = U[layout.iE, i]
        wpr = model.primrec.work[1]
        T, μ, ur, n, e, P, ok = hydro.cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, 0.0, 0.0, 0.0, 0.0,
            grid.rC[i], 0.4, eos;
            yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
            maxit=50
        )
        @test ok
        @test isfinite(T) && T >= hydro.T_MIN
        @test isfinite(P) && P >= 0
        @test isfinite(e) && e >= 0
        @test isfinite(ur)
        @test isfinite(μ)
    end

    @testset "Primitive recovery hard states" begin
        eos = hydro.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0)
        layout = hydro.StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])
        model  = hydro.IdealDiffViscModel(
            eos, layout, hydro.IdealPrimRec(),
            # diffusion
            false, :alpha, 0.1, 1.0, 0.0, 0.02, 0.3, 0.3, -1.0,
            0.0, 0.0, 0.0, 0.0,
            true, false, 16, false, true,
            # viscosity
            false, false, hydro.ZeroViscosity(), hydro.ZeroBulkViscosity(),
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            -1.0, -1.0,
            false, false,
            # relax advect
            true, true
        )

        wpr = model.primrec.work[1]
        τ = 0.4
        r = 1.0

        # 1) round-trip: prim -> cons -> prim for a few extreme states
        # Keep within caps (PHI_CAP/Y_CAP), but include near-causal v and large μ.
        states = [
            (T=0.18, φ=0.0,   y=atanh(0.0)),
            (T=0.25, φ=5.0,   y=atanh(0.95)),
            (T=0.45, φ=-5.0,  y=atanh(-0.98)),
            (T=0.90, φ=15.0,  y=atanh(0.999)),
        ]

        Utmp = zeros(length(layout.names), 1)
        for st in states
            yT = log(st.T)
            okU, prim0 = hydro.prim_to_cons_col_ideal_phi_diff_visc!(
                Utmp, 1,
                yT, st.φ, st.y,
                0.0, 0.0, 0.0, 0.0,
                r, τ,
                eos, layout,
            )
            @test okU
            @test prim0.ok

            D = Utmp[layout.iDtau,1] / τ
            Sr = Utmp[layout.iSr,1]
            E = Utmp[layout.iE,1]

            T, μ, ur, n, e, P, ok = hydro.cons_to_prim_ideal_phi_diff_visc!(
                wpr, D, Sr, E, 0.0, 0.0, 0.0, 0.0,
                r, τ, eos;
                yT0=yT, φ0=st.φ, y0=st.y,
                maxit=120,
            )
            @test ok
            @test isfinite(T) && T >= hydro.T_MIN
            @test isfinite(μ)
            @test isfinite(ur)
            @test isfinite(P) && P >= 0
            @test isfinite(e) && e >= 0

            # Consistency: reconstructed conserved vars match within a loose tolerance.
            Uchk = zeros(length(layout.names), 1)
            yT1 = log(T)
            φ1 = (μ - hydro.hq_mass(eos)) / max(T, hydro.T_MIN)
            y1 = asinh(ur)
            okChk, _ = hydro.prim_to_cons_col_ideal_phi_diff_visc!(
                Uchk, 1,
                yT1, φ1, y1,
                0.0, 0.0, 0.0, 0.0,
                r, τ,
                eos, layout,
            )
            @test okChk
            @test isapprox(Uchk[layout.iDtau,1], Utmp[layout.iDtau,1]; rtol=1e-8, atol=1e-12)
            @test isapprox(Uchk[layout.iSr,1],   Utmp[layout.iSr,1];   rtol=1e-8, atol=1e-12)
            @test isapprox(Uchk[layout.iE,1],    Utmp[layout.iE,1];    rtol=1e-8, atol=1e-12)
        end

        # 2) deliberately inconsistent conservative state: large |Sr| at tiny E.
        # Should fail cleanly and record a reason.
        D = 0.1
        Sr = 10.0
        E  = 1e-6
        T, μ, ur, n, e, P, ok = hydro.cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, 0.0, 0.0, 0.0, 0.0,
            r, τ, eos;
            maxit=60,
        )
        @test !ok
        @test wpr.last_reason != hydro.PRR_UNSET
        @test isfinite(wpr.last_iters)
    end

    @testset "Short run does not throw" begin
        mktempdir() do outdir
            hydro.run_sim_ideal_diff_visc(
                outdir=outdir,
                Nr=64,
                rmax=5.0,
                τ0=0.4,
                τfinal=0.401,
                dump_dt=10.0,
                log_every=10_000,
                log_corrections_every=10_000,
                init_csv=nothing,
                enable_diff=false,
                enable_shear=false,
                enable_bulk=false,
                time_integrator=:ssprk2,
                postprocess=false,
            )
            @test _snapshots_finite(outdir)
        end
    end

    @testset "Short dissipative run does not throw" begin
        mktempdir() do outdir
            hydro.run_sim_ideal_diff_visc(
                outdir=outdir,
                Nr=64,
                rmax=5.0,
                τ0=0.4,
                # a couple steps, but still fast enough for default CI
                τfinal=0.405,
                dump_dt=10.0,
                log_every=10_000,
                log_corrections_every=10_000,
                init_csv=nothing,
                enable_diff=true,
                enable_shear=true,
                enable_bulk=true,
                eta_over_s=0.1,
                zeta_over_s=0.083,
                time_integrator=:ssprk2,
                postprocess=false,
            )
            @test _snapshots_finite(outdir)
        end
    end

    @testset "relaxation_laws.jl reference vs the solver's backward-Euler update" begin
        # src/relaxation_laws.jl is a REFERENCE (exact-exponential) integrator of the scalar MIS law
        #   τn uτ dν/dτ + (1 + δθ) ν = ν_NS ;
        # relax_dissipative! (src/dissipation.jl) discretises the SAME law with backward Euler,
        #   ν_new = (A ν_old + ν_NS) / (A + 1 + δθ),  A = τn uτ / Δ   (no advection/λNN/Dy terms here).
        # Neither calls the other; this test pins that they are two discretisations of one ODE.
        law = hydro.DefaultRelaxationLaw()
        νold, νNS, τn, δ, θ, uτ = 0.3, -0.05, 1.2, 0.4, 0.8, 1.3
        g = 1 + δ*θ
        νexact(Δ) = hydro.relaxation_update_nur_phys(law; νold_phys=νold, νNS_phys=νNS, adv_src=0.0,
                                                     Δ=Δ, τn=τn, δ=δ, θ=θ, uτ=uτ)
        νBE(Δ)    = ((τn*uτ/Δ)*νold + νNS) / ((τn*uτ/Δ) + g)
        # same fixed point
        @test isapprox(νexact(1e9), νNS/g; rtol=1e-9)
        @test isapprox(νBE(1e9),    νNS/g; rtol=1e-6)     # BE fixed-point error ∝ A = τn uτ/Δ ≈ 1.6e-9
        # first-order consistency: |BE − exact| = O(Δ²) per step ⇒ ratio of errors ≈ 4 when Δ halves
        e(Δ) = abs(νBE(Δ) - νexact(Δ))
        @test 3.5 < e(0.02)/e(0.01) < 4.5
        @test e(0.01) < 1e-4
        # the bulk hook is the exact exponential of τΠ uτ dΠ + Π = Π_NS
        Πn = hydro.relaxation_update_Pi_phys(law; Πold_phys=0.2, ΠNS_phys=-0.1, adv_src=0.0, Δ=0.5, τΠ=1.0, θ=0.0, uτ=1.0)
        @test isapprox(Πn, -0.1 + 0.3*exp(-0.5); rtol=1e-12)
        # the shear hook relaxes both mixed components with one rate
        πφ, πη = hydro.relaxation_update_pi_phi_eta_phys(law; πφ_old_phys=0.1, πη_old_phys=-0.2,
                    πφNS_phys=0.0, πηNS_phys=0.0, adv_πφ=0.0, adv_πη=0.0, Δ=0.3, τπ=0.6, δπ=0.0, θ=0.0, uτ=1.0)
        @test isapprox(πφ, 0.1*exp(-0.5); rtol=1e-12) && isapprox(πη, -0.2*exp(-0.5); rtol=1e-12)
    end

    # Optional longer run (e.g. local stress testing):
    #   FIVOHYDRO_LONG_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
    let long = lowercase(get(ENV, "FIVOHYDRO_LONG_TESTS", "0")) in ("1", "true", "yes", "y")
        if long
            @testset "Longer dissipative run does not throw" begin
                mktempdir() do outdir
                    hydro.run_sim_ideal_diff_visc(
                        outdir=outdir,
                        Nr=64,
                        rmax=5.0,
                        τ0=0.4,
                        τfinal=0.6,
                        dump_dt=10.0,
                        log_every=10_000,
                        log_corrections_every=10_000,
                        init_csv=nothing,
                        enable_diff=true,
                        enable_shear=true,
                        enable_bulk=true,
                        eta_over_s=0.1,
                        zeta_over_s=0.083,
                        time_integrator=:ssprk2,
                    )
                    @test _snapshots_finite(outdir)
                end
            end
        end
    end
end

# EOS thermodynamic-consistency unit tests (in-process; the `hydro` module is already loaded above).
include(joinpath(@__DIR__, "test_eos_consistency.jl"))

# Charm-solver regression tests run as ISOLATED subprocesses: each (re)builds its own solver
# module (hydro_current_IS2 / density-frame flux via main2IS2.jl / main.jl), which would clash
# with the in-process `hydro` module if `include`d here. Subprocesses keep them independent.
let JLBIN = joinpath(Sys.BINDIR, Base.julia_exename()),
    PROJ  = normpath(joinpath(@__DIR__, "..")),     # FiVoHydro.jl package dir (full dep tree)
    long  = lowercase(get(ENV, "FIVOHYDRO_LONG_TESTS", "0")) in ("1", "true", "yes", "y")

    regression = ["test_is2_drive.jl", "test_density_frame_flux.jl", "test_bdnk_causal.jl",
                  "test_bdnk_frame_coeffs.jl",   # BDNK general-frame σ_T/σ_a; needs u^r ≠ 0 AND ∂_rT ≠ 0
                  "test_is2_causality.jl", "test_m1_gates.jl",   # M1 validation ladder (9 gates; G5 skips if the LP1 bundle is absent)
                  # the 1+1D analytic ladder's fast tier (test/run1d_gates.jl), 2026-09-11:
                  "test_terms1d.jl",            # T  — the shared term switches in the 1-D solvers (~5 s)
                  "test_gubser_viscous1d.jl"]   # A4 — viscous Gubser vs its semi-analytic ODE (~10 s)
    long && append!(regression, ["test_is2_stability.jl",   # heavier fresh IS2 solve
                                 "test_bjorken1d.jl",       # A1/A2 — Bjorken ideal + the full DNMR set (~3 min)
                                 "test_diffusion_mode.jl"]) # X1 — the diffusion mode, three solvers (~1.5 min)

    @testset "charm-solver regression (subprocess): $script" for script in regression
        path = joinpath(@__DIR__, script)
        @test isfile(path)
        ok = success(run(ignorestatus(`$JLBIN --project=$PROJ $path`)))
        @test ok
    end
end
