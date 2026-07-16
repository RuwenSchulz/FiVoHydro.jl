using Test

# Load the solver module as currently defined by the repo entrypoint.
include(joinpath(@__DIR__, "..", "main.jl"))
using .hydro

# Load the current-only 2nd-moment module for targeted identity tests — only if present.
# (main3.jl was removed from the repo; guard the include so the suite still runs.)
const _MAIN3 = joinpath(@__DIR__, "..", "main3.jl")
const HAVE_2ND_MOMENT = isfile(_MAIN3)
if HAVE_2ND_MOMENT
    include(_MAIN3)
    using .hydro_current_2nd_moment
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
        end
        @test true
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
        end
        @test true
    end

    if !HAVE_2ND_MOMENT
        @testset "Current-only 2nd-moment identities (skipped: main3.jl absent)" begin
            @test_skip true
        end
    else
    @testset "Current-only 2nd-moment identities" begin
        # Build a tiny grid in the current-only module.
        grid_full = hydro_current_2nd_moment.make_grid(33; rmax=2.0, nghost=1)
        grid = hydro_current_2nd_moment.CurrentGrid1D(grid_full)

        # Minimal callable splines (functions) are sufficient for the module.
        r_grid = collect(range(0.0, 2.0; length=4))
        t_grid = collect(range(0.0, 2.0; length=4))

        # Static background with ur=0 => v=0.
        T_spl  = (r, t) -> 0.25
        ur_spl = (r, t) -> 0.0
        α_spl  = (r, t) -> 0.1
        bg = hydro_current_2nd_moment.BackgroundFields(
            r_grid, t_grid,
            T_spl,
            ur_spl,
            nothing,
            α_spl,
            nothing,
            nothing,
            (r, t) -> 0.0,
        )

        τ = 1.0
        q = @. grid.r * exp(-grid.r)
        nu_r = @. 0.01 * sin(grid.r)
        Phi = zeros(length(grid.r))

        snap = hydro_current_2nd_moment.reconstruct_snapshot(q, nu_r, Phi, τ, grid, bg;
            DsT=0.24,
            T_floor=1e-6,
            eos=hydro_current_2nd_moment.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0),
        )

        @testset "Jr identity" begin
            @test all(isfinite, snap.Jtau)
            @test all(isfinite, snap.Jr)
            @test all(isfinite, snap.u_tau)
            @test all(isfinite, snap.nu_r)
            # With ur=0 => v=0 and u_tau=1, so Jr must equal nu_r/u_tau^2.
            @test maximum(abs.(snap.Jr .- snap.nu_r ./ (snap.u_tau .^ 2))) < 1e-12
        end

        @testset "q conservation geometry" begin
            # With ur=0 and DsT=0 => nu_r is driven to 0 and Jr=0,
            # so the update reduces to q_new = q - dt*(q/τ).
            ws = hydro_current_2nd_moment.CurrentWorkspace1D(grid)
            q2 = copy(q)
            nu2 = zeros(length(q2))
            Phi2 = zeros(length(q2))
            dt = 0.1

            hydro_current_2nd_moment.step_current_imex!(q2, nu2, Phi2, τ, dt, grid, ws, bg;
                DsT=0.0,
                T_floor=1e-6,
                eos=hydro_current_2nd_moment.ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0),
            )
            @test maximum(abs.(q2 .- (q .* (1 .- dt/τ)))) < 1e-12
        end
    end
    end  # if HAVE_2ND_MOMENT

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
                end
                @test true
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
                  "test_is2_causality.jl"]
    long && push!(regression, "test_is2_stability.jl")   # heavier fresh IS2 solve

    @testset "charm-solver regression (subprocess): $script" for script in regression
        path = joinpath(@__DIR__, script)
        @test isfile(path)
        ok = success(run(ignorestatus(`$JLBIN --project=$PROJ $path`)))
        @test ok
    end
end
