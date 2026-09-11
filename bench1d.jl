#!/usr/bin/env julia
# ==============================================================================
# bench1d.jl — what the 1+1D solvers cost, per sector. The 1-D twin of bench2d.jl.
#
#   julia -t 8 --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/bench1d.jl
#
# Reports ns per cell-step (wall time / (cells × steps)), best of three after a
# warm-up run (the first run carries the compilation), for:
#   * the bulk solver on the production-like IC, sector by sector, at two resolutions;
#   * the charm IS2 solver on an analytic background, shipped vs consistent closures,
#     and its allocation per step (README.md "Known open items" #4 quoted ~41 MB/step).
# A throughput number is meaningless on a loaded machine: run it alone.
# ==============================================================================
using Printf

const _ROOT = @__DIR__
include(joinpath(_ROOT, "main.jl"));     using .hydro;             const H = hydro
include(joinpath(_ROOT, "main2IS2.jl")); using .hydro_current_IS2; const HI = hydro_current_IS2

Tof(r) = 0.05 + 0.42*exp(-r^2/(2*3.2^2))
αof(r) = -4.0 + 1.2*exp(-r^2/(2*2.5^2))

function bulk_case(Nr; τf = 2.0, kw...)
    g = H.make_grid_1d(Nr; rmax = 15.0)
    m = H.build_model_1d(; kw...)
    U = H.allocate_state(g, m)
    H.initialize_from_radial!(U, g, m, 0.4, Tof, αof)
    t = @elapsed res = H.run_sim_1d!(U, g, m; τ0 = 0.4, τfinal = τf)
    res.ok || error("bench run failed")
    return t, res.nsteps
end

best3(f) = minimum(first(f()) for _ in 1:3)

function main()
    @printf("FiVo 1+1D cost — %d threads, %s\n\n", Threads.nthreads(), gethostname())
    visc = (enable_shear = true, eta_over_s = 0.1, enable_bulk = true, zeta_over_s = 0.1)
    diff = (enable_diff = true, kappa_coeff = 0.1163)
    configs = [("ideal", NamedTuple()), ("+ shear", (enable_shear = true, eta_over_s = 0.1)),
               ("+ bulk", (enable_bulk = true, zeta_over_s = 0.1)), ("+ diffusion", diff),
               ("all medium sectors + diffusion", (; visc..., diff...)),
               ("  + consistent_fm", (; visc..., diff..., consistent_fm = true))]
    bulk_case(100; τf = 0.6)                                   # warm-up (compilation)
    for Nr in (400, 800)
        @printf("bulk solver, Nr = %d, τ 0.4 → 2 fm\n", Nr)
        base = NaN
        for (lbl, kw) in configs
            ns = Ref(0)
            tbest = minimum(begin
                tt, n = bulk_case(Nr; kw...); ns[] = n; tt
            end for _ in 1:3)
            cs = 1e9*tbest/(Nr*ns[])
            isnan(base) && (base = cs)
            @printf("  %-34s %5d steps  %7.2f s  %7.0f ns/cell-step  (%+4.0f %% vs ideal)\n",
                    lbl, ns[], tbest, cs, 100*(cs/base - 1))
        end
        println()
    end

    # the charm IS2 solver on an analytic Bjorken-like background
    bg = HI.analytic_background(; T = (τ, r) -> (0.08 + 0.35*exp(-r^2/18))*(0.6/τ)^(1/3),
                                  ur = (τ, r) -> 0.1*r*(τ - 0.5)/(1 + 0.04r^2),
                                  r_grid = collect(0.0:0.05:16.0), t_grid = collect(0.5:0.01:4.0))
    is2(Nr; kw...) = HI.run_static_IS2_test(; background = bg, DsT = 0.1163, τ0 = 0.6, τfinal = 1.6, Nr,
                                            rmax = 12.0, dump_dt = 1.0, init_mode = :n_profile,
                                            n_profile = r -> 0.05*exp(-r^2/10), log_every = 10^9, kw...)
    is2(60)                                                    # warm-up
    for Nr in (300, 600)
        @printf("charm IS2, Nr = %d, τ 0.6 → 1.6 fm\n", Nr)
        for (lbl, kw) in (("shipped", NamedTuple()), ("consistent fm", (consistent_fm = true,)),
                          ("consistent fm + m2", (consistent_fm = true, consistent_m2 = true)))
            local res
            t = minimum(begin
                s = @timed (res = is2(Nr; kw...)); s.time
            end for _ in 1:3)
            al = @allocated is2(Nr; kw...)
            ns = res["diagnostics"]["steps"]
            @printf("  %-22s %5d steps  %6.2f s  %8.0f ns/cell-step  %7.1f MB/step allocated\n",
                    lbl, ns, t, 1e9*t/(Nr*ns), al/ns/2^20)
        end
        println()
    end
end
main()
