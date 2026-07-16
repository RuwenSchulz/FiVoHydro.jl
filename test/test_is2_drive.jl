#!/usr/bin/env julia
# test_is2_drive.jl — REGRESSION TEST for the IS2 diffusion-current drive.
#
# Guards the path-conservative intra-cell drive term in `_compute_dUdt!` (main2IS2.jl): the HLL
# face fluctuations use B·(reconstructed face jump), which VANISHES for smooth fields, so without
# the explicit intra-cell B·∂_rU term the diffusion drive κ·∂_rα on the ν^r row is lost and ν^r is
# driven to ≈0 instead of its Navier–Stokes value.
#
# On a static, flow-free, constant-T background (u^r=0 ⟹ no source coupling, no flow shift), the
# IS2 ν^r equation reduces to  τ_n·∂_τν^r + ν^r = -κ·∂_rα,  whose attractor is the first-order NS
# current  ν^r_NS = -κ·∂_rα  with  κ = D_sT·n/(T·fmGeV).  We evolve a smooth Gaussian charm blob and
# require the realized ν^r/n to track -(D_sT/(T·fmGeV))·∂_rα across the bulk (it is ≈0 if the drive
# term is missing — the regression this test catches).
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_is2_drive.jl

using Test, JLD2, Dierckx, Printf
include(joinpath(@__DIR__, "..", "main2IS2.jl"))
using .hydro_current_IS2

const FMGEV = 1.0 / 0.19733   # GeV·fm
const DST   = 0.1163
const TBG   = 0.30            # constant background temperature [GeV]

# ── build a synthetic constant-T, zero-flow background file ───────────────────
function make_bg(path)
    rg = collect(range(0.0, 15.0; length=160))
    tg = collect(range(0.4,  3.0; length=40))
    Tgrid  = fill(TBG, length(rg), length(tg))
    urgrid = zeros(length(rg), length(tg))
    Tspl  = Spline2D(rg, tg, Tgrid;  kx=3, ky=3)
    urspl = Spline2D(rg, tg, urgrid; kx=1, ky=1)
    jldsave(path; r_grid=rg, t_grid=tg, T_spline=Tspl, ur_spline=urspl)
end

@testset "IS2 diffusion-current drive (ν^r → -κ∂_rα)" begin
    bg = tempname() * ".jld2"; make_bg(bg)
    res = hydro_current_IS2.run_static_IS2_test(;
        background_file=bg, DsT=DST, τ0=0.4, τfinal=2.0, Nr=300, rmax=15.0,
        CFL=0.15, CFLτ=0.03, dump_dt=0.2, T_floor=0.05, init_mode=:auto,
        n_profile = r -> exp(-r^2 / (2*3.0^2)), use_cM=false,
        eos = hydro_current_IS2.LatticeHRGEOS(canon_factor=1.0))

    r = Float64.(res["r_grid"]); t = Float64.(res["t_grid"])
    α = Array{Float64}(res["alpha"]); n = Array{Float64}(res["n"]); nur = Array{Float64}(res["nur"])
    jt = findlast(<=(1.5), t)                              # a late, relaxed snapshot
    αspl = Spline2D(r, t, α; kx=3, ky=1)
    drα(rr) = (Float64(αspl(rr+0.02, t[jt])) - Float64(αspl(rr-0.02, t[jt]))) / 0.04
    ridx(rr) = argmin(abs.(r .- rr))

    @printf("\n  %5s | %12s %12s %9s\n", "r", "ν^r/n", "NS=-κ∂α/n", "ratio")
    maxabs = 0.0
    for rr in (1.5, 2.0, 3.0, 4.0, 5.0)
        i = ridx(rr)
        ratio_fv = nur[i, jt] / n[i, jt]
        ns       = -(DST / (TBG * FMGEV)) * drα(rr)        # ν^r_NS / n
        rel      = ratio_fv / ns
        maxabs   = max(maxabs, abs(ratio_fv))
        @printf("  %5.1f | %12.6f %12.6f %9.3f\n", rr, ratio_fv, ns, rel)
        @test 0.6 < rel < 1.4                              # ν^r tracks NS (≈0 ⟹ drive lost ⟹ FAIL)
    end
    @test maxabs > 0.01                                    # hard floor: not the broken ν^r≈0 state
    rm(bg; force=true)
end
