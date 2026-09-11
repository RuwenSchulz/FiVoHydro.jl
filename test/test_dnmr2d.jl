# ==============================================================================
# test/test_dnmr2d.jl — gate Gd: the medium's second-order (DNMR) couplings in 2+1D.
#
# Until 2026-09-11 the 2-D solver REFUSED τ_ππ, λ_πΠ, δ_ΠΠ and λ_Ππ (`_UNWIRED_KNOBS_2D`):
# the knobs existed for parity with the 1-D model, and nothing read them. They are now
# wired (src2d/dissipation2d.jl) with the same equations as the 1-D solver, which had
# its own signs corrected the same day (EQUATIONS1D.md §2–3):
#
#     τ_π Δ Dπ + π = −2ησ − δ_ππθπ − τ_ππ π^{λ⟨i}σ^{j⟩}_λ − λ_πΠ Π σ     (mostly plus)
#     τ_Π DΠ + Π   = −ζθ − δ_ΠΠ θΠ − λ_Ππ π:σ
#
# Gd1  a transversely uniform 2-D state is Bjorken: every configuration against the
#      0+1D DNMR ODEs (analytic_referees.jl `bjorken_dnmr`, the referee gate A2 uses
#      for 1-D), at three steps — first order, Richardson-extrapolated error < 1e-3 —
#      and a wrong-sign referee must miss.
# Gd2  the SAME uniform state through the 1-D solver: 1-D and 2-D agree at each step
#      (they integrate the same split scheme in 0+1D).
# Gd3  the contractions: π:σ and the ⟨⟩-projected π·σ used by the medium are the
#      charm second moment's (gated there against the 1-D reduction, Gm1) — the shared
#      helper `pi_sigma_contractions_2d` is checked against brute-force index algebra
#      with the metric at random flowing states.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_dnmr2d.jl
# ==============================================================================

using Printf
using Test

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
include(joinpath(@__DIR__, "analytic_referees.jl"))
using .hydro, .hydro2d
const H1 = hydro; const H2 = hydro2d

const T0, α0, TAU0, TAU1 = 0.45, -20.0, 0.6, 3.0

function run2d(kw; CFLτ)
    g = H2.make_grid2d(12, 12; xmax = 6.0, ymax = 6.0)
    m = H2.build_model_2d(; eos = H2.LatticeHRGEOS(), kw...)
    U = H2.allocate_state(g, m)
    H2.initialize_uniform!(U, g, m, TAU0; T0, alpha0 = α0)
    res = H2.run_sim_2d!(U, g, m; τ0 = TAU0, τfinal = TAU1, CFL = 10.0, CFLτ)
    @test res.ok
    f = H2.fields_2d(g, U, res.work, m)
    return (e = f.e[6, 6], φ = haskey(f, :pieta) ? -f.pieta[6, 6] : 0.0, Π = haskey(f, :Pi) ? f.Pi[6, 6] : 0.0)
end

function run1d(kw; CFLτ)
    g = H1.make_grid_1d(40; rmax = 10.0)
    m = H1.build_model_1d(; eos = H1.LatticeHRGEOS(), kw...)
    U = H1.allocate_state(g, m)
    H1.initialize_uniform!(U, g, m, TAU0; T0, alpha0 = α0)
    res = H1.run_sim_1d!(U, g, m; τ0 = TAU0, τfinal = TAU1, CFL = 10.0, CFLτ)
    f = H1.fields_1d(g, U, m; τ = res.τ, work = res.work)
    return (e = f.e[20], φ = -f.piEta[20], Π = f.Pi[20])
end

function referee(m, kr)
    eos = m.eos
    th(T) = (P = H1.eos_Pne(T, α0*T, eos); H1.local_thermo(T, α0*T, P[2], P[3], P[1], eos))
    eos_eP(T) = (P = H1.eos_Pne(T, α0*T, eos); (P[3], P[1]))
    Tof_e(e) = (lo = 0.01; hi = 2.0; for _ in 1:100; mid = (lo+hi)/2; eos_eP(mid)[1] < e ? (lo = mid) : (hi = mid); end; (lo+hi)/2)
    _, eref, φref, Πref = bjorken_dnmr(; eos_eP, Tof_e,
        η = T -> m.enable_shear ? H1.viscosity(T, th(T), m.shear) : 0.0,
        τπ = T -> m.enable_shear ? H1.τ_shear(T, th(T), m.shear) : 0.0,
        ζ = T -> m.enable_bulk ? H1.bulk_viscosity(T, th(T), m.bulk) : 0.0,
        τΠ = T -> m.enable_bulk ? H1.τ_bulk(T, th(T), m.bulk) : 1.0,
        e0 = eos_eP(T0)[1], τ0 = TAU0, τ1 = TAU1, kr...)
    return eref(TAU1), φref(TAU1), Πref(TAU1)
end

function gate_Gd1_Gd2()
    println("\nGd1/Gd2 — the DNMR couplings on 2-D Bjorken, vs the 0+1D ODEs and vs the 1-D solver")
    sh = (enable_shear = true, eta_over_s = 0.2, tauShear_coeff = 0.2)
    bu = (enable_bulk = true, zeta_over_s = 0.1, tauPi_coeff = 15.0)
    cases = [
        ("shear, δ_ππ = 4/3, τ_ππ = 10/7", (; sh..., deltaShear_factor = 4/3, taupi_pi_factor = 10/7), (δππ = 4/3, τππ = 10/7), (δππ = 4/3, τππ = -10/7), :φ),
        ("bulk, δ_ΠΠ = 2/3",              (; bu..., deltaPi_factor = 2/3), (δΠΠ = 2/3,), (δΠΠ = -2/3,), :Π),
        ("shear+bulk, λ_πΠ = 6/5",        (; sh..., bu..., deltaShear_factor = 4/3, lambda_pi_Pi_factor = 6/5),
                                          (δππ = 4/3, λπΠ = 6/5), (δππ = 4/3, λπΠ = -6/5), :φ),
        ("shear+bulk, λ_Ππ = 1.2",        (; sh..., bu..., deltaShear_factor = 4/3, lambda_Pi_pi_factor = 1.2),
                                          (δππ = 4/3, λΠπ = 1.2), (δππ = 4/3, λΠπ = -1.2), :Π),
    ]
    for (label, km, kr, kbad, key) in cases
        m1 = H1.build_model_1d(; eos = H1.LatticeHRGEOS(), km...)
        e, φ, Π = referee(m1, kr); eb, φb, Πb = referee(m1, kbad)
        r2 = [run2d(km; CFLτ = c) for c in (0.02, 0.01, 0.005)]
        r1 = [run1d(km; CFLτ = c) for c in (0.01, 0.005)]
        ext = (e = 2r2[3].e - r2[2].e, φ = 2r2[3].φ - r2[2].φ, Π = 2r2[3].Π - r2[2].Π)
        rel(x, y, on) = on ? x/y - 1 : 0.0
        eext = (rel(ext.e, e, true), rel(ext.φ, φ, m1.enable_shear), rel(ext.Π, Π, m1.enable_bulk))
        ebad = key === :φ ? ext.φ/φb - 1 : ext.Π/Πb - 1
        p = log2(maximum(abs, (rel(r2[2].e, e, true), rel(r2[2].φ, φ, m1.enable_shear), rel(r2[2].Π, Π, m1.enable_bulk))) /
                 maximum(abs, (rel(r2[3].e, e, true), rel(r2[3].φ, φ, m1.enable_shear), rel(r2[3].Π, Π, m1.enable_bulk))))
        d12 = maximum(abs, (r2[3].e/r1[2].e - 1, m1.enable_shear ? r2[3].φ/r1[2].φ - 1 : 0.0,
                            m1.enable_bulk ? r2[3].Π/r1[2].Π - 1 : 0.0))
        @printf("  %-32s 2D extrapolated (e, φ, Π) %+.1e %+.1e %+.1e | order %.2f | wrong-sign referee %+.1e | 2D vs 1D %.1e\n",
                label, eext..., p, ebad, d12)
        @test maximum(abs, eext) < 1e-3
        @test p > 0.7
        @test abs(ebad) > 10*abs(key === :φ ? eext[2] : eext[3])
        @test d12 < 5e-2
    end
end

function gate_Gd3()
    println("\nGd3 — the π·σ contractions vs brute-force index algebra with the metric")
    s = UInt64(7); u() = (s = s*0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(s >> 11)*2.0^-53 - 0.5)
    worst = 0.0
    for _ in 1:300
        ux, uy = 1.2u(), 1.2u(); uτ = sqrt(1 + ux^2 + uy^2); τ = 0.6 + 2(u() + 0.5)
        pxx, pxy, pyy = 0.2u(), 0.2u(), 0.2u()
        pe, _ = H2.project_shear_traceless_2d(ux, uy, uτ, pxx, pxy, pyy, 0.0)
        # a traceless, u-orthogonal σ from consistent random kinematics
        dtux, dtuy, dxux, dxuy, dyux, dyuy = 0.3u(), 0.3u(), 0.3u(), 0.3u(), 0.3u(), 0.3u()
        ax = uτ*dtux + ux*dxux + uy*dyux; ay = uτ*dtuy + ux*dxuy + uy*dyuy
        θ = (ux*dtux + uy*dtuy)/uτ + dxux + dyuy + uτ/τ
        σxx, σxy, σyy, ση = H2.ns_shear_target_2d(ux, uy, uτ, τ, θ, ax, ay, dxux, dxuy, dyux, dyuy, -0.5)
        πσ, cxx, cxy, cyy = H2.pi_sigma_contractions_2d(ux, uy, pxx, pxy, pyy, pe, σxx, σxy, σyy, ση)
        # brute force on the (τ, x, y) block, g = diag(−1, 1, 1); τ-rows from orthogonality;
        # the η-η pieces enter π:σ as π^η_η σ^η_η
        P = H2.shear_tensor_contravariant_2d(ux, uy, uτ, τ, pxx, pxy, pyy, pe)
        Π3 = [P.tt P.tx P.ty; P.tx P.xx P.xy; P.ty P.xy P.yy]
        stx = (ux*σxx + uy*σxy)/uτ; sty = (ux*σxy + uy*σyy)/uτ; stt = (ux*stx + uy*sty)/uτ
        Σ3 = [stt stx sty; stx σxx σxy; sty σxy σyy]
        G = [-1.0 0 0; 0 1 0; 0 0 1]
        πσ_bf = sum(Π3 .* (G*Σ3*G)) + pe*ση
        C = Π3*G*Σ3; C = (C + C')/2                   # π^{(i}_λ σ^{j)λ}
        worst = max(worst, abs(πσ - πσ_bf)/(abs(πσ_bf) + 1e-3), abs(cxx - C[2,2])/(abs(C[2,2]) + 1e-3),
                    abs(cxy - C[2,3])/(abs(C[2,3]) + 1e-3), abs(cyy - C[3,3])/(abs(C[3,3]) + 1e-3))
    end
    @printf("  π:σ, c_xx, c_xy, c_yy over 300 random flowing states: worst rel %.1e\n", worst)
    @test worst < 1e-12
end

function main()
    @testset "Gd — DNMR couplings (2+1D)" begin
        gate_Gd1_Gd2(); gate_Gd3()
    end
end
main()
