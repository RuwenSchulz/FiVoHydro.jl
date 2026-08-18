#!/usr/bin/env julia
# ═════════════════════════════════════════════════════════════════════════════════════════════════
# test_m1_gates.jl — the validation ladder for main2M1.jl, in order.
#
# Steps 1-4 use SYNTHETIC backgrounds built from closures (M1Background's spline fields are `Any` and
# only ever called as spl(r, τ)), so they need no data products and test the formulation alone.
# Step 5 runs on the shipped LP1 background and step 6 turns the regulators off — except there are no
# regulators to turn off, which is the point of the exercise.
#
#   G1  static uniform background, no gradient  ⇒  NOTHING moves (the fixed point)
#   G2  charm number is conserved to round-off
#   G3  the Navier-Stokes limit reproduces D_s = D_sT/T
#   G4  the boosted IC is recovered exactly: (n, ν^r) in ⇒ (n, ν^r) out
#   G5  the production LP1 background runs, with NO cell ever leaving the realizability cone
#   G6  strong-gradient stress test: |w| stays inside the cone with no clamp anywhere
#
# Run: julia --project=Julia Julia/FiVoHydro.jl/test_m1_gates.jl
# ═════════════════════════════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "main2M1.jl"))
using .hydro_current_M1
using .hydro_current_M1: M1Background, M1Grid1D, solve_M1, recover, source_moments,
                         eos_Pne, LatticeHRGEOS,
                         build_source_tables!, _build_h_table!, eta_drag, hpp
using Printf

const RES = Tuple{String,Bool,String}[]
check(n, ok, d="") = (push!(RES, (n, ok, String(d))); ok)

const T0BG  = 0.30                      # GeV, hot enough that the whole grid is fluid
const MASS  = 1.5
const RG    = collect(range(0.0, 30.0; length = 2))
const TG    = collect(range(0.1, 40.0; length = 2))

"""Synthetic background: uniform temperature `T`, radial flow `ur(r)`, charm fugacity `α(r)`."""
mk_bg(; T = T0BG, ur = (r, τ) -> 0.0, α = (r, τ) -> 0.0) =
    M1Background(RG, TG, (r, τ) -> T, ur, α, nothing, nothing)

"""⟨r²⟩ over the Milne measure τ r dr, from a snapshot row of `n`."""
function r2_of(r, n)
    num = sum(i -> r[i]^2 * n[i] * r[i], eachindex(r))
    den = sum(i -> n[i] * r[i], eachindex(r))
    num/max(den, 1e-300)
end

println("═"^96); println("main2M1.jl — validation ladder"); println("═"^96)

# ── G1 : the fixed point ────────────────────────────────────────────────────────────────────────
# Uniform α, no flow, uniform T.  The only thing left in the equations is the Milne 1/τ dilution, so
# n must fall exactly as 1/τ and w must stay identically zero.  Anything else is a bug in the
# geometry source terms, which is precisely where a Milne port goes wrong.
let
    bg = mk_bg()
    grid = M1Grid1D(200, 20.0)
    res = solve_M1(grid, zeros(200), zeros(200), 1.0, 5.0, bg;
                   CFL = 0.25, save_dt = 4.0, DsT = 0.1163, log_every = 0)
    n = res["n"]; w = res["w"]; τ = res["t_grid"]
    core = 20:150                                        # away from both boundaries
    ratio = n[core, end] ./ n[core, 1]
    expect = τ[1]/τ[end]
    dev = maximum(abs.(ratio ./ expect .- 1))
    wmax = maximum(abs, w[core, end])
    @printf("\nG1  uniform background:  n(τ_f)/n(τ_0) = %.6f  (1/τ expects %.6f),  max|w| = %.2e\n",
            ratio[length(ratio) ÷ 2], expect, wmax)
    check("G1a uniform state dilutes exactly as 1/τ", dev < 2e-3, @sprintf("max dev %.2e", dev))
    check("G1b no current is generated without a gradient", wmax < 1e-10, @sprintf("max|w| %.2e", wmax))
end

# ── G2 : conservation ───────────────────────────────────────────────────────────────────────────
let
    bg = mk_bg(α = (r, τ) -> -r^2/(2*4.0^2))
    grid = M1Grid1D(300, 25.0)
    res = solve_M1(grid, [-grid.r[i]^2/(2*4.0^2) for i in 1:300], zeros(300), 1.0, 6.0, bg;
                   CFL = 0.25, save_dt = 5.0, DsT = 0.1163, log_every = 0)
    r = res["r_grid"]; n = res["n"]; τ = res["t_grid"]
    Nt = res["Ntau"]
    Ntot(j) = τ[j]*sum(i -> Nt[i, j]*r[i], eachindex(r))
    drift = abs(Ntot(length(τ))/Ntot(1) - 1)
    @printf("\nG2  charm number drift over τ = %.1f → %.1f :  %.2e\n", τ[1], τ[end], drift)
    check("G2 charm number is conserved", drift < 1e-6, @sprintf("drift %.2e", drift))
end

# ── G3 : the Navier-Stokes limit ────────────────────────────────────────────────────────────────
# A wide blob on a static uniform bath spreads diffusively: in the 2-D transverse plane
# ⟨r²⟩ = ⟨r²⟩₀ + 4 D_s Δτ.  This is the one gate that checks the SOURCE normalisation against the
# transport coefficient the rest of the pipeline is calibrated to.
let
    σ = 5.0; DsT = 0.1163
    Ds_fm = DsT/T0BG*0.1973269804                        # D_s = D_sT/T, GeV⁻¹ → fm
    bg = mk_bg(α = (r, τ) -> -r^2/(2σ^2))
    grid = M1Grid1D(400, 30.0)
    res = solve_M1(grid, [-grid.r[i]^2/(2σ^2) for i in 1:400], zeros(400), 2.0, 10.0, bg;
                   CFL = 0.25, save_dt = 2.0, DsT = DsT, log_every = 0)
    r = res["r_grid"]; n = res["Ntau"]; τ = res["t_grid"]
    # ⚠ MEASURE D AFTER THE TRANSIENT.  The current starts at zero and builds up on
    # τ_rel = h/(η_D M) ≈ 0.6 fm/c here; including that in the fit biases D low by ~τ_rel/Δτ, which
    # is most of the discrepancy the first version of this gate reported.
    println()
    for j in 2:length(τ)
        @printf("G3  τ %5.2f → %5.2f : ⟨r²⟩ %.3f → %.3f fm²  ⇒ D = %.4f fm  (D_s = %.4f)\n",
                τ[j-1], τ[j], r2_of(r, n[:, j-1]), r2_of(r, n[:, j]),
                (r2_of(r, n[:, j]) - r2_of(r, n[:, j-1]))/(4*(τ[j] - τ[j-1])), Ds_fm)
    end
    D_meas = (r2_of(r, n[:, end]) - r2_of(r, n[:, end-1]))/(4*(τ[end] - τ[end-1]))
    @printf("G3  post-transient D = %.4f fm vs D_s = %.4f fm  (%.1f%%)\n",
            D_meas, Ds_fm, 100*abs(D_meas/Ds_fm - 1))
    check("G3 the NS limit reproduces D_s", abs(D_meas/Ds_fm - 1) < 0.08,
          @sprintf("D %.4f vs %.4f fm (%.1f%%)", D_meas, Ds_fm, 100*abs(D_meas/Ds_fm - 1)))
end

# ── G4 : the IC round trip ──────────────────────────────────────────────────────────────────────
# The boosted-Jüttner IC is built from (n, ν^r) and the recovery must return them.  Run zero steps by
# asking for τf = τ0 + one CFL step and reading the FIRST snapshot.  A flowing background is used on
# purpose: this is where the Milne↔LRF boost composition of eq:boostsubs can go wrong silently.
let
    grid = M1Grid1D(200, 20.0)
    bg = mk_bg(ur = (r, τ) -> 0.4*r/10, α = (r, τ) -> -r^2/(2*4.0^2))
    α0 = [-grid.r[i]^2/(2*4.0^2) for i in 1:200]
    # w is prescribed directly as a fraction of the bound, so the IC is realizable by construction
    wtgt(r) = 0.35*sin(π*min(r/12, 1.0))
    eos0 = LatticeHRGEOS(canon_factor = 1.0)
    νr0 = map(1:200) do i
        _, nn, _ = eos_Pne(T0BG, α0[i]*T0BG, eos0)
        uτ = sqrt(1 + (0.4*grid.r[i]/10)^2)
        nn*wtgt(grid.r[i])*uτ
    end
    res = solve_M1(grid, α0, νr0, 1.0, 1.0 + 1e-9, bg; CFL = 0.25, save_dt = 1e-9,
                   DsT = 0.1163, log_every = 0)
    νout = res["nur"][:, 1]
    core = 5:180
    # ⚠ SCALE THE RESIDUAL BY THE PROFILE, NOT POINTWISE.  wtgt vanishes at r = 0 and r ≥ 12, so a
    # pointwise relative error divides round-off by zero and reports nonsense — it did (6.5), while
    # the absolute residual was 6e-12.
    scale = maximum(abs, νr0[core])
    dev = maximum(abs.(νout[core] .- νr0[core]))/scale
    @printf("\nG4  ν^r round trip on a FLOWING background: max rel dev = %.2e\n", dev)
    check("G4 the boosted IC returns its own (n, ν^r)", dev < 1e-8, @sprintf("max rel dev %.2e", dev))
end

# ── G5 : the production LP1 background ──────────────────────────────────────────────────────────
let
    p = normpath(joinpath(@__DIR__, "..", "Projects", "LangevinPaper1", "data",
                          "hydro_splines_matchedIC", "charm_physical_constFluidum.jld2"))
    if !isfile(p)
        @printf("\nG5  skipped — %s not found\n", p)
    else
        res = run_static_M1_test(background_file = p, DsT = 0.1163, τ0 = 0.4, τfinal = 8.0,
                                 Nr = 300, rmax = 25.0, dump_dt = 2.0, log_every = 0)
        w = res["w"]; n = res["n"]; nb = res["diagnostics"]["realizability_floor"]
        wmax = 0.0; loc = (0.0, 0.0)
        for j in axes(w, 2), i in axes(w, 1)
            n[i, j] > 1e-9*maximum(n[:, j]) || continue
            abs(w[i, j]) > wmax && (wmax = abs(w[i, j]); loc = (res["t_grid"][j], res["r_grid"][i]))
        end
        @printf("\nG5  LP1 const-D_sT background:  max|w| = %.4f at (τ=%.2f, r=%.2f),  floored cells = %d\n",
                wmax, loc..., nb)
        check("G5a the production background runs inside the cone", wmax < 1.0,
              @sprintf("max|w| = %.4f", wmax))
        check("G5b no cell needed the realizability floor", nb == 0, @sprintf("%d floored", nb))
    end
end

# ── G6 : strong-gradient stress test ────────────────────────────────────────────────────────────
# A sharp edge on a WEAKLY coupled charm sector is the configuration that makes IS2 leave the cone
# and forces the vacuum ramp.  There is no ramp, no α clamp and no ν bound in this module, so if the
# formulation needs one it will show up here.
let
    grid = M1Grid1D(400, 25.0)
    edge(r) = log(max(0.5*(1 - tanh((r - 6.0)/0.4)), 1e-40))
    bg = mk_bg(α = (r, τ) -> edge(r))
    res = solve_M1(grid, [edge(grid.r[i]) for i in 1:400], zeros(400), 1.0, 9.0, bg;
                   CFL = 0.25, save_dt = 8.0, DsT = 3.0, log_every = 0)
    w = res["w"]; n = res["n"]; nb = res["diagnostics"]["realizability_floor"]
    dyn = maximum(n[:, end])
    wmax = maximum(abs(w[i, end]) for i in eachindex(res["r_grid"]) if n[i, end] > 1e-16*dyn)
    @printf("\nG6  sharp edge, D_sT = 3.0 (τ_drag = %.1f fm/c):  max|w| = %.4f over 16 decades of n, floored = %d\n",
            1/eta_drag(T0BG, MASS, 3.0), wmax, nb)
    check("G6a a sharp edge stays inside the cone with no regulator", wmax < 1.0,
          @sprintf("max|w| = %.4f", wmax))
    check("G6b and needs no realizability floor", nb == 0, @sprintf("%d floored", nb))
end

println("\n" * "═"^96); println("GATES")
for (n, ok, d) in RES
    @printf("  %-4s %-52s %s\n", ok ? "PASS" : "FAIL", n, d)
end
@printf("\n%d/%d passed\n", count(r -> r[2], RES), length(RES))
