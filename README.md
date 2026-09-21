# FiVoHydro.jl — FiVo

[![CI](https://github.com/RuwenSchulz/FiVoHydro.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/RuwenSchulz/FiVoHydro.jl/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22790954.svg)](https://doi.org/10.5281/zenodo.22790954)

A finite-volume solver for boost-invariant relativistic viscous hydrodynamics with a diffusing
heavy-quark (charm) charge, written for heavy-quark transport in heavy-ion collisions.

There are three solvers. They share one term interface, one test suite and one output format:

| solver | driver → module | evolves | geometry |
|---|---|---|---|
| **1+1D bulk** | `main.jl` → `hydro` | the medium $(T, u^r, \Pi, \pi_\phi, \pi_\eta)$ + a diffusing charge $(n, \nu^r)$ | radial Milne $(\tau, r)$ |
| **1+1D charm IS2** | `main2IS2.jl` → `hydro_current_IS2` | the charm $(\alpha, \nu^r, \pi_Q^r, \pi_Q^\perp, \Pi_Q)$ on a frozen background | radial Milne |
| **2+1D** | `main2D.jl` → `hydro2d` | medium + charge + the charm second moment | transverse Cartesian Milne $(\tau, x, y)$ |

The bulk solvers (1+1D and 2+1D) are conservative finite volume: HLLE fluxes, MUSCL reconstruction
(MC limiter) in primitive variables, SSPRK2/3 in time, operator-split relaxation of the dissipative
fields and a MOOD fallback. The charm relaxation equations have no conservative form. The IS2 solver
uses the quasi-linear form $A_t\,\partial_\tau U + A_x\,\partial_r U = S$ with RK4, and the characteristic
speeds are the eigenvalues of $A_t^{-1}A_x$. The charm charge itself is still updated conservatively.

![a real Pb+Pb event](examples2d/figures/ex10_showcase_N400.png)

<sub>One MC-Glauber Pb+Pb event at N = 400, τ = 8 fm/c, from `examples2d/10_showcase.jl`. The white
line is freeze-out.</sub>

---

## Validation

All numbers below come from the scripts named next to them. The validation ladders were last run on
2026-09-15, the examples on 2026-09-14.

| | | |
|---|---|---|
| **1+1D validation ladder** | `test/run1d_gates.jl` | **10 / 10** |
| **2+1D validation ladder** | `test/run2d_gates.jl` | **22 / 22** |
| **unit + fast tier (CI)** | `Pkg.test()` | green |
| **worked examples** | `examples1d/`, `examples2d/` | **14 / 14** run clean |
| **cross-check with Fluidum.jl** | 5 algebraic gate files + a 15-gate 1+1D solve comparison | all pass (below) |

### Measured errors

| what | against | measured |
|---|---|---|
| ideal Bjorken, 1+1D | $T\tau^{1/3}$ = const, and the $(e,n)$ ODE | convergence order **2.00** (SSPRK2) / **2.99** (SSPRK3); $T$ to 2e-9 |
| viscous Bjorken, 1+1D | the 0+1D DNMR ODEs, term by term | Richardson-extrapolated error ≤ **3e-4** |
| **viscous Gubser, 1+1D** | its semi-analytic ODE | $L_2(T)$ = **2.8e-5**, $L_2(\bar\pi)$ = **0.19 %** at $N_r$ = 800; order **2.12** in $T$ |
| charm diffusion mode | closed $J_0/J_1$ ODE, all three solvers | amplitude ≤ 3e-3, current ≤ 1 % |
| Bjorken, 2+1D | the 0+1D Israel–Stewart system | order **2.00** |
| Gubser, 2+1D | the exact solution | order **1.96** |
| sound attenuation, 2+1D | the exact MIS dispersion root | excess damping first order in $\Delta x$ (numerical viscosity) |

The dissipative fields are first order in $\Delta\tau$ because the relaxation is operator-split.
Halve `CFLτ` and their error halves. The ideal sector is second order, and so is $T$ on viscous
Gubser (order 2.12). $\bar\pi$ is order 1.01.

![analytic benchmarks](examples1d/figures/ex03_analytic_benchmarks.png)

<sub>`examples1d/03_analytic_benchmarks.jl`: ideal Bjorken at orders 2 and 3, viscous Bjorken against
the DNMR ODEs, viscous Gubser at three resolutions, and the charm diffusion mode through both 1-D
solvers. The inset convergence test is taken at a fixed radius.</sub>

---

## Cross-check with Fluidum.jl

FiVo is compared with Fluidum.jl, a hydro code with a different discretisation: primitive-variable
upwind on a quasi-linear matrix, adaptive Tsit5, and dissipation inside the matrix instead of
operator-split.

| | measured |
|---|---|
| the two codes discretise the same PDE, every row of the 2+1D viscous system | **1e-11** |
| charm second moment, 2+1D, on 2-D states | **2.2e-16**, all four switch combinations |
| transverse charm first moment, vs a closed form | FiVo converges to it: **1.05 → 0.53 → 0.43 %** at N = 32/48/64, worst single cell 0.8 % |
| **1+1D medium, Bjorken** | **2.3e-6** after Richardson-extrapolating FiVo's operator split, below the 1.3e-5 floor from the two codes' different $\hbar c$ |
| **1+1D medium, fireball** | the difference converges in every field and sector: $T$ 0.33 → 0.16 % (ideal), 0.28 → 0.09 % (shear), 0.30 → 0.13 % (shear+bulk) over $N$ = 100 → 200 |
| **distance to the continuum** (2+1D, real Pb+Pb event) | each code against its own Richardson limit at 384²: FiVo's error is **8.7–25× smaller in $T$** and **3.4–19× smaller in $u^x$**; the two limits agree to 0.24–0.38 % in $T$. In the charm fields the limits are 13–44 % apart at τ = 4 and 8, so no ranking there |

![FiVo vs Fluidum, 1+1D](docs/figures/crosscheck_1p1d.png)

<sub>1+1D. Top: both codes against a closed form, and the Bjorken difference, which is FiVo's
operator split. Bottom: fireball profiles and the code-to-code difference against resolution. The
profiles stop at r = 9 fm, beyond which $|\pi|/P > 1$.</sub>

![distance to the continuum](docs/figures/crosscheck_self_convergence.png)

<sub>2+1D, a real Pb+Pb event. Each code against its own Richardson limit at 384². Hollow markers:
the two limits do not agree, so no ranking there. `fit failed` means the order fit did not converge.</sub>

![the 2+1D code-to-code difference against resolution](docs/figures/crosscheck_2p1d_convergence.png)

<sub>2+1D. The code-to-code difference in $T$ against resolution, three initial conditions × three
sectors. In `full` it stalls because FiVo has the DNMR coefficient $\delta_{\pi\pi} = \tfrac43
\tau_\pi$ and Fluidum's Israel–Stewart does not. With $\delta_{\pi\pi} = 0$ it converges again.</sub>

Figure provenance: [`docs/figures/README.md`](docs/figures/README.md).

---

## Worked examples

Fourteen examples, three 1+1D and eleven 2+1D. Each takes seconds to minutes and makes the figure
next to it.

| | |
|---|---|
| [`examples1d/`](examples1d/README.md) | the 1+1D library interface, the term switches, the analytic benchmarks |
| [`examples2d/`](examples2d/README.md) | elliptic flow, resolution, fluctuating and real events, the dissipative sectors, the charm terms, vorticity, Gubser, and two showcases |

The examples need `Plots`, which is not a dependency of the package.

![elliptic flow](examples2d/figures/ex02_elliptic_flow.png)

<sub>`examples2d/02_elliptic_flow.jl`: ε₂ → momentum anisotropy, sector by sector from one initial
condition, with an ε₂ = 0 control.</sub>

---

## Licence and citation

© 2026 Ruwen Schulz. Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

If you use FiVo in published work, please cite it ([`CITATION.cff`](CITATION.cff), or GitHub's
"Cite this repository" button) and say which version or commit you ran.

| | DOI |
|---|---|
| all versions (resolves to the newest) | [10.5281/zenodo.22790954](https://doi.org/10.5281/zenodo.22790954) |
| v0.1.0 | [10.5281/zenodo.22791452](https://doi.org/10.5281/zenodo.22791452) |

## 1. Which solver

| you want | use |
|---|---|
| a medium (with or without viscosity) and a diffusing charge, axisymmetric | 1+1D bulk: `build_model_1d` + `run_sim_1d!` |
| the charm current and second moment on a given medium | 1+1D IS2: `run_static_IS2_test` |
| anything non-axisymmetric (ε₂, ε₃, single events), or the vorticity couplings | 2+1D: `build_model_2d` + `run_sim_2d!` |

The first-order (`main2.jl`), BDNK (`main2BDNK.jl`, `mainBDNK*.jl`), MaxEnt (`main2M1.jl`, `main2M2.jl`)
and density-frame current solvers are other closures of the same charm problem (Entry points, below).

---

## 2. Quickstart

### Install

```sh
git clone https://github.com/RuwenSchulz/FiVoHydro.jl.git
cd FiVoHydro.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'      # the unit + fast tier
```

CI runs **Julia 1.11**, and `Manifest.toml` is resolved for 1.11.5. It also runs on 1.12, where
`instantiate` warns about the manifest version and a stale project hash. Both warnings are harmless.
Don't `Pkg.resolve()`, it would break the 1.11 CI.

### A first run

**1+1D medium + charge**, in a REPL started with `julia -t auto --project=.`:

```julia
include("main.jl"); using .hydro; const H = hydro
g = H.make_grid_1d(300; rmax = 15.0)
m = H.build_model_1d(; enable_shear = true, eta_over_s = 0.1,
                       enable_diff = true, kappa_coeff = 0.1163,     # D_s T
                       consistent_fm = true, terms = :default)       # §3
H.show_equations(m)                                                  # what m integrates, term by term
U = H.allocate_state(g, m)
H.initialize_from_radial!(U, g, m, 0.4, r -> 0.05 + 0.42exp(-r^2/20), r -> -4.0)
res = H.run_sim_1d!(U, g, m; τ0 = 0.4, τfinal = 5.0)                 # in memory
res.ok || error("step failed at τ = $(res.τ)")
f = H.fields_1d(g, U, m; τ = res.τ, work = res.work)                  # r, T, mu, alpha, n, e, P, ur, nur, Pi, piR, piPhi, piEta
H.save_fields("run.jld2", f; model = m)                               # §4
```

**1+1D charm on a frozen background.** Give either a JLD2 bundle or functions of (τ, r):

```julia
include("main2IS2.jl"); using .hydro_current_IS2; const HI = hydro_current_IS2
bg  = HI.analytic_background(; T = (τ, r) -> 0.4*(0.6/τ)^(1/3)*exp(-r^2/30))     # or background_file = "bg.jld2"
res = HI.run_static_IS2_test(; background = bg, DsT = 0.1163, τ0 = 0.6, τfinal = 5.0, Nr = 300, rmax = 12.0,
                               consistent_fm = true, consistent_m2 = true, terms = HI.without(:acceleration))
res["n"], res["nur"], res["piQr"], res["PiQ"], res["terms"], res["diagnostics"]
HI.show_equations_IS2(; consistent_fm = true, consistent_m2 = true)
```

**2+1D:** `README2D.md` §1. Same keywords (`build_model_2d(; enable_shear, eta_over_s, enable_diff,
kappa_coeff, consistent_fm, consistent_m2, terms)`) and same defaults as `build_model_1d`.

Two things to watch:

1. `res.ok` means every step completed, not that the result is physical. Check `res.dQ`, `res.maxu`
   and the correction counters, and quote dissipative fields only where $T > T_{\rm fo}$.
2. Call `finalize_ic!` after `set_cell!`. The `initialize_*` helpers do it for you.

---

## 3. Switching terms off

Every solver takes a `terms` keyword with the same spellings (`src/terms.jl`):

```julia
terms = :default                               # the equations as they stand
terms = :homogeneous                           # a homogeneous medium at rest (see below)
terms = without(:vorticity, :acceleration)     # drop every term built from these ingredients
terms = (fm_inertial = false,)                 # one named term
terms = (preset = :full, without = (:acceleration,), fm_dlnh = false)
terms = (with = (:vorticity,),)                # the default plus both vorticity couplings
```

`show_terms()` prints the register. `show_equations(model)` (and `show_equations_IS2(; …)`) print the
equations a configuration integrates, with every term marked `[x]`, `[ ]`, or `≡0` if it vanishes in
that geometry.

**Presets**

| preset | meaning |
|---|---|
| `:default` | every derived term on except the two vorticity couplings |
| `:full` | every term on, vorticity included |
| `:homogeneous` | drops every term built from a gradient or rate of the medium ($\nabla T$, $a = Du$, $\nabla u$, $DT$), inertial terms included. What is left is driven by the charm's own gradients ($\nabla\alpha$, $\nabla\nu$, $D\alpha$). With `consistent_fm = true` this reproduces `consistent_fm = false` bit for bit (gates T3, Gt7, X1). θ includes the Bjorken $1/\tau$, so the longitudinal dilution is dropped too |
| `:none` | every term off; each field relaxes to zero on its own clock |

**Ingredients** (what `without` / `with` take). A term is dropped if it is built from any of them:

| ingredient | terms it removes |
|---|---|
| `:acceleration` (alias `:inertia`) | `fm_inertial` ($\tau_n n a$), `m2_accel_nu`, `m2_projector` |
| `:vorticity` | `m2_vorticity`, `shear_vorticity` |
| `:temperature_gradient` | `fm_gradT`, `m2_nu_gradTh` |
| `:shear` / `:expansion` | `m2_bg_gradu`, `m2_pi_sigma`, `m2_PiQ_sigma` / `fm_expansion`, `m2_bg_gradu`, `m2_expansion` |
| `:velocity_gradient` | all of ∇u: `fm_nu_gradu` and everything under `:shear`, `:expansion`, `:vorticity` |
| `:cooling` ($DT$), `:fugacity_rate` ($D\alpha$) | `fm_dlnh`, `m2_bg_DlnT`, `m2_expansion` / `m2_bg_Dalpha` |
| `:fugacity_gradient`, `:current_gradient` | `nu_gradalpha` (the shipped drive) / `m2_nu_gradient` |
| `:medium_gradients` | the `:homogeneous` set, relative to any base |

**Rules**
- Naming a term in a disabled sector (e.g. `fm_inertial = false` without `consistent_fm`), or
  misspelling one, is an error. Changes from a preset or an ingredient in a disabled sector are ignored.
- A switched-off term is removed exactly (gate T2: 3e-15; Gt2/Gt3).
- In 1+1D the vorticity couplings and the Δ-projector of the second moment vanish. Switching them
  does nothing and shows as `≡0`.

---

## 4. Input and output

| input | how |
|---|---|
| grid | `make_grid_1d(Nr; rmax, nghost)`, `make_grid2d(Nx, Ny; xmax, ymax, nghost)` (also spelled `make_grid_2d`) |
| model | `build_model_1d(; …)`, `build_model_2d(; …)`, same keyword names and defaults; IS2 takes keywords per solve |
| initial state | `initialize_uniform!`, `initialize_from_radial!(…, T(r), α(r); urof)`, or `set_cell!` + `finalize_ic!` per cell (1-D and 2-D); 2-D also `initialize_from_grid_csv!`; CSVs through `run_sim_ideal_diff_visc(; init_csv)` |
| IS2 background | `background_file` (a JLD2 bundle with `r_grid`, `t_grid`, `T_spline`, `ur_spline`, optional α/n/ν/κ/τ_diff/gradient splines) or `background = analytic_background(T = (τ, r) -> …, ur = …)` |

| output | what |
|---|---|
| `fields_1d(g, U, m; τ, work)` | NamedTuple of vectors over the interior: `r, T, mu, alpha, n, e, P, ur, utau, v, nur, Pi, piR, piPhi, piEta, ok, τ`, in GeV and fm |
| `fields_2d(g, U, work, m)` | NamedTuple of Nx×Ny matrices: `x, y, T, mu, alpha, n, e, P, ux, uy` + every dissipative field. Unlike `fields_1d`, `work` is positional and there is no `τ` |
| `run_static_IS2_test` | Dict: `r_grid, t_grid, n, nur, alpha, piQr, piQperp, PiQ` (Nr × Nt), `diagnostics` (per solve), `terms`, `consistent_fm`, `consistent_m2` |
| `save_fields(path, f; model, meta)` / `load_fields(path)` | all three solvers. One JLD2 file with the fields, the resolved term switches, the `show_equations` printout, every model knob, the git revision (`-dirty` if uncommitted) and a time stamp. Plain types only, so it reads without FiVo loaded. Schema: `src/fields_io.jl` |
| `run_sim_ideal_diff_visc` | legacy: `snapshot_tau_*.csv` per dump (+ `_meta.csv`) |

---

## 5. Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                  # ~4 min, in CI: unit + the 1-D fast tier (T, A4, IO)
julia -t auto --project=. test/run1d_gates.jl                 # 1+1D ladder, ~10 min
julia -t auto --project=. test/run2d_gates.jl                 # 2+1D ladder, ~30 min
FIVO1D_TIER=fast … run1d_gates.jl;  FIVO2D_TIER=fast … run2d_gates.jl   # the CI tiers, ~70 s / ~60 s
FIVOHYDRO_LONG_TESTS=1 julia … Pkg.test()                     # + A1/A2, X1, the IS2 stability run
```

Each runner exits 1 on any failure, including a missing gate file.

**1+1D** (`test/run1d_gates.jl`, 10 gates, **10/10 on 2026-09-14**):

| gate | tested against | measured |
|---|---|---|
| **T** term switches | the switched pieces vs the shipped expressions; `:homogeneous` ≡ shipped on a solve (bulk, IS2); Euler cancellation of ∇T + inertia on an ideal fluid; refusals | pieces 3e-15; bit for bit; pair cancels to 0.5 % of either term alone |
| **A1** ideal Bjorken | $T\tau^{1/3}$ = const (conformal); the (e, n) ODE with a charge (LatticeHRGEOS) | order 2.00 (SSPRK2) / 2.99 (SSPRK3); T to 2e-9, nτ to 4e-16 |
| **A2** viscous Bjorken | the 0+1D DNMR ODEs, each of δ_ππ, τ_ππ, λ_πΠ, δ_ΠΠ, λ_Ππ; wrong-sign versions must fail | first order (the split); extrapolated error ≤ 3e-4 |
| **A4** viscous Gubser | the semi-analytic ODE (as 2-D G1v) | L2(T) 2.8e-5, L2(π̄) 0.19 % at Nr = 800, order 2.1 / 1.0 |
| **X1** diffusion mode | $\delta n = AJ_0(kr)$, $\nu^r = BJ_1(kr)$ closed ODE; 1D bulk, IS2 and 2D, four term configurations | A to ≤ 3e-3 (1D, IS2), B to ≤ 1 %; 2D converges with dx |
| **IO** output format | `save_fields` → `load_fields` for all three solvers: fields bit for bit, the terms and knobs that made them, readable with only JLD2 loaded | 21/21, 12 s |
| IS2 drive, IS2 speeds, density frame, BDNK, M1 | also in `Pkg.test()` | — |

**2+1D:** 22 gates, **22/22 on 2026-09-15** (≈29 min), listed with what each measured in `README2D.md` §5.

---

## 6. Cost

`bench/bench1d.jl` (1+1D) and `bench/bench2d.jl` (2+1D, `README2D.md` §6) report the cost per
cell-step, best of three after a warm-up. Measured 2026-09-11, 8 threads, on a Pb+Pb-like profile,
τ 0.4 → 2 fm:

| 1+1D bulk | Nr = 400 | Nr = 800 |
|---|---|---|
| ideal | 2.9 µs (144 steps, 0.17 s) | 2.7 µs (288 steps, 0.61 s) |
| + shear / + bulk | +41 % / +36 % | +45 % / +45 % |
| + diffusion | +47 %, and 1.5–2.2× the steps (diffusion step cap) | +47 % |
| all sectors + diffusion | 4.5 µs (223 steps, 0.40 s) | 4.0 µs (640 steps, 2.1 s) |
| + `consistent_fm` | −2 % (noise) | +6 % |

| charm IS2 (analytic background, τ 0.6 → 1.6) | Nr = 300 | Nr = 600 |
|---|---|---|
| shipped closure | 48 µs, 18 MB allocated per step | 50 µs, 36 MB per step |
| `consistent_fm` | +25 % | +21 % |
| `consistent_fm + consistent_m2` | +40 % | +33 % |

At 4 threads (2026-09-14) the per-cell-step costs roughly double: ideal 6.6 µs at Nr = 400 and 6.4 µs
at Nr = 800, all sectors + diffusion 10.7 / 8.3 µs, charm IS2 76 µs at Nr = 300. Allocation does not
depend on the thread count (17.9 MB/step at Nr = 300, 35.9 at Nr = 600).

A 1+1D bulk solve takes seconds. The Pb+Pb background (Nr = 800 to τ = 6 fm, 4 threads) took 184 s.
The IS2 solver costs ~15× the bulk solver per cell and allocates ~60 kB per cell per step: every cell
builds three 5×5 systems per RK stage, each with an `lu`, two solves and an `eigvals`.

---

# Reference

## How the solvers share code

`main2D.jl` includes six files from `src/`: `constants.jl`, `utils.jl`, `eos.jl`, `terms.jl`,
`primitives.jl` (transport-coefficient models) and `relaxation_laws.jl`. Everything that depends on
the dimension (grid, primitive recovery, fluxes, reconstruction, RHS, timestepper, dissipation,
floors, boundaries) is re-implemented in `src2d/`, so changing the 2-D solver cannot move 1-D
results. Tests check the overlap: `test_primrec2d_vs_1d.jl`, `test_reproduction2d.jl`,
`test_dissipative_vs_1d.jl`, `test_consistent_fm2d.jl` Gc1, `test_consistent_m22d.jl` Gm1, and X1.

Shared: the term register (`src/terms.jl`) and the closure formulas used by both the 1-D bulk and the
IS2 solver (`src/hq_consistent_firstmoment.jl`). The 2-D solver reads no ENV variables. The 1-D
drivers read many; `ENV_FLAGS.md` lists them.

## Entry points

| file | module | physics |
|---|---|---|
| `main.jl` | `hydro` | bulk fluid + charge: `run_sim_ideal_diff_visc(; kwargs…)` (library) / `main()` (CLI, ENV-driven). Includes all of `src/` |
| `main2IS2.jl` | `hydro_current_IS2` | charm current, Israel–Stewart 5-field `(α, ν^r, π_Q^{rr}, π_Q^⊥, Π_Q)` on a frozen bulk: `run_static_IS2_test(; background_file, …)` |
| `main2.jl` | `hydro_current` | charm current, first-order (ν^r relaxes to NS): `solve_current_only` |
| `main2M1.jl` | `hydro_current_M1` | charm current as a 3-field maximum-entropy (M1) moment system, no transport coefficients, no regulators: `run_static_M1_test`, `solve_M1` |
| `main2M2.jl` | `hydro_current_M2` | four-field MaxEnt M2 system (parallel to M1): `run_static_M2_test` |
| `main2BDNK.jl` | `hydro_current_bdnk` | BDNK current-only (keeps `∂_τ α`, `σ_T/σ_a`); has an `EPS_NU=density_frame` branch |
| `mainBDNK.jl` / `mainBDNK_causal.jl` | — / `bdnk_causal` | BDNK bulk driver `run_sim_bdnk`; the causal telegraph variant |
| `mainDensityFrame.jl` | — | thin driver: `charge_mode=:density_frame` (ν-less parabolic flux) then `hydro.main()` |
| `mainBGonly.jl` | `hydro_bgonly` | background-only copy of `main.jl` |
| **`main2D.jl`** | **`hydro2d`** | the 2+1D solver (transverse Cartesian, boost-invariant Milne): bulk + charge `(T, u^x, u^y, Π, π^{xx}, π^{xy}, π^{yy}, π^{ηη}, n, ν^x, ν^y)` via `run_sim_2d!`, with the consistent first moment (`consistent_fm`) and a passive second moment (`consistent_m2`). Includes all of `src2d/`. See `README2D.md` |
| `src/FiVoHydro.jl` | `FiVoHydro` | package wrapper: includes `main.jl`, exports `hydro` |

`src/` is the bulk solver: `eos.jl` (ConformalHQEOS, LatticeHRGEOS, TabulatedHQEOS), `primitives.jl`
(`IdealDiffViscModel`, viscosity models), `primrec.jl` (3-unknown Newton recovery),
`fluxes.jl`/`reconstruction.jl`/`rhs.jl`/`timestepper.jl` (the FV scheme), `dissipation.jl`
(transport coefficients, `relax_dissipative!`, stabilizers), `mood.jl`, `floors.jl`, `io.jl`,
`gubser.jl` (analytic Gubser + `initialize_gubser!`), `is2_second_moment_builder.jl` (IS2 matrices),
`runtime_flags.jl` (the `HYDRO_*` ENV reader), `terms.jl` (§3), `api1d.jl` (the 1+1D library
interface), `fields_io.jl` (§4). `relaxation_laws.jl` is a reference exponential integrator for tests
and benches, not a solver hook.

## Stabilizers (bulk solver), all off by default

| kwarg (`run_sim_ideal_diff_visc`) | default | what it does | where |
|---|---|---|---|
| `nur_clip_factor` | `-1` (off) | clip `\|ν^r\| ≤ f·n_smoothed·u^τ` after relaxation | `src/dissipation.jl` `relax_dissipative!` (end) |
| `alpha_filter_eps`, `nur_filter_eps`, `visc_filter_eps` | `0` | Kreiss–Oliger filter strength, clamped to `[0, 0.24]` | `smooth_alpha!` etc. |
| `alpha_smooth_len`, `nur_smooth_len`, `visc_smooth_len` | `0` | smoothing length → ε via `eps_from_len(·, dr)` | same |
| `dtau_u_smooth_len` | `0` (off) | band-limit `∂_τu^r` at a fixed physical length (fm). On fine grids the split scheme rings at short wavelength, and the ringing grows with resolution (onset between dr = 0.026 and 0.013 fm on the O+O bulk; `EQUATIONS1D.md` §8). `diag.dtau_ur_q_max` says whether a run is affected | `src/dissipation.jl` `bandlimit_centered!` |
| `do_soft_project_nur` | `false` | tanh projection of ν_NS and ν_new into `\|ν^r\| < n u^τ` | `_soft_project_nur_phys` |
| `do_axis_project_nur`, `axis_project_nfit` | `false`, `2` | polynomial axis projection of ν^r | `axis_project_nur_tapered!` |
| `Pi_clip_factor`, `pi_clip_factor` | `-1` (off) | clip bulk/shear relative to the pressure | `relax_dissipative!` |

The CLI `main()` falls back to `last_ic_diagnostics.env` (NUR_CLIP_FACTOR=0.98, ALPHA_FILTER_EPS=0.01,
VISC_FILTER_EPS=0.01, PI/SHEAR_CLIP_FACTOR=0.35, TAU_SHEAR_COEFF=0.2, TAU_PI_COEFF=15) when an ENV
var is unset. That is a different operating point from the kwarg defaults, so keep the file.

## Charm-sector regulators (IS2)

| knob (ENV) | default | production | meaning |
|---|---|---|---|
| `FIVO_VACUUM_N_LO` / `FIVO_VACUUM_N_HI` | `1e-6` / **`2e-3`** | Pb+Pb: default | density-gated ramp `w=(n−n_lo)/(n_hi−n_lo)` multiplying the whole RHS, i.e. an effective `τ_n/w` in the dilute tail. Calibrated on Pb+Pb (~24 charm quarks) |
| `FIVO_VACUUM_N_REL_HI` / `_LO` | `0` (off) / `1e-4` | O+O: `0.003` | the same ramp relative to the slice maximum; needed on O+O (0.27 charm quarks) |
| `FIVO_VACUUM_T_HI` / `_LO` | `0` (off) | off | T-gated ramp; leaves the warm flat tail undamped |
| `FIVO_IS2_ALPHA_MIN` / `_MAX` / `_SOFT` | `-20` / `200` / `1` | default | softplus floor on α |
| `FIVO_IS2_NU_BOUND`, `_FRAME`, `_SMOOTH`, `_KNEE` | `0` (off), `"lab"`, `0`, `0.8` | O+O: `0.7` | soft saturation `\|ν^r\| ≤ f·n`. `"lab"` is tighter than the LRF bound by `u^τ`; `"lrf"` is the physical frame |
| `FIVO_IS2_CONE_PROJECT` | `0` (off) | O+O c_M variant: `0.995` | projection into the admissible cone |
| `FIVO_IS2_JTAU_FLOOR` | `0` | default | floor on J^τ |
| `FIVO_NU_RAPIDITY` | `0` | off | rapidity chart; the α recovery undoes it, so it has no effect as is |
| `FIVO_CM_SIGN` | **`+1`** | default | `-1` is the elliptic (ill-posed) branch |
| `FIVO_IS2_TAUN_DEGENERACY`, `_TAUM_DEGENERACY` | `0` (bare) | bare | τ_n = D_s·z·K₃/K₂ without `÷g_hq`; τ_M = τ_n/2 |
| `FIVO_IS2_CAUSAL_COEFF` | `1` (on) | on | `τ_n ← max(τ_n, κ/χ)` keeps the diffusion signal speed ≤ c |
| `FIVO_IS2_TAUN_SCALE`, `FIVO_IS2_TAUPI_REL` | `1`, `0` | default | diagnostic dials; don't combine `TAUN_SCALE≠1` with `FIVO_IS2_USE_CM=1` |
| `FIVO_ORIGIN_ODD_FIRST_ORDER_CELLS` | `4` | default | first-order cells at the axis for the odd fields |

The ramp masks a ν^r runaway in the dilute tail (unregulated, `max\|ν^r\|/n` reaches 2528 c on O+O)
and also damps the physical freeze-out shell. The M1/M2 solvers don't need it. The history and the
negative results are in the header of `main2IS2.jl`.
