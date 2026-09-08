# FiVoHydro.jl

A 1+1D (radial, boost-invariant Milne) finite-volume solver for a relativistic viscous fluid carrying
a diffusing conserved charge, written for heavy-quark (charm) transport in heavy-ion collisions. One
bulk solver evolves `(T, u^r, Π, π^{φφ}, π^{ηη})` together with the charge `(n, ν^r)`; a family of
*current-only* solvers evolves the charm sector on a frozen background with different closures
(first-order MIS, Israel–Stewart second moments, BDNK, maximum-entropy M1/M2, density frame).

Scheme: HLLE Riemann solver + MUSCL (MC limiter) reconstruction in primitive variables + SSPRK2/3
method-of-lines, with operator-split relaxation of the dissipative fields and a MOOD fallback. Geometry
is radial Milne, so the native analytic references are Gubser and Bjorken flow; planar shock tubes are
covered by the flat-Cartesian sibling `FiVo2DIdeal` (`Julia/FiVo2DIdeal.jl`, its own submodule).

This package is a git submodule of `phd-git`; it is consumed almost entirely by `include`-ing one of
the driver files below from project scripts run with `--project=Julia/FiVoHydro.jl`.

## Two solvers in one package — read this first

This repository holds **two independent solvers** that share a package but almost no code:

| | 1+1D (radial) | 2+1D (transverse) |
|---|---|---|
| driver / module | `main.jl` → `hydro` | `main2D.jl` → `hydro2d` |
| numerics | `src/` (28 files, ~7.8 k lines) | `src2d/` (14 files, ~3.3 k lines) |
| geometry | radial Milne `(τ, r)`, axisymmetric | transverse Cartesian Milne `(τ, x, y)`, boost-invariant |
| charm / IS2 sector | yes (`main2IS2.jl` and the other `main2*.jl`) | **no** |
| tests | `test/runtests.jl` (`Pkg.test()`, in CI) | `test/run2d_gates.jl` (18 gates; only the ~10 s fast tier is in CI — see below) |

`main2D.jl` includes exactly **five** files from `src/` — `constants.jl`, `utils.jl`, `eos.jl`,
`primitives.jl` (transport-coefficient models only) and `relaxation_laws.jl`. Everything that knows
about dimensionality — grid, primitive recovery, fluxes, reconstruction, RHS, timestepper,
dissipation, floors, boundary conditions — is **re-implemented** in `src2d/`.

That duplication is deliberate, and `main2D.jl`'s header states the reason: *"main.jl and src/ are
NOT touched by this module — the 1-D production path stays bit-identical so every published number
remains reproducible."* Three gates police the overlap rather than the source sharing it:
`test_primrec2d_vs_1d.jl`, `test_reproduction2d.jl` and `test_dissipative_vs_1d.jl` compare the two
solvers on a common locus. `src2d/transport2d.jl` also re-states two pure functions from
`src/dissipation.jl` (noted in `test_charge2d.jl`).

The 2-D solver reads **no ENV variables at all** — it is configured purely by keyword arguments,
unlike the 1-D bulk and charm solvers. This is why `ENV_FLAGS.md` lists nothing from `src2d/`.

⚠ **`FiVo2DIdeal` is a third, unrelated codebase** (`Julia/FiVo2DIdeal.jl`, its own submodule and its
own git repo): flat Minkowski, ideal only, one 839-line file, `Printf` its only dependency. It shares
no code with this package. "2-D FiVo" is ambiguous between it and `hydro2d` — say which you mean.

Design, derivation and the full 2-D validation record: `TWOD_PROGRAM.md` (a chronological build log,
not a reference manual — sections are in the order they were written, including retractions).

## Entry points

| file | module | physics | primary consumer |
|---|---|---|---|
| `main.jl` | `hydro` | **bulk** fluid + charge: `run_sim_ideal_diff_visc(; kwargs…)` (library) / `main()` (CLI, ENV-driven). Includes all of `src/`. | DPM `bulk_background{solver=fivo}`, `bulk_background_oo` via `Projects/LangevinPaper{1,OO}/generate_physical_background_fivo.jl`; FiVoBenchmark |
| `main2IS2.jl` | `hydro_current_IS2` | **charm current, Israel–Stewart 5-field** `(α, ν^r, π_Q^{rr}, π_Q^⊥, Π_Q)` on a frozen bulk: `run_static_IS2_test(; background_file, …)`. Most-consumed entry (≈56 call sites). | DPM `charm_hydro_oo`; `Projects/LangevinPaper{1,OO}/is2_dropin.jl`; CMExperiment |
| `main2.jl` | `hydro_current` | charm current, first-order (ν^r relaxes to NS): `solve_current_only` | AttractorPaper1/5, AttractorFoundations |
| `main2M1.jl` | `hydro_current_M1` | charm current as a 3-field **maximum-entropy (M1)** moment system, no transport coefficients, no regulators: `run_static_M1_test`, `solve_M1` | `Projects/LangevinPaper1/m1_dropin.jl`, `Tex/HeavyQuarkHydro/diag_m1_*` |
| `main2M2.jl` | `hydro_current_M2` | four-field MaxEnt **M2** system (parallel to M1): `run_static_M2_test` | `m2_dropin.jl`, `Tex/HeavyQuarkHydro/plot_m2_*` |
| `main2BDNK.jl` | `hydro_current_bdnk` | BDNK current-only (keeps `∂_τ α`, `σ_T/σ_a`); has an `EPS_NU=density_frame` branch | CrossSolverComparison, `Tex/ModePaper1` |
| `mainBDNK.jl` / `mainBDNK_causal.jl` | — / `bdnk_causal` | BDNK bulk driver `run_sim_bdnk`; the genuinely causal telegraph variant | FiVoBenchmark `bench_bdnk_*` |
| `mainDensityFrame.jl` | — | thin driver: `charge_mode=:density_frame` (ν-less parabolic flux) then `hydro.main()` | DensityFrame project |
| `mainJonly.jl`, `mainJonly2nd.jl` | — | charge-only variants used by AttractorPaper5 / `Julia/tools/diag_piQ_*` | legacy |
| `mainBGonly.jl` | `hydro_bgonly` | background-only copy of `main.jl` (≈900 duplicated lines) — kept: it is MainFiVo's `:background_only` pipeline mode (`Code/case_specs.jl:81`) | MainFiVo |
| **`main2D.jl`** | **`hydro2d`** | **the 2+1D solver** (transverse Cartesian, boost-invariant Milne): bulk + charge `(T, u^x, u^y, Π, π^{xx}, π^{xy}, π^{yy}, π^{ηη}, n, ν^x, ν^y)` via `run_sim_2d!`. Includes all of `src2d/`. **No charm/IS2 sector.** | `tools/export_background2d.jl`; the 18-gate ladder `test/run2d_gates.jl` |
| `src/FiVoHydro.jl` | `FiVoHydro` | package wrapper: includes `main.jl`, exports `hydro` (only `Projects/SoftPionPaper` uses `import FiVoHydro`) | — |

`src/` (28 files, incl. the `FiVoHydro.jl` package shim) is the bulk solver: `eos.jl` (ConformalHQEOS, LatticeHRGEOS, TabulatedHQEOS),
`primitives.jl` (`IdealDiffViscModel`, viscosity models), `primrec.jl` (3-unknown Newton recovery),
`fluxes.jl`/`reconstruction.jl`/`rhs.jl`/`timestepper.jl` (the FV scheme), `dissipation.jl` (transport
coefficients, `relax_dissipative!`, all stabilizers), `mood.jl`, `floors.jl`, `io.jl`, `gubser.jl`
(analytic Gubser + `initialize_gubser!`), `is2_second_moment_builder.jl` (IS2 matrices),
`runtime_flags.jl` (the single `HYDRO_*` ENV reader). `relaxation_laws.jl` is a *reference*
exact-exponential integrator used by tests/benches, not a solver hook.

## Quickstart

Bulk solve (the production call, from `generate_physical_background_fivo.jl`):

```julia
include("Julia/FiVoHydro.jl/main.jl"); using .hydro
hydro.run_sim_ideal_diff_visc(
    outdir="scratch/run1", Nr=800, rmax=20.0, τ0=0.4, τfinal=13.0, dump_dt=0.1,
    init_csv="data/fivo_bulk_ic_pbpb.csv",             # columns r,T,ur,(alpha|n),...
    eos=hydro.LatticeHRGEOS(),
    enable_diff=true,  DsT=0.1163,                      # D_s·T [GeV·fm]; κ = DsT·n/(T·fmGeV)
    enable_shear=true, eta_over_s=0.1, tauShear_coeff=0.2,
    enable_bulk=true,  zeta_over_s=0.1, tauPi_coeff=15.0,
    deltaShear_factor=4/3, taupi_pi_factor=0.0,         # δ_ππ damping that matches Fluidum MIS (2026-07-02)
    CFL=0.15, time_integrator=:ssprk2)
```
Output: `snapshot_tau_*.csv` (+ `_meta.csv`) per dump in `outdir`; with `postprocess=true` also a
Langevin-style `hydro_currents_*.jld2` and spline bundle.

Charm IS2 on a frozen background (`background_file` is a JLD2 with `r_grid`, `t_grid`, `T_spline`,
`ur_spline` and optionally `alpha`/`nur`/`kappa`/`tau_diff` splines — see `load_IS2_background`):

```julia
include("Julia/FiVoHydro.jl/main2IS2.jl"); using .hydro_current_IS2
res = hydro_current_IS2.run_static_IS2_test(; background_file="bg.jld2", DsT=0.1163,
          τ0=0.4, τfinal=8.0, Nr=1000, rmax=25.0, CFL=0.15, CFLτ=0.03, dump_dt=0.1,
          use_cM=false, eos=hydro_current_IS2.LatticeHRGEOS(canon_factor=1.0))
res["r_grid"], res["t_grid"], res["n"], res["nur"], res["alpha"], res["piQr"], res["piQperp"], res["PiQ"]
res["diagnostics"]   # steps, linear/eigen failures, nu_bound_hits, cone_projections, jtau_nonpositive
```
Every IS2 regulator is an ENV-overridable `const` read at module load (table below); production sets
them in the DPM recipe (`Projects/LangevinPaper1/dpm_recipes.jl`, `charm_hydro_oo`).

Tests / benchmarks:

```sh
julia --project=Julia/FiVoHydro.jl -e 'using Pkg; Pkg.test()'            # ~1 min; FIVOHYDRO_LONG_TESTS=1 adds the heavy IS2 run
julia --project=Julia/FiVoHydro.jl Julia/Projects/FiVoBenchmark/run_all_benchmarks.jl   # ~5 min, 42 PASS, writes BENCHMARK_REPORT.md, exits 1 on FAIL/ERROR
julia --project=Julia Julia/Projects/FiVoBenchmark/plot_benchmarks.jl
julia Julia/FiVoHydro.jl/tools/list_env_flags.jl > Julia/FiVoHydro.jl/ENV_FLAGS.md            # regenerate the flag table
```
`test/runtests.jl` runs the in-process smoke/EOS/relaxation tests and then, as isolated subprocesses,
`test_is2_drive.jl` (the intra-cell drive term), `test_density_frame_flux.jl`, `test_bdnk_causal.jl`,
`test_is2_causality.jl` and `test_m1_gates.jl` (the 9-gate M1 ladder). CI: `.github/workflows/ci.yml`.

**There are two test ladders, and CI runs only the first.**

```sh
julia --project=Julia/FiVoHydro.jl -e 'using Pkg; Pkg.test()'                    # 1-D, ~1 min, IN CI
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run2d_gates.jl  # 2-D, 18 gates, NOT in CI
```

`test/run2d_gates.jl` is the whole 2+1D validation ladder — shear-closure algebra, primitive recovery
(and recovery vs 1-D on the production locus), G0/G0b Bjorken, G1/G1v Gubser ideal and viscous, Gs
sound, G2 shear+bulk, G3/G3g charge, Gk charge dispersion, G4 reproduction vs 1-D production,
G5 all-sectors production IC, G6 non-axisymmetric, G7 dissipative vs 1-D, G8 un-averaged and G9
fluctuating ICs. Each gate runs in its own subprocess (several include both `main.jl`-side files and
`main2D.jl`, whose modules would collide in one session) and the script exits 1 if any fails, so it
is already usable as a CI step.

Last full run: **18/18 PASS, ≈70 min** (2026-09-08). The full ladder is too slow to put in CI as it
stands — ~70 min against ~1 min for `Pkg.test()` — so CI runs only `FIVO2D_TIER=fast`: the three
algebra and primitive-recovery gates, **~10 s**, no time evolution. Treat that as a smoke test; it
cannot see the timestepper, the fluxes or the regulators. **Run the full ladder by hand after
touching `src2d/` or `main2D.jl`.** (Per-gate cost: the three fast gates are 2.4 / 5.2 / 3.5 s, then
G0 Bjorken alone is 70 s — that cliff is where the fast tier stops. Splitting the remainder behind a
schedule or a label is the obvious next step and has not been done.) What each gate is for and what
it measured: `TWOD_PROGRAM.md` §6.

## Stabilizers (bulk solver) — all OFF by default

The shipped defaults exercise the bare scheme; FiVoBenchmark's D1 audit shows the validated
observables are stabilizer-independent. Production bulk runs enable only what the DPM recipe passes.

| kwarg (`run_sim_ideal_diff_visc`) | default | what it does | where |
|---|---|---|---|
| `nur_clip_factor` | `-1` (off) | clip `\|ν^r\| ≤ f·n_smoothed·u^τ` after relaxation | `src/dissipation.jl` `relax_dissipative!` (end) |
| `alpha_filter_eps`, `nur_filter_eps`, `visc_filter_eps` | `0` | Kreiss–Oliger filter strength, clamped to `[0, 0.24]` | `smooth_alpha!` etc. |
| `alpha_smooth_len`, `nur_smooth_len`, `visc_smooth_len` | `0` | smoothing length → ε via `eps_from_len(·, dr)` | same |
| `do_soft_project_nur` | `false` | tanh projection of ν_NS and ν_new into `\|ν^r\| < n u^τ` | `_soft_project_nur_phys` |
| `do_axis_project_nur`, `axis_project_nfit` | `false`, `2` | polynomial axis projection of ν^r | `axis_project_nur_tapered!` |
| `Pi_clip_factor`, `pi_clip_factor` | `-1` (off) | clip bulk/shear relative to the pressure | `relax_dissipative!` |

Production operating points: Pb+Pb bulk (`bulk_background{solver=fivo}`): Nr=800, rmax=20, CFL=0.15,
`deltaShear_factor=4/3`, clips/filters off. O+O bulk (`bulk_background_oo`): nr=500, rmax=6.5,
rdrop=5.0, cfl=0.1, `nurclip=0.3`, diffdt=0.01. The CLI `main()` falls back to
`last_ic_diagnostics.env` (NUR_CLIP_FACTOR=0.98, ALPHA_FILTER_EPS=0.01, VISC_FILTER_EPS=0.01,
PI/SHEAR_CLIP_FACTOR=0.35, TAU_SHEAR_COEFF=0.2, TAU_PI_COEFF=15) when an ENV var is unset — a
different operating point from the kwarg defaults.

## Charm-sector regulators (IS2) — what is on in production

| knob (ENV) | default | production | meaning |
|---|---|---|---|
| `FIVO_VACUUM_N_LO` / `FIVO_VACUUM_N_HI` | `1e-6` / **`2e-3`** (was 1e-2 before 2026-07-26) | Pb+Pb: default | density-gated ramp `w=(n−n_lo)/(n_hi−n_lo)` multiplying the WHOLE RHS ⇒ effective `τ_n/w` in the dilute tail. Calibrated on Pb+Pb (~24 charm quarks). |
| `FIVO_VACUUM_N_REL_HI` / `_LO` | `0` (off) / `1e-4` | O+O: `0.003` | the same ramp relative to the slice maximum — needed on O+O (0.27 charm quarks) where the absolute ramp is wrong in both directions |
| `FIVO_VACUUM_T_HI` / `_LO` | `0` (off) | off | T-gated ramp — **documented negative result**, leaves the warm flat tail undamped |
| `FIVO_IS2_ALPHA_MIN` / `_MAX` / `_SOFT` | `-20` / `200` / `1` | default | softplus floor on α (the old ±200 was an overflow guard mistaken for a bound; 2026-08-03) |
| `FIVO_IS2_NU_BOUND`, `_FRAME`, `_SMOOTH`, `_KNEE` | `0` (off), `"lab"`, `0`, `0.8` | O+O: `0.7` | soft saturation `\|ν^r\| ≤ f·n`. ⚠ `"lab"` is tighter than the physical LRF bound by `u^τ` (≈1.37 on Σ_fo); `"lrf"` is the physical frame. |
| `FIVO_IS2_CONE_PROJECT` | `0` (off) | O+O c_M variant: `0.995` | projection into the admissible cone |
| `FIVO_IS2_JTAU_FLOOR` | `0` | default | floor on J^τ |
| `FIVO_NU_RAPIDITY` | `0` | off | rapidity chart — **do not promote as is** (undone by the α recovery one line later) |
| `FIVO_CM_SIGN` | **`+1`** | default | `-1` was the elliptic (ill-posed) branch — a real bug until 2026-07 |
| `FIVO_IS2_TAUN_DEGENERACY`, `_TAUM_DEGENERACY` | `0` (bare) | bare | τ_n = D_s·z·K₃/K₂ with NO `÷g_hq` (2026-07-16/21); τ_M = τ_n/2 |
| `FIVO_IS2_CAUSAL_COEFF` | `1` (on) | on | `τ_n ← max(τ_n, κ/χ)` keeps the diffusion signal speed ≤ c (never binds for bare τ_n) |
| `FIVO_IS2_TAUN_SCALE`, `FIVO_IS2_TAUPI_REL` | `1`, `0` | default | diagnostic dials; do not combine `TAUN_SCALE≠1` with `FIVO_IS2_USE_CM=1` |
| `FIVO_ORIGIN_ODD_FIRST_ORDER_CELLS` | `4` | default | first-order cells at the axis for the odd fields |

The ramp is a mitigation, not a cure: it masks a real ν^r runaway in the dilute tail (unregulated,
`max\|ν^r\|/n` reaches 2528 c on O+O and the charm drift grows with resolution) and at the same time
damps the physical freeze-out shell. The M1/M2 modules exist because their cone is structural and need
none of this. The full record:

- `main2IS2.jl` header (lines 1–300) — the regulators, their history, and every negative result.
- `Projects/LangevinPaper1/HydroFieldsDiagnostic/README.md` — the `n_hi` scan (clamp binds at r≈8.6 fm below 1e-3).
- `Projects/FiVoFluidumComparison/BARE_TAUN_REGEN.md` — the 6× τ_n convention mismatch and the bare regen.
- `Projects/CMExperiment/CM_PBPB_REPORT.md` — the c_M closure study; the c_M=0 FiVo–Fluidum gap (15–24% on Σ_fo) that exceeds LP1's 10% gate.
- `Projects/FiVoBenchmark/CAUSAL_HYDRO_AUDIT.md` — BDNK/IS2 causality audit; "BDNK" charge sector is parabolic in practice (B-2, open).
- `Tex/HeavyQuarkHydro/MAXENT_M1_PROGRAM.md` — why M1 retires the regulators.
- `Projects/LangevinPaperOO/README.md` — the withdrawn "Fluidum is unstable on O+O" claims (2026-08-18).

## The thermodynamically consistent first moment (full ∇P)

`src/hq_consistent_firstmoment.jl` implements the **full-∇P (thermodynamically consistent) first
moment** for the charm sector, transcribed from the xAct derivation
`Julia/tools/derive_hq_consistent.wls` (7/7 gates) and the twin of Fluidum's
`src/Matrix/HQ_const_BG_consistent.jl`. The shipped first-moment row is the homogeneous-rest-frame
reduction of `(D_s/T) Δ^r_λ ∇_μ T_Q^{μλ} + ν^r = 0`; dropping the homogeneity step adds five source
terms (∇⊥T pressure gradient, `τ_n n a^r` inertial, `τ_n (ν·∇u)^r`, expansion + `D ln h`, geometric
dilution). All five vanish in the frame the shipped derivation was performed in, and on an *ideal*
baryon-free background the first two cancel by Euler — which is why the shipped fugacity drive works
there and fails on a viscous background.

**It is production for O+O.** ⚠ Two things about this are easy to get wrong:

1. **The live switch is not the module's own ENV var.** `main2IS2.jl:345` defines
   `IS2_CONSISTENT_FM = Ref(get(ENV, "FIVO_HQ_CONSISTENT", "0") == "1")`, but **nothing in the repo
   ever sets `FIVO_HQ_CONSISTENT`** — grepping for it finds only its own definition. The switch that
   is actually used is `FIVO_IS2_CONSISTENT`, read by
   `Projects/LangevinPaperOO/is2_dropin.jl:139`, which assigns the `Ref` directly. Two names for one
   flag; only the project-side one is live.
2. **Pb+Pb and O+O reach "consistent" through different codes.** `LP1_CLOSURE=consistent` (the LP1
   default) resolves `charm_hydro_consistent`, which runs **Fluidum's**
   `:HQ_const_BG_consistent_5f` matrix — not this package. `OO_CLOSURE=consistent` (the O+O default)
   resolves `charm_hydro_oo_consistent`, which is **FiVo's** `IS2_CONSISTENT_FM`. So this file is on
   the production path for O+O only.

Scope, and what is *not* corrected:

- The consistent projection corrects the **first** moment only. The second-moment sources stay the
  shipped ones, so a consistent run carries shipped `c_M` sources riding on a consistent `(α, ν^r)`;
  `Tex/MaxEntHydro/diag_lp1_deltacm.jl` and `oo_deltacm_corrected` measure the residual.
- Second-moment **back-coupling is off in production** (`use_cM` defaults to `"0"`). With
  `IS2_CONSISTENT_FM` *and* `use_cM` both on, the matrix `c_M` is zeroed and the coupling is applied
  as a source instead (`main2IS2.jl:958`).
- Not every product has a consistent twin, and the exceptions are named rather than silently falling
  back: the `c_M` variant bundle, the analytic τ₀ second-moment IC (closure-independent by
  construction), and `case == "ideal"` (at `D_sT → 0` there is no current for a drive to act on).
- **The 2-D solver has no consistent first moment**, and no charm sector at all — `main2D.jl` does
  not include `is2_second_moment_builder.jl` or `hq_consistent_firstmoment.jl`. This is a gap in
  `hydro2d`, not on the O+O production path.

Gates: `Tex/MaxEntHydro/diag_fivo_consistent_gates.jl` (24 xAct reference points shared with the
Fluidum gates, the `h == m K₃/K₂` tie, the identity `n + T dn/dT == n h/T`, and a cross-code
comparison against Fluidum's `hqc_matrices` on the real production background);
`Projects/LangevinPaperOO/diag_oo_consistent_control.jl` (charm conserved to 0.4 % over the physical
region). Runner for the Pb+Pb-side comparison: `Tex/MaxEntHydro/run_fivo_consistent.jl`.

## Environment flags

All ENV reads are enumerated by `tools/list_env_flags.jl` into `ENV_FLAGS.md` (151 distinct flags,
with default and `file:line`). Conventions: `FIVO_*` are the charm-solver constants (read once at
module load, so set them before `include`), `HYDRO_*` are the bulk solver's runtime/debug toggles
(single reader `src/runtime_flags.jl`, cached — mutate ENV before the first `hydro_flags()` call), and
the un-prefixed names (`DS_T`, `NUR_CLIP_FACTOR`, `ENABLE_DIFF`, …) are the CLI `main()` knobs with
`last_ic_diagnostics.env` fallbacks.

## Conventions and convention-changing commits

- Units: fm, GeV; `fmGeV = 1/ħc`; κ = D_sT·n/(T·fmGeV). Stored dissipatives are `DISS_SIGN × physical`.
- ν_NS = −κ[(u^τ)² ∂_r α + u^r u^τ ∂_τ α] (the covariant ∂_τα piece is ~2× on a flowing background).
- **τ_n is bare**: D_s·z·K₃/K₂, a moment ratio in which the degeneracy cancels. `39da649` (2026-07-16,
  `diff_tauN_bg`) and `c4fe4a0` (main2IS2) removed a spurious `1/g_hq` that made τ_n 6× too small and
  the diffusion signal speed superluminal above T≈0.48 GeV.
- `9b7d79d` (2026-07-26) `FIVO_VACUUM_N_HI` 1e-2 → 2e-3 (every FiVo charm product re-solved).
- `914356b`/`d5b8ee0` (2026-08-03) α floor −20 (softplus) instead of ±200.
- `FIVO_CM_SIGN` default −1 → +1 (the ill-posed branch was the default until the CMExperiment).
- `deltaShear_factor=4/3` at all production bulk callers (2026-07-02): matches Fluidum MIS shear to 0.1%;
  the library default stays 0 because the Gubser benchmark needs bare MIS.
- 2026-08-21: `run_sim_ideal_diff_visc` `DsT` default 5.24 → 0.24 (every production caller passes it
  explicitly; 0.24 is what `main()`, the benches and the tests use); `axis_project_nur_tapered!` default
  `nfit` 10 → 2 (the kwarg default); `mainBGonly.jl` module renamed `hydro_bgonly`; `test_m1_gates.jl`
  moved into `test/` and wired into `Pkg.test()`; dead `compute_dvdr!`/`_finite_or_nan` removed;
  IS2 results gained `diagnostics["steps"]`.

## Known open items (flagged, not changed)

1. **ν^r bound frame — `f` is not frame-transferable.** `FIVO_IS2_NU_BOUND_FRAME` defaults to `"lab"`,
   which is the physically wrong frame: production's `nu_bound=0.7` is a physical drift bound of
   ≈0.51 c, and the paper must quote it in lab variables. The correct `"lrf"` is implemented but not
   default, because `f=0.7` was calibrated empirically *in the lab frame* against the depletion
   runaway. Measured A/B (O+O bulk, Nr=300, τ 0.4→5, same f=0.7): `max|ν^r|/n` 0.700 → **1.167**
   (i.e. above the admissibility bound the clamp exists to enforce) and `J^τ≤0` cell-steps 579 →
   **1884**. Switching frames therefore needs its own calibration (expect `f_lrf ≈ f/u^τ ≈ 0.5`,
   already runaway-free in the recipe's scan) and re-mints every O+O charm product. See
   `main2IS2.jl:149` and the note at `dpm_recipes.jl` `"nu_bound"`.
2. The absolute ramp (Pb+Pb) vs the relative ramp (O+O): the relative form is not validated on Pb+Pb.
3. c_M=0 FiVo–Fluidum ν^r gap on Σ_fo (15.5% const / 23.9% linear) exceeds LP1's own 10% gate — a
   pre-existing cross-solver issue on the freeze-out contour, independent of c_M (`CM_PBPB_REPORT.md`).
4. Performance: `main2M1.jl` allocates ≈33 MB per time step at Nr=500 (≈7 GB for a τ 0.4→2 solve) and
   `main2IS2.jl` ≈41 MB/step (`bench_perf` cases `m1_500`, `is2_500`) — orders of magnitude above the
   bulk solver. Not a correctness issue.

Resolved since the 2026-08-21 pass (kept here so they are not re-reported as open): the BDNK charge
sector is no longer parabolic — finding B-2R redesigned it into a genuine causal telegraph scheme with
the validated production driver `mainBDNK_causal.jl` (`CAUSAL_HYDRO_AUDIT.md`); and
`Tex/LangevinPaperOO` Fig. `fig:viscous` now divides by a same-system O+O denominator (commit
`f9792a8c`), so the withdrawn "Fluidum is unstable on O+O" claim no longer appears in that manuscript.

See `CLEANUP_CANDIDATES.md` for tracked artifacts and orphaned files, and
`Projects/FiVoBenchmark/BENCHMARK_REPORT.md` for the current validation ladder and timing baseline.

## IC diagnostics and Gubser validation (legacy CLI workflows)

`bench/ic_diagnostics.jl` analyses an initial-profile CSV for smoothness/admissibility and writes a
diagnostics CSV + PNGs (`--input`, `--outdir`, `--taper-width`, `--interp`); `run_ic_diagnostics.sh`
wraps it interactively and writes `last_ic_diagnostics.env`. `bench/gubser_validation.jl --outdir …`
then `bench/gubser_plot.jl --indir …` reproduce the ideal-Gubser convergence figure. Both are
superseded for validation purposes by `Projects/FiVoBenchmark`.
