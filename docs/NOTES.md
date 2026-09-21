# FiVo — notes moved out of the README

Moved here from `README.md` on 2026-09-21, unchanged, so the pointers elsewhere that cite them
("README §7", "README §8", "Environment flags") still land somewhere. Section numbers and `§`
references inside refer to the README as it was then.

---

## What the terms are worth (was §3)

**What the terms are worth** (`examples1d/02_terms.jl`, a viscous fireball, τ = 5 fm, fluid cells only). The
consistent first moment changes the charm current by 4.4× its shipped maximum. Removing the inertial term alone
moves it 9×, the ∇T channel alone 12×, and **both together only 0.6×**: on a near-ideal fluid Euler makes them
cancel as a pair (gate T4). Dropping one of them is therefore a much bigger change than dropping both. The 2+1D
numbers are in `examples2d/06_charm_terms.jl`.

## FiVoBenchmark (was §5)

`Julia/Projects/FiVoBenchmark/` (`run_all_benchmarks.jl`, about 5 min, writes `BENCHMARK_REPORT.md`) is the older
benchmark suite. It runs the scheme through its own harness (`bench_common.jl`) and is wired into no gate. Two of
its checks were repaired on 2026-09-11. The viscous Bjorken check had passed at 5% while the heating was 4× too
small, and is now a gate on the heating itself. The viscous Gubser run used an acausal C_s = 1, which the ∂_τu^r
fix exposed (§8), and now uses 0.2. Re-run at HEAD on 2026-09-14: **48 PASS / 1 CHECK** (a documented historical note) **/ 0 FAIL**, with every
PASS line and every allocation figure identical to the 2026-09-11 report — only the wall times moved (a busier
machine). The ladders above are the validation of record.

## IS2 cost (was §6)

The obvious lever, not taken: rows 3–5 of that system are trivialised (the second moments are evolved by
`_second_moment_eigen_rhs!`), so `B = At⁻¹Ax` is block upper-triangular and its spectrum is that of its 2×2
(α, ν) block plus zeros — a closed form instead of a 5×5 `eigvals`. It would move results at round-off (the CFL
step and the Rusanov dissipation read the speed), so it was left for a deliberate, gated change.

---

## 7. Known limitations — read before quoting a number

| | |
|---|---|
| **1-D bulk products before 2026-09-11 used a ∂_τu^r ~4 % of its value** | every viscous 1-D FiVo background (Pb+Pb, O+O) was made with it. On the Pb+Pb production IC the fix moves u^r by +1–1.6 % in the bulk, Π by 5–9 %, π^η_η by 1–17 % (τ = 4 fm, T > T_fo). `hydro.HYDRO_LEGACY_DTAU_UR[] = true` reproduces the old arithmetic. Nothing has been re-minted |
| **on a fine radial grid the corrected ∂_τu^r rings** | at full strength it makes the operator-split NS target short-wavelength unstable, and **refining `dr` makes it worse** while a 5× smaller Δτ changes nothing — it is a scheme instability, not a discretisation error. Measured on the O+O bulk: stable at dr = 0.026 fm, ringing at dr = 0.013 fm (|d²T| growing 6.9e-5 → 2.6e-4 over τ ∈ [3.4, 4.7]). **The onset is measured, not derived** — a new grid must measure its own; `diag.dtau_ur_q_max` says whether ∂_τu^r is grid-scale at all. Cure: `dtau_u_smooth_len` (fm, default 0 = off = bit-identical to every result before 2026-09-13) band-limits it at a fixed **physical** length. O+O production uses 0.10 fm (`OO_BG_DTAUUSMOOTH`); the Pb+Pb bulk (dr = 0.025 fm) sits just on the stable side and **has not been checked**. Fluidum does not have this — it keeps the ∂_τu^r coefficient inside `A_t` and is grid-converged to dr = 0.0043 fm (`EQUATIONS1D.md` §8) |
| first order in time once anything dissipative is on | the operator split (1-D and 2-D); halve `CFLτ` to check. The IS2 solver is unsplit RK4 |
| the 1-D cold-start recovery with `ConformalHQEOS` fails at α ≲ −20 | its initial guess $T_0 = E^{1/4}$ ignores $a_{SB}\hbar c^{-3}$, and the φ direction is too badly scaled when $n \sim e^{-24}$. `LatticeHRGEOS` converges at every α tried; production is unaffected. Use a charge-free EOS (`ConformalHQEOS(m_hq = 0, g_hq = 0)`) for charge-free tests |
| **the shear sector is acausal for C_s = `tauShear_coeff` > 1/2** | conformal IS needs η/(τ_π(e+P)) = C_s ≤ 1/2, and an acausal IS theory is unstable in a moving frame. With the ∂_τu^r fix the 1-D solver shows it: viscous Gubser runs at C_s ≤ 0.6 and runs away at 0.8 (growing with resolution). The builders warn. ⚠ `run_sim_ideal_diff_visc` still defaults to C_s = 1 (production passes 0.2) |
| the axis cell | **repaired 2026-09-15** (§8). It still carries an O(dr) mismatch between the solver's acceleration and $\nabla T$ (gate T4: halves with each refinement), but the first cell is no longer special-cased: $L_2(T)$ on viscous Gubser at $N_r$ = 800 is **2.78e-05** (was 2.61e-04) and the order in $T$ is **2.12** (was 1.77). `FIVO_AXIS_CELL_EXACT=0` restores the old arithmetic bit for bit |
| the dilute edge | **Israel–Stewart stops being valid before the grid does.** On a Woods–Saxon fireball (τ 0.4 → 3, shear+bulk) $|\pi|/P$ crosses **1 at r = 8.29 fm** — freeze-out is at 7.71 — and reaches **2.9 by r = 13**, where the shear stress is three times the pressure and the gradient expansion carries no content. Any structure out there is the *model* failing, not the solver: the second difference of $u^r$ *grows* with refinement (1.5e-3 → 2.2e-3 → 9.1e-3 at $N_r$ = 400/800/1200) while above $T_{\rm fo}$ it falls (6.6e-4 → 1.8e-4 → 8.5e-5). ⛔ **No numerical knob removes it**, and six were measured and rejected: the timestep (non-monotonic, resolution-dependent optimum, wrong choice gives $\max|u^r|$ = 4.1), filters (up to **17× worse** — they damp the dissipative fields, and it is viscosity that suppresses the front), `dtau_u_smooth_len` (saturates at 6.5e-4), the box size (identical to **six digits** at rmax = 20/30/40 fm), the near-vacuum floor (ten orders away), and axis-face reconstruction. 🔑 **Quote from $T > T_{\rm fo}$, and plot to the validity edge** — floors and the vacuum ramp shape the tail below it |
| the vorticity couplings are off by default | they vanish in 1+1D; in 2+1D see README2D §7 below |
| no thermal fluctuations, no $c_M$ back-coupling in 2+1D | not implemented |

---

## 8. Corrections of 2026-09-11, 09-13 and 09-14

Found while building the 1+1D ladder. Each one is at its code and in `EQUATIONS1D.md` §8:
- **∂_τu^r in the 1-D relaxation (production path).** It came out at ~4% of its value. Against viscous Gubser
  L2(π̄) is 132% with the old history and 1.5% with the fix.
- **SSPRK3 was first order.** Its last stage ran at τ+Δ instead of τ+Δ/2.
- **π was zeroed on every other step for charge-free fluids at rest.** The recovery bisection was capped 15
  iterations short of float resolution.
- **The dormant DNMR couplings.** λ_πΠ and λ_Ππ had the wrong sign and τ_ππ lacked its trace (no caller used
  them).
- **IS2 diagnostic counters** accumulated across solves.
- **FiVoBenchmark's viscous Gubser ran an acausal shear sector** (C_s = 1). It completed only because of the
  ∂_τu^r defect. With the fix it crashes, as acausal IS must; the bench now uses C_s = 0.2.

**2026-09-13.** The 09-11 ∂_τu^r fix is right and the closed-form gates agree with it, but at full strength it
makes the split NS target short-wavelength unstable and the O+O bulk ran into it (§7). `dtau_u_smooth_len`
band-limits the term at a fixed physical length; default 0.0 reproduces everything before that date bit for bit.
The full measurement, the linear estimate that does **not** give a usable threshold, and why Fluidum is immune:
`EQUATIONS1D.md` §8.

**2026-09-14 — `main2D.jl` had an UNDECLARED DEPENDENCY, and no test could see it.** `main2D.jl:27` does
`using DelimitedFiles` (for `readdlm` in `initialize_from_grid_csv!`), and `DelimitedFiles` was in neither
`Project.toml` nor `Manifest.toml`. Interactively it resolves from the default environment, so nothing ever
failed — but `Pkg.test()` runs in a sandbox that does not include it, so **the 2+1D solver could not be loaded
in the package's own declared environment.** `Pkg.test()` never noticed because until today no test loaded
`main2D.jl`; the new IO gate does, and failed on its first run under `Pkg.test()`. Now declared. ⚠ `Plots` is
the same class of problem and is still undeclared — the examples need it and nothing in `src/`, `src2d/` or
`test/` does, so it is called out in the example READMEs instead of added as a dependency of the solver.
⚠ `Manifest.toml` still records `julia_version = 1.11.5` (what CI pins) while this machine runs 1.12.6, so the
`DelimitedFiles` entry was added by hand rather than by a re-resolve; `Pkg` warns that the project hash is
stale. `instantiate` and `test` both pass with the warning.

**2026-09-15 — THE FIRST CELL WAS TREATED AS IF IT WERE ON THE AXIS, and it is not.** The first
physical cell sits at $r = \Delta r/2$. Three places acted as though it sat at $r = 0$:
`apply_bc!` zeroed its conserved radial momentum $S_r$, `rhs!` zeroed its $u^r$, and
`reconstruct_muscl_prims!` left its face first order. On Gubser $u^r(\Delta r/2) = \Delta r/2$
exactly — an **O(Δr) quantity being discarded every step**, which is precisely the first-order error
the cell used to show. The zeroing entered in an early debugging commit and was documented nowhere.

Measured on viscous Gubser (η/s = 0.02, τ = 1 → 2), against the exact time derivative at τ₀ the
solver's RHS in cell 1 was **−18.7 %** and did not improve with resolution; it is now **+4.5e-5**,
the same as the interior. Consequences on the gate:

| | before | after |
|---|---|---|
| $L_2(T)$, $N_r$ = 800 | 2.614e-04 | **2.780e-05** (9.4×) |
| $L_2(\bar\pi)$, $N_r$ = 800 | 1.500e-02 | **1.924e-03** (7.8×) |
| observed order in $T$ | 1.77 | **2.12** |
| cell-1 $\bar\pi$ error at $N_r$ = 400 | 1.56e-03 | **3.96e-06** (394×) |

⚠ **The three act together.** Turning the reconstruction on alone makes cell 1 *worse* (−18.7 % →
−38 %), which is why an earlier attempt at exactly that was measured and rejected.
✅ Well-balancedness is preserved **exactly**: a uniform static fluid still gives $\max|u^r| = 0$ to
machine zero at every resolution. The 1-D ladder is **10/10** with the repair, T4 (the axis Euler
cancellation) and X1 (the diffusion mode through all three solvers) included. The 2+1D solver is
untouched — it is transverse Cartesian and has no axis (`src2d/rhs2d.jl`: "no special
first-physical-cell treatment").
⚠ **This moves 1-D results.** On a Woods–Saxon fireball the change is small (the profile is flat at
the axis: self-convergence 9.74e-4 → 9.41e-4 at $N_r$ = 200), but anything with structure near
$r = 0$ moves. `FIVO_AXIS_CELL_EXACT=0` reproduces every pre-2026-09-15 number bit for bit.

**2026-09-14.** `bench/benchmark.jl`, `bench/gubser_validation.jl` and `bench/ic_diagnostics.jl` had been unable to
construct a model since `do_axis_project_nur` was added: they passed 35 positional fields where the back-compatible
constructors cover 36/37/39. The missing argument was restored in all three. They are the legacy CLI workflows at the
end of this file; the validation of record is still the two ladders (§5).


---

## Production calls

The production drivers, as the DPM recipes call them. For new work use the library interface (§2);
these keep their legacy defaults because ten callers rely on them. Bulk solve (from
`generate_physical_background_fivo.jl`):

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

Tests, the validation ladders and the benchmarks: §5 above.

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

1. **The switch has two names, and only one of them was ever set.** `main2IS2.jl` used to read
   `FIVO_HQ_CONSISTENT` alone, and **nothing in the repo ever set it** — grepping the module's own
   variable name concluded "never enabled anywhere", which is wrong: the live switch is
   `FIVO_IS2_CONSISTENT`, read by `Projects/LangevinPaperOO/is2_dropin.jl:139`, which assigns the
   `Ref` directly. Since 2026-09-08 `main2IS2.jl:358` accepts **both**, `FIVO_IS2_CONSISTENT` first,
   so reading the module no longer suggests a dead switch. The second moment is independent:
   `FIVO_IS2_CONSISTENT_M2`.
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
- **The 2-D solver now has its own consistent first moment** — ⚠ and until 2026-09-10 the 2-D
  relaxation ADDED it where it had to subtract it (every consistent term had the wrong sign; gate Gc7
  now evolves a Bjorken state to catch that; `EQUATIONS2D.md` §10) —
  `src2d/hq_consistent_firstmoment2d.jl`, flag `IdealDiffVisc2DModel.consistent_fm`, default OFF
  (2026-09-08). It is a RE-DERIVATION, not a port: the 1-D function is the same covariant object
  contracted in radial Milne, and retyping it with `x` for `r` is wrong in two specific ways that
  gate `test_consistent_fm2d.jl` exists to catch (both were made, and caught — see
  `TWOD_PROGRAM.md` §6ae). In its 1-D limit it reproduces `hq_consistent_extras` bit for bit.
  (This bullet said until 2026-09-10 that 2-D had no second-moment sector. It has had one since
  2026-09-08: `src2d/hq_consistent_m2_2d.jl`, `consistent_m2`, passive — no `c_M` back-coupling, which
  O+O production runs without in any case.)

Gates: `Tex/MaxEntHydro/diag_fivo_consistent_gates.jl` (24 xAct reference points shared with the
Fluidum gates, the `h == m K₃/K₂` tie, the identity `n + T dn/dT == n h/T`, and a cross-code
comparison against Fluidum's `hqc_matrices` on the real production background);
`Projects/LangevinPaperOO/diag_oo_consistent_control.jl` (charm conserved to 0.4 % over the physical
region). Runner for the Pb+Pb-side comparison: `Tex/MaxEntHydro/run_fivo_consistent.jl`.

## Environment flags

All ENV reads are enumerated by `tools/list_env_flags.jl` into `ENV_FLAGS.md` (169 distinct flags,
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
- **2026-09-11** (EQUATIONS1D.md §8): ∂_τu^r in every 1-D relaxation was ~4 % of its value (the
  production path — see §7); SSPRK3 was first order; charge-free states at rest had π zeroed every other
  step; λ_πΠ/λ_Ππ signs and the τ_ππ trace (default-off couplings); IS2 diagnostic counters were cumulative.
  `run_static_IS2_test` gained `terms`, `consistent_fm`, `consistent_m2` and `background` keywords; the
  1-D bulk model gained `consistent_fm` and `terms`, and on 2026-09-13 `dtau_u_smooth_len` — three trailing
  fields; the 36/37/39-argument positional constructors still work).
- 2026-08-21: `run_sim_ideal_diff_visc` `DsT` default 5.24 → 0.24 (every production caller passes it
  explicitly; 0.24 is what `main()`, the benches and the tests use); `axis_project_nur_tapered!` default
  `nfit` 10 → 2 (the kwarg default); `mainBGonly.jl` module renamed `hydro_bgonly`; `test_m1_gates.jl`
  moved into `test/` and wired into `Pkg.test()`; dead `compute_dvdr!`/`_finite_or_nan` removed;
  IS2 results gained `diagnostics["steps"]`.

## Known open items (flagged, not changed)

(§7 above has the limitations found on 2026-09-11; this list predates it.)

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

⚠ **These CLI paths write outside the package by default.** `main.jl`'s `main()` (the CLI entry, not
the library API) resolves `HYDRO_OUTDIR` to `../../Julia/Plot/snapshots/FiVo` — two levels above the
clone, a directory that exists in the research repository this package is developed in and **not in a
standalone clone**. Set `HYDRO_OUTDIR` explicitly before using them. The library API
(`run_sim_1d!`, `run_sim_2d!`, `run_static_IS2_test`) writes nothing unless you ask it to, and is
unaffected.

⚠ `last_ic_diagnostics.env` at the package root **is not a stray artifact**, despite the
"Autogenerated" header: `main()` falls back to it for every CLI knob that has no environment variable
set (`main.jl:850`), so deleting it silently changes CLI defaults. It is tracked deliberately.

---

# From README2D.md (moved 2026-09-21)

## README2D §5, outside this package

Outside this package, in `Julia/Projects/FiVoFluidumComparison/`:

| | |
|---|---|
| `gate_2p1d_viscous.jl` | medium viscous rows vs a referee that is neither code (1e-11); commit-level in `programme.jl check` |
| `gate_transverse_fm.jl` | the transverse first moment vs a closed form with real transverse structure, supplied by neither code, **both codes**. FiVo converges to it: 1.05 → 0.53 → 0.43 % at N = 32/48/64, worst single cell 0.8 %, `cos` = 1.000000 at every resolution; commit-level |
| `gate_2p1d_m2.jl` | second-moment rows, FiVo vs Fluidum at identical states |
| `test/test_diffusion_mode.jl` (X1, the 1-D ladder) | the charm current on a radial diffusion mode: **this solver, the 1-D bulk solver and the 1-D IS2 solver** against one closed-form referee, four term configurations |
| `COMPARISON_2P1D.md` | the solve-vs-solve record (a real Pb+Pb event to freeze-out, §37) |
| `compare_examples_1p1d.jl`, `COMPARISON_1P1D.md` | the **1+1D** medium against Fluidum on the worked-example configurations — 15/15, and the first such comparison of that sector |

## README2D §6, the performance pass

**Measured 2026-09-11 after the performance pass** (`TWOD_PROGRAM.md` §6ai), which made the solver
**3.4× faster** — 60 % of its runtime had been a single Bessel function inside the equation of
state. On the same machine the earlier code gave 1880 ns/cell-step at N = 300 against 582 now.
The benchmark does a warm-up run first (without it the first row carries the solver's compilation
and reads 6× high) and takes the best of three for the per-sector rows (with one repetition that
section reported diffusion as *cheaper* than ideal).

## README2D §7, known limitations

| | |
|---|---|
| **2-D `consistent_fm` results before 2026-09-10 are wrong** | every consistent term had the wrong sign (EQUATIONS2D §10) |
| the charm second moment lacked its projector term before 2026-09-10 | 0.02–0.6 % of the rate; `terms = (m2_projector = false,)` reproduces the old rows. Fluidum's twin lacked it too until 2026-09-11, when both the projector (default ON) and the vorticity coupling (default OFF) were derived and added to `Fluidum.jl/src/Matrix/HQ_2p1d_BG_m2.jl`; the two codes now agree on genuinely 2-D states (`FiVoFluidumComparison/gate_2p1d_m2_newterms.jl`) |
| the vorticity couplings are off by default | `m2_vorticity`: up to 12 % of the σ coupling in the worst cells of a lumpy event; neither code carried it before. `shear_vorticity` (the medium's, 2026-09-11): not carried by Fluidum |
| the medium's DNMR couplings τ_ππ, λ_πΠ, δ_ΠΠ, λ_Ππ | default 0; wired since 2026-09-11 with the 1+1D solver's equations and signs (gate Gd). Only δ_ππ is set by the production callers |
| first order in time once anything dissipative is on | the operator split; halve `CFLτ` to check |
| the dilute edge | the vacuum ramp throttles the charge sector below $n = 2\cdot10^{-3}$ fm⁻³, which reaches above $T_{\rm fo}$; quote dissipative fields from $T > T_{\rm fo}$ only |
| no diffusive signal speed in the CFL | `wavespeeds_2d` bounds the HLLE fan by the sound speed. COMPARISON_2P1D §28 measured `\|ν\|/n` climbing to 0.93 at the edge with `consistent_fm` on — but under the wrong sign, whose expansion term anti-damped the current. Not re-measured since the fix |
| `consistent_fm` defaults differ **between the codes** | `build_model_2d` defaults to **false**; Fluidum's 2-D second-moment driver `matrix2d_HQ_BG_m2!` defaults `consistent_fm` to **true**, and its other two 2-D charm drivers (`matrix2d_HQ_BG!`, `matrix2d_visc_HQ_BG!`) carry the shipped ∇α row with no switch at all. Set it explicitly on both sides |
| `deltaShear_factor` differs between the two 1-D **entry points** | `build_model_1d` and `build_model_2d` both default to 4/3; the legacy `run_sim_ideal_diff_visc` defaults to 0 (production passes 4/3). Set it explicitly in any comparison |
| `transport_mass` is only partly decoupled | $\tau_n$, $h$, $h'$ still use the EoS mass |
| no thermal fluctuations, no $c_M$ back-coupling | not implemented |
| the 2-D charm density is not bit-identical to `eos_Pne` | since 2026-09-11 the 2-D EOS uses a fast $K_2$ (19× faster, agreeing to 1.6e-15). Fields move ≤ 4e-14 absolute on a solve; P and e are bit-identical, and `src/eos.jl` (the 1-D production path) is untouched |

The full list of what was found and fixed, with measurements: `TWOD_PROGRAM.md` (D1–D16, §6) and
`EQUATIONS2D.md` §10.
