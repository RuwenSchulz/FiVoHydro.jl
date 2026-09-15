# FiVo 2+1D — `main2D.jl` / module `hydro2d`

A finite-volume solver for **boost-invariant, transversely Cartesian** relativistic viscous
hydrodynamics in Milne coordinates $(\tau, x, y, \eta_s)$. It carries the medium, the
Israel–Stewart shear and bulk stresses, and a diffusing heavy-quark (charm) charge. Optionally it
also carries the **thermodynamically consistent** charm first moment and a passive charm second
moment. It is the 2-D sibling of the 1+1D radial solver in `main.jl` and shares six
dimension-agnostic files with it, plus the common I/O layer `src/fields_io.jl`
(`README.md`, "How the solvers share code").

| document | what it is for |
|---|---|
| **this file** | how to use the solver: quickstart, every knob, the examples, the validation, the cost |
| [`EQUATIONS2D.md`](EQUATIONS2D.md) | every equation, term by term — formula, code, switch, gate — and the corrections log |
| [`examples2d/`](examples2d/README.md) | ten runnable examples, each seconds to minutes, each with a figure; four also write animations |
| [`TWOD_PROGRAM.md`](TWOD_PROGRAM.md) | the chronological build log: every derivation, measurement and retraction |

**Status (2026-09-14).** The ladder is 22 gates, **22/22 on 2026-09-14**: two analytic solutions (Bjorken,
Gubser), sound propagation, reproduction of the 1-D production solver, closed-form referees for the
charm sector, and cross-code agreement with Fluidum. Not on any manuscript's production path; used
by `Projects/FiVoFluidumComparison` and one O+O animation. Read §7 before quoting a number from
the dilute edge or from a charm closure.

---

## 1. Quickstart

From the repository root, in a REPL started with `julia -t auto --project=Julia/FiVoHydro.jl`:

```julia
include("Julia/FiVoHydro.jl/main2D.jl"); using .hydro2d; const H = hydro2d

g = H.make_grid2d(96, 96; xmax = 12.0, ymax = 12.0)       # ±12 fm, 96² cells, 3 ghost layers
m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s  = 0.10,           # τ_π = η/(C_s T s), C_s = tauShear_coeff
        enable_bulk  = true, zeta_over_s = 0.10,
        enable_diff  = true, kappa_coeff = 0.1163,         # D_s·T
        consistent_fm = true)                              # the full-∇P charm first moment
H.show_equations(m)                                        # what this model integrates, term by term

U = H.allocate_state(g, m)
for ix in 1:g.Nxtot, iy in 1:g.Nytot                       # an elliptic fireball at rest, τ0 = 0.4
    x, y = g.xC[ix], g.yC[iy]
    T = 0.05 + 0.40*exp(-(x^2/1.3 + y^2/0.8)/(2*2.5^2))
    H.set_cell!(U, H.lin(g, ix, iy), T, -4.0, 0.0, 0.0, 0.4, m)   # (T, α = μ/T, u^x, u^y)
end
H.finalize_ic!(U, g, m; τ0 = 0.4)                          # floors + admissibility — never skip

res = H.run_sim_2d!(U, g, m; τ0 = 0.4, τfinal = 4.0)
res.ok || error("step failed at τ = $(res.τ)")
f = H.fields_2d(g, U, res.work, m)                         # every field as an Nx×Ny matrix
println("T_max = ", maximum(f.T), " GeV,  max|u| = ", res.maxu, ",  charge drift = ", res.dQ)
```

About 8 s including compilation. `f` has `x, y, T, mu, alpha, n, e, P, ux, uy` and every dissipative
field the model carries (`nux, nuy, Pi, pixx, pixy, piyy, pieta, pQxx, …, PiQ`), in physical units.

**Three things that bite** (each has cost a debugging session here):

1. `finalize_ic!` is not optional after `set_cell!`: a vacuum tail starts below the energy floor.
   `initialize_uniform!`, `initialize_from_radial!` and `initialize_from_grid_csv!` call it for you.
2. `res.ok` means "every step completed", not "the result is physical". Read `res.maxu`, `res.dQ` and
   `res.minPtot` as well, and `|π|/P`, `|ν|/n` over cells **above** freeze-out.
3. Julia soft scope: an accumulator updated inside a *top-level* `for` is a new local every
   iteration. Put loops in functions (`CLAUDE.md`, trap 1).

---

## 2. The model — every knob

`build_model_2d(; kwargs...)` builds an `IdealDiffVisc2DModel` with a state layout matching the
enabled sectors (an absent sector's fields are not in `U` at all). **Defaults are the bare ideal
scheme**; every sector and every regulator is opt-in. Any field below can be passed as a keyword.

**Sectors and coefficients**

| keyword | default | meaning |
|---|---|---|
| `eos` | `LatticeHRGEOS()` | equation of state (`ConformalHQEOS()` for the conformal tests) |
| `with_charge` | `true` | carry the conserved charge $\tilde D = \tau J^\tau$ at all |
| `enable_shear`, `eta_over_s`, `tauShear_coeff` | `false`, 0.1, 0.2 | $\eta = (\eta/s)s$, $\tau_\pi = \eta/(C_s T s)$ |
| `deltaShear_factor` | **4/3** | $\delta_{\pi\pi}/\tau_\pi$. ⚠ `build_model_1d` agrees (4/3 since 2026-09-11), but the legacy 1-D entry `run_sim_ideal_diff_visc` still defaults to **0** — production passes 4/3 explicitly. Fluidum has no such term |
| `enable_bulk`, `zeta_over_s`, `tauPi_coeff` | `false`, 0.1, 15.0 | peaked $\zeta/s$ at $T = 0.175$ GeV, $\tau_\Pi$ from $C_\zeta$ |
| `enable_diff`, `kappa_coeff` | `false`, **0.0** | charge diffusion with $D_sT$ = `kappa_coeff`. ⚠ set it: 0 means no diffusion at all |
| `consistent_fm` | `false` | the full-∇P first moment (four extra terms, EQUATIONS2D §4) |
| `consistent_m2` | `false` | the passive charm second moment $(\pi_Q^{ij}, \Pi_Q)$ (EQUATIONS2D §5) |
| `terms` | `:default` | per-term switches: a preset (`:homogeneous`, `:full`, `:none`), `without(:acceleration, …)`, or a NamedTuple such as `(fm_inertial = false,)` — §3 |
| `transport_mass` | 0.0 (= EoS mass) | charm mass in the second moment's Bessel ratios and $\tau_M$ |
| `tauN_coeff`, `deltaN_factor` | 1.0, 0.0 | $\tau_n$ multiplier (refused with `consistent_fm`), $\delta_{nn}/\tau_n$ |

**Numerics and switches that are part of the equations**

| keyword | default | meaning |
|---|---|---|
| `shear_projected_deriv`, `diff_projected_deriv` | `true` | the projector correction on $D\pi$ and $D\nu$ |
| `relax_advect_pi`, `relax_advect_Pi`, `relax_advect_nu`, `relax_advect_m2` | `true` | transverse advection of the dissipative fields, upwinded inside the relaxation |
| `advect_pi`, `advect_Pi`, `advect_nu` | `false` | the alternative: advect them as passive scalars in the flux. Unstable for the charge sector on the production IC — leave off |
| `shear_constraint` | `:project` | restore tracelessness each step (`:monitor` only measures it) |

**Regulators — they act where they bind; see EQUATIONS2D §8**

| keyword | default | meaning |
|---|---|---|
| `vacuum_n_lo`, `vacuum_n_hi`, `vacuum_ramp_relax` | 1e-6, 2e-3 fm⁻³, `true` | density-gated ramp on the charge drive and on $\tau_n$ |
| `T_vac_cut` (or `E_vac_cut`) | 0.05 GeV | colder cells are vacuum |
| `pi_clip_factor`, `Pi_clip_factor`, `nu_clip_factor` | −1 (off) | $\lvert\pi\rvert, \lvert\Pi\rvert \le fP$, $\lvert\nu\rvert \le f\,n\,u^\tau$. Use `pi_clip_factor = 1` on lumpy events |
| `r_domain` | `Inf` | evolve only the disc $r \le r_{\rm domain}$ |

**Second-order medium couplings** (DNMR; all default 0, EQUATIONS2D §2–3, gate Gd): `taupi_pi_factor`
(τ_ππ/τ_π), `lambda_pi_Pi_factor` (λ_πΠ/τ_π), `deltaPi_factor` (δ_ΠΠ/τ_Π), `lambda_Pi_pi_factor` (λ_Ππ/τ_Π).

Refused at non-default values, because nothing in `src2d/` reads them (they exist for signature
parity with the 1-D model): `diffusion_drive`, `diff_dt_coeff`, `shear_dt_coeff`, `bulk_dt_coeff`.

**The driver**

```julia
res = run_sim_2d!(U, g, m; τ0 = 0.4, τfinal = 13.0, CFL = 0.2, CFLτ = 0.05,
                  integrator = :ssprk2,          # or :ssprk3 (convergence studies; no MOOD)
                  bc = :outflow,                 # or :periodic
                  on_dump = (τ, U, work) -> …, dump_dt = 0.5,
                  work = nothing, reset_history = (work === nothing))
# res: ok, τ, nsteps, nprimfail, nvacuum, max_shear_res, work, maxu, dQ, Q0, Q1, minPtot
```

To continue a run, pass `work = res.work, reset_history = false`: the $\partial_\tau$ history lives in
the work array, and a restart without it drops the $\partial_\tau$ pieces of the NS targets for one step.

**Initial conditions**

| | |
|---|---|
| `initialize_uniform!(U, g, m, τ0; T0, alpha0)` | a transversely uniform state (Bjorken) |
| `initialize_from_radial!(U, g, m, τ0, Tof, αof)` | from radial profiles `T(r)`, `α(r)` — the repo's production ICs are radial |
| `initialize_from_grid_csv!(U, g, m, τ0, csv)` | from an `x,y,T0,alpha0` grid, as `Projects/ALICE_IC_Creation/BuildIC2D.jl` writes (real ε₂, ε₃) |
| `set_cell!(U, i, T, α, ux, uy, τ, m; Pi, pixx, pixy, piyy, nux, nuy)` then `finalize_ic!` | anything else |

---

## 3. Switching individual terms off

Whole sectors switch with the `enable_*` / `consistent_*` flags, and a few medium terms with the knobs
above. Every other term has its own switch in the shared register `Terms` (`src/terms.jl`, the same one
the 1+1D solvers use; `Terms2D` is its alias here). The interface is one screen, in `README.md` §3:

```julia
m = H.build_model_2d(; enable_diff = true, kappa_coeff = 0.1163,
                       consistent_fm = true, consistent_m2 = true,
                       terms = (fm_inertial = false, m2_vorticity = true))
m = H.build_model_2d(; …, terms = :homogeneous)                       # a homogeneous medium at rest
m = H.build_model_2d(; …, terms = H.without(:vorticity, :acceleration)) # drop by physical ingredient
H.show_equations(m)
```

`show_equations` prints every equation with each term marked `[x]` or `[ ]` and the knob that
controls it; `show_terms()` prints the register with each term's ingredients. The switches:

| first moment | second moment |
|---|---|
| `nu_gradalpha` — the fugacity drive $-\kappa\nabla^{\langle i\rangle}\alpha$ | `m2_nu_gradient` — $2\eta_Q\sigma_{(\nu)}$, $\zeta_Q\theta_{(\nu)}$ |
| `fm_gradT` — pressure gradient, T channel | `m2_bg_gradu` — $2\bar\eta\sigma$, $\tfrac53\bar\eta\theta$ |
| `fm_inertial` — $\tau_n n a^i$ | `m2_bg_DlnT`, `m2_bg_Dalpha` — the trace drives |
| `fm_nu_gradu` — $\tau_n\nu^m\nabla_m u^i$ | `m2_expansion`, `m2_pi_sigma`, `m2_PiQ_sigma` — class (iii) |
| `fm_expansion` — $\tau_n\theta\nu^i$ | `m2_vorticity` — **off by default**, class (iii) |
| `fm_dlnh` — $(D_s/T)h'DT\,\nu^i$ | `m2_projector` — the Δ-projector on $D\pi_Q$ |
| **medium:** `shear_vorticity` — $2\tau_\pi\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle}$, **off by default** | `m2_accel_nu`, `m2_nu_gradTh` — class (iv) |

A switch NAMED in a sector that is off is **refused**, not ignored, and so is a misspelled name; a change that
comes from a preset or an ingredient in an off sector is inert and reset. Defaults are the equations as they
stand; switching a term off reproduces the arithmetic without it bit for bit (gate Gt). `terms = :homogeneous`
with `consistent_fm = true` reproduces `consistent_fm = false` bit for bit (Gt7). What each term is worth on a
fireball: example 06.

**The medium's vorticity coupling** (`shear_vorticity`, added 2026-09-11) is the medium twin of `m2_vorticity`:
the same function, `vorticity_coupling_2d`, with τ_π for τ_M — the antisymmetric half of the kinematic
contraction the Boltzmann streaming term produces, with a fixed coefficient (DNMR's $+2\tau_\pi\pi_\lambda^{\langle\mu}\omega^{\nu\rangle\lambda}$).
Gt8: inert on a round fireball (to the O(h²) vorticity a Cartesian stencil manufactures), acts on a swirling
one (795× more), and only rotates π ($\pi_{\mu\nu}X^{\mu\nu} = 0$ to 5e-15). Fluidum does not carry it; off by default.

---

## 4. Examples

```sh
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/06_charm_terms.jl
```

| | what it shows | time |
|---|---|---|
| 01 first run | the whole API on a uniform state, checked against the 0+1D equations; why a viscous run is first order | 12 s |
| 02 elliptic flow | ε₂ → momentum anisotropy, sector by sector, with the ε₂ = 0 control | 34 s |
| 03 choosing a resolution | a three-grid Richardson study; fields, observables and conserved quantities converge differently | 52 s |
| 04 a fluctuating event | v₃ from lumps, event by event, about the participant plane | 1.7 min |
| 05 the dissipative sectors | what each sector changes and costs, how big \|π\|/P gets, whether the regulators are inert | 1.2 min |
| 06 the charm terms | `show_equations`, then every closure term removed one at a time and what it moves | 18 s |
| 07 a real event | one un-averaged MC-Glauber Pb+Pb event to freeze-out, raw vs smoothed (`smooth_fm`); **animations** | 40 s |
| 08 vorticity on vs off | `m2_vorticity` measured on that event: \|ω\|/\|σ\|, and what it moves in π_Q; **animation**. `EX08_N=480`/`640` render it at 3×/4×, and the answer **converges** (median 3.01e-2 → 2.84e-2 → 2.79e-2) | 30 s |
| 09 Gubser flow | the solver against an exact solution, and the convergence order; **animation** | 1.5 min |
| 10 the showcase | one real Pb+Pb event, every sector on and `m2_vorticity = true`, run long at high resolution and rendered for a slide rather than for a table; **animation**. `EX10_N`/`EX10_TAUF`/`EX10_NFRAME` override the defaults | 72 s |

Times are the suite baseline (`Julia/Projects/suite_baseline.toml`), re-recorded 2026-09-11 after the
performance pass (10 on 2026-09-14) — the ten together are 591 s, just under 10 minutes. All ten pass: **13/13
FiVo examples on 2026-09-14**, the three 1+1D ones included. The suite runs them all with
`julia --project=Julia Julia/Projects/run_suite.jl examples --only FiVoHydro`.

---

## 5. Validation

```sh
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run2d_gates.jl           # 22 gates, ~35 min (22/22 on 2026-09-14)
FIVO2D_TIER=fast julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run2d_gates.jl   # 6 gates, ~30 s, in CI
```

Each gate runs in its own process; the runner exits 1 on any failure, and a gate file that has gone
missing counts as a failure. Run the full ladder after touching `src2d/` or `main2D.jl`.

| gate | file | checks against |
|---|---|---|
| shear algebra | `test_shear2d_algebra.jl` | orthogonality/trace identities, an independent closure |
| recovery | `test_primrec2d.jl`, `test_primrec2d_vs_1d.jl` | round trips; the 1-D recovery on the production locus |
| **Gc** first moment | `test_consistent_fm2d.jl` | the 1-D source bit for bit; rotation; Bjorken; **Gc7: the sign, on a solve** |
| **Gm** second moment | `test_consistent_m22d.jl` | the 1-D reduction (4e-16, with the moving projection); rotation; trace; Bjorken |
| **Gd** DNMR couplings | `test_dnmr2d.jl` | the 0+1D DNMR ODEs (Richardson-extrapolated), wrong-sign referees, the 1-D solver, and the π·σ contractions vs brute force |
| **Gt** term switches | `test_terms2d.jl` | every switch wired and additive; the vorticity coupling vs brute-force index algebra; Gt7 presets/`without` and `:homogeneous` ≡ shipped on a solve; Gt8 the medium vorticity coupling |
| G0, G0b | `test_bjorken2d.jl`, `test_bjorken_bulk2d.jl` | Bjorken, ideal and viscous (order 2.00) |
| G1, G1v | `test_gubser2d.jl`, `test_gubser_viscous2d.jl` | Gubser flow, analytic and semi-analytic (order 1.96) |
| Gs | `test_sound2d.jl` | sound speed and viscous attenuation |
| G2, G3, G3g, Gk | `test_dissipation2d.jl`, `test_charge2d.jl`, `test_charge_gubser2d.jl`, `test_charge_dispersion2d.jl` | shear/bulk targets, charge sector vs 1-D, charge on Gubser, the diffusion dispersion relation |
| G4, G7 | `test_reproduction2d.jl`, `test_dissipative_vs_1d.jl` | the 1-D production run from the production IC |
| G5, G6, G8, G9 | `test_production_allsectors2d.jl`, `test_elliptic2d.jl`, `test_unaveraged_ic2d.jl`, `test_fluctuating_ic2d.jl` | all sectors to late times; deformed, un-averaged and single-event ICs |

The 2-D leg of two more gates lives in the 1-D ladder, because each judges all three solvers with one
referee: **X1** (`test/test_diffusion_mode.jl`, the radial diffusion mode) and **IO**
(`test/test_fields_io.jl`, `save_fields`/`load_fields` round-trip — `fields_2d` included).

Outside this package, in `Julia/Projects/FiVoFluidumComparison/`:

| | |
|---|---|
| `gate_2p1d_viscous.jl` | medium viscous rows vs a referee that is neither code (1e-11); commit-level in `programme.jl check` |
| `gate_transverse_fm.jl` | the transverse first moment vs a closed form, **both codes** (FiVo 0.37 %, Fluidum 2.5e-6); commit-level |
| `gate_2p1d_m2.jl` | second-moment rows, FiVo vs Fluidum at identical states |
| `test/test_diffusion_mode.jl` (X1, the 1-D ladder) | the charm current on a radial diffusion mode: **this solver, the 1-D bulk solver and the 1-D IS2 solver** against one closed-form referee, four term configurations |
| `COMPARISON_2P1D.md` | the solve-vs-solve record (a real Pb+Pb event to freeze-out, §37) |
| `compare_examples_1p1d.jl`, `COMPARISON_1P1D.md` | the **1+1D** medium against Fluidum on the worked-example configurations — 15/15, and the first such comparison of that sector |

---

## 6. Cost

`bench/bench2d.jl` reports ns per cell-step on the production IC. Measured 2026-09-11, 16 threads,
$\tau = 0.4 \to 4$, all medium sectors on:

| run | N | ns / cell-step |
|---|---|---|
| all medium sectors + diffusion | 100 | 813 |
| ″ | 150 | 689 |
| ″ | 200 | 628 |
| ″ | 300 | **582** |

At N = 200, best of three, as a percentage over the ideal scheme (468 ns/cell-step): shear +27 %,
bulk +19 %, diffusion +16 %, all three +35 %, `consistent_fm` +51 %, `+ consistent_m2` +126 %. The
second moment's share is its five source evaluations per cell per step — one for the rate, one per
channel to measure the implicit coefficient. The per-cell cost falls with N because thread
utilisation improves. Projected: N = 400 to τ = 13 in ~1.3 min, N = 800 in ~11 min.

**Measured 2026-09-11 after the performance pass** (`TWOD_PROGRAM.md` §6ai), which made the solver
**3.4× faster** — 60 % of its runtime had been a single Bessel function inside the equation of
state. On the same machine the earlier code gave 1880 ns/cell-step at N = 300 against 582 now.
The benchmark does a warm-up run first (without it the first row carries the solver's compilation
and reads 6× high) and takes the best of three for the per-sector rows (with one repetition that
section reported diffusion as *cheaper* than ideal).

**Re-measured 2026-09-14 at 4 threads** (what `run_suite.jl bench` records, 269 s): 2641 → 2010 ns/cell-step from
N = 100 to N = 300, i.e. ~3.5× the 16-thread numbers above, and the per-sector percentages reproduce within a few
points — shear +30.0 %, bulk +21.3 %, diffusion +18.9 %, all three +33.7 %, `consistent_fm` +55.3 %,
`+ consistent_m2` +140.7 %. Quote the thread count with any of these numbers.

Which step limit binds depends on the box and the $\tau$ range. The step count tracks $N$ when the
transverse CFL binds (cost ~$N^3$) and stays flat when the Bjorken clock `CFLτ` does (cost ~$N^2$).
A lumpy event that goes unphysical is *slower* (MOOD retries): wall time per step is an early
warning.

---

## 7. Known limitations — read before quoting a number

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
