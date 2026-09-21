# FiVo 2+1D — `main2D.jl` / module `hydro2d`

A finite-volume solver for boost-invariant, transversely Cartesian relativistic viscous
hydrodynamics in Milne coordinates $(\tau, x, y, \eta_s)$. It carries the medium, the
Israel–Stewart shear and bulk stresses and a diffusing heavy-quark (charm) charge. Optionally it
also carries the thermodynamically consistent charm first moment and a passive charm second moment.
It is the 2-D version of the 1+1D radial solver in `main.jl` and shares six dimension-independent
files with it, plus the I/O layer `src/fields_io.jl` (`README.md`, "How the solvers share code").

| document | what it is for |
|---|---|
| **this file** | how to use the solver: quickstart, knobs, examples, tests, cost |
| [`EQUATIONS2D.md`](EQUATIONS2D.md) | every equation, term by term (formula, code, switch, gate) and the corrections log |
| [`examples2d/`](examples2d/README.md) | eleven runnable examples, seconds to minutes each, each with a figure; four also write animations |
| [`TWOD_PROGRAM.md`](TWOD_PROGRAM.md) | the build log |

**Status (2026-09-15).** 22 gates, **22/22 on 2026-09-15** (≈29 min): two analytic solutions
(Bjorken, Gubser), sound propagation, reproduction of the 1-D solver, closed-form tests for the charm
sector, and cross-checks with Fluidum.

---

## 1. Quickstart

In a REPL started with `julia -t auto --project=.`:

```julia
include("main2D.jl"); using .hydro2d; const H = hydro2d

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

Two things to watch:

1. Call `finalize_ic!` after `set_cell!`, otherwise a vacuum tail starts below the energy floor.
   `initialize_uniform!`, `initialize_from_radial!` and `initialize_from_grid_csv!` do it for you.
2. `res.ok` means every step completed, not that the result is physical. Also check `res.maxu`,
   `res.dQ`, `res.minPtot`, and `|π|/P`, `|ν|/n` over cells above freeze-out.

---

## 2. The model

`build_model_2d(; kwargs...)` builds an `IdealDiffVisc2DModel` whose state layout matches the enabled
sectors (a disabled sector's fields are not in `U` at all). The defaults are the bare ideal scheme;
every sector and every regulator is opt-in. Any field below can be passed as a keyword.

**Sectors and coefficients**

| keyword | default | meaning |
|---|---|---|
| `eos` | `LatticeHRGEOS()` | equation of state (`ConformalHQEOS()` for the conformal tests) |
| `with_charge` | `true` | carry the conserved charge $\tilde D = \tau J^\tau$ at all |
| `enable_shear`, `eta_over_s`, `tauShear_coeff` | `false`, 0.1, 0.2 | $\eta = (\eta/s)s$, $\tau_\pi = \eta/(C_s T s)$ |
| `deltaShear_factor` | **4/3** | $\delta_{\pi\pi}/\tau_\pi$. `build_model_1d` also uses 4/3; the legacy `run_sim_ideal_diff_visc` defaults to 0. Fluidum has no such term |
| `enable_bulk`, `zeta_over_s`, `tauPi_coeff` | `false`, 0.1, 15.0 | peaked $\zeta/s$ at $T = 0.175$ GeV, $\tau_\Pi$ from $C_\zeta$ |
| `enable_diff`, `kappa_coeff` | `false`, **0.0** | charge diffusion with $D_sT$ = `kappa_coeff`. Set it; 0 means no diffusion |
| `consistent_fm` | `false` | the full-∇P first moment (four extra terms, EQUATIONS2D §4) |
| `consistent_m2` | `false` | the passive charm second moment $(\pi_Q^{ij}, \Pi_Q)$ (EQUATIONS2D §5) |
| `terms` | `:default` | per-term switches: a preset (`:homogeneous`, `:full`, `:none`), `without(:acceleration, …)`, or a NamedTuple such as `(fm_inertial = false,)` (§3) |
| `transport_mass` | 0.0 (= EoS mass) | charm mass in the second moment's Bessel ratios and $\tau_M$ |
| `tauN_coeff`, `deltaN_factor` | 1.0, 0.0 | $\tau_n$ multiplier (refused with `consistent_fm`), $\delta_{nn}/\tau_n$ |

**Numerics and switches that are part of the equations**

| keyword | default | meaning |
|---|---|---|
| `shear_projected_deriv`, `diff_projected_deriv` | `true` | the projector correction on $D\pi$ and $D\nu$ |
| `relax_advect_pi`, `relax_advect_Pi`, `relax_advect_nu`, `relax_advect_m2` | `true` | transverse advection of the dissipative fields, upwinded inside the relaxation |
| `advect_pi`, `advect_Pi`, `advect_nu` | `false` | advect them as passive scalars in the flux instead. Unstable for the charge sector on the production IC; leave off |
| `shear_constraint` | `:project` | restore tracelessness each step (`:monitor` only measures it) |

**Regulators** (EQUATIONS2D §8)

| keyword | default | meaning |
|---|---|---|
| `vacuum_n_lo`, `vacuum_n_hi`, `vacuum_ramp_relax` | 1e-6, 2e-3 fm⁻³, `true` | density-gated ramp on the charge drive and on $\tau_n$ |
| `T_vac_cut` (or `E_vac_cut`) | 0.05 GeV | colder cells are vacuum |
| `pi_clip_factor`, `Pi_clip_factor`, `nu_clip_factor` | −1 (off) | $\lvert\pi\rvert, \lvert\Pi\rvert \le fP$, $\lvert\nu\rvert \le f\,n\,u^\tau$. Use `pi_clip_factor = 1` on lumpy events |
| `r_domain` | `Inf` | evolve only the disc $r \le r_{\rm domain}$ |

**Second-order medium couplings** (DNMR; all default 0, EQUATIONS2D §2–3, gate Gd): `taupi_pi_factor`
(τ_ππ/τ_π), `lambda_pi_Pi_factor` (λ_πΠ/τ_π), `deltaPi_factor` (δ_ΠΠ/τ_Π), `lambda_Pi_pi_factor` (λ_Ππ/τ_Π).

`diffusion_drive`, `diff_dt_coeff`, `shear_dt_coeff` and `bulk_dt_coeff` exist only for signature
parity with the 1-D model. Nothing in `src2d/` reads them, so non-default values are refused.

**The driver**

```julia
res = run_sim_2d!(U, g, m; τ0 = 0.4, τfinal = 13.0, CFL = 0.2, CFLτ = 0.05,
                  integrator = :ssprk2,          # or :ssprk3 (convergence studies; no MOOD)
                  bc = :outflow,                 # or :periodic
                  on_dump = (τ, U, work) -> …, dump_dt = 0.5,
                  work = nothing, reset_history = (work === nothing))
# res: ok, τ, nsteps, nprimfail, nvacuum, max_shear_res, work, maxu, dQ, Q0, Q1, minPtot
```

To continue a run, pass `work = res.work, reset_history = false`. The $\partial_\tau$ history lives in
the work array; without it the restart drops the $\partial_\tau$ pieces of the NS targets for one step.

**Initial conditions**

| | |
|---|---|
| `initialize_uniform!(U, g, m, τ0; T0, alpha0)` | a transversely uniform state (Bjorken) |
| `initialize_from_radial!(U, g, m, τ0, Tof, αof)` | from radial profiles `T(r)`, `α(r)` |
| `initialize_from_grid_csv!(U, g, m, τ0, csv)` | from an `x,y,T0,alpha0` grid (real ε₂, ε₃) |
| `set_cell!(U, i, T, α, ux, uy, τ, m; Pi, pixx, pixy, piyy, nux, nuy)` then `finalize_ic!` | anything else |

---

## 3. Switching individual terms off

Whole sectors switch with the `enable_*` / `consistent_*` flags, and a few medium terms with the
knobs above. Every other term has its own switch in the shared register `Terms` (`src/terms.jl`, the
same one the 1+1D solvers use; `Terms2D` is its alias here). The interface is in `README.md` §3:

```julia
m = H.build_model_2d(; enable_diff = true, kappa_coeff = 0.1163,
                       consistent_fm = true, consistent_m2 = true,
                       terms = (fm_inertial = false, m2_vorticity = true))
m = H.build_model_2d(; …, terms = :homogeneous)                       # a homogeneous medium at rest
m = H.build_model_2d(; …, terms = H.without(:vorticity, :acceleration)) # drop by physical ingredient
H.show_equations(m)
```

`show_equations` prints every equation with each term marked `[x]` or `[ ]` and the knob that
controls it. `show_terms()` prints the register with each term's ingredients. The switches:

| first moment | second moment |
|---|---|
| `nu_gradalpha` — the fugacity drive $-\kappa\nabla^{\langle i\rangle}\alpha$ | `m2_nu_gradient` — $2\eta_Q\sigma_{(\nu)}$, $\zeta_Q\theta_{(\nu)}$ |
| `fm_gradT` — pressure gradient, T channel | `m2_bg_gradu` — $2\bar\eta\sigma$, $\tfrac53\bar\eta\theta$ |
| `fm_inertial` — $\tau_n n a^i$ | `m2_bg_DlnT`, `m2_bg_Dalpha` — the trace drives |
| `fm_nu_gradu` — $\tau_n\nu^m\nabla_m u^i$ | `m2_expansion`, `m2_pi_sigma`, `m2_PiQ_sigma` — class (iii) |
| `fm_expansion` — $\tau_n\theta\nu^i$ | `m2_vorticity` — **off by default**, class (iii) |
| `fm_dlnh` — $(D_s/T)h'DT\,\nu^i$ | `m2_projector` — the Δ-projector on $D\pi_Q$ |
| **medium:** `shear_vorticity` — $2\tau_\pi\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle}$, **off by default** | `m2_accel_nu`, `m2_nu_gradTh` — class (iv) |

Naming a switch in a sector that is off, or misspelling one, is an error. Changes from a preset or an
ingredient in an off sector are ignored. Switching a term off reproduces the arithmetic without it
bit for bit (gate Gt). `terms = :homogeneous` with `consistent_fm = true` reproduces
`consistent_fm = false` bit for bit (Gt7). Example 06 shows what each term does on a fireball.

**The medium's vorticity coupling** `shear_vorticity` is the medium version of `m2_vorticity`: the
same function, `vorticity_coupling_2d`, with τ_π for τ_M (DNMR's
$+2\tau_\pi\pi_\lambda^{\langle\mu}\omega^{\nu\rangle\lambda}$). Gt8: inert on a round fireball, acts
on a swirling one (795× more), and only rotates π ($\pi_{\mu\nu}X^{\mu\nu} = 0$ to 5e-15). Fluidum
does not carry it. Off by default.

---

## 4. Examples

```sh
julia -t auto --project=. examples2d/06_charm_terms.jl
```

| | what it shows | time |
|---|---|---|
| 01 first run | the whole API on a uniform state, checked against the 0+1D equations; why a viscous run is first order | 12 s |
| 02 elliptic flow | ε₂ → momentum anisotropy, sector by sector, with the ε₂ = 0 control | 34 s |
| 03 choosing a resolution | a three-grid Richardson study; fields, observables and conserved quantities converge differently | 52 s |
| 04 a fluctuating event | v₃ from lumps, event by event, about the participant plane | 1.7 min |
| 05 the dissipative sectors | what each sector changes and costs, how big \|π\|/P gets, whether the regulators are inert | 1.2 min |
| 06 the charm terms | `show_equations`, then every closure term removed one at a time and what it moves | 18 s |
| 07 a real event | one MC-Glauber Pb+Pb event to freeze-out, raw vs smoothed (`smooth_fm`); animations | 40 s |
| 08 vorticity on vs off | `m2_vorticity` on that event: \|ω\|/\|σ\| and what it moves in π_Q; animation. `EX08_N=480`/`640` render it at 3×/4× (median 3.01e-2 → 2.84e-2 → 2.79e-2) | 30 s |
| 09 Gubser flow | the solver against an exact solution, and the convergence order; animation | 1.5 min |
| 10 the showcase | one Pb+Pb event, every sector on and `m2_vorticity = true`, long run at high resolution; animation. `EX10_N`/`EX10_TAUF`/`EX10_NFRAME` override the defaults | 72 s |
| 11 the swirl showcase | 50 hot spots + a rigid-body swirl, every sector on, τ → 12 fm/c | 2.5 min |

Times are from the suite baseline (2026-09-11; example 10 from 2026-09-14, 11 from 2026-09-15). All
**14/14** FiVo examples, the three 1+1D ones included, passed on 2026-09-14.

### The figures they produce

Each example writes its figure into `examples2d/figures/`. Longer captions are in
[`examples2d/README.md`](examples2d/README.md).

![the first run](examples2d/figures/ex01_bjorken.png)
![elliptic flow](examples2d/figures/ex02_elliptic_flow.png)

<sub>**01** the whole API on a transversely uniform state, against the 0+1D DNMR equations.
**02** ε₂ → momentum anisotropy, sector by sector from one initial condition, with the ε₂ = 0
control (−2.7e-15).</sub>

![choosing a resolution](examples2d/figures/ex03_convergence.png)
![the dissipative sectors](examples2d/figures/ex05_sectors.png)

<sub>**03** a three-grid Richardson study of a field, an observable and a conserved quantity.
**05** what each dissipative sector changes and costs, how large \|π\|/P gets, and whether the
regulators are inert.</sub>

![a fluctuating event](examples2d/figures/ex04_fluctuating_event.png)
![the charm terms](examples2d/figures/ex06_charm_terms.png)

<sub>**04** v₃ from lumps, event by event, about the participant plane. **06** every charm closure
term removed one at a time, and what each one moves.</sub>

![a real event](examples2d/figures/ex07_real_event.png)
![vorticity on vs off](examples2d/figures/ex08_vorticity_N640.png)

<sub>**07** one MC-Glauber Pb+Pb event to freeze-out, raw and lightly smoothed. **08** `m2_vorticity`
on vs off, rendered at N = 640 (`EX08_N=640`, ~12 min). Median \|ω\|/\|σ\| 3.01e-2 → 2.84e-2 →
2.79e-2 over N = 160/480/640.</sub>

![Gubser flow](examples2d/figures/ex09_gubser.png)

<sub>**09** Gubser flow against the exact solution, and the convergence order. Gate G1 measures T
1.90–1.96 and u 2.01–2.07 at N = 100/200/400.</sub>

![the showcase](examples2d/figures/ex10_showcase_N400.png)

<sub>**10** one Pb+Pb event with every sector on, run long at high resolution.</sub>

---

## 5. Tests

```sh
julia -t auto --project=. test/run2d_gates.jl                     # 22 gates, ~30 min (22/22 on 2026-09-15)
FIVO2D_TIER=fast julia -t auto --project=. test/run2d_gates.jl    # 6 gates, ~30 s, in CI
```

Each gate runs in its own process. The runner exits 1 on any failure, including a missing gate
file. Run the full ladder after changing `src2d/` or `main2D.jl`.

| gate | file | tested against |
|---|---|---|
| shear algebra | `test_shear2d_algebra.jl` | orthogonality/trace identities, an independent closure |
| recovery | `test_primrec2d.jl`, `test_primrec2d_vs_1d.jl` | round trips; the 1-D recovery on the production locus |
| **Gc** first moment | `test_consistent_fm2d.jl` | the 1-D source bit for bit; rotation; Bjorken; Gc7: the sign on a solve |
| **Gm** second moment | `test_consistent_m22d.jl` | the 1-D reduction (4e-16, with the moving projection); rotation; trace; Bjorken |
| **Gd** DNMR couplings | `test_dnmr2d.jl` | the 0+1D DNMR ODEs (Richardson-extrapolated), wrong-sign versions, the 1-D solver, and the π·σ contractions vs brute force |
| **Gt** term switches | `test_terms2d.jl` | every switch wired and additive; the vorticity coupling vs brute-force index algebra; Gt7 presets/`without` and `:homogeneous` ≡ shipped on a solve; Gt8 the medium vorticity coupling |
| G0, G0b | `test_bjorken2d.jl`, `test_bjorken_bulk2d.jl` | Bjorken, ideal and viscous (order 2.00) |
| G1, G1v | `test_gubser2d.jl`, `test_gubser_viscous2d.jl` | Gubser flow, analytic and semi-analytic (order 1.96) |
| Gs | `test_sound2d.jl` | sound speed and viscous attenuation |
| G2, G3, G3g, Gk | `test_dissipation2d.jl`, `test_charge2d.jl`, `test_charge_gubser2d.jl`, `test_charge_dispersion2d.jl` | shear/bulk targets, charge sector vs 1-D, charge on Gubser, the diffusion dispersion relation |
| G4, G7 | `test_reproduction2d.jl`, `test_dissipative_vs_1d.jl` | the 1-D production run from the production IC |
| G5, G6, G8, G9 | `test_production_allsectors2d.jl`, `test_elliptic2d.jl`, `test_unaveraged_ic2d.jl`, `test_fluctuating_ic2d.jl` | all sectors to late times; deformed, un-averaged and single-event ICs |

### What the gates measure

From the run of 2026-09-15:

| | measured |
|---|---|
| **G0** Bjorken, ideal | order **2.00** (SSPRK2) and **2.99** (SSPRK3); τJ^τ drift, transverse spread and \|u\| all identically zero |
| **G0b** Bjorken + nonlinear bulk | T to 1e-4 and Π to 7.5e-3 against the 0+1D system at ζ/s = 0.05, 0.15, 0.30, i.e. up to \|Π\|/P = 0.60 |
| **G1** Gubser, exact | L2(T) 3.29e-3 → 8.81e-4 → 2.26e-4 and L2(u) 7.73e-3 → 1.92e-3 → 4.57e-4 at N = 100/200/400, order **T 1.90, 1.96 · u 2.01, 2.07**. x↔y symmetry at 1e-14; the entropy outflow is resolution-independent (−0.116) while the error against the analytic answer falls 1.28e-3 → 9.56e-5 |
| **G1v** Gubser, viscous | the semi-analytic ODE reduces to the analytic ideal Gubser at η/s = 0 to **3.6e-11** |
| **G2** shear + bulk | the projected NS target to **2.4e-15**; the 2-D and 1-D π^{yy}/π^{η} forms agree exactly; order **1.05** |
| **G3** charge | charge drift on a closed domain **1.9e-15**; **0 of 2104** cells carry an up-gradient ν^x; x↔y asymmetry exactly zero |
| **G3g** charge on Gubser | core invariant order **1.97**, L2(n) order **2.05** |
| **Gk** dispersion | τ_n matches the bare moment ratio D_s z K₃/K₂ to **1.0000** at four temperatures; the measured propagating/overdamped transition brackets the predicted k* = 0.733 1/fm |
| **Gc** first moment | the 1-D source reproduced bit for bit (rel 0.00e+00 at four of five probes, 2.5e-16 at the fifth); the derived sign tracks the ODE to 7.3e-4, the flipped sign misses by 5.3e-1 |
| **Gm** second moment | the 1-D reduction to **4.0e-16**, rotation to 7.8e-16 |
| **Gd** DNMR couplings | each of δ_ππ, δ_ΠΠ, λ_πΠ, λ_Ππ Richardson-extrapolated to ≤3e-4 at order **0.97–0.99**; wrong-sign versions miss by 5e-3 to 1.7 |
| **Gt** term switches | 18 terms registered; Σ(per-term pieces) vs the whole source **1.4e-15** over 40 states; the vorticity piece is exactly zero in the axisymmetric limit and matches brute-force index algebra to 1.1e-14 |
| **Gs** sound | c_s to **0.05 %** of the exact value |
| **G4/G7** vs the 1-D solver | L2(T) 1.18e-3 → 6.31e-4, order **0.90** in dx; sector by sector and ring by ring in G7 |
| **G5/G6/G8/G9** production ICs | the ε₂ = 0 control returns anisotropy **−2.7e-15**; ε₂ = 0.25 gives 0.20786 → 0.20798 over N = 150 → 300 (**0.06 %**); single-event v₃ converges to 0.92 % and is **9.5×** the ensemble value; the regulators move v₂ by 0.05 % of signal |

G2's order is 1.05 because of the operator split: every dissipative sector is first order in Δτ. The
ideal sector (G0, G1) is second order.

Two more gates cover this solver but sit in the 1-D ladder: **X1** (`test/test_diffusion_mode.jl`,
the radial diffusion mode through all three solvers) and **IO** (`test/test_fields_io.jl`,
`save_fields`/`load_fields`, `fields_2d` included). The cross-checks with Fluidum are in `README.md`.

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

At N = 200, best of three, relative to the ideal scheme (468 ns/cell-step): shear +27 %, bulk
+19 %, diffusion +16 %, all three +35 %, `consistent_fm` +51 %, `+ consistent_m2` +126 %. The
second moment costs five source evaluations per cell per step. The per-cell cost falls with N
because thread utilisation improves. Projected: N = 400 to τ = 13 in ~1.3 min, N = 800 in ~11 min.

At 4 threads (2026-09-14): 2641 → 2010 ns/cell-step from N = 100 to N = 300, about 3.5× the
16-thread numbers. The per-sector percentages are similar: shear +30.0 %, bulk +21.3 %, diffusion
+18.9 %, all three +33.7 %, `consistent_fm` +55.3 %, `+ consistent_m2` +140.7 %.

The step count tracks $N$ when the transverse CFL binds (cost ~$N^3$) and stays flat when the
Bjorken clock `CFLτ` does (cost ~$N^2$). A lumpy event that goes unphysical is slower (MOOD retries).
