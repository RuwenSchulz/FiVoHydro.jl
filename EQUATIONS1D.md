# FiVo 1+1D — the equations, term by term

Every equation the two 1+1D solvers integrate, one term at a time: what it is, where it lives in the
code, how to switch it off, and which gate checks it. The 2+1D twin is [`EQUATIONS2D.md`](EQUATIONS2D.md);
the front door is [`README.md`](README.md).

> ⚠ Paths beginning `Julia/Projects/…` or `Tex/…` name the **private research repository** this
> package is developed in. They are cited for provenance — so a number can be traced to the script
> that produced it — and are not links you can follow from a clone of this package.


| solver | file | what it evolves |
|---|---|---|
| **bulk** | `main.jl` (module `hydro`) | the medium $(T, u^r, \Pi, \pi)$ and a diffusing charge $(n, \nu^r)$ — §1–§4 |
| **charm IS2** | `main2IS2.jl` (module `hydro_current_IS2`) | the charm current and its second moment $(\alpha, \nu^r, \pi_Q^r, \pi_Q^\perp, \Pi_Q)$ on a **frozen** background — §5–§6 |

The same information at run time, for the configuration you actually built:

```julia
hydro.show_equations(hydro.build_model_1d(; enable_diff = true, kappa_coeff = 0.1163, consistent_fm = true))
hydro_current_IS2.show_equations_IS2(; consistent_fm = true, consistent_m2 = true, terms = :homogeneous)
```

---

## 0. Conventions

| | |
|---|---|
| chart | radial Milne $(\tau, r, \phi, \eta_s)$, $g_{\mu\nu} = \mathrm{diag}(-1, 1, r^2, \tau^2)$; boost invariant and axisymmetric |
| velocity | $u^\mu = (u^\tau, u^r, 0, 0)$, $u^\tau = \sqrt{1 + (u^r)^2}$, $v = u^r/u^\tau$, rapidity $y = \operatorname{asinh} u^r$ |
| comoving derivative | $D = u^\tau\partial_\tau + u^r\partial_r$; also $D_l = u^r\partial_\tau + u^\tau\partial_r$ (along $l$ below) |
| expansion | $\theta = \partial_\tau u^\tau + \partial_r u^r + u^\tau/\tau + u^r/r$ |
| comoving triad | $l = (u^r, u^\tau, 0, 0)$, $\hat\phi = (0,0,1/r,0)$, $\hat\eta = (0,0,0,1/\tau)$ — orthonormal, orthogonal to $u$, parallel transported ($Dl = a_l u$, $D\hat\phi = D\hat\eta = 0$) |
| shear rates | $\sigma_l = \partial_r u^r + v\,\partial_\tau u^r - \theta/3$, $\sigma_\phi = u^r/r - \theta/3$, $\sigma_\eta = u^\tau/\tau - \theta/3$ |
| units | fm and GeV; $\hbar c = 0.19733$ GeV fm (`fmGeV` $= 1/\hbar c$); `DsT` is $D_sT$, dimensionless |
| dissipatives | stored = `DISS_SIGN` × physical (`DISS_SIGN = +1`) |

In radial symmetry every tensor is diagonal in the triad $(l, \hat\phi, \hat\eta)$. So the shear carries two
independent channels, $\pi_\phi$ and $\pi_\eta = \tau^2\pi^{\eta\eta}$, with $\pi_l = -\pi_\phi - \pi_\eta$. Three kinds of
term vanish **identically** here and are carried only by the 2+1D solver: any vorticity coupling (a radial flow has
none), and the Δ-projector on $D\pi$ in these channels (they are orthogonal to the $(\tau, r)$ plane the flow lives in).
The switches for them are accepted in 1+1D and do nothing (`src/terms.jl`, "RADIAL SYMMETRY").

**The ∂_τ history.** $\partial_\tau u^r$, $\partial_\tau\alpha$ and $\partial_\tau T$ are differenced against the
previous step (`work.y_prev`, `alpha_prev`, `T_prev`), NaN-seeded, and dropped on the first step. ⚠ Until
2026-09-11 `y_prev` was copied at the START of the relaxation, where it holds the last RK stage, not the previous
step: $\partial_\tau u^r$ came out at about 4% of its value (§8).

---

## 1. The medium — conservation laws (bulk solver)

In flux form with $\tilde D = \tau J^\tau$, $S = T^{\tau r}$, $E = T^{\tau\tau}$:

$$
\begin{aligned}
\partial_\tau \tilde D + \frac1r\partial_r\!\left(r\,\tau J^r\right) &= 0 \\
\partial_\tau S + \frac1r\partial_r\!\left(r\,T^{rr}\right) &= -\frac{S}{\tau} + \frac{P + \Pi + \pi^\phi{}_\phi}{r} \\
\partial_\tau E + \frac1r\partial_r\!\left(r\,T^{\tau r}\right) &= -\frac{E + P + \Pi + \pi^\eta{}_\eta}{\tau}
\end{aligned}
\qquad
T^{\mu\nu} = (e + P + \Pi)u^\mu u^\nu + (P + \Pi)g^{\mu\nu} + \pi^{\mu\nu},\quad J^\mu = nu^\mu + \nu^\mu .
$$

| piece | code | gate |
|---|---|---|
| fluxes, sources | `flux_cell!`, `source_cell_fast_col!` (`src/fluxes.jl`), `rhs!` (`src/rhs.jl`) | A1 (Bjorken, order 2.00 / 2.99), A4 (Gubser) |
| $\pi^{\mu\nu}$ from the stored channels | `shear_tensor_contravariant` (`src/shear_tensor.jl`) | A4 |
| primitive recovery $(\log T, \phi, y)$ | `cons_to_prim_ideal_phi_diff_visc!` (`src/primrec.jl`) | `runtests.jl` round trips, A1 b |
| EoS | `LatticeHRGEOS`, `ConformalHQEOS` (`src/eos.jl`) | `test_eos_consistency.jl` |
| HLLE + MUSCL(MC), SSPRK2 / SSPRK3 | `step_ssprk2!`, `step_ssprk3!` (`src/timestepper.jl`) | A1 a |

Nothing here is switchable: these are the conservation laws.

---

## 2. Shear (bulk solver)

For each evolved channel $i \in \{\phi, \eta\}$ (mixed diagonal components = triad components):

$$
\tau_\pi D\pi_i + \pi_i = -2\eta\sigma_i \;-\; \delta_{\pi\pi}\theta\,\pi_i \;-\; \tau_{\pi\pi}\Big(\pi_i\sigma_i - \tfrac13\pi\!:\!\sigma\Big) \;-\; \lambda_{\pi\Pi}\,\Pi\,\sigma_i ,
\qquad \pi\!:\!\sigma = \pi_l\sigma_l + \pi_\phi\sigma_\phi + \pi_\eta\sigma_\eta .
$$

This is the mostly-plus form of DNMR's $-\tau_{\pi\pi}\pi^{\langle\mu}{}_\lambda\sigma^{\nu\rangle\lambda} + \lambda_{\pi\Pi}\Pi\sigma^{\mu\nu}$ (mostly-minus),
under which the first term keeps its sign and the second flips.

| term | code | switch | gate |
|---|---|---|---|
| $-2\eta\sigma_i$ | `relax_dissipative!` (`pp_NS`, `pe_NS`) | `enable_shear` | A2, A4 |
| $-\delta_{\pi\pi}\theta\pi_i$ | ″ (denominator) | `deltaShear_factor = 0` | A2 |
| $-\tau_{\pi\pi}(\pi_i\sigma_i - \pi:\sigma/3)$ | ″ | `taupi_pi_factor` (default 0) | A2 |
| $-\lambda_{\pi\Pi}\Pi\sigma_i$ | ″ | `lambda_pi_Pi_factor` (default 0) | A2, incl. the sign check (c) |
| $\tau_\pi u^r\partial_r\pi_i$ (upwind) | ″ | `relax_advect_pi = false` | — |
| $2\tau_\pi\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle}$ | — | `terms.shear_vorticity` | ≡ 0 in radial flow |

Coefficients: $\eta = (\eta/s)\,s\,\hbar c$, $\tau_\pi = \eta/(C_s T s)$ (`eta_over_s`, `tauShear_coeff` $= C_s$); every
second-order coefficient is a factor times its relaxation time (`deltaShear_factor` × τ_π, …).

> ⚠ **Defaults differ between the two entry points.** `build_model_1d` defaults to δ_ππ = 4/3 τ_π and C_s = 0.2, the
> same as the 2+1D solver. The production driver `run_sim_ideal_diff_visc` defaults to δ_ππ = 0 and C_s = 1, and its
> production callers pass 4/3 and 0.2 explicitly.

---

## 3. Bulk (bulk solver)

$$
\tau_\Pi D\Pi + \Pi = -\zeta\theta \;-\; \delta_{\Pi\Pi}\theta\,\Pi \;-\; \lambda_{\Pi\pi}\,\pi\!:\!\sigma
$$

| term | code | switch | gate |
|---|---|---|---|
| $-\zeta\theta$ | `relax_dissipative!` (`ΠNS_phys`) | `enable_bulk` | A2 |
| $-\delta_{\Pi\Pi}\theta\Pi$ | ″ | `deltaPi_factor` | A2 |
| $-\lambda_{\Pi\pi}\pi:\sigma$ | ″ | `lambda_Pi_pi_factor` | A2, incl. the sign check (c) |
| $\tau_\Pi u^r\partial_r\Pi$ | ″ | `relax_advect_Pi = false` | — |

$\zeta = (\zeta/s)(T)\,s\,\hbar c$ with the Lorentzian peak at $T = 0.175$ GeV, $\tau_\Pi = \zeta/[C_\zeta T s(1/3 - c_s^2)^2] + 0.1$ fm.

---

## 4. The charge current (bulk solver)

$$
\tau_n\,\Delta^r{}_\nu D\nu^\nu + \nu^r = \text{drive}^r - \delta_{nn}\theta\,\nu^r - \lambda_{nn}\sigma_l\,\nu^r ,
\qquad \Delta^r{}_\nu D\nu^\nu = D\nu^r - v\,\nu^r\,Dy ,
$$

$$
\text{drive}^r = \underbrace{-\kappa\,\nabla^{\langle r\rangle}\alpha}_{\text{shipped}}
\;\underbrace{-\;\frac{D_s}{T}\Big(n + T\frac{\partial n}{\partial T}\Big)\nabla^{\langle r\rangle}T
\;-\;\tau_n n\,a^r \;-\;\tau_n(\nu\!\cdot\!\nabla u)^r \;-\;\nu^r\Big(\tau_n\theta + \frac{D_s}{T}h'DT\Big)}_{\text{consistent\_fm} \;=\; -s^r},
$$

with $\nabla^{\langle r\rangle}X = (u^\tau)^2\partial_r X + u^r u^\tau\partial_\tau X$, $\kappa = D_s n$, $a^r = u^\tau\partial_\tau u^r + u^r\partial_r u^r$,
$(\nu\cdot\nabla u)^r = \nu^r(\partial_r u^r + v\partial_\tau u^r)$, $h = mK_3/K_2$, and $\tau_n = D_s h/T$.

| term | code | switch | gate |
|---|---|---|---|
| $-\kappa\nabla^{\langle r\rangle}\alpha$ | `relax_dissipative!` (`νNS_raw`) | `terms.nu_gradalpha` | X1 (the telegraph mode: τ_n and D_s) |
| $-(D_s/T)(n + T\partial_T n)\nabla^{\langle r\rangle}T$ | `hq_consistent_extras` (`src/hq_consistent_firstmoment.jl`) | `terms.fm_gradT` | T2, T4 (Euler cancellation) |
| $-\tau_n n a^r$ | ″ | `terms.fm_inertial` | T2, T4 |
| $-\tau_n(\nu\cdot\nabla u)^r$ | ″ | `terms.fm_nu_gradu` | T2 |
| $-\tau_n\theta\,\nu^r$ | ″ | `terms.fm_expansion` | T2, **X1** (the sign, on a solve) |
| $-(D_s/T)h'DT\,\nu^r$ | ″ | `terms.fm_dlnh` | T2, X1 |
| all consistent terms | | `consistent_fm = false` | T3 (`:homogeneous` ≡ off, bit for bit) |
| $\tau_n v\,\nu^r Dy$ (projector) | ″ | always | — |
| $\tau_n u^r\partial_r\nu^r$ | ″ | `relax_advect_nur = false` | — |
| $-\delta_{nn}\theta\nu$, $-\lambda_{nn}\sigma_l\nu$ | ″ | `deltaN_factor`, `lambda_NN_factor` | — |

`hq_consistent_extras` is **the same function** the charm IS2 solver adds to its matrix source (§5). It returns
$s^r$ with the source on the LEFT, $\tau_n\Delta D\nu + \nu + \kappa\nabla\alpha + s = 0$, and the relaxation, which
integrates the right-hand side, **subtracts** it. The switched path splits the shipped expression into its five
terms, and T2 checks that the pieces sum to it (3e-15). ⚠ In radial Milne the $\nu\cdot\nabla u$ term and the flat part of $\theta$ are
numerically equal. The shipped `2·∂_r u^r·h` is both of them, and the split separates them.

**Physics checks** built into the gates:
- T4. On an ideal fluid Euler gives $a = -\nabla^{\langle r\rangle}T/T$, and $n + T\partial_T n = nh/T$, so the
  pressure-gradient and inertial terms cancel **as a pair**. That is the hidden justification of the shipped
  $\nabla\alpha$-only drive. Measured: each term alone moves ν by 16–18× the $\nabla\alpha$-driven current, while the
  pair moves it by $7\cdot10^{-2}$ (Nr = 120) → $6\cdot10^{-3}$ (Nr = 480) away from the axis.
- T3. `terms = :homogeneous` with `consistent_fm = true` reproduces `consistent_fm = false` bit for bit. The shipped row
  *is* the homogeneous-medium reduction.
- The sources wait for the $\partial_\tau$ history (first step) because the two terms cancel only as a pair.

Density frame (`charge_mode = :density_frame`): no $\nu$ field; $\mu$ from the on-slice $n$ and a parabolic flux
$J^r = -\kappa(u^\tau)^2\partial_r\alpha$ (`add_density_frame_charge_flux!`); no consistent first moment.

---

## 5. The charm current (IS2 solver)

`main2IS2.jl` evolves the charm sector on a **frozen** background $T(\tau, r)$, $u^r(\tau, r)$ (a JLD2 bundle, or
`analytic_background(T = (τ, r) -> …)`). The first moment is written as a quasi-linear system
$A_t\partial_\tau U + A_x\partial_r U + \text{src} = 0$ for $U = (\alpha, \nu^r, \pi_Q^r, \pi_Q^\perp, \Pi_Q)$
(`build_IS2_system!`, `src/is2_second_moment_builder.jl`). Rows 1–2:

$$
\text{row 1 (charge)}\qquad \partial_\tau(\tau r J^\tau) + \partial_r(\tau r J^r) = 0 \quad(\text{conservative } q\text{-transport, then } \alpha \text{ recovered})
$$
$$
\text{row 2 (current)}\qquad \tau_n\Delta^r{}_\nu D\nu^\nu + \nu^r + \kappa\nabla^{\langle r\rangle}\alpha + s^r_{\text{consistent}} + c_M(\ldots) = 0
$$

| term | code | switch | gate |
|---|---|---|---|
| $\kappa\nabla^{\langle r\rangle}\alpha$ | `At[2,1]`, `Ax[2,1]` | `terms.nu_gradalpha` | X1, T3 (off ⇒ no current) |
| $s^r$ (the five consistent terms, §4) | `hq_consistent_extras` → `src[2]` | `consistent_fm`, `terms.fm_*` | X1, T2, T3; Fluidum twin: `Tex/MaxEntHydro/diag_fivo_consistent_gates.jl` |
| $c_M$ back-coupling of the second moment | `hq_cm_force` / matrix `cM` | `use_cM` (default off) | CMExperiment |

`consistent_fm`, `consistent_m2` and `terms` are keywords of `run_static_IS2_test`. They set the module Refs
`IS2_CONSISTENT_FM`, `IS2_CONSISTENT_M2` and `IS2_TERMS` for one solve and restore them afterwards. O+O production
sets `IS2_CONSISTENT_FM[]` directly (`LangevinPaperOO/is2_dropin.jl`), and that still works.

Coefficients (`transport_all`): $\tau_n = D_s zK_3/K_2$ (bare, no $g_{hq}$), $\kappa = D_s n_{GC}$, $\partial n/\partial\alpha = n$,
$\partial n/\partial T = (n/T)(3 + zK_1/K_2)$, $\tau_M = (D_s z/2)K_4/K_3$, $\eta_M = (D_sT/2)(4 + zK_1/K_2)$.

---

## 6. The charm second moment (IS2 solver)

Two systems, **not** superposable:

* `consistent_m2 = false` — the SHIPPED rows (`_second_moment_eigen_rhs!`, the inverse of Fluidum's
  `HQ_const_BG_2nd_moment` matrix), with its $\sigma_{(\nu)}$ drive of the opposite sign (gate M3,
  `Tex/MaxEntHydro/diag_fivo_m2_gates.jl`). Not switchable term by term.
* `consistent_m2 = true` — the covariant reduction in the transported triad (`hq_consistent_m2_rhs`,
  `src/hq_consistent_m2.jl`). All on the left, = 0, traceless channels $i \in \{l, \phi\}$ and the trace:

$$
\begin{aligned}
&\tau_M Dp_i + p_i + 2\eta_Q\sigma_{(\nu)i} + \tau_M\big[(\tfrac53\theta + D\ln C)p_i + 2(p_i s_i - \pi:\sigma/3) + 2\Pi_Q s_i\big] + 2\bar\eta s_i
+ \big[2\lambda_a a_l\nu_l + \tfrac{D_s}{T}\nu_l D_l(Th)\big](\delta_{il} - \tfrac13) = 0 \\
&\tau_M D\Pi_Q + \Pi_Q + \zeta_Q\theta_{(\nu)} + \tau_M(\tfrac53\theta + D\ln C)\Pi_Q + \tfrac23\tau_M\pi:\sigma
+ \bar\eta\big[D\alpha + \tfrac AB D\ln T + \tfrac53\theta\big] + \tfrac{D_sA}{3T}a\cdot\nu + \tfrac{5D_s}{6T}\nu_l D_l(Th) = 0
\end{aligned}
$$

| class | term | switch |
|---|---|---|
| (i) | $2\eta_Q\sigma_{(\nu)}$ ∣ $\zeta_Q\theta_{(\nu)}$ | `terms.m2_nu_gradient` |
| (ii) | $2\bar\eta\sigma$ ∣ $\tfrac53\bar\eta\theta$ | `terms.m2_bg_gradu` |
| (ii) | ∣ $\bar\eta(A/B)D\ln T$ | `terms.m2_bg_DlnT` |
| (ii) | ∣ $\bar\eta D\alpha$ | `terms.m2_bg_Dalpha` |
| (iii) | $\tau_M(\tfrac53\theta + D\ln C)$ × field | `terms.m2_expansion` |
| (iii) | $2\tau_M(p_is_i - \pi:\sigma/3)$ ∣ $\tfrac23\tau_M\pi:\sigma$ | `terms.m2_pi_sigma` |
| (iii) | $2\tau_M\Pi_Qs_i$ | `terms.m2_PiQ_sigma` |
| (iv) | $2\lambda_a a_l\nu_l$ ∣ $(D_sA/3T)a\cdot\nu$ | `terms.m2_accel_nu` |
| (iv) | $(D_s/T)\nu_lD_l(Th)$ ∣ $(5D_s/6T)\nu_lD_l(Th)$ | `terms.m2_nu_gradTh` |
| — | vorticity coupling, the Δ-projector on $D\pi_Q$ | `terms.m2_vorticity`, `terms.m2_projector` — ≡ 0 here |

Gate T2 checks that the switched pieces sum to the all-on rows (1.7e-14). The all-on arithmetic is the expression as
shipped, and Fluidum's `hqc_m2_rows(:consistent)` matches it to 1.7e-15 (gate M4).

---

## 7. Regulators and guards — not physics, but they act

| | where | default | note |
|---|---|---|---|
| clips `nur_clip_factor`, `pi_clip_factor`, `Pi_clip_factor` | bulk | off | README.md "Stabilizers" |
| filters `*_filter_eps`, `*_smooth_len`; `do_soft_project_nur`, `do_axis_project_nur` | bulk | off | ″ |
| vacuum ramp $w(n)$ on every $dU$ | IS2 | $n_{lo}/n_{hi} = 10^{-6}/2\cdot10^{-3}$ fm⁻³ | ENV `FIVO_VACUUM_*`; O+O uses the relative ramp |
| α floor (softplus at −20), ν bound, cone projection, $J^\tau$ floor | IS2 | floor on, others off | ENV `FIVO_IS2_*` — README.md "Charm-sector regulators" |
| Kreiss–Oliger dissipation | IS2 | σ_KO = 0.2 | keyword |

---

## 8. Corrections log

Newest first. Each is also recorded at the code.

**2026-09-13 — ∂_τu^r at full strength makes the split NS target short-wavelength unstable.** The
09-11 fix above is right, and the closed-form gates agree with it — but the O+O bulk background ran
straight into what it exposed. ∂_τu^r enters θ_full, so Π_NS = −ζθ_full and the shear target carry it;
Π feeds back into the momentum equation, and for a mode e^{ikr} the viscous damping −ζk² picks up a
factor (e+P)²/[(e+P)²+k²ζ²v²], i.e. above k ≈ (e+P)/(ζv) the k² damping is gone. That is the
structure, not a threshold: taken literally ζ|v|/(e+P) is 0.4–0.9 fm here, which would condemn every
grid in use and does not. **The onset is measured, not derived.** On the O+O bulk over τ ∈ [3.4, 4.7]
(sign changes of ∂_r T over r ∈ [1, 4.2] fm, and max |second difference of T|, early → late):

| dr (fm) | ∂_τu^r | sign changes | max \|d²T\| | |
|---|---|---|---|---|
| 0.0260 | raw | 0–10 | 4.8e-5 → 1.2e-5 | decaying |
| 0.0130 | raw | 0–41 | 6.9e-5 → 2.6e-4 | **growing** |
| 0.0130 | limited, 0.10 fm | 0–6 | 1.2e-5 → 3.1e-6 | decaying |

**Refining the grid makes it worse** — an instability of the scheme, not a discretisation error, and it
does not converge away. A 5× smaller Δτ changes nothing (43 vs 43 sign changes, amplitudes within
10%), so no timestep rule can catch it and none was added. Measured here too: at Nr = 1000
(dr = 0.0065 fm) the solve CRAWLS — τ = 0.61 in 22 min against τ = 6.0 in 50 min at Nr = 500. The
09-11 acausal-Gubser entry below is the same phenomenon: its runaway also grew with resolution.

> ⚠ **Corrected 2026-09-14.** The first version of this entry also cited the 09-08 note "at
> nr=1000–2000 the solve CFL-CRAWLS rather than failing" as a second sighting **in FiVo**. That was
> wrong twice over: the line is about **Fluidum's** MIS solve, and about the longer **Pb+Pb** `t1`,
> not O+O — see `Projects/AttractorPaper1/Code/momentum_thermalization/fluidum_oo/README.md`, which
> records the O+O grid scan explicitly. The FiVo Nr = 1000 crawl above was measured directly and
> stands on its own.

**The cross-code referee says the corrections went the right way.** `diag_bulk_backend_comparison.jl` puts
both codes' splines on one 261×225 lattice and measures inside Σ_fo (32 522 cells). Measured 2026-09-08
against the PRE-fix FiVo, and again 2026-09-14 against the fixed one: **T max 1.04 % → 0.29 %, T p95
0.83 % → 0.14 %, |Δu^r| max 0.0519 → 0.0092** — 3.6–5.9× better agreement between two independent codes,
with central T matching to four digits at τ₀.

**Fluidum does not have this, and the reason is structural.** It writes the system quasilinearly,
`A_t ∂_τφ + A_x ∂_xφ + src = 0` for `φ = (T, u^r, π^φ_φ, π^η_η, Π)`, and puts the coefficient of
`∂_τu^r` **inside `A_t`**: in `one_d_viscous_matrix_derived` (`Fluidum.jl/src/Matrix/viscous_generated.jl`)
entry (5,2) — the Π row against the `∂_τu^r` column — is `ζ u^r/u^τ = ζv`, precisely the quantity
`relax_dissipative!` reaches back a step for. Entry (5,5) is `τ_B u^τ` and the two shear rows carry
`−2u^r η/(3R²u^τ)`. Solving for all five `∂_τφ` together makes the term implicit. **Measured:**
Fluidum's O+O bulk is grid-converged at 600/900/1200/1500 points over `rmax = 6.5` — dr down to
**0.0043 fm, 3× finer than where FiVo rings** — all reaching τ = 6 with `T(r)` and `ν^r(r)`
overlaying, and its stored O+O bulk passes the ringing gate (worst 2 sign changes, |d²T| decaying
9.1e-6 → 7.6e-7 over 105 stored times). **That, not the band limit, is the real fix for FiVo**; the
filter is a stopgap that is measured to work. Neither code dominates: in the **charm current**
Fluidum's explicit Tsit5 on the coupled IS matrix overshoots by 3.6× at τ₀ and ~70× by τ = 4, where
FiVo's operator-split exact exponential relaxation is unconditionally stable. Each is stiff where
the other is not. ⚠ The 2+1D cross-code gate (15/15) compares terms at a point and would not have
caught this: it is blind to long-time stability on a fine grid. `dtau_u_smooth_len` (fm, default 0.0 = off = every result before this date, bit for
bit) band-limits ∂_τu^r at a fixed PHYSICAL length before use, so the term is grid-convergent. The
check that it removes the artefact and not physics: the limited dr = 0.013 run agrees with the
INDEPENDENT stable dr = 0.026 run to 1e-3 relative in T inside the fireball and to 0.08% in max|Π|,
while the raw dr = 0.013 run additionally carries a 4% inflated max|π^r_r|. O+O production sets
0.10 fm (`OO_BG_DTAUUSMOOTH`). `diag.dtau_ur_q_max` reports whether ∂_τu^r is grid-scale at all —
measured 2.36 (dr=0.013, rings hard), 2.16 (dr=0.026, rings mildly), 0.014 (limited); it says IN THE
BAND, not how bad. run1d_gates 9/9 with the change.

*Not established here:* whether the Pb+Pb bulk (`bulk_background{solver=fivo}`, Nr=800, rmax=20 ⇒
dr = 0.025 fm) is affected. It sits just on the stable side of the O+O measurement, but that was
measured on a different fireball and has not been checked. Any rerun reports `dtau_ur_q_max`.

**2026-09-11 — ∂_τu^r was ~4% of its value in every 1-D relaxation (production path).** `y_prev` was copied at the
start of `relax_dissipative!`, where `work.y` holds the last RK stage, not the previous step. Measured on a flowing
fireball: |y_prev − y_new| = 3e-4 against |y_old − y_new| = 7e-3. So θ's $\partial_\tau u^\tau$, the NS targets built from
θ, $\sigma_l$ and the charge projector term were all short of it. Against viscous Gubser (A4) at Nr = 800, L2(π̄) is
132% with the old history and 1.5% (converging, order 1.1) with the fix. On the production Pb+Pb bulk IC the fix moves
u^r by +1–1.6% in the bulk, Π by 5–9% and π^η_η by 1–17% (τ = 4 fm, cells above T_fo). **Every 1-D FiVo bulk
background with viscosity was made with the old arithmetic.** `hydro.HYDRO_LEGACY_DTAU_UR[] = true` reproduces it.

**2026-09-11 — the ∂_τu^r fix exposed an acausal benchmark.** FiVoBenchmark's viscous Gubser used
C_s = `tauShear_coeff` = 1, i.e. η/(τ_π(e+P)) = 1, while conformal Israel–Stewart is causal only for C_s ≤ 1/2, and
an acausal IS theory is unstable in a moving frame. With ∂_τu^r restored the solver shows it: stable at
C_s ≤ 0.6, a runaway at 0.8 whose size grows with resolution (max u^r 9.9 → 21, Nr = 100 → 200), and a crash at
1.0. The old arithmetic ran every C_s because it lacked the term that carries the instability. The bench now
uses C_s = 0.2 (production), and `build_model_1d`, `build_model_2d` and `run_sim_ideal_diff_visc` warn above 1/2.

**2026-09-11 — SSPRK3 was first order.** Its third stage was evaluated at τ+Δ instead of τ+Δ/2, and the RHS depends on
τ explicitly. Uniform ideal Bjorken errors went 6.7e-3 / 4.3e-3 / 2.2e-3 as CFLτ halved; they are now order 2.99.
Production uses SSPRK2 (always correct); the CLI `main()` and FiVoBenchmark defaulted to SSPRK3.

**2026-09-11 — π and Π were zeroed on every other step for charge-free fluids at rest.** The at-rest, charge-free
branch of the recovery is a bisection that `relax_dissipative!` capped at 40 iterations, about 15 short of float
resolution. A correct root was reported as a failure and the relaxation's fail-safe zeroed the dissipatives. Viscous
heating in baryonless Bjorken came out 1.1% against 4.4%, and FiVoBenchmark's `bench_bjorken` still passed at 5%.

**2026-09-11 — the dormant DNMR couplings.** λ_πΠ and λ_Ππ had the mostly-minus sign, τ_ππ lacked its −π:σ/3 trace (2×
too strong on Bjorken), and $\sigma_l$ lacked $v\partial_\tau u^r$. All default to 0 and no caller sets them.
A2 checks each against the 0+1D DNMR ODEs.

**2026-09-11 — the IS2 diagnostic counters accumulated across solves** in one process (`nu_bound_hits`,
`jtau_nonpositive`, `cone_projections`). They are now per solve.

Earlier corrections (τ_n short by $g_{hq}$, the ∂_τα history, the α floor, the vacuum ramp) are in README.md
"Conventions and convention-changing commits" and at the code.

---

## 9. Time integration and the operator split

⚠ §9–§11 were added on 2026-09-14, after §8; the corrections log stays **§8** because six files across the
repository cite it by that number.

**Bulk solver** (`src/timestepper.jl`, `src/api1d.jl`):

| | |
|---|---|
| advection | SSPRK2 (Heun) on the conserved set, with MOOD; `integrator = :ssprk3` is third order for smooth ideal flow and runs without MOOD |
| step | $\Delta\tau = \min\big(\mathrm{CFL}\,dr/a_{\max},\ \mathrm{CFL}_\tau\,\tau\big)$, defaults 0.2 and 0.05, then capped by the parabolic limit `diff_dt_coeff` $dr^2/\kappa_{\rm eff}$ and by `shear_dt_coeff`$\,\tau_\pi$, `bulk_dt_coeff`$\,\tau_\Pi$ when those sectors are on (`compute_dt_from_work`, `main.jl`) |
| relaxation | once per accepted step, on the updated state at $\tau + \Delta$ (`relax_dissipative!`), between two rounds of floors/admissibility — first order in $\Delta\tau$ |
| $\partial_\tau$ history | $u^r$, $\alpha$, $T$ of the previous **step** (`work.y_prev`, `alpha_prev`, `T_prev`), NaN-seeded and dropped on step one; `reset_history = false` continues a run. ⚠ this is the history that was being read at the wrong point until 2026-09-11 (§8), and the one `dtau_u_smooth_len` band-limits since 2026-09-13 |
| failure handling | stage → BC → $\tilde D$ positivity → floors → θ-admissibility → $S$–$E$ bound → MOOD (first order locally, up to `max_stage_retries` = 3) → halve $\Delta\tau$ (up to 24 times) |

Order: 2.00 on ideal Bjorken (SSPRK2), 2.99 with SSPRK3 (gate A1); 1.8 on viscous Gubser in $T$ (A4).
First order once any dissipative sector is on — that is the split, not the integrator, and halving
`CFLτ` halves the error rather than quartering it.

**Charm IS2 solver** (`main2IS2.jl`): a quasi-linear $5\times5$ system integrated by classical **RK4**,
unsplit, with Kreiss–Oliger dissipation ($\sigma_{\rm KO} = 0.2$) and a step set by the characteristic
speeds of the system (its `eigvals` per face). The relaxation is exact-exponential rather than split,
which is why it is unconditionally stable where an explicit scheme on the same matrix is not (§8).

---

## 10. What is *not* carried

| | why |
|---|---|
| any vorticity coupling — `m2_vorticity`, `shear_vorticity` | $\omega^{\mu\nu} \equiv 0$ for a radial flow. The switches exist and are inert (`show_equations` marks them `≡0`); the 2+1D solver carries both |
| the Δ-projector on $D\pi_Q$ — `m2_projector` | the 1-D second moment lives in the parallel-transported triad $(l, \hat\phi, \hat\eta)$, orthogonal to $u$, so $-\tau_M(u^ic^j + u^jc^i)$ has no component there. Also inert, also real in 2-D |
| medium: $\varphi_7\pi\pi$ | not implemented (no knob) — the same truncation Fluidum makes |
| medium: $\tau_{\pi\pi}$, $\lambda_{\pi\Pi}$, $\lambda_{\Pi\pi}$, $\lambda_{NN}$ | implemented and gated (A2), but **default 0** and no production caller sets them. Only $\delta_{\pi\pi} = 4/3$ is set, by the production bulk callers |
| thermal / hydrodynamic fluctuations | none, in any solver |
| a transverse plane | by construction: $\varepsilon_2$, $\varepsilon_3$, $v_2$, $v_3$ and every vorticity effect need `main2D.jl` |
| charm $c_M$ back-coupling in production | implemented in the IS2 solver (`hq_cm_force`, `use_cM`), **off** in every production recipe; the 2-D solver has no twin |

---

## 11. Where each piece lives

| file | what |
|---|---|
| `main.jl` | module `hydro`, the CLI `main()`, `run_sim_ideal_diff_visc`, `compute_dt_from_work`, `initialize!` |
| `src/api1d.jl` | the library interface: `make_grid_1d`, `build_model_1d`, `allocate_state`, `set_cell!`/`finalize_ic!`/`initialize_*`, `run_sim_1d!`, `fields_1d`, `show_equations` |
| `src/terms.jl` | `Terms`, `TERM_REGISTER`, presets, ingredients, `without`, `show_terms` — **shared with the 2+1D solver** |
| `src/primitives.jl` | `IdealDiffViscModel` — every knob with its default and history; the viscosity models |
| `src/dissipation.jl` | kinematics, the NS targets, `relax_dissipative!`, `bandlimit_centered!`, every stabilizer |
| `src/hq_consistent_firstmoment.jl`, `src/hq_consistent_m2.jl` | the full-∇P first moment and the consistent second moment (both also called by the IS2 solver) |
| `src/primrec.jl` | the three-unknown Newton primitive recovery |
| `src/fluxes.jl`, `reconstruction.jl`, `rhs.jl`, `timestepper.jl`, `mood.jl` | the finite-volume scheme |
| `src/eos.jl` | `ConformalHQEOS`, `LatticeHRGEOS`, `TabulatedHQEOS` |
| `src/gubser.jl` | the analytic Gubser solution and `initialize_gubser!` |
| `src/fields_io.jl` | `save_fields` / `load_fields` — **shared with both other solvers** (gate IO) |
| `src/floors.jl`, `grid.jl`, `state_layout.jl`, `work.jl`, `boundary_conditions.jl` | floors, grid, layout, work arrays, boundaries |
| `src/runtime_flags.jl`, `diagnostics.jl`, `debugging.jl`, `io.jl` | the single `HYDRO_*` ENV reader, counters, debug dumps, snapshot I/O |
| `main2IS2.jl` | module `hydro_current_IS2`: the whole charm IS2 solver in one file (2211 lines) — `run_static_IS2_test`, `analytic_background`, `show_equations_IS2`, the RK4 stepper and its regulators |
| `src/is2_second_moment_builder.jl` | the IS2 matrices |
