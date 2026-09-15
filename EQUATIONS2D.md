# FiVo 2+1D — the equations, term by term

Every equation the 2+1D solver (`main2D.jl`, module `hydro2d`) integrates, one term at a time:
what it is, where it lives in the code, how to switch it off, and which gate checks it. This is
the reference; the chronological build log with every wrong turn is `TWOD_PROGRAM.md`, and the
front door is `README2D.md`.

> ⚠ Paths beginning `Julia/Projects/…` or `Tex/…` name the **private research repository** this
> package is developed in. They are cited for provenance — so a number can be traced to the script
> that produced it — and are not links you can follow from a clone of this package.


The same information at run time, for the model you actually built:

```julia
m = hydro2d.build_model_2d(; enable_diff = true, consistent_fm = true, consistent_m2 = true)
hydro2d.show_equations(m)        # every equation, every term [x]/[ ], and the knob for it
```

Math renders on GitHub and in the VS Code preview.

---

## 0. Conventions

| | |
|---|---|
| chart | Milne $(\tau, x, y, \eta_s)$, $g_{\mu\nu} = \mathrm{diag}(-1, 1, 1, \tau^2)$, boost invariant ($\partial_\eta = 0$, $u^\eta = 0$) |
| velocity | $u^\mu = (u^\tau, u^x, u^y, 0)$, $u^\tau = \sqrt{1 + (u^x)^2 + (u^y)^2}$, $v^i = u^i/u^\tau$ |
| comoving derivative | $D = u^\mu\partial_\mu = u^\tau\partial_\tau + u^x\partial_x + u^y\partial_y$ |
| expansion | $\theta = \nabla_\mu u^\mu = \partial_\tau u^\tau + \partial_x u^x + \partial_y u^y + u^\tau/\tau$ |
| acceleration | $a^i = D u^i$, $\;a^\tau = (u^x a^x + u^y a^y)/u^\tau$ (from $u\cdot a = 0$) |
| projector | $\Delta^{\mu\nu} = g^{\mu\nu} + u^\mu u^\nu$, $\;X^{\langle\mu\rangle} = \Delta^\mu{}_\nu X^\nu$ |
| units | fm and GeV; $\hbar c = 0.19733$ GeV fm (`fmGeV` $= 1/\hbar c$) |
| dissipatives | stored = physical (`DISS_SIGN = +1`) |

The only non-zero Christoffels are $\Gamma^\tau_{\eta\eta} = \tau$ and $\Gamma^\eta_{\tau\eta} = 1/\tau$. So for
transverse indices $\nabla_\mu u^i = \partial_\mu u^i$, and every geometric effect sits in the
$u^\tau/\tau$ of $\theta$ and in the source of the energy equation.

`kinematics_2d` (`src2d/dissipation2d.jl`) builds $\theta$, $a^i$, $a^\tau$ and the four transverse velocity
gradients **once** per cell; every sector below uses those. The $\partial_\tau u^i$ in them is differenced
against the previous step (`work.ux_prev`), and is dropped on the very first step of a run.

---

## 1. The medium — conservation laws

Evolved in flux form, with $\tilde D = \tau J^\tau$, $S^i = T^{\tau i}$, $E = T^{\tau\tau}$:

$$
\begin{aligned}
\partial_\tau \tilde D + \partial_x(\tau J^x) + \partial_y(\tau J^y) &= 0 \\
\partial_\tau S^x + \partial_x T^{xx} + \partial_y T^{xy} &= -S^x/\tau \\
\partial_\tau S^y + \partial_x T^{xy} + \partial_y T^{yy} &= -S^y/\tau \\
\partial_\tau E + \partial_x T^{\tau x} + \partial_y T^{\tau y} &= -\left(E + P + \Pi + \pi^\eta{}_\eta\right)/\tau
\end{aligned}
$$

with

$$
T^{\mu\nu} = (e + P + \Pi)\,u^\mu u^\nu + (P + \Pi)\,g^{\mu\nu} + \pi^{\mu\nu},
\qquad J^\mu = n\,u^\mu + \nu^\mu .
$$

| piece | code | gate |
|---|---|---|
| flux table and sources | `flux_cell_2d!`, `source_cell_2d!` (`fluxes2d.jl`) | W8 in `FiVoFluidumComparison/gate_2p1d_viscous.jl` (1.9e-16) |
| $\tau$-row of $\pi^{\mu\nu}$ from $u_\mu\pi^{\mu\nu} = 0$ | `shear_tensor_contravariant_2d` (`shear2d.jl`) | `test_shear2d_algebra.jl` |
| primitive recovery $(\log T, \phi, u^x, u^y)$ | `cons_to_prim_2d!` (`primrec2d.jl`) | `test_primrec2d.jl`, `test_primrec2d_vs_1d.jl` |
| EoS | `LatticeHRGEOS` (`src/eos.jl`) | `test_eos_consistency.jl` |
| HLLE + MUSCL(MC) in primitives, SSPRK2 | `hlle_flux_2d!`, `reconstruct_muscl_2d!`, `step_ssprk2_2d!` | G0 Bjorken (order 2.00), G1 Gubser (order 1.96) |

Nothing here is switchable: these are the conservation laws. `with_charge = false` removes $\tilde D$
and the charge sector altogether.

---

## 2. Shear

$$
\tau_\pi\,\Delta^{ij}{}_{\alpha\beta}\,D\pi^{\alpha\beta} + \pi^{ij}
= -2\eta\,\sigma^{ij} - \delta_{\pi\pi}\,\theta\,\pi^{ij}
- \tau_{\pi\pi}\,\pi^{\lambda\langle i}\sigma^{j\rangle}{}_\lambda - \lambda_{\pi\Pi}\,\Pi\,\sigma^{ij}
- 2\tau_\pi\,\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle}
$$

The last three terms are DNMR's $-\tau_{\pi\pi}\pi^{\langle\mu}{}_\lambda\sigma^{\nu\rangle\lambda} + \lambda_{\pi\Pi}\Pi\sigma^{\mu\nu} + 2\tau_\pi\pi_\lambda{}^{\langle\mu}\omega^{\nu\rangle\lambda}$
(mostly-minus) in this code's mostly-plus metric: the $\tau_{\pi\pi}$ and ω terms keep their form, λ_πΠ flips.
All three default to off/0 (since 2026-09-11 they are wired; before, the two factors were refused).

For a symmetric, traceless, $u$-orthogonal tensor the projector acts as

$$
\Delta^{ij}{}_{\alpha\beta}\,D\pi^{\alpha\beta} = D\pi^{ij} - u^i c^j - u^j c^i ,
\qquad c^j = \pi^{j\beta} a_\beta = -\pi^{j\tau}a^\tau + \pi^{jx}a^x + \pi^{jy}a^y ,
$$

so what is integrated, for $(ij) \in \{xx, xy, yy\}$, is

$$
\tau_\pi\left(u^\tau\partial_\tau + u^k\partial_k\right)\pi^{ij} + \pi^{ij}
= -2\eta\,\sigma^{ij} - \delta_{\pi\pi}\theta\,\pi^{ij} + \tau_\pi\left(u^i c^j + u^j c^i\right).
$$

$\pi^\eta{}_\eta = \tau^2\pi^{\eta\eta}$ is **never evolved**: tracelessness fixes it,
$\pi^\eta{}_\eta = \pi^{\tau\tau} - \pi^{xx} - \pi^{yy}$ (`project_shear_traceless_2d`).

The shear tensor, from $\sigma^{\mu\nu} = \nabla^{\langle\mu}u^{\nu\rangle}$ in this chart:

$$
\begin{aligned}
\sigma^{xx} &= \partial_x u^x + u^x a^x - \tfrac{\theta}{3}\left(1 + (u^x)^2\right) \\
\sigma^{yy} &= \partial_y u^y + u^y a^y - \tfrac{\theta}{3}\left(1 + (u^y)^2\right) \\
\sigma^{xy} &= \tfrac12\left(\partial_x u^y + \partial_y u^x\right) + \tfrac12\left(u^x a^y + u^y a^x\right) - \tfrac{\theta}{3}\,u^x u^y \\
\sigma^\eta{}_\eta &= u^\tau/\tau - \theta/3
\end{aligned}
$$

| term | code | switch | gate |
|---|---|---|---|
| $-2\eta\sigma^{ij}$ | `ns_shear_target_2d` | `enable_shear` | W2/W3 (`gate_2p1d_viscous.jl`, referee 1e-11), G2, Gs (attenuation) |
| $-\delta_{\pi\pi}\theta\pi^{ij}$ | `relax_shear_cell_2d!` (denominator) | `deltaShear_factor = 0` | W5 (closed form 1.1e-12) |
| $\tau_\pi(u^ic^j + u^jc^i)$ | `relax_shear_cell_2d!` | `shear_projected_deriv = false` | W2 |
| $\tau_\pi u^k\partial_k\pi^{ij}$ (upwind) | `relax_shear_cell_2d!` | `relax_advect_pi = false` | W6 (live update, config B) |
| $2\tau_\pi\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle}$ — **off by default** | `relax_shear_cell_2d!` → `vorticity_coupling_2d` | `terms.shear_vorticity` | Gt4 (the function), **Gt8** (on a solve) |
| $\tau_{\pi\pi}\pi^{\lambda\langle i}\sigma^{j\rangle}{}_\lambda = \tau_{\pi\pi}(c^{ij} - \Delta^{ij}\pi\!:\!\sigma/3)$ | ″ → `pi_sigma_contractions_2d` | `taupi_pi_factor` (default 0) | **Gd1** (0+1D DNMR ODE), Gd3 (the contraction vs brute force) |
| $\lambda_{\pi\Pi}\Pi\sigma^{ij}$ | ″ | `lambda_pi_Pi_factor` (default 0) | Gd1, Gd2 (= the 1-D solver to 1e-15) |

Coefficients: $\eta = (\eta/s)\,s\,\hbar c$, $\;\tau_\pi = \eta/(C_s\,T\,s)$ (`eta_over_s`, `tauShear_coeff` $= C_s$),
$\;\delta_{\pi\pi} = $ `deltaShear_factor` $\times\,\tau_\pi$.

> ⚠ **`deltaShear_factor` defaults to 4/3 here and in `build_model_1d`, but to 0 in `main.jl`'s legacy
> `run_sim_ideal_diff_visc`**, and Fluidum has no
> such term at all (`TWOD_PROGRAM.md` D16). Set it explicitly when comparing codes.

---

## 3. Bulk

$$
\tau_\Pi\left(u^\tau\partial_\tau + u^k\partial_k\right)\Pi + \Pi = -\zeta\,\theta - \delta_{\Pi\Pi}\,\theta\,\Pi - \lambda_{\Pi\pi}\,\pi\!:\!\sigma
$$

$\zeta = (\zeta/s)(T)\,s\,\hbar c$ with the Lorentzian peak
$(\zeta/s)(T) = (\zeta/s)_{\max} / \left[1 + ((T - 0.175)/0.024)^2\right]$, and
$\tau_\Pi = \zeta / \left[C_\zeta\,T\,s\,(1/3 - c_s^2)^2\right] + 0.1$ fm
(`zeta_over_s`, `tauPi_coeff` $= C_\zeta$).

| term | code | switch | gate |
|---|---|---|---|
| $-\zeta\theta$ | `relax_bulk_cell_2d!` | `enable_bulk` | G0b (nonlinear Bjorken), W2 (Π row 7.9e-14) |
| $\tau_\Pi u^k\partial_k\Pi$ | `relax_bulk_cell_2d!` | `relax_advect_Pi = false` | — |
| $-\delta_{\Pi\Pi}\theta\Pi$ (implicit) | ″ | `deltaPi_factor` (default 0) | Gd1 |
| $-\lambda_{\Pi\pi}\pi:\sigma$ (explicit; mostly-plus sign) | ″ → `pi_sigma_contractions_2d` | `lambda_Pi_pi_factor` (default 0) | Gd1, Gd2, Gd3 |
| positivity $P + \Pi > 0.01P$ | `relax_bulk_cell_2d!` | always on (a guard, §8) | — |

---

## 4. The charm current — first moment

The charge sector evolves $\nu^i$ (and $n$ through $\tilde D$). The row is derived from

$$
\frac{D_s}{T}\,\Delta^\mu{}_\lambda\,\nabla_\nu T_Q^{\nu\lambda} + \nu^\mu = 0,
\qquad
T_Q^{\mu\nu} = \varepsilon_Q u^\mu u^\nu + P_0\Delta^{\mu\nu} + h\left(u^\mu\nu^\nu + \nu^\mu u^\nu\right),
$$

with $P_0 = nT$, $\varepsilon_Q + P_0 = nh$, $h = m K_3(z)/K_2(z)$ the enthalpy per particle
($z = m/T$), and $\tau_n = D_s h/T$. Working the divergence out:

$$
\tau_n\,\Delta^i{}_\nu D\nu^\nu + \nu^i
= \underbrace{-\kappa\,\nabla^{\langle i\rangle}\alpha}_{\text{shipped drive}}
\;\underbrace{-\;\frac{D_s}{T}\Big(n + T\frac{\partial n}{\partial T}\Big)\nabla^{\langle i\rangle}T
\;-\;\tau_n\,n\,a^i
\;-\;\tau_n\,\nu^m\nabla_m u^i
\;-\;\nu^i\Big(\tau_n\theta + \frac{D_s}{T}h'\,DT\Big)}_{\text{consistent\_fm} \;=\; -s^i}
$$

with $\nabla^{\langle i\rangle}X = \partial_i X + u^i DX$, $\;\kappa = D_s n$,
$\;\Delta^i{}_\nu D\nu^\nu = D\nu^i - u^i(\nu\cdot a)$ and
$\;\nu^m\nabla_m u^i = \nu^\tau\partial_\tau u^i + \nu^x\partial_x u^i + \nu^y\partial_y u^i$,
$\nu^\tau = (u^x\nu^x + u^y\nu^y)/u^\tau$.

Every consistent term enters with a **minus**: they all come from the same divergence as $\nu$ itself.
Two checks on that sign. The expansion term gives the Bjorken dilution $\partial_\tau(\tau\nu) = 0$ in
the collisionless limit. And on an ideal baryon-free fluid, Euler ($a^i = -\nabla^{\langle i\rangle}T/T$)
together with $n + T\partial_T n = nh/T$ makes the pressure-gradient and inertial terms cancel
exactly — the hidden justification of the shipped $\nabla\alpha$-only drive.

What is integrated (backward Euler in $u^\tau\partial_\tau$, `relax_charge_cell_2d!`):

$$
\tau_n^{\rm eff}\left(u^\tau\partial_\tau + u^k\partial_k\right)\nu^i + \nu^i
= w(n)\left[-\kappa\nabla^{\langle i\rangle}\alpha - s^i\right] + \tau_n^{\rm eff}\,u^i(\nu\cdot a),
\qquad \tau_n^{\rm eff} = w(n)\,\tau_n ,
$$

where $w(n)$ is the vacuum ramp (§8).

| term | code | switch | gate |
|---|---|---|---|
| $-\kappa\nabla^{\langle i\rangle}\alpha$ | `ns_diffusion_target_2d` | `terms.nu_gradalpha` | G3, Gk (dispersion), G3g |
| $-(D_s/T)(n + T\partial_T n)\nabla^{\langle i\rangle}T$ | `consistent_fm_source_2d` | `terms.fm_gradT` | Gc1–Gc4, E3 (`gate_charm_eos.jl`), T1–T4 (`gate_transverse_fm.jl`) |
| $-\tau_n n a^i$ | ″ | `terms.fm_inertial` | Gc1, Gc2 |
| $-\tau_n\nu^m\nabla_m u^i$ | ″ | `terms.fm_nu_gradu` | Gc1, Gc2 |
| $-\tau_n\theta\,\nu^i$ | ″ | `terms.fm_expansion` | Gc3, **Gc7** (the sign, on a solve) |
| $-(D_s/T)h'DT\,\nu^i$ | ″ | `terms.fm_dlnh` | Gc1, Gc7 |
| $\tau_n u^i(\nu\cdot a)$ | `relax_charge_cell_2d!` | `diff_projected_deriv = false` | — |
| $\tau_n u^k\partial_k\nu^i$ (upwind) | ″ | `relax_advect_nu = false` | — |
| all of the consistent terms | | `consistent_fm = false` | Gc5 (off = bit-identical to the shipped row) |

`consistent_fm_source_2d` returns $s^i$ **source-on-the-LHS** — the convention of the 1-D
`hq_consistent_extras` and of Fluidum's generated row, which is what lets Gc1 compare them bit for
bit. The relaxation **subtracts** it. Until 2026-09-10 it added it (§10).

Coefficients:

| | formula | code |
|---|---|---|
| $D_s$ | $(D_sT)/T\cdot\hbar c$, from `kappa_coeff` $= D_sT$ | `diff_coeffs_2d` |
| $\kappa$ | $D_s\,n$ | ″ |
| $\tau_n$ | $D_s h/T = D_sT\;zK_3/K_2\,/\,T$ — a moment ratio, no $g_{hq}$ | `_diff_tauN_impl_2d` |
| $h,\;h'$ | $mK_3/K_2$, $\;(mz/T)\left[(K_2+K_4)K_2 - (K_1+K_3)K_3\right]/(2K_2^2)$ | `hq_h_hprime_2d` |
| $\partial n/\partial T\vert_\alpha$ | $(n/T)(3 + zK_1/K_2)$ | `hq_dn_dT_2d` |

> **The EOS the 2-D solver evaluates** is `eos_Pne_2d` (`src2d/primrec2d.jl`), not `eos_Pne`
> directly: same arithmetic, with $K_2(m/T)$ from the fast `src2d/bessel2d.jl` and the T-only part
> memoised inside each Newton solve. P and e are bit-identical to `eos_Pne`; the charm density agrees
> to ~1e-15 (it is a different, equally accurate Bessel algorithm). `src/eos.jl`, shared with the 1-D
> production solver, is untouched. Gate: `test_primrec2d.jl`; the measurement: `TWOD_PROGRAM.md` §6ai.

---

## 5. The charm second moment

With `consistent_m2 = true` the charm stress $\pi_Q^{ij}$ (stored like the medium shear: $xx, xy, yy$
and a redundant $\pi_Q{}^\eta{}_\eta$) and its trace $\Pi_Q$ are evolved beside $\nu$. They are **passive**
($c_M = 0$): they read the medium and the current and feed back into neither.
Derivation: `Tex/LangevinPaper1/M2_CONSISTENT_DERIVATION.md` §4. All on the left, $= 0$:

$$
\begin{aligned}
&\tau_M\,\Delta^{ij}{}_{\alpha\beta}D\pi_Q^{\alpha\beta} + \pi_Q^{ij}
+ 2\eta_Q\,\sigma_{(\nu)}^{ij}
+ \tau_M\Big(\tfrac53\theta + D\ln C\Big)\pi_Q^{ij}
+ 2\tau_M\,\pi_Q^{\lambda\langle i}\sigma^{j\rangle}{}_\lambda
+ 2\tau_M\,\pi_Q^{\lambda\langle i}\omega_\lambda{}^{j\rangle}
+ 2\tau_M\,\Pi_Q\,\sigma^{ij} \\
&\qquad + 2\bar\eta\,\sigma^{ij}
+ 2\lambda_a\,a^{\langle i}\nu^{j\rangle}
+ \frac{D_s}{T}\,\nu^{\langle i}\nabla^{j\rangle}(Th) = 0 ,
\\[4pt]
&\tau_M\,D\Pi_Q + \Pi_Q + \zeta_Q\,\theta_{(\nu)}
+ \tau_M\Big(\tfrac53\theta + D\ln C\Big)\Pi_Q
+ \tfrac23\tau_M\,\pi_Q\!:\!\sigma
+ \bar\eta\Big(D\alpha + \tfrac{A}{B}D\ln T + \tfrac53\theta\Big) \\
&\qquad + \frac{D_sA}{3T}\,a\cdot\nu + \frac{5D_s}{6T}\,\nu\cdot\nabla(Th) = 0 .
\end{aligned}
$$

$\sigma_{(\nu)}$ is the traceless projected gradient of the **current**. It is *not* the $u$-shear with
$u \to \nu$: the projector drags in gradients of $u$ through $\nu^\tau$. It is generated
(`Julia/tools/derive_signu_2p1d.wls`), and so are $\pi\!:\!\sigma$ and the (iii) contraction, because
every contraction of these contravariant components needs the metric.

| class | term (traceless $\;\vert\;$ trace) | switch | code | gate |
|---|---|---|---|---|
| — | $\tau_M\Delta D\pi_Q$: the projector part $-\tau_M(u^ic^j + u^jc^i)$ | `terms.m2_projector` | `consistent_m2_source_2d` | **Gm1** (with the moving projection) |
| (i) | $2\eta_Q\sigma_{(\nu)}^{ij}$ $\;\vert\;$ $\zeta_Q\theta_{(\nu)}$ | `terms.m2_nu_gradient` | ″, `sigma_nu_2d` | Gm1, Gm2, N1–N4 |
| (ii) | $2\bar\eta\,\sigma^{ij}$ $\;\vert\;$ $\tfrac53\bar\eta\,\theta$ | `terms.m2_bg_gradu` | ″ | Gm1, Bjorken trace referee |
| (ii) | $\;\vert\;$ $\bar\eta\,(A/B)\,D\ln T$ | `terms.m2_bg_DlnT` | ″ | Gm1, Bjorken trace referee (10 digits) |
| (ii) | $\;\vert\;$ $\bar\eta\,D\alpha$ | `terms.m2_bg_Dalpha` | ″ | Gm1 |
| (iii) | $\tau_M(\tfrac53\theta + D\ln C)\,\pi_Q$ $\;\vert\;$ same $\times\Pi_Q$ | `terms.m2_expansion` | ″ | Gm1 |
| (iii) | $2\tau_M\pi_Q^{\lambda\langle i}\sigma^{j\rangle}{}_\lambda$ $\;\vert\;$ $\tfrac23\tau_M\pi_Q\!:\!\sigma$ | `terms.m2_pi_sigma` | ″ | Gm1, Gm3 |
| (iii) | $2\tau_M\Pi_Q\sigma^{ij}$ (δM-extension) | `terms.m2_PiQ_sigma` | ″ | Gm1 |
| (iii) | $2\tau_M\pi_Q^{\lambda\langle i}\omega_\lambda{}^{j\rangle}$ — **off by default** | `terms.m2_vorticity` | `vorticity_coupling_2d` | **Gt4** |
| (iv) | $2\lambda_a a^{\langle i}\nu^{j\rangle}$ $\;\vert\;$ $(D_sA/3T)\,a\cdot\nu$ | `terms.m2_accel_nu` | ″, `rank1_traceless_2d` | Gm1, Gm2 |
| (iv) | $(D_s/T)\nu^{\langle i}\nabla^{j\rangle}(Th)$ $\;\vert\;$ $(5D_s/6T)\,\nu\cdot\nabla(Th)$ | `terms.m2_nu_gradTh` | ″ | Gm1, Gm2 |
| — | $\tau_M u^k\partial_k(\pi_Q, \Pi_Q)$ (upwind) | `relax_advect_m2 = false` | `relax_charm_m2_cell_2d!` | — |

**The vorticity coupling**, with $\omega^{\alpha\nu} = \tfrac12(\nabla_\perp^\alpha u^\nu - \nabla_\perp^\nu u^\alpha)$
(derivative index first), is

$$
2\pi^{\lambda\langle i}\omega_\lambda{}^{j\rangle} = (\pi g\,\omega)^{ij} + (\pi g\,\omega)^{ji},
\qquad (\pi g\,\omega)^{i\nu} = \pi^{i\lambda}g_{\lambda\lambda}\,\omega^{\lambda\nu}.
$$

It vanishes identically for a radial flow — which is why the 1-D code never needed it, and why both
2-D codes, written from the 1-D reduction, lacked it. Measured $|\omega|/|\sigma|$ over cells above
$T_{\rm fo}$: a smooth elliptic fireball $4\cdot10^{-5}$ at $\tau = 1$ rising to $10^{-3}$ at $\tau = 3$
(median; max $3\cdot10^{-2}$); a lumpy event $8\cdot10^{-4} \to 8\cdot10^{-3}$ (max 0.12). It is off
by default so that every existing number and the parity with Fluidum stay put; example 06 measures
it.

Coefficients (`hq_m2_coeffs_2d`, `tauM_charm_2d`, `ηM_charm_2d`), with $z = m/T$:

| | formula |
|---|---|
| $\tau_M$ | $\tau_n\,K_4K_2/(2K_3^2) = (D_s z/2)\,K_4/K_3$ |
| $\eta_Q$ | $T\tau_n/2$, $\quad\zeta_Q = \tfrac53\eta_Q$ |
| $\bar\eta$ | $\tau_n P_0/2 = \tau_n nT/2$ — the charm's effective background-shear viscosity |
| $\lambda_a$ | $D_s(m^2 + 6Th)/(2T)$ |
| $A$, $\;A/B$ | $m^2 + 5Th$, $\;5 + zK_2/K_3$ |
| $d\ln C/d\ln T$ | $z\left[(K_3 + K_5)/(2K_4) - (K_2 + K_4)/(2K_3)\right]$ |

The update (`relax_charm_m2_cell_2d!`): each channel is affine in its own field. Its coefficient is
**measured** by one extra source evaluation (the $(1 + \text{geo})$ it was once assumed to be is 12 %
off, because the (iii) coupling is linear in $\pi$ too), and the backward-Euler step is then exact.
The update waits for the $\partial_\tau$ history, because $D\alpha$ and $D\ln T$ dominate the trace row.

---

## 6. Time integration and the operator split

| | |
|---|---|
| advection | SSPRK2 (Heun) on the conserved set; SSPRK3 available (`integrator = :ssprk3`, no MOOD) |
| step | $\Delta\tau = \min(\mathrm{CFL}\,\min(dx,dy)/a_{\max},\ \mathrm{CFL}_\tau\,\tau)$, defaults 0.2 and 0.05. $\mathrm{CFL}_\tau$ binds below $N \approx 150$ |
| relaxation | once per accepted step, after one fresh primitive recovery (`update_primitives_2d!`, then `relax_dissipative_2d!`) — first order in $\Delta\tau$ |
| $\partial_\tau$ history | $u^i$, $\alpha$, $T$, $\nu^i$ of the previous step (`Work2D.*_prev`), NaN-seeded; the $\partial_\tau$ pieces are dropped on step one |
| failure handling | stage → sanitise (floors, $S$–$E$ bound, $\nu$ bound) → admissibility → MOOD (first order locally, twice) → halve $\Delta\tau$ |

Order: 2.00 on Bjorken, 1.9–2.1 on Gubser (ideal); first order once a dissipative sector is on
(the split).

---

## 7. What is *not* carried

| | why |
|---|---|
| ~~medium: $\tau_{\pi\pi}$, $\lambda_{\pi\Pi}$, $\lambda_{\Pi\pi}$, $\delta_{\Pi\Pi}$~~ | **wired since 2026-09-11** (§2, §3; gate Gd). Until then they were refused (`reject_unwired_knobs_2d`) |
| medium: $\varphi_7\pi\pi$ | not implemented (no knob) — the same truncation Fluidum makes. (The medium's $\pi\omega$ coupling IS available since 2026-09-11: `terms.shear_vorticity`, §2, off by default) |
| charm $c_M$ back-coupling ($\pi_Q \to \nu$) | the second moment is passive; the 1-D `hq_cm_force` has no 2-D twin |
| thermal / hydrodynamic fluctuations | none, in either solver |
| a $\tau_n$ / $h$ decoupled from the EoS mass | `transport_mass` reaches the second moment's coefficients and $\tau_M$, but $\tau_n$, $h$ and $h'$ still use the EoS mass |

---

## 8. Regulators and guards — not physics, but they act

| | default | where it acts | note |
|---|---|---|---|
| vacuum ramp $w(n) = \mathrm{clamp}\big((n - n_{\rm lo})/(n_{\rm hi} - n_{\rm lo}), 0, 1\big)$ on the charge drive **and** $\tau_n$ | $n_{\rm lo} = 10^{-6}$, $n_{\rm hi} = 2\cdot10^{-3}$ fm⁻³ | dilute tail; reaches above $T_{\rm fo}$ ($w_{\min} = 0.22$ there on the production IC) | moves the relaxation rate, not the fixed point. The second moment is handed the ramped $\tau_n$ too |
| `T_vac_cut` | 0.05 GeV | cells colder are vacuum | measured insensitive (G4: 1.3 % over a 4.5× change) |
| $P + \Pi \ge 0.01P$ | on | bulk | |
| relaxation denominator $\ge (A+1)/2$ | on | shear, charge, second moment | inert in production ($A \approx 100$) |
| `pi_clip_factor`, `Pi_clip_factor`, `nu_clip_factor` | off ($-1$) | $\lvert\pi\rvert, \lvert\Pi\rvert \le fP$; $\lvert\nu\rvert \le f n u^\tau$ | needed on lumpy events (G9 sets `pi_clip_factor = 1`); measured inert on smooth ICs |
| `r_domain` | $\infty$ | cells outside the disc held at vacuum | for like-for-like comparison with the 1-D grid |

---

## 9. Where each piece lives

| file | what |
|---|---|
| `main2D.jl` | module, `build_model_2d`, ICs (`set_cell!`, `finalize_ic!`, `initialize_*`), `run_sim_2d!` |
| `src2d/terms2d.jl` | `Terms2D`, `TERMS_2D`, `show_equations` |
| `src2d/primitives2d.jl` | `IdealDiffVisc2DModel` — every knob, with its default and history |
| `src2d/dissipation2d.jl` | kinematics, NS targets, the four `relax_*_cell_2d!` |
| `src2d/hq_consistent_firstmoment2d.jl` | $s^i$, $h$, $h'$, $\partial_T n$ |
| `src2d/hq_consistent_m2_2d.jl` | second-moment sources, $\sigma_{(\nu)}$, the vorticity coupling |
| `src2d/transport2d.jl` | $\tau_n$, $\tau_M$, $\eta_M$, the vacuum ramp |
| `src2d/bessel2d.jl` | the fast scaled $K_2$ the 2-D EOS uses (vendored from Bessels.jl, MIT) |
| `src2d/shear2d.jl` | $\pi^{\mu\nu}$ storage, orthogonality, tracelessness |
| `src2d/fluxes2d.jl`, `rhs2d.jl`, `reconstruction2d.jl`, `timestepper2d.jl` | the finite-volume scheme |
| `src2d/primrec2d.jl`, `floors2d.jl`, `bc2d.jl`, `grid2d.jl`, `state_layout2d.jl`, `work2d.jl` | recovery, floors/MOOD, boundaries, grid, layout, work arrays |

---

## 10. Corrections log

Newest first. Each entry is also recorded at the code it concerns.

**2026-09-11 — the medium's DNMR couplings are wired** (τ_ππ, λ_πΠ, δ_ΠΠ, λ_Ππ; §2–§3), with the
equations and signs the 1-D solver was corrected to the same day, and gated against the 0+1D DNMR ODEs (Gd).
π:σ and the ⟨⟩-projected π·σ come from `pi_sigma_contractions_2d`, factored out of the charm second moment
without changing its arithmetic (a 2-D run with every sector and both closures is bit-identical to the
previous commit).

**2026-09-11 — the term switches became the shared `Terms` (src/terms.jl)** with presets, `without(...)` by
ingredient and a sector rule; `Terms2D` is an alias, and every default is unchanged (Gt1–Gt6 pass at the same
numbers). Added `terms.shear_vorticity`, the medium's vorticity coupling (§2), off by default. No 2-D arithmetic
changed with the defaults. The 1+1D solver's corrections of the same day (EQUATIONS1D.md §8) touch 2-D only
through the gates that compare against it (G4, G7).

**2026-09-10 — the charm second moment lacked the projector on its comoving derivative.**
The medium shear always carried $\tau_\pi(u^ic^j + u^jc^i)$; the charm second moment integrated a bare
$\tau_M D\pi_Q^{ij}$. At gate Gm1's states the missing term is 0.02–0.57 % of the rate. Gm1 could not
see it: it mapped the 2-D rate onto the 1-D $p_l$ with the projection held fixed in time. But
$l^\mu = (u^r, u^\tau)$ moves with the flow, and the difference is exactly this term. It also compared
states with different gradients (the 1-D side had $\partial_r p_l = 0$, the 2-D side $\partial_x\pi^{xx} = 0$).
Gm1 now differentiates the projection along the trajectory and matches the states; it holds at 4e-16
with the term, and fails at 5.7e-3 without it. `terms = (m2_projector = false,)` reproduces the old
rows. Fluidum's `HQ_2p1d_BG_m2.jl` agreed with the old rows at 2.2e-16 (N3), so it lacked the term
too — **until 2026-09-11**, when both terms were derived symbolically (`derive_hq_m2_2p1d.wls`
G6/G7, 22/22) and implemented there as well. The two codes now agree at ~1e-16 with the projector on,
with the vorticity coupling on, and with both (`gate_2p1d_m2_newterms.jl` M1-M3)
— not changed there.

**2026-09-10 — the consistent first moment had the wrong sign.**
`consistent_fm_source_2d` returns $s^i$ source-on-the-LHS and the relaxation *added* it to the
right-hand target, from 2026-09-08 until today. So every consistent term entered with the wrong sign.
On a uniform Bjorken state the current *grew* ($\tau\nu$: 1.00 → 1.63 over $\tau = 0.4 \to 1.2$) where
it must dilute (→ 0.24). Every gate that checks $s^i$ at a fixed state passed, because the value was
right. New gate **Gc7** evolves the state, and fails on the old code. The at-rest referee
`FiVoFluidumComparison/gate_transverse_fm.jl` had been adjusted to the solver's sign; with the derived
sign restored, FiVo matches it at 0.37 % and **Fluidum's independently generated row matches the same
fixed point at 2.5e-6** (amplitude 1.00000, previously "0.129, unexplained"). Only runs with
`consistent_fm = true` are affected — none of them production.

**2026-09-10 — the vorticity coupling of the second moment was absent** (§5). Added as
`terms.m2_vorticity`, off by default, gated by Gt4.

Earlier corrections to this solver — the four second-moment update defects, the $\partial_\tau$ history
defects, the reflecting vacuum wall, $\tau_n$ short by $g_{hq}$ — are in `TWOD_PROGRAM.md` (§6ae–§6ag
and D1–D16) and at the code.
