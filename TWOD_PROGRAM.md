# FiVo 2+1D — build programme

Living doc for the 2+1D (transverse Cartesian, boost-invariant Milne) extension of the FiVo bulk
solver, carrying **bulk + charge**: `(T, u^x, u^y, Π, π^{xx}, π^{xy}, π^{yy}, π^{ηη}, n, ν^x, ν^y)`.

Status board is at the bottom. The narrative of *why* belongs here; numbers belong to the gates.

---

## 0. What this is and is not

The production solver (`main.jl`, `src/`) is **1+1D radial Milne**: boost-invariant *and*
azimuthally symmetric. This project adds a sibling `main2D.jl` + `src2d/` that drops the azimuthal
symmetry. It does **not** touch `src/` or `main.jl` — the 1-D production path stays bit-identical so
every published number remains reproducible.

`Julia/Projects/FiVoBenchmark/fivo2d.jl` (`FiVo2D`) is a *different* thing: a 249-line self-contained
flat-Minkowski Γ-law solver for standard shock-tube benchmarks. It shares the scheme but no code. It
is useful here only as a flat-limit reference.

### The IC caveat — read this before interpreting any result

The repo's current ICs are **azimuthally symmetric by construction**, and the symmetry is imposed
*upstream of FiVo*: `Julia/Projects/ALICE_IC_Creation/MCGCollisionDensity.jl:34` deposits each binary
collision as `azimuthal_gauss(r, ρ, w)` — the φ-average of a unit Gaussian. `data/initial_profiles_physical.csv`
is `r,T0,alpha0` on 1002 radial points. MonteCarloGlauber's 2-D participant field `evt(x,y)` exists
and is documented (`MCGCollisionDensity.jl:4`) but is discarded.

Consequence: **running the current ICs in 2+1D is a validation exercise, not a physics result.** The
2-D answer must reproduce the 1-D answer. That is milestone G4 below and it is the single most
valuable gate in the ladder — but it produces no new physics. New physics needs the φ-averaging
removed from the IC builder, which is a separate and much smaller job (tracked as F1).

---

## 1. Geometry and the conservative form

Coordinates `x^μ = (τ, x, y, η)`, metric `g_{μν} = diag(-1, 1, 1, τ²)`, `√-g = τ`.
Boost invariance: `∂_η = 0`, `u^η = 0`. Non-zero Christoffels: `Γ^τ_{ηη} = τ`, `Γ^η_{τη} = 1/τ`.

`u^μ = (u^τ, u^x, u^y, 0)` with `u^τ = sqrt(1 + (u^x)² + (u^y)²)`.

Using `∇_μ A^μ = (1/τ) ∂_μ (τ A^μ)` and `∇_μ T^{μν} = (1/τ)∂_μ(τ T^{μν}) + Γ^ν_{μλ}T^{μλ}`:

| conserved | definition | x-flux | y-flux | source |
|---|---|---|---|---|
| `D̃` | `τ J^τ` | `τ J^x` | `τ J^y` | `0` |
| `S^x` | `T^{τx}` | `T^{xx}` | `T^{xy}` | `-S^x/τ` |
| `S^y` | `T^{τy}` | `T^{xy}` | `T^{yy}` | `-S^y/τ` |
| `E` | `T^{ττ}` | `T^{τx}` | `T^{τy}` | `-(E + P + Π + τ²π^{ηη})/τ` |

**This matches the 1-D code's conventions exactly** (`src/primrec.jl:778-780`, `src/fluxes.jl`
`source_cell_fast_col!`): only `D` carries the explicit `τ` weight; the `τ` factor for `E`, `S` is
absorbed into the geometric source. Verified against the 1-D energy source
`-(E + (P+Π) + τ²π^{ηη})/τ`, which is reproduced term for term.

**What disappears relative to 1-D radial:** the `+(P + Π + r²π^{φφ})/r` momentum source, the
`1/r ∂_r(r F)` flux weighting (`src/rhs.jl:266-280`), the `r=0` axis and its odd-parity ghost
reflection, and the `π^r_r = π^φ_φ` axis-regularity patch (`src/boundary_conditions.jl:20-30`).
2-D Cartesian Milne is geometrically *simpler* than 1-D cylindrical Milne.

### Fluxes

`T^{μν} = w_eff u^μu^ν + (P+Π) g^{μν} + π^{μν}`, `w_eff = e + P + Π`, `J^μ = n u^μ + ν^μ`.

```
T^{xx} = w_eff (u^x)² + P + Π + π^{xx}      T^{τx} = w_eff u^τ u^x + π^{τx}
T^{yy} = w_eff (u^y)² + P + Π + π^{yy}      T^{τy} = w_eff u^τ u^y + π^{τy}
T^{xy} = w_eff u^x u^y + π^{xy}             T^{ττ} = w_eff (u^τ)² - P - Π + π^{ττ}
```

---

## 2. The shear sector — the one real derivation

Symmetric `π^{μν}`, traceless, orthogonal to `u`. Boost invariance kills every η-mixed component
(`π^{τη} = π^{xη} = π^{yη} = 0`, odd under `η → -η`). That leaves 7 components
(`ττ, τx, τy, xx, xy, yy, ηη`) constrained by 3 orthogonality relations plus 1 trace ⇒
**3 independent dofs**, the same count Fluidum uses (`piyy, pizz, pixy`).

With `u_τ = -u^τ`, `u_x = u^x`, `u_y = u^y`:

**Orthogonality** `u_μ π^{μν} = 0`:
```
π^{τx} = (u^x π^{xx} + u^y π^{xy}) / u^τ
π^{τy} = (u^x π^{xy} + u^y π^{yy}) / u^τ
π^{ττ} = (u^x π^{τx} + u^y π^{τy}) / u^τ
       = [ (u^x)² π^{xx} + 2 u^x u^y π^{xy} + (u^y)² π^{yy} ] / (u^τ)²
```
(the `ν = η` relation is satisfied identically.)

**Tracelessness** `g_{μν}π^{μν} = 0`:
```
τ² π^{ηη} = π^{ττ} - π^{xx} - π^{yy}
```

**Why `π^{xx}` is the safe one to eliminate.** Substituting and using
`(u^x)² - (u^τ)² = -(1 + (u^y)²)`:
```
π^{xx} = [ 2 u^x u^y π^{xy} + (u^y)² π^{yy} - (u^τ)² (π^{yy} + τ² π^{ηη}) ] / (1 + (u^y)²)
```
The denominator is `≥ 1` for any flow — never singular. This is why Fluidum stores `(yy, zz, xy)`
and not `xx`.

### Decision: store 4, project to 3

We store **`π^{xx}, π^{xy}, π^{yy}, π^{ηη}`** (one redundant), advect them as scalars exactly as the
1-D code advects `piR, piEta`, and after each relaxation substep project onto the constraint surface.

Rationale:
- **x↔y symmetry stays manifest.** The 3-dof closure singles out `x`, so an azimuthally symmetric IC
  would develop a truncation-level `x`/`y` asymmetry from the *closure*, not the physics. Since G4
  (the reproduction gate) is exactly a symmetric-IC test, a formulation that breaks the symmetry by
  construction would poison the most important measurement in the ladder.
- **The redundancy is a free correctness monitor.** The constraint residual
  `R = (-π^{ττ} + π^{xx} + π^{yy} + τ²π^{ηη}) / |π|` is a per-cell diagnostic that should sit at
  round-off. A drifting `R` is an early warning nothing else in the scheme provides. It becomes a
  gate, in the house style.
- Cost is one extra advected scalar out of eleven fields.

The projection is cross-checked against the closed form above (they must agree to round-off when the
constraint is already satisfied).

---

## 3. The charge sector

`u_μ ν^μ = 0` ⇒ `ν^τ = (u^x ν^x + u^y ν^y) / u^τ`, so `J^τ = n u^τ + (u^x ν^x + u^y ν^y)/u^τ`.
The 1-D code's `D = n u^τ + v ν^r` (`src/primrec.jl:773`) is the `u^y = 0` case. ✓

Open item (**D2**): the 1-D relaxation carries a covariant NS drive with a temporal piece,
`∇^⟨r⟩α = u_τ² ∂_r α + u^r u^τ ∂_τ α` (`src/dissipation.jl`, `work.alpha_prev`). The comment there
records that dropping `∂_τ α` left `ν^r` **≈2× under-driven** against Fluidum. The 2-D analogue
`∇^⟨i⟩α = Δ^{iμ}∂_μ α` must be *derived*, not pattern-matched, and gated the same way.

---

## 4. Primitive recovery — the schedule risk

1-D solves a 3-unknown Newton on `(yT=log T, φ=(μ-m)/T, y=asinh u^r)`
(`src/primrec.jl:94` `evalF_ideal_phi_diff_visc!`):
```
F1 = n u^τ + v ν^r - D
F2 = w_eff u^τ u^r + π^{τr} - S_r
F3 = w_eff (u^τ)² - P - Π + π^{ττ} - E
```
In 1-D the stored parametrisation makes `π^{τr} ∝ u^τ u^r`, so the shear contribution to momentum is
**collinear with u by construction**. In 2-D with `π^{xy} ≠ 0` that collinearity is gone: the flow
*direction* becomes a genuine unknown. The system goes to **4 unknowns** `(yT, φ, u^x, u^y)`:
```
F1 = n u^τ + (u^x ν^x + u^y ν^y)/u^τ - D
F2 = w_eff u^τ u^x + π^{τx} - S^x
F3 = w_eff u^τ u^y + π^{τy} - S^y
F4 = w_eff (u^τ)² - P - Π + π^{ττ} - E
```
The 1-D guards that make this robust (`Y_CAP` on a scalar rapidity, the `|y| ≤ 1e-12` degenerate
branch, `copysign` seeding off `sign(Sr)`) do not generalise directly. Expect this file to consume a
disproportionate share of the effort. Needs a `solve4x4` to sit beside `solve3x3_gauss!`
(`src/utils.jl:33`).

---

## 5. Reuse map

Reused **unchanged** from `src/` (~1,800 lines): `eos.jl`, `constants.jl`, `utils.jl`,
`relaxation_laws.jl`, `runtime_flags.jl`, `logging_setup.jl`, and the transport-coefficient models in
`primitives.jl` (`QGPViscosity`, `SimpleBulkViscosity`, `LocalThermo`).

Reused **with interface adaptation**: `timestepper.jl` (411 lines, only 2 references to `grid.rC`,
both in error messages — it operates on `(a,i)` matrices and is index-agnostic), `mood.jl`,
`diagnostics.jl`, `work.jl`. **The state is already stored on a flat cell index everywhere**, so with
`i = (ix-1)*Nytot + iy` the RK stages, MOOD repair, primfail plumbing and work arrays carry over with
their shapes unchanged. This is the single biggest reason the port is tractable.

**Rewritten** in `src2d/`: `grid2d.jl`, `state_layout2d.jl`, `shear2d.jl`, `primrec2d.jl`,
`fluxes2d.jl`, `rhs2d.jl`, `bc2d.jl`, `reconstruction2d.jl`, `dissipation2d.jl`, `io2d.jl`.

---

## 6. Validation ladder

| gate | test | criterion |
|---|---|---|
| G0 | Bjorken — uniform transverse, ideal and viscous | exact; error at round-off |
| G1 | Constraint residual `R` on every gate | at round-off, non-drifting |
| G2 | Gubser ideal + viscous | analytic. Note Gubser is *also* azimuthally symmetric — tests the 2-D machinery, not azimuthal structure. **IDEAL Gubser IMPLEMENTED as G1 (§6k), 2nd order, 13/13. VISCOUS Gubser still open** — it needs the semi-analytic de Sitter ODE solution, which is not written. |
| G3 | flat-Minkowski limit vs `FiVo2D`: 2-D Riemann, cylindrical explosion, sound | matches the existing 42-PASS benchmark values |
| **G4** | **reproduction gate** — 2-D run seeded from `data/initial_profiles_physical.csv` vs the 1+1D production run | radial profiles at truncation error; azimuthal variance at round-off; x↔y asymmetry at round-off |
| G5 | conservation of `∫ τ dV` of `D` and `E` | at round-off over the full run |
| G6 | symbolic: FiVo-2D vs `two_d_viscous_HQ_matrix`, in the style of `verify_fivo_vs_fluidum_1st_moment.wl` | same continuum equations term by term |
| G7 | genuinely non-symmetric IC (b≠0) | runs to completion; `ε₂ → v₂` right sign and magnitude |

G4 is the money gate. G6 is the strongest correctness argument and is cheap once the algebra is written down.

---

## 6c. P2/P3 findings

### The 1-D shear NS target is exact, and it looked as though it wasn't

`src/shear_tensor.jl`'s `shear_NS_target_contravariant` builds `σφ = u^r/r - θ/3`,
`ση = u^τ/τ - θ/3` with no acceleration term `u^i Du^j`. That is not an approximation: the 1-D
solver evolves only `π^φ_φ` and `π^η_η` and reconstructs `π^r_r` by tracelessness, and for those two
components `u^φ = u^η = 0`, so the acceleration term vanishes identically.

In 2-D the stored components are LAB-frame `π^{xx}, π^{xy}, π^{yy}` with `u^i ≠ 0`, so the term is
genuinely present and is carried. Gate G2b measures the reduction: `π_NS^{yy}` against the 1-D
`π_NS^{φ}_{φ}` on the symmetry axis agrees at **exactly 0.0**.

### The projected comoving derivative

`Δ^{ij}_{αβ}Dπ^{αβ} = Dπ^{ij} - u^ic^j - u^jc^i` with `c^j = π^{jβ}Du_β`; the trace part of the
projector drops out entirely because `g_{αβ}π^{αβ} = 0`. Same structure for the charge current:
`Δ^i_ν Dν^ν = Dν^i - u^i(ν·a)`. Both corrections vanish at `u^i = 0`, which is again why the 1-D code
does not carry them for the components it evolves.

Writing that out is what surfaced **D5** (above): the 1-D code's version of this term for `ν` uses
`Dy = (∂_τu^r + u^r∂_ru^r)/u^τ`, but `Dy = u^τ∂_τy + u^r∂_ry = ∂_τu^r + (u^r/u^τ)∂_ru^r`. The
`∂_τ` piece appears to carry a spurious `1/u^τ`. **Not measured, not acted on.**

### The recovery tolerance was set below what an FD Jacobian can reach

The first realistic (elliptic, non-symmetric) run showed a **1.5-3% primfail rate** and three of five
sector configurations **aborting** near τ ≈ 4.5. Two false leads before the real cause:

1. "They must be vacuum cells" — every flagged cell read `T = T_MIN`. But that is the FALLBACK
   writing `T_MIN` after a failure, not the cause. Instrumented properly: `vacuum = 0`, so none of
   them took the vacuum branch. (The counter now separates the two, which is worth having anyway.)
2. "The line search is failing" — true, but a symptom. Representative failure:
   `PRR_LINESEARCH_FAILED` at `resnorm = 1.14e-14`. A converged state cannot be improved, so the
   backtracking search necessarily stalls at the floor; returning `false` there discarded good cells.
   Fixed by always falling through to residual acceptance, as `src/primrec.jl` does — **and that
   alone changed nothing**, because the residual test rejected too.

The actual cause: `tol_res = 1e-14` on the row-scaled max-norm sits AT the achievable floor. The
Jacobian is finite-difference, relative accuracy ~1e-10 (h² truncation + eps/h roundoff at h=1e-6),
so the residual bottoms out around 1e-14 and bounces. Set to **1e-12**.

| | before | after |
|---|---|---|
| primfail rate (5 sector configs) | 1.38 - 2.99% | **0.000%** in all five |
| configurations completing | 2 of 5 | **5 of 5** |
| steps to τ=6 (spread across configs) | 119 - 271 | 147 - 148 |
| round-trip recovery accuracy | ~1e-14 | ~1e-12 |

The cost is three orders of accuracy that nothing needs: 1e-12 is still tighter than the 1-D
production solver at the dilute edge (2.15e-12) and ten orders below any quoted number. The
round-trip gates measure it rather than assuming it.

### 6d. P4/P5: what the production IC actually needed

The bare scheme passed every controlled gate and then **failed on the first step** of the real IC.
Three distinct causes, each found by measurement rather than guessed:

1. **No floors.** The production IC tapers T to `T_MIN = 1e-20`, so the outer grid starts below the
   energy floor. Ported `enforce_floors_2d!` + `enforce_S_energy_constraint_2d!` from `src/floors.jl`
   (the S-bound rescales the VECTOR `(S^x,S^y)`, preserving direction — scaling components
   separately would manufacture flow along an axis). After this the run completed with L2 = 2.0e-4.
2. **A vacuum cut is needed, and the `1e6` multiplier does not transfer.** Cells in the box CORNERS
   (`r > rmax`, a region the 1-D radial grid does not have at all) sit at T ≈ 16-40 MeV, where
   `e(T)` falls five orders per 10 MeV and the lattice-HRG fit is a wild extrapolation. Added a
   cut stated as a TEMPERATURE. First attempt applied the 1-D's near-vacuum `1e6×` factor to it,
   giving a blanking threshold of 0.74 that wiped out most of the fireball — caught immediately as
   **L2 = 0.32**. The factor is calibrated against the hard floor, not a physical energy.
3. **Failures poisoned the warm-start cache.** On failure the code stored the fallback state
   (`T_MIN`) into `work.x0_*`. `log(T_MIN)` is finite, so the next step accepted it as a guess and
   skipped `seed_from_conserved_2d` — starting the Newton 37 e-folds below the answer. A single
   failure was self-sustaining. Storing `NaN` on failure cut residual primfails **5.5×** with the
   answer unchanged to four digits.

**The cut is demonstrably inert.** Measured at N=200 against the 1-D run: `T_vac_cut` = 0.02 / 0.05 /
0.09 gives L2 = 4.670e-4 / 4.640e-4 / 4.702e-4 — 1.3% relative across a 4.5× change — while residual
primfails fall 3760 → 564 → 24. Operating point 0.05, three times below freeze-out.

### 6e. The transport convention was inverted — and the charge sector is still open

**Fixed.** `main.jl` defaults to `advect_* = false, relax_advect_* = true`
(main.jl:540-564): the dissipative dofs are transported by UPWINDING INSIDE the relaxation substep,
not as passive scalars through the finite-volume flux. All three of my sectors had it backwards, and
the relaxation transport was never implemented. Correcting it improved charge conservation on the
production IC by **128×** (dQ 1.14e-1 → 8.9e-4, and 3.6e-7 with all sectors on).

**D6 — CLOSED.** With diffusion on the production IC the charge sector produced ~5e4 recovery
failures and an x↔y asymmetry of **1.0**, while shear and bulk were clean on the same IC. The 1-D
solver on that IC with diffusion is clean (charge conserved to 5e-16), so it was a 2-D defect.

**Mechanism.** The failures sat at the fluid-vacuum interface. There the charge row
`n u^τ + ν^τ = D` is nearly DEGENERATE: `n` depends exponentially on `φ`, so a tiny error in `D`
swings `φ` wildly, `α = μ/T` with it, and `ν_NS = -κ ∇α` feeds straight back into that same row.
Measured in the tail: **|∇α| = 119 per fm** and **|ν|/n = 418**. Shear and bulk are immune because
their targets are gradients of `u` and `T`, which stay smooth there.

**Fix: the density-gated vacuum ramp**, the same device `main2IS2.jl:_vacuum_weight` applies
(`FIVO_VACUUM_N_LO = 1e-6`, `_N_HI = 2e-3`, Pb+Pb production values). Below `n_lo` the current is
held at zero; between lo and hi its NS drive ramps in linearly.

| production IC, N=150 | before | after |
|---|---|---|
| diffusion only: primfails / x↔y | 44548 / **1.00** | 4464 / **1.3e-13** |
| all sectors: primfails / x↔y | 18910 / **1.00** | 6304 / **3.8e-11** |
| G3 closed-domain charge drift | 9.1e-16 | 3.6e-15 |
| G3 outflow charge drift | 1.3e-8 | **1.9e-9** |

**Four hypotheses were tested before this one; two were real bugs, two were not:**
1. inverted transport convention — a real bug (§ above), fixed, improved dQ 128×, not D6;
2. raw central-difference α gradient — replaced with the 1-D's limited face-reconstructed one,
   correct to match production, no effect on D6;
3. causality clip `|ν| ≤ n u^τ` — no effect; reverted to the production default (off);
4. **restricting the domain to the disc `r ≤ rmax`** — this made it *worse* (44.5k → 163k failures),
   and that is what cracked it: a hard vacuum wall is a sharp interface, so the problem could not be
   "the corners" (which the disc removes) and had to be the INTERFACE. `r_domain` remains available
   as a model option, defaulting to `Inf`.

**D7 — CLOSED.** With all sectors at N=300 the shear field grew to `max|π^xy| = 1.0e+03` by τ=4
and `max|u|` reached 4.3 (v = 0.97), while shear ALONE was clean to τ=8. Two independent causes,
both found by measurement:

**(a) A charge-row failure discarded the converged hydro block.** The four residual rows are not
equally robust: rows 2-4 (momentum, momentum, energy) are well conditioned everywhere, row 1
(charge) is nearly degenerate in the dilute tail. Solving them together let the fragile row veto the
three robust ones. Instrumented per-row at N=300:

| failing cells, τ=1.33 | count | share |
|---|---|---|
| **charge row only** (hydro converged) | 2458 | **68.2%** |
| hydro rows only | 0 | 0.0% |
| both | 0 | 0.0% |

Every one of those cells was then reset to vacuum, punching holes into a perfectly good `T` and `u`
field — which is what inflated `max|u|` (1.16 → 3.02 on adding diffusion to shear) and fed the shear
blow-up. Fixed with a **staged fallback** in `primrec2d.jl`: if the coupled solve fails, retry
(yT, uˣ, uʸ) from rows 2-4 with φ frozen, then close row 1 for φ alone by bisection (n is monotone
in φ, so it cannot diverge). Exact for `LatticeHRGEOS`, where `∂P/∂μ = 0` means rows 2-4 do not
contain φ at all; approximate for a general EOS, which is why it is a fallback and not the primary
path.

**(b) The bulk pressure drove the total pressure negative.** With bulk+diffusion, `|Π|/P` reached
**3.50** and `min(P+Π)` went to **-2.7e-4**, making the effective enthalpy and sound speed
meaningless. Added a positivity guard `Π ≥ -0.99 P` — not a tuning knob.

**(c) One more, exposed by (a)'s fix.** The staged fallback initially returned `ok = true` even when
row 1 was *unclosable* (`J^τ - ν^τ ≤ 0`, i.e. the diffusion current exceeds the charge). That kept a
hydro state whose charge contradicted its own conserved variable: total charge drift **1.85e+05** at
N=150, τ=8. Fixed with `enforce_nu_charge_constraint_2d!`, the charge-sector analogue of the S–E
bound: `|ν_⊥| ≤ χ J^τ` guarantees solvability without needing the velocity, since
`|ν^τ| = |u·ν|/u^τ < |ν_⊥|`.

| production IC, all sectors | before | after |
|---|---|---|
| N=300, τ=4: `max|π^xy|` | **1.0e+03** | **3.8e-2** |
| N=300, τ=4: `max|u|` | 4.34 | **1.20** |
| N=300, τ=4: x↔y | 9.4e-13 | **6.6e-13** |
| N=300, τ=8: x↔y | 2.2e-2 | **1.8e-13** |
| N=150, τ=8: primfails | 12740 | **40** |
| N=150, τ=8: charge drift | 1.85e+05 | **2.7e-5** |

**New gate G5** (`test_production_allsectors2d.jl`) runs the production IC with all four sectors to
late time at two resolutions, asserting on exactly the quantities D6 and D7 moved. Nothing in the
previous ladder covered this: every other gate runs either a controlled problem or a single sector,
and both defects needed the real IC, sector interaction AND late time together.

⚠ **Observation, not a defect**: G5 measures `max|π|/P = 0.95`. A shear correction comparable to the
pressure is the edge of applicability for viscous hydro. It is at the dilute edge rather than the
core, but it is worth knowing before quoting anything from the outer region.

### Charge conservation is exact; the outflow number is physics

The `D̃` source is identically zero and the flux divergence telescopes, so any drift in total
`τJ^τ` must be boundary flux. Measured: **9.1e-16 on a closed (periodic) domain**, 1.3e-8 with
outflow boundaries. The latter is real charge leaving the box — the charm background `n(T,α₀)` fills
the whole domain and is not compactly supported — not a scheme error. The gate asserts on the closed
domain and reports both.

---

## 6a. What the first two gates cost — findings

### The shear closure reduces exactly to the 1-D parametrisation

`src/shear_tensor.jl` does NOT store `π^r_r`; despite the docstring, `piR_phys` is the
LOCAL-REST-FRAME diagonal entry, boosted:
`Π^{ττ} = (u^r)² piR`, `Π^{τr} = u^τu^r piR`, `Π^{rr} = (u^τ)² piR`. The 2-D closure written
here reproduces that limit to 2e-16 without being told about it (`test_shear2d_algebra.jl`,
"reduces to the 1-D production parametrisation"), and independently reconstructs `piEta` from
`Πtt - pixx - piyy`. That anchors the 2-D shear sector to the trusted solver rather than to a
fresh derivation only.

### Primitive recovery: two real defects found and fixed

**(1) Row scaling.** The four residual rows differ by ~7 orders of magnitude: with the charm EOS
the charge row is `O(n) ~ 1e-4` while the energy row is `O(e) ~ 1e3`. Against a single global
norm the charge equation is effectively unconstrained — μ carried errors up to **1.4e-8** where
the solve "converged". Residual and Jacobian rows are now equilibrated by per-equation scales.
Measured effect: μ error **1.4e-8 -> 1.8e-13**.

**D3 — does the 1-D production solver share this weakness? MEASURED: mechanism real, magnitude
irrelevant. No published number is affected and no production file was changed.**

Measured on the locus the production IC actually occupies (`data/initial_profiles_physical.csv`:
T ∈ [0.128, 0.563], φ ∈ [-9.84, -4.23]), worst relative charge error over 400 states per point,
guesses perturbed 1% so the seed is not the answer:

| T | φ | E | n | **1-D** dn/n | **2-D** dn/n |
|---|---|---|---|---|---|
| 0.130 | -9.94 | 5.4e-01 | 2.3e-04 | **3.3e-12** | 1.5e-14 |
| 0.200 | -9.10 | 1.1e+01 | 1.1e-03 | 4.7e-13 | 1.4e-14 |
| 0.300 | -6.60 | 7.8e+01 | 2.7e-02 | 3.5e-14 | 1.6e-14 |
| 0.563 | -4.26 | 1.2e+03 | 9.4e-01 | 7.9e-15 | 1.1e-14 |

The 1-D error grows toward the dilute edge exactly as the loose bound predicts — **220× worse than
a row-equilibrated solve at T = 0.13** — while the 2-D version is flat. But 3.3e-12 is ~10 orders of
magnitude below any quoted number. Zero hard failures in all three regimes tested (production seed,
10% stressed seed, cold start).

**Why the 1-D solver is safe, and it is partly luck:** its loop test is `nrm < tol` with
`tol = 1e-15` **absolute**, which is unreachable once `E ~ 1e3`, so the iteration runs on until the
line search stagnates — i.e. to full floating-point convergence. The loose acceptance gate only
defines what it would *accept*, not what it produces. (My first measurement of this was circular —
it seeded the Newton with the exact solution, giving `iters = 0` and proving nothing. The numbers
above use perturbed seeds.)

Recorded permanently as `test/test_primrec2d_vs_1d.jl`, which also checks that the 2-D forward map
reproduces the 1-D conserved variables — measured **8.4e-16**, so the two solvers provably invert
the same problem and the accuracy comparison is meaningful.

**(2) Global convergence.** From a fixed seed (`u = 0`, `T = 0.15`) the Newton STALLED on
**0.95%** of physical-band states — every one of them fast-flow, `|u| ~ 1.5`, failing at
`iters = 0`. The Jacobian at those solutions is well conditioned, so the failure was global, not
local. Fixed with (a) a seed built from the conserved variables by the ideal closed form
`v = M/(E+P)`, `e = (E+P)(1-v²) - P` with `e(T)` inverted by bisection, and (b) a backtracking
line search requiring an actual decrease in the scaled residual. Measured: **0.95% -> 0.00%** over
6000 states, and 0/9000 across the full test matrix.

At 6.4e5 cells a 1% primfail rate is ~6000 bad cells per RHS evaluation, so this had to be closed
before any grid work — it would have been indistinguishable from a physics bug later.

### 6b. The EOS carries charm as a non-back-reacting tracer

`LatticeHRGEOS` returns `e` and `P` **exactly independent of μ** (12.947010000 / 3.2671369339 at
both φ = -5 and φ = 0, T = 0.30). Only `n` responds. Two consequences:
- the `(yT, u^x, u^y)` block DECOUPLES from φ — the Jacobian's φ column is non-zero only in the
  charge row, which is why the block-structure survives pivoting;
- states with `φ ≳ 0` are outside what the EOS models (at φ = 4 the charm rest-mass energy is
  **127×** the total energy density). Production sits at `φ ≈ -4.25` (from
  `data/initial_profiles_physical.csv`, `alpha0 ≈ -1.569`). Gates should sample `φ ∈ [-6, 0]`.

### The production EOS carries charm as a tracer with `dP/dμ = 0` — exactly

Beyond `e` and `P` being μ-independent (§6a), the sharper statement is that
**`LatticeHRGEOS` has `∂P/∂μ = 0` identically while `n ≠ 0`**. Measured over T ∈ {0.15, 0.30, 0.50},
μ ∈ {-1, 0}: `dP/dμ = 0.0000e+00` in every case, against `n` up to 2.6. The Euler relation
`e = T ∂P/∂T + μ ∂P/∂μ - P` holds to **2e-10**, but only because the μ term is zero.

Consequence: `eos_entropy(T,μ,n,e,P) = (e+P-μn)/T` (src/eos.jl:306) subtracts a term the pressure
never contained. **It is not the thermodynamic entropy of this EOS**; the consistent one is
`s = (e+P)/T = ∂P/∂T`. Using the former as a Bjorken invariant puts a dt-INDEPENDENT floor of
**3.0e-4** on gate G0 — which is exactly how the convergence test caught it (order 0.00 instead of
2.00). The G0 reference now uses `s = (e+P)/T`; see the derivation in `test/test_bjorken2d.jl`.

⚠ Note for elsewhere, **not measured and not acted on**: `eos_entropy` feeds `local_thermo`, and
hence `viscosity(T, th, sh) = (η/s)·s`. The spurious `μn/T` term is ~1.4e-4 relative at the
production operating point — far under the η/s uncertainty, so this is a remark, not a defect.
Tracked as **D4**.

## 6f. What is and is not validated — read before quoting anything

The ladder is green, and that is worth exactly as much as the gates cover. Three limits:

**1. The dissipative sectors agree with the 1-D IN THE CORE; the edge is a different story.**
**P6 CLOSED** by gate G7 (`test_dissipative_vs_1d.jl`): both solvers run the production IC with
shear + bulk + diffusion at the Pb+Pb operating point, passed explicitly on both sides. Measured at
N=200, relative L2 of (2-D − 1-D) per radial shell, scaled by the 1-D rms:

| field | global | r<2 | 2-4 | 4-6 | 6-8 |
|---|---|---|---|---|---|
| T | 1.37e-03 | 3.6e-04 | 6.5e-04 | 1.5e-03 | 2.4e-03 |
| α | 1.18e-01 | **5.1e-03** | 7.8e-03 | 5.1e-02 | 2.5e-01 |
| u^r | 2.80e-02 | 5.3e-03 | 6.1e-03 | 1.3e-02 | 2.7e-02 |
| Π | 1.13e-01 | **4.0e-03** | 2.3e-02 | 9.5e-02 | 1.2e-01 |
| π^{rr} | 1.25e-01 | **3.8e-03** | 4.1e-02 | 2.7e-01 | 4.1e-01 |
| π^η_η | 2.93e-02 | **4.8e-03** | 9.2e-03 | 5.7e-02 | 1.3e-01 |
| ν^r | 4.87e-02 | 1.4e-02 | 1.5e-02 | 3.3e-02 | 1.0e-01 |

*(local T in those shells: 0.327, 0.308, 0.259, 0.191; freeze-out is 0.1565)*

**The global L2 is misleading** — it is dominated by the outer shells, where the fields are small,
the profile is steepest, and the 2-D carries guards the 1-D does not have AT ALL: the density-gated
vacuum ramp on the charge sector, the bulk positivity guard, and `E_vac_cut`. Those were not
optional; without them the 2-D solver does not run on this IC (§6d-§6e). A difference there is
therefore expected, and a tight global bound would either fail or force the guards out.

**T agrees to 2.4e-03 even in the 6-8 shell**, so the bulk is sound everywhere — it is specifically
the dissipative fields that degrade outward.

⚠ **CONSEQUENCE FOR USE: do not quote the 2-D dissipative fields from the outer region
(r ≳ 6 fm, T ≲ 0.19 GeV) without further work.** G7 guards the core at ~2× the measured values and
guards T globally.

**2. MOOD is written but not wired.** `mark_bad_2d!` and `expand_bad_2d!` exist in `floors2d.jl` and
are called from nowhere. The solver has floors, the S–E bound, the ν bound, the vacuum cut and the
vacuum ramp — but no first-order fallback for cells whose reconstruction is bad. The 1-D has one.
Tracked as **P7**.

**3. Azimuthal structure is exercised but cannot be verified against anything.** No reference
solution for a deformed IC exists in this repo — that is why the 2-D solver is being built. G6 checks
the four things that ARE decidable: the ε₂=0 control gives identically zero anisotropy, the
anisotropy converges with resolution, reflection in x and y are exact, and conservation holds. It
cannot check correctness.

⚠ **Open observation (P8).** On a DEFORMED IC at N=300, reflection symmetry degrades to ~1.8e-3,
while it is at round-off for the same deformation at N=200 and for ε₂=0 at N=300:

| | refl_x | refl_y |
|---|---|---|
| ε₂=0, N=300 | 3.1e-13 | 3.5e-13 |
| ε₂=0.10, N=300 | 1.2e-03 | 1.2e-03 |
| ε₂=0.25, N=300 | 1.8e-03 | 3.4e-13 |
| ε₂=0.25, N=200 | 1.5e-10 | 1.8e-10 |

So it needs deformation AND high resolution together. The bulk observable is unaffected — the
anisotropy converges to 0.08% across N=100→300 — so this is likely a small number of cells, but it
is a max-norm over cells and has not been localised. G6 therefore gates at N ≤ 200.

## 6g. F1 — the un-averaged IC, and the rest of the queue

**F1 CLOSED.** `Julia/Projects/ALICE_IC_Creation/MCGCollisionDensity2D.jl` +
`BuildIC2D.jl` keep the binary-collision midpoint VECTORS that
`MCGCollisionDensity.jl` reduces to radii before depositing `azimuthal_gauss`. That
one φ-average was where the azimuthal information died — upstream of every solver.

| Pb+Pb 0-5%, 8000 min-bias, 400 selected | value |
|---|---|
| `∫ n_coll dx dy` vs `⟨N_coll⟩` | **-0.000%** (self-normalising, as in 1-D) |
| N_charm (pairs) vs the 1-D build | **24.09 vs 24.04 — ratio 1.0023** |
| T_max vs the 1-D central temperature | **0.5814 vs 0.5814** |
| ε₂ (participant-plane aligned) | **0.1054** |
| ε₂ (`align_pp = false` control) | **0.0165** ≈ 0 |

The control matters: averaging a centrality class WITHOUT rotating each event into
its own participant plane washes the deformation straight back out. That it returns
ε₂ ≈ 0 and reproduces the 1-D radial profile is what shows the deformation is real
and not an artefact of the 2-D deposit.

⚠ **The normalisation is INHERITED, not re-derived.** `TemperatureCalibration.jl`
fixes T(r,τ₀) by MCG → Fluidum → Cooper-Frye tuned to ALICE N_π; redoing that in
2-D needs a 2-D freeze-out surface, which does not exist (**P9**). `BuildIC2D.jl`
instead uses the calibrated radial pair as a density → T map applied pointwise. It
reproduces the 1-D charm count to 0.2% and the central temperature exactly, but
**the pion yield of a 2-D run built this way has not been checked against ALICE.**

🔑 A factor-2 bug was caught here by the N_charm cross-check: the first version
hardcoded σ_NN and dσ/dy in **mb** where the pipeline works in **fm²**, and dropped
the 0.8 shadowing. Both are now READ from `metadata_DensityHard.txt`, so they
cannot drift from the 1-D build.

### First 2-D physics run (gate G8)

| | ε₂(IC) | momentum anisotropy | response |
|---|---|---|---|
| τ=2 | 0.0903 | +0.0260 | 0.288 |
| τ=4 | 0.0903 | +0.0345 | 0.383 |
| τ=8 | 0.0903 | +0.0559 | 0.619 |
| τ=8, N=250 | 0.0903 | +0.0552 | 0.612 |

Spatial eccentricity converting to momentum anisotropy, building with time — the
defining hydrodynamic response, resolution-stable to 1.2%.

### The rest of the queue

- **P7 CLOSED**: MOOD wired (`force_first_order` threaded through `rhs_2d!` →
  `_faces_2d!`; `_stage_2d!` escalates mark → expand → retry before halving dt).
  Cut primfails on the elliptic IC 200 → 108 at N=150.
- **P8 CLOSED**: the deformed-IC reflection residual is **entirely below
  freeze-out in the box corners**. At N=300, ε₂=0.25: globally 1.8e-3, but
  **above T_fo it is 5.3e-15**. Every contributing cell is at r ∈ [18.6, 24.7],
  T ∈ [0.067, 0.116]. G6 now asserts on the observable-producing region and
  reports both.
- **D5 CLOSED, negligible.** Measured on a real 1-D run: the error from the
  `Dy` form is **5.9e-5** relative to the ν-update denominator, and the whole
  `τ_n v Dy` term is only **4.8e-4** — both swamped by the backward-Euler weight
  `A ≈ 80`. Real algebra, no production change warranted.
- **D4 CLOSED, negligible where used.** The `eos_entropy` `-μn/T` term costs
  **3.5e-3** on η along the production locus (my earlier estimate of 1.4e-4 was
  25× too small — measured, not guessed). It reaches 17% only at high T AND high
  charm fugacity, which production does not visit. η/s uncertainty is O(50%).

## 6h. The un-averaged IC, second pass — and a measured trade-off

**T and the charm density now carry DIFFERENT geometries.** The first version of
`BuildIC2D.jl` derived T from the collision density, which slaved the two profiles
together. But hard processes scale with BINARY COLLISIONS and the soft entropy that
sets T scales with PARTICIPANTS, and the 1-D pipeline already keeps them distinct
(`nhard_profile.txt` from N_coll, `T_profile.txt` from MCG's entropy background).
`mcg_fields_grid` now returns both fields from the same events with the same
per-event Ψ₂ rotation:

| Pb+Pb 0-5%, 8000 min-bias | value |
|---|---|
| ε₂, binary-collision field | **0.1054** |
| ε₂, entropy/participant field | **0.0784** |
| N_charm vs the 1-D build | 24.09 vs 24.04 (**ratio 1.0023**) |
| T_max vs the 1-D central T | 0.5814 vs 0.5814 |

α then follows from BOTH: writing `n = A(T)·exp(φ)`, the 1-D triple
(T, alpha, n_hard) tabulates `A(T)` — verified, `n_EOS(T, α·T)` matches
`nhard_profile.txt` to ~1% through the fireball — and
`φ(x,y) = ln(n_hard(x,y)/A(T(x,y)))`.

🔑 **Two bugs caught by cross-checks, neither visible in the fields themselves:**
1. σ_NN and dσ/dy hardcoded in **mb** where the pipeline works in **fm²**, plus the
   0.8 shadowing dropped: N_charm 7.8× wrong. Now read from
   `metadata_DensityHard.txt`, with a **hard assert** at 5%.
2. `azimuthal_average` returned **0.0** for empty bins. Averaging onto the 1-D
   radial grid (dr=0.1) from a 2-D grid with dx=0.25 leaves most bins empty, so the
   density → T map was built from a comb of zeros. Empty bins are now NaN and
   interpolated, and the average is taken on a radius grid matched to the 2-D
   spacing.

### The charge-loss / stability trade-off (gate G8)

Charge drift on the un-averaged IC grows with resolution and time: **4.7e-6** at
N=150 but **3.3e-3** at N=250, τ=8. The mechanism is the cold edge — the vacuum cut
and the density-gated ramp both delete charge there. Loosening them fixes it, and
breaks the charge sector:

| N=250, τ=8 | dQ | x↔y (elliptic IC) |
|---|---|---|
| `T_vac_cut`=0.05, `n_lo`=1e-6 (default) | 3.3e-3 | **4.5e-13** |
| 0.02, 1e-9 | **2.1e-6** | **1.00** ← D6 returns |

So the loss is the PRICE of a stable charge sector, not a defect to be tuned away.
G8 asserts tightly at N=150 (1e-4) and loosely at N=250 (2e-2) with this recorded
next to the assertion, so nobody "fixes" it by loosening the thresholds.

### P10 CLOSED — the fluid-vacuum face was a REFLECTING WALL

The T slices showed spikes at r ≈ ±10 fm at late times, above freeze-out. Cause, from
dumping the cells across the interface at τ=8:

| x [fm] | T | E |
|---|---|---|
| 9.20 | 0.1760 | 1.43e+00 |
| **9.36** | **0.2671** | **1.84e+01** |
| 9.52 | 0.0000 | 1.00e-20 (vacuum) |

**20× the neighbour's energy, in one cell.** In `_faces_2d!`, a face whose neighbour is
vacuum failed `prim_to_cons_2d!` (`eos_Pne` underflows to `e = 0` at `T_MIN`) and the code
did `continue`, **leaving the face flux at zero**. A zero flux at a fluid/vacuum face is a
perfectly reflecting wall: the fireball pours energy into its outermost cell and none can
leave. It had been there since P1c and affected every run.

Fix: give the vacuum side an explicit FLOOR state (`_floor_prim_2d!`) so HLLE runs
normally — it then supplies its own upwinding and dissipation, and the vacuum side
contributes nothing. Afterwards the profile is monotone straight through:

| x [fm] | T | E |
|---|---|---|
| 9.20 | 0.1588 | 8.71e-01 |
| 9.36 | 0.1576 | 8.31e-01 |
| 9.52 | 0.1564 | 7.92e-01 |

🔑 The first attempt — substituting the fluid side's own flux, restricted to outflow —
made it far WORSE (E reached 4e17). A one-sided flux carries no dissipation, and across a
13-order jump that is unconditionally unstable. The lesson is to let the Riemann solver
see a valid state on both sides rather than hand-rolling the flux.

**The wall was doing far more damage than the spikes, and it SUBSUMES P8.** It had been
present since P1c, in every 2-D run:

| | with the wall | fixed |
|---|---|---|
| E in the outermost fluid cell, τ=8 | 18.4 | **0.83** |
| T half-width at τ=8 [fm] | 9.52 | **12.56** |
| charge drift, N=250, τ=8 (G8) | 3.3e-03 | **6.8e-06** |
| reflection symmetry, deformed IC, N=300 (G6) | 1.8e-03 | **2.7e-13** |
| primfails, N=300 elliptic (G6) | 1432 | **436** |

So the fireball was being **artificially confined** — late-time expansion and the
freeze-out surface were both wrong — and two effects I had written off as unavoidable
were this one bug:
- the **P8** reflection residual, which I had characterised as a benign sub-freeze-out
  corner effect. It was the wall. **P8 is therefore closed by this fix, not by the
  earlier reasoning**, and G6's global reflection figure is now 2.7e-13.
- most of the **edge charge loss** I had presented as the price of the vacuum ramp
  (§6h). The trade-off is real but far smaller than measured: 6.8e-6, not 3.3e-3.

⚠ Also corrected: I flagged `max|π|/P = 0.95` as "the edge of applicability". Measured
ABOVE FREEZE-OUT, where observables come from, it is **0.202** (N=150, τ=8) and **0.340**
(N=300, τ=4) — a healthy 20-34%. The 0.95 was entirely dilute tail. G5 now asserts on the
freeze-out-restricted value and reports the global one, as G6 does for reflection.

---

## 7. Status

| id | item | state |
|---|---|---|
| P0 | derivation: geometry, conservative form, shear closure, charge closure | **done** — §1-3 |
| P1a | `grid2d.jl`, `state_layout2d.jl`, `shear2d.jl` | **done**, gated: `test_shear2d_algebra.jl` 2010/2010 at round-off |
| P1b | `primitives2d.jl`, `primrec2d.jl` (4-unknown Newton + seed + line search) | **done**, gated: `test_primrec2d.jl` 33/33, 0/9000 recovery failures; `test_primrec2d_vs_1d.jl` 49/49 |
| P1c | `fluxes2d.jl`, `reconstruction2d.jl`, `bc2d.jl`, `work2d.jl`, `rhs2d.jl`, `timestepper2d.jl`, `main2D.jl` | **done**, gated: `test_bjorken2d.jl` 9/9 |
| P2 | `dissipation2d.jl` — shear + bulk NS targets, IS relaxation, projected comoving derivative | **done**, gated: `test_dissipation2d.jl` |
| P3 | `transport2d.jl` + charge block — `(α, ν^x, ν^y)`, closes **D2** | **done**, gated: `test_charge2d.jl` |
| **G0** | Bjorken: `τJ^τ` drift **0.0**, `u^x=u^y` **0.0**, transverse spread **0.0**, n err 4.4e-16, SSPRK2 order **2.00**, SSPRK3 order **2.99** | **PASS** |
| **G2** | shear/bulk: NS traceless 2.4e-15; `π_NS^{yy}` vs 1-D **0.0 exactly**; viscous Bjorken vs independent ODE e 9.7e-5, π 2.1e-3; viscous 13.8% hotter than ideal with `π^η_η<0`; splitting order 1.05 | **PASS** |
| **G4** | **reproduction**: 2-D from the production IC vs the 1-D production run — T agrees **L2 6.4e-4 / Linf 2.7e-3** at dx=0.2, converging at order **0.88** in dx; azimuthal spread 2.4e-3 and falling; x↔y **3.5e-14** | **PASS** |
| **P10** | fluid-vacuum face flux left at ZERO = a reflecting wall | **CLOSED**, and it subsumes P8 |
| **G5** | **production IC, all four sectors, late time**: N=150/τ=8 primfail **40**, x↔y **5.0e-12**; N=300/τ=4 x↔y **6.6e-13**, `max\|u\|` 1.20, min(P+Π) > 0, dQ 9.3e-6 | **PASS** |
| **G7** | **dissipative sectors vs the 1-D on the production IC**, all sectors both sides: core (r<4) α 7.8e-3, Π 2.3e-2, π^{rr} 4.1e-2, π^η_η 9.2e-3, ν^r 1.5e-2; T **1.4e-3 globally** | **PASS** |
| **G8** | **un-averaged production IC** (F1's product): ε₂(IC) 0.090 → momentum anisotropy **+0.056** at τ=8, building with time, resolution-stable to 1.2% | **PASS** |
| **G6** | **non-axisymmetric IC** (production profile on an elliptical radius): ε₂=0 control anisotropy **identically 0**, anisotropy **converges to 0.08%** over N=100→300, reflection symmetry 1.5e-10 at N≤200 | **PASS** |
| **INT** | integration, all 11 fields on an ELLIPTIC (ε₂≠0) IC — the case 1-D cannot represent: runs to τ=6, primfail **0**, `π^xy` ≠ 0, momentum anisotropy **+0.186** | **PASS** |
| **G3** | charge: τ_n copies **bit-identical**; `ν_NS` vs 1-D 6.2e-16; **closed-domain charge drift 9.1e-16**; 0/2104 cells up-gradient; x↔y asymmetry **0.0 exactly** | **PASS** |
| P2 | shear + bulk sector, NS targets, IS relaxation, constraint projection | not started |
| P3 | charge sector `(α, ν^x, ν^y)`, incl. the D2 covariant-drive derivation | not started |
| P4 | G4 reproduction gate on the production IC | **done** — `test_reproduction2d.jl`; runs BOTH solvers in-process from the 1-D's own `load_initial_interpolants`, so an IC mismatch is impossible |
| P5 | robustness: floors, S–E bound, ν admissibility bound, vacuum cut, vacuum ramp, staged recovery, bulk positivity | **done** — all four sectors run the production IC to τ=8 at N=150 and N=300 with x↔y ≤ 5e-12; gated by G5 | — full 11-field elliptic IC runs clean: `ok`, **primfail 0**, shear constraint 0.0, 148 steps / 3.8 s on 96². No MOOD, floors or vacuum ramps yet; those are the remaining P5 work |
| F1 | un-averaged 2-D IC | **CLOSED** — `MCGCollisionDensity2D.jl` + `BuildIC2D.jl`, ε₂=0.105, N_charm within 0.2% of the 1-D; see §6g |
| D2 | derive the 2-D covariant NS drive `∇^⟨i⟩α`; gate it | **CLOSED** — `ν_NS^i = -κ(∂_iα + u^i Dα)`; setting `u^y=0` gives `-κ[(u^τ)²∂_rα + u^ru^τ∂_τα]`, the production expression exactly (6.2e-16, `test_charge2d.jl` G3b) |
| D5 | 1-D `Dy` carries a spurious `1/u^τ` on its `∂_τ` piece | **CLOSED, negligible** — 5.9e-5 relative to the ν-update denominator; production untouched; see §6g |
| **D6** | charge sector unstable on the production IC — degenerate charge row at the fluid-vacuum interface | **CLOSED** by the density-gated vacuum ramp (x↔y 1.00 → 1.3e-13); see §6e |
| **D7** | with ALL sectors at N=300, `\|π^xy\|` grew to 1e3 by τ=4 | **CLOSED** — a charge-row failure was discarding the converged hydro block (68.2% of failures), plus a negative total pressure; see §6e |
| P6 | compare the DISSIPATIVE sectors against the 1-D solver on the production IC | **CLOSED** by gate G7 — core agreement sub-percent to a few percent, edge degrades to tens of percent where the 2-D carries guards the 1-D lacks; see §6f |
| P7 | wire MOOD | **CLOSED** — see §6g |
| P8 | reflection symmetry on a deformed IC at N=300 | **CLOSED** — entirely below freeze-out in the box corners; above T_fo it is 5.3e-15; see §6g |
| P10 | T spikes at the fluid-vacuum interface | **CLOSED** — the face flux was left at ZERO there, i.e. a reflecting wall; 20x energy pile-up in one cell. Fixed with a floor state so HLLE runs. See §6h |
| P9 | re-derive the ALICE N_π calibration in 2-D (needs a 2-D freeze-out surface) — until then `BuildIC2D.jl`'s normalisation is inherited | **open** |
| D4 | `eos_entropy`'s `-μn/T` term is not thermodynamic for `LatticeHRGEOS` | **CLOSED, negligible** — 3.5e-3 on η at the production locus (not 1.4e-4 as estimated); see §6g |
| D3 | measure whether the 1-D production primrec shares the row-scaling weakness | **CLOSED, negative** — see §6a. Mechanism real (220× at the dilute edge), magnitude 3.3e-12, no manuscript affected, production untouched |

### Cost — MEASURED (`bench2d.jl`), and my early estimate was too pessimistic

At the top of this document I guessed that 2-D would take runtime "from minutes to hours".
Measured on 24 threads, production IC, all four sectors:

| N | cells | steps (τ 0.4→4) | wall | ns / cell-step |
|---|---|---|---|---|
| 100 | 10 000 | 63 | 3.0 s | 4740 |
| 150 | 22 500 | 83 | 3.4 s | 1792 |
| 200 | 40 000 | 105 | 7.0 s | 1668 |
| 300 | 90 000 | 152 | 20.7 s | 1513 |

Per-cell cost *improves* 3.1× from N=100 to N=300 — thread utilisation, not algorithmics;
the small grids simply cannot fill 24 threads. Total cost still scales ~N³ (cells ×N², steps
×N through the CFL).

Projection: **N=400 (dx=0.10 fm) to τ=13 is ~3 minutes; N=800 (dx=0.05 fm) ~26 minutes.**
Matching the 1-D grid spacing exactly (dr=0.025 ⇒ N=1600) would be ~3.5 hours. So the
original "hours" estimate was right only for the very finest grid, and wrong by ~2 orders
of magnitude for the resolutions actually used.

⚠ **The per-sector cost split is NOT resolved.** Single-rep wall times were
6.22 / 7.75 / 7.41 / 6.55 / 6.91 s for ideal / +shear / +bulk / +diffusion / all-three —
and "all three" (+10.0% over ideal) came in *below* "shear alone" (+22.2%), which is
impossible. The spread is run-to-run noise. Attributing cost per sector needs repetitions
that this benchmark does not do.

---

## 6i. Centrality classes — a much more anisotropic IC, and a map that would have hidden it

`BuildIC2D.jl all` now builds three classes in one pass (0–5%, 20–30%, 30–40%). The point is
the deformation: the 0–5% entropy field carries ε₂ = 0.079, which is a barely-visible almond.

| class | ⟨N_coll⟩ | ε₂ (collision) | ε₂ (entropy) | N_charm | T_max [GeV] |
|---|---|---|---|---|---|
| 0–5%   | 1786.98 | 0.1072 | **0.0794** | 23.79 | 0.5814 |
| 20–30% |  574.52 | 0.3054 | **0.3283** |  7.65 | 0.5346 |
| 30–40% |  321.96 | 0.3655 | **0.4011** |  4.29 | 0.5109 |

### 🔴 The density → T map has to be calibrated on ONE class, and it was not

`T_of_s = monotone_map(s_avg, T_1D(rmap))` is *self-calibrating*: it matches the φ-average of
this file's own 2-D entropy field to the 1-D calibrated `T_profile.txt`. That is exactly right
for 0–5%, the class the 1-D pipeline was calibrated on — and it is a trap for every other class.
Rebuilding the map per class matches **each** class's own φ-average to the **same** 1-D profile,
so a peripheral collision is handed the central temperature by construction: T_max identical for
every centrality, the whole centrality dependence of T erased, and only the *shape* anisotropy
surviving.

The guard that was supposed to catch this (`T_max ≤ T_max(0–5%)·1.01`) would have **passed**,
because it would have been comparing the profile to itself. The same shape as the CharmTempLib
trap: *a gate that reuses the quantity under test can never fail.*

Fixed by `reference_s_avg` — the map is built once on 0–5%, cached in `ic2d_ref_smap.txt` under a
key covering seed / n_minbias / dx / xmax / align_pp / W / √s, and applied unchanged to every
other class. The monotone T_max ladder 0.5814 → 0.5346 → 0.5109 above is what says it works;
before the fix all three read 0.5814.

### The normalisation invariant is centrality-independent, and τ₀ cancels

`N_charm = (∫n_coll dx dy)·(dσ/dy)/(σ_NN·τ₀)·τ₀ = N_coll·(dσ/dy)/σ_NN`. Comparing the absolute
`N_charm` to the 0–5% value only works for 0–5% (it falls to 0.32× and 0.18× of it); the class-
independent statement is the ratio, and it holds at 0.013314 for all three to 6 digits. The first
version of the guard divided by τ₀ as well and failed by exactly 1/τ₀ = 2.5.

### The hydrodynamic response, measured

Half-width at half the central value, `y/x` (out-of-plane / in-plane), 30–40%:

| τ [fm/c] | 0.4 | 1.0 | 2.0 | 4.0 | 6.0 | 8.0 |
|---|---|---|---|---|---|---|
| T  y/x | 1.490 | 1.471 | 1.393 | 1.304 | 1.194 | **1.160** |
| n  y/x | 1.640 | 1.640 | 1.519 | 1.242 | 0.962 | **0.904** |

The almond rounds out monotonically, and the *charm* distribution crosses 1 and inverts by τ ≈ 6 —
the in-plane pressure gradient is the steeper one, so the fluid pushes further in x than in y and
drags the tracer with it. On 0–5% the same numbers barely move (T y/x 1.047 → 1.039). That is the
whole reason to run a peripheral class.

### |ν|/n crosses 1 — and every one of those cells is below freeze-out

The global maximum of |ν|/n (over cells with n > 1e-6 fm⁻³) reaches **1.25** at τ = 1 on 20–30%,
which would be outside the domain of a first-order current. It is not: measured, every one of
those maxima sits at n ≈ 2e-6 fm⁻³ and T ≈ 0.09–0.12 GeV — five decades below the peak density
and **below T_fo = 0.1565**, i.e. in the dilute tail that produces no observables.

| class | max |ν|/n, global | max |ν|/n, above T_fo |
|---|---|---|
| 0–5%   | 0.97 (τ=2) | **0.211** (τ=1) → 0.036 (τ=8) |
| 20–30% | 1.25 (τ=1) | **0.250** (τ=1) → 0.023 (τ=8) |
| 30–40% | 1.10 (τ=2) | **0.275** (τ=1) → 0.011 (τ=8) |

`plot2d_evolution.jl` now prints both, for the same reason G5 gates `max|π|/P` above freeze-out
and only reports the global figure: an applicability statement belongs where the observables are.

### The late-time central dip is physical, not a grid artefact

At τ = 8 the 30–40% freeze-out contour splits into two lobes on the x-axis: T(0) has fallen just
below an off-centre maximum. Resolution test, to τ = 8 on the same IC:

| N | dx [fm] | T(0) | max_x T | at x | dip |
|---|---|---|---|---|---|
| 150 | 0.213 | 0.15524 | 0.15739 | −4.16 | 1.37% |
| 200 | 0.160 | 0.15523 | 0.15742 | −4.24 | 1.39% |
| 300 | 0.107 | 0.15521 | 0.15744 | −4.21 | **1.41%** |

Converged in both depth and position — the late-time flattening/hollowing of a strongly flowing
fireball (max|u| ≈ 3), not the scheme. The lobes sitting at x < 0 rather than symmetrically is the
residual event-by-event asymmetry of an un-averaged IC, which is not mirror-symmetric.

### Gate status after the rebuild

`ic2d_00-05.csv` was regenerated by this pass (⟨N_coll⟩ 1809.59 → 1786.98), so the whole ladder was
re-run against it: **11/11 pass**, G8 charge drift 2.8e-6 / 1.0e-6 / 6.8e-6, anisotropy response
resolution-stable to 1.40%.

---

## 6j. Initial-state fluctuations — G9, the shear regulator, and a reproducibility defect

Two different things are called "fluctuations" here. **There is no thermal/hydrodynamic
fluctuation sector**: `grep -rn randn src/ src2d/` returns nothing, in either solver. This
section is about the other one — **initial-state** fluctuations, which the pipeline was
silently averaging away.

### The ensemble IC has no ε₃, by construction

`mcg_fields_grid` rotates each event into its own participant plane and then adds it. That
preserves ε₂ exactly — it is why the ensemble almond survives — and destroys everything else.
Measured on the ensemble ICs: **ε₃ = 0.0013 / 0.0070 / 0.0068** for 0–5% / 20–30% / 30–40%,
i.e. zero. Triangular flow is a pure fluctuation observable, so no ensemble IC can produce it,
and a 1-D radial solver cannot represent it at all.

`n_average` / `event_offset` (MCGCollisionDensity2D.jl, threaded through `CFG2D`) select a
single event instead. Six events at 20–30%:

| | ev01 | ev02 | ev03 | ev04 | ev05 | ev06 | ensemble |
|---|---|---|---|---|---|---|---|
| ε₂ (entropy) | 0.323 | 0.200 | 0.443 | 0.305 | 0.331 | 0.430 | 0.328 |
| ε₃ (entropy) | 0.114 | 0.223 | 0.074 | 0.099 | 0.135 | 0.030 | **0.007** |
| T_max / ensemble peak | 1.26 | 1.22 | 1.23 | 1.20 | 1.09 | 1.24 | 1.00 |

ε₂ fluctuates about a nonzero mean (the almond); ε₃ has no mean and is pure fluctuation.

### 🔴 The solver did not survive a lumpy IC, and said it had

A single event's hot spots are set by the sub-nucleon width W = 0.5 fm. The shear sector could
not resolve their gradients: |π|/P ran past 1 and then to 1e7–1e12, P + Π went negative,
recovery began failing in bulk, and the solve cascaded. Two things made this worse than a crash:

* **`run_sim_2d!` returned `ok = true` throughout.** `state_ok_2d` checks only that every cell
  is finite and above the floors — so `ok` was correct by its own definition and useless as a
  physics statement. Measured with the regulator off on ev02: **max|u| = 28.1 (N=200) and 32.7
  (N=300), dQ = 0.39 and 0.54**, with 729k and 1.17M recovery failures. Half the conserved
  charge left the grid and the run reported success.
* **Resolution does not fix it.** The v_n it reports differ at every resolution.

The sector split identifies it: at N=200 the **ideal** run is perfectly stable (max|u| = 1.94,
dQ = 8.7e-4, min(P+Π) = 6.11e-2 flat), and `shear+bulk, no diffusion` diverges exactly as the
full system does. It is the shear sector, not the charge sector.

`run_sim_2d!` now also returns `maxu` and `dQ` so a caller can disbelieve `ok`. ⚠ that `maxu`
is over ALL interior cells including the dilute tail, so it is larger than the above-freeze-out
figure the gates assert on: 3.46 against 1.36 on ev02 at τ = 8.

### The fix is the regulator that was already there, switched off

`pi_clip_factor` (dissipation2d.jl) caps the *evolved* π component-wise at f·|P|. It defaults
to −1.0 (off), inherited from the 1-D solver, whose smooth radial profile never needs it. At
f = 1 on ev02:

| N | dx | max\|u\| | dQ | min(P+Π) | max\|π\|/P | v₂ | v₃ |
|---|---|---|---|---|---|---|---|
| 200 | 0.160 | 1.39 | 3.1e-6 | +3.78e-2 | 0.408 | 0.22258 | 0.14651 |
| 300 | 0.107 | 1.40 | 9.8e-6 | +3.80e-2 | 0.400 | 0.22266 | 0.14473 |
| 400 | 0.080 | 1.40 | 2.4e-6 | +3.80e-2 | — | 0.22259 | 0.14402 |

**v₃ converges to 1.2% from N=200 to N=300** (1.7% to N=400).

**It does not depend on the cap value**, which is what makes it a measurement rather than a
property of the regulator — v₃ = 0.14651 / 0.14605 / 0.14605 / 0.14605 at f = 1 / 2 / 5 / 10,
and 0.14764 at f = 0.5. Any finite cap stops the runaway; the converged answer is the same.

**Inertness is the licence for using it at all**, and it is measured: on the smooth ensemble IC,
clip on vs off agrees to Δv₂ = 4.1e-6, Δv₃ = 4.2e-7, Δmax|u| = 2.5e-6 — every printed digit, at
f = 0.5 through 10. The default stays **off** (1-D parity); it is set explicitly by G9 and by
`plot2d_evolution.jl`. Whether it should become the 2-D default is a decision, not a measurement.

### ε_n → v_n: the response

v_n is the harmonic of the transverse momentum density above freeze-out,
`V_n = Σ|S| e^{inφ_p} / Σ|S|`, `S = (T^{τx}, T^{τy})`.

| IC | ε₂ | ε₃ | v₂ | v₃ | v₂/ε₂ | v₃/ε₃ |
|---|---|---|---|---|---|---|
| ensemble 20–30% | 0.336 | **0.008** | 0.387 | **0.016** | 1.150 | — |
| ev01 | 0.331 | 0.107 | 0.319 | 0.159 | 0.963 | 1.49 |
| ev02 | 0.204 | 0.227 | 0.223 | 0.147 | 1.093 | 0.65 |
| ev03 | 0.456 | 0.077 | 0.441 | 0.067 | 0.968 | 0.87 |
| ev04 | 0.301 | 0.101 | 0.239 | 0.117 | 0.796 | 1.16 |
| ev05 | 0.340 | 0.147 | 0.263 | 0.189 | 0.776 | 1.28 |
| ev06 | 0.427 | 0.019 | 0.309 | 0.108 | 0.724 | 5.73 |

The single event produces **9.5× the ensemble's v₃**, and across the six events
**r(ε₂,v₂) = +0.833, r(ε₃,v₃) = +0.546**. ⚠ Six events is a weak statistic and r(ε₃,v₃) is not
stable under a single event changing — it read +0.772 before ev02 was corrected. Treat the ε₂
correlation as solid and the ε₃ one as suggestive; a response coefficient needs tens of events.
ev06's v₃/ε₃ = 5.7 is a small-denominator artefact (ε₃ = 0.019), not a large response.

The **Ψ₂ plane alignment is the tighter check**: (Ψ₂^momentum − Ψ₂^IC) mod π, in units of π/2,
reads 0.979 / 0.906 / 0.935 / 1.130 / 0.919 / 0.867 across the events and 0.984 for the
ensemble — clustered at 1, i.e. momentum builds perpendicular to the ε₂ plane, as it must. The
n=3 equivalent scatters over 0.39–1.47, expected since Ψ₃ is itself a fluctuating quantity.

### 🔴🔴 The single-event ICs were not reproducible — one draw in 8000

`BuildIC2D.jl events 20-30 2`, run twice, selected **different collisions**: event 2044
(N_coll 734) and event 2992 (N_coll 891), with the window multiplicity bounds moving
53.59–81.81 → 51.73–77.96. Seeding `MersenneTwister(seed)` was not enough.

Localised by elimination, not by guessing — two hypotheses died first (the global RNG alone; a
tie in the multiplicity sort, refuted by 8000 distinct multiplicities and a stable `sortperm`).
The `rng` stream itself is clean: the same impact parameter and the same next `rand(rng)` in
every process. Only `rand(rng, parts.nucl1)` moved. The type says why —
`NucleiWoodSaxon3D{…Metropolis_Hastings{3, density_WS_deformed, …}}` is a **stateful MCMC
chain**, and its *first* draw takes randomness from the global RNG, not from `rng`. Every later
draw is deterministic, which is what hid it:

* within one process, repeat calls agreed exactly, so it looked reproducible;
* across processes only event #1 of 8000 differed — but that moved its rank in the multiplicity
  sort, and the 20–30% window starts at rank 1601, so the selection at position 2 flipped;
* **5 of 6 events and all 3 ensembles were unchanged**, so it read as a one-off. (My first
  reading — that a stateful chain would poison every later event — was wrong, and the rebuild
  is what said so: the ensembles came back identical to 4 decimals.)

Fixed in `mcg_fields_grid` by seeding the global RNG **and burning one draw per chain**, since
seeding alone pins the second build in a process but not the first (measured: event #2 became
byte-reproducible while event #1 still moved, 207 / 4579 / 207). Verified: three independent
processes now select 207 / 2992 / 4535 and produce **byte-identical** CSVs.

`mcg_fields_grid` now returns `sel_idx` / `sel_ncoll`, `build` prints them and writes them to
the metadata, and **G9 asserts `selected_events == [2992]`** — so a regeneration that picks a
different collision fails loudly instead of silently re-measuring a different event.

⚠ All ev02 numbers in this section are the corrected (2992) event. The ensemble numbers in §6i,
and the 43-cell clamp measurement, are unaffected — those ICs came back identical.

### Cost

Pinned to 12 of 24 cores with foreign jobs on the machine (`taskset`, load recorded); **relative**
numbers are sound, absolute ns/cell-step is not portable.

| IC | N=150 | N=200 | N=300 | N=400 |
|---|---|---|---|---|
| ensemble, ns/cell-step | 2213 | 1858 | 1675 | 1634 |
| single event, ns/cell-step | 1797 | 1715 | 1605 | 1572 |

**A fluctuating IC costs no more than a smooth one** — slightly less, since it is a smaller
fireball. But an *unregulated* one costs 3.8×: 6511 ns/cell-step at N=200 (82.6 s vs 21.7 s),
5566 at N=300 (236 s vs 67.9 s), with 729k/1.17M recovery failures. **Wall time per step is
therefore a usable early-warning signal** — a run that has gone unphysical is also slow, because
it spends its time on MOOD escalation retries.

**Per-sector cost, resolved.** `bench2d.jl` left this open because single-rep timings put "all
three sectors" below "shear alone". With 3 reps and medians, on the single event at N=200
(two independent runs):

| | ideal | +shear | +bulk | +diffusion | all three |
|---|---|---|---|---|---|
| run 1 | 15.15 s | +26.9% | +21.4% | +26.7% | +31.1% |
| run 2 | 16.67 s | +32.1% | +19.2% | +28.5% | +29.4% |

Rep spread is 3.4–11.8%, so shear, diffusion and all-three are not separable from one another —
but bulk is clearly the cheapest addition, and **the three sectors together cost about what the
most expensive one costs alone**. They are not additive: the per-step cost is dominated by the
shared primitive-recovery and relaxation passes, not by the individual closures. That is the
resolution of the question `bench2d.jl` left open, and it reproduces across both runs.

### 🟠 OPEN: the charge current is over its bound in hot spots

Measured on ev02 (regulated shear, stable run), max |ν|/n **above freeze-out**:

| τ [fm/c] | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|
| single event, `nu_clip` off | **1.455** | 0.410 | 0.207 | 0.079 |
| single event, `nu_clip` = 1 | **1.214** | 0.924 | 0.207 | 0.079 |
| ensemble, either | 0.250 | 0.155 | 0.090 | 0.023 |

On the ensemble the first-order current is comfortably inside its domain. On a single event it
reaches **1.46 at τ = 1 — 5.8× the ensemble, and past 1**, i.e. outside where a first-order
diffusion current means anything, while the run stays stable and conserves charge to 3e-6, so
nothing announces it.

`nu_clip_factor` does bind there (1.455 → 1.214) but it does **not** bring the current back
inside its domain, and at τ = 2 it is *higher* with the regulator on (0.924 vs 0.410) — capping
the current early leaves more charge to be moved later. Either way it changes the charm field by
only **1.6e-4 of peak**, with n_max identical to five digits at every τ. So the excursion lives
in cells carrying negligible charm: an applicability flag, not a source of error in n(x,y). The
regulator is not the answer here and is left off. Do not quote a charm-diffusion number from a
single-event run at τ ≲ 2 without saying this.

⚠ These are the corrected (2992) event. The pre-fix ev02 gave 3.95 at τ = 1; the excursion is
strongly event-dependent, which is itself a reason not to quote one event's number as a bound.

### Ladder

**12/12 pass** with G9 added (`test_fluctuating_ic2d.jl`), on ICs rebuilt after both the map fix
and the RNG fix.

---

## 6k. Correctness pass — Gubser (G1), and physical results

### 🔴 The analytic acceptance test had been dropped

The §6 plan listed `G2 | Gubser ideal + viscous | analytic`. The implemented ladder renumbered
around it and it was never written, which left the solver with **no comparison against an exact
solution**: G0 is Bjorken (transversally uniform, so it exercises none of the 2-D machinery), and
every other gate compares against the 1-D solver, against itself at another resolution, or against
a physical expectation. `src/gubser.jl` existed but only for `Grid1D`.

`test_gubser2d.jl` (G1) closes that. Gubser flow is azimuthally symmetric, which is the point: the
solver is told nothing of that, so a Cartesian (x,y) run tests accuracy against truth *and* whether
the discretisation manufactures azimuthal structure out of a radially symmetric solution.

**The reference is validated, not assumed** — two preliminaries run before the solver is touched:

| check | result |
|---|---|
| EOS actually conformal over the Gubser range, max\|e/(3P) − 1\| | **4.2e-13** |
| analytic solution satisfies ∂_μ(s u^μ) = 0, FD residual (validates T *and* u together) | **3.7e-9** |

(`ConformalHQEOS` carries a *massive* heavy-quark Boltzmann sector on top of the conformal light
sector; it is suppressed here by the tracer fugacity α = −20, and the residual is what is quoted.)

### Result: clean second-order convergence to the exact solution

τ = 1 → 2 fm/c, errors inside r ≤ 3 fm:

| N | dx [fm] | L2(T) | L∞(T) | L2(u) | x↔y | ring spread of T/T_exact | entropy error |
|---|---|---|---|---|---|---|---|
| 100 | 0.200 | 3.287e-3 | 8.17e-3 | 7.73e-3 | 1.8e-14 | 5.48e-3 | −1.28e-3 |
| 200 | 0.100 | 8.809e-4 | 2.38e-3 | 1.92e-3 | 5.1e-15 | 1.86e-3 | −3.54e-4 |
| 400 | 0.050 | 2.257e-4 | 6.34e-4 | 4.57e-4 | 5.6e-15 | 6.41e-4 | −9.56e-5 |

**Convergence order: T 1.90 → 1.96, u 2.01 → 2.07.** That is the formal order of MUSCL + SSPRK2 on
smooth data, and it settles a question G4 left ambiguous — G4 measured "order 0.90 in dx", but that
is convergence toward the *1-D solver's* answer, which carries its own error; against an exact
solution the scheme is second order.

**x↔y at 5e-15** — round-off. The Cartesian grid does not break the symmetry of the solution.

⚠ Two metrics in the first version of this gate were wrong, and both failed for that reason:

* *ring spread* binned raw T by radius. Cells in one bin sit at different radii, so the "spread"
  was the genuine radial variation of T across the bin width — it does not converge (it read a flat
  0.11 at every resolution) and is not an error. Normalising each cell by `T_exact(r_cell)` isolates
  grid imprinting, which then converges as above.
* *entropy* was asserted to be conserved in a fixed disc. Gubser flow **expands**, so entropy
  legitimately leaves: the drift converges to a physical **−11.6%, the same at every resolution**,
  which is how you can tell it is not an error. The meaningful quantity is the error against the
  analytic entropy in the same disc, which converges at ~2nd order (last column above).

### Physical results — `physics2d.jl`

Three centrality classes and two single events, N = 300, τ = 0.4 → 10 fm/c, η/s = ζ/s = 0.10.
`plots2d/physics_{v2,radialflow,lifetime,entropy}.*`, `plots2d/gubser_{T,ux,convergence}.*`.

| IC | ε₂(τ₀) | ε_p(4) | ε_p(8) | ε_p/ε₂ | ⟨u_T⟩(8) | A_fo(4) [fm²] | ΔN_c/N_c |
|---|---|---|---|---|---|---|---|
| 0–5% | 0.0852 | 0.0183 | 0.0381 | 0.447 | 0.695 | 247.1 | −3.0e-5 |
| 20–30% | 0.3362 | 0.1143 | 0.1521 | 0.452 | 0.636 | 153.5 | −1.5e-4 |
| 30–40% | 0.4092 | 0.1523 | 0.1744 | 0.426 | 0.651 | 125.0 | −3.0e-4 |
| 20–30% ev01 | 0.3312 | 0.1195 | 0.1668 | 0.504 | 0.779 | 157.3 | +4.5e-6 |
| 20–30% ev02 | 0.2036 | 0.0686 | 0.0968 | 0.476 | 0.807 | 160.6 | −4.1e-6 |
| 0–5% ideal | 0.0852 | 0.0209 | 0.0402 | 0.472 | 0.592 | 224.3 | −3.0e-4 |
| 30–40% ideal | 0.4092 | 0.1776 | 0.2059 | 0.503 | — | 106.7 | −2.9e-3 |

`ε_p` is the standard stress-tensor momentum anisotropy over the whole fireball. ⚠ The
freeze-out-restricted `v₂` is *not* usable late: by τ = 8 the 30–40% class has 15 fm² above T_fo, a
handful of cells, and `v₂` there read 0.95. That is a measurement artefact, not flow.

Four things here are physics with a known sign, and all four come out right:

* **ε_p increases with centrality**, tracking ε₂ — and **ε_p/ε₂ = 0.43–0.50 across every class and
  both single events**, i.e. a near-universal linear response, which is the defining hydrodynamic
  behaviour.
* **Viscosity suppresses elliptic flow**: ideal 0.0402 vs viscous 0.0381 at 0–5% (−5%), and
  0.2059 vs 0.1744 at 30–40% (−15%) — a larger suppression in the smaller system.
* **Viscosity increases radial flow**: ⟨u_T⟩(8) = 0.695 viscous vs 0.592 ideal at 0–5%. Opposite
  sign to its effect on ε_p, which is the textbook combination.
* **Lifetime orders with size**: T(0) crosses T_fo between τ = 9–10 (20–30%) and 7–8 fm/c
  (30–40%); 0–5% is still above at τ = 10.

**Charm number is conserved to 3e-6 … 3e-4.** ⚠ the first version of this script reported a drift of
exactly **+24.0** = τ_f/τ₀ − 1: `iDtau` *is* τ·J^τ and I multiplied by τ again.

### The second law, and what the ideal run does

**Viscous entropy rises monotonically in every run — it never decreases at any of the 15 sampled
times.** Nothing in the scheme imposes that; Israel–Stewart guarantees it in the continuum, a
discretisation need not. Production is +21.1% / +26.2% / +28.7% for 0–5% / 20–30% / 30–40%: more in
the more peripheral system, as the gradients are larger relative to its size.

The **ideal** runs lose 0.16%, and the loss does *not* shrink with resolution (−0.00121 / −0.00149 /
−0.00162 at N = 150 / 225 / 300), so it is not plain numerical diffusion. Localised by splitting
S(τ) by radius and temperature:

| τ | 1.0 | 2.0 | 4.0 | 7.0 | 10.0 |
|---|---|---|---|---|---|
| ΔS/S | **+1.6e-4** | **+1.0e-4** | −1.6e-4 | −7.3e-4 | −1.61e-3 |
| fraction of S below T_fo | 0.004 | 0.009 | 0.039 | 0.200 | **0.565** |
| fraction of S beyond r = 10 fm | ~0 | 9e-6 | 0.003 | 0.083 | **0.277** |

Ideal entropy is conserved to **1.6e-4 while the fireball is hot and compact**, and the drift
appears only as matter becomes cold and dilute (56% of the entropy is below freeze-out by τ = 10)
and approaches the box edge (0.5% beyond r = 15 fm of an 18 fm half-box). Together with G1 — where
the entropy error converges at 2nd order in a region with no cold tail — that places the drift in
the vacuum/floor treatment and the outer boundary, not in the core algorithm.

### Ladder

**13/13**, G1 added.

🪤 The Julia soft-scope trap bit **three times** in this session's scripts (`base`, `τ` twice) — each
time in a top-level `for`. It is in CLAUDE.md and it still costs a run every time.

---

## 6l. Second correctness pass — conservation identities, sound, and the charm sector

### The two exact Milne identities (the planned G5 was only ever done for charge)

With g = diag(−1,1,1,τ²), √−g = τ, Γ^τ_{ηη} = τ:

* **ν = τ:** d/dτ[τ ∫T^{ττ}] = −∫(P + Π + π^η_η). Has a source — this is where a Milne
  geometric-term error would hide.
* **ν = x:** d/dτ[τ ∫T^{τx}] = 0. **No source at all.** The ICs have u = 0, so total transverse
  momentum starts at zero and must stay there while a lumpy event generates plenty of it locally.

| | result |
|---|---|
| net transverse momentum, τ∫T^{τx} / τ∫\|S\| | **1e-6 … 8e-6** over every run |
| energy identity, relative residual (production CFL) | ~5e-3 |

The energy residual needed chasing. It was **identical (0.1976) at N=150 and N=300**, so not spatial;
refining my τ-stencil dropped it and then it **saturated**, so not the stencil either. Two controls
settled it:

* **CFL scan** (viscous): 4.47e-3 → 1.15e-3 → 6.67e-4 as CFL halves. It converges with the
  *solver's* timestep.
* **Ideal** (no dissipative sectors, hence no operator splitting): 6.23e-3 → 1.72e-3 → 6.01e-4 with
  dτ, converging at ~2nd order with **no floor at all**.

So the geometric source terms are right, and the viscous floor is the **first-order operator
splitting** between the hydro step and the dissipative relaxation — which `main2D.jl` documents as
deliberate ("inside an already first-order splitting"). At production CFL = 0.15 it is 5e-3.

### Sound: the first quantitative test of the VISCOUS sector against theory

`bc2d.jl`'s `:periodic` carries the comment "provided for the sound-wave benchmark (gate G3)".
That benchmark was never written. Uniform background at rest, τ₀ = 40 fm so θ = 1/τ = 0.025 fm⁻¹ is
small against ω ≈ 0.5 fm⁻¹, amplitude 1e-3, tracking the complex Fourier amplitude of the k-mode.

**Sound speed**, ideal, three wavelengths (L = 4, 6, 8 fm):

| c_s measured | 0.53807 | 0.53808 | 0.53812 |
|---|---|---|---|
| exact √c_s² | 0.54272 | 0.54272 | 0.54272 |

**0.85% low and wavelength-independent to 5 digits** — non-dispersive, as sound must be. The offset
is the background cooling over the 20 fm measurement window.

**Attenuation**, L = 6 fm, against the Navier–Stokes/IS prediction Γ = (4/3)η k²/2(e+P):

| η/s | Γ measured − floor | predicted | ratio |
|---|---|---|---|
| 0.01 | 0.00440 | 0.00412 | **1.068** |
| 0.02 | 0.00880 | 0.00824 | **1.067** |
| 0.04 | 0.01756 | 0.01649 | **1.065** |

Exactly linear in η/s with a constant 6.6% offset — the Bjorken expansion the static dispersion
omits. **The shear sector reproduces the theoretical attenuation to 7%.**

⚠ Two of my own errors, both in the *test*: the first version used `cos(kx)` with u = 0, which is a
**standing** wave — its Fourier coefficient oscillates through zero, the phase does not advance, and
it measured c_s = 0.13 against 0.54. And the first damping reference mixed unit systems (the code's
`viscosity()` carries an internal `invfmGeV`), inflating the predicted Γ by ~26× and making the
solver look wrong by a factor of 17. In consistent natural units τ_π = 0.06 fm, so ωτ_π ≈ 0.03 and
**Navier–Stokes is the correct limit**, not the IS cubic I had thought was needed.

### 🔴 The charm sector: the sign is right, and the "shortfall" was the vacuum gate

The centrality scan found ⟨r²⟩ of the charm *decreasing* with D_sT (34.9 → 21.7 fm² at τ = 8), which
looks like a sign error. It is not. On a **uniform-temperature** background, where ∇α ∥ ∇n and the
answer must be ordinary Fick diffusion:

| D_sT | ⟨r²⟩(0) → ⟨r²⟩(12) | slope | slope/4D_s |
|---|---|---|---|
| 0 | 6.9595 → 6.9610 | 1.2e-4 | — (no spurious diffusion) |
| 0.0582 | 6.9595 → 7.3554 | 0.03307 | 0.216 |
| 0.1163 | 6.9595 → 7.7395 | 0.06527 | 0.213 |
| 0.2326 | 6.9595 → 8.4853 | 0.12806 | 0.209 |

It **broadens**, and the slope is **exactly linear in D_sT** (ratios 1.974, 1.962 against 2.000). So
the fireball result is real physics: the current drives the *fugacity* α uniform, not the density,
and with a hot centre uniform α means n ~ n_eq(T), which is peaked — charm is pulled inward.

The constant 0.21 shortfall against the code's own `D = (D_sT/T)/fmGeV` was **not** the relaxation
time (τ_n = 0.025–0.4 fm; scaling `tauN_coeff` 16× moved the slope by 1%). It is the **density-gated
vacuum ramp** — the D6 fix — which throttles κ linearly between n = 1e-6 and 2e-3:

| α | n_bg | vacuum weight | slope/4D_s |
|---|---|---|---|
| −6 | 3.35e-4 | 0.167 | 0.213 |
| −4 | 2.47e-3 | 1.000 | **1.021** |
| −2 … +2 | ≥1.8e-2 | 1.000 | **1.021** |

**Above the ramp the realized diffusion matches the requested D_s to 2%.** My test background had
been sitting inside it. ⚠ Practically: the ramp throttles charm transport wherever n < 2e-3 fm⁻³,
which in a real fireball is a large outer region — by design, but it means the *effective* D_s there
is below the nominal one. Worth stating before quoting a diffusion number from the dilute edge.

### Physics scans — `physics_scans.jl`

Six centrality classes (ε₂ = 0.085 … 0.484), N = 250, τ → 8 fm/c.

| class | ε₂ | ε_p viscous | ε_p ideal | suppression |
|---|---|---|---|---|
| 0–5% | 0.0852 | 0.0381 | 0.0402 | **5.2%** |
| 5–10% | 0.1610 | 0.0746 | 0.0796 | 6.3% |
| 10–20% | 0.2384 | 0.1105 | 0.1197 | 7.7% |
| 20–30% | 0.3361 | 0.1520 | 0.1705 | 10.8% |
| 30–40% | 0.4091 | 0.1743 | 0.2060 | 15.3% |
| 40–50% | 0.4842 | 0.1917 | 0.2411 | **20.5%** |

ε_p rises monotonically with centrality, and the **viscous suppression grows monotonically from 5%
to 21%** — smaller systems are more sensitive to viscosity, which is the textbook result.

Viscosity scan at 20–30%, η/s = 0 → 0.24 (ζ = 0):

| η/s | 0.00 | 0.02 | 0.05 | 0.08 | 0.12 | 0.16 | 0.24 |
|---|---|---|---|---|---|---|---|
| ε_p(8) | 0.1705 | 0.1643 | 0.1559 | 0.1483 | 0.1392 | 0.1311 | 0.1175 |
| ⟨u_T⟩(8) | 0.4968 | 0.5141 | 0.5395 | 0.5619 | 0.5889 | 0.6120 | 0.6522 |
| ΔS/S | −0.0025 | +0.027 | +0.065 | +0.099 | +0.138 | +0.170 | +0.223 |

All three monotone and in the right direction: shear viscosity **suppresses elliptic flow (−31%),
enhances radial flow (+31%), and produces entropy**, with the ideal run at ΔS/S ≈ 0.

`plots2d/scan_{centrality,response,viscosity,charm}.*`.

---

## 6m. VISCOUS GUBSER (G1v) — the last analytic gap closed

The plan asked for "Gubser ideal + viscous". §6k did the ideal half; this is the viscous one, and
it is now the only test in the ladder that compares the **nonlinear** shear sector against an exact
solution (sound waves cover the linear regime, everything else compares against the 1-D solver or
against a sign).

### The reference, derived rather than quoted

Gubser flow is a fixed point of the conformal symmetry, so the **kinematics are viscosity-independent** —
only T and π change. Weyl-rescaling Milne by 1/τ² into de Sitter leaves the fluid at rest, with
θ̂ = 2tanhρ and σ̂^η_η = +(2/3)tanhρ. Writing π̄ ≡ π^η_η/(e+P) — Weyl-invariant, so the *same number*
in Milne — energy conservation and the IS shear equation with δ_ππ = (4/3)τ_π reduce to

    dT̂/dρ = −(2/3)T̂ tanhρ + (1/3)T̂ π̄ tanhρ
    dπ̄/dρ = −(4/3)π̄² tanhρ − π̄/τ̂_π + (4/15) tanhρ,      τ̂_π = 5(η/s)/T̂_nat

The δ_ππ term cancels the (8/3)π̄tanhρ from the chain rule exactly, which is why the second equation
is this clean.

**Three validations of that reference, all before the solver is touched:**

| check | result |
|---|---|
| η/s → 0 must integrate to T̂ = T̂₀cosh^{−2/3}ρ, which via cosh²ρ = Den/(4q²τ²) is the code's own `gubser_temperature` | **3.56e-11** |
| π̄ must relax onto Navier–Stokes, π̄_NS = (4/3)(η/s)tanhρ/T̂_nat | 2.7% (η/s=0.005), 10.4% (0.02) |
| the attractor: seeding π̄ at NS vs at 0 must give the same solution | 6.0e-5, 2.4e-3 |

### Result

τ = 1 → 2 fm/c, errors inside r ≤ 3 fm:

| N | L2(T), η/s=0.005 | L2(π̄) | L2(T), η/s=0.02 | L2(π̄) |
|---|---|---|---|---|
| 100 | 3.242e-3 | 5.284e-2 | 3.146e-3 | 4.992e-2 |
| 200 | 8.689e-4 | 1.699e-2 | 8.391e-4 | 1.790e-2 |
| 400 | 2.215e-4 | 6.756e-3 | 2.101e-4 | 7.742e-3 |
| **order** | **1.97** | **1.33** | **2.00** | **1.21** |

Temperature at full second order, shear converging at ~1.2–1.3 (π is limiter-sensitive and derived
through the tracelessness projection). x↔y stays at 1e-14.

### 🪤 Three traps, all mine, and how each was caught

1. **Backward ODE integration is exponentially unstable.** The relaxation −π̄/τ̂_π is anti-damping in
   reverse; measured, it dies at |Δρ|/τ̂_π = 16–19 for every η/s (ρ = −0.32 at η/s = 0.002, −0.69 at
   0.005, −2.77 at 0.02) — exactly where 1e-16 has grown to O(1). Smaller η/s dies *sooner*, which
   is the giveaway. Fixed by integrating forward only, from the analytically-known ideal T̂ at ρ_min
   onto the attractor.

2. **The sign of σ̂^η_η.** The system is invariant under π̄ → −π̄ with both source terms flipped, so a
   wrong sign looks perfectly self-consistent and only a comparison against the solver exposes it.
   **Anchored on Bjorken, not assumed**: there P_L = P − (4/3)η/τ, i.e. π^η_η < 0, and G0 confirms
   the code reproduces it (piEta = −8.8e-1). In de Sitter the η direction is the one that does *not*
   expand — the mirror image — so the sign is the opposite one.

3. **τ_π in GeV instead of fm⁻¹**, the same trap as the sound gate: it makes τ̂_π and hence π̄_NS a
   factor invfmGeV = 5.07 too large. Diagnosed by measuring σ^η_η from the solver's *own* velocity
   field: 2ησ^η_η/(e+P) with the sound gate's η normalisation gave +0.00363 against the solver's
   +0.00345, 5% — which is what identified the missing factor rather than a solver defect.

After all three, π̄ agrees pointwise at **1–3%** and T at **0.06%** at N = 200.

### Ladder

**14/14.** The analytic coverage is now: Bjorken (G0), ideal Gubser (G1), **viscous Gubser (G1v)**,
sound speed and attenuation, the two Milne conservation identities, and the second law.

---

## 6n. BULK VISCOSITY vs THEORY (Gs) — the last untested sector

Gubser is conformal, so no exact solution can constrain ζ: G2 and the 1-D comparison only check the
bulk sector against another implementation. Sound does constrain it —
Γ = ½k²[(4/3)η + ζ]/(e+P) — so running with η = 0 isolates it.

### 🔴 ζ is a Lorentzian, not (ζ/s)·s

`SimpleBulkViscosity` is peaked at T = 0.175 GeV with width 0.024:
ζ = ζs_peak/(1 + ((T−0.175)/0.024)²)·invfmGeV·s. **At the shear gate's background (T = 0.35) that
factor is 1/54 = 0.018** — bulk is effectively switched off there. The test therefore runs *on* the
peak.

### Magnitude, at T = 0.175

| ζ/s | Γ − floor | Navier–Stokes | ratio |
|---|---|---|---|
| 0.02 | 0.01258 | 0.01237 | **1.018** |
| 0.05 | 0.03156 | 0.03091 | **1.021** |
| 0.10 | 0.06402 | 0.06183 | **1.035** |

### Temperature dependence — the Lorentzian shape, not just its normalisation

| T₀ | lorentz(T₀) | ⟨lorentz⟩ | Γ − floor | ratio using T₀ | ratio using ⟨lorentz⟩ |
|---|---|---|---|---|---|
| 0.175 | 1.0000 | 0.9657 | 0.03156 | 1.021 | **1.057** |
| 0.190 | 0.7191 | 0.8473 | 0.02587 | 1.263 | **1.072** |
| 0.210 | 0.3198 | 0.4174 | 0.01156 | 1.404 | **1.075** |
| 0.250 | 0.0929 | 0.1170 | 0.00273 | 1.356 | **1.076** |

⚠ The middle column is a trap I fell into: ζ changes by 3× across the Lorentzian's width and the
Bjorken background **cools ~4% during the run**, so evaluating the Lorentzian at the *initial*
temperature is wrong — at T₀ = 0.175 the run drifts off the peak, at T₀ = 0.21 it drifts toward it.
That produced a ratio rising 1.02 → 1.40 with T₀, which is the drift and not the solver. With the
run-averaged factor the ratio is **constant at 1.057–1.076 across a factor of 8.5 in ζ**.

That constancy is the point: a coefficient error would scale with ζ, a background effect does not.
The residual ~7% is the same offset the shear channel shows (1.073/1.072/1.071 at η/s =
0.01/0.02/0.04), i.e. the Bjorken expansion θ = 1/τ that the static dispersion relation omits.

**Both dissipative sectors are now quantitatively confirmed against theory to ~7%**, and the sound
speed to 0.55%.

### Gate

`test_sound2d.jl` (Gs) folds all three into the ladder: sound speed, shear attenuation linear in
η/s with a constant ratio, and bulk attenuation at two points on the Lorentzian with the ratio
required to be the same at both — so the temperature dependence is gated, not just the magnitude.

---

## 6o. Sub-percent: how far it goes, and where it stops

> 🔴 **RETRACTED 2026-09-02 — see §6q.** The conclusion below, that "the realized shear viscosity is
> ~3.4–4.0% above (4/3)(η/s)s/(e+P)" and that this is a normalisation question about a function
> shared with the 1-D solver, is **WRONG**. The 4% was my sound-wave *initialisation* exciting the
> backward mode, not the solver. Measured directly, π matches its target to 0.02%, and with the exact
> eigenmode the damping matches theory to 0.2–0.3%. The scans below are still correct as measurements
> — they are what established that the effect is independent of resolution, timestep, amplitude, τ_π
> and τ₀, which is precisely why it looked like a coefficient error. Kept in place, dated.

The sound gate's ~7% was attributed to the Bjorken background. That is a sharp prediction — the
offset must fall like 1/τ₀ — and it half holds:

| τ₀ | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|
| excess in Γ | 11.07% | 7.15% | 5.05% | 3.96% | 3.41% |
| c_s error | 0.933% | 0.433% | 0.176% | 0.046% | **0.020%** |

The c_s error vanishes cleanly: **the ideal sector reaches sub-percent, and then two more decades**.
Γ does not — it fits **excess = 2.86% + 176/τ₀ %**, a 1/τ₀ piece on top of a constant.

That constant survives everything:

| excluded | evidence |
|---|---|
| resolution | 3.83 / 3.96 / 4.07% at N = 96 / 128 / 192 (converged, rising slightly) |
| timestep / operator splitting | 1.0396 / 1.0397 at CFL = 0.15 / 0.075 |
| perturbation amplitude | 1.0397 / 1.0396 at 1e-4 / 1e-3 |
| relaxation time | saturates at 1.040 as τ_π → 0 (Cs = 0.05 / 0.2 / 0.8 → 1.0244 / 1.0396 / 1.0404) |
| the background estimator | fitting Γ against k² puts the background in the intercept; slope still gives **1.0337** |
| the entropy definition | mine vs the code's: 92.270229 both, ratio 1.000000 |

and the constitutive relation is right by hand: the code's `ns_shear_target_2d` is π_NS = −2ησ, and
for Bjorken that gives the textbook −(4/3)η/τ.

**So the realized shear viscosity is ~3.4–4.0% above (4/3)(η/s)s/(e+P).** That is a normalisation
question about what η the code intends, and `viscosity()` is shared with the 1-D production solver —
so it is not a 2-D bug to fix here. Sub-percent on the dissipative coefficients needs that decision
first.

---

## 6p. P9 — a 2-D freeze-out surface and Cooper-Frye yield

`freezeout2d.jl`. The surface is boost-invariant and parameterised as τ = τ_fo(x,y), the last time
each transverse cell crosses T_fo downwards, with
dΣ_μ = τ_fo(1, −∂_xτ_fo, −∂_yτ_fo, 0) dx dy dη. Conventions taken from the 1-D calibration:
T_fo = 0.156, m_π = 0.138, p_T ∈ [0.1, 5.0], `pion_yield_factor` = 2.0×2.5.

⚠ **(ħc)³**: dΣ_μp^μ carries fm³·GeV and d³p/E carries GeV², so the measure is (2πħc)³, not (2π)³.
Omitting it returned dN/dy = 8.8 instead of ~1300.

### Yields

| class | dN_π/dy | ratio to 0–5% |
|---|---|---|
| 0–5% | 1266.0 | 1.000 |
| 5–10% | 1020.4 | 0.806 |
| 10–20% | 761.1 | 0.601 |
| 20–30% | 513.6 | 0.406 |
| 30–40% | 338.4 | 0.267 |
| 40–50% | 227.2 | 0.179 |

Bose statistics; Boltzmann runs 12% lower throughout. Negative-dΣ·p contributions are ≤0.06%.

### 🔴 The inherited normalisation does NOT carry over

The control is the 2-D solver seeded from the *same* radial profile the 1-D calibration used
(`initial_profiles_physical.csv`), run through the *same* Cooper-Frye:

**axisymmetric control 1087.4 against the 1-D calibration's 1339.63 — a ratio of 0.812.**

So a 2-D re-derivation of `Norm` would land ~23% above the inherited value. That is the number P9
was asking for, and it says the caveat in `BuildIC2D.jl`'s header is real and quantified rather than
hypothetical.

⚠ It is **not yet attributed**. Candidates, none excluded: the 1-D calibration ran through Fluidum's
`spectra_internal`, not this Cooper-Frye, so the two may differ in feed-down or statistics
conventions; the 1-D profile is tapered (`taper_width = 1.0`) while the 2-D IC carries its own cold
tail; and this surface omits δf. Until one of those is pinned down, **quote the ratio, not the
absolute yield.**

### What is trustworthy now

The *shape* — the ratios above are independent of the overall normalisation, and they are the
quantity to compare against ALICE's measured centrality dependence. I have not done that comparison
here because it needs the published table rather than a recalled number.

⚠ 335–405 cells per run re-heat above T_fo after first freezing (3.6–5.0% of the hot cells), which a
single-valued τ_fo(x,y) cannot represent — the 2-D analogue of the non-bijective-contour problem
`FreezeOutContour.jl` documents in 1-D. Reported by the script rather than assumed away.

---

## 6q. The 4% was mine, not the solver's — sub-percent reached

§6o reported a hard ~4% floor in the sound attenuation and concluded the realized η was 4% high.
That was wrong, and the way it was wrong is worth keeping.

### What the scans could not see

The excess was independent of resolution, timestep, amplitude, τ_π, τ₀ and the estimator, and it was
**the same 1.0345 for η/s = 0.01, 0.02 and 0.04**. A coefficient error looks exactly like that. What
it actually was: the initialisation used the *ideal* eigenvector — δu in phase with δT, and π = 0 —
which is wrong at O(Γ/ω) ≈ 1.5%. That residue excites the backward-propagating mode, |c(τ)| beats,
and a straight-line fit to log|c| returns a window average. **The eigenvector error scales with η,
and Γ scales with η, so the ratio is η-independent** — which is why every scan came back flat.

### What caught it

Measuring π **directly** against its target instead of through the dispersion relation:

| η/s | \|π^xx\| / \|−2ησ^xx\| | phase offset | ωτ_π |
|---|---|---|---|
| 0.01 | **0.99978** | 0.90° | 0.0158 rad = 0.90° |
| 0.02 | **0.99947** | — | — |
| 0.04 | **0.99856** | — | — |

π = 2ησ/(1 − iωτ_π), textbook Israel–Stewart, to 0.02%. Alongside: the regulator was **off**
(`pi_clip_factor = −1`), |π^xx|/P was 1.8e-4 – 7e-4 so nothing was near clipping, and transverse
momentum was conserved to **7e-12 – 1.2e-11** on the periodic box. The solver was never the problem.

The confirming symptom: splitting the fit window in three gave apparent ratios **1.275 / 0.484 /
0.918** — not a decaying exponential at all, but beats.

### Sub-percent, with the exact eigenmode

δu from ω/(wk) with the complex ω of the IS dispersion, and δπ from the IS response:

| η/s | ideal eigenvector | exact eigenmode | half-window split, ideal → exact |
|---|---|---|---|
| 0.01 | 1.03454 | **1.00337** | 0.089 → **0.0018** |
| 0.02 | 1.03466 | **1.00301** | 0.067 → **0.0019** |
| 0.04 | 1.03442 | **1.00213** | 0.023 → **0.0019** |

The half-split is the diagnostic that separates the two hypotheses: a genuine coefficient error
leaves the decay exponential, mode contamination does not.

### Gate Gs, now at τ₀ = 320 with the exact eigenmode

| quantity | agreement |
|---|---|
| sound speed | **0.05%** |
| shear attenuation, η/s = 0.01 / 0.02 / 0.04 | **1.00330 / 1.00287 / 1.00168** |
| bulk attenuation, T = 0.175 / 0.210 | **0.995 / 0.990** |
| half-window split | 0.005 |

**Every sector now agrees with theory at or below 1%**, and the gate asserts it: ratios in
[0.98, 1.02] for shear and a half-split below 0.01, so a future change that reintroduces either a
coefficient error or mode contamination fails.

---

## 6r. The benchmark figure set, and what breaks under exotic initial conditions

### `plot_benchmarks2d.jl` — the four cases with a known answer

| benchmark | reference | result |
|---|---|---|
| Bjorken | analytic (τs and τn conserved) | max rel. error in T over τ = 0.4–12: **9.27e-5** |
| Gubser, ideal | analytic (PRD82 085027) | L2(T) 3.29e-3 → 2.26e-4, **order 1.96** |
| Gubser, viscous | semi-analytic de Sitter ODE | L2(T) 3.15e-3 → 2.10e-4, **order 2.00** |
| Sound | Israel–Stewart dispersion | Γ/Γ_IS = **1.00334 / 1.00299 / 1.00210** |

`plots2d/bench_{bjorken, gubser_ideal_T, gubser_viscous_T, gubser_viscous_pi, sound, convergence}`.
The viscous-Gubser π̄ panel is the one worth looking at: the solver's shear sits on the ODE curve at
every radius and every time, which is the statement that the nonlinear shear sector is right.

⚠ The sound panel subtracts the ideal run, as gate Gs does. Without that subtraction the same
figure reads 0.964 / 0.983 / 0.992 — the ideal curve is not flat, because the perturbation grows
adiabatically as the background cools, and that offset is not viscous damping.

### `exotic_ic2d.jl` — deliberately nasty initial conditions

Every validated case so far is smooth or nearly so. These are not: multiplicative white noise on T at
10/30/60% correlated over 0.5 fm, the same at 30% correlated over **one cell**, five σ = 0.4 fm hot
spots reaching T = 1.34 GeV, a hollow ring, two blobs 8 fm apart, and a 0.5 × 12 fm filament.

**All nine run to completion. None crashes.** The worst values over the run (not the final state —
reading only the end is vacuous here, since most of these have no cell above T_fo left at τ = 8, so
max|u| comes back 0.00 and min(P+Π) comes back +Inf):

| case | max\|u\| | min(P+Π) | max\|π\|/P | ΔQ/Q | |
|---|---|---|---|---|---|
| smooth | 0.70 | +6.4e-3 | 0.533 | 1.0e-4 | ok |
| noise_10 | 0.80 | +6.2e-4 | 0.782 | 1.1e-4 | ok |
| noise_30 | 1.01 | +6.2e-4 | 1.000 | 1.1e-4 | ok |
| **noise_60** | 1.63 | **−4.0e-2** | 1.006 | 1.8e-4 | **inadmissible** |
| noise_cell | 0.92 | +6.1e-4 | 0.852 | 1.2e-4 | ok |
| **hotspots** | 2.84 | **−2.9e-2** | 1.119 | 4.9e-6 | **inadmissible** |
| ring | 1.21 | +5.4e-3 | 0.955 | 8.2e-5 | ok |
| binary | 0.84 | +1.3e-3 | 1.000 | 3.7e-4 | ok |
| filament | 1.24 | +6.4e-4 | 1.003 | 9.3e-4 | ok |

Charge is conserved to ≤9e-4 in every case, including grid-scale noise. The evolution is physically
sensible throughout (`plots2d/exotic_ics.png`): the hot spots each launch an expanding shock ring and
the rings intersect in a star pattern, the hollow ring implodes to a hot focus, the two blobs merge
across a compression ridge, and the filament expands transversely.

### 🔴 `Pi_clip_factor = 1.0` cannot guarantee P + Π > 0

The two failures are both min(P+Π) < 0 — the bulk pressure. `Pi_clip_factor` is off by default,
exactly as `pi_clip_factor` was before G9, so the obvious move is to switch it on. **It changes
nothing at f = 1.0**, to every printed digit. Scanning it on `hotspots` says why:

| Pi_clip | min(P+Π) | min(Π/P) |
|---|---|---|
| off | −2.944e-2 | −1.1077 |
| **1.0** | **−2.944e-2** | **−1.1077** |
| 0.5 | **+3.06e-2** | −0.5472 |
| 0.2 | +4.89e-2 | −0.2171 |

The regulator works exactly as designed — min(Π/P) tracks −f — but Π is capped against the pressure
**at relaxation time**, and P then falls ~10% before the state is next sampled. The realized bound is
therefore ≈ 1.1 f, and f = 1.0 leaves Π/P at −1.108: not enough. **f ≲ 0.9 is required for the
positivity it is supposed to provide, and f = 0.5 gives margin.** The flag does reach the model
(verified directly); this is a lag, not a wiring defect.

So: the solver survives grid-scale noise, hollow geometries, colliding blobs and filaments without
special handling, and the boundary of what it will do is 60% local fluctuations or 1.3 GeV hot spots,
where the bulk pressure goes negative unless the bulk regulator is set below 1.

---

## 6s. The physical single-event Pb+Pb initial condition, 0–5%

`BuildIC2D.jl events 00-05 4` (or `build(CFG2D(; cent_lo=0, cent_hi=5, n_average=1,
event_offset=k, dx=0.125))`). What actually fluctuates, from `MCGCollisionDensity2D`: Woods-Saxon
nucleon positions, per-participant `Gamma(k, 1/k)` weights, and the reduced-thickness generalised
mean, deposited as W = 0.5 fm Gaussians. Charm follows binary collisions, the soft entropy follows
participants, and the two keep separate geometries.

| | ev01 | ev02 | ev03 | ev04 | ensemble |
|---|---|---|---|---|---|
| event index / N_coll | 5014 / 2193 | 5301 / 2124 | 1487 / 2129 | 1044 / 2036 | — / 1787 |
| ε₂ of the seeded energy density | 0.166 | 0.046 | 0.210 | 0.090 | 0.085 |
| **ε₃** | **0.095** | **0.114** | **0.133** | **0.082** | **0.0018** |
| T_max [GeV] | 0.866 | 0.838 | 0.926 | 0.777 | 0.588 |
| N_charm | 29.20 | 28.28 | 28.35 | 27.11 | 23.79 |

ε₃ is 45–70× the ensemble value; T_max is 1.34–1.59× the ensemble peak. N_coll exceeds the ensemble
mean because the 0–5% window selects the highest-multiplicity events, and N_charm follows it.

Admissibility, N = 280 (dx = 0.129), `pi_clip_factor = 1.0`, `Pi_clip_factor = 0.5`, worst over the run:

| | max\|u\| | min(P+Π) | max\|π\|/P | ΔQ/Q | ic bad |
|---|---|---|---|---|---|
| ev01–ev04 | 1.62–1.72 | +2.9e-2 … +3.1e-2 | 1.00–1.07 | 7.6e-6 … 9.4e-6 | 0 |
| ensemble | 1.05 | +3.05e-2 | 0.567 | 1.4e-5 | 0 |

### 🔴 The IC grid spacing must divide the domain

Built at dx = 0.15 fm, every event reported **`ic bad = 8`** where the ensemble reported 0.
`collect(-14.0:0.15:14.0)` ends at **13.9**, because 28/0.15 is not an integer — the IC is short by
0.1 fm on the +x and +y edges and those solver cells have no data. Silent, small, and exactly the
kind of thing that becomes a mystery later. `BuildIC2D.jl` now asserts that dx divides 2·xmax;
0.25, 0.125 and 0.1 do, 0.15 does not. Rebuilt at dx = 0.125: `ic bad = 0` for all four.

### ⚠ The geometry is resolved at dx = 0.25; the peak temperature is not

Same event (index 5014) at three IC spacings:

| dx | grid | ε₂(entropy) | ε₃(entropy) | T_max |
|---|---|---|---|---|
| 0.25 | 113² | 0.0993 | 0.0758 | 0.7516 |
| 0.15 | 187² | 0.0993 | 0.0758 | 0.8016 |
| 0.10 | 281² | 0.0993 | 0.0758 | 0.8539 |

ε₂ and ε₃ agree to four digits at every spacing — flow observables are safe on the coarse grid. T_max
is still climbing at dx = 0.10, so anything driven by the hottest cells (charm production in hot
spots, |π|/P, the sub-nucleon structure) depends on the IC resolution and the choice must be quoted.
**dx = 0.125 is the one used here**, matched to the solver's own dx ≈ 0.129 — a finer IC would be
interpolated away on the solver grid.

### ⚠ The charm current is over its bound early, as at 20–30%

max|ν|/n above freeze-out reaches **1.64 (ev01) and 2.00 (ev03)** at τ = 1, falling to 0.13–0.14 by
τ = 8. Same statement as §6l: on a single event the first-order diffusion current is outside its
domain for τ ≲ 2, and it is worse at 0–5% than at 20–30% (1.46) because the hot spots are hotter.

Figures: `plots2d/{T,n}_{evolution,slices}_00-05_ev01.*` and `_00-05_ev03.*`.

---

## 6t. dx = 0.05 fm: the production runs are already converged

The benchmarks were effectively there already — Gubser at N = 400 over ±10 fm **is** dx = 0.05, and
the sound gate at N = 128 over 6 fm is dx = 0.047. The open question was the production case.

Physical single event `00-05_ev01`, IC at dx = 0.05, solver box ±18 fm, to τ = 8:

| N | dx [fm] | ε_p | v₃ | max\|u\| | max\|π\|/P | ΔQ/Q | wall |
|---|---|---|---|---|---|---|---|
| 200 | 0.180 | 0.05479 | 0.08671 | 1.69 | 1.002 | 9.3e-6 | 0.4 min |
| 280 | 0.129 | 0.05478 | 0.08682 | 1.70 | 1.030 | 8.6e-6 | 0.9 min |
| 400 | 0.090 | 0.05485 | 0.08729 | 1.71 | 1.029 | 7.2e-6 | 2.3 min |
| 560 | 0.064 | 0.05485 | 0.08740 | 1.71 | 1.000 | 4.3e-6 | 5.9 min |
| **720** | **0.050** | **0.05490** | **0.08748** | **1.72** | **1.000** | **2.7e-6** | **12.3 min** |

**ε_p moves 0.2% and v₃ 0.9% across a 3.6× change in dx**, with v₃ flattening (the last two differ by
0.1%). Charge conservation improves monotonically, 9.3e-6 → 2.7e-6. The seeded T_max is stable to
0.13%. So the production numbers quoted elsewhere in this document, taken at N = 250–300, are
converged; dx = 0.05 buys nothing except 14× the cost (1546 ns/cell-step at N = 720).

### 🪤 I contaminated my own first attempt

The first version of this table had N = 280 on a *different* initial condition. `BuildIC2D.jl` writes
`ic2d_00-05_ev01.csv` with no dx in the filename, and I launched the dx = 0.05 IC build and the
solver scan at the same time: the file was rewritten at 14:57:52, the N = 280 run had read the old
dx = 0.125 version at 14:57:20, and everything from N = 400 on read the new one. The giveaway was the
seeded T_max — 0.8629 for N = 280 against 0.8474 / 0.8480 / 0.8479 for the rest — and ε₂ jumping
0.1657 → 0.1588 between two rows that should differ only by resolution. Re-run on a single IC, the
sequence above is clean. Exactly the concurrency trap CLAUDE.md warns about, self-inflicted.

⚠ **The IC filename does not encode dx.** Two builds at different spacings collide silently. The
spacing *is* recorded in the `_meta.txt` (`grid = 561x561, dx=0.05`), so provenance is recoverable
after the fact, but the collision is real.

### The IC's own peak converges too, non-monotonically

| IC dx | 0.25 | 0.15 | 0.125 | 0.10 | 0.05 |
|---|---|---|---|---|---|
| T_max [GeV] | 0.7516 | 0.8016 | 0.8656 | 0.8539 | **0.8483** |

⚠ This is **not** a clean convergence sequence, and §6s overstated it as "still climbing". The
density→T map's reference (the 0–5% ensemble φ-average) is rebuilt at each spacing, so the map moves
with dx as well as the sampling. Taken together the values settle to ≈0.848 for dx ≤ 0.10, so
dx = 0.125 is within 2% and dx = 0.25 is 11% low. ε₂ and ε₃ are unaffected at any spacing.

### ⚠ `ic.nbad` is nonzero at fine solver resolution

Seeding the single-event IC reports nbad = 8 / 0 / 16 / 40 / 32 at N = 200 / 280 / 400 / 560 / 720 —
non-monotonic. **Harmless**: every cell ends with finite positive energy, charge is conserved to
2.7e-6, and every run is admissible. It is the documented α runaway in the cold tail (§6d), where a
finer grid samples more of the extreme tail and the fallback catches it. But G8 asserts
`ic.nbad == 0`, which holds only at the resolutions it runs (N = 150, 250) on the ensemble IC — the
assertion would not survive being applied to a single event at N ≥ 400.

### All four physical events at dx = 0.05

IC and solver both at dx = 0.05 (561² IC, N = 720 over ±18 fm), τ → 8, ~12 min each:

| event | ε₂ | ε₃ | ε_p | v₃ | ε_p/ε₂ | v₃/ε₃ | max\|u\| | min(P+Π) | ΔQ/Q |
|---|---|---|---|---|---|---|---|---|---|
| ev01 | 0.1588 | 0.0936 | 0.05490 | 0.08748 | 0.346 | 0.935 | 1.72 | +2.9e-2 | 2.7e-6 |
| ev02 | 0.0433 | 0.1143 | 0.03354 | 0.14011 | 0.775 | 1.226 | 1.74 | +3.1e-2 | 3.5e-6 |
| ev03 | 0.2039 | 0.1311 | 0.10581 | 0.03679 | 0.519 | 0.281 | 1.71 | +3.1e-2 | 3.9e-6 |
| ev04 | 0.0893 | 0.0815 | 0.07143 | 0.08497 | 0.800 | 1.043 | 1.63 | +3.1e-2 | 2.3e-6 |

All admissible: max|u| 1.63–1.74, min(P+Π) > 0, charge to ≤3.9e-6, max|π|/P ≈ 1.0.

⚠ The response ratios scatter far more than the ensemble's tight ε_p/ε₂ = 0.43–0.50 (§6k):
0.35–0.80 here, and v₃/ε₃ spans 0.28–1.23. ev03 is the extreme — the largest ε₂ *and* the largest
ε₃, but the **smallest** v₃. On a single event the harmonic planes need not align and the response
is not a scalar times ε_n, so per-event ratios are not the ensemble response coefficient. This is
the physics of event-by-event fluctuations, not a solver artefact — but it means a single event
cannot be used to read off a response.

---

## 6u. Nonlinear bulk (G0b), and the P9 19% decomposed

### G0b — bulk beyond the linear regime

The bulk sector's only quantitative check was gate Gs, which is *linear* (a 1e-4 perturbation about a
uniform background), and Gubser cannot help because it is conformal. **G0 never exercised bulk at
all.** `test_bjorken_bulk2d.jl` closes that: Bjorken flow reduces the solver's bulk sector to

    de/dτ = −(e + P + Π)/τ ,   dΠ/dτ = (−Π − ζ/τ)/τ_Π

(the code's law is τ_Π DΠ + Π = −ζθ with **no** δ_ΠΠ term — read off `dissipation2d.jl`, not
assumed), integrated by RK4 far below the solver's tolerance. Started at T₀ = 0.30 and run to τ = 6
so Bjorken cooling carries the system **through the ζ Lorentzian peak at T = 0.175** — a bulk test
that never enters the bulk regime is not one.

| ζ/s | T vs reference | Π vs reference | \|Π\|/P | x,y spread |
|---|---|---|---|---|
| 0.05 | 9.90e-5 | 5.20e-3 | 0.066 | 0.0 |
| 0.15 | 9.93e-5 | 7.51e-3 | 0.236 | 0.0 |
| 0.30 | 3.53e-5 | 1.57e-3 | **0.600** | 0.0 |

**Sub-percent at |Π|/P = 0.6**, and transverse uniformity is *exact* (0.0, not merely small).
Convergence in the timestep, ζ/s = 0.15:

| CFL | 0.20 | 0.10 | 0.05 | 0.025 |
|---|---|---|---|---|
| \|ΔT\|/T | 1.42e-4 | 4.92e-5 | 1.16e-5 | 2.17e-5 |
| \|ΔΠ\|/Π | 9.72e-3 | 5.00e-3 | 2.17e-3 | **9.34e-4** |

The reference calls the solver's own `bulk_coeffs_2d`, so this gate tests the **dynamics** — the
implicit relaxation, its back-reaction on the energy equation, the resulting entropy production —
not the coefficient formulas, which Gs already pins to ~1%. It also asserts dP/dμ = 0 (verified to
1e-12) since that is what makes T(e) a one-dimensional inversion.

### P9: the 19% is now decomposed, and only one third of it was mine

The decisive measurement is my Cooper–Frye against **Fluidum's own `spectra_internal`, on one and
the same surface**:

| | dN_π/dy |
|---|---|
| my Cooper–Frye | 1189.89 |
| Fluidum's | 1182.43 |
| **ratio** | **1.0063** |

**The two agree to 0.63%.** So the CF conventions — statistics, feed-down factor 2.0×2.5, phase
space, (2πħc)³ — are right, and the 19% is not there. (The sign of mine is negative: my surface
normal orientation is the opposite convention. Magnitudes are what is compared.)

That splits the gap three ways:

| piece | ratio | |
|---|---|---|
| CF conventions | 1.0063 | 0.6% — correct |
| my 2-D τ_fo(x,y) surface vs the 1-D contour | 0.914 | **8.6% — a defect of mine** |
| IC + solver | 0.883 | 11.7% — a genuine setup difference |

and 0.914 × 0.883 × 1.006 = 0.812, the observed ratio.

**🔴 The surface defect, fixed.** `pion_yield` skipped any cell whose neighbours had not frozen, to
avoid a NaN gradient. That discards the **outer rim of the surface**, which is exactly where dΣ is
largest (nearly parallel to the time axis). Replaced by one-sided differences at the rim: the yield
goes 1087.4 → **1171.3** at N = 200 and **1183.9** at N = 300, i.e. **−0.50%** against the contour,
from −8.6%.

**The remaining 11.7% is not an error.** `TemperatureCalibration.jl` runs Pb+Pb through
`:fluidum`, and `_run_fluidum` **floors the initial condition at T_fo** — the whole cold tail sits
*on* the isotherm, giving a materially larger fireball, and Fluidum's own tracker (not an isotherm
scan) defines the edge, because against a constant-T_fo tail the isotherm is not well defined. The
FiVo path floors far below T_fo. So my axisymmetric "control" was never the same physical setup as
the calibration. **A 2-D re-derivation of `Norm` would legitimately differ**, and the size of that
difference is the ~12% above, not 19%.

⚠ **A τ-sampling bias remains in the absolute yield.** Refining dτ from 0.05 to 0.02 at N = 300
moves the yield 1183.9 → 1172.9 (−0.9%): a cell is recorded as frozen at the first *sampled* τ below
T_fo, so coarse sampling places the surface systematically late. The comparison above is at matched
dτ = 0.05 on both sides, so it is like-for-like; the dτ → 0 limit needs both refined together.

### Where the sectors stand now

| sector | best quantitative check | agreement |
|---|---|---|
| ideal | Bjorken, Gubser (2nd order), c_s | 1e-4 … 0.05% |
| shear | viscous Gubser (2nd order), π vs IS target, attenuation | 0.02% … 0.3% |
| **bulk** | **G0b nonlinear at \|Π\|/P = 0.6**, plus Gs attenuation | **0.09% … 0.75%** |
| charge | Fick on uniform T; 1-D limit exact | 2% |

⚠ **Charge is the one still standing where it was.** It has no analytic 2-D test *with flow*: the
Fick check is on a static uniform background (2%), and the 1-D reduction is exact but 1-D. The
concrete next step is Gubser advection — with κ = 0 the charge obeys ∂_μ(nu^μ) = 0 on a known
non-trivial 2-D flow, so n̂ ∝ 1/cosh²ρ is an exact reference for the transport at second order.

---

## 6v. G3g — charge transport on Gubser flow, and a charge sink in the vacuum cut

The charge sector's only quantitative checks were Fick diffusion on a **static uniform** background
(2%) and an exact but **one-dimensional** reduction. Neither exercised charge transport on a
non-trivial 2-D flow. `test_charge_gubser2d.jl` does.

### The exact statement

With κ = 0 the charge obeys ∂_μ(nu^μ) = 0. For ideal conformal flow the entropy obeys the same
equation and s = aT³, so **n/T³ is conserved exactly along the flow**. Seed it uniform and it must
stay uniform in space and constant in time, on a background reaching u^r ≈ 2 (v ≈ 0.89). That is
sharper than comparing n to a profile: a single number that must not move, with nothing to fit.

The seeding is closed form — for the Boltzmann tracer n(T,μ)/n(T,0) = e^α, so
**α = log(n_target / n(T, μ=0))**, needing no knowledge of A(T). Verified exact to 1e-12.

### Result, τ = 1 → 2

| N | dx | invariant (r≤2) | L2(n) (r≤2) | x↔y | ΔQ/Q |
|---|---|---|---|---|---|
| 100 | 0.280 | 8.967e-3 | 1.545e-2 | 7.3e-15 | 9.99e-5 |
| 200 | 0.140 | 3.436e-3 | 2.833e-3 | 9.2e-15 | 9.95e-5 |
| 400 | 0.070 | **1.905e-3** | **2.216e-3** | 1.1e-14 | 9.99e-5 |

**Sub-percent (0.19%) on the invariant**, x↔y at round-off. ⚠ **Not second order** — the invariant
converges at ~0.85 and L2 at ~0.35, toward a floor near 2e-3. The gate asserts a bound and continued
improvement rather than claiming an order it does not have. Why the charge advection is only ~first
order where T and u are second order is **not established** and is the obvious next question.

### 🔴 The vacuum cut is a charge sink

ΔQ/Q came out at **6.1e-3** and was independent of resolution *and* of box size (10 vs 14 fm) — which
rules out both discretisation and boundary outflow, and was what forced the question. Scanning the
cut, everything else fixed:

| T_vac_cut | 0.05 (production) | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|
| ΔQ/Q | **6.18e-3** | 4.01e-4 | 9.95e-5 | 9.95e-5 |
| core invariant | 3.436e-3 | 3.436e-3 | 3.436e-3 | 3.436e-3 |
| isotherm radius at τ=2 | **7.9 fm** | 15.4 fm | 25.7 fm | 43.2 fm |

Gubser's tail crosses T = 0.05 at r = 7.9 fm, **inside** the box, and every cell beyond is zeroed —
taking its charge. The loss falls 62× and saturates once the cut radius leaves the box, while the
core invariant does not move at all: the two effects separate cleanly, and the residual 1e-4 is
genuine outflow.

**Charge is lost wherever the fluid is colder than T_vac_cut.** On the production IC that is small
(ΔQ/Q ≈ 8e-6) only because its cold tail carries almost no charm — it is a property of how much
charge sits below the cut, not of the scheme. Worth stating before quoting charm conservation on a
configuration with a cold, charge-carrying tail.

### Sector status after this pass

| sector | best quantitative check | agreement |
|---|---|---|
| ideal | Bjorken, Gubser (2nd order), c_s | 1e-4 … 0.05% |
| shear | viscous Gubser (2nd order), π vs IS target, attenuation | 0.02% … 0.3% |
| bulk | **G0b nonlinear at \|Π\|/P = 0.6**, Gs attenuation | 0.09% … 0.75% |
| charge | **G3g advection on Gubser**, Fick on uniform T | **0.19%** … 2% |

All four are now sub-percent somewhere, and the charge sector has a 2-D test with flow for the first
time. What is still missing for charge: a test of **diffusion** on a flowing background — G3g has
κ = 0 by construction, because a non-zero κ destroys the exact solution (α = log(C T³/n(T,0)) is not
spatially uniform, so ∇α ≠ 0).

---

## 6w. Diffusion WITH flow — the configuration the charm actually lives in

G3g has κ = 0; the Fick test has κ > 0 but u = 0. The physical case has
T(x,y,τ), u^x, u^y and κ all non-zero at once, and nothing tested that.

### An exact statement is still available: uniformly boosted background

Uniform fields stay uniform (every ∂_i vanishes, so the Milne equations reduce to ODEs in τ), and
charm is a pure tracer with dP/dμ = 0, so a charge blob cannot back-react. The background is
therefore exact at any boost. Diffusion is a rest-frame process, and for the direction **transverse**
to the boost there is no length contraction either, so the lab-frame spreading rate is slowed by time
dilation alone:

    d⟨y²⟩/dτ_lab = 2 D_s / γ ,   and the RATIO to the unboosted run must be exactly 1/γ.

Nothing about D_s, the EOS or the units survives in that ratio. It isolates the LRF projection of
ν^i = −κ(∂_iα + u^i Dα) and the relativistic kinematics.

| u₀ | ⟨γ⟩ | d⟨y²⟩/dτ | measured ratio | 1/⟨γ⟩ | dev | ⟨1/γ⟩ | dev |
|---|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 0.156674 | — | — | — | — | — |
| 0.50 | 1.1245 | 0.139587 | 0.89094 | 0.88931 | +0.18% | 0.88932 | +0.18% |
| 1.00 | 1.4365 | 0.109572 | 0.69936 | 0.69614 | +0.46% | 0.69620 | +0.45% |
| 1.50 | 1.8446 | 0.085498 | 0.54570 | 0.54212 | +0.66% | 0.54223 | +0.64% |
| 2.00 | 2.2982 | 0.068696 | 0.43846 | 0.43513 | +0.77% | 0.43525 | +0.74% |

**Sub-percent up to u^x = 2, i.e. v = 0.87.** The diffusion current is correctly projected into the
local rest frame on a flowing background.

⚠ I predicted the small residual trend was Jensen bias from using 1/⟨γ⟩ where ⟨1/γ⟩ is correct. The
measurement **refutes that**: the two agree to 1e-4, because γ barely moves over the run at τ₀ = 60.
The residual 0.2–0.8%, growing with γ, is real and **unexplained**.

### 🔴 The full configuration is NOT sub-percent at production resolution

With T(x,y,τ), u^x, u^y and κ all on there is no exact solution — α is not spatially uniform, so
∇α ≠ 0 destroys any analytic ansatz. The available statement is self-convergence of the charm field
on the physical single event (τ → 6, sampled on a fixed 60² grid):

| | value |
|---|---|
| ‖n(150) − n(600)‖ / ‖n(600)‖ | 3.563e-2 |
| ‖n(300) − n(600)‖ / ‖n(600)‖ | **1.625e-2** |
| observed order | **1.13** |

**At the production N = 300 the charm field carries ~1.6% error, and it converges at first order** —
so sub-percent needs N ≳ 600, not 300. Charge conservation is fine throughout (ΔQ/Q = 1.1e-5 →
4.5e-6); it is the *shape* of n that is only first-order accurate.

### The blocker, stated plainly

Charge transport converges at ~first order in **two independent tests** — the G3g invariant at order
0.85 with κ = 0 on Gubser, and the full-physics charm field at 1.13 — while T and u converge at 2.00
in the very same runs. The cause is **not identified**. It is not the diffusion-current advection
(G3g has κ = 0), not charge conservation (exact to 1e-5), and not the vacuum ramp (the G3g core is
far above it).

Until that is found, the honest position is: **the bulk fields are second-order and sub-percent; the
charm field is first-order and ~1.6% at production resolution.** Anything quoted from n(x,y) inherits
that, and running at N = 600 buys a factor of two rather than the factor of four a second-order
scheme would give.

