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
| G2 | Gubser ideal + viscous | analytic. Note Gubser is *also* azimuthally symmetric — tests the 2-D machinery, not azimuthal structure |
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

**1. The only comparison against an independent trusted solver is IDEAL.** Gate G4 runs the 1-D
production reference with `enable_diff = enable_shear = enable_bulk = false`
(`test_reproduction2d.jl:56`). The dissipative sectors are validated by *derivation* checks (the NS
targets reduce exactly to the 1-D forms: `π_NS^{yy}` at **0.0**, `ν_NS` at **6.2e-16**), by viscous
Bjorken against an independently integrated ODE, and by internal consistency (conservation,
symmetry, convergence) — but they have **never been compared against the 1-D solver on the real IC
with dissipation on**. That comparison is straightforward to run and is the single highest-value
thing still missing. Tracked as **P6**.

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
| **G5** | **production IC, all four sectors, late time**: N=150/τ=8 primfail **40**, x↔y **5.0e-12**; N=300/τ=4 x↔y **6.6e-13**, `max\|u\|` 1.20, min(P+Π) > 0, dQ 9.3e-6 | **PASS** |
| **G6** | **non-axisymmetric IC** (production profile on an elliptical radius): ε₂=0 control anisotropy **identically 0**, anisotropy **converges to 0.08%** over N=100→300, reflection symmetry 1.5e-10 at N≤200 | **PASS** |
| **INT** | integration, all 11 fields on an ELLIPTIC (ε₂≠0) IC — the case 1-D cannot represent: runs to τ=6, primfail **0**, `π^xy` ≠ 0, momentum anisotropy **+0.186** | **PASS** |
| **G3** | charge: τ_n copies **bit-identical**; `ν_NS` vs 1-D 6.2e-16; **closed-domain charge drift 9.1e-16**; 0/2104 cells up-gradient; x↔y asymmetry **0.0 exactly** | **PASS** |
| P2 | shear + bulk sector, NS targets, IS relaxation, constraint projection | not started |
| P3 | charge sector `(α, ν^x, ν^y)`, incl. the D2 covariant-drive derivation | not started |
| P4 | G4 reproduction gate on the production IC | **done** — `test_reproduction2d.jl`; runs BOTH solvers in-process from the 1-D's own `load_initial_interpolants`, so an IC mismatch is impossible |
| P5 | robustness: floors, S–E bound, ν admissibility bound, vacuum cut, vacuum ramp, staged recovery, bulk positivity | **done** — all four sectors run the production IC to τ=8 at N=150 and N=300 with x↔y ≤ 5e-12; gated by G5 | — full 11-field elliptic IC runs clean: `ok`, **primfail 0**, shear constraint 0.0, 148 steps / 3.8 s on 96². No MOOD, floors or vacuum ramps yet; those are the remaining P5 work |
| F1 | un-averaged 2-D IC (keep `evt(x,y)`, b≠0) — *separate, smaller job* | not started |
| D2 | derive the 2-D covariant NS drive `∇^⟨i⟩α`; gate it | **CLOSED** — `ν_NS^i = -κ(∂_iα + u^i Dα)`; setting `u^y=0` gives `-κ[(u^τ)²∂_rα + u^ru^τ∂_τα]`, the production expression exactly (6.2e-16, `test_charge2d.jl` G3b) |
| D5 | 1-D `Dy = (∂_τu^r + u^r∂_ru^r)/u^τ` in the ν projected-derivative term; with `y=asinh u^r` the comoving derivative is `∂_τu^r + (u^r/u^τ)∂_ru^r` — the `∂_τ` piece should not carry the `1/u^τ` | **open, ALGEBRA ONLY** — magnitude UNMEASURED, production untouched. It is a second-order damping coefficient in the charm relaxation, so worth measuring before it is dismissed |
| **D6** | charge sector unstable on the production IC — degenerate charge row at the fluid-vacuum interface | **CLOSED** by the density-gated vacuum ramp (x↔y 1.00 → 1.3e-13); see §6e |
| **D7** | with ALL sectors at N=300, `\|π^xy\|` grew to 1e3 by τ=4 | **CLOSED** — a charge-row failure was discarding the converged hydro block (68.2% of failures), plus a negative total pressure; see §6e |
| P6 | compare the DISSIPATIVE sectors against the 1-D solver on the production IC — the biggest remaining validation gap | **open** |
| P7 | wire MOOD (`mark_bad_2d!`/`expand_bad_2d!` are written but never called) | **open** |
| P8 | reflection symmetry ~1.8e-3 on a deformed IC at N=300 only; not localised | **open** |
| D4 | `eos_entropy`'s `-μn/T` term is not thermodynamic for `LatticeHRGEOS`; it feeds η via `local_thermo` (~1.4e-4 relative) | **open, low priority** — see §6b |
| D3 | measure whether the 1-D production primrec shares the row-scaling weakness | **CLOSED, negative** — see §6a. Mechanism real (220× at the dilute edge), magnitude 3.3e-12, no manuscript affected, production untouched |

### Cost note

Production 1-D is `Nr=800`, `rmax=20`. The same `dr` in 2-D over `[-20,20]²` is `800² = 6.4e5` cells
with 11 fields and a 2-D CFL. Runtime goes from minutes to hours per run. Grid-resolution choice is
a live question — G4 should be run at matched `dr` first, then at whatever the production 2-D grid
turns out to be.
