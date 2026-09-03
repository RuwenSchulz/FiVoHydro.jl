# FiVoHydro 2+1D — worked examples

Five runnable setups for `main2D.jl` / `src2d/`, each a few minutes, each producing a figure in
`figures/`. They are written to be **copied and edited**, and every trap this solver has actually
shipped is called out in a comment where it would bite.

```sh
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/01_first_run.jl
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/02_elliptic_flow.jl
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/03_choosing_a_resolution.jl
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/04_fluctuating_event.jl
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples2d/05_dissipative_sectors.jl
```

| | what it shows |
|---|---|
| **01** the first run | the whole API — grid, model, state, IC, driver, readout — on a transversely uniform state, which must integrate the 0+1D DNMR equations; and why the agreement is 1e-3 rather than 1e-13 |
| **02** elliptic flow | ε₂ → momentum anisotropy, the reason to run 2+1D at all, with the sector ladder run from one initial condition and the ε₂ = 0 control |
| **03** choosing a resolution | the three-grid Richardson study, and why a field, an observable and a conserved quantity converge at three different rates |
| **04** a fluctuating event | v₃ from lumps, why it needs one event at a time, and harmonics about the participant plane rather than the grid axes |
| **05** the dissipative sectors | what each sector changes and costs, how big \|π\|/P really gets, and whether the regulators are inert |

These are **examples, not gates**. The validation ladder is `test/run2d_gates.jl` — 18 gates, all
passing as of 2026-09-03 — and the physics behind each of them is `TWOD_PROGRAM.md`. Run the ladder
before trusting a change; run these to learn the API or to start a new study.

```sh
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run2d_gates.jl
```

## The four things that bite hardest here

1. **`finalize_ic!` is not optional on a hand-built IC.** `set_cell!` writes primitives into
   conserved variables cell by cell and applies no floors; a profile with a vacuum tail starts below
   the energy floor in its outer cells and the run dies within a few steps.
   `initialize_uniform!` and `initialize_from_radial!` call it for you — nothing else does.

2. **Julia soft scope kills accumulators, and it killed one of these examples while it was being
   written.** `prev = e` inside a *top-level* `for` creates a new local every iteration. In an
   example you get an `UndefVarError`; in a PASS/FAIL gate you get a silent PASS. Put the loop in a
   function. (`CLAUDE.md`, trap #1.)

3. **The repo's own initial conditions carry no ε₂ and no ε₃.** `data/initial_profiles_physical.csv`
   is `r, T, α` on 1002 radial points because the IC builder φ-averages every binary collision
   *upstream* of FiVo (`TWOD_PROGRAM.md` §0). Running them in 2+1D is a validation exercise — the
   2-D answer must reproduce the 1-D one — and yields zero elliptic flow. Examples 02 and 04 build
   their own deformed and lumpy initial conditions for that reason.

4. **The scheme is second order in the ideal sector and first order once anything dissipative is
   on.** Advection is SSPRK2 (gate G0 measures exactly 2.00); the dissipative sectors are relaxed by
   an operator split once per accepted step. Example 01 measures the crossover: halving `CFLτ`
   halves the error, not quarters it.

## And two that are specific to reading the numbers

**Measure the response, not the observable.** ε₂ → anisotropy is a transfer function. Quoting the
output alone hides whether a change moved the medium or moved the initial condition, so every
comparison here divides by the input and runs the ε₂ = 0 control that must return identically zero.

**A statistic can improve because its denominator grew.** Refining the grid changes the number of
cells in every average; report the quantity and its scale together. The same failure mode cost this
programme a headline number once already (`CLAUDE.md`).
