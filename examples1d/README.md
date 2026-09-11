# FiVo 1+1D — examples

Runnable, each with a figure in `figures/`. From the repository root:

```sh
julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/examples1d/01_first_run.jl
```

| | what it shows | time |
|---|---|---|
| 01 first run | the library interface in one screen: `build_model_1d`, `show_equations`, `run_sim_1d!`, `fields_1d`, on a viscous, diffusing fireball with the consistent first moment | 15 s |
| 02 switching terms off | the `terms` interface in both 1+1D solvers. A fireball in the bulk solver with its first-moment terms removed by ingredient: `:homogeneous` ≡ shipped bit for bit, and the ∇T and inertial terms cancel as a pair. Then its T(τ, r), u^r(τ, r) as the frozen background of the charm IS2 solver, with the second moment's terms switched the same way | 3 min |
| 03 analytic benchmarks | the solvers against known results, as pictures: ideal Bjorken (orders 2 and 3), viscous Bjorken with shear, bulk and the λ couplings vs the DNMR ODEs, viscous Gubser vs its semi-analytic solution at three resolutions, and the charm diffusion mode through the bulk and IS2 solvers vs its closed ODE | ~3 min |

The same checks, asserted rather than plotted, are the 1+1D validation ladder `test/run1d_gates.jl`
(`README.md` §5). The 2+1D examples are in [`../examples2d/`](../examples2d/README.md). `run_suite.jl` discovers this
directory by its prefix and runs everything here.
