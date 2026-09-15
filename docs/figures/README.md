# `docs/figures/` — figures the front README shows that are produced OUTSIDE this repository

Everything here is a **copy**. FiVoHydro is developed as a submodule of a private research
repository, and the cross-code comparison against a second, independent hydrodynamics code
(Fluidum.jl) lives on that side — this package cannot reach it, and a reader of the public
repository cannot either. So the figures that make the cross-check claims checkable are copied in,
with their provenance and the command that regenerates them.

⚠ **They are copies, so they can go stale.** If a number in `README.md` §"Cross-checks" disagrees
with a figure here, the figure is the older thing. Regenerate, then re-copy.

| file | produced by | shows |
|---|---|---|
| `crosscheck_1p1d.png` | `FiVoFluidumComparison/plot_examples_1p1d.jl` → `figures1d/cmp1d_examples.png` | FiVo vs Fluidum in 1+1D on the worked-example configurations: Bjorken against one closed form, the operator-split ladder, the fireball profiles, the three-sector convergence ladder, the charm current |
| `crosscheck_self_convergence.png` | `FiVoFluidumComparison/plot_self_convergence.jl` → `figures2d/self_convergence.png` | each code against its OWN Richardson limit at 384² on a real Pb+Pb event — which code is closer to the continuum. Hollow markers are points where the two limits do not meet and no ranking may be read off |
| `crosscheck_2p1d_convergence.png` | `FiVoFluidumComparison/plot_2p1d_comparison.jl` → `figures2d/g_conv_T.png` | the 2+1D code-to-code difference in T against resolution, three initial conditions × three sectors |

Regenerating them (inside the research repository):

```sh
julia --project=Julia Julia/Projects/FiVoFluidumComparison/compare_examples_1p1d.jl   # the numbers
julia --project=Julia Julia/Projects/FiVoFluidumComparison/plot_examples_1p1d.jl      # → crosscheck_1p1d
julia --project=Julia Julia/Projects/FiVoFluidumComparison/plot_self_convergence.jl   # → crosscheck_self_convergence
julia --project=Julia Julia/Projects/FiVoFluidumComparison/plot_2p1d_comparison.jl    # → crosscheck_2p1d_convergence
```

Every other figure the documentation shows is produced **by this repository**, by the example it sits
next to (`examples1d/figures/`, `examples2d/figures/`), and needs no copying.
