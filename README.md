# FiVoHydro
Finite Volume O(2)-Hydrodynamics Implementation in 1+1D

## IC diagnostics

Runs a solver-independent initial-condition analysis and writes CSV outputs (per-cell fields + first/second derivatives) under `ic_diagnostics/` by default.

- Suite mode (recommended): `julia --project=. --threads=auto bench/ic_diagnostics.jl --suite`
- Single case from CSV: `julia --project=. --threads=auto bench/ic_diagnostics.jl --init_csv=data/initial_profiles_physical.csv --Nr=600 --rmax=22 --tau0=0.4`
- Analytic IC: `julia --project=. --threads=auto bench/ic_diagnostics.jl --analytic`

Optional terminal output:

- Summary table (default on): pass `--no-table` to disable
- Field summary table: `--field-table`
- ASCII derivative “plots” (sparklines): `--plots --plot_kind=both --plot_width=80 --plot_fields=T,mu,alpha,phi`

PNG output (saved under `ic_diagnostics/<case>/plots/`):

- Enable: `--png`
- Control what gets plotted:
	- `--png_fields=T,mu,alpha,phi`
	- `--png_kind=both` (or `d1`, `d2`)
	- `--png_w=1100 --png_h=900`

To match the solver configuration, enable the corresponding physics in the diagnostics run:

- Charge diffusion: `--diff`
- Shear viscosity: `--shear`
- Bulk viscosity: `--bulk`

## Gubser-flow validation

Runs an end-to-end ideal-hydro evolution initialized from the analytic ideal Gubser solution and
writes both snapshots and an error time series (relative L2 norms) against the analytic solution.

- Run: `julia --project=. --threads=auto bench/gubser_validation.jl --outdir=snapshots/gubser --Nr=400 --rmax=15 --tau0=0.6 --taufinal=4.0 --q=1.0 --Tc0=0.5`
- Outputs:
	- Snapshots: `snapshot_tau_*.csv`
	- Errors: `gubser_errors.csv` (columns `tau, err_T, err_e, err_ur`)

Notes:
- This validation uses a conformal, baryonless EOS by default: `ConformalHQEOS(g_eff=40, m_hq=0, g_hq=0)`.
- Primitive recovery now supports charge-less states (`Dtau==0`), which is required for baryonless Gubser flow.

### Plotting

Generate quick PNG plots (error vs τ, plus numeric-vs-analytic profiles at a chosen τ):

- `julia --project=. --threads=auto bench/gubser_plot.jl --indir=snapshots/gubser --tau=1.0`

This writes into `--indir/plots` by default.
