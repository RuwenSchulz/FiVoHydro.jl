#!/usr/bin/env bash
set -euo pipefail

# Interactive wrapper for bench/ic_diagnostics.jl
#
# Prompts:
#   - shear? (yes/no)
#   - bulk?  (yes/no)
#   - diff?  (yes/no)
# PNG output is always enabled.
#
# You can override any default via environment variables, e.g.
#   OUTDIR=ic_diagnostics_custom NR=800 RMAX=25 TAU0=0.4 ./run_ic_diagnostics.sh
#
# Supported env vars (mirrors bench/ic_diagnostics.jl flags):
#   MODE:        suite | single | analytic            (default: suite)
#   OUTDIR:      output directory                    (default: ic_diagnostics_run_YYYYmmdd_HHMMSS)
#   NR:          grid cells                          (default: 400)
#   RMAX:        maximum radius                      (default: 22.0)
#   NGHOST:      ghost cells per side                (default: 3)
#   TAU0:        initial proper time                 (default: 0.4)
#   TAUFINAL:    final proper time for solver reuse  (default: 5.0)
#   CFL:         hyperbolic CFL                      (default: 0.2)
#   CFLTAU:      geometric CFL in tau                (default: 0.05)
#   EOS:         EOS kind                            (default: latticehrg)
#   INIT_CSV:    CSV IC file (single/suite uses it)  (default: data/initial_profiles_physical.csv)
#   AUTO_MATCH_CSV_GRID: 1/0  (infer NR/RMAX/NGHOST from CSV if not set) (default: 1)
#   MATCH_IC_GRID: 1/0  (auto-pick NR to match target IC_DR)            (default: 0)
#   IC_DR:       target CSV dr when MATCH_IC_GRID=1   (default: 0.028)
#   FUGACITY:    alpha | lambda                      (default: alpha)
#   TAPER_WIDTH: taper width for single run          (default: 0.0)
#   PRINT_TABLE: 1/0                                 (default: 1)
#   FIELD_TABLE: 1/0                                 (default: 0)
#   PLOTS:       1/0                                 (default: 0)
#   PLOT_WIDTH:  ASCII plot width                    (default: 90)
#   PLOT_KIND:   both | d1 | d2                      (default: both)
#   PLOT_FIELDS: comma list                          (default: T,mu,alpha,phi)
#   PNG_KIND:    both | d1 | d2                      (default: both)
#   PNG_FIELDS:  comma list                          (default: (same as PLOT_FIELDS))
#   PNG_W:       PNG width in px                     (default: 1100)
#   PNG_H:       PNG height in px                    (default: 900)
#   POSTPLOT:    1/0  (generate extra diagnosis PNGs) (default: 1)
#   JULIA:       julia executable                    (default: julia)
#   THREADS:     julia threads setting               (default: auto)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ask_yn() {
  # Usage: ask_yn "Question?" "Y"|"N"; returns 0=yes, 1=no
  local prompt="$1"
  local default="${2:-N}"
  local reply

  while true; do
    if [[ "$default" == "Y" ]]; then
      read -r -p "$prompt [Y/n] " reply || true
      reply="${reply:-Y}"
    else
      read -r -p "$prompt [y/N] " reply || true
      reply="${reply:-N}"
    fi

    case "${reply,,}" in
      y|yes) return 0 ;;
      n|no)  return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

print_run_summary() {
  local diag_status="$1"
  local both_local="${both:-0}"
  local interp_modes=""
  local diag_dirs=""
  local hydro_outdir_abs

  hydro_outdir_abs="$(python3 - <<'PY' "$ROOT_DIR" "$HYDRO_OUTDIR"
import os, sys
root = sys.argv[1]
path = sys.argv[2]
if os.path.isabs(path):
    print(os.path.normpath(path))
else:
    print(os.path.normpath(os.path.join(root, path)))
PY
)"

  if [[ "$RUN_LINEAR" == "1" ]]; then
    interp_modes="linear"
    if [[ "$diag_status" == "completed" && "$both_local" == "1" ]]; then
      diag_dirs+="  - ${OUTDIR}_linear"$'\n'
    elif [[ "$diag_status" == "completed" ]]; then
      diag_dirs+="  - ${OUTDIR}"$'\n'
    fi
  fi
  if [[ "$RUN_CUBIC" == "1" ]]; then
    if [[ -n "$interp_modes" ]]; then
      interp_modes+=", "
    fi
    interp_modes+="cubic"
    if [[ "$diag_status" == "completed" && "$both_local" == "1" ]]; then
      diag_dirs+="  - ${OUTDIR}_cubic"$'\n'
    elif [[ "$diag_status" == "completed" && "$RUN_LINEAR" != "1" ]]; then
      diag_dirs+="  - ${OUTDIR}"$'\n'
    fi
  fi
  if [[ -z "$interp_modes" ]]; then
    interp_modes="(none)"
  fi
  if [[ -z "$diag_dirs" ]]; then
    diag_dirs="  - (no diagnostics output)"
  else
    diag_dirs="${diag_dirs%$'\n'}"
  fi

  cat <<EOF

========================================
Effective IC diagnostic parameters
========================================
status          : $diag_status
mode            : $MODE
interp modes    : $interp_modes
outdir          : $OUTDIR
init_csv        : $INIT_CSV
hydro_outdir    : $hydro_outdir_abs
last_diag_env   : $LAST_DIAG_ENV

tau0            : $TAU0
tau_final       : $TAUFINAL
Emin            : $EMIN
chi             : $CHI

Nr              : $NR
nghost          : $NGHOST
Ntot            : $((NR + 2*NGHOST))
rmin            : $RMIN_ACTUAL
rmax            : $RMAX
rmax_data       : $RMAX_DATA_ACTUAL
dr              : $DR_ACTUAL

CFL             : $CFL
CFLtau          : $CFLTAU
dump_dt         : $DUMP_DT
EOS             : $EOS
fugacity        : $FUGACITY
taper_width     : $TAPER_WIDTH
interp default  : $INTERP
interp_dr       : ${INTERP_DR:-auto}
time integrator : $TIME_INTEGRATOR
auto_cfl        : $AUTO_CFL
cfl_safety      : $CFL_SAFETY
cfl_max         : $CFL_MAX
cfltau_max      : $CFLTAU_MAX
log_every       : $LOG_EVERY
log_corr_every  : $LOG_CORRECTIONS_EVERY

shear           : $enable_shear
bulk            : $enable_bulk
diff            : $enable_diff
postplot        : $POSTPLOT

diffusion params:
  DsT                 : $DS_T
  kappa_coeff         : ${KAPPA_COEFF:-auto}
  diffusion_drive     : $DIFFUSION_DRIVE
  tauN_coeff          : $TAU_N_COEFF
  deltaN_factor       : $DELTA_N_FACTOR
  diff_dt_coeff       : $DIFF_DT_COEFF
  nur_clip_factor     : $SOLVER_NUR_CLIP_FACTOR
  alpha_filter_eps    : $SOLVER_ALPHA_FILTER_EPS
  nur_filter_eps      : $SOLVER_NUR_FILTER_EPS
  alpha_smooth_len    : $SOLVER_ALPHA_SMOOTH_LEN
  nur_smooth_len      : $SOLVER_NUR_SMOOTH_LEN
  do_soft_project_nur : $DO_SOFT_PROJECT_NUR
  do_axis_project_nur : $DO_AXIS_PROJECT_NUR
  axis_project_nfit   : $AXIS_PROJECT_NFIT
  advect_nur          : $ADVECT_NUR
  relax_advect_nur    : $RELAX_ADVECT_NUR

viscosity params:
  eta_over_s               : $ETA_OVER_S
  zeta_over_s              : $ZETA_OVER_S
  tau_shear_coeff          : $TAU_SHEAR_COEFF
  tau_pi_coeff             : $TAU_PI_COEFF
  shear_dt_coeff           : $SHEAR_DT_COEFF
  bulk_dt_coeff            : $BULK_DT_COEFF
  delta_bulk_factor        : $DELTA_BULK_FACTOR
  delta_shear_factor       : $DELTA_SHEAR_FACTOR
  taupi_pi_factor          : $TAUPI_PI_FACTOR
  lambda_bulk_shear_factor : $LAMBDA_BULK_SHEAR_FACTOR
  lambda_shear_bulk_factor : $LAMBDA_SHEAR_BULK_FACTOR
  lambda_NN_factor         : $LAMBDA_NN_FACTOR
  visc_filter_eps          : $SOLVER_VISC_FILTER_EPS
  visc_smooth_len          : $SOLVER_VISC_SMOOTH_LEN
  Pi_clip_factor           : $SOLVER_PI_CLIP_FACTOR
  shear_clip_factor        : $SOLVER_SHEAR_CLIP_FACTOR
  advect_bulk_pi           : $ADVECT_BULK_PI
  relax_advect_bulk_pi     : $RELAX_ADVECT_BULK_PI
  advect_shear_pi          : $ADVECT_SHEAR_PI
  relax_advect_shear_pi    : $RELAX_ADVECT_SHEAR_PI

grid adaptation:
  expand_grid        : $EXPAND_GRID
  expand_factor      : $EXPAND_FACTOR
  expand_tail_cells  : $EXPAND_TAIL_CELLS
  expand_tail_frac   : $EXPAND_TAIL_FRAC
  expand_tail_abs_E  : $EXPAND_TAIL_ABS_E
  expand_min_cells   : $EXPAND_MIN_CELLS
  expand_max_nr      : $EXPAND_MAX_NR
  expand_max_rmax    : $EXPAND_MAX_RMAX
  expand_cooldown    : $EXPAND_COOLDOWN
  init_good_range    : $INIT_GOOD_RANGE
  init_good_min      : $INIT_GOOD_MIN_CELLS
  init_good_buffer   : $INIT_GOOD_BUFFER

repair / guards:
  hydro_repair_theta : $HYDRO_REPAIR_THETA
  hydro_srscale_margin : $HYDRO_SRSCALE_MARGIN

diagnostic dirs :
$diag_dirs
========================================
EOF
}

JULIA_BIN="${JULIA:-julia}"
THREADS="${THREADS:-auto}"

MODE_WAS_SET=0
[[ -n "${MODE+x}" ]] && MODE_WAS_SET=1
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-ic_diagnostics_run_${TS}}"
HYDRO_OUTDIR="${HYDRO_OUTDIR:-../../Julia/Plot/snapshots/FiVo}"

MODE="${MODE:-single}"

# Grid control
# If INIT_CSV is present, we default to matching its grid exactly (including
# the implied ghost-cell convention), unless NR/RMAX/NGHOST are explicitly set.
AUTO_MATCH_CSV_GRID="${AUTO_MATCH_CSV_GRID:-1}"

NR_WAS_SET=0
RMAX_WAS_SET=0
NGHOST_WAS_SET=0
[[ -n "${NR+x}" ]] && NR_WAS_SET=1
[[ -n "${RMAX+x}" ]] && RMAX_WAS_SET=1
[[ -n "${NGHOST+x}" ]] && NGHOST_WAS_SET=1

NR="${NR:-400}"
RMAX="${RMAX:-22.0}"
NGHOST="${NGHOST:-3}"
TAU0="${TAU0:-0.4}"
TAUFINAL="${TAUFINAL:-5.0}"
EMIN="${EMIN:-1e-20}"
CHI="${CHI:-0.99999999}"
CFL="${CFL:-0.2}"
CFLTAU="${CFLTAU:-0.05}"
EOS="${EOS:-latticehrg}"

INIT_CSV="${INIT_CSV:-data/initial_profiles_physical.csv}"
MATCH_IC_GRID="${MATCH_IC_GRID:-0}"
IC_DR="${IC_DR:-0.028}"
FUGACITY="${FUGACITY:-alpha}"
TAPER_WIDTH="${TAPER_WIDTH:-0.0}"

PRINT_TABLE="${PRINT_TABLE:-1}"
FIELD_TABLE="${FIELD_TABLE:-0}"
PLOTS="${PLOTS:-0}"

PLOT_WIDTH="${PLOT_WIDTH:-90}"
PLOT_KIND="${PLOT_KIND:-both}"
PLOT_FIELDS="${PLOT_FIELDS:-T,mu,alpha,phi}"

PNG_KIND="${PNG_KIND:-both}"
PNG_FIELDS="${PNG_FIELDS:-$PLOT_FIELDS}"
PNG_W="${PNG_W:-1100}"
PNG_H="${PNG_H:-900}"

# CSV interpolation control (relevant when INIT_CSV is used)
# Default: run a SINGLE diagnostic using linear interpolation.
# You can opt into cubic via `./run_ic_diagnostics.sh --cubic` (or INTERP=cubic).
INTERP="${INTERP:-linear}"          # linear | cubic
INTERP_DR="${INTERP_DR:-0.02}"      # spacing for cubic resampling (set empty to auto)

# Solver-side robustness defaults written into last_ic_diagnostics.env.
TIME_INTEGRATOR="${TIME_INTEGRATOR:-ssprk3}"
AUTO_CFL="${AUTO_CFL:-1}"
CFL_SAFETY="${CFL_SAFETY:-0.85}"
CFL_MAX="${CFL_MAX:-0.35}"
CFLTAU_MAX="${CFLTAU_MAX:-0.12}"

DUMP_DT="${DUMP_DT:-0.1}"
LOG_EVERY="${LOG_EVERY:-50}"
LOG_CORRECTIONS_EVERY="${LOG_CORRECTIONS_EVERY:-50}"

DS_T="${DS_T:-0.24}"
KAPPA_COEFF="${KAPPA_COEFF:-}"
DIFFUSION_DRIVE="${DIFFUSION_DRIVE:-n}"
TAU_N_COEFF="${TAU_N_COEFF:-1.0}"
DELTA_N_FACTOR="${DELTA_N_FACTOR:-0.0}"

DIFF_DT_COEFF="${DIFF_DT_COEFF:-0.01}"
SHEAR_DT_COEFF="${SHEAR_DT_COEFF:-0.01}"
BULK_DT_COEFF="${BULK_DT_COEFF:-0.01}"

NUR_CLIP_FACTOR="${NUR_CLIP_FACTOR:-0.98}"
ALPHA_FILTER_EPS="${ALPHA_FILTER_EPS:-0.01}"
NUR_FILTER_EPS="${NUR_FILTER_EPS:-0.0}"
ALPHA_SMOOTH_LEN="${ALPHA_SMOOTH_LEN:-0.0}"
NUR_SMOOTH_LEN="${NUR_SMOOTH_LEN:-0.0}"
DO_SOFT_PROJECT_NUR="${DO_SOFT_PROJECT_NUR:-0}"
DO_AXIS_PROJECT_NUR="${DO_AXIS_PROJECT_NUR:-0}"
AXIS_PROJECT_NFIT="${AXIS_PROJECT_NFIT:-2}"
ADVECT_NUR="${ADVECT_NUR:-0}"
RELAX_ADVECT_NUR="${RELAX_ADVECT_NUR:-1}"

VISC_FILTER_EPS="${VISC_FILTER_EPS:-0.01}"
VISC_SMOOTH_LEN="${VISC_SMOOTH_LEN:-0.0}"
PI_CLIP_FACTOR="${PI_CLIP_FACTOR:-0.35}"
SHEAR_CLIP_FACTOR="${SHEAR_CLIP_FACTOR:-0.35}"
DELTA_BULK_FACTOR="${DELTA_BULK_FACTOR:-0.0}"
DELTA_SHEAR_FACTOR="${DELTA_SHEAR_FACTOR:-0.0}"
TAUPI_PI_FACTOR="${TAUPI_PI_FACTOR:-0.0}"
LAMBDA_BULK_SHEAR_FACTOR="${LAMBDA_BULK_SHEAR_FACTOR:-0.0}"
LAMBDA_SHEAR_BULK_FACTOR="${LAMBDA_SHEAR_BULK_FACTOR:-0.0}"
LAMBDA_NN_FACTOR="${LAMBDA_NN_FACTOR:-0.0}"
ADVECT_BULK_PI="${ADVECT_BULK_PI:-0}"
RELAX_ADVECT_BULK_PI="${RELAX_ADVECT_BULK_PI:-1}"
ADVECT_SHEAR_PI="${ADVECT_SHEAR_PI:-0}"
RELAX_ADVECT_SHEAR_PI="${RELAX_ADVECT_SHEAR_PI:-1}"

EXPAND_GRID="${EXPAND_GRID:-1}"
EXPAND_FACTOR="${EXPAND_FACTOR:-1.5}"
EXPAND_TAIL_CELLS="${EXPAND_TAIL_CELLS:-8}"
EXPAND_TAIL_FRAC="${EXPAND_TAIL_FRAC:-1e-6}"
EXPAND_TAIL_ABS_E="${EXPAND_TAIL_ABS_E:-1e-18}"
EXPAND_MIN_CELLS="${EXPAND_MIN_CELLS:-64}"
EXPAND_MAX_NR="${EXPAND_MAX_NR:-200000}"
EXPAND_MAX_RMAX="${EXPAND_MAX_RMAX:-inf}"
EXPAND_COOLDOWN="${EXPAND_COOLDOWN:-25}"
INIT_GOOD_RANGE="${INIT_GOOD_RANGE:-1}"
INIT_GOOD_MIN_CELLS="${INIT_GOOD_MIN_CELLS:-32}"
INIT_GOOD_BUFFER="${INIT_GOOD_BUFFER:-4}"

ETA_OVER_S="${ETA_OVER_S:-0.1}"
ZETA_OVER_S="${ZETA_OVER_S:-0.1}"
TAU_SHEAR_COEFF="${TAU_SHEAR_COEFF:-0.2}"
TAU_PI_COEFF="${TAU_PI_COEFF:-15.0}"

HYDRO_REPAIR_THETA="${HYDRO_REPAIR_THETA:-1}"
HYDRO_SRSCALE_MARGIN="${HYDRO_SRSCALE_MARGIN:-0.995}"

POSTPLOT="${POSTPLOT:-1}"

# By default we run only the linear mode.
# To also run cubic (for comparisons), set e.g. ONLY_LINEAR=0 RUN_CUBIC=1.
ONLY_LINEAR="${ONLY_LINEAR:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_CUBIC="${RUN_CUBIC:-0}"

if [[ "$ONLY_LINEAR" == "1" ]]; then
  RUN_LINEAR=1
  RUN_CUBIC=0
fi

# Optional CLI overrides (keep env var support; CLI wins).
#   --linear          -> run only linear
#   --cubic           -> run only cubic
#   --both-interp     -> run linear + cubic (separate output dirs)
#   --interp=linear|cubic (alias)
for arg in "$@"; do
  case "$arg" in
    --linear)
      ONLY_LINEAR=0
      RUN_LINEAR=1
      RUN_CUBIC=0
      INTERP="linear"
      ;;
    --cubic)
      ONLY_LINEAR=0
      RUN_LINEAR=0
      RUN_CUBIC=1
      INTERP="cubic"
      ;;
    --both-interp)
      ONLY_LINEAR=0
      RUN_LINEAR=1
      RUN_CUBIC=1
      ;;
    --interp=*)
      v="${arg#*=}"
      v="${v,,}"
      if [[ "$v" == "cubic" ]]; then
        ONLY_LINEAR=0
        RUN_LINEAR=0
        RUN_CUBIC=1
        INTERP="cubic"
      else
        ONLY_LINEAR=0
        RUN_LINEAR=1
        RUN_CUBIC=0
        INTERP="linear"
      fi
      ;;
  esac
done

csv_grid_info() {
  local csv_path="$1"
  python3 - <<'PY' "$csv_path"
import csv, sys, math
path = sys.argv[1]
rs=[]
with open(path, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        rs.append(float(row['r']))
N=len(rs)
if N < 2:
    raise SystemExit("CSV has <2 rows")
rmin=rs[0]
rmax=rs[-1]
dr=rs[1]-rs[0]
if not (math.isfinite(dr) and dr>0):
    raise SystemExit("Invalid dr")
print(f"{N} {rmin:.17g} {rmax:.17g} {dr:.17g}")
PY
}

if [[ "$AUTO_MATCH_CSV_GRID" == "1" && "${MODE,,}" != "analytic" && -n "$INIT_CSV" && -f "$INIT_CSV" ]]; then
  if [[ "$NR_WAS_SET" == "0" && "$RMAX_WAS_SET" == "0" && "$NGHOST_WAS_SET" == "0" ]]; then
    read -r CSV_N CSV_RMIN CSV_RMAX CSV_DR < <(csv_grid_info "$INIT_CSV")
    # Infer ghost-cell count per side from rmin and dr.
    # For cell-centered grids, rmin = -(nghost - 0.5)*dr  =>  nghost = -rmin/dr + 0.5
    NG_EST=$(awk -v rmin="$CSV_RMIN" -v dr="$CSV_DR" 'BEGIN{ x = (-rmin/dr) + 0.5; printf "%d", int(x + 0.5) }')
    if [[ "$NG_EST" -lt 0 ]]; then NG_EST=0; fi
    NR_EST=$((CSV_N - 2*NG_EST))
    # Physical rmax is at the last interior cell center: rmax_phys = rmax_data - (nghost - 0.5)*dr
    RMAX_EST=$(awk -v rmax="$CSV_RMAX" -v dr="$CSV_DR" -v ng="$NG_EST" 'BEGIN{ printf "%.12g", rmax - (ng-0.5)*dr }')
    if [[ "$NR_EST" -ge 10 ]]; then
      NR="$NR_EST"
      NGHOST="$NG_EST"
      RMAX="$RMAX_EST"
      MATCH_IC_GRID=0
      IC_DR="$CSV_DR"
      echo "AUTO_MATCH_CSV_GRID=1 -> from $INIT_CSV: N=$CSV_N rmin=$CSV_RMIN rmax=$CSV_RMAX dr=$CSV_DR"
    else
      echo "AUTO_MATCH_CSV_GRID=1 but inferred NR=$NR_EST looks invalid; keeping NR=$NR RMAX=$RMAX NGHOST=$NGHOST" >&2
    fi
  fi
fi

# When comparing solver FV gradients to simple FD derivatives, the first/last
# few cells can differ a lot due to boundary stencils. This trims those cells
# from max-diff summaries (defaults to NGHOST).
DIAG_TRIM="${DIAG_TRIM:-$NGHOST}"

extra_flags=()

if [[ "$MATCH_IC_GRID" == "1" ]]; then
  # Choose NR so that dr = RMAX/NR is as close as possible to IC_DR.
  # This reduces interpolation artifacts when the CSV IC is on a uniform grid.
  NR_MATCH=$(awk -v rmax="$RMAX" -v dr="$IC_DR" 'BEGIN{ if (dr<=0) {print 0} else {printf "%d", int(rmax/dr + 0.5)} }')
  if [[ "$NR_MATCH" -ge 33 ]]; then
    NR="$NR_MATCH"
    DR_ACTUAL=$(awk -v rmax="$RMAX" -v nr="$NR" 'BEGIN{printf "%.12g", rmax/nr}')
    DR_RELERR=$(awk -v dra="$DR_ACTUAL" -v drt="$IC_DR" 'BEGIN{ if (drt==0) {print "NaN"} else {printf "%.3g", (dra-drt)/drt} }')
    echo "MATCH_IC_GRID=1 -> using NR=$NR so dr≈$DR_ACTUAL (target $IC_DR, relerr $DR_RELERR)"
  else
    echo "MATCH_IC_GRID=1 but computed NR=$NR_MATCH too small; keeping NR=$NR" >&2
  fi
fi

# Interactive tau overrides
read -r -p "Initial proper time [TAU0=$TAU0]: " TAU0_INPUT || true
if [[ -n "$TAU0_INPUT" ]]; then
  if [[ "$TAU0_INPUT" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v v="$TAU0_INPUT" 'BEGIN{exit !(v>0)}'; then
    TAU0="$TAU0_INPUT"
  else
    echo "Invalid input '$TAU0_INPUT' (need positive number); keeping TAU0=$TAU0" >&2
  fi
fi

read -r -p "Final proper time [TAUFINAL=$TAUFINAL]: " TAUFINAL_INPUT || true
if [[ -n "$TAUFINAL_INPUT" ]]; then
  if [[ "$TAUFINAL_INPUT" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v v="$TAUFINAL_INPUT" -v tau0="$TAU0" 'BEGIN{exit !(v>tau0)}'; then
    TAUFINAL="$TAUFINAL_INPUT"
  else
    echo "Invalid input '$TAUFINAL_INPUT' (need number > TAU0=$TAU0); keeping TAUFINAL=$TAUFINAL" >&2
  fi
fi

# Interactive grid-point override
read -r -p "Number of grid points [NR=$NR]: " NR_INPUT || true
if [[ -n "$NR_INPUT" ]]; then
  if [[ "$NR_INPUT" =~ ^[0-9]+$ ]] && [[ "$NR_INPUT" -ge 10 ]]; then
    NR="$NR_INPUT"
  else
    echo "Invalid input '$NR_INPUT' (need integer >= 10); keeping NR=$NR" >&2
  fi
fi

# Interactive rmax override
read -r -p "Maximum radius [RMAX=$RMAX]: " RMAX_INPUT || true
if [[ -n "$RMAX_INPUT" ]]; then
  # Accept integer or float (e.g. 25, 22.5, .5)
  if [[ "$RMAX_INPUT" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v v="$RMAX_INPUT" 'BEGIN{exit !(v>0)}'; then
    RMAX="$RMAX_INPUT"
  else
    echo "Invalid input '$RMAX_INPUT' (need positive number); keeping RMAX=$RMAX" >&2
  fi
fi

DR_ACTUAL=$(awk -v rmax="$RMAX" -v nr="$NR" 'BEGIN{ if (nr<=0) {print "NaN"} else {printf "%.12g", rmax/nr} }')
RMIN_ACTUAL=$(awk -v dr="$DR_ACTUAL" -v ng="$NGHOST" 'BEGIN{ printf "%.12g", -(ng-0.5)*dr }')
RMAX_DATA_ACTUAL=$(awk -v rmax="$RMAX" -v dr="$DR_ACTUAL" -v ng="$NGHOST" 'BEGIN{ printf "%.12g", rmax + (ng-0.5)*dr }')
echo "Grid: NR=$NR (physical), NGHOST=$NGHOST per side -> NTOT=$((NR + 2*NGHOST))"
echo "Grid extents (cell centers): rmin≈$RMIN_ACTUAL rmax≈$RMAX_DATA_ACTUAL (physical rmax=$RMAX, dr≈$DR_ACTUAL)"

# Mode selection
case "${MODE,,}" in
  suite)    extra_flags+=("--suite") ;;
  single)   ;; # default behavior is a single case
  analytic) extra_flags+=("--analytic") ;;
  *)
    echo "Unknown MODE='$MODE' (expected: suite|single|analytic)" >&2
    exit 2
    ;;
esac

# Always enable PNG output
extra_flags+=("--png")

# Interactive physics toggles
enable_shear=0
enable_bulk=0
enable_diff=0
if ask_yn "Enable shear viscosity?" "N"; then
  extra_flags+=("--shear")
  enable_shear=1
fi
if ask_yn "Enable bulk viscosity?" "N"; then
  extra_flags+=("--bulk")
  enable_bulk=1
fi
if ask_yn "Enable charge diffusion?" "N"; then
  extra_flags+=("--diff")
  enable_diff=1
fi

# Decide what interpolation to record for the solver (main.jl).
# This should reflect what was actually run here (linear-only vs cubic-only),
# or fall back to the user's INTERP when both are requested.
SOLVER_INTERP="${INTERP,,}"
SOLVER_INTERP_DR="${INTERP_DR}"
if [[ "$RUN_LINEAR" == "1" && "$RUN_CUBIC" == "0" ]]; then
  SOLVER_INTERP="linear"
  SOLVER_INTERP_DR=""
elif [[ "$RUN_CUBIC" == "1" && "$RUN_LINEAR" == "0" ]]; then
  SOLVER_INTERP="cubic"
  # keep SOLVER_INTERP_DR as-is (empty -> auto)
elif [[ "$RUN_LINEAR" == "1" && "$RUN_CUBIC" == "1" ]]; then
  # keep user's preference in INTERP (default: cubic)
  if [[ "$SOLVER_INTERP" != "cubic" ]]; then
    SOLVER_INTERP="linear"
    SOLVER_INTERP_DR=""
  fi
else
  # Shouldn't happen, but be safe.
  SOLVER_INTERP="linear"
  SOLVER_INTERP_DR=""
fi

SOLVER_NUR_CLIP_FACTOR="$NUR_CLIP_FACTOR"
SOLVER_ALPHA_FILTER_EPS="$ALPHA_FILTER_EPS"
SOLVER_NUR_FILTER_EPS="$NUR_FILTER_EPS"
SOLVER_ALPHA_SMOOTH_LEN="$ALPHA_SMOOTH_LEN"
SOLVER_NUR_SMOOTH_LEN="$NUR_SMOOTH_LEN"
if [[ "$enable_diff" != "1" ]]; then
  SOLVER_NUR_CLIP_FACTOR="-1.0"
  SOLVER_ALPHA_FILTER_EPS="0.0"
  SOLVER_NUR_FILTER_EPS="0.0"
  SOLVER_ALPHA_SMOOTH_LEN="0.0"
  SOLVER_NUR_SMOOTH_LEN="0.0"
fi

SOLVER_VISC_FILTER_EPS="$VISC_FILTER_EPS"
SOLVER_VISC_SMOOTH_LEN="$VISC_SMOOTH_LEN"
SOLVER_PI_CLIP_FACTOR="$PI_CLIP_FACTOR"
SOLVER_SHEAR_CLIP_FACTOR="$SHEAR_CLIP_FACTOR"
if [[ "$enable_bulk" != "1" ]]; then
  SOLVER_PI_CLIP_FACTOR="-1.0"
fi
if [[ "$enable_shear" != "1" ]]; then
  SOLVER_SHEAR_CLIP_FACTOR="-1.0"
fi
if [[ "$enable_shear" != "1" && "$enable_bulk" != "1" ]]; then
  SOLVER_VISC_FILTER_EPS="0.0"
  SOLVER_VISC_SMOOTH_LEN="0.0"
fi

# Persist effective settings for the solver (main.jl) to reuse.
# main.jl can read this file when USE_DIAG_SETTINGS=1.
LAST_DIAG_ENV="${LAST_DIAG_ENV:-last_ic_diagnostics.env}"
{
  echo "# Autogenerated by run_ic_diagnostics.sh on $(date -Iseconds)"
  echo "INIT_CSV=$INIT_CSV"
  echo "HYDRO_OUTDIR=$HYDRO_OUTDIR"
  echo "NR=$NR"
  echo "RMAX=$RMAX"
  echo "NGHOST=$NGHOST"
  echo "RMIN=$RMIN_ACTUAL"
  echo "DR=$DR_ACTUAL"
  echo "TAU0=$TAU0"
  echo "TAUFINAL=$TAUFINAL"
  echo "EMIN=$EMIN"
  echo "CHI=$CHI"
  echo "CFL=$CFL"
  echo "CFLTAU=$CFLTAU"
  echo "EOS=$EOS"
  echo "FUGACITY=$FUGACITY"
  echo "TAPER_WIDTH=$TAPER_WIDTH"
  echo "INTERP=$SOLVER_INTERP"
  echo "INTERP_DR=$SOLVER_INTERP_DR"
  echo "ENABLE_SHEAR=$enable_shear"
  echo "ENABLE_BULK=$enable_bulk"
  echo "ENABLE_DIFF=$enable_diff"
  echo "MODE=$MODE"
  echo "TIME_INTEGRATOR=$TIME_INTEGRATOR"
  echo "AUTO_CFL=$AUTO_CFL"
  echo "CFL_SAFETY=$CFL_SAFETY"
  echo "CFL_MAX=$CFL_MAX"
  echo "CFLTAU_MAX=$CFLTAU_MAX"
  echo "DUMP_DT=$DUMP_DT"
  echo "LOG_EVERY=$LOG_EVERY"
  echo "LOG_CORRECTIONS_EVERY=$LOG_CORRECTIONS_EVERY"
  echo "DS_T=$DS_T"
  echo "KAPPA_COEFF=$KAPPA_COEFF"
  echo "DIFFUSION_DRIVE=$DIFFUSION_DRIVE"
  echo "TAU_N_COEFF=$TAU_N_COEFF"
  echo "DELTA_N_FACTOR=$DELTA_N_FACTOR"
  echo "DIFF_DT_COEFF=$DIFF_DT_COEFF"
  echo "SHEAR_DT_COEFF=$SHEAR_DT_COEFF"
  echo "BULK_DT_COEFF=$BULK_DT_COEFF"
  echo "NUR_CLIP_FACTOR=$SOLVER_NUR_CLIP_FACTOR"
  echo "ALPHA_FILTER_EPS=$SOLVER_ALPHA_FILTER_EPS"
  echo "NUR_FILTER_EPS=$SOLVER_NUR_FILTER_EPS"
  echo "ALPHA_SMOOTH_LEN=$SOLVER_ALPHA_SMOOTH_LEN"
  echo "NUR_SMOOTH_LEN=$SOLVER_NUR_SMOOTH_LEN"
  echo "DO_SOFT_PROJECT_NUR=$DO_SOFT_PROJECT_NUR"
  echo "DO_AXIS_PROJECT_NUR=$DO_AXIS_PROJECT_NUR"
  echo "AXIS_PROJECT_NFIT=$AXIS_PROJECT_NFIT"
  echo "ADVECT_NUR=$ADVECT_NUR"
  echo "RELAX_ADVECT_NUR=$RELAX_ADVECT_NUR"
  echo "VISC_FILTER_EPS=$SOLVER_VISC_FILTER_EPS"
  echo "VISC_SMOOTH_LEN=$SOLVER_VISC_SMOOTH_LEN"
  echo "PI_CLIP_FACTOR=$SOLVER_PI_CLIP_FACTOR"
  echo "SHEAR_CLIP_FACTOR=$SOLVER_SHEAR_CLIP_FACTOR"
  echo "DELTA_BULK_FACTOR=$DELTA_BULK_FACTOR"
  echo "DELTA_SHEAR_FACTOR=$DELTA_SHEAR_FACTOR"
  echo "TAUPI_PI_FACTOR=$TAUPI_PI_FACTOR"
  echo "LAMBDA_BULK_SHEAR_FACTOR=$LAMBDA_BULK_SHEAR_FACTOR"
  echo "LAMBDA_SHEAR_BULK_FACTOR=$LAMBDA_SHEAR_BULK_FACTOR"
  echo "LAMBDA_NN_FACTOR=$LAMBDA_NN_FACTOR"
  echo "ADVECT_BULK_PI=$ADVECT_BULK_PI"
  echo "RELAX_ADVECT_BULK_PI=$RELAX_ADVECT_BULK_PI"
  echo "ADVECT_SHEAR_PI=$ADVECT_SHEAR_PI"
  echo "RELAX_ADVECT_SHEAR_PI=$RELAX_ADVECT_SHEAR_PI"
  echo "EXPAND_GRID=$EXPAND_GRID"
  echo "EXPAND_FACTOR=$EXPAND_FACTOR"
  echo "EXPAND_TAIL_CELLS=$EXPAND_TAIL_CELLS"
  echo "EXPAND_TAIL_FRAC=$EXPAND_TAIL_FRAC"
  echo "EXPAND_TAIL_ABS_E=$EXPAND_TAIL_ABS_E"
  echo "EXPAND_MIN_CELLS=$EXPAND_MIN_CELLS"
  echo "EXPAND_MAX_NR=$EXPAND_MAX_NR"
  echo "EXPAND_MAX_RMAX=$EXPAND_MAX_RMAX"
  echo "EXPAND_COOLDOWN=$EXPAND_COOLDOWN"
  echo "INIT_GOOD_RANGE=$INIT_GOOD_RANGE"
  echo "INIT_GOOD_MIN_CELLS=$INIT_GOOD_MIN_CELLS"
  echo "INIT_GOOD_BUFFER=$INIT_GOOD_BUFFER"
  echo "ETA_OVER_S=$ETA_OVER_S"
  echo "ZETA_OVER_S=$ZETA_OVER_S"
  echo "TAU_SHEAR_COEFF=$TAU_SHEAR_COEFF"
  echo "TAU_PI_COEFF=$TAU_PI_COEFF"
  echo "HYDRO_REPAIR_THETA=$HYDRO_REPAIR_THETA"
  echo "HYDRO_SRSCALE_MARGIN=$HYDRO_SRSCALE_MARGIN"
} > "$LAST_DIAG_ENV"
echo "Wrote solver env file: $LAST_DIAG_ENV"

# Ask whether to actually run the (expensive) IC diagnostics
RUN_DIAGNOSTICS=1
if ! ask_yn "Run IC diagnostics?" "Y"; then
  RUN_DIAGNOSTICS=0
  echo "Skipping diagnostics. Settings saved to $LAST_DIAG_ENV."
fi

# Optional terminal output toggles
if [[ "$PRINT_TABLE" == "0" ]]; then
  extra_flags+=("--no-table")
fi
if [[ "$FIELD_TABLE" == "1" ]]; then
  extra_flags+=("--field-table")
fi
if [[ "$PLOTS" == "1" ]]; then
  extra_flags+=("--plots")
fi

run_one_interp() {
  local interp_kind="$1"
  local outdir_run="$2"

  local -a cmd
  cmd=(
    "$JULIA_BIN" --project=. --threads="$THREADS" bench/ic_diagnostics.jl
    "--outdir=$outdir_run"
    "--Nr=$NR"
    "--rmax=$RMAX"
    "--nghost=$NGHOST"
    "--tau0=$TAU0"
    "--CFL=$CFL"
    "--CFLtau=$CFLTAU"
    "--eos=$EOS"
    "--init_csv=$INIT_CSV"
    "--fugacity=$FUGACITY"
    "--taper_width=$TAPER_WIDTH"
    "--interp=$interp_kind"
    "--plot_width=$PLOT_WIDTH"
    "--plot_kind=$PLOT_KIND"
    "--plot_fields=$PLOT_FIELDS"
    "--png_kind=$PNG_KIND"
    "--png_fields=$PNG_FIELDS"
    "--png_w=$PNG_W"
    "--png_h=$PNG_H"
  )

  # Only pass interp_dr for cubic; allow user to set empty to auto-select.
  if [[ "$interp_kind" == "cubic" && -n "${INTERP_DR}" ]]; then
    cmd+=("--interp_dr=$INTERP_DR")
  fi

  cmd+=("${extra_flags[@]}")

  echo "Running ($interp_kind):" \
    "${cmd[@]}"
  echo
  "${cmd[@]}"
  echo

  echo "Done. Outputs under: $outdir_run"

  if [[ "$POSTPLOT" == "1" ]]; then
    echo
    echo "Post-plotting full diagnostics (incl. gradAlpha_fv) for $interp_kind…"

    # Use a headless-safe GR backend.
    local outdir_abs
    local init_csv_abs
    outdir_abs="$(cd "$(dirname "$outdir_run")" && pwd)/$(basename "$outdir_run")"
    init_csv_abs="$(cd "$(dirname "$INIT_CSV")" && pwd)/$(basename "$INIT_CSV")"

    OUTDIR_ABS="$outdir_abs" INIT_CSV_ABS="$init_csv_abs" FUGACITY="$FUGACITY" DIAG_TRIM="$DIAG_TRIM" \
      "$JULIA_BIN" --project=. --threads=1 -e 'begin
      using CSV
      using Tables
      using Statistics
      using Plots

      # Headless-safe default for GR (still fine if another backend is active)
      try
        ENV["GKSwstype"] = "100"
      catch
      end

      outdir = get(ENV, "OUTDIR_ABS", "")
      init_csv = get(ENV, "INIT_CSV_ABS", "")
      fugacity = lowercase(get(ENV, "FUGACITY", "alpha"))
      trim = try
        parse(Int, get(ENV, "DIAG_TRIM", "3"))
      catch
        3
      end

      function hascol(tbl, name::Symbol)
        try
          return haskey(tbl, name)
        catch
          return false
        end
      end

      function col(tbl, name::Symbol)
        return hascol(tbl, name) ? collect(getproperty(tbl, name)) : nothing
      end

      function finite_minmax(v)
        vmin = Inf
        vmax = -Inf
        n = 0
        for x in v
          xf = Float64(x)
          if isfinite(xf)
            vmin = min(vmin, xf)
            vmax = max(vmax, xf)
            n += 1
          end
        end
        return (n == 0) ? (NaN, NaN, 0) : (vmin, vmax, n)
      end

      function maxabs(v)
        m = 0.0
        for x in v
          xf = Float64(x)
          if isfinite(xf)
            m = max(m, abs(xf))
          end
        end
        return m
      end

      function maxabsdiff(a, b)
        m = 0.0
        for i in eachindex(a, b)
          ai = a[i]; bi = b[i]
          if isfinite(ai) && isfinite(bi)
            m = max(m, abs(ai - bi))
          end
        end
        return m
      end

      function maxabsdiff_trim(a, b, trim::Int)
        n = min(length(a), length(b))
        if n <= 2*trim
          return maxabsdiff(a, b)
        end
        m = 0.0
        for i in (trim+1):(n-trim)
          ai = a[i]; bi = b[i]
          if isfinite(ai) && isfinite(bi)
            m = max(m, abs(ai - bi))
          end
        end
        return m
      end

      function saveplot(p, path)
        mkpath(dirname(path))
        savefig(p, path)
      end

      function read_csv_columntable(path)
        return Tables.columntable(CSV.File(path))
      end

      function read_first_row(path)
        rows = Tables.rowtable(CSV.File(path))
        return isempty(rows) ? nothing : rows[1]
      end

      # Load raw CSV data once (optional; used for sanity overlays)
      raw = nothing
      if !isempty(init_csv) && isfile(init_csv)
        raw_tbl = read_csv_columntable(init_csv)
        raw_r = hascol(raw_tbl, :r)  ? collect(raw_tbl.r)  : nothing
        raw_T = hascol(raw_tbl, :T0) ? collect(raw_tbl.T0) : nothing

        raw_f = nothing
        if fugacity == "alpha" && hascol(raw_tbl, :alpha0)
          raw_f = collect(raw_tbl.alpha0)
        elseif fugacity == "lambda" && hascol(raw_tbl, :lambda0)
          raw_f = collect(raw_tbl.lambda0)
        end
        raw = (; r=raw_r, T0=raw_T, f0=raw_f)
      end

      if isempty(outdir) || !isdir(outdir)
        @warn "Postplot: OUTDIR missing or not a directory" outdir
        exit(0)
      end

      cases = filter(p -> isdir(p), readdir(outdir; join=true))
      if isempty(cases)
        @warn "Postplot: no case directories found" outdir
        exit(0)
      end

      summary_lines = String[]
      push!(summary_lines, "Full IC diagnosis summary for $(outdir)")
      push!(summary_lines, "fugacity=$(fugacity) init_csv=$(init_csv)")

      for case_dir in sort(cases)
        fields_csv = joinpath(case_dir, "ic_fields.csv")
        isfile(fields_csv) || continue

        # Per-case run meta
        run_summary = read_first_row(joinpath(case_dir, "ic_run_summary.csv"))
        τ0 = (run_summary === nothing || !haskey(run_summary, :tau0)) ? NaN : Float64(run_summary.tau0)

        tbl = read_csv_columntable(fields_csv)
        r = col(tbl, :r)
        r === nothing && continue

        diagnosis_dir = joinpath(case_dir, "diagnosis")

        # Base fields
        T = col(tbl, :T)
        mu = col(tbl, :mu)
        alpha = col(tbl, :alpha)
        phi = col(tbl, :phi)
        ok = col(tbl, :ok)

        v = col(tbl, :v)
        ur = col(tbl, :ur)
        n = col(tbl, :n)
        e = col(tbl, :e)
        P = col(tbl, :P)
        D = col(tbl, :D)
        Dtau = col(tbl, :Dtau)

        # Derivatives from diagnostics
        dT = col(tbl, :dT)
        dmu = col(tbl, :dmu)
        dalpha = col(tbl, :dalpha)
        dphi = col(tbl, :dphi)

        d2T = col(tbl, :d2T)
        d2mu = col(tbl, :d2mu)
        d2mu_sg = col(tbl, :d2mu_sg)
        d2alpha = col(tbl, :d2alpha)
        d2phi = col(tbl, :d2phi)

        dn = col(tbl, :dn)
        d2n = col(tbl, :d2n)
        de = col(tbl, :de)
        d2e = col(tbl, :d2e)
        d2e_sg = col(tbl, :d2e_sg)
        dP = col(tbl, :dP)
        d2P = col(tbl, :d2P)
        dD = col(tbl, :dD)
        d2D = col(tbl, :d2D)
        dDtau = col(tbl, :dDtau)
        d2Dtau = col(tbl, :d2Dtau)
        dur = col(tbl, :dur)
        d2ur = col(tbl, :d2ur)
        dv = col(tbl, :dv)
        d2v = col(tbl, :d2v)

        # Solver-style diffusion gradient
        alpha_smooth = col(tbl, :alpha_smooth)
        gradAlpha_fv = col(tbl, :gradAlpha_fv)

        casename = basename(case_dir)
        push!(summary_lines, "")
        push!(summary_lines, "case=$(casename)")

        # Include run summary table row if present
        if run_summary !== nothing
          # Keep this compact; the dt breakdown is what we usually care about
          if haskey(run_summary, :dt)
            push!(summary_lines, "  dt=$(round(Float64(run_summary.dt), sigdigits=6)) dt_cfl=$(round(Float64(run_summary.dt_cfl), sigdigits=6)) dt_diff=$(round(Float64(run_summary.dt_diff), sigdigits=6)) dt_shear=$(round(Float64(run_summary.dt_shear), sigdigits=6)) dt_bulk=$(round(Float64(run_summary.dt_bulk), sigdigits=6))")
          end
          if haskey(run_summary, :ok_cells) && haskey(run_summary, :bad_cells)
            push!(summary_lines, "  ok_cells=$(run_summary.ok_cells) bad_cells=$(run_summary.bad_cells)")
          end
        end

        # Load field summary and record the top curvature offenders
        field_summary_path = joinpath(case_dir, "ic_field_summary.csv")
        if isfile(field_summary_path)
          fs = Tables.rowtable(CSV.File(field_summary_path))
          if !isempty(fs) && haskey(fs[1], :max_dimless_curv)
            sorted = sort(collect(fs); by = r -> (isfinite(Float64(r.max_dimless_curv)) ? -Float64(r.max_dimless_curv) : 0.0))
            topk = min(length(sorted), 5)
            push!(summary_lines, "  top max_dimless_curv fields:")
            for k in 1:topk
              rr = sorted[k]
              push!(summary_lines, "    $(rr.name): κ=$(round(Float64(rr.max_dimless_curv), sigdigits=6)) max|d1|=$(round(Float64(rr.max_abs_d1), sigdigits=6)) max|d2|=$(round(Float64(rr.max_abs_d2), sigdigits=6))")
            end
          end
        end

        # -----------------------------
        # Consistency checks (reliability)
        # -----------------------------
        if T !== nothing && mu !== nothing
          # alpha consistency: alpha == mu/T
          if alpha !== nothing
            alpha_check = [ (isfinite(mu[i]) && isfinite(T[i]) && abs(T[i]) > 0) ? (mu[i]/T[i]) : NaN for i in eachindex(T) ]
            aerr = [ (isfinite(alpha[i]) && isfinite(alpha_check[i])) ? (alpha[i] - alpha_check[i]) : NaN for i in eachindex(alpha) ]
            push!(summary_lines, "  max|alpha - mu/T| = $(round(maxabs(aerr), sigdigits=6))")
            p = plot(r, aerr; label="alpha - mu/T", lw=2, xlabel="r")
            saveplot(p, joinpath(diagnosis_dir, "alpha_minus_mu_over_T.png"))
          end

          # mhq consistency from definition phi = (mu - mhq)/T  => mhq = mu - phi*T
          if phi !== nothing
            mhq_infer = [ (isfinite(mu[i]) && isfinite(phi[i]) && isfinite(T[i])) ? (mu[i] - phi[i]*T[i]) : NaN for i in eachindex(T) ]
            mn, mx, nn = finite_minmax(mhq_infer)
            spread = (isfinite(mn) && isfinite(mx)) ? (mx - mn) : NaN
            push!(summary_lines, "  inferred mhq spread (max-min) = $(round(spread, sigdigits=6))")
            p = plot(r, mhq_infer; label="mhq inferred = mu - phi*T", lw=2, xlabel="r")
            saveplot(p, joinpath(diagnosis_dir, "mhq_inferred.png"))
          end
        end

        # v(ur) kinematics consistency
        if v !== nothing && ur !== nothing
          vchk = [ isfinite(ur[i]) ? (ur[i] / sqrt(1 + ur[i]^2)) : NaN for i in eachindex(ur) ]
          verr = [ (isfinite(v[i]) && isfinite(vchk[i])) ? (v[i] - vchk[i]) : NaN for i in eachindex(v) ]
          push!(summary_lines, "  max|v - ur/sqrt(1+ur^2)| = $(round(maxabs(verr), sigdigits=6))")
          p = plot(r, verr; label="v - ur/sqrt(1+ur^2)", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "v_minus_ur_kinematics.png"))
        end

        # Dtau / tau0 consistency: D ~ Dtau/tau0
        if D !== nothing && Dtau !== nothing && isfinite(τ0) && τ0 > 0
          Dchk = [ isfinite(Dtau[i]) ? (Dtau[i] / τ0) : NaN for i in eachindex(Dtau) ]
          Derr = [ (isfinite(D[i]) && isfinite(Dchk[i])) ? (D[i] - Dchk[i]) : NaN for i in eachindex(D) ]
          push!(summary_lines, "  max|D - Dtau/tau0| = $(round(maxabs(Derr), sigdigits=6))")
          p = plot(r, Derr; label="D - Dtau/tau0", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "D_minus_Dtau_over_tau0.png"))
        end

        # 1) alpha & smoothing
        if alpha !== nothing
          p = plot(r, alpha; label="alpha", lw=2, xlabel="r")
          if alpha_smooth !== nothing
            plot!(p, r, alpha_smooth; label="alpha_smooth", lw=2)
            saveplot(p, joinpath(diagnosis_dir, "alpha_vs_smooth.png"))

            diff = [ (isfinite(alpha[i]) && isfinite(alpha_smooth[i])) ? (alpha[i] - alpha_smooth[i]) : NaN for i in eachindex(alpha) ]
            p2 = plot(r, diff; label="alpha - alpha_smooth", lw=2, xlabel="r")
            saveplot(p2, joinpath(diagnosis_dir, "alpha_minus_smooth.png"))
          else
            saveplot(p, joinpath(diagnosis_dir, "alpha.png"))
          end
        end

        # 2) diffusion-relevant gradient check.
        # NOTE: gradAlpha_fv is computed from alpha_smooth with an FV limiter.
        # dalpha is a simple FD derivative of the *raw* alpha. Large differences can be normal
        # for non-smooth CSV ICs and are not necessarily an issue for the solver.
        if dalpha !== nothing && gradAlpha_fv !== nothing
          p = plot(r, dalpha; label="dalpha (FD on raw alpha)", lw=2, xlabel="r")
          plot!(p, r, gradAlpha_fv; label="gradAlpha_fv (solver, on alpha_smooth)", lw=2)
          saveplot(p, joinpath(diagnosis_dir, "dalpha_vs_gradAlpha_fv.png"))

          d = [ (isfinite(dalpha[i]) && isfinite(gradAlpha_fv[i])) ? (gradAlpha_fv[i] - dalpha[i]) : NaN for i in eachindex(dalpha) ]
          # Trim boundary-adjacent cells (stencil mismatch / BC effects)
          if length(d) > 2*trim
            for i in 1:trim
              d[i] = NaN
              d[end - i + 1] = NaN
            end
          end
          p2 = plot(r, d; label="gradAlpha_fv - dalpha", lw=2, xlabel="r")
          saveplot(p2, joinpath(diagnosis_dir, "gradAlpha_fv_minus_dalpha.png"))

          mad_full = maxabsdiff(dalpha, gradAlpha_fv)
          mad = maxabsdiff_trim(dalpha, gradAlpha_fv, trim)
          push!(summary_lines, "  max|gradAlpha_fv - dalpha(raw FD)| (interior, trim=$(trim)) = $(round(mad, sigdigits=6))")
          push!(summary_lines, "  max|gradAlpha_fv - dalpha(raw FD)| (full) = $(round(mad_full, sigdigits=6))")

          if alpha_smooth !== nothing
            # FD derivative of alpha_smooth on the same cell-centered grid.
            dr = (length(r) >= 2) ? (r[2] - r[1]) : NaN
            dalpha_s = Vector{Float64}(undef, length(alpha_smooth))
            fill!(dalpha_s, NaN)
            if isfinite(dr) && dr > 0 && length(alpha_smooth) >= 2
              # one-sided at ends
              if isfinite(alpha_smooth[1]) && isfinite(alpha_smooth[2])
                dalpha_s[1] = (alpha_smooth[2] - alpha_smooth[1]) / dr
              end
              if isfinite(alpha_smooth[end]) && isfinite(alpha_smooth[end-1])
                dalpha_s[end] = (alpha_smooth[end] - alpha_smooth[end-1]) / dr
              end
              # centered interior
              for i in 2:(length(alpha_smooth)-1)
                if isfinite(alpha_smooth[i-1]) && isfinite(alpha_smooth[i+1])
                  dalpha_s[i] = (alpha_smooth[i+1] - alpha_smooth[i-1]) / (2*dr)
                end
              end
            end

            p3 = plot(r, dalpha_s; label="d(alpha_smooth) (FD)", lw=2, xlabel="r")
            plot!(p3, r, gradAlpha_fv; label="gradAlpha_fv (solver)", lw=2)
            saveplot(p3, joinpath(diagnosis_dir, "dalpha_smooth_vs_gradAlpha_fv.png"))

            d3 = [ (isfinite(dalpha_s[i]) && isfinite(gradAlpha_fv[i])) ? (gradAlpha_fv[i] - dalpha_s[i]) : NaN for i in eachindex(dalpha_s) ]
            if length(d3) > 2*trim
              for i in 1:trim
                d3[i] = NaN
                d3[end - i + 1] = NaN
              end
            end
            p4 = plot(r, d3; label="gradAlpha_fv - d(alpha_smooth)", lw=2, xlabel="r")
            saveplot(p4, joinpath(diagnosis_dir, "gradAlpha_fv_minus_dalpha_smooth.png"))

            mad3_full = maxabsdiff(dalpha_s, gradAlpha_fv)
            mad3 = maxabsdiff_trim(dalpha_s, gradAlpha_fv, trim)
            push!(summary_lines, "  max|gradAlpha_fv - d(alpha_smooth)| (interior, trim=$(trim)) = $(round(mad3, sigdigits=6))")
            push!(summary_lines, "  max|gradAlpha_fv - d(alpha_smooth)| (full) = $(round(mad3_full, sigdigits=6))")
          end
        end

        # 2b) second derivatives for key IC smoothness triage
        if d2alpha !== nothing
          p = plot(r, d2alpha; label="d2alpha", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "d2alpha.png"))
        end
        if d2phi !== nothing
          p = plot(r, d2phi; label="d2phi", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "d2phi.png"))
        end
        if d2T !== nothing
          p = plot(r, d2T; label="d2T", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "d2T.png"))
        end
        if d2mu !== nothing
          p = plot(r, d2mu; label="d2mu (FD)", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "d2mu.png"))
        end
        if d2mu_sg !== nothing
          p = plot(r, d2mu_sg; label="d2mu_sg (smoothed)", lw=2, xlabel="r")
          if d2mu !== nothing
            plot!(p, r, d2mu; label="d2mu (FD)", lw=1, ls=:dash, alpha=0.5)
          end
          saveplot(p, joinpath(diagnosis_dir, "d2mu_sg.png"))
        end

        # 3b) additional hydro variables + derivatives (helpful to spot CSV kinks)
        if n !== nothing
          saveplot(plot(r, n; label="n", lw=2, xlabel="r"), joinpath(diagnosis_dir, "n.png"))
          dn !== nothing && saveplot(plot(r, dn; label="dn", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dn.png"))
          d2n !== nothing && saveplot(plot(r, d2n; label="d2n", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2n.png"))
        end
        if e !== nothing
          saveplot(plot(r, e; label="e", lw=2, xlabel="r"), joinpath(diagnosis_dir, "e.png"))
          de !== nothing && saveplot(plot(r, de; label="de", lw=2, xlabel="r"), joinpath(diagnosis_dir, "de.png"))
          d2e !== nothing && saveplot(plot(r, d2e; label="d2e", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2e.png"))
          if d2e_sg !== nothing
            p = plot(r, d2e_sg; label="d2e_sg (smoothed)", lw=2, xlabel="r")
            if d2e !== nothing
              plot!(p, r, d2e; label="d2e (FD)", lw=1, ls=:dash, alpha=0.5)
            end
            saveplot(p, joinpath(diagnosis_dir, "d2e_sg.png"))
          end
        end
        if P !== nothing
          saveplot(plot(r, P; label="P", lw=2, xlabel="r"), joinpath(diagnosis_dir, "P.png"))
          dP !== nothing && saveplot(plot(r, dP; label="dP", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dP.png"))
          d2P !== nothing && saveplot(plot(r, d2P; label="d2P", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2P.png"))
        end
        if D !== nothing
          saveplot(plot(r, D; label="D", lw=2, xlabel="r"), joinpath(diagnosis_dir, "D.png"))
          dD !== nothing && saveplot(plot(r, dD; label="dD", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dD.png"))
          d2D !== nothing && saveplot(plot(r, d2D; label="d2D", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2D.png"))
        end
        if Dtau !== nothing
          saveplot(plot(r, Dtau; label="Dtau", lw=2, xlabel="r"), joinpath(diagnosis_dir, "Dtau.png"))
          dDtau !== nothing && saveplot(plot(r, dDtau; label="dDtau", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dDtau.png"))
          d2Dtau !== nothing && saveplot(plot(r, d2Dtau; label="d2Dtau", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2Dtau.png"))
        end
        if ur !== nothing
          saveplot(plot(r, ur; label="ur", lw=2, xlabel="r"), joinpath(diagnosis_dir, "ur.png"))
          dur !== nothing && saveplot(plot(r, dur; label="dur", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dur.png"))
          d2ur !== nothing && saveplot(plot(r, d2ur; label="d2ur", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2ur.png"))
        end
        if v !== nothing
          saveplot(plot(r, v; label="v", lw=2, xlabel="r"), joinpath(diagnosis_dir, "v.png"))
          dv !== nothing && saveplot(plot(r, dv; label="dv", lw=2, xlabel="r"), joinpath(diagnosis_dir, "dv.png"))
          d2v !== nothing && saveplot(plot(r, d2v; label="d2v", lw=2, xlabel="r"), joinpath(diagnosis_dir, "d2v.png"))
        end

        # 3) T and mu with first derivatives
        if T !== nothing
          p = plot(r, T; label="T", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "T.png"))
          if dT !== nothing
            p2 = plot(r, dT; label="dT", lw=2, xlabel="r")
            saveplot(p2, joinpath(diagnosis_dir, "dT.png"))
          end
        end
        if mu !== nothing
          p = plot(r, mu; label="mu", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "mu.png"))
          if dmu !== nothing
            p2 = plot(r, dmu; label="dmu", lw=2, xlabel="r")
            saveplot(p2, joinpath(diagnosis_dir, "dmu.png"))
          end
        end

        # 4) phi with derivative
        if phi !== nothing
          p = plot(r, phi; label="phi", lw=2, xlabel="r")
          saveplot(p, joinpath(diagnosis_dir, "phi.png"))
          if dphi !== nothing
            p2 = plot(r, dphi; label="dphi", lw=2, xlabel="r")
            saveplot(p2, joinpath(diagnosis_dir, "dphi.png"))
          end
        end

        # 5) Raw CSV overlays (if available)
        if raw !== nothing && raw.r !== nothing
          if T !== nothing && raw.T0 !== nothing
            p = plot(raw.r, raw.T0; seriestype=:scatter, ms=2, label="T0 (raw CSV)", xlabel="r")
            plot!(p, r, T; lw=2, label="T (on grid)")
            saveplot(p, joinpath(diagnosis_dir, "T_raw_vs_grid.png"))
          end

          if alpha !== nothing && raw.f0 !== nothing
            raw_alpha = if fugacity == "alpha"
              raw.f0
            else
              map(x -> (isfinite(x) && x > 0) ? log(x) : NaN, raw.f0)
            end
            p = plot(raw.r, raw_alpha; seriestype=:scatter, ms=2, label="alpha (raw CSV)", xlabel="r")
            plot!(p, r, alpha; lw=2, label="alpha (on grid)")
            saveplot(p, joinpath(diagnosis_dir, "alpha_raw_vs_grid.png"))
          end
        end

        # 6) ok mask
        if ok !== nothing
          okf = map(x -> x ? 1.0 : 0.0, ok)
          p = plot(r, okf; label="ok (1=true)", lw=2, xlabel="r", ylim=(-0.1, 1.1))
          saveplot(p, joinpath(diagnosis_dir, "ok_mask.png"))
        end
      end

      open(joinpath(outdir, "diagnosis_summary.txt"), "w") do io
        for ln in summary_lines
          println(io, ln)
        end
      end
      println("Wrote: " * joinpath(outdir, "diagnosis_summary.txt"))
    end'
  fi
}

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then

# Run selected interpolation modes.
# If only one mode is enabled, write directly into OUTDIR.
both=0
if [[ "$RUN_LINEAR" == "1" && "$RUN_CUBIC" == "1" ]]; then
  both=1
fi

if [[ "$RUN_LINEAR" == "1" ]]; then
  od="$OUTDIR"
  [[ "$both" == "1" ]] && od="${OUTDIR}_linear"
  run_one_interp linear "$od"
fi
if [[ "$RUN_CUBIC" == "1" ]]; then
  od="$OUTDIR"
  [[ "$both" == "1" ]] && od="${OUTDIR}_cubic"
  run_one_interp cubic "$od"
fi

# After diagnostics, shrink solver NR/RMAX only if the IC has bad outer cells.
update_env_from_good_range() {
  local outdir_use="$1"
  local case_dir="$outdir_use/single"
  local summary_csv="$case_dir/ic_run_summary.csv"

  [[ -f "$summary_csv" ]] || return 0

  local buf
  buf="${INIT_GOOD_BUFFER:-4}"

  local vals
  vals=$(python3 - "$summary_csv" "$buf" <<'PY'
import csv
import math
import sys


def as_int(x, default=0):
    try:
        s = "" if x is None else str(x).strip()
        return int(float(s)) if s != "" else default
    except Exception:
        return default


def as_float(x, default=float("nan")):
    try:
        s = "" if x is None else str(x).strip()
        v = float(s) if s != "" else default
        return v if math.isfinite(v) else default
    except Exception:
        return default


path = sys.argv[1]
buf = as_int(sys.argv[2], 0)

with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    sys.exit(0)

r = rows[0]
nr_good = as_int(r.get("Nr_good", ""), 0)
nr_full = as_int(r.get("Nr", ""), 0)

# Only shrink if diagnostics found bad cells at large radii:
# OK-prefix (Nr_good) shorter than full domain (Nr).
if not (nr_full > 0 and nr_good > 0 and nr_good < nr_full):
    sys.exit(0)

rmax_good = as_float(r.get("rmax_good", ""))
dr = as_float(r.get("dr", ""))

if buf > 0:
    nr_good = max(nr_good - buf, 1)
    if math.isfinite(rmax_good) and math.isfinite(dr):
        rmax_good = rmax_good - buf * dr

print(nr_good, rmax_good)
PY
)

  local nr_good rmax_good
  read -r nr_good rmax_good <<< "$vals"

  if [[ -n "$nr_good" && "$nr_good" -gt 0 ]] && [[ "$rmax_good" != "nan" ]]; then
    echo "Large-r bad cells detected; shrinking solver domain: NR=$nr_good RMAX=$rmax_good (buffer=$buf)"
    local tmp
    tmp=$(mktemp)
    awk -v nr="$nr_good" -v rmax="$rmax_good" -v buf="$buf" '
      BEGIN {done_nr=0; done_rmax=0; done_expand=0; done_buf=0}
      /^NR=/   {print "NR="nr; done_nr=1; next}
      /^RMAX=/ {print "RMAX="rmax; done_rmax=1; next}
      /^EXPAND_GRID=/ {print "EXPAND_GRID=1"; done_expand=1; next}
      /^INIT_GOOD_BUFFER=/ {print "INIT_GOOD_BUFFER="buf; done_buf=1; next}
      {print}
      END {
        if (!done_nr)   print "NR="nr
        if (!done_rmax) print "RMAX="rmax
        if (!done_expand) print "EXPAND_GRID=1"
        if (!done_buf) print "INIT_GOOD_BUFFER="buf
      }
    ' "$LAST_DIAG_ENV" > "$tmp" && mv "$tmp" "$LAST_DIAG_ENV"
  fi
}

if [[ "$both" == "1" ]]; then
  if [[ "$SOLVER_INTERP" == "cubic" ]]; then
    update_env_from_good_range "${OUTDIR}_cubic"
  else
    update_env_from_good_range "${OUTDIR}_linear"
  fi
else
  update_env_from_good_range "$OUTDIR"
fi

fi  # end if RUN_DIAGNOSTICS

diag_status="skipped"
if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
  diag_status="completed"
fi
print_run_summary "$diag_status"
