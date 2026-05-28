#!/usr/bin/env bash
set -euo pipefail

# Interactive launcher for main2.jl using environment variables.
# Empty input keeps the shown default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

prompt() {
  local var_name="$1"
  local label="$2"
  local default="$3"
  local value

  read -r -p "${label} [${default}]: " value || true
  value="$(printf '%s' "$value" | xargs)"
  if [[ -z "${value}" ]]; then
    printf '%s' "${default}"
  else
    printf '%s' "${value}"
  fi
}

prompt_mode() {
  local default="$1"
  local value
  read -r -p "PHIS_MODE (on/off) [${default}]: " value || true
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [[ -z "${value}" ]]; then
    value="$default"
  fi
  case "$value" in
    on|coupled|1|true|yes|y) printf '%s' "coupled" ;;
    off|0|false|no|n)       printf '%s' "off" ;;
    *)
      echo "Invalid PHIS_MODE: '$value' (use on/off)" >&2
      exit 2
      ;;
  esac
}

prompt_optional() {
  local label="$1"
  local value
  read -r -p "${label} (optional): " value || true
  value="$(printf '%s' "$value" | xargs)"
  printf '%s' "$value"
}

TAU0="$(prompt TAU0 "tau0" "0.4")"
TAUFINAL="$(prompt TAUFINAL "tau final" "15.0")"
RMAX="$(prompt RMAX "rmax" "20.0")"
NR="$(prompt NR "nr" "400")"
DS_T="$(prompt DS_T "DsT" "0.24")"
PHIS_MODE="$(prompt_mode "on")"
RUN_LABEL="$(prompt_optional "RUN_LABEL")"

export TAU0 TAUFINAL RMAX NR DS_T PHIS_MODE RUN_LABEL

echo ""
echo "Launching main2.jl with:"
echo "  TAU0=$TAU0"
echo "  TAUFINAL=$TAUFINAL"
echo "  RMAX=$RMAX"
echo "  NR=$NR"
echo "  DS_T=$DS_T"
echo "  PHIS_MODE=$PHIS_MODE"
if [[ -n "${RUN_LABEL}" ]]; then
  echo "  RUN_LABEL=$RUN_LABEL"
fi
echo ""

exec julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/main2.jl"
