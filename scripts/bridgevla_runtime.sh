#!/usr/bin/env bash
# BridgeVLA machine-independent runtime bootstrap.
# Source this file from a shell; do not execute it as a standalone process.
#
# Optional per-host configuration:
#   ~/.config/bridgevla/$(hostname -s).env
#
# Supported overrides:
#   BRIDGEVLA_CONDA_SH       path to conda.sh
#   BRIDGEVLA_CONDA_ENV      conda environment name (default: bridgevla)
#   BRIDGEVLA_COPPELIASIM_ROOT
#   BRIDGEVLA_DISPLAY         default display when DISPLAY is unset
#   BRIDGEVLA_START_XVFB      start Xvfb (default: 0; eval enables it)
#   BRIDGEVLA_CHECK_GPU       require and print nvidia-smi (default: 1)
#   BRIDGEVLA_EXPECTED_HOSTNAME  short hostname or comma-separated hostnames
#   BRIDGEVLA_SKIP_CONDA      use an already activated/external Python env

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[bridgevla-runtime] source this file; do not execute it" >&2
    exit 2
fi

_bridgevla_runtime_error() {
    echo "[bridgevla-runtime] ERROR: $*" >&2
    return 1
}

_bridgevla_runtime_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_bridgevla_runtime_host="$(hostname -s 2>/dev/null || hostname)"
_bridgevla_runtime_config="${BRIDGEVLA_RUNTIME_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/bridgevla/${_bridgevla_runtime_host}.env}"

if [[ -f "$_bridgevla_runtime_config" ]]; then
    # shellcheck disable=SC1090
    source "$_bridgevla_runtime_config"
fi

if [[ -n "${BRIDGEVLA_EXPECTED_HOSTNAME:-}" ]]; then
    case ",${BRIDGEVLA_EXPECTED_HOSTNAME}," in
        *,"${_bridgevla_runtime_host}",*) ;;
        *)
            _bridgevla_runtime_error "expected hostname '${BRIDGEVLA_EXPECTED_HOSTNAME}', got '${_bridgevla_runtime_host}'"
            return 1
            ;;
    esac
fi

export BRIDGEVLA_HOSTNAME="$_bridgevla_runtime_host"

if [[ -z "${BRIDGEVLA_ROOT:-}" ]]; then
    BRIDGEVLA_ROOT="$(cd -- "$_bridgevla_runtime_script_dir/.." && pwd)"
fi
export BRIDGEVLA_ROOT

[[ -d "$BRIDGEVLA_ROOT/finetune" ]] || {
    _bridgevla_runtime_error "invalid repository root: $BRIDGEVLA_ROOT"
    return 1
}

if [[ "${BRIDGEVLA_SKIP_CONDA:-0}" != "1" ]]; then
    if ! declare -F conda >/dev/null 2>&1; then
        _bridgevla_conda_sh="${BRIDGEVLA_CONDA_SH:-}"
        if [[ -z "$_bridgevla_conda_sh" ]]; then
            for _candidate in \
                "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "$HOME/miniforge3/etc/profile.d/conda.sh" \
                "/opt/miniconda3/etc/profile.d/conda.sh" \
                "/opt/miniforge3/etc/profile.d/conda.sh"; do
                if [[ -f "$_candidate" ]]; then
                    _bridgevla_conda_sh="$_candidate"
                    break
                fi
            done
        fi
        if [[ -z "$_bridgevla_conda_sh" || ! -f "$_bridgevla_conda_sh" ]]; then
            _bridgevla_runtime_error "cannot find conda.sh; set BRIDGEVLA_CONDA_SH or create $_bridgevla_runtime_config"
            return 1
        fi
        # shellcheck disable=SC1090
        source "$_bridgevla_conda_sh"
    fi

    declare -F conda >/dev/null 2>&1 || {
        _bridgevla_runtime_error "conda shell function is unavailable after sourcing conda.sh"
        return 1
    }

    _bridgevla_conda_env="${BRIDGEVLA_CONDA_ENV:-bridgevla}"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$_bridgevla_conda_env" ]]; then
        conda activate "$_bridgevla_conda_env" || {
            _bridgevla_runtime_error "failed to activate conda environment '$_bridgevla_conda_env'"
            return 1
        }
    fi
fi

export COPPELIASIM_ROOT="${BRIDGEVLA_COPPELIASIM_ROOT:-$BRIDGEVLA_ROOT/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
[[ -d "$COPPELIASIM_ROOT" ]] || {
    _bridgevla_runtime_error "CoppeliaSim directory does not exist: $COPPELIASIM_ROOT"
    return 1
}

case ":${LD_LIBRARY_PATH:-}:" in
    *":$COPPELIASIM_ROOT:"*) ;;
    *) export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"

_bridgevla_python_paths=(
    "$BRIDGEVLA_ROOT/finetune"
    "$BRIDGEVLA_ROOT/finetune/bridgevla/libs/YARR"
    "$BRIDGEVLA_ROOT/finetune/bridgevla/libs/RLBench"
    "$BRIDGEVLA_ROOT/finetune/bridgevla/libs/PyRep"
    "$BRIDGEVLA_ROOT/finetune/bridgevla/libs/peract_colab"
    "$BRIDGEVLA_ROOT/finetune/bridgevla/libs/point-renderer"
)
for _path in "${_bridgevla_python_paths[@]}"; do
    case ":${PYTHONPATH:-}:" in
        *":$_path:"*) ;;
        *) PYTHONPATH="$_path${PYTHONPATH:+:$PYTHONPATH}" ;;
    esac
done
export PYTHONPATH

export DISPLAY="${DISPLAY:-${BRIDGEVLA_DISPLAY:-:1.0}}"

if [[ "${BRIDGEVLA_START_XVFB:-0}" == "1" ]]; then
    command -v Xvfb >/dev/null 2>&1 || {
        _bridgevla_runtime_error "BRIDGEVLA_START_XVFB=1 but Xvfb is not installed"
        return 1
    }

    _bridgevla_display_ready=0
    if command -v xdpyinfo >/dev/null 2>&1 && DISPLAY="$DISPLAY" xdpyinfo >/dev/null 2>&1; then
        _bridgevla_display_ready=1
    fi

    if [[ "$_bridgevla_display_ready" -eq 0 ]]; then
        _bridgevla_display_num="${DISPLAY%%.*}"
        if ! [[ "$_bridgevla_display_num" =~ ^:[0-9]+$ ]]; then
            _bridgevla_runtime_error "unsupported DISPLAY '$DISPLAY'; expected :N or :N.S"
            return 1
        fi

        if ! ps -eo args= | grep -E "(^|[[:space:]])Xvfb[[:space:]]+${_bridgevla_display_num}([[:space:]]|$)" | grep -v grep >/dev/null 2>&1; then
            _bridgevla_xvfb_log="${BRIDGEVLA_XVFB_LOG:-/tmp/bridgevla-xvfb-${_bridgevla_display_num#:}.log}"
            Xvfb "$_bridgevla_display_num" -screen 0 1024x768x24 -ac >"$_bridgevla_xvfb_log" 2>&1 &
            export BRIDGEVLA_XVFB_PID=$!
        fi

        if command -v xdpyinfo >/dev/null 2>&1; then
            for _attempt in $(seq 1 20); do
                if DISPLAY="$DISPLAY" xdpyinfo >/dev/null 2>&1; then
                    _bridgevla_display_ready=1
                    break
                fi
                sleep 0.2
            done
        fi

        [[ "$_bridgevla_display_ready" -eq 1 ]] || {
            _bridgevla_runtime_error "X display '$DISPLAY' did not become ready"
            return 1
        }
    fi
fi

if [[ "${BRIDGEVLA_CHECK_GPU:-1}" == "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
        _bridgevla_runtime_error "nvidia-smi is unavailable; set BRIDGEVLA_CHECK_GPU=0 only for a CPU-only operation"
        return 1
    }
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || {
        _bridgevla_runtime_error "nvidia-smi failed"
        return 1
    }
fi

if ! command -v python >/dev/null 2>&1; then
    _bridgevla_runtime_error "python is unavailable after runtime setup"
    return 1
fi

if [[ "${BRIDGEVLA_RUNTIME_QUIET:-0}" != "1" ]]; then
    echo "[bridgevla-runtime] host=$_bridgevla_runtime_host repo=$BRIDGEVLA_ROOT env=${CONDA_DEFAULT_ENV:-external} python=$(command -v python) display=$DISPLAY"
fi

unset _bridgevla_conda_sh _bridgevla_conda_env _bridgevla_candidate _candidate
unset _bridgevla_python_paths _path _bridgevla_display_ready _bridgevla_display_num _bridgevla_xvfb_log _attempt
unset _bridgevla_runtime_script_dir _bridgevla_runtime_host _bridgevla_runtime_config
unset -f _bridgevla_runtime_error
