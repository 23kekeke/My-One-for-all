#!/usr/bin/env bash
# Source before any Dongguan infer command on deploy_dgx.
#   source /home/yichu/check/data/dongguan_data_checkpoint_0807/env.sh

export DONGGUAN_DEPLOY_ROOT="${DONGGUAN_DEPLOY_ROOT:-/home/yichu/check/data/dongguan_data_checkpoint_0807}"
export DEPLOY="$DONGGUAN_DEPLOY_ROOT"
export CODE="$DEPLOY/code"

export GR00T_COSMOS_MODEL_PATH="$DEPLOY/models/Cosmos-Reason2-2B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PIPELINE2_QLORA=1
export GROOT_PATCH_MISTRAL=1
export GROOT_HF_LOCAL_FIRST=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Spark aarch64: user-local NVPL (no sudo) + CUDA 13 + venv NVIDIA libs
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
_GR00T_SP="${CODE:-$DONGGUAN_DEPLOY_ROOT/code}/isaacGr00t/.venv/lib/python3.12/site-packages"
_LD_PREPEND=""
for _d in \
  "$_GR00T_SP/nvidia/cu13/lib" \
  "$_GR00T_SP/nvidia/cudnn/lib" \
  "$_GR00T_SP/torch/lib" \
  "${HOME}/.local/nvpl/usr/lib/aarch64-linux-gnu"
do
  if [[ -d "$_d" ]]; then
    _LD_PREPEND="${_LD_PREPEND:+$_LD_PREPEND:}$_d"
  fi
done
if [[ -n "$_LD_PREPEND" ]]; then
  export LD_LIBRARY_PATH="${_LD_PREPEND}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _d _LD_PREPEND _GR00T_SP

export PYTHONPATH="$CODE/isaacGr00t:$CODE/pipeline5_dongguan_relative:$CODE/pipeline3_biman:$CODE/pipeline2:${PYTHONPATH:-}"

# Prefer local GR00T venv; fall back if symlink broken / not yet rebuilt on this host.
_DEFAULT_PY="$CODE/isaacGr00t/.venv/bin/python"
if [[ ! -x "$_DEFAULT_PY" ]]; then
  for cand in \
    "$HOME/miniconda3/envs/gr00t/bin/python" \
    "$HOME/miniconda3/envs/gr00t_py310/bin/python" \
    /home/ubuntu/projects/zibianliang_env/isaacGr00t/.venv/bin/python
  do
    if [[ -x "$cand" ]]; then
      _DEFAULT_PY="$cand"
      break
    fi
  done
fi
export PY="${PY:-$_DEFAULT_PY}"

# SDK daemon / capture must use xr_lerobot (not GR00T venv)
_DEFAULT_PY_SDK=""
for cand in \
  "$HOME/miniconda3/envs/xr_lerobot/bin/python" \
  /home/ubuntu/anaconda3/envs/xr_lerobot/bin/python \
  /home/yichu/anaconda3/envs/xr_lerobot/bin/python
do
  if [[ -x "$cand" ]]; then
    _DEFAULT_PY_SDK="$cand"
    break
  fi
done
export PY_SDK="${PY_SDK:-${_DEFAULT_PY_SDK:-python3}}"

_DEFAULT_CHECKPOINT="$DEPLOY/checkpoints/multi345_v1/checkpoint-15000"
# If CHECKPOINT is unset OR points at a missing path (e.g. old
# checkpoints/checkpoint-15000 after the multi345_v1 move), use default.
if [[ -z "${CHECKPOINT:-}" || ! -e "${CHECKPOINT}" ]]; then
  if [[ -n "${CHECKPOINT:-}" && ! -e "${CHECKPOINT}" ]]; then
    echo "[env.sh] CHECKPOINT missing: $CHECKPOINT" >&2
    echo "[env.sh] -> using $_DEFAULT_CHECKPOINT" >&2
  fi
  export CHECKPOINT="$_DEFAULT_CHECKPOINT"
fi
unset _DEFAULT_CHECKPOINT
export P5="$CODE/pipeline5_dongguan_relative"
export P7="$CODE/pipeline7/pipeline7_dongguan_relative_eef_only"
export CHECKPOINT_P7="${CHECKPOINT_P7:-$DEPLOY/checkpoints/multi345_end_only/checkpoint-15000}"
# Robot gRPC (Dongguan LAN). Override if needed.
export ROBOT_SERVER="${ROBOT_SERVER:-192.168.1.103:50051}"
export SDK_DAEMON_URL="${SDK_DAEMON_URL:-127.0.0.1:15101}"

cd "$P5"
# For pipeline7 eef-only infer, also: source "$DEPLOY/env_p7.sh"

# Friendly diagnostics (non-fatal)
if [[ ! -x "$PY" ]]; then
  echo "[env.sh] WARNING: PY not executable: $PY" >&2
  echo "[env.sh] This host is $(uname -m). Packed .venv was built for another machine/arch." >&2
  echo "[env.sh] Rebuild GR00T venv on this Spark, then: export PY=/path/to/python" >&2
fi
if [[ ! -x "$PY_SDK" ]]; then
  echo "[env.sh] WARNING: PY_SDK not executable: $PY_SDK" >&2
fi
