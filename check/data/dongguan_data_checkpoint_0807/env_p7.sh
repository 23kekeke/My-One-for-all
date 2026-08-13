#!/usr/bin/env bash
# Source AFTER env.sh for pipeline7 eef-only inference.
#   source /home/yichu/check/data/dongguan_data_checkpoint_0807/env.sh
#   source /home/yichu/check/data/dongguan_data_checkpoint_0807/env_p7.sh

export DONGGUAN_DEPLOY_ROOT="${DONGGUAN_DEPLOY_ROOT:-/home/yichu/check/data/dongguan_data_checkpoint_0807}"
export DEPLOY="${DEPLOY:-$DONGGUAN_DEPLOY_ROOT}"
export CODE="${CODE:-$DEPLOY/code}"

export P7="$CODE/pipeline7/pipeline7_dongguan_relative_eef_only"

# Prefer pipeline7 on PYTHONPATH (ahead of pipeline5) so eef-only modality wins.
export PYTHONPATH="$CODE/isaacGr00t:$P7:$CODE/pipeline3_biman:$CODE/pipeline2:${PYTHONPATH:-}"

_DEFAULT_CHECKPOINT_P7="$DEPLOY/checkpoints/multi345_end_only/checkpoint-15000"
_FALLBACK_CHECKPOINT_P7="$DEPLOY/checkpoints/multi345_end_only/checkpoint-13000"

_pick_p7_ckpt() {
  local cand="$1"
  if [[ -d "$cand" ]] && [[ -f "$cand/model.safetensors.index.json" ]]; then
    # Reject incomplete Drive downloads
    if compgen -G "$cand/model-*.safetensors.drivedownload" > /dev/null; then
      return 1
    fi
    if [[ -f "$cand/model-00001-of-00002.safetensors" ]] || [[ -f "$cand/model-00002-of-00002.safetensors" ]]; then
      return 0
    fi
  fi
  return 1
}

if [[ -z "${CHECKPOINT_P7:-}" ]] || ! _pick_p7_ckpt "${CHECKPOINT_P7}"; then
  if [[ -n "${CHECKPOINT_P7:-}" ]]; then
    echo "[env_p7.sh] CHECKPOINT_P7 incomplete/missing: $CHECKPOINT_P7" >&2
  fi
  if _pick_p7_ckpt "$_DEFAULT_CHECKPOINT_P7"; then
    export CHECKPOINT_P7="$_DEFAULT_CHECKPOINT_P7"
  elif _pick_p7_ckpt "$_FALLBACK_CHECKPOINT_P7"; then
    echo "[env_p7.sh] -> falling back to $_FALLBACK_CHECKPOINT_P7" >&2
    export CHECKPOINT_P7="$_FALLBACK_CHECKPOINT_P7"
  else
    export CHECKPOINT_P7="$_DEFAULT_CHECKPOINT_P7"
    echo "[env_p7.sh] WARNING: no complete multi345_end_only checkpoint found" >&2
  fi
fi
# Alias for paste commands that still use $CHECKPOINT
export CHECKPOINT="$CHECKPOINT_P7"
unset _DEFAULT_CHECKPOINT_P7 _FALLBACK_CHECKPOINT_P7

cd "$P7"
echo "[env_p7.sh] P7=$P7"
echo "[env_p7.sh] CHECKPOINT_P7=$CHECKPOINT_P7"
