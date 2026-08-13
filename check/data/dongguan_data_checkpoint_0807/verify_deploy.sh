#!/usr/bin/env bash
set -euo pipefail
DEPLOY="${DONGGUAN_DEPLOY_ROOT:-/data/dongguan_data_checkpoint_0807/deploy_dgx}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

echo "=== verify deploy_dgx ==="
echo "DEPLOY=$DEPLOY"
echo "PY=$PY"
echo "CHECKPOINT=$CHECKPOINT"

for p in \
  "$DEPLOY/models/GR00T-N1.7-3B" \
  "$DEPLOY/models/Cosmos-Reason2-2B" \
  "$DEPLOY/checkpoints/multi345_v1/checkpoint-14000" \
  "$DEPLOY/checkpoints/multi345_v1/checkpoint-15000" \
  "$CODE/isaacGr00t/gr00t" \
  "$CODE/pipeline5_dongguan_relative/dongguan_inference/live_runner.py" \
  "$PY"
do
  if [[ ! -e "$p" ]]; then
    echo "MISSING: $p" >&2
    exit 1
  fi
  echo "  ok $p"
done

"$PY" parity_check_relative.py --checkpoint "$CHECKPOINT"
"$PY" parity_check_relative.py --checkpoint "$CHECKPOINT" --load-policy

echo "=== verify_deploy OK (parity + load-policy) ==="
