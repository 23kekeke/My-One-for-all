#!/usr/bin/env bash
# Pipeline7 data flow: Step4 -> Step5 -> Step6 -> pre_train_confirm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_PY="${ROOT}/../isaacGr00t/.venv/bin/python"
DATA_ROOT="/data/dongguan_data_checkpoint_0807"
STEP3="${DATA_ROOT}/tmp/step3_rot6d"
P5_LEROBOT="${DATA_ROOT}/lerobot/multi_345"
P7_LEROBOT="${DATA_ROOT}/lerobot_p7/multi_345"
SMOKE="${DATA_ROOT}/tmp/step_smoke_p7"

mkdir -p "${SMOKE}"

echo "=== Step4 export (state 32 / action 20) ==="
python "${ROOT}/step4_export_lerobot.py" \
  --input-root "${STEP3}" \
  --output-root "${P7_LEROBOT}" \
  --link-videos-from "${P5_LEROBOT}"

echo "=== Step5 relative stats ==="
"${GR00T_PY}" "${ROOT}/step5_generate_relative_stats.py" \
  --dataset-path "${P7_LEROBOT}" \
  --output "${SMOKE}/multi_345_relative_stats_report.json"

echo "=== Step6 loader smoke ==="
"${GR00T_PY}" "${ROOT}/step6_smoke_loader_relative.py" \
  --dataset-path "${P7_LEROBOT}" \
  --output "${SMOKE}/multi_345_loader_report.json"

echo "=== Pre-train confirm ==="
"${GR00T_PY}" "${ROOT}/pre_train_confirm.py" \
  --dataset-path "${P7_LEROBOT}" \
  --output "${SMOKE}/pre_train_confirm.json"

echo "Done. See ${SMOKE}/pre_train_confirm.json"
