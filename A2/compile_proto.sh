#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  echo "Create /home/yichu/A2/.venv first or set PYTHON_BIN explicitly." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$SCRIPT_DIR/proto" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$SCRIPT_DIR/proto/a2_data.proto"

# grpc_tools generates an absolute local import. Make it package-relative.
sed -i \
  's/^import a2_data_pb2 as a2__data__pb2$/from . import a2_data_pb2 as a2__data__pb2/' \
  "$OUT_DIR/a2_data_pb2_grpc.py"

echo "Generated:"
echo "  $OUT_DIR/a2_data_pb2.py"
echo "  $OUT_DIR/a2_data_pb2_grpc.py"
