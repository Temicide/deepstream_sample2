#!/usr/bin/env bash
set -euo pipefail

MODEL_PT="${1:-models/yolov8s.pt}"
MODEL_ONNX="${2:-models/yolov8s.onnx}"
IMG_SIZE="${IMG_SIZE:-224}"
OPSET="${OPSET:-12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_SOURCE="$MODEL_PT"
if [[ ! -f "$MODEL_PT" ]]; then
  if [[ "$(basename "$MODEL_PT")" == "yolov8s.pt" ]]; then
    echo "[INFO] $MODEL_PT not found locally; Ultralytics will download yolov8s.pt automatically."
    MODEL_SOURCE="yolov8s.pt"
  else
    echo "[ERROR] Missing input model: $MODEL_PT"
    echo "Place the .pt file at that path or pass a valid file as the first argument."
    exit 1
  fi
fi

"$PYTHON_BIN" - <<PY
from pathlib import Path
from ultralytics import YOLO

pt_path = Path("$MODEL_SOURCE")
onnx_path = Path("$MODEL_ONNX")
imgsz = int("$IMG_SIZE")
opset = int("$OPSET")

model = YOLO(str(pt_path))
exported_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=opset)
exported_path = Path(exported_path)

if exported_path.resolve() != onnx_path.resolve():
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx_path.exists():
        onnx_path.unlink()
    exported_path.replace(onnx_path)

print(f"[INFO] Exported ONNX: {onnx_path}")
print(f"[INFO] input size: {imgsz}x{imgsz}")
PY
