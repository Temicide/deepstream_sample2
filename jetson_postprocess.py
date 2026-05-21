from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pipline as core


DEFAULT_RUNTIME_LOG = "jetson_postprocess.jsonl"
DEFAULT_TARGET_FPS = 10.0


def runtime_record(event: str, **data: Any) -> dict[str, Any]:
    return {"event": event, **data}


def write_runtime_log(path: str | None, record: dict[str, Any]) -> None:
    message = json.dumps(record, ensure_ascii=False)
    print(message, flush=True)
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def parse_args() -> argparse.Namespace:
    args = core.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = DEFAULT_RUNTIME_LOG
    if args.target_fps <= 0:
        args.target_fps = DEFAULT_TARGET_FPS
    args.show = False
    return args


def run(args: argparse.Namespace) -> None:
    yolo_path = core.resolve_model_path(args.yolo, fallback="yolo.onnx")
    cls_path = core.resolve_model_path(args.cls, fallback="cls.onnx")
    args.resolved_yolo_path = yolo_path

    yolo_labels, brand_labels = core.load_labels(args.labels, yolo_path, cls_path)
    sources = core.source_specs_from_args(args)
    if not sources:
        raise RuntimeError("No sources configured")

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_runtime_log(
        args.output_jsonl,
        runtime_record(
            "jetson_postprocess_start",
            yolo=yolo_path,
            classifier=cls_path,
            labels=args.labels,
            yolo_runner=args.yolo_runner,
            ultralytics_tracker=args.ultralytics_tracker,
            api_url=None if args.no_api else args.api_url,
            output_jsonl=args.output_jsonl,
            target_fps=args.target_fps,
            rtsp_count=len(sources),
            sources=sources,
        ),
    )

    cls_model = core.OnnxModel(
        cls_path,
        backend=args.backend,
        input_size=args.cls_size,
        allow_fallback=True,
        role="classifier",
    )

    if len(sources) > 1:
        yolo_model = None
        if args.yolo_runner == "onnx":
            yolo_model = core.OnnxModel(
                yolo_path,
                backend=args.backend,
                input_size=args.yolo_size,
                allow_fallback=False,
                role="YOLO",
            )
        core.process_multi_stream(args, yolo_model, cls_model, yolo_labels, brand_labels, sources)
        return

    if args.yolo_runner == "ultralytics":
        yolo_model = core.UltralyticsTrackDetector(yolo_path, yolo_labels, args)
    else:
        yolo_model = core.OnnxModel(
            yolo_path,
            backend=args.backend,
            input_size=args.yolo_size,
            allow_fallback=False,
            role="YOLO",
        )

    args.camera_id = sources[0]["camera_id"]
    args.source = sources[0]["source"]
    if core.is_image_path(args.source):
        core.process_image(args, yolo_model, cls_model, yolo_labels, brand_labels)
    else:
        core.process_stream(args, yolo_model, cls_model, yolo_labels, brand_labels)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
