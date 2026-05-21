#!/usr/bin/env python3

import argparse
import json
from collections import deque
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

import pyds

from config import (
    CLASSIFIER_BACKEND,
    CLASSIFIER_ENGINE_PATH,
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_LABELS_PATH,
    CLASSIFIER_MIN_CONFIDENCE,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_OPERATE_ON_CLASS_IDS,
    DETECTOR_CLUSTER_MODE,
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_CUSTOM_LIB_PATH,
    DETECTOR_ENGINE_PATH,
    DETECTOR_INFER_DIMS,
    DETECTOR_INTERVAL,
    DETECTOR_LABELS_PATH,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS_IOU_THRESHOLD,
    DETECTOR_NETWORK_MODE,
    DETECTOR_NUM_CLASSES,
    DETECTOR_OUTPUT_BLOB_NAMES,
    DETECTOR_PARSE_BBOX_FUNC,
    DETECTION_LOG_API_URL,
    DETECTION_LOG_BATCH_SIZE,
    DETECTION_LOG_FLUSH_INTERVAL_SEC,
    DETECTION_LOG_MAX_RECENT,
    DETECTION_LOG_TIMEOUT_SEC,
    JPEG_QUALITY,
    JETSON_ID,
    MAX_RTSP_SOURCES,
    MUX_HEIGHT,
    MUX_TIMEOUT_USEC,
    MUX_WIDTH,
    OUTPUT_FPS,
    PRIMARY_GIE_ID,
    PROJECT_ROOT,
    SOURCE_BIN_FACTORY,
    RTSP_URLS,
    TILER_COLUMNS,
    TILER_HEIGHT,
    TILER_ROWS,
    TILER_WIDTH,
    VEHICLE_CLASS_IDS,
)


Gst.init(None)


def abs_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_text_labels(path):
    file_path = Path(abs_path(path))
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json_labels(path):
    file_path = Path(abs_path(path))
    if not file_path.exists():
        return {}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "classifier" in data and isinstance(data["classifier"], dict):
        data = data["classifier"]
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    return {}


def make_element(factory_name: str, name: str):
    element = Gst.ElementFactory.make(factory_name, name)
    if not element:
        raise RuntimeError(f"Could not create GStreamer element: {factory_name} name={name}")
    return element


def request_mux_sink_pad(mux, index: int):
    sinkpad = mux.get_request_pad(f"sink_{index}")
    if sinkpad:
        return sinkpad
    return mux.get_request_pad("sink_%u")


def encode_jpeg(frame_bgr: np.ndarray) -> Optional[bytes]:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
    )
    if not ok:
        return None
    return encoded.tobytes()


class SharedFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._updated_at = 0.0

    def set_jpeg(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg
            self._updated_at = time.time()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def age_sec(self) -> Optional[float]:
        with self._lock:
            if self._updated_at == 0.0:
                return None
            return time.time() - self._updated_at


class FpsMeter:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_ts = 0.0
        self._fps = 0.0

    def tick(self) -> float:
        now = time.time()
        with self._lock:
            if self._last_ts > 0:
                delta = now - self._last_ts
                if delta > 0:
                    instant_fps = 1.0 / delta
                    if self._fps <= 0:
                        self._fps = instant_fps
                    else:
                        self._fps = (self._fps * 0.9) + (instant_fps * 0.1)
            self._last_ts = now
            return self._fps


class CudaContextScope:
    _cuda = None
    _ctx = None
    _lock = threading.RLock()

    def __enter__(self):
        if self.__class__._ctx is None:
            import pycuda.driver as cuda

            cuda.init()
            self.__class__._cuda = cuda
            self.__class__._ctx = cuda.Device(0).make_context()
            cuda.Context.pop()
        self.__class__._lock.acquire()
        self.__class__._ctx.push()
        return self.__class__._cuda

    def __exit__(self, exc_type, exc, tb):
        self.__class__._cuda.Context.pop()
        self.__class__._lock.release()
        return False


class TensorRTBackend:
    def __init__(self, model_path: str, role: str = "model"):
        self.model_path = str(model_path)
        self.role = role
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_shape = []

        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with CudaContextScope() as cuda:
            with open(self.model_path, "rb") as handle:
                engine_bytes = handle.read()
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError(f"TensorRT failed to load engine: {self.model_path}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(f"TensorRT failed to create execution context: {self.model_path}")

            self.bindings = [None] * int(self.engine.num_bindings)
            for index in range(int(self.engine.num_bindings)):
                shape = tuple(int(v) for v in self.engine.get_binding_shape(index))
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                if any(v < 0 for v in shape):
                    raise RuntimeError(f"Dynamic TensorRT shapes are not supported here: {shape}")
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[index] = int(device_mem)
                item = {
                    "index": index,
                    "shape": shape,
                    "dtype": dtype,
                    "host": host_mem,
                    "device": device_mem,
                }
                if self.engine.binding_is_input(index):
                    self.inputs.append(item)
                    self.input_shape = [int(v) for v in shape]
                else:
                    self.outputs.append(item)
            if len(self.inputs) != 1:
                raise RuntimeError(f"Expected one input binding, got {len(self.inputs)}")
            self.stream = cuda.Stream()

    def run(self, tensor):
        tensor = np.ascontiguousarray(tensor.astype(self.inputs[0]["dtype"], copy=False))
        if tuple(tensor.shape) != tuple(self.inputs[0]["shape"]):
            raise ValueError(
                f"{self.role} input shape mismatch: got {tuple(tensor.shape)}, expected {tuple(self.inputs[0]['shape'])}"
            )

        with CudaContextScope() as cuda:
            np.copyto(self.inputs[0]["host"], tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError(f"TensorRT execute_async_v2 failed for {self.model_path}")
            for item in self.outputs:
                cuda.memcpy_dtoh_async(item["host"], item["device"], self.stream)
            self.stream.synchronize()
            return [item["host"].reshape(item["shape"]).copy() for item in self.outputs]

    def nchw_size(self, default: int) -> int:
        if len(self.input_shape) == 4 and self.input_shape[2] == self.input_shape[3] and self.input_shape[2] > 0:
            return int(self.input_shape[2])
        return int(default)


class RuntimeModel:
    def __init__(self, model_path: str, backend: str = "auto", input_size: int = 224, role: str = "model"):
        self.model_path = str(model_path)
        self.backend = backend
        self.input_size = int(input_size)
        self.role = role
        self.enabled = Path(self.model_path).exists()
        self.trt_model = None
        self.session = None
        self.net = None
        self.input_shape = []

        if not self.enabled:
            return

        suffix = Path(self.model_path).suffix.lower()
        if backend in ("auto", "trt") and suffix == ".engine":
            self.backend = "trt"
            self.trt_model = TensorRTBackend(self.model_path, role=role)
            self.input_shape = list(self.trt_model.input_shape)
            return

        if backend in ("auto", "trt") and suffix != ".onnx":
            candidate = Path(self.model_path).with_suffix(".engine")
            if candidate.exists():
                self.backend = "trt"
                self.trt_model = TensorRTBackend(str(candidate), role=role)
                self.input_shape = list(self.trt_model.input_shape)
                self.model_path = str(candidate)
                return
            if backend == "trt":
                raise RuntimeError(f"TensorRT backend requires a .engine file: {self.model_path}")

        if backend in ("auto", "ort") and suffix == ".onnx":
            try:
                import onnxruntime as ort

                providers = []
                for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
                    if provider in ort.get_available_providers():
                        providers.append(provider)
                if not providers:
                    providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                input_shape = self.session.get_inputs()[0].shape
                self.input_shape = [int(v) if isinstance(v, int) else int(self.input_size) for v in input_shape]
                self.backend = "ort"
                return
            except Exception as exc:
                if backend == "ort":
                    raise RuntimeError(f"onnxruntime backend failed for {self.role} ({self.model_path}): {exc}")

        if backend in ("auto", "opencv"):
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.backend = "opencv"
            return

        raise RuntimeError(f"Could not load model: {self.model_path}")

    def nchw_size(self, default: int) -> int:
        if self.trt_model is not None:
            return self.trt_model.nchw_size(default)
        if len(self.input_shape) == 4 and self.input_shape[2] == self.input_shape[3] and self.input_shape[2] > 0:
            return int(self.input_shape[2])
        return int(default)

    def run(self, tensor):
        if self.trt_model is not None:
            return self.trt_model.run(tensor)
        if self.session is not None:
            inputs = {self.session.get_inputs()[0].name: tensor.astype(np.float32, copy=False)}
            outputs = self.session.run(None, inputs)
            return [np.asarray(output) for output in outputs]
        if self.net is not None:
            self.net.setInput(tensor.astype(np.float32, copy=False))
            output_names = self.net.getUnconnectedOutLayersNames()
            outputs = self.net.forward(output_names)
            if isinstance(outputs, np.ndarray):
                return [outputs]
            return [np.asarray(output) for output in outputs]
        raise RuntimeError(f"Model is not loaded: {self.model_path}")


def preprocess_classifier(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.reshape(-1).astype(np.float32)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-12)


def resolve_label(label_map, class_id):
    return label_map.get(str(class_id), f"class_{class_id}")


def clamp_bbox(left, top, width, height, frame_w, frame_h):
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(frame_w, x1 + max(0, width))
    y2 = min(frame_h, y1 + max(0, height))
    return x1, y1, x2, y2


def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DetectionBatchLogger:
    def __init__(self, api_url, jetson_id, timeout_sec, max_recent):
        self.api_url = str(api_url).strip()
        self.jetson_id = str(jetson_id).strip()
        self.timeout_sec = float(timeout_sec)
        self.max_recent = int(max_recent)
        self.enabled = bool(self.api_url)
        self._lock = threading.Lock()
        self._recent = deque(maxlen=max(1, self.max_recent))
        self._pending = []
        self._last_post_at = 0.0
        self._last_error = ""

    def add(self, record):
        if not self.enabled:
            return
        with self._lock:
            self._recent.append(record)
            self._pending.append(record)

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "api_url": self.api_url,
                "jetson_id": self.jetson_id,
                "pending_count": len(self._pending),
                "last_post_at": self._last_post_at,
                "last_error": self._last_error,
                "recent": list(self._recent),
            }

    def flush(self, batch_size):
        if not self.enabled:
            return 0
        batch_size = max(1, int(batch_size))
        with self._lock:
            if not self._pending:
                return 0
            batch = self._pending[:batch_size]
            del self._pending[:batch_size]

        payload = json.dumps({"data": batch}).encode("utf-8")
        request = urlrequest.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(request, timeout=self.timeout_sec) as response:
                response.read()
        except Exception as exc:
            with self._lock:
                self._pending = batch + self._pending
                self._last_error = str(exc)
            return -1

        with self._lock:
            self._last_post_at = time.time()
            self._last_error = ""
        return len(batch)


def make_detection_record(class_name, rect, cam_idx, confidence, track_id=""):
    type_name = str(class_name or "object").strip().lower() or "object"
    left = float(rect.left)
    top = float(rect.top)
    width = float(rect.width)
    height = float(rect.height)
    record = {
        "timestamp": utc_timestamp(),
        "type": type_name,
        "color": "",
        "brand": "",
        "x": left,
        "y": top,
        "width": width,
        "height": height,
        "camera_id": "cam{0}".format(cam_idx + 1),
        "jetson_id": JETSON_ID,
        "track_id": "" if track_id is None else str(track_id),
    }
    if confidence is not None:
        record["confidence"] = float(confidence)
    return record


def build_detector_config() -> str:
    config_path = PROJECT_ROOT / "models" / "generated" / "config_infer_primary.txt"
    ensure_dir(config_path)

    model_path = Path(abs_path(DETECTOR_MODEL_PATH))
    model_is_engine = model_path.suffix.lower() == ".engine"
    if DETECTOR_ENGINE_PATH:
        engine_path = Path(abs_path(DETECTOR_ENGINE_PATH))
    elif model_is_engine:
        engine_path = model_path
    else:
        precision = {0: "fp32", 1: "int8", 2: "fp16"}.get(DETECTOR_NETWORK_MODE, "fp16")
        engine_path = model_path.with_name(f"{model_path.name}_b{len(RTSP_URLS)}_gpu0_{precision}.engine")

    lines = [
        "[property]",
        "gpu-id=0",
        f"batch-size={len(RTSP_URLS)}",
        f"model-engine-file={engine_path}",
        f"labelfile-path={abs_path(DETECTOR_LABELS_PATH)}",
        "net-scale-factor=0.00392156862745098",
        "model-color-format=0",
        f"infer-dims={DETECTOR_INFER_DIMS}",
        f"network-mode={DETECTOR_NETWORK_MODE}",
        f"interval={DETECTOR_INTERVAL}",
        f"gie-unique-id={PRIMARY_GIE_ID}",
        "process-mode=1",
        "network-type=0",
        f"num-detected-classes={DETECTOR_NUM_CLASSES}",
        f"cluster-mode={DETECTOR_CLUSTER_MODE}",
    ]

    if not model_is_engine:
        lines.insert(4, f"onnx-file={model_path}")

    if DETECTOR_CUSTOM_LIB_PATH:
        lines.append(f"custom-lib-path={abs_path(DETECTOR_CUSTOM_LIB_PATH)}")
    if DETECTOR_PARSE_BBOX_FUNC:
        lines.append(f"parse-bbox-func-name={DETECTOR_PARSE_BBOX_FUNC}")
    if DETECTOR_OUTPUT_BLOB_NAMES:
        lines.append(f"output-blob-names={DETECTOR_OUTPUT_BLOB_NAMES}")

    lines.extend(
        [
            "",
            "[class-attrs-all]",
            "topk=300",
            f"nms-iou-threshold={DETECTOR_NMS_IOU_THRESHOLD}",
            f"pre-cluster-threshold={DETECTOR_CONFIDENCE_THRESHOLD}",
            "",
        ]
    )

    config_path.write_text("\n".join(lines), encoding="utf-8")
    return str(config_path)


class CombinedDeepStreamPipeline:
    def __init__(self):
        if len(RTSP_URLS) > MAX_RTSP_SOURCES:
            raise ValueError(f"Supports up to {MAX_RTSP_SOURCES} RTSP sources")

        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self._mosaic_thread = None
        self._log_thread = None
        self._stop_event = threading.Event()
        self.shared_frame = SharedFrame()
        self.camera_frames = [SharedFrame() for _ in RTSP_URLS]
        self._fps_meters = {f"cam{i}": FpsMeter() for i in range(len(RTSP_URLS))}
        self._mosaic_fps = FpsMeter()
        self._source_bins = {}
        self._mux_sinkpads = {}
        self._detection_logger = DetectionBatchLogger(
            DETECTION_LOG_API_URL,
            JETSON_ID,
            DETECTION_LOG_TIMEOUT_SEC,
            DETECTION_LOG_MAX_RECENT,
        )
        self._classifier_labels = load_json_labels(CLASSIFIER_LABELS_PATH)
        self._detector_labels = load_text_labels(DETECTOR_LABELS_PATH)
        self._detector_label_map = {str(i): label for i, label in enumerate(self._detector_labels)}
        self._classify_class_ids = {
            int(item)
            for item in CLASSIFIER_OPERATE_ON_CLASS_IDS.split(";")
            if item.strip().isdigit()
        }
        if not self._classify_class_ids:
            self._classify_class_ids = set(VEHICLE_CLASS_IDS)
        classifier_model_path = CLASSIFIER_ENGINE_PATH if CLASSIFIER_ENGINE_PATH and Path(CLASSIFIER_ENGINE_PATH).exists() else CLASSIFIER_MODEL_PATH
        self.classifier = RuntimeModel(
            classifier_model_path,
            backend=CLASSIFIER_BACKEND,
            input_size=CLASSIFIER_INPUT_SIZE,
            role="classifier",
        )

    def _placeholder_mosaic(self, message):
        rows = max(1, TILER_ROWS)
        cols = max(1, TILER_COLUMNS)
        canvas = np.zeros((rows * MUX_HEIGHT, cols * MUX_WIDTH, 3), dtype=np.uint8)
        draw_label(canvas, message, 18, 38, (90, 90, 90))
        return encode_jpeg(canvas)

    def _build_source_bin(self, index: int, uri: str):
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            raise RuntimeError(f"Unable to create source bin {bin_name}")

        source_factory = SOURCE_BIN_FACTORY or "uridecodebin"
        if source_factory not in ("uridecodebin", "nvurisrcbin"):
            source_factory = "uridecodebin"

        source = Gst.ElementFactory.make(source_factory, f"uri-decode-bin-{index}")
        if not source:
            raise RuntimeError(f"{source_factory} is required but was not found")
        source.set_property("uri", uri)
        try:
            source.set_property("drop-on-latency", True)
        except Exception:
            pass
        try:
            source.set_property("latency", 150)
        except Exception:
            pass
        try:
            source.set_property("select-rtp-protocol", 4)
        except Exception:
            pass

        queue = make_element("queue", f"source-queue-{index}")
        queue.set_property("max-size-buffers", 4)
        queue.set_property("leaky", 2)

        convert = make_element("nvvideoconvert", f"source-convert-{index}")
        caps = make_element("capsfilter", f"source-caps-{index}")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        source.connect("pad-added", self._decodebin_pad_added, nbin)
        nbin.add(source)
        nbin.add(queue)
        nbin.add(convert)
        nbin.add(caps)

        if not queue.link(convert):
            raise RuntimeError(f"Failed to link source queue in {bin_name}")
        if not convert.link(caps):
            raise RuntimeError(f"Failed to link source converter in {bin_name}")

        ghost_pad = Gst.GhostPad.new("src", caps.get_static_pad("src"))
        if not ghost_pad:
            raise RuntimeError("Failed to create ghost pad")
        nbin.add_pad(ghost_pad)
        return nbin

    def _decodebin_pad_added(self, decodebin, pad, source_bin):
        caps = pad.get_current_caps() or pad.query_caps(None)
        structure_name = caps.get_structure(0).get_name()
        if not structure_name.startswith("video"):
            return

        source_name = source_bin.get_name()
        suffix = source_name.rsplit("-", 1)[-1]
        queue = source_bin.get_by_name(f"source-queue-{suffix}")
        if not queue:
            print(f"[ERROR] Source queue not found for {source_name}")
            return

        sink_pad = queue.get_static_pad("sink")
        if not sink_pad:
            print(f"[ERROR] Source queue sink pad not found for {source_name}")
            return

        if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
            print(f"[INFO] Linked source pad for {source_name}: {caps.to_string()}")
        else:
            print(f"[ERROR] Failed to link decodebin pad for {source_name}: {caps.to_string()}")

    def _make_convert_chain(self, suffix: str, width: int, height: int):
        nvvidconv = make_element("nvvideoconvert", f"nvvconv-{suffix}")
        caps_rgba = make_element("capsfilter", f"caps-rgba-{suffix}")
        caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))

        videoconv = make_element("videoconvert", f"vconv-{suffix}")
        caps_bgr = make_element("capsfilter", f"caps-bgr-{suffix}")
        caps_bgr.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw, format=BGR, width={width}, height={height}"),
        )
        return [nvvidconv, caps_rgba, videoconv, caps_bgr]

    def _make_appsink(self, name: str, callback):
        sink = make_element("appsink", name)
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.connect("new-sample", callback)
        return sink

    def _overlay_fps(self, frame_bgr: np.ndarray, fps: float, label: str) -> np.ndarray:
        fps_text = f"{label} FPS: --" if fps <= 0 else f"{label} FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        padding = 8
        x = 12
        y = 12 + text_size[1]
        cv2.rectangle(
            frame_bgr,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + baseline + padding),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(frame_bgr, fps_text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        return frame_bgr

    def _classify_crop(self, crop_bgr):
        if not self.classifier.enabled or crop_bgr.size == 0:
            return None, None
        tensor = preprocess_classifier(crop_bgr, self.classifier.nchw_size(CLASSIFIER_INPUT_SIZE))
        outputs = self.classifier.run(tensor)
        if not outputs:
            return None, None
        logits = outputs[0]
        probs = softmax(logits)
        brand_id = int(np.argmax(probs))
        confidence = float(probs[brand_id])
        if confidence < CLASSIFIER_MIN_CONFIDENCE:
            return None, confidence
        brand_name = self._classifier_labels.get(str(brand_id), f"class_{brand_id}")
        return brand_name, confidence

    def _queue_detection_log(self, class_name, rect, cam_idx, confidence, track_id=""):
        record = make_detection_record(class_name, rect, cam_idx, confidence, track_id=track_id)
        self._detection_logger.add(record)

    def _log_worker(self):
        delay = max(0.2, float(DETECTION_LOG_FLUSH_INTERVAL_SEC))
        while not self._stop_event.is_set():
            try:
                while True:
                    sent = self._detection_logger.flush(DETECTION_LOG_BATCH_SIZE)
                    if sent <= 0:
                        break
            except Exception as exc:
                print(f"[ERROR] detection log worker failed: {exc!r}")
            time.sleep(delay)

    def _render_objects(self, frame_bgr: np.ndarray, frame_meta) -> np.ndarray:
        frame_h, frame_w = frame_bgr.shape[:2]
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            class_id = int(obj_meta.class_id)
            class_name = getattr(obj_meta, "label", "") or resolve_label(self._detector_label_map, class_id)
            rect = obj_meta.rect_params
            x1, y1, x2, y2 = clamp_bbox(
                int(rect.left),
                int(rect.top),
                int(rect.width),
                int(rect.height),
                frame_w,
                frame_h,
            )
            if x2 <= x1 or y2 <= y1:
                l_obj = l_obj.next
                continue

            if class_id in self._classify_class_ids:
                crop = frame_bgr[y1:y2, x1:x2]
                brand, confidence = self._classify_crop(crop)
            else:
                brand, confidence = None, None

            self._queue_detection_log(class_name, rect, int(frame_meta.source_id), confidence, getattr(obj_meta, "object_id", ""))

            color = (40, 180, 40)
            if class_id == 7:
                color = (30, 120, 220)
            elif class_id == 5:
                color = (220, 140, 30)
            elif class_id == 3:
                color = (180, 60, 200)

            label = f"{class_name}"
            if brand:
                label += f" | {brand} {confidence:.2f}"
            elif confidence is not None:
                label += f" | {confidence:.2f}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            draw_label(frame_bgr, label, x1, y1, color)

            l_obj = l_obj.next

        return frame_bgr

    def _on_camera_sample(self, sink, cam_idx: int):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            arr = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = height * width * 3
            if arr.size < expected:
                return Gst.FlowReturn.OK
            frame_bgr = arr[:expected].reshape((height, width, 3)).copy()

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if batch_meta:
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    except StopIteration:
                        break
                    if int(frame_meta.source_id) != cam_idx:
                        l_frame = l_frame.next
                        continue
                    frame_bgr = self._render_objects(frame_bgr, frame_meta)
                    l_frame = l_frame.next

            fps = self._fps_meters[f"cam{cam_idx}"].tick()
            frame_bgr = self._overlay_fps(frame_bgr, fps, f"Cam{cam_idx + 1}")
            jpg = encode_jpeg(frame_bgr)
            if jpg:
                self.camera_frames[cam_idx].set_jpeg(jpg)

        except Exception as exc:
            print(f"[ERROR] cam{cam_idx} appsink failed: {exc!r}")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _build_mosaic(self) -> Optional[bytes]:
        frames = []
        for idx, shared in enumerate(self.camera_frames):
            jpeg = shared.get_jpeg()
            if not jpeg:
                empty = np.zeros((MUX_HEIGHT, MUX_WIDTH, 3), dtype=np.uint8)
                draw_label(empty, f"Cam{idx + 1} waiting", 12, 40, (80, 80, 80))
                frames.append(empty)
                continue
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                frame = np.zeros((MUX_HEIGHT, MUX_WIDTH, 3), dtype=np.uint8)
            frames.append(frame)

        if not frames:
            return self._placeholder_mosaic("Waiting for camera frames")

        cell_w = MUX_WIDTH
        cell_h = MUX_HEIGHT
        rows = max(1, TILER_ROWS)
        cols = max(1, TILER_COLUMNS)
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for index, frame in enumerate(frames):
            row = index // cols
            col = index % cols
            if row >= rows:
                break
            resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = resized

        fps = self._mosaic_fps.tick()
        canvas = self._overlay_fps(canvas, fps, "FPS")
        return encode_jpeg(canvas)

    def _mosaic_worker(self):
        delay = 1.0 / max(1, OUTPUT_FPS)
        while not self._stop_event.is_set():
            try:
                jpg = self._build_mosaic()
                if jpg:
                    self.shared_frame.set_jpeg(jpg)
            except Exception as exc:
                print(f"[ERROR] mosaic worker failed: {exc!r}")
                fallback = self._placeholder_mosaic("Mosaic error - check logs")
                if fallback:
                    self.shared_frame.set_jpeg(fallback)
            time.sleep(delay)

    def build(self):
        if len(RTSP_URLS) > MAX_RTSP_SOURCES:
            raise ValueError(f"DetectPipeline supports up to {MAX_RTSP_SOURCES} RTSP sources")

        self.pipeline = Gst.Pipeline.new("deepstream-semifinal-pipeline")
        if not self.pipeline:
            raise RuntimeError("Could not create pipeline")

        streammux = make_element("nvstreammux", "stream-muxer")
        streammux.set_property("width", MUX_WIDTH)
        streammux.set_property("height", MUX_HEIGHT)
        streammux.set_property("batch-size", len(RTSP_URLS))
        streammux.set_property("batched-push-timeout", MUX_TIMEOUT_USEC)
        streammux.set_property("live-source", 1)
        self.pipeline.add(streammux)

        for index, uri in enumerate(RTSP_URLS):
            source_bin = self._build_source_bin(index, uri)
            self.pipeline.add(source_bin)
            self._source_bins[index] = source_bin
            sinkpad = request_mux_sink_pad(streammux, index)
            if not sinkpad:
                raise RuntimeError(f"Unable to get streammux sink pad for source {index}")
            self._mux_sinkpads[index] = sinkpad
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(f"Unable to get source bin src pad for source {index}")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source {index} to streammux")

        pgie = make_element("nvinfer", "primary-inference")
        infer_config_path = build_detector_config()
        pgie.set_property("config-file-path", infer_config_path)
        self.pipeline.add(pgie)

        pre_osd_conv = make_element("nvvideoconvert", "pre-osd-converter")
        pre_osd_caps = make_element("capsfilter", "pre-osd-caps")
        pre_osd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self.pipeline.add(pre_osd_conv)
        self.pipeline.add(pre_osd_caps)

        nvosd = make_element("nvdsosd", "onscreendisplay")
        try:
            nvosd.set_property("process-mode", 0)
        except Exception:
            pass
        self.pipeline.add(nvosd)

        demux = make_element("nvstreamdemux", "stream-demux")
        self.pipeline.add(demux)

        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> pgie")
        if not pgie.link(pre_osd_conv):
            raise RuntimeError("Failed to link pgie -> pre-osd-converter")
        if not pre_osd_conv.link(pre_osd_caps):
            raise RuntimeError("Failed to link pre-osd-converter -> pre-osd-caps")
        if not pre_osd_caps.link(nvosd):
            raise RuntimeError("Failed to link pre-osd-caps -> nvosd")
        if not nvosd.link(demux):
            raise RuntimeError("Failed to link nvosd -> demux")

        for index in range(len(RTSP_URLS)):
            demux_src = demux.get_request_pad(f"src_{index}")
            if not demux_src:
                raise RuntimeError(f"nvstreamdemux: cannot get src_{index}")

            branch_queue = make_element("queue", f"q-cam{index}")
            branch_queue.set_property("max-size-buffers", 2)
            branch_queue.set_property("leaky", 2)

            convert_els = self._make_convert_chain(f"cam{index}", MUX_WIDTH, MUX_HEIGHT)
            cam_sink = self._make_appsink(f"appsink-cam{index}", lambda sink, idx=index: self._on_camera_sample(sink, idx))

            for element in [branch_queue] + convert_els + [cam_sink]:
                self.pipeline.add(element)

            chain = [branch_queue] + convert_els + [cam_sink]
            for left, right in zip(chain[:-1], chain[1:]):
                if not left.link(right):
                    raise RuntimeError(f"Failed to link {left.get_name()} -> {right.get_name()}")

            if demux_src.link(branch_queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux src_{index} -> q-cam{index}")

        print(f"[INFO] Built DeepStream pipeline with {len(RTSP_URLS)} RTSP source(s)")
        print(f"[INFO] Detector config: {infer_config_path}")
        print(f"[INFO] Detector model: {abs_path(DETECTOR_MODEL_PATH)}")
        print(f"[INFO] Classifier model: {abs_path(CLASSIFIER_MODEL_PATH)}")
        return self.pipeline

    def start(self):
        self.build()
        self.loop = GLib.MainLoop()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self.loop)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set pipeline to PLAYING")

        self._stop_event.clear()
        if not self.shared_frame.get_jpeg():
            placeholder = self._build_mosaic()
            if placeholder:
                self.shared_frame.set_jpeg(placeholder)
        self._mosaic_thread = threading.Thread(target=self._mosaic_worker, daemon=True)
        self._mosaic_thread.start()
        self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._log_thread.start()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()
        print("[INFO] DeepStream pipeline started")

    def stop(self):
        self._stop_event.set()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        print("[INFO] Pipeline stopped")

    def _bus_call(self, bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            print("[INFO] End-of-stream")
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("[ERROR] GStreamer error:", err)
            print("[ERROR] Debug:", debug)
            loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print("[WARN] GStreamer warning:", err)
            print("[WARN] Debug:", debug)
        return True

    def get_jpeg(self) -> Optional[bytes]:
        return self.shared_frame.get_jpeg()

    def get_camera_jpeg(self, cam_idx: int) -> Optional[bytes]:
        if 0 <= cam_idx < len(self.camera_frames):
            return self.camera_frames[cam_idx].get_jpeg()
        return None

    def get_detection_logs(self):
        return self._detection_logger.snapshot()


def _multipart_generator(frame_getter, delay_sec: float):
    while True:
        frame = frame_getter()
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(delay_sec)


def create_app():
    from fastapi import FastAPI, Path
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="DeepStream Semifinal")
    state = {
        "mode": "mosaic",
        "started_at": time.time(),
        "pipeline": None,
    }

    @app.get("/health")
    def health():
        pipeline = state["pipeline"]
        return {
            "ok": True,
            "uptime_sec": round(time.time() - state["started_at"], 2),
            "source_count": len(RTSP_URLS),
            "has_frame": pipeline.get_jpeg() is not None if pipeline else False,
        }

    @app.get("/monitor", response_class=HTMLResponse)
    def monitor():
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepStream Semifinal</title>
  <style>
    body {{ margin: 0; background: #0b1020; color: #e5e7eb; font-family: Arial, sans-serif; }}
    header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 18px; background: #111827; border-bottom: 1px solid #243244; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(320px, 1fr); gap: 16px; padding: 16px; }}
    section {{ background: #0f172a; border: 1px solid #243244; border-radius: 16px; overflow: hidden; }}
    img {{ width: 100%; display: block; background: #020617; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; padding: 12px; }}
    .card {{ border: 1px solid #243244; border-radius: 12px; overflow: hidden; background: #020617; }}
    .card h3 {{ margin: 0; padding: 8px 10px; font-size: 13px; background: #111827; border-bottom: 1px solid #243244; }}
    .pill {{ padding: 4px 10px; border-radius: 999px; background: #14532d; color: #dcfce7; font-size: 12px; }}
    @media (max-width: 1000px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
        <header>
        <strong>DeepStream Semifinal</strong>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
            <span class="pill">sources: {len(RTSP_URLS)}</span>
            <a class="pill" href="/logs/detect" style="text-decoration:none; background:#1d4ed8; color:#eff6ff;">detect logs</a>
        </div>
    </header>
  <main>
    <section>
      <img src="/video/mosaic" alt="mosaic stream">
    </section>
    <section>
      <div class="grid">
        {''.join(f'<div class="card"><h3>Cam {index + 1}</h3><img src="/video/cam/{index + 1}" alt="cam {index + 1}"></div>' for index in range(len(RTSP_URLS)))}
      </div>
    </section>
  </main>
</body>
</html>
"""

    @app.get("/video/mosaic")
    def video_mosaic():
        pipeline = state["pipeline"]
        if not pipeline:
            return JSONResponse({"error": "pipeline not started"}, status_code=400)
        return StreamingResponse(
            _multipart_generator(pipeline.get_jpeg, 1.0 / max(1, OUTPUT_FPS)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video/cam/{cam_id}")
    def video_cam(cam_id: int = Path(ge=1, le=len(RTSP_URLS))):
        pipeline = state["pipeline"]
        if not pipeline:
            return JSONResponse({"error": "pipeline not started"}, status_code=400)
        return StreamingResponse(
            _multipart_generator(lambda: pipeline.get_camera_jpeg(cam_id - 1), 1.0 / max(1, OUTPUT_FPS)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

        @app.get("/api/logs/detect")
        def detect_logs_json():
                pipeline = state["pipeline"]
                if not pipeline:
                        return JSONResponse({"error": "pipeline not started"}, status_code=400)
                return JSONResponse(pipeline.get_detection_logs())

        @app.get("/logs/detect", response_class=HTMLResponse)
        def detect_logs_page():
                pipeline = state["pipeline"]
                if not pipeline:
                        return HTMLResponse("<html><body><h3>pipeline not started</h3></body></html>", status_code=400)
                snapshot = pipeline.get_detection_logs()
                rows = []
                for item in reversed(snapshot.get("recent", [])):
                        rows.append(
                                "<tr>"
                                "<td>{timestamp}</td><td>{camera_id}</td><td>{type}</td><td>{x:.1f}</td><td>{y:.1f}</td>"
                                "<td>{width:.1f}</td><td>{height:.1f}</td><td>{brand}</td><td>{color}</td><td>{track_id}</td>"
                                "</tr>".format(
                                        timestamp=item.get("timestamp", ""),
                                        camera_id=item.get("camera_id", ""),
                                        type=item.get("type", ""),
                                        x=float(item.get("x", 0.0)),
                                        y=float(item.get("y", 0.0)),
                                        width=float(item.get("width", 0.0)),
                                        height=float(item.get("height", 0.0)),
                                        brand=item.get("brand", ""),
                                        color=item.get("color", ""),
                                        track_id=item.get("track_id", ""),
                                )
                        )
                table_rows = "".join(rows) or "<tr><td colspan='10'>No detections yet</td></tr>"
                return f"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="3">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Detection Logs</title>
    <style>
        body {{ margin: 0; padding: 18px; background: #0b1020; color: #e5e7eb; font-family: Arial, sans-serif; }}
        a {{ color: #93c5fd; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #0f172a; }}
        th, td {{ border: 1px solid #243244; padding: 8px; font-size: 12px; text-align: left; }}
        th {{ background: #111827; }}
        .meta {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; color: #cbd5e1; }}
        .pill {{ padding: 4px 10px; border-radius: 999px; background: #111827; border: 1px solid #243244; }}
    </style>
</head>
<body>
    <div><a href="/monitor">Back to monitor</a></div>
    <h2>Detection Logs</h2>
    <div class="meta">
        <span class="pill">enabled: {snapshot.get('enabled')}</span>
        <span class="pill">pending: {snapshot.get('pending_count')}</span>
        <span class="pill">jetson: {snapshot.get('jetson_id', '')}</span>
        <span class="pill">api: {snapshot.get('api_url', '')}</span>
        <span class="pill">last error: {snapshot.get('last_error', '') or 'none'}</span>
    </div>
    <table>
        <thead>
            <tr>
                <th>timestamp</th>
                <th>camera_id</th>
                <th>type</th>
                <th>x</th>
                <th>y</th>
                <th>width</th>
                <th>height</th>
                <th>brand</th>
                <th>color</th>
                <th>track_id</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""

    @app.on_event("startup")
    def _startup():
        pipeline = CombinedDeepStreamPipeline()
        state["pipeline"] = pipeline
        state["started_at"] = time.time()
        pipeline.start()

    @app.on_event("shutdown")
    def _shutdown():
        pipeline = state.get("pipeline")
        if pipeline:
            pipeline.stop()

    return app


def main():
    parser = argparse.ArgumentParser(description="DeepStream semifinal combined pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

