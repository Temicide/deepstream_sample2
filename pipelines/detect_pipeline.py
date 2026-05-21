import time
import threading
import json
import queue
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

import vehicle_color_rule

from config import (
    CLASSIFIER_BACKEND,
    CLASSIFIER_CACHE_MAX_SIZE,
    CLASSIFIER_CACHE_TTL_SEC,
    CLASSIFIER_ENGINE_PATH,
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_LABELS_PATH,
    CLASSIFIER_MAX_PER_CAMERA_FRAME,
    CLASSIFIER_MIN_CONFIDENCE,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_OPERATE_ON_CLASS_IDS,
    DETECTION_MIN_CONFIDENCE,
    DETECTION_POST_INTERVAL_SEC,
    DETECTION_POST_TIMEOUT_SEC,
    DETECTION_SERVER_URL,
    JETSON_ID,
    MAX_RTSP_SOURCES,
    MUX_HEIGHT,
    MUX_WIDTH,
    RTSP_URLS,
    TRACKER_COMPUTE_HW,
    TRACKER_CONFIG_PATH,
    TRACKER_DISPLAY_ID,
    TRACKER_ENABLE,
    TRACKER_GPU_ID,
    TRACKER_HEIGHT,
    TRACKER_LIB_FILE,
    TRACKER_WIDTH,
    VEHICLE_CLASS_IDS,
)
from pipelines.common import BaseDeepStreamPipeline, encode_jpeg, make_element, set_optional_property
from pipelines.classifier_runtime import RuntimeModel, classify_crop, load_classifier_labels
from pipelines.model_config import abs_project_path, ensure_primary_infer_config


def _parse_class_ids(value) -> set:
    if isinstance(value, str):
        raw_items = value.replace(",", ";").split(";")
    else:
        raw_items = list(value or [])

    class_ids = set()
    for item in raw_items:
        try:
            class_ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return class_ids


class DetectionJsonPublisher:
    def __init__(self, url: str, timeout_sec: float, min_interval_sec: float):
        self.url = url.strip()
        self.timeout_sec = timeout_sec
        self.min_interval_sec = min_interval_sec
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._last_post_ts = 0.0
        self._last_error = ""
        self._last_ok_ts = 0.0
        self._last_item_count = 0

    def start(self):
        if not self.url or self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[INFO] Detection JSON publisher enabled: {self.url}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def publish_latest(self, payload: Dict[str, Any]):
        if not self.url:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.url),
            "url": self.url,
            "last_ok_ts": self._last_ok_ts,
            "last_error": self._last_error,
            "last_item_count": self._last_item_count,
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            elapsed = time.time() - self._last_post_ts
            if elapsed < self.min_interval_sec:
                time.sleep(self.min_interval_sec - elapsed)

            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    response.read()
                self._last_ok_ts = time.time()
                self._last_error = ""
                self._last_item_count = len(payload.get("data", []))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self._last_error = repr(exc)
                print(f"[WARN] Detection JSON POST failed: {self._last_error}")
            finally:
                self._last_post_ts = time.time()


class DetectPipeline(BaseDeepStreamPipeline):
    """
    Phase 3:
    RTSP x5 -> decode -> nvstreammux -> nvinfer -> optional nvtracker -> tiler -> nvdsosd -> appsink

    เพิ่ม /detections โดยอ่าน metadata หลัง tracker เพื่อให้ track_id ต่อเนื่องเมื่อเปิด tracker
    """

    def __init__(self):
        super().__init__(mode_name="detect")
        if len(RTSP_URLS) > MAX_RTSP_SOURCES:
            raise ValueError(f"DetectPipeline supports up to {MAX_RTSP_SOURCES} RTSP sources")
        self._det_lock = threading.Lock()
        self._detections: Dict[str, Any] = {
            "timestamp": 0,
            "cameras": {},
        }
        self._publisher = DetectionJsonPublisher(
            DETECTION_SERVER_URL,
            DETECTION_POST_TIMEOUT_SEC,
            DETECTION_POST_INTERVAL_SEC,
        )
        self._tracker_enabled = TRACKER_ENABLE
        self._cam_color_lock = threading.Lock()
        self._cam_color_queue: Dict[int, Dict] = {}
        self._cam_color_enriched: Dict[int, tuple] = {}
        self._color_cycle = 0
        self._classify_class_ids = _parse_class_ids(CLASSIFIER_OPERATE_ON_CLASS_IDS) or set(VEHICLE_CLASS_IDS)
        self._classifier_labels = load_classifier_labels(abs_project_path(CLASSIFIER_LABELS_PATH))
        classifier_model_path = CLASSIFIER_ENGINE_PATH.strip() or CLASSIFIER_MODEL_PATH
        self._classifier = RuntimeModel(
            abs_project_path(classifier_model_path),
            backend=CLASSIFIER_BACKEND,
            input_size=CLASSIFIER_INPUT_SIZE,
            role="classifier",
        )
        self._brand_cache_lock = threading.Lock()
        self._brand_cache: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        self._classifier_last_error = self._classifier.error
        self._classifier_run_count = 0
        self._classifier_cache_hits = 0
        if self._classifier.enabled:
            print(
                "[INFO] Vehicle brand classifier enabled: "
                f"backend={self._classifier.backend} model={self._classifier.model_path}"
            )
        else:
            print(f"[WARN] Vehicle brand classifier disabled: {self._classifier.error}")

    def _create_tracker(self):
        tracker = make_element("nvtracker", "object-tracker")
        tracker.set_property("ll-lib-file", TRACKER_LIB_FILE)
        tracker.set_property("ll-config-file", abs_project_path(TRACKER_CONFIG_PATH))
        tracker.set_property("tracker-width", TRACKER_WIDTH)
        tracker.set_property("tracker-height", TRACKER_HEIGHT)
        tracker.set_property("display-tracking-id", TRACKER_DISPLAY_ID)
        tracker.set_property("gpu-id", TRACKER_GPU_ID)
        set_optional_property(tracker, "compute-hw", TRACKER_COMPUTE_HW)
        return tracker

    def _attach_detection_probe(self, element):
        srcpad = element.get_static_pad("src")
        if not srcpad:
            raise RuntimeError(f"Unable to get {element.get_name()} src pad")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._tracked_src_pad_buffer_probe, None)

    def _attach_osd_style_probe(self, element):
        sinkpad = element.get_static_pad("sink")
        if not sinkpad:
            raise RuntimeError(f"Unable to get {element.get_name()} sink pad")
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self._osd_sink_pad_buffer_probe, None)

    @staticmethod
    def _configure_leaky_queue(element):
        element.set_property("max-size-buffers", 2)
        element.set_property("leaky", 2)

    @staticmethod
    def _link_tee_to_queue(tee, queue_element, branch_name: str):
        tee_src = tee.get_request_pad("src_%u")
        if not tee_src:
            raise RuntimeError(f"Unable to get tee src pad for {branch_name} branch")
        queue_sink = queue_element.get_static_pad("sink")
        if tee_src.link(queue_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link tee -> {queue_element.get_name()}")

    def _build_mosaic_osd_branch(
        self,
        tee,
        rows: int,
        columns: int,
        width: int,
        height: int,
    ):
        q_mosaic = make_element("queue", "q-mosaic-osd")
        self._configure_leaky_queue(q_mosaic)

        tiler = make_element("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows", rows)
        tiler.set_property("columns", columns)
        tiler.set_property("width", width)
        tiler.set_property("height", height)

        nvvidconv_preosd = make_element("nvvideoconvert", "pre-osd-converter")
        caps_rgba_preosd = make_element("capsfilter", "caps-rgba-preosd")
        caps_rgba_preosd.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"),
        )

        nvosd = make_element("nvdsosd", "onscreendisplay")
        set_optional_property(nvosd, "process-mode", 0)
        set_optional_property(nvosd, "display-bbox", True)
        set_optional_property(nvosd, "display-text", False)
        set_optional_property(nvosd, "display-mask", False)
        self._attach_osd_style_probe(nvosd)

        mosaic_conv, _, _ = self._make_convert_chain(
            suffix="mosaic", width=width, height=height
        )
        mosaic_sink = self._make_appsink("appsink-mosaic", self._on_new_sample)

        for el in [q_mosaic, tiler, nvvidconv_preosd, caps_rgba_preosd, nvosd] + mosaic_conv + [mosaic_sink]:
            self.pipeline.add(el)

        self._link_tee_to_queue(tee, q_mosaic, "mosaic")

        mosaic_chain = [q_mosaic, tiler, nvvidconv_preosd, caps_rgba_preosd, nvosd] + mosaic_conv + [mosaic_sink]
        for a, b in zip(mosaic_chain[:-1], mosaic_chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

        print("[INFO] Built mosaic branch (tracker -> OSD, no remux)")

    def _build_per_camera_branch(self, tee):
        q_demux = make_element("queue", "q-per-camera-demux")
        self._configure_leaky_queue(q_demux)
        demux = make_element("nvstreamdemux", "nvstream-demux")

        self.pipeline.add(q_demux)
        self.pipeline.add(demux)

        self._link_tee_to_queue(tee, q_demux, "per-camera")
        if not q_demux.link(demux):
            raise RuntimeError("Failed to link q-per-camera-demux -> nvstreamdemux")

        for i in range(len(RTSP_URLS)):
            demux_src = demux.get_request_pad(f"src_{i}")
            if not demux_src:
                raise RuntimeError(f"nvstreamdemux: cannot get src_{i}")

            q_cam = make_element("queue", f"q-cam{i}")
            self._configure_leaky_queue(q_cam)

            conv_els, _, _ = self._make_convert_chain(
                suffix=f"cam{i}", width=MUX_WIDTH, height=MUX_HEIGHT
            )

            def _cam_cb(sink, idx=i):
                return self._on_camera_sample(sink, idx)

            cam_sink = self._make_appsink(f"appsink-cam{i}", _cam_cb)

            for el in [q_cam] + conv_els + [cam_sink]:
                self.pipeline.add(el)

            if demux_src.link(q_cam.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux src_{i} -> q-cam{i}")

            cam_chain = [q_cam] + conv_els + [cam_sink]
            for a, b in zip(cam_chain[:-1], cam_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

            print(f"[INFO] Built per-camera branch for cam {i}")

    def build(self):
        """
        Override build เพื่อแทรก nvinfer ก่อน nvstreamdemux
        และให้ mosaic OSD รับ metadata ตรงจาก tracker ก่อนแยกเป็น per-camera
        """
        self.pipeline = Gst.Pipeline.new(f"deepstream-{self.mode_name}-pipeline")
        if not self.pipeline:
            raise RuntimeError("Could not create pipeline")

        from config import (
            MUX_WIDTH, MUX_HEIGHT, MUX_BATCH_SIZE, MUX_TIMEOUT_USEC,
            TILER_ROWS, TILER_COLUMNS, TILER_WIDTH, TILER_HEIGHT,
        )

        # ── nvstreammux ──────────────────────────────────────────────
        streammux = make_element("nvstreammux", "stream-muxer")
        streammux.set_property("width", MUX_WIDTH)
        streammux.set_property("height", MUX_HEIGHT)
        streammux.set_property("batch-size", MUX_BATCH_SIZE)
        streammux.set_property("batched-push-timeout", MUX_TIMEOUT_USEC)
        streammux.set_property("live-source", 1)
        set_optional_property(streammux, "enable-padding", True)
        self.pipeline.add(streammux)
        self._build_sources(streammux)

        # ── nvinfer ──────────────────────────────────────────────────
        infer_config_path = ensure_primary_infer_config()
        pgie = make_element("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", infer_config_path)
        self.pipeline.add(pgie)
        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> pgie")

        pre_demux = pgie
        if TRACKER_ENABLE:
            tracker = self._create_tracker()
            self.pipeline.add(tracker)
            if not pgie.link(tracker):
                raise RuntimeError("Failed to link pgie -> tracker")
            pre_demux = tracker
            print(f"[INFO] nvtracker enabled: {TRACKER_CONFIG_PATH}")
        else:
            print("[INFO] nvtracker disabled")
        self._attach_detection_probe(pre_demux)

        # ── split tracked metadata: mosaic OSD keeps original batch metadata
        #    while per-camera appsinks still receive individual decoded streams.
        tee = make_element("tee", "tracked-output-tee")
        self.pipeline.add(tee)
        if not pre_demux.link(tee):
            raise RuntimeError(f"Failed to link {pre_demux.get_name()} -> tracked-output-tee")

        self._build_mosaic_osd_branch(
            tee,
            rows=TILER_ROWS,
            columns=TILER_COLUMNS,
            width=TILER_WIDTH,
            height=TILER_HEIGHT,
        )
        self._build_per_camera_branch(tee)

        print(f"[INFO] nvinfer config: {infer_config_path}")
        return self.pipeline

    def _tracked_src_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        self._color_cycle += 1
        cycle = self._color_cycle

        now = time.time()
        raw_data_batch = {"data": []}
        result = {
            "timestamp": now,
            "pipeline": self.mode_name,
            "source_count": len(RTSP_URLS),
            "cameras": {},
        }

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = int(frame_meta.source_id)
            cam_key = str(camera_id)

            cam_result = {
                "camera_id": camera_id + 1,
                "source_id": camera_id,
                "rtsp_url": RTSP_URLS[camera_id] if camera_id < len(RTSP_URLS) else "",
                "frame_num": int(frame_meta.frame_num),
                "ntp_timestamp": int(getattr(frame_meta, "ntp_timestamp", 0)),
                "objects": [],
            }

            cam_entries = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                rect = obj_meta.rect_params
                confidence = float(obj_meta.confidence)
                tracker_confidence = float(getattr(obj_meta, "tracker_confidence", -1.0))
                obj_class_id = int(obj_meta.class_id)
                if self._should_skip_object(confidence):
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                    continue
                output_confidence = self._output_confidence(confidence, tracker_confidence)

                label = ""
                try:
                    label = obj_meta.obj_label
                except Exception:
                    label = str(obj_class_id)

                bbox = {
                    "left": float(rect.left),
                    "top": float(rect.top),
                    "width": float(rect.width),
                    "height": float(rect.height),
                }
                raw_bbox = self._scale_bbox_to_source_frame(frame_meta, bbox)
                detection_uuid = str(uuid.uuid4())
                stable_track_id = self._format_track_id(
                    getattr(obj_meta, "object_id", None),
                    allow_zero=self._tracker_enabled,
                )
                track_id = stable_track_id or detection_uuid
                cached_color = ""
                cached_brand = ""
                cached_brand_confidence = None
                if stable_track_id and obj_class_id in VEHICLE_CLASS_IDS:
                    cached, fresh = self._get_cached_brand(
                        self._brand_cache_key(camera_id, obj_class_id, stable_track_id)
                    )
                    if cached and fresh:
                        cached_color = cached.get("color", "") or ""
                        cached_brand = cached.get("brand", "") or ""
                        cached_brand_confidence = cached.get("confidence")

                object_result = {
                    "class_id": obj_class_id,
                    "label": label,
                    "confidence": output_confidence,
                    "detector_confidence": confidence,
                    "tracker_confidence": tracker_confidence,
                    "uuid": detection_uuid,
                    "track_id": track_id,
                    "bbox": bbox,
                    "bbox_norm": {
                        "left": bbox["left"] / max(1, MUX_WIDTH),
                        "top": bbox["top"] / max(1, MUX_HEIGHT),
                        "width": bbox["width"] / max(1, MUX_WIDTH),
                        "height": bbox["height"] / max(1, MUX_HEIGHT),
                    },
                    "color": cached_color,
                    "brand": cached_brand,
                    "brand_confidence": cached_brand_confidence,
                }
                cam_result["objects"].append(object_result)
                raw_entry = {
                    "uuid": detection_uuid,
                    "timestamp": self._format_timestamp(now),
                    "type": label,
                    "color": cached_color,
                    "brand": cached_brand,
                    "x": raw_bbox["left"],
                    "y": raw_bbox["top"],
                    "width": raw_bbox["width"],
                    "height": raw_bbox["height"],
                    "camera_id": f"cam{camera_id + 1}",
                    "jetson_id": JETSON_ID,
                    "track_id": track_id,
                }
                raw_data_batch["data"].append(raw_entry)
                cam_entries.append({
                    "class_id": obj_class_id,
                    "bbox": bbox,
                    "raw_entry": raw_entry,
                    "object_result": object_result,
                    "stable_track_id": stable_track_id,
                })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            with self._cam_color_lock:
                self._cam_color_queue[camera_id] = {
                    "cycle": cycle,
                    "entries": cam_entries,
                }

            result["cameras"][cam_key] = cam_result

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        with self._det_lock:
            self._detections = result

        return Gst.PadProbeReturn.OK

    def _osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                confidence = float(obj_meta.confidence)
                self._style_rect_for_osd(
                    obj_meta.rect_params,
                    visible=not self._should_skip_object(confidence),
                )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    @staticmethod
    def _format_timestamp(timestamp_sec: float) -> str:
        return (
            datetime.fromtimestamp(timestamp_sec, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _scale_bbox_to_source_frame(frame_meta, bbox: Dict[str, float]) -> Dict[str, float]:
        source_width = int(getattr(frame_meta, "source_frame_width", 0) or MUX_WIDTH)
        source_height = int(getattr(frame_meta, "source_frame_height", 0) or MUX_HEIGHT)

        scale = min(MUX_WIDTH / max(1, source_width), MUX_HEIGHT / max(1, source_height))
        if scale <= 0:
            scale = 1.0

        scaled_width = source_width * scale
        scaled_height = source_height * scale
        pad_x = max(0.0, (MUX_WIDTH - scaled_width) * 0.5)
        pad_y = max(0.0, (MUX_HEIGHT - scaled_height) * 0.5)

        left = (bbox["left"] - pad_x) / scale
        top = (bbox["top"] - pad_y) / scale
        right = (bbox["left"] + bbox["width"] - pad_x) / scale
        bottom = (bbox["top"] + bbox["height"] - pad_y) / scale

        left = min(max(left, 0.0), float(source_width))
        top = min(max(top, 0.0), float(source_height))
        right = min(max(right, 0.0), float(source_width))
        bottom = min(max(bottom, 0.0), float(source_height))

        return {
            "left": left,
            "top": top,
            "width": max(0.0, right - left),
            "height": max(0.0, bottom - top),
        }

    def _should_skip_object(self, detector_confidence: float) -> bool:
        if detector_confidence < 0 and self._tracker_enabled:
            return False
        return detector_confidence < DETECTION_MIN_CONFIDENCE

    @staticmethod
    def _style_rect_for_osd(rect, visible: bool):
        try:
            rect.border_width = 2 if visible else 0
            if visible:
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)
        except Exception:
            pass

    @staticmethod
    def _output_confidence(detector_confidence: float, tracker_confidence: float) -> float:
        if detector_confidence >= 0:
            return detector_confidence
        if tracker_confidence >= 0:
            return tracker_confidence
        return 0.0

    @staticmethod
    def _format_track_id(object_id, allow_zero: bool = False) -> str:
        if object_id is None:
            return ""
        try:
            object_id_int = int(object_id)
        except (TypeError, ValueError, OverflowError):
            return ""
        if object_id_int < 0 or object_id_int >= 0xFFFFFFFFFFFFFFFE:
            return ""
        if object_id_int == 0 and not allow_zero:
            return ""
        return str(object_id_int)

    @staticmethod
    def _brand_cache_key(cam_idx: int, class_id: int, stable_track_id: str) -> Tuple[int, int, str]:
        return int(cam_idx), int(class_id), str(stable_track_id)

    def _get_cached_brand(self, key: Tuple[int, int, str]) -> Tuple[Optional[Dict[str, Any]], bool]:
        ttl = max(0.0, float(CLASSIFIER_CACHE_TTL_SEC))
        now = time.time()
        with self._brand_cache_lock:
            cached = self._brand_cache.get(key)
            if not cached:
                return None, False
            fresh = ttl <= 0.0 or (now - float(cached.get("updated_at", 0.0))) <= ttl
            self._classifier_cache_hits += 1
            return dict(cached), fresh

    def _set_cached_brand(
        self,
        key: Tuple[int, int, str],
        brand: str,
        confidence: Optional[float],
        color: Optional[str] = None,
    ) -> None:
        max_size = int(CLASSIFIER_CACHE_MAX_SIZE)
        if max_size <= 0:
            return

        with self._brand_cache_lock:
            previous = self._brand_cache.get(key, {})
            stored_brand = brand or previous.get("brand", "")
            stored_confidence = confidence if confidence is not None else previous.get("confidence")
            stored_color = color if color is not None else previous.get("color", "")
            self._brand_cache[key] = {
                "brand": stored_brand,
                "confidence": stored_confidence,
                "color": stored_color or "",
                "updated_at": time.time(),
            }
            while len(self._brand_cache) > max_size:
                oldest_key = min(
                    self._brand_cache,
                    key=lambda item: float(self._brand_cache[item].get("updated_at", 0.0)),
                )
                self._brand_cache.pop(oldest_key, None)

    def _classify_vehicle_brand(self, crop_bgr: np.ndarray) -> Tuple[str, Optional[float]]:
        if not self._classifier.enabled:
            return "", None
        try:
            brand, confidence = classify_crop(
                self._classifier,
                self._classifier_labels,
                crop_bgr,
                CLASSIFIER_INPUT_SIZE,
                CLASSIFIER_MIN_CONFIDENCE,
            )
            self._classifier_run_count += 1
            self._classifier_last_error = ""
            return brand, confidence
        except Exception as exc:
            self._classifier_last_error = repr(exc)
            print(f"[WARN] Vehicle brand classification failed: {self._classifier_last_error}")
            return "", None

    def _update_detection_attributes(
        self,
        raw_entry: Dict[str, Any],
        object_result: Optional[Dict[str, Any]],
        color: str,
        brand: str,
        brand_confidence: Optional[float],
    ) -> None:
        raw_entry["color"] = color or ""
        raw_entry["brand"] = brand or ""

        if object_result is None:
            return

        with self._det_lock:
            object_result["color"] = color or ""
            object_result["brand"] = brand or ""
            object_result["brand_confidence"] = brand_confidence

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
                print(f"[WARN] Cam{cam_idx} buffer too small: got={arr.size}, expected={expected}")
                return Gst.FlowReturn.OK

            frame = arr[:expected].reshape((height, width, 3)).copy()

            self._process_camera_colors(cam_idx, frame)

            fps = self._fps_meters[f"cam{cam_idx}"].tick()
            frame = self._overlay_fps(frame, fps, f"Cam{cam_idx + 1}")
            jpg = encode_jpeg(frame)
            if jpg:
                self.camera_frames[cam_idx].set_jpeg(jpg)

            self._try_publish_enriched_batch()

        except Exception as e:
            print(f"[ERROR] cam{cam_idx} appsink failed:", repr(e))
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _process_camera_colors(self, cam_idx: int, frame_bgr: np.ndarray):
        with self._cam_color_lock:
            cam_data = self._cam_color_queue.pop(cam_idx, None)

        if cam_data is None:
            return

        entries = cam_data.get("entries", [])
        classification_budget = max(0, int(CLASSIFIER_MAX_PER_CAMERA_FRAME))
        enriched = []
        for entry in entries:
            raw_entry = entry["raw_entry"]
            obj_class_id = entry["class_id"]
            object_result = entry.get("object_result")
            color = raw_entry.get("color", "")
            brand = raw_entry.get("brand", "")
            brand_confidence = object_result.get("brand_confidence") if object_result else None
            crop = None

            if obj_class_id in VEHICLE_CLASS_IDS:
                bbox = entry["bbox"]
                x1 = max(0, int(bbox["left"]))
                y1 = max(0, int(bbox["top"]))
                x2 = min(frame_bgr.shape[1], x1 + int(bbox["width"]))
                y2 = min(frame_bgr.shape[0], y1 + int(bbox["height"]))

                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]
                    color = vehicle_color_rule.detect_color(crop)
                else:
                    color = "Unknown"

                stable_track_id = entry.get("stable_track_id") or ""
                cache_key = None
                if stable_track_id:
                    cache_key = self._brand_cache_key(cam_idx, obj_class_id, stable_track_id)

                if crop is not None and obj_class_id in self._classify_class_ids and cache_key:
                    cached, fresh = self._get_cached_brand(cache_key)
                    if cached and fresh:
                        brand = cached.get("brand", "")
                        brand_confidence = cached.get("confidence")
                    elif classification_budget > 0:
                        brand, brand_confidence = self._classify_vehicle_brand(crop)
                        classification_budget -= 1
                    elif cached:
                        brand = cached.get("brand", "")
                        brand_confidence = cached.get("confidence")

                if cache_key:
                    self._set_cached_brand(cache_key, brand, brand_confidence, color)

            self._update_detection_attributes(raw_entry, object_result, color, brand, brand_confidence)

            enriched.append(raw_entry)

        with self._cam_color_lock:
            self._cam_color_enriched[cam_idx] = (cam_data["cycle"], enriched, time.time())

    def _try_publish_enriched_batch(self):
        n_cameras = len(RTSP_URLS)
        now = time.time()

        with self._cam_color_lock:
            if len(self._cam_color_enriched) < n_cameras:
                if self._cam_color_enriched:
                    oldest_ts = min(ts for _, _, ts in self._cam_color_enriched.values())
                    if now - oldest_ts < 0.5:
                        return
                else:
                    return

            all_entries = []
            for cam_idx in sorted(self._cam_color_enriched):
                _, entries, _ = self._cam_color_enriched[cam_idx]
                all_entries.extend(entries)

            self._cam_color_enriched.clear()

        if all_entries:
            self._publisher.publish_latest({"data": all_entries})

    def get_detections(self):
        with self._det_lock:
            return self._detections

    def get_detection_publisher_status(self):
        return self._publisher.status()

    def get_classifier_status(self):
        status = self._classifier.status()
        with self._brand_cache_lock:
            cache_size = len(self._brand_cache)
            cache_hits = self._classifier_cache_hits
        status.update({
            "labels_count": len(self._classifier_labels),
            "operate_on_class_ids": sorted(self._classify_class_ids),
            "min_confidence": CLASSIFIER_MIN_CONFIDENCE,
            "cache_ttl_sec": CLASSIFIER_CACHE_TTL_SEC,
            "cache_max_size": CLASSIFIER_CACHE_MAX_SIZE,
            "cache_size": cache_size,
            "cache_hits": cache_hits,
            "max_per_camera_frame": CLASSIFIER_MAX_PER_CAMERA_FRAME,
            "run_count": self._classifier_run_count,
            "last_error": self._classifier_last_error,
        })
        return status

    def start(self):
        self._publisher.start()
        super().start()

    def stop(self):
        self._publisher.stop()
        super().stop()
