import time
import threading
import json
import queue
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

from config import (
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
)
from pipelines.common import BaseDeepStreamPipeline, make_element, set_optional_property
from pipelines.model_config import abs_project_path, ensure_primary_infer_config


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

    def build(self):
        """
        Override build เพื่อแทรก nvinfer ก่อน nvstreamdemux
        และ nvdsosd เฉพาะใน mosaic path หลัง remux
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

        # ── nvstreamdemux -> per-camera branches + remux ─────────────
        _, remux = self._build_demux_branches(pre_demux)

        # ── mosaic path: remux -> tiler -> osd -> convert -> appsink ──
        tiler = make_element("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows", TILER_ROWS)
        tiler.set_property("columns", TILER_COLUMNS)
        tiler.set_property("width", TILER_WIDTH)
        tiler.set_property("height", TILER_HEIGHT)

        nvvidconv_preosd = make_element("nvvideoconvert", "pre-osd-converter")
        caps_rgba_preosd = make_element("capsfilter", "caps-rgba-preosd")
        caps_rgba_preosd.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"),
        )

        nvosd = make_element("nvdsosd", "onscreendisplay")
        try:
            nvosd.set_property("process-mode", 0)
        except Exception:
            pass

        mosaic_conv, _, _ = self._make_convert_chain(
            suffix="mosaic", width=TILER_WIDTH, height=TILER_HEIGHT
        )
        mosaic_sink = self._make_appsink("appsink-mosaic", self._on_new_sample)

        for el in [tiler, nvvidconv_preosd, caps_rgba_preosd, nvosd] + mosaic_conv + [mosaic_sink]:
            self.pipeline.add(el)

        mosaic_chain = [remux, tiler, nvvidconv_preosd, caps_rgba_preosd, nvosd] + mosaic_conv + [mosaic_sink]
        for a, b in zip(mosaic_chain[:-1], mosaic_chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

        print("[INFO] Built mosaic branch (detect mode with OSD)")
        print(f"[INFO] nvinfer config: {infer_config_path}")
        return self.pipeline

    def _tracked_src_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

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

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                rect = obj_meta.rect_params
                confidence = float(obj_meta.confidence)
                tracker_confidence = float(getattr(obj_meta, "tracker_confidence", -1.0))
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
                    label = str(obj_meta.class_id)

                bbox = {
                    "left": float(rect.left),
                    "top": float(rect.top),
                    "width": float(rect.width),
                    "height": float(rect.height),
                }
                raw_bbox = self._scale_bbox_to_source_frame(frame_meta, bbox)
                detection_uuid = str(uuid.uuid4())
                track_id = (
                    self._format_track_id(
                        getattr(obj_meta, "object_id", None),
                        allow_zero=self._tracker_enabled,
                    )
                    or detection_uuid
                )

                cam_result["objects"].append({
                    "class_id": int(obj_meta.class_id),
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
                })
                raw_data_batch["data"].append({
                    "uuid": detection_uuid,
                    "timestamp": self._format_timestamp(now),
                    "type": label,
                    "color": "",
                    "brand": "",
                    "x": raw_bbox["left"],
                    "y": raw_bbox["top"],
                    "width": raw_bbox["width"],
                    "height": raw_bbox["height"],
                    "camera_id": f"cam{camera_id + 1}",
                    "jetson_id": JETSON_ID,
                    "track_id": track_id,
                })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            result["cameras"][cam_key] = cam_result

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        with self._det_lock:
            self._detections = result

        if raw_data_batch["data"]:
            self._publisher.publish_latest(raw_data_batch)
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

    def get_detections(self):
        with self._det_lock:
            return self._detections

    def get_detection_publisher_status(self):
        return self._publisher.status()

    def start(self):
        self._publisher.start()
        super().start()

    def stop(self):
        self._publisher.stop()
        super().stop()
