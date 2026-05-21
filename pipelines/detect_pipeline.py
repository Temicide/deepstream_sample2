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
)
from pipelines.common import BaseDeepStreamPipeline, make_element
from pipelines.model_config import ensure_primary_infer_config


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
    RTSP x5 -> decode -> nvstreammux -> nvinfer -> tiler -> nvdsosd -> appsink -> FastAPI MJPEG

    เพิ่ม /detections โดยอ่าน metadata จาก nvinfer ผ่าน pad probe
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

    def create_custom_elements(self):
        pgie = make_element("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", ensure_primary_infer_config())

        # วาง probe ที่ src pad ของ pgie เพื่ออ่าน metadata ก่อนเข้า tiler/osd
        srcpad = pgie.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Unable to get pgie src pad")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_src_pad_buffer_probe, None)

        # nvdsosd ต้องอยู่หลัง tiler ใน common chain ไม่ได้แทรกเอง
        # ดังนั้นเราจะ return pgie ก่อน แล้ว override build chain ด้วยการเพิ่ม osd หลัง tiler ไม่ได้ใน base เดิม
        # วิธีง่ายสุด: ใช้ pgie เป็น custom element และเพิ่ม osd โดย override create_post_tiler_elements
        self._needs_osd = True
        return [pgie]

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
        self.pipeline.add(streammux)
        self._build_sources(streammux)

        # ── nvinfer ──────────────────────────────────────────────────
        infer_config_path = ensure_primary_infer_config()
        pgie = make_element("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", infer_config_path)
        srcpad = pgie.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Unable to get pgie src pad")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_src_pad_buffer_probe, None)
        self.pipeline.add(pgie)
        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> pgie")

        # ── nvstreamdemux -> per-camera branches + remux ─────────────
        _, remux = self._build_demux_branches(pgie)

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

    def _pgie_src_pad_buffer_probe(self, pad, info, u_data):
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
                if confidence < DETECTION_MIN_CONFIDENCE:
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                    continue

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
                    self._format_track_id(getattr(obj_meta, "object_id", None))
                    or detection_uuid
                )

                cam_result["objects"].append({
                    "class_id": int(obj_meta.class_id),
                    "label": label,
                    "confidence": confidence,
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
        scale_x = source_width / max(1, MUX_WIDTH)
        scale_y = source_height / max(1, MUX_HEIGHT)
        return {
            "left": bbox["left"] * scale_x,
            "top": bbox["top"] * scale_y,
            "width": bbox["width"] * scale_x,
            "height": bbox["height"] * scale_y,
        }

    @staticmethod
    def _format_track_id(object_id) -> str:
        if object_id is None:
            return ""
        try:
            object_id_int = int(object_id)
        except (TypeError, ValueError, OverflowError):
            return ""
        if object_id_int <= 0 or object_id_int >= 0xFFFFFFFFFFFFFFFE:
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
