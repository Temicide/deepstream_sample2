import time
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject, GLib

from config import (
    RTSP_URLS,
    MUX_WIDTH,
    MUX_HEIGHT,
    MUX_BATCH_SIZE,
    MUX_TIMEOUT_USEC,
    TILER_ROWS,
    TILER_COLUMNS,
    TILER_WIDTH,
    TILER_HEIGHT,
    JPEG_QUALITY,
)


Gst.init(None)


class SharedFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._last_update = 0.0

    def set_jpeg(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg
            self._last_update = time.time()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def age_sec(self) -> Optional[float]:
        with self._lock:
            if self._last_update == 0:
                return None
            return time.time() - self._last_update


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


def make_element(factory_name: str, name: str):
    element = Gst.ElementFactory.make(factory_name, name)
    if not element:
        raise RuntimeError(f"Could not create GStreamer element: {factory_name} name={name}")
    return element

def set_optional_property(element, property_name: str, value) -> bool:
    if not element.find_property(property_name):
        return False
    element.set_property(property_name, value)
    return True

def request_mux_sink_pad(mux, index: int):
    sinkpad = mux.get_request_pad(f"sink_{index}")
    if sinkpad:
        return sinkpad

    sinkpad = mux.get_request_pad("sink_%u")
    if sinkpad:
        return sinkpad

    return None

def encode_jpeg(frame_bgr: np.ndarray) -> Optional[bytes]:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
    ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not ok:
        return None
    return encoded.tobytes()


class BaseDeepStreamPipeline:
    """
    Common DeepStream pipeline (Option B — nvstreamdemux per-camera appsink):

    mosaic path:
      nvstreammux -> [custom] -> tiler -> nvvidconv -> appsink (shared_frame)

    per-camera path (parallel, no CPU crop):
      nvstreammux -> [custom] -> nvstreamdemux -> src_0 -> nvvidconv_0 -> appsink_0
                                               -> src_1 -> nvvidconv_1 -> appsink_1
                                               -> src_N -> ...

    แต่ละ appsink_i รัน callback บน GStreamer thread ของตัวเอง
    ไม่มี crop บน CPU อีกต่อไป
    """

    def __init__(self, mode_name: str):
        self.mode_name = mode_name
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self.shared_frame = SharedFrame()
        self._fps_meters = {"mosaic": FpsMeter()}

        self.appsink_width = TILER_WIDTH
        self.appsink_height = TILER_HEIGHT

        self._streammux = None
        self._source_bins: Dict[int, Gst.Bin] = {}
        self._mux_sinkpads: Dict[int, Gst.Pad] = {}
        self._reconnect_timers: Dict[int, threading.Timer] = {}
        self._reconnect_delay = 5.0

        self.camera_frames: List[SharedFrame] = [SharedFrame() for _ in RTSP_URLS]
        for idx in range(len(RTSP_URLS)):
            self._fps_meters[f"cam{idx}"] = FpsMeter()

    def get_jpeg(self) -> Optional[bytes]:
        return self.shared_frame.get_jpeg()

    def get_camera_jpeg(self, cam_idx: int) -> Optional[bytes]:
        if 0 <= cam_idx < len(self.camera_frames):
            return self.camera_frames[cam_idx].get_jpeg()
        return None

    def _crop_camera_frame(self, mosaic_bgr: np.ndarray, cam_idx: int) -> np.ndarray:
        cell_w = TILER_WIDTH // TILER_COLUMNS
        cell_h = TILER_HEIGHT // TILER_ROWS
        col = cam_idx % TILER_COLUMNS
        row = cam_idx // TILER_COLUMNS
        x = col * cell_w
        y = row * cell_h
        return mosaic_bgr[y:y + cell_h, x:x + cell_w].copy()

    def _build_sources(self, streammux):
        """Create all source bins and link them to the muxer. Stores refs for reconnect."""
        self._streammux = streammux
        for i, uri in enumerate(RTSP_URLS):
            source_bin = self._create_source_bin(i, uri)
            self.pipeline.add(source_bin)
            self._source_bins[i] = source_bin

            #sinkpad = streammux.get_request_pad("sink_%u")
            sinkpad = request_mux_sink_pad(streammux, i)
            if not sinkpad:
                raise RuntimeError(f"Unable to get streammux sink pad for source {i}")
            self._mux_sinkpads[i] = sinkpad

            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(f"Unable to get source bin src pad for source {i}")

            ret = srcpad.link(sinkpad)
            if ret != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source {i} to streammux: {ret}")

    def _make_convert_chain(self, suffix: str, width: int, height: int):
        """
        สร้าง element chain: nvvideoconvert -> capsfilter(RGBA) -> videoconvert -> capsfilter(BGR)
        ใช้ซ้ำได้ทั้ง mosaic path และ per-camera path
        คืนค่า (list of elements, head element, tail element)
        """
        nvvidconv  = make_element("nvvideoconvert", f"nvvconv-{suffix}")
        caps_rgba  = make_element("capsfilter",     f"caps-rgba-{suffix}")
        caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))

        videoconv  = make_element("videoconvert",   f"vconv-{suffix}")
        caps_bgr   = make_element("capsfilter",     f"caps-bgr-{suffix}")
        caps_bgr.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw, format=BGR, width={width}, height={height}"),
        )
        elements = [nvvidconv, caps_rgba, videoconv, caps_bgr]
        return elements, nvvidconv, caps_bgr

    def _make_appsink(self, name: str, callback):
        """สร้าง appsink พร้อม callback"""
        sink = make_element("appsink", name)
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        set_optional_property(sink, "async", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.connect("new-sample", callback)
        return sink

    def _overlay_fps(self, frame_bgr: np.ndarray, fps: float, label: str) -> np.ndarray:
        if fps <= 0:
            fps_text = f"{label} FPS: --"
        else:
            fps_text = f"{label} FPS: {fps:.1f}"

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
        cv2.putText(
            frame_bgr,
            fps_text,
            (x, y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        return frame_bgr

    def _build_demux_branches(self, upstream_element):
        """
        ต่อ nvstreamdemux หลัง upstream_element แล้วสร้างสองส่วน:

        1) per-camera branch (parallel):
             nvstreamdemux -> src_i -> tee_i -> q_cam_i -> convert -> appsink_cam_i

        2) mosaic branch (re-mux กลับก่อนเข้า tiler):
             nvstreamdemux -> src_i -> tee_i -> q_remux_i -> remux -> tiler -> ...

        tee ต่อหลัง demux src pad แต่ละตัว เพราะ nvstreamdemux ให้ request
        pad เดิมซ้ำไม่ได้ (1 src pad = 1 downstream เท่านั้น)
        """
        n = len(RTSP_URLS)

        demux = make_element("nvstreamdemux", "nvstream-demux")
        self.pipeline.add(demux)
        if not upstream_element.link(demux):
            raise RuntimeError(
                f"Failed to link {upstream_element.get_name()} -> nvstreamdemux"
            )

        # ── nvstreammux ตัวที่ 2 สำหรับ mosaic path ──────────────────
        remux = make_element("nvstreammux", "stream-remuxer")
        remux.set_property("width", MUX_WIDTH)
        remux.set_property("height", MUX_HEIGHT)
        remux.set_property("batch-size", n)
        remux.set_property("batched-push-timeout", MUX_TIMEOUT_USEC)
        remux.set_property("live-source", 1)
        set_optional_property(remux, "enable-padding", True)
        self.pipeline.add(remux)

        for i in range(n):
            # ── demux src pad (1 pad ต่อ 1 stream) ───────────────────
            demux_src = demux.get_request_pad(f"src_{i}")
            if not demux_src:
                raise RuntimeError(f"nvstreamdemux: cannot get src_{i}")

            # ── tee: แยก 1 stream ออกสองทาง ──────────────────────────
            tee_i = make_element("tee", f"tee-cam{i}")
            self.pipeline.add(tee_i)

            tee_sink = tee_i.get_static_pad("sink")
            if demux_src.link(tee_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux src_{i} -> tee-cam{i}")

            # ── branch A: per-camera appsink ──────────────────────────
            q_cam = make_element("queue", f"q-cam{i}")
            q_cam.set_property("max-size-buffers", 2)
            q_cam.set_property("leaky", 2)

            conv_els, conv_head, _ = self._make_convert_chain(
                suffix=f"cam{i}", width=MUX_WIDTH, height=MUX_HEIGHT
            )

            def _cam_cb(sink, idx=i):
                return self._on_camera_sample(sink, idx)

            cam_sink = self._make_appsink(f"appsink-cam{i}", _cam_cb)

            for el in [q_cam] + conv_els + [cam_sink]:
                self.pipeline.add(el)

            cam_chain = [q_cam] + conv_els + [cam_sink]
            for a, b in zip(cam_chain[:-1], cam_chain[1:]):
                if not a.link(b):
                    raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

            tee_src_cam = tee_i.get_request_pad("src_%u")
            if tee_src_cam.link(q_cam.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link tee-cam{i} -> q-cam{i}")

            # ── branch B: remux สำหรับ mosaic ─────────────────────────
            q_remux = make_element("queue", f"q-remux{i}")
            q_remux.set_property("max-size-buffers", 2)
            q_remux.set_property("leaky", 2)
            self.pipeline.add(q_remux)

            #remux_sink = remux.get_request_pad("sink_%u")
            remux_sink = request_mux_sink_pad(remux, i)

            if not remux_sink:
                raise RuntimeError(f"stream-remuxer: cannot get sink pad for source {i}")

            tee_src_remux = tee_i.get_request_pad("src_%u")
            if tee_src_remux.link(q_remux.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link tee-cam{i} -> q-remux{i}")

            q_remux_src = q_remux.get_static_pad("src")
            if q_remux_src.link(remux_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link q-remux{i} -> stream-remuxer sink pad")

            print(f"[INFO] Built per-camera branch for cam {i}")

        return demux, remux

    def _build_mosaic_sink(self, remux):
        """
        ต่อ tiler -> convert -> appsink_mosaic หลัง remux
        """
        tiler = make_element("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows", TILER_ROWS)
        tiler.set_property("columns", TILER_COLUMNS)
        tiler.set_property("width", TILER_WIDTH)
        tiler.set_property("height", TILER_HEIGHT)

        mosaic_conv, _, _ = self._make_convert_chain(
            suffix="mosaic", width=TILER_WIDTH, height=TILER_HEIGHT
        )
        mosaic_sink = self._make_appsink("appsink-mosaic", self._on_new_sample)

        for el in [tiler] + mosaic_conv + [mosaic_sink]:
            self.pipeline.add(el)

        mosaic_chain = [remux, tiler] + mosaic_conv + [mosaic_sink]
        for a, b in zip(mosaic_chain[:-1], mosaic_chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")

        print("[INFO] Built mosaic branch")

    def build(self):
        self.pipeline = Gst.Pipeline.new(f"deepstream-{self.mode_name}-pipeline")
        if not self.pipeline:
            raise RuntimeError("Could not create pipeline")

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

        # ── custom elements (nvinfer ฯลฯ จาก subclass) ───────────────
        custom_elements = self.create_custom_elements()
        for el in custom_elements:
            self.pipeline.add(el)

        pre_demux = streammux
        for el in custom_elements:
            if not pre_demux.link(el):
                raise RuntimeError(f"Failed to link {pre_demux.get_name()} -> {el.get_name()}")
            pre_demux = el

        # ── nvstreamdemux -> per-camera branches + remux ─────────────
        _, remux = self._build_demux_branches(pre_demux)

        # ── mosaic path: remux -> tiler -> convert -> appsink ─────────
        self._build_mosaic_sink(remux)

        return self.pipeline

    def create_custom_elements(self):
        """
        Override in subclasses.
        Return list of elements to insert between streammux and tiler.
        """
        return []

    def process_frame_before_jpeg(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Override in subclasses.
        Phase 1 returns original.
        Phase 2 converts grayscale.
        Phase 3 returns frame with OSD already drawn by DeepStream.
        """
        return frame_bgr

    def _find_source_index(self, element) -> Optional[int]:
        """Walk up the GStreamer parent chain to find which source bin an element belongs to."""
        obj = element
        while obj is not None:
            for idx, sbin in self._source_bins.items():
                if obj == sbin:
                    return idx
            parent = obj.get_parent() if hasattr(obj, "get_parent") else None
            if parent is None or parent == obj:
                break
            obj = parent
        return None

    def _schedule_reconnect(self, index: int):
        if index in self._reconnect_timers:
            self._reconnect_timers[index].cancel()
        t = threading.Timer(
            self._reconnect_delay,
            lambda: GLib.idle_add(self._do_reconnect, index),
        )
        t.daemon = True
        self._reconnect_timers[index] = t
        t.start()
        print(f"[INFO] Will reconnect source {index} in {self._reconnect_delay}s")

    def _do_reconnect(self, index: int) -> bool:
        """Runs on GLib main loop thread — safe for GStreamer operations."""
        uri = RTSP_URLS[index]
        print(f"[INFO] Reconnecting source {index}: {uri}")

        old_bin = self._source_bins.pop(index, None)
        sinkpad = self._mux_sinkpads.get(index)

        if old_bin:
            old_srcpad = old_bin.get_static_pad("src")
            if old_srcpad and sinkpad and old_srcpad.is_linked():
                old_srcpad.unlink(sinkpad)
            old_bin.set_state(Gst.State.NULL)
            self.pipeline.remove(old_bin)

        new_bin = self._create_source_bin(index, uri)
        self.pipeline.add(new_bin)
        self._source_bins[index] = new_bin

        new_srcpad = new_bin.get_static_pad("src")
        if new_srcpad and sinkpad:
            ret = new_srcpad.link(sinkpad)
            if ret == Gst.PadLinkReturn.OK:
                new_bin.sync_state_with_parent()
                print(f"[INFO] Source {index} reconnected OK")
            else:
                print(f"[WARN] Source {index} relink failed ({ret}), will retry")
                self._source_bins.pop(index, None)
                self.pipeline.remove(new_bin)
                self._schedule_reconnect(index)
        else:
            new_bin.sync_state_with_parent()

        return False  # don't repeat GLib idle

    def _create_source_bin(self, index: int, uri: str):
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            raise RuntimeError(f"Unable to create source bin {bin_name}")

        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            print("[WARN] nvurisrcbin unavailable, falling back to uridecodebin")
            uri_decode_bin = make_element("uridecodebin", f"uri-decode-bin-{index}")
        uri_decode_bin.set_property("uri", uri)
        try:
            uri_decode_bin.set_property("drop-on-latency", True)
        except Exception:
            pass
        uri_decode_bin.connect("pad-added", self._decodebin_pad_added, nbin)
        try:
            uri_decode_bin.connect("child-added", self._decodebin_child_added, nbin)
        except TypeError:
            pass

        nbin.add(uri_decode_bin)

        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if not ghost_pad:
            raise RuntimeError("Failed to create ghost pad")
        nbin.add_pad(ghost_pad)

        return nbin

    def _decodebin_child_added(self, child_proxy, obj, name, user_data):
        # Optional: tune rtspsrc latency when uridecodebin creates it.
        # This helps stability over low quality network.
        if "source" in name.lower() or "rtspsrc" in name.lower():
            try:
                obj.set_property("latency", 200)
                obj.set_property("drop-on-latency", True)
            except Exception:
                pass

    def _decodebin_pad_added(self, decodebin, pad, source_bin):
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.query_caps(None)

        structure_name = caps.get_structure(0).get_name()
        if not structure_name.startswith("video"):
            return

        features = caps.get_features(0)
        # DeepStream wants NVMM memory from NVIDIA decoder.
        if not features or not features.contains("memory:NVMM"):
            print("[WARN] Decodebin did not pick NVIDIA decoder / NVMM memory.")
            print("[WARN] caps:", caps.to_string())
            return

        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad.set_target(pad):
            print(f"[INFO] Linked decodebin pad for {source_bin.get_name()}")
        else:
            print(f"[ERROR] Failed to link decodebin pad for {source_bin.get_name()}")

    def _on_new_sample(self, sink):
        """Mosaic appsink callback — อัป shared_frame สำหรับ /video/<mode>"""
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
                print(f"[WARN] Mosaic buffer too small: got={arr.size}, expected={expected}")
                return Gst.FlowReturn.OK

            frame = arr[:expected].reshape((height, width, 3)).copy()
            frame = self.process_frame_before_jpeg(frame)
            fps = self._fps_meters["mosaic"].tick()
            frame = self._overlay_fps(frame, fps, "Mosaic")

            jpg = encode_jpeg(frame)
            if jpg:
                self.shared_frame.set_jpeg(jpg)

        except Exception as e:
            print("[ERROR] mosaic appsink failed:", repr(e))
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_camera_sample(self, sink, cam_idx: int):
        """Per-camera appsink callback — อัป camera_frames[cam_idx] โดยตรง ไม่มี crop"""
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
            fps = self._fps_meters[f"cam{cam_idx}"].tick()
            frame = self._overlay_fps(frame, fps, f"Cam{cam_idx + 1}")
            jpg = encode_jpeg(frame)
            if jpg:
                self.camera_frames[cam_idx].set_jpeg(jpg)

        except Exception as e:
            print(f"[ERROR] cam{cam_idx} appsink failed:", repr(e))
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def start(self):
        self.build()
        self.loop = GLib.MainLoop()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self.loop)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set pipeline to PLAYING")

        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()
        print(f"[INFO] Started DeepStream pipeline mode={self.mode_name}")

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        print("[INFO] Pipeline stopped")

    def _bus_call(self, bus, message, loop):
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            print("[INFO] End-of-stream")
            loop.quit()

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_idx = self._find_source_index(message.src)
            if src_idx is not None:
                # One RTSP source failed — reconnect it without stopping others
                print(f"[WARN] RTSP source {src_idx} error: {err} — reconnecting")
                self._schedule_reconnect(src_idx)
            else:
                # Fatal pipeline error unrelated to a single source
                print("[ERROR] GStreamer error:", err)
                print("[ERROR] Debug:", debug)
                loop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print("[WARN] GStreamer warning:", err)
            print("[WARN] Debug:", debug)

        return True
