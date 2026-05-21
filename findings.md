# Findings

## Codebase Analysis

### config.py
- Contains RTSP URLs, pipeline dimensions, inference settings, tracker settings, detection server URL
- No vehicle class filter yet → need VEHICLE_CLASS_IDS = {2, 3, 5, 7}

### vehicle_color_rule.py
- Provides `detect_color(vehicle_bgr_img: np.ndarray) -> str`
- Handles None/empty with "Unknown"
- Uses HSV color space voting
- No changes needed

### pipelines/detect_pipeline.py
- DetectPipeline extends BaseDeepStreamPipeline
- Overrides build() with nvinfer + nvtracker + tee → mosaic/per-camera branches
- _tracked_src_pad_buffer_probe: parses batch metadata, builds raw_data_batch + result, publishes via _publisher
- Does NOT override _on_camera_sample (inherits from common.py)
- Base class sets up per-camera appsinks with _on_camera_sample callback
- Has _det_lock for detection data, _publisher for JSON POST

### pipelines/common.py
- BaseDeepStreamPipeline defines pipeline structure, appsink callbacks
- _on_camera_sample extracts BGR frame, encodes JPEG, stores in camera_frames
- encode_jpeg() utility function
- _make_convert_chain creates convert elements at specified width/height

## Data Flow (Proposed)
1. Probe (on post-tracker buffer): cycle++, store entries per camera in _cam_color_queue
2. Per-camera appsink (_on_camera_sample): extract BGR frame, encode JPEG, _process_camera_colors
3. _process_camera_colors: look up cached entries, crop vehicle bboxes, detect_color(), store enriched
4. _try_publish_enriched_batch: when all cameras ready (or timeout), combine and publish

## Coordinate Matching
- Probe bbox: MUX_WIDTH x MUX_HEIGHT (e.g., 640x640)
- Per-camera appsink frame: MUX_WIDTH x MUX_HEIGHT (via _make_convert_chain)
- → No coordinate transform needed for cropping

## Source ID Mapping
- frame_meta.source_id from streammux = source index (0-4)
- cam_idx in appsink callback = demux src index (0-4)
- → source_id == cam_idx
