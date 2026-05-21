import argparse
import time

import uvicorn
from fastapi import FastAPI, Path
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse

from config import OUTPUT_FPS, RTSP_URLS
from pipelines.live_pipeline import LivePipeline
from pipelines.gray_pipeline import GrayPipeline
from pipelines.detect_pipeline import DetectPipeline


app = FastAPI(title="Jetson Nano DeepStream RTSP API")

STATE = {
    "mode": None,
    "pipeline": None,
    "started_at": time.time(),
}


@app.get("/")
def index():
    return RedirectResponse(url="/monitor")


def mjpeg_generator(pipeline):
    delay = 1.0 / max(1, OUTPUT_FPS)
    while True:
        frame = pipeline.get_jpeg()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(delay)


def mjpeg_generator_cam(pipeline, cam_idx: int):
    delay = 1.0 / max(1, OUTPUT_FPS)
    while True:
        frame = pipeline.get_camera_jpeg(cam_idx)
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(delay)


@app.get("/health")
def health():
    pipeline = STATE["pipeline"]
    payload = {
        "ok": True,
        "mode": STATE["mode"],
        "uptime_sec": round(time.time() - STATE["started_at"], 2),
        "source_count": len(RTSP_URLS),
        "has_frame": pipeline.get_jpeg() is not None if pipeline else False,
    }
    if pipeline and STATE["mode"] == "detect":
        payload["publisher"] = pipeline.get_detection_publisher_status()
        payload["classifier"] = pipeline.get_classifier_status()
    return payload


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    mode = STATE["mode"] or "live"
    video_path = f"/video/{mode}"
    show_detections = mode == "detect"
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepStream Monitor</title>
  <style>
    body {{
      margin: 0;
      background: #111827;
      color: #e5e7eb;
      font-family: Arial, sans-serif;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #0f172a;
      border-bottom: 1px solid #334155;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 16px;
      padding: 16px;
    }}
    img {{
      width: 100%;
      background: #020617;
      border: 1px solid #334155;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      min-height: 360px;
      overflow: auto;
      background: #020617;
      border: 1px solid #334155;
      color: #cbd5e1;
      font-size: 12px;
    }}
    .pill {{
      padding: 4px 8px;
      border-radius: 999px;
      background: #14532d;
      color: #dcfce7;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <strong>DeepStream Monitor</strong>
    <span class="pill">mode: {mode}</span>
  </header>
  <main>
    <section>
      <img src="{video_path}" alt="DeepStream video stream">
    </section>
    <section>
      <pre id="detections">{"Waiting for detections..." if show_detections else "Start server with --mode detect to see detection JSON."}</pre>
    </section>
  </main>
  <script>
    const showDetections = {str(show_detections).lower()};
    async function refreshDetections() {{
      if (!showDetections) return;
      try {{
        const res = await fetch("/detections", {{ cache: "no-store" }});
        const data = await res.json();
        document.getElementById("detections").textContent = JSON.stringify(data, null, 2);
      }} catch (err) {{
        document.getElementById("detections").textContent = String(err);
      }}
    }}
    refreshDetections();
    setInterval(refreshDetections, 500);
  </script>
</body>
</html>
"""


@app.get("/video/live")
def video_live():
    if STATE["mode"] != "live":
        return JSONResponse(
            {"error": "server is not running in live mode"},
            status_code=400,
        )
    return StreamingResponse(
        mjpeg_generator(STATE["pipeline"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/gray")
def video_gray():
    if STATE["mode"] != "gray":
        return JSONResponse(
            {"error": "server is not running in gray mode"},
            status_code=400,
        )
    return StreamingResponse(
        mjpeg_generator(STATE["pipeline"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/detect")
def video_detect():
    if STATE["mode"] != "detect":
        return JSONResponse(
            {"error": "server is not running in detect mode"},
            status_code=400,
        )
    return StreamingResponse(
        mjpeg_generator(STATE["pipeline"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/cam/{cam_id}/live")
def video_cam_live(cam_id: int = Path(ge=1, le=len(RTSP_URLS))):
    if STATE["mode"] != "live":
        return JSONResponse({"error": "server is not running in live mode"}, status_code=400)
    return StreamingResponse(
        mjpeg_generator_cam(STATE["pipeline"], cam_id - 1),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/cam/{cam_id}/gray")
def video_cam_gray(cam_id: int = Path(ge=1, le=len(RTSP_URLS))):
    if STATE["mode"] != "gray":
        return JSONResponse({"error": "server is not running in gray mode"}, status_code=400)
    return StreamingResponse(
        mjpeg_generator_cam(STATE["pipeline"], cam_id - 1),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/cam/{cam_id}/detect")
def video_cam_detect(cam_id: int = Path(ge=1, le=len(RTSP_URLS))):
    if STATE["mode"] != "detect":
        return JSONResponse({"error": "server is not running in detect mode"}, status_code=400)
    return StreamingResponse(
        mjpeg_generator_cam(STATE["pipeline"], cam_id - 1),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections")
def detections():
    pipeline = STATE["pipeline"]
    if STATE["mode"] != "detect":
        return JSONResponse(
            {"error": "detections are available only in detect mode"},
            status_code=400,
        )
    return pipeline.get_detections()


@app.get("/detections/cam/{cam_id}")
def detections_cam(cam_id: int = Path(ge=1, le=len(RTSP_URLS))):
    pipeline = STATE["pipeline"]
    if STATE["mode"] != "detect":
        return JSONResponse(
            {"error": "detections are available only in detect mode"},
            status_code=400,
        )
    detections_payload = pipeline.get_detections()
    return detections_payload.get("cameras", {}).get(str(cam_id - 1), {
        "camera_id": cam_id,
        "source_id": cam_id - 1,
        "objects": [],
    })


def create_pipeline(mode: str):
    if mode == "live":
        return LivePipeline()
    if mode == "gray":
        return GrayPipeline()
    if mode == "detect":
        return DetectPipeline()
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["live", "gray", "detect"],
        required=True,
        help="select one pipeline mode",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    pipeline = create_pipeline(args.mode)
    STATE["mode"] = args.mode
    STATE["pipeline"] = pipeline
    STATE["started_at"] = time.time()

    pipeline.start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
