# DeepStream Semifinal

RTSP sources go through DeepStream for decode + detection, then each detected vehicle crop is classified in Python with the same preprocessing style used by `firm-pipeline`.

## Files

```text
server.py
pipeline.py
config.py
requirements.txt
models/
  labels.txt
  labels.json
```

## Model swap

To change models later, replace the files referenced in `config.py`:

- `DETECTOR_MODEL_PATH` for the primary YOLO detector
- `CLASSIFIER_MODEL_PATH` for the secondary brand classifier

Both paths can point to either `.onnx` or `.engine` files.

If the detector points to ONNX, DeepStream will build the TensorRT engine automatically on first run.
If the classifier points to ONNX, the app will run it directly with `onnxruntime` or OpenCV DNN.

## Install

Make sure the DeepStream Python environment is active, then install the Python packages:

```bash
pip install -r requirements.txt
```

For Jetson Nano with Python 3.6.9, use a 3.6-compatible `numpy` wheel if your environment still has a mismatched build. The pinned versions in `requirements.txt` are chosen to avoid the `numpy.core._multiarray_umath` import error.

If `onnxruntime` is not available for your Jetson Python build, you can skip it. The classifier will fall back to OpenCV DNN when needed.

If you use a virtual environment, also install `pyservicemaker` inside it when needed by other DeepStream tasks.

## Run

```bash
python3 server.py --host 0.0.0.0 --port 8000
```

Open:

```text
http://JETSON_IP:8000/monitor
http://JETSON_IP:8000/video/mosaic
http://JETSON_IP:8000/video/cam/1
http://JETSON_IP:8000/logs/detect
http://JETSON_IP:8000/api/logs/detect
```

The `/logs/detect` page shows the latest detection records in a browser. The `/api/logs/detect` endpoint returns the same log buffer as JSON for debugging.

Detected objects are batched and posted to `http://10.0.11.153:8080/api/v1/raw_data/batch` by default. Override that target with `DEEPSTREAM_DETECTION_LOG_API_URL` if needed.

## Notes

- The default RTSP URLs are in `config.py`.
- The detector labels file is `models/labels.txt`.
- The classifier labels file is `models/labels.json`.
- If your detector is a YOLO variant with raw outputs, set the custom parser values in `config.py`.
- If `/video/mosaic` stays blank, check `/health` and `/logs/detect` first. The app now keeps a recent detection buffer and reports sender errors on the log page.
