import os
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def _env_list(name, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


RTSP_URLS = _env_list(
    "DEEPSTREAM_RTSP_URLS",
    [
        "rtsp://10.0.11.153:8554/cctv01",
        "rtsp://10.0.11.153:8554/cctv02",
        "rtsp://10.0.11.153:8554/cctv03",
        "rtsp://10.0.11.153:8554/cctv04",
        "rtsp://10.0.11.153:8554/cctv05",
    ],
)

MAX_RTSP_SOURCES = 5
SOURCE_BIN_FACTORY = os.getenv("DEEPSTREAM_SOURCE_BIN_FACTORY", "uridecodebin").strip()

OUTPUT_FPS = int(os.getenv("DEEPSTREAM_OUTPUT_FPS", "8"))
JPEG_QUALITY = int(os.getenv("DEEPSTREAM_JPEG_QUALITY", "70"))
DETECTION_LOG_API_URL = os.getenv(
    "DEEPSTREAM_DETECTION_LOG_API_URL",
    "http://10.0.11.153:8080/api/v1/raw_data/batch",
).strip()
DETECTION_LOG_TIMEOUT_SEC = float(os.getenv("DEEPSTREAM_DETECTION_LOG_TIMEOUT_SEC", "2.0"))
DETECTION_LOG_BATCH_SIZE = int(os.getenv("DEEPSTREAM_DETECTION_LOG_BATCH_SIZE", "25"))
DETECTION_LOG_FLUSH_INTERVAL_SEC = float(os.getenv("DEEPSTREAM_DETECTION_LOG_FLUSH_INTERVAL_SEC", "1.0"))
DETECTION_LOG_MAX_RECENT = int(os.getenv("DEEPSTREAM_DETECTION_LOG_MAX_RECENT", "300"))
JETSON_ID = os.getenv("DEEPSTREAM_JETSON_ID", "jetson-nano-01").strip()

MUX_WIDTH = int(os.getenv("DEEPSTREAM_MUX_WIDTH", "640"))
MUX_HEIGHT = int(os.getenv("DEEPSTREAM_MUX_HEIGHT", "360"))
MUX_TIMEOUT_USEC = int(os.getenv("DEEPSTREAM_MUX_TIMEOUT_USEC", "40000"))

TILER_ROWS = int(os.getenv("DEEPSTREAM_TILER_ROWS", "2"))
TILER_COLUMNS = int(os.getenv("DEEPSTREAM_TILER_COLUMNS", "3"))
TILER_WIDTH = int(os.getenv("DEEPSTREAM_TILER_WIDTH", "960"))
TILER_HEIGHT = int(os.getenv("DEEPSTREAM_TILER_HEIGHT", "720"))

DETECTOR_MODEL_PATH = os.getenv("DEEPSTREAM_DETECTOR_MODEL", str(MODELS_DIR / "yolo.onnx"))
DETECTOR_ENGINE_PATH = os.getenv("DEEPSTREAM_DETECTOR_ENGINE", "").strip()
DETECTOR_LABELS_PATH = os.getenv("DEEPSTREAM_DETECTOR_LABELS", str(MODELS_DIR / "labels.txt"))
DETECTOR_INFER_DIMS = os.getenv("DEEPSTREAM_DETECTOR_INFER_DIMS", "3;640;640")
DETECTOR_NUM_CLASSES = int(os.getenv("DEEPSTREAM_DETECTOR_NUM_CLASSES", "80"))
DETECTOR_INTERVAL = int(os.getenv("DEEPSTREAM_DETECTOR_INTERVAL", "0"))
DETECTOR_NETWORK_MODE = int(os.getenv("DEEPSTREAM_DETECTOR_NETWORK_MODE", "2"))
DETECTOR_CLUSTER_MODE = int(os.getenv("DEEPSTREAM_DETECTOR_CLUSTER_MODE", "4"))
DETECTOR_CONFIDENCE_THRESHOLD = float(os.getenv("DEEPSTREAM_DETECTOR_CONF", "0.25"))
DETECTOR_NMS_IOU_THRESHOLD = float(os.getenv("DEEPSTREAM_DETECTOR_IOU", "0.45"))
DETECTOR_CUSTOM_LIB_PATH = os.getenv(
    "DEEPSTREAM_DETECTOR_CUSTOM_LIB",
    str(PROJECT_ROOT / "models" / "libnvdsinfer_custom_impl_Yolo.so"),
).strip()
DETECTOR_PARSE_BBOX_FUNC = os.getenv("DEEPSTREAM_DETECTOR_PARSE_FUNC", "NvDsInferParseYolo").strip()
DETECTOR_OUTPUT_BLOB_NAMES = os.getenv("DEEPSTREAM_DETECTOR_OUTPUT_BLOB_NAMES", "").strip()

CLASSIFIER_MODEL_PATH = os.getenv("DEEPSTREAM_CLASSIFIER_MODEL", str(MODELS_DIR / "cls.onnx"))
CLASSIFIER_ENGINE_PATH = os.getenv("DEEPSTREAM_CLASSIFIER_ENGINE", "").strip()
CLASSIFIER_LABELS_PATH = os.getenv("DEEPSTREAM_CLASSIFIER_LABELS", str(MODELS_DIR / "labels.json"))
CLASSIFIER_INPUT_SIZE = int(os.getenv("DEEPSTREAM_CLASSIFIER_INPUT_SIZE", "224"))
CLASSIFIER_BACKEND = os.getenv("DEEPSTREAM_CLASSIFIER_BACKEND", "auto")
CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("DEEPSTREAM_CLASSIFIER_MIN_CONFIDENCE", "0.0"))
CLASSIFIER_OPERATE_ON_CLASS_IDS = os.getenv("DEEPSTREAM_CLASSIFIER_CLASS_IDS", "2;3;5;7")

PRIMARY_GIE_ID = 1
SECONDARY_GIE_ID = 2

VEHICLE_CLASS_IDS = {2, 3, 5, 7}

