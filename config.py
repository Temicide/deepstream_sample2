from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# แก้ RTSP URL ให้ตรงกับกล้องของคุณ
RTSP_URLS = [
    "rtsp://10.0.11.153:8554/cctv01",
    "rtsp://10.0.11.153:8554/cctv02",
    "rtsp://10.0.11.153:8554/cctv03",
    "rtsp://10.0.11.153:8554/cctv04",
    "rtsp://10.0.11.153:8554/cctv05",
]

# Pipeline นี้ออกแบบให้รับได้สูงสุด 5 RTSP sources
MAX_RTSP_SOURCES = 5

# ค่าเริ่มต้นที่เหมาะกับ Jetson Nano 4GB
MUX_WIDTH = 640
MUX_HEIGHT = 640
MUX_BATCH_SIZE = len(RTSP_URLS)
MUX_TIMEOUT_USEC = 40000

# Tiler รวมภาพ 5 กล้องเป็น mosaic เดียว
TILER_ROWS = 2
TILER_COLUMNS = 3
TILER_WIDTH = 1280
TILER_HEIGHT = 720

# API MJPEG setting
OUTPUT_FPS = 8
JPEG_QUALITY = 70

# Phase 3
INFER_CONFIG_PATH = "models/config_infer_primary.txt"
LABELS_PATH = "models/labels.txt"

# YOLO model settings
# เปลี่ยนไฟล์โมเดลในอนาคตได้โดยแก้ YOLO_ONNX_PATH แล้ว restart server
AUTO_GENERATE_INFER_CONFIG = True
YOLO_ONNX_PATH = "models/yolov8s.onnx"
YOLO_ENGINE_PATH = ""  # ว่างไว้เพื่อ generate path อัตโนมัติจากชื่อ ONNX + batch + precision
YOLO_INPUT_DIMS = "3;640;640"
YOLO_NUM_CLASSES = 80
YOLO_INTERVAL = 1
YOLO_NETWORK_MODE = 2  # 0=FP32, 1=INT8, 2=FP16
YOLO_CLUSTER_MODE = 2  # raw YOLOv8 output with this parser; post-NMS models need a matching parser
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_NMS_IOU_THRESHOLD = 0.45
YOLO_CUSTOM_LIB_PATH = "nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"  # เช่น ./libnvdsinfer_custom_impl_Yolo.so
YOLO_PARSE_BBOX_FUNC = "NvDsInferParseYolo"  # เช่น NvDsInferParseYolo
YOLO_OUTPUT_BLOB_NAMES = "output0"  # optional, ใช้เมื่อ parser/model ต้องระบุ output layer

# Tracker settings
# เปิด nvtracker เพื่อให้ bbox และ track_id ต่อเนื่องข้าม frame โดยยังคุม latency/FPS
TRACKER_ENABLE = True
TRACKER_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CONFIG_PATH = "models/config_tracker_NvSORT.yml"
TRACKER_WIDTH = 640
TRACKER_HEIGHT = 384
TRACKER_DISPLAY_ID = 1
TRACKER_GPU_ID = 0
TRACKER_COMPUTE_HW = 0  # 0=default, 1=GPU, 2=VIC on Jetson

# Vehicle color detection: COCO 0-indexed class IDs (car, motorcycle, bus, truck)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Vehicle brand classifier.
# Uses the semifinal classifier artifacts, but runs inside the root detect pipeline.
CLASSIFIER_MODEL_PATH = str(PROJECT_ROOT / "deepstream_semifinal" / "models" / "cls.onnx")
CLASSIFIER_ENGINE_PATH = ""
CLASSIFIER_LABELS_PATH = str(PROJECT_ROOT / "deepstream_semifinal" / "models" / "labels.json")
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_BACKEND = "auto"  # auto, ort, opencv, or trt for .engine files
CLASSIFIER_MIN_CONFIDENCE = 0.0
CLASSIFIER_OPERATE_ON_CLASS_IDS = VEHICLE_CLASS_IDS
CLASSIFIER_CACHE_TTL_SEC = 30.0
CLASSIFIER_CACHE_MAX_SIZE = 512
CLASSIFIER_MAX_PER_CAMERA_FRAME = 1

# Optional: POST detection JSON to another backend.
# ถ้าว่างไว้จะยังเปิด /detections ให้ server นี้อ่านได้ตามปกติ
DETECTION_SERVER_URL = "http://10.0.11.153:8080/api/v1/raw_data/batch"
DETECTION_POST_INTERVAL_SEC = 0.25
DETECTION_POST_TIMEOUT_SEC = 1.0
DETECTION_MIN_CONFIDENCE = 0.25
JETSON_ID = "jetson-nano-01"
