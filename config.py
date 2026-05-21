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
YOLO_INPUT_DIMS = "3;224;224"
YOLO_NUM_CLASSES = 80
YOLO_INTERVAL = 1
YOLO_NETWORK_MODE = 2  # 0=FP32, 1=INT8, 2=FP16
YOLO_CLUSTER_MODE = 2  # YOLOv8/v11 raw output: 2, YOLOv10/v26 post-NMS: 4
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_NMS_IOU_THRESHOLD = 0.45
YOLO_CUSTOM_LIB_PATH = "nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"  # เช่น ./libnvdsinfer_custom_impl_Yolo.so
YOLO_PARSE_BBOX_FUNC = "NvDsInferParseYolo"  # เช่น NvDsInferParseYolo
YOLO_OUTPUT_BLOB_NAMES = "output0"  # optional, ใช้เมื่อ parser/model ต้องระบุ output layer

# Optional: POST detection JSON to another backend.
# ถ้าว่างไว้จะยังเปิด /detections ให้ server นี้อ่านได้ตามปกติ
DETECTION_SERVER_URL = ""
DETECTION_POST_INTERVAL_SEC = 0.25
DETECTION_POST_TIMEOUT_SEC = 1.0
DETECTION_MIN_CONFIDENCE = 0.0
