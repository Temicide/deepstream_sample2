# DeepStream RTSP API Project for Jetson Nano 4GB

โปรเจกต์นี้แยกเป็น 3 phase:

- Phase 1: รับ RTSP 5 กล้องด้วย DeepStream แล้วแสดงเป็น MJPEG API
- Phase 2: รับ RTSP 5 กล้อง แล้วแปลงภาพ mosaic เป็น grayscale ก่อนแสดงผ่าน API
- Phase 3: รับ RTSP 5 กล้อง แล้ว detect ด้วย YOLO ผ่าน DeepStream `nvinfer`, แสดง monitor, และส่ง detection JSON

ออกแบบให้รันทีละ phase เท่านั้น เพื่อไม่ให้ Jetson Nano decode กล้องซ้ำหลายรอบพร้อมกัน

## Tested target concept

เหมาะกับ Jetson Nano 4GB + JetPack 4.6.x + DeepStream 6.x

## Project structure

```text
deepstream_rtsp_api_project/
├── server.py
├── config.py
├── requirements.txt
├── scripts/
│   └── export_yolov8s_onnx.sh
├── pipelines/
│   ├── common.py
│   ├── model_config.py
│   ├── live_pipeline.py
│   ├── gray_pipeline.py
│   └── detect_pipeline.py
└── models/
    ├── config_infer_primary.txt
    └── labels.txt
```

## Install Python packages

แนะนำให้ใช้ Python environment ที่ DeepStream Python bindings ใช้ได้อยู่แล้ว

```bash
sudo apt update
sudo apt install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools python3-opencv

python3 -m pip install fastapi uvicorn
```

ถ้า `cv2` ใช้ไม่ได้ใน pip env ให้ใช้ system Python ก่อน:

```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import gi; print('gi ok')"
```

Phase 3 ต้องมี `pyds`:

```bash
python3 -c "import pyds; print('pyds ok')"
```

## Edit RTSP URLs

แก้ใน `config.py`

```python
RTSP_URLS = [
    "rtsp://user:pass@192.168.1.101:554/stream1",
    ...
]
```

## Run phase 1

```bash
python3 server.py --mode live --host 0.0.0.0 --port 8000
```

เปิด:

```text
http://JETSON_IP:8000/video/live
http://JETSON_IP:8000/health
```

## Run phase 2

```bash
python3 server.py --mode gray --host 0.0.0.0 --port 8000
```

เปิด:

```text
http://JETSON_IP:8000/video/gray
```

## Run phase 3

ใส่ YOLO ONNX model ไว้ที่ `models/yolov8s.onnx` หรือแก้ `YOLO_ONNX_PATH` ใน `config.py`

```python
YOLO_ONNX_PATH = "models/yolov8s.onnx"
YOLO_INPUT_DIMS = "3;224;224"
YOLO_NUM_CLASSES = 80
YOLO_CLUSTER_MODE = 2
```

เมื่อเริ่ม server ระบบจะ generate `models/config_infer_primary.txt` ให้อัตโนมัติ และตั้ง `model-engine-file` ตามชื่อ ONNX + batch 5 + precision เช่น:

```text
models/yolov8s.onnx_b5_gpu0_fp16.engine
```

ถ้าเปลี่ยนเป็น model ใหม่ ให้แก้ `YOLO_ONNX_PATH`, `YOLO_INPUT_DIMS`, `YOLO_NUM_CLASSES`, parser settings ถ้าจำเป็น แล้ว restart server. DeepStream/TensorRT จะ build engine ใหม่ในครั้งแรกที่รันกับ model นั้น

ถ้าคุณมีไฟล์ `yolov8s.pt` แล้ว อยาก export เป็น ONNX ให้รัน:

```bash
python3 -m pip install ultralytics
bash scripts/export_yolov8s_onnx.sh models/yolov8s.pt models/yolov8s.onnx
```

สคริปต์นี้จะ export ด้วย `imgsz=224` และบันทึกผลไว้ที่ `models/yolov8s.onnx`

สำหรับ Ultralytics YOLO:

- YOLOv8/v11 raw output มักใช้ `YOLO_CLUSTER_MODE = 2` และต้องมี custom bbox parser
- YOLOv10/v26 post-NMS output มักใช้ `YOLO_CLUSTER_MODE = 4`
- ถ้า ONNX เป็น dynamic shape ต้องมี `YOLO_INPUT_DIMS` เช่น `3;640;640`

ถ้า model ต้องใช้ custom parser:

```python
YOLO_CUSTOM_LIB_PATH = "./libnvdsinfer_custom_impl_Yolo.so"
YOLO_PARSE_BBOX_FUNC = "NvDsInferParseYolo"
```

```bash
python3 server.py --mode detect --host 0.0.0.0 --port 8000
```

ตอนแสดงผลระบบจะเขียน FPS overlay ลงบนภาพ mosaic และภาพรายกล้องให้อัตโนมัติ

เปิด:

```text
http://JETSON_IP:8000/monitor
http://JETSON_IP:8000/video/detect
http://JETSON_IP:8000/detections
http://JETSON_IP:8000/detections/cam/1
```

## Send detection JSON to another server

ถ้าต้องการให้ pipeline POST detection JSON ไป backend อื่น ให้แก้ใน `config.py`:

```python
DETECTION_SERVER_URL = "http://YOUR_SERVER:9000/detections"
DETECTION_POST_INTERVAL_SEC = 0.25
DETECTION_MIN_CONFIDENCE = 0.25
```

ถ้า `DETECTION_SERVER_URL` ว่างไว้ ระบบจะไม่ POST ออก แต่ยังดู JSON ได้จาก `/detections`

## Recommended camera setting for Jetson Nano 4GB

- H.264
- 640x360 หรือ 640x480
- 5-10 FPS
- bitrate 512 kbps - 1500 kbps ต่อกล้อง
- เริ่มจาก 2 กล้องก่อน แล้วค่อยเพิ่มเป็น 5 กล้อง

## Important

ถ้าภาพเขียว แตก หรือกระพริบ:

1. ลด resolution ของกล้อง
2. ลด FPS
3. เพิ่ม bitrate ให้พอดี ไม่สูงเกิน
4. ใช้ H.264 ก่อน
5. เพิ่ม network stability
6. ทดสอบ RTSP ทีละกล้องด้วย `gst-launch-1.0`
