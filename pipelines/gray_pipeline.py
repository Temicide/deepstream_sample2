import cv2
import numpy as np

from pipelines.common import BaseDeepStreamPipeline


class GrayPipeline(BaseDeepStreamPipeline):
    """
    Phase 2:
    RTSP x5 -> decode -> nvstreammux -> tiler -> appsink -> grayscale -> FastAPI MJPEG

    ทำ grayscale หลังจาก tiler เพื่อลดภาระ CPU:
    5 frames -> 1 mosaic frame -> grayscale
    """

    def __init__(self):
        super().__init__(mode_name="gray")

    def process_frame_before_jpeg(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
