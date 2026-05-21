from pipelines.common import BaseDeepStreamPipeline


class LivePipeline(BaseDeepStreamPipeline):
    """
    Phase 1:
    RTSP x5 -> decode -> nvstreammux -> tiler -> appsink -> FastAPI MJPEG
    """

    def __init__(self):
        super().__init__(mode_name="live")
