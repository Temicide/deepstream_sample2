import json
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def load_classifier_labels(path: str) -> Dict[str, str]:
    labels_path = Path(path)
    if not labels_path.exists():
        return {}

    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("classifier"), dict):
        data = data["classifier"]
    if not isinstance(data, dict):
        return {}

    return {str(key): str(value) for key, value in data.items()}


class CudaContextScope:
    _cuda = None
    _ctx = None
    _lock = threading.RLock()

    def __enter__(self):
        if self.__class__._ctx is None:
            import pycuda.driver as cuda

            cuda.init()
            self.__class__._cuda = cuda
            self.__class__._ctx = cuda.Device(0).make_context()
            cuda.Context.pop()
        self.__class__._lock.acquire()
        self.__class__._ctx.push()
        return self.__class__._cuda

    def __exit__(self, exc_type, exc, tb):
        self.__class__._cuda.Context.pop()
        self.__class__._lock.release()
        return False


class TensorRTBackend:
    def __init__(self, model_path: str, role: str = "model"):
        self.model_path = str(model_path)
        self.role = role
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_shape = []

        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with CudaContextScope() as cuda:
            with open(self.model_path, "rb") as handle:
                engine_bytes = handle.read()
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError("TensorRT failed to load engine: {0}".format(self.model_path))
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT failed to create execution context: {0}".format(self.model_path))

            self.bindings = [None] * int(self.engine.num_bindings)
            for index in range(int(self.engine.num_bindings)):
                shape = tuple(int(v) for v in self.engine.get_binding_shape(index))
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                if any(v < 0 for v in shape):
                    raise RuntimeError("Dynamic TensorRT shapes are not supported here: {0}".format(shape))

                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[index] = int(device_mem)
                item = {
                    "index": index,
                    "shape": shape,
                    "dtype": dtype,
                    "host": host_mem,
                    "device": device_mem,
                }
                if self.engine.binding_is_input(index):
                    self.inputs.append(item)
                    self.input_shape = [int(v) for v in shape]
                else:
                    self.outputs.append(item)

            if len(self.inputs) != 1:
                raise RuntimeError("Expected one input binding, got {0}".format(len(self.inputs)))
            self.stream = cuda.Stream()

    def nchw_size(self, default: int) -> int:
        if len(self.input_shape) == 4 and self.input_shape[2] == self.input_shape[3] and self.input_shape[2] > 0:
            return int(self.input_shape[2])
        return int(default)

    def run(self, tensor: np.ndarray):
        tensor = np.ascontiguousarray(tensor.astype(self.inputs[0]["dtype"], copy=False))
        if tuple(tensor.shape) != tuple(self.inputs[0]["shape"]):
            raise ValueError(
                "{0} input shape mismatch: got {1}, expected {2}".format(
                    self.role,
                    tuple(tensor.shape),
                    tuple(self.inputs[0]["shape"]),
                )
            )

        with CudaContextScope() as cuda:
            np.copyto(self.inputs[0]["host"], tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 failed for {0}".format(self.model_path))
            for item in self.outputs:
                cuda.memcpy_dtoh_async(item["host"], item["device"], self.stream)
            self.stream.synchronize()
            return [item["host"].reshape(item["shape"]).copy() for item in self.outputs]


class RuntimeModel:
    def __init__(self, model_path: str, backend: str = "auto", input_size: int = 224, role: str = "model"):
        self.model_path = str(model_path)
        self.backend = backend
        self.input_size = int(input_size)
        self.role = role
        self.enabled = Path(self.model_path).exists()
        self.error = ""
        self.trt_model = None
        self.session = None
        self.net = None
        self.input_shape = []
        self._run_lock = threading.Lock()

        if not self.enabled:
            self.error = "model file does not exist: {0}".format(self.model_path)
            self.backend = "disabled"
            return

        try:
            self._load()
        except Exception as exc:
            self.enabled = False
            self.error = repr(exc)
            self.backend = "disabled"

    def _load(self) -> None:
        suffix = Path(self.model_path).suffix.lower()
        if self.backend in ("auto", "trt") and suffix == ".engine":
            self.backend = "trt"
            self.trt_model = TensorRTBackend(self.model_path, role=self.role)
            self.input_shape = list(self.trt_model.input_shape)
            return

        if self.backend in ("auto", "ort") and suffix == ".onnx":
            try:
                import onnxruntime as ort

                providers = [
                    provider
                    for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                    if provider in ort.get_available_providers()
                ]
                if not providers:
                    providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                input_shape = self.session.get_inputs()[0].shape
                self.input_shape = [int(v) if isinstance(v, int) else int(self.input_size) for v in input_shape]
                self.backend = "ort"
                return
            except Exception:
                if self.backend == "ort":
                    raise

        if self.backend in ("auto", "opencv"):
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.backend = "opencv"
            return

        if self.backend == "trt":
            raise RuntimeError("TensorRT backend requires a .engine file: {0}".format(self.model_path))
        raise RuntimeError("Could not load model: {0}".format(self.model_path))

    def nchw_size(self, default: int) -> int:
        if self.trt_model is not None:
            return self.trt_model.nchw_size(default)
        if len(self.input_shape) == 4 and self.input_shape[2] == self.input_shape[3] and self.input_shape[2] > 0:
            return int(self.input_shape[2])
        return int(default)

    def run(self, tensor: np.ndarray):
        if not self.enabled:
            raise RuntimeError("Model is not loaded: {0}".format(self.model_path))

        with self._run_lock:
            if self.trt_model is not None:
                return self.trt_model.run(tensor)
            if self.session is not None:
                inputs = {self.session.get_inputs()[0].name: tensor.astype(np.float32, copy=False)}
                outputs = self.session.run(None, inputs)
                return [np.asarray(output) for output in outputs]
            if self.net is not None:
                self.net.setInput(tensor.astype(np.float32, copy=False))
                output_names = self.net.getUnconnectedOutLayersNames()
                outputs = self.net.forward(output_names)
                if isinstance(outputs, np.ndarray):
                    return [outputs]
                return [np.asarray(output) for output in outputs]

        raise RuntimeError("Model is not loaded: {0}".format(self.model_path))

    def status(self):
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "model_path": self.model_path,
            "input_shape": list(self.input_shape),
            "error": self.error,
        }


def preprocess_classifier(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.reshape(-1).astype(np.float32)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-12)


def classify_crop(
    model: RuntimeModel,
    labels: Dict[str, str],
    crop_bgr: np.ndarray,
    input_size: int,
    min_confidence: float,
) -> Tuple[str, Optional[float]]:
    if not model.enabled or crop_bgr is None or crop_bgr.size == 0:
        return "", None

    tensor = preprocess_classifier(crop_bgr, model.nchw_size(input_size))
    outputs = model.run(tensor)
    if not outputs:
        return "", None

    probs = softmax(outputs[0])
    brand_id = int(np.argmax(probs))
    confidence = float(probs[brand_id])
    if confidence < float(min_confidence):
        return "", confidence

    return labels.get(str(brand_id), "class_{0}".format(brand_id)), confidence
