from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PYLIBS = os.path.join(ROOT, "pylibs")
if _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

from onnx_yolo import OnnxYolo

PI5_IP = ""
RTSP_PORT = 8554

YOLO_ONNX = os.path.join(ROOT, "models", "yolo11s.onnx")
PAR_ONNX = os.path.join(ROOT, "models", "color_par_v4_multi_resnet50_sleeve.onnx")
TRT_CACHE = os.path.join(ROOT, "models", "trt_cache")
VERDICT = os.path.join(ROOT, "models", "provider_verdict.json")

YOLO_PROVIDER = "trt"
PAR_PROVIDER = "trt"
FORCE_PROVIDER = ""
PAR_GPU_MEM_MB = 0

IMG_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MAX_DET = 20
YOLO_BUCKETS = (1, 2, 4)

PAR_MAX_CROPS = 8
PAR_CACHE_TTL = 12
PAR_CACHE_IOU = 0.5
PAR_BUCKETS = (1, 2, 4, 8)

SLEEVE_LONG_MIN = 0.95

FULLBODY_EDGE_MARGIN = 6
FULLBODY_MIN_RATIO = 1.8
FULLBODY_MIN_HEIGHT = 120

FULLBODY_MAX_RATIO = 4.5
FULLBODY_MIN_WIDTH = 70

GENDERS = ["male", "female"]
COLORS = ["black", "blue", "brown", "green", "gray", "orange", "pink",
          "purple", "red", "white", "yellow"]
SLEEVES = ["short", "long"]
INPUT_HW = (256, 128)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

_PROV = {"trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                 "CPUExecutionProvider"],
         "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
         "cpu": ["CPUExecutionProvider"]}


class ColorPAR:
    def __init__(self, path, provider="trt", gpu_mem_mb=0, trt_cache=""):
        import onnxruntime as ort

        self.H, self.W = INPUT_HW
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        avail = ort.get_available_providers()
        specs = []
        for p in _PROV.get(provider, _PROV["cuda"]):
            if p not in avail:
                continue
            if p == "TensorrtExecutionProvider":
                if trt_cache:
                    os.makedirs(trt_cache, exist_ok=True)
                specs.append((p, {"trt_fp16_enable": True,
                                  "trt_engine_cache_enable": bool(trt_cache),
                                  "trt_engine_cache_path": trt_cache,
                                  "trt_timing_cache_enable": True}))
            elif p == "CUDAExecutionProvider":
                opt = {"arena_extend_strategy": "kSameAsRequested",
                       "cudnn_conv_use_max_workspace": "0"}
                if gpu_mem_mb:
                    opt["gpu_mem_limit"] = str(gpu_mem_mb * 1024 * 1024)
                specs.append((p, opt))
            else:
                specs.append(p)

        last = None
        for i in range(len(specs)):
            try:
                self.sess = ort.InferenceSession(path, sess_options=so,
                                                 providers=specs[i:])
                break
            except Exception as e:
                last = e
                name = specs[i][0] if isinstance(specs[i], tuple) else specs[i]
                print(f"[PAR] {name} 사용 불가 ({type(e).__name__}), 다음 프로바이더로")
        else:
            raise RuntimeError(f"PAR ONNX 세션 생성 실패: {last}")

        self.iname = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()
        self._scale = ((1.0 / 255.0) / _STD).astype(np.float32)
        self._bias = (-_MEAN / _STD).astype(np.float32)
        self._bufs = {}

    def _buf(self, n):
        b = next((x for x in PAR_BUCKETS if n <= x), n)
        buf = self._bufs.get(b)
        if buf is None:
            buf = self._bufs[b] = np.zeros((b, 3, self.H, self.W), np.float32)
        return buf

    def _pre(self, crops_bgr):
        buf = self._buf(len(crops_bgr))
        for i, c in enumerate(crops_bgr):
            r = cv2.resize(c, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            buf[i, 0] = r[:, :, 2]
            buf[i, 1] = r[:, :, 1]
            buf[i, 2] = r[:, :, 0]
        n = len(crops_bgr)
        view = buf[:n]
        np.multiply(view, self._scale, out=view)
        np.add(view, self._bias, out=view)
        return buf, n

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, crops_bgr):
        buf, n = self._pre(crops_bgr)
        g, u, d, s = self.sess.run(None, {self.iname: buf})
        return (self._softmax(g[:n]), self._softmax(u[:n]),
                self._softmax(d[:n]), self._softmax(s[:n]))

    def warmup(self):
        crop = np.zeros((128, 64, 3), np.uint8)
        for n in PAR_BUCKETS:
            self.predict([crop] * n)


def score_probs(p, q):
    s = 1.0
    for row, idx in zip(p, q):
        if idx is not None:
            s *= float(row[idx])
    return s


def sleeve_label(s_row):
    return "long" if float(s_row[1]) >= SLEEVE_LONG_MIN else "short"


def attr_text(p):
    return (f"{GENDERS[int(p[0].argmax())]},{COLORS[int(p[1].argmax())]},"
            f"{COLORS[int(p[2].argmax())]},{sleeve_label(p[3])}")


def fullbody_reject(box, fw, fh):
    x0, y0, x1, y1 = box
    m = FULLBODY_EDGE_MARGIN
    if x0 <= m or y0 <= m or x1 >= fw - m or y1 >= fh - m:
        return "cut"
    w, h = x1 - x0, y1 - y0
    if h < FULLBODY_MIN_HEIGHT:
        return "small"
    if w < FULLBODY_MIN_WIDTH:
        return "sliver"
    r = h / w
    if r < FULLBODY_MIN_RATIO:
        return "upper"
    if r > FULLBODY_MAX_RATIO:
        return "thin"
    return ""


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


_TRACK_SEQ = itertools.count(1)


class AttrTracker:
    def __init__(self, ttl, iou_thr):
        self.ttl, self.iou_thr = ttl, iou_thr
        self.items = []
        self._claimed = set()

    def tick(self):
        for it in self.items:
            it["age"] += 1
        self.items = [it for it in self.items if it["age"] <= self.ttl * 3]
        self._claimed = set()

    def take(self, box):
        best, bi = self.iou_thr, -1
        for k, it in enumerate(self.items):
            if k in self._claimed:
                continue
            v = _iou(box, it["box"])
            if v > best:
                best, bi = v, k
        if bi < 0:
            return -1, None, None
        self._claimed.add(bi)
        it = self.items[bi]
        it["box"] = box
        if it["p"] is not None and it["age"] <= self.ttl:
            return bi, it["p"], it["tid"]
        return bi, None, it["tid"]

    def put(self, slot, box, probs):
        if 0 <= slot < len(self.items):
            it = self.items[slot]
            it.update(box=box, p=probs, age=0)
            return it["tid"]
        tid = next(_TRACK_SEQ)
        self.items.append({"box": box, "p": probs, "age": 0, "tid": tid})
        self._claimed.add(len(self.items) - 1)
        return tid


class CameraReceiver(threading.Thread):
    def __init__(self, url, name=""):
        super().__init__(daemon=True)
        self.url = url
        self.name_ = name or url
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self.connected = False
        self._stopev = threading.Event()

    def stop(self):
        self._stopev.set()

    def snapshot(self):
        with self._lock:
            return self._frame, self._count

    def run(self):
        announced = False
        while not self._stopev.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                if not announced:
                    print(f"[cam] {self.name_} 대기중", flush=True)
                    announced = True
                self._stopev.wait(3.0)
                continue
            announced = False
            self.connected = True
            print(f"[cam] connected {self.name_} "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
            while not self._stopev.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with self._lock:
                    self._frame, self._count = frame, self._count + 1
            cap.release()
            self.connected = False
            print(f"[cam] lost {self.name_}, 재접속 대기", flush=True)
            self._stopev.wait(2.0)


def resolve_providers():
    if FORCE_PROVIDER:
        print(f"[prov] FORCE_PROVIDER={FORCE_PROVIDER} (판정 파일 무시)", flush=True)
        return FORCE_PROVIDER, FORCE_PROVIDER
    try:
        with open(VERDICT, encoding="utf-8") as f:
            v = json.load(f)
        y, p = v.get("yolo"), v.get("par")
        if y in _PROV and p in _PROV:
            print(f"[prov] prepare.py 판정 적용: YOLO={y} PAR={p}", flush=True)
            for k, why in (("YOLO", v.get("yolo_why")), ("PAR", v.get("par_why"))):
                if why:
                    print(f"[prov]   {k}: {why}", flush=True)
            return y, p
        print(f"[prov] 판정 파일 내용이 이상하다({y},{p}). 기본값을 쓴다.", flush=True)
    except FileNotFoundError:
        print(f"[prov] 판정 파일 없음 -> 기본값 YOLO={YOLO_PROVIDER} "
              f"PAR={PAR_PROVIDER}", flush=True)
        print("[prov] scripts/prepare.py 를 한 번 돌리면 fp16 안전 여부를 재서 "
              "자동으로 정해준다.", flush=True)
    except (OSError, ValueError) as e:
        print(f"[prov] 판정 파일 읽기 실패({type(e).__name__}). 기본값을 쓴다.",
              flush=True)
    return YOLO_PROVIDER, PAR_PROVIDER


def warn_if_cpu(name, providers):
    if any(p.startswith(("CUDA", "Tensorrt")) for p in providers):
        return False
    bar = "!" * 70
    print(f"\n{bar}\n[{name}] ⚠ GPU를 못 쓰고 CPU로 돌고 있다. 수십 배 느리다.\n"
          f"[{name}]   providers = {providers}\n"
          f"[{name}]   GPU 메모리 부족일 가능성이 크다. 확인할 것:\n"
          f"[{name}]   1) 이 스크립트가 이미 떠 있지 않은지"
          f"  ->  pgrep -af jetson_par_sender\n"
          f"[{name}]   2) 브라우저 / VS Code 를 닫았는지\n"
          f"[{name}]   3) 그래도 안 되면 PAR_GPU_MEM_MB = 320 으로 아레나를 조여라\n"
          f"{bar}\n", flush=True)
    return True


def build_models():
    ymode, pmode = resolve_providers()

    print(f"[YOLO] loading {YOLO_ONNX} ({ymode})", flush=True)
    yolo = OnnxYolo(YOLO_ONNX, imgsz=IMG_SIZE, providers=_PROV[ymode],
                    trt_cache=TRT_CACHE if ymode == "trt" else "",
                    buckets=YOLO_BUCKETS)
    print(f"[YOLO] providers: {yolo.providers}", flush=True)

    print(f"[PAR] loading {PAR_ONNX} ({pmode})", flush=True)
    par = ColorPAR(PAR_ONNX, pmode, PAR_GPU_MEM_MB,
                   TRT_CACHE if pmode == "trt" else "")
    print(f"[PAR] providers: {par.providers}", flush=True)

    print("[warmup] 배치별 엔진 준비중 (TensorRT 첫 실행이면 몇 분 걸린다)",
          flush=True)
    t0 = time.time()
    yolo.warmup()
    par.warmup()
    print(f"[warmup] done {time.time() - t0:.1f}s", flush=True)

    cpu = warn_if_cpu("YOLO", yolo.sess.get_providers())
    cpu |= warn_if_cpu("PAR", par.sess.get_providers())
    if not cpu:
        print("[dev] YOLO/PAR 모두 GPU 사용중", flush=True)
    return yolo, par
