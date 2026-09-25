from __future__ import annotations

import os

import cv2
import numpy as np

PERSON_CLASS = 0
_PAD = 114


class OnnxYolo:
    def __init__(self, path, imgsz=640, providers=None, gpu_mem_mb=0,
                 trt_cache="", buckets=(1, 2, 4)):
        import onnxruntime as ort

        self.imgsz = imgsz
        self.buckets = tuple(sorted(buckets))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        avail = ort.get_available_providers()
        specs = []
        for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]):
            if p not in avail:
                continue
            if p == "TensorrtExecutionProvider":
                opt = {"trt_fp16_enable": True,
                       "trt_engine_cache_enable": bool(trt_cache),
                       "trt_engine_cache_path": trt_cache,
                       "trt_timing_cache_enable": True}
                if trt_cache:
                    os.makedirs(trt_cache, exist_ok=True)
                specs.append((p, opt))
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
                print(f"[YOLO] {name} 사용 불가 ({type(e).__name__}), 다음 프로바이더로")
        else:
            raise RuntimeError(f"YOLO ONNX 세션 생성 실패: {last}")

        self.iname = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()
        self._bufs = {}
        self._canvas = None

    def _buf(self, n):
        buf = self._bufs.get(n)
        if buf is None:
            buf = self._bufs[n] = np.zeros((n, 3, self.imgsz, self.imgsz),
                                           np.float32)
        return buf

    def _letterbox_into(self, img, dst_chw):
        s = self.imgsz
        if self._canvas is None:
            self._canvas = np.empty((s, s, 3), np.uint8)
        h, w = img.shape[:2]
        r = min(s / h, s / w)
        nh, nw = min(s, int(round(h * r))), min(s, int(round(w * r)))
        top, left = (s - nh) // 2, (s - nw) // 2
        cv = self._canvas
        cv[:] = _PAD
        cv2.resize(img, (nw, nh), dst=cv[top:top + nh, left:left + nw],
                   interpolation=cv2.INTER_LINEAR)
        dst_chw[0] = cv[:, :, 2]
        dst_chw[1] = cv[:, :, 1]
        dst_chw[2] = cv[:, :, 0]
        return r, left, top

    @staticmethod
    def _decode(pred, r, padx, pady, W, H, conf, iou, max_det):
        sc = pred[4 + PERSON_CLASS]
        keep = np.flatnonzero(sc >= conf)
        if keep.size == 0:
            return []
        sc = sc[keep]
        cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
        x = (cx - bw * 0.5 - padx) / r
        y = (cy - bh * 0.5 - pady) / r
        w = bw / r
        h = bh / r
        rects = np.stack([x, y, w, h], 1)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), sc.tolist(), conf, iou)
        if len(idx) == 0:
            return []
        idx = np.array(idx).reshape(-1)
        idx = idx[np.argsort(-sc[idx])][:max_det]
        out = []
        for i in idx:
            x0 = max(0, int(round(x[i])))
            y0 = max(0, int(round(y[i])))
            x1 = min(W - 1, int(round(x[i] + w[i])))
            y1 = min(H - 1, int(round(y[i] + h[i])))
            if x1 > x0 and y1 > y0:
                out.append((x0, y0, x1, y1))
        return out

    def detect(self, frames, conf=0.35, iou=0.5, max_det=20):
        n = len(frames)
        b = next((x for x in self.buckets if n <= x), n)
        buf = self._buf(b)
        meta = [self._letterbox_into(f, buf[i]) for i, f in enumerate(frames)]
        view = buf[:n]
        np.multiply(view, np.float32(1.0 / 255.0), out=view)
        pred = self.sess.run(None, {self.iname: buf})[0]
        res = []
        for i, f in enumerate(frames):
            r, padx, pady = meta[i]
            H, W = f.shape[:2]
            res.append(self._decode(pred[i], r, padx, pady, W, H,
                                    conf, iou, max_det))
        return res

    def warmup(self, hw=(480, 640)):
        dummy = np.zeros((hw[0], hw[1], 3), np.uint8)
        for n in self.buckets:
            self.detect([dummy] * n)
