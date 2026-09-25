import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pylibs"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np

import jetson_par_sender as J
from onnx_yolo import OnnxYolo

SAMPLE = os.path.join(ROOT, "samples", "bus.jpg")
LOG = os.path.join(ROOT, "output", "prepare_report.txt")
REPEAT = 20


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def hr(t):
    print(f"\n{'─' * 68}\n{t}\n{'─' * 68}", flush=True)


def build_yolo(mode):
    return OnnxYolo(J.YOLO_ONNX, imgsz=J.IMG_SIZE, providers=J._PROV[mode],
                    trt_cache=J.TRT_CACHE if mode == "trt" else "",
                    buckets=J.YOLO_BUCKETS)


def build_par(mode):
    return J.ColorPAR(J.PAR_ONNX, mode, J.PAR_GPU_MEM_MB,
                      J.TRT_CACHE if mode == "trt" else "")


def bench(fn, n_per_call, label):
    fn()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn()
    dt = (time.perf_counter() - t0) / REPEAT
    print(f"  {label:22s} {dt * 1000:7.1f} ms/call  "
          f"{dt * 1000 / n_per_call:6.1f} ms/장  {n_per_call / dt:6.1f} fps",
          flush=True)


def box_iou(a, b):
    return J._iou(a, b)


def compare_yolo(img):
    hr("1) YOLO 정확도: TensorRT fp16 vs CUDA fp32")
    a = build_yolo("trt").detect([img])[0]
    b = build_yolo("cuda").detect([img])[0]
    print(f"  검출 인원   fp16={len(a)}  fp32={len(b)}", flush=True)
    if not a or not b:
        print("  ⚠ 한쪽이 0명이다. 안전하게 fp32(cuda)로 간다.", flush=True)
        return False, f"검출 인원 fp16={len(a)} fp32={len(b)}"
    ious, dpx = [], []
    for bb in b:
        j = max(range(len(a)), key=lambda k: box_iou(a[k], bb))
        ious.append(box_iou(a[j], bb))
        dpx.append(max(abs(a[j][k] - bb[k]) for k in range(4)))
    print(f"  박스 IoU    최소 {min(ious):.4f}  평균 {sum(ious) / len(ious):.4f}",
          flush=True)
    print(f"  좌표 차이   최대 {max(dpx)} px", flush=True)
    ok = (len(a) == len(b)) and min(ious) >= 0.90
    why = (f"인원 {len(a)}={len(b)}, 최소 IoU {min(ious):.4f}, "
           f"좌표차 {max(dpx)}px")
    print(f"  판정: {'fp16 사용 가능' if ok else 'fp16 부적합 -> fp32로'}  ({why})",
          flush=True)
    return ok, why


def crops_from(img, boxes, want=8):
    out = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    while out and len(out) < want:
        out.append(out[len(out) % len(boxes)])
    return out[:want]


def compare_par(crops):
    hr("2) PAR 정확도: TensorRT fp16 vs CUDA fp32")
    a = build_par("trt").predict(crops)
    b = build_par("cuda").predict(crops)
    names = ("성별", "상의색", "하의색", "소매")
    flips = 0
    for nm, x, y in zip(names, a, b):
        d = float(np.abs(x - y).max())
        f = int((x.argmax(1) != y.argmax(1)).sum())
        flips += f
        print(f"  {nm:6s} 확률 최대차 {d:.5f}   라벨 뒤집힘 {f}/{len(x)}명",
              flush=True)
    q = (0, 9, 0, 0)
    sa = np.array([J.score_probs(tuple(v[i] for v in a), q) for i in range(len(crops))])
    sb = np.array([J.score_probs(tuple(v[i] for v in b), q) for i in range(len(crops))])
    cross = int(((sa >= J.MATCH_THRESHOLD) != (sb >= J.MATCH_THRESHOLD)).sum())
    print(f"  매칭 점수 최대차 {np.abs(sa - sb).max():.5f}  "
          f"문턱({J.MATCH_THRESHOLD}) 판정 뒤집힘 {cross}/{len(crops)}명", flush=True)
    ok = (flips == 0 and cross == 0)
    why = f"라벨 뒤집힘 {flips}건, 매칭 판정 뒤집힘 {cross}건 (총 {len(crops)}명)"
    print(f"  판정: {'fp16 사용 가능' if ok else 'fp16 부적합 -> fp32로'}  ({why})",
          flush=True)
    return ok, why


def speed(img, crops):
    hr("3) 속도")
    for mode in ("trt", "cuda"):
        print(f"[YOLO {mode}]", flush=True)
        y = build_yolo(mode)
        for n in J.YOLO_BUCKETS:
            frames = [img] * n
            bench(lambda: y.detect(frames), n, f"batch{n}")
        del y
    for mode in ("trt", "cuda"):
        print(f"[PAR {mode}]", flush=True)
        p = build_par(mode)
        for n in J.PAR_BUCKETS:
            c = crops[:n] if len(crops) >= n else (crops * n)[:n]
            bench(lambda: p.predict(c), n, f"batch{n}")
        del p


def main():
    sys.stdout = Tee(LOG)
    check_only = "--check" in sys.argv
    img = cv2.imread(SAMPLE)
    if img is None:
        sys.exit(f"샘플 이미지가 없다: {SAMPLE}")

    if not check_only:
        hr("0) TensorRT 엔진 빌드 (처음이면 배치마다 몇 분씩 걸린다)")
        free = int(os.popen("free -m | awk '/^Mem:/{print $7}'").read() or 0)
        print(f"  가용 RAM {free} MB", flush=True)
        if free < 3000:
            print("  ⚠ 가용 RAM이 3GB 미만이다. 브라우저/VS Code를 끄고 다시 실행해라.\n"
                  "    이대로 구우면 TensorRT가 tactic을 건너뛰어 느린 엔진이 박힌다.",
                  flush=True)
            if not sys.stdin.isatty():
                sys.exit("  입력받을 터미널이 없어서 중단한다. 메모리를 확보하고 다시 실행해라.")
            if input("  그래도 진행할까? [y/N] ").strip().lower() != "y":
                sys.exit("  중단했다. 메모리를 확보하고 다시 실행해라.")
        os.makedirs(J.TRT_CACHE, exist_ok=True)
        for what, mk in (("YOLO", build_yolo), ("PAR", build_par)):
            t0 = time.time()
            m = mk("trt")
            m.warmup()
            print(f"  {what} 엔진 준비 완료 {time.time() - t0:.0f}s", flush=True)
            del m

    boxes = build_yolo("trt" if not check_only else "trt").detect([img])[0]
    print(f"\n샘플 bus.jpg 검출 {len(boxes)}명 (정답 4명)", flush=True)
    crops = crops_from(img, boxes, max(J.PAR_BUCKETS))

    yolo_ok, yolo_why = compare_yolo(img)
    par_ok, par_why = compare_par(crops)
    speed(img, crops)

    hr("결론")
    result = {"yolo": "trt" if yolo_ok else "cuda",
              "par": "trt" if par_ok else "cuda",
              "yolo_why": yolo_why, "par_why": par_why}
    with open(J.VERDICT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  YOLO -> {result['yolo']}   ({yolo_why})", flush=True)
    print(f"  PAR  -> {result['par']}    ({par_why})", flush=True)
    if yolo_ok and par_ok:
        print("\n  fp16이 fp32와 같은 결과를 냈다. TensorRT로 간다.", flush=True)
    else:
        print("\n  fp16에서 결과가 달라진 부분이 있어 그쪽은 fp32(cuda)로 정했다.",
              flush=True)
    print(f"\n  판정 저장: {J.VERDICT}", flush=True)
    print(f"  전체 기록: {LOG}", flush=True)
    print("  이 판정은 jetson_par_sender.py 가 시작할 때 자동으로 읽는다.\n"
          "  손댈 것 없이 그냥 실행하면 된다:\n"
          "      python3 ~/project/service/scripts/jetson_par_sender.py", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
