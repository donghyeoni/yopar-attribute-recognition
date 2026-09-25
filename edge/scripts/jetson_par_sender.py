from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pylibs"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0",
)

import cv2

import server_config as scfg
from server_link import DetectionReporter, TargetSync, rfc3339, utc_now

from capture_core import (
    COLORS, CONF_THRESHOLD, FORCE_PROVIDER, FULLBODY_EDGE_MARGIN,
    FULLBODY_MAX_RATIO, FULLBODY_MIN_HEIGHT, FULLBODY_MIN_RATIO,
    FULLBODY_MIN_WIDTH, GENDERS, IMG_SIZE, IOU_THRESHOLD, MAX_DET,
    PAR_BUCKETS, PAR_CACHE_IOU, PAR_CACHE_TTL, PAR_GPU_MEM_MB, PAR_MAX_CROPS,
    PAR_ONNX, PAR_PROVIDER, PI5_IP, RTSP_PORT, SLEEVES, SLEEVE_LONG_MIN,
    TRT_CACHE, VERDICT, YOLO_BUCKETS, YOLO_ONNX, YOLO_PROVIDER, _PROV,
    AttrTracker, CameraReceiver, ColorPAR, attr_text, build_models,
    fullbody_reject, resolve_providers, score_probs, sleeve_label,
    warn_if_cpu, _iou,
)


ALWAYS_ON_CAMERAS = ["camera-01", "camera-02", "camera-03", "camera-04"]

PC_IP = ""
VIEW_PORT = 5006
SEND_TO_PC = True
SAVE_LAST_DIR = os.path.join(ROOT, "output")
SAVE_EVERY = 30

TOPK = 3

MATCH_THRESHOLD = 0.25

JPEG_QUALITY = 75

FULLBODY_ONLY = True

SEND_MAX_W = 960
CV_THREADS = 4
STAT_EVERY = 60

LABELS = (GENDERS, COLORS, SLEEVES)


_COLOR = {"black": "black", "검정": "black", "검은": "black", "블랙": "black",
          "white": "white", "흰": "white", "하양": "white", "화이트": "white",
          "red": "red", "빨강": "red", "빨간": "red", "레드": "red",
          "purple": "purple", "보라": "purple", "퍼플": "purple",
          "yellow": "yellow", "노랑": "yellow", "노란": "yellow", "옐로": "yellow",
          "gray": "gray", "grey": "gray", "회색": "gray", "그레이": "gray",
          "blue": "blue", "파랑": "blue", "파란": "blue", "블루": "blue", "청": "blue",
          "green": "green", "초록": "green", "녹색": "green", "그린": "green",
          "pink": "pink", "분홍": "pink", "핑크": "pink",
          "brown": "brown", "갈색": "brown", "브라운": "brown",
          "orange": "orange", "주황": "orange", "오렌지": "orange"}
_UP_MARK = ["상의", "윗옷", "top", "upper", "shirt", "sleeve", "jacket", "coat",
            "tee", "hoodie", "hood", "후드", "sweater", "니트", "blouse"]
_DOWN_MARK = ["하의", "아래", "bottom", "lower", "pants", "trouser", "jeans",
              "shorts", "skirt", "바지", "치마"]
_SHORT_MARK = ["short sleeve", "short-sleeve", "shortsleeve", "반팔", "반소매"]
_LONG_MARK = ["long sleeve", "long-sleeve", "longsleeve", "긴팔", "긴소매",
              "hoodie", "후드", "coat", "코트"]


def _nearest_color_before(t, pos):
    best, bp = None, -1
    for alias, canon in _COLOR.items():
        p = t.rfind(alias, 0, pos)
        if p > bp:
            bp, best = p, canon
    return best


def parse_query(text):
    t = text.lower()
    if any(k in t for k in ("female", "woman", "women", "여자", "여성", "여")):
        g = 1
    elif any(k in t for k in ("male", "man", "men", "남자", "남성", "남")):
        g = 0
    else:
        g = None

    def fc(marks):
        ps = [t.find(m) for m in marks if m in t]
        if not ps:
            return None
        c = _nearest_color_before(t, min(ps))
        return COLORS.index(c) if c in COLORS else None

    sl = None
    if any(m in t for m in _SHORT_MARK):
        sl = 0
    elif any(m in t for m in _LONG_MARK):
        sl = 1
    return g, fc(_UP_MARK), fc(_DOWN_MARK), sl


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def wrap_text(text, max_w, scale, thick=1):
    lines, cur = [], ""
    for w in text.split():
        cand = w if not cur else cur + " " + w
        if cv2.getTextSize(cand, _FONT, scale, thick)[0][0] <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_lines(frame, lines, x, y, scale, color, lh=None):
    lh = lh or int(20 * scale / 0.5)
    for ln in lines:
        cv2.putText(frame, ln, (x, y), _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, ln, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += lh
    return y


def attr_dict(p):
    if p is None:
        return None
    return {"gender": GENDERS[int(p[0].argmax())],
            "upperColor": COLORS[int(p[1].argmax())],
            "lowerColor": COLORS[int(p[2].argmax())],
            "sleeve": sleeve_label(p[3])}


class FrameSender(threading.Thread):
    def __init__(self, ip, port, enabled=True):
        super().__init__(daemon=True)
        self.ip, self.port, self.sock = ip, port, None
        self.enabled = enabled
        self.slots = {}
        self.cv = threading.Condition()
        self._stopev = threading.Event()
        self.dropped = 0
        self.sent = 0

    def stop(self):
        self._stopev.set()
        with self.cv:
            self.cv.notify_all()

    def submit(self, cam, frame, matches):
        with self.cv:
            if cam in self.slots:
                self.dropped += 1
            self.slots[cam] = (frame, matches)
            self.cv.notify()

    def connect(self):
        if not self.enabled:
            print("[send] SEND_TO_PC=False -> PC 전송 없음", flush=True)
            return
        while not self._stopev.is_set():
            try:
                self.sock = socket.create_connection((self.ip, self.port), timeout=5.0)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[send] PC 연결됨 {self.ip}:{self.port}", flush=True)
                return
            except OSError as e:
                print(f"[send] PC 대기중 ({e}) 3초 후 재시도", flush=True)
                self._stopev.wait(3.0)

    def _send(self, cam, buf, matches):
        name = cam.encode("ascii", "replace")[:255]
        pkt = (struct.pack("B", len(name)) + name
               + struct.pack("B", min(255, matches))
               + struct.pack(">I", len(buf)) + buf)
        if self.sock is None:
            self.connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall(pkt)
        except OSError:
            print("[send] 끊김, 재접속", flush=True)
            self.sock = None
            self.connect()
            if self.sock is not None:
                try:
                    self.sock.sendall(pkt)
                except OSError:
                    self.sock = None

    def run(self):
        nframe = 0
        self.connect()
        while not self._stopev.is_set():
            with self.cv:
                while not self.slots and not self._stopev.is_set():
                    self.cv.wait(0.2)
                items = list(self.slots.items())
                self.slots.clear()
            for cam, (frame, matches) in items:
                if SEND_MAX_W and frame.shape[1] > SEND_MAX_W:
                    s = SEND_MAX_W / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s,
                                       interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue
                data = buf.tobytes()
                if self.enabled:
                    self._send(cam, data, matches)
                self.sent += 1
                nframe += 1
                if SAVE_LAST_DIR and nframe % SAVE_EVERY == 0:
                    path = os.path.join(SAVE_LAST_DIR, f"par_last_{cam}.jpg")
                    try:
                        with open(path + ".tmp", "wb") as f:
                            f.write(data)
                        os.replace(path + ".tmp", path)
                    except OSError:
                        pass


class CameraPool:
    def __init__(self):
        self.cams = {}
        self.state = {}

    def reconcile(self, wanted):
        wanted = list(wanted)
        for path in [p for p in self.cams if p not in wanted]:
            self.cams.pop(path).stop()
            self.state.pop(path, None)
            print(f"[cam] {path} 해제 (지시에서 빠짐)", flush=True)
        for path in wanted:
            if path in self.cams:
                continue
            c = CameraReceiver(scfg.rtsp_url(PI5_IP, RTSP_PORT, path), path)
            c.start()
            self.cams[path] = c
            self.state[path] = {"last": -1, "fps": 0.0, "tprev": time.time(),
                                "tracker": AttrTracker(PAR_CACHE_TTL,
                                                       PAR_CACHE_IOU)}
            print(f"[cam] {path} 추가", flush=True)

    def stop_all(self):
        for c in self.cams.values():
            c.stop()
        self.cams.clear()
        self.state.clear()

    def live(self):
        return sum(1 for c in self.cams.values() if c.connected)


def start_server_link():
    key = scfg.device_key()
    if not key:
        print(f"[link] ⚠ Device Key가 비어 있다. 서버가 401을 낼 것이다.\n"
              f"[link]   {scfg.DEVICE_KEY_FILE} 에 키를 한 줄로 넣을 것.\n"
              f"[link]   실행 중에 넣어도 재시작 없이 반영된다.", flush=True)
    else:
        print(f"[link] Device Key 로드됨 ({len(key)}자, ...{key[-4:]})", flush=True)

    sync = TargetSync(parse_query, LABELS)
    reporter = DetectionReporter(sync)
    sync.start()
    reporter.start()

    mq = None
    try:
        from mq_listener import MQListener
        mq = MQListener(sync)
        mq.start()
    except Exception as e:
        print(f"[mq] 시작 실패({type(e).__name__}) — 폴링만으로 동기화", flush=True)
    return sync, reporter, mq


def main():
    if CV_THREADS:
        cv2.setNumThreads(CV_THREADS)
    if SAVE_LAST_DIR:
        os.makedirs(SAVE_LAST_DIR, exist_ok=True)

    yolo, par = build_models()

    pool = CameraPool()
    print(f"[cam] 대기 — 서버가 지정한 카메라만 연다 "
          f"(매핑: {sorted(scfg.CAMERA_CODE_TO_PATH)})", flush=True)

    sender = FrameSender(PC_IP, VIEW_PORT, SEND_TO_PC)
    sender.start()

    sync, reporter, mq = start_server_link()

    def current_snapshot():
        return sync.current()

    nframe = par_calls = par_crops_total = persons_total = 0
    skipped_total = 0
    skip_reasons = {}
    max_batch = YOLO_BUCKETS[-1]

    if FULLBODY_ONLY:
        print(f"[gate] 전신만 캡처 — 경계여백 {FULLBODY_EDGE_MARGIN}px, "
              f"높이/너비 {FULLBODY_MIN_RATIO}~{FULLBODY_MAX_RATIO}, "
              f"최소높이 {FULLBODY_MIN_HEIGHT}px", flush=True)
    else:
        print("[gate] FULLBODY_ONLY=False — 잘린 사람도 캡처한다", flush=True)

    try:
        while True:
            now = time.time()
            snap = current_snapshot()
            wanted = list(ALWAYS_ON_CAMERAS)
            for p in snap.camera_paths():
                if p not in wanted:
                    wanted.append(p)
            pool.reconcile(wanted)
            if not pool.cams:
                time.sleep(0.2)
                continue

            batch = []
            for path, cam in list(pool.cams.items()):
                stt = pool.state[path]
                frame, count = cam.snapshot()
                if frame is None or count == stt["last"]:
                    continue
                stt["last"] = count
                batch.append((cam, frame.copy()))
                if len(batch) >= max_batch:
                    break
            if not batch:
                time.sleep(0.003)
                continue

            now_utc = utc_now()
            active = {cam.name_: snap.for_camera(cam.name_,
                                                 scfg.SEARCH_ALL_CAMERAS)
                      for cam, _f in batch}

            dets = yolo.detect([f for _, f in batch], conf=CONF_THRESHOLD,
                               iou=IOU_THRESHOLD, max_det=MAX_DET)

            work = []
            for (cam, frame), boxes in zip(batch, dets):
                boxes = [b for b in boxes if b[2] - b[0] >= 8 and b[3] - b[1] >= 16]
                tr = pool.state[cam.name_]["tracker"]
                tr.tick()
                probs = [None] * len(boxes)
                tids = [None] * len(boxes)
                todo = []
                conds = active[cam.name_]
                if conds:
                    for i, b in enumerate(boxes):
                        slot, p, tid = tr.take(b)
                        tids[i] = tid
                        if p is not None:
                            probs[i] = p
                        else:
                            todo.append((i, slot))
                    todo.sort(key=lambda t: -((boxes[t[0]][2] - boxes[t[0]][0]) *
                                              (boxes[t[0]][3] - boxes[t[0]][1])))
                work.append((cam, frame, boxes, probs, tr, todo, tids, conds))
                persons_total += len(boxes)

            sel = []
            depth = 0
            while len(sel) < PAR_MAX_CROPS:
                added = False
                for wi, w in enumerate(work):
                    todo = w[5]
                    if depth < len(todo) and len(sel) < PAR_MAX_CROPS:
                        sel.append((wi, todo[depth]))
                        added = True
                if not added:
                    break
                depth += 1
            if sel:
                crops = []
                for wi, (bi, _slot) in sel:
                    frame, boxes = work[wi][1], work[wi][2]
                    x0, y0, x1, y1 = boxes[bi]
                    crops.append(frame[y0:y1, x0:x1])
                gp, up, dp, sp = par.predict(crops)
                par_calls += 1
                par_crops_total += len(crops)
                for j, (wi, (bi, slot)) in enumerate(sel):
                    boxes, probs, tr = work[wi][2], work[wi][3], work[wi][4]
                    p = (gp[j], up[j], dp[j], sp[j])
                    probs[bi] = p
                    work[wi][6][bi] = tr.put(slot, boxes[bi], p)

            for cam, frame, boxes, probs, _tr, _todo, tids, conds in work:
                scores = [0.0] * len(boxes)
                per_case = {}
                fh_, fw_ = frame.shape[:2]
                gate = ([fullbody_reject(b, fw_, fh_) for b in boxes]
                        if FULLBODY_ONLY else [""] * len(boxes))
                for t in conds:
                    for i, p in enumerate(probs):
                        if p is None:
                            continue
                        s = score_probs(p, t.attrs)
                        if s > scores[i]:
                            scores[i] = s
                        if s >= MATCH_THRESHOLD and tids[i] is not None and not gate[i]:
                            d = per_case.setdefault(t.case_id, {})
                            if s > d.get(tids[i], (0.0, -1))[0]:
                                d[tids[i]] = (s, i)

                if (scfg.LOG_SEEN and conds
                        and nframe % scfg.LOG_SEEN_EVERY == 0):
                    for i, p in enumerate(probs):
                        if p is None:
                            continue
                        x0, y0, x1, y1 = boxes[i]
                        print(f"[seen] {cam.name_} {attr_text(p)} "
                              f"score={scores[i]:.3f} "
                              f"{'통과' if scores[i] >= MATCH_THRESHOLD else '미달'} "
                              f"gate={gate[i] or 'full'} "
                              f"{x1 - x0}x{y1 - y0}px", flush=True)

                order = sorted(range(len(boxes)), key=lambda i: -scores[i])
                matched = {i for i in order[:TOPK]
                           if probs[i] is not None and scores[i] >= MATCH_THRESHOLD}
                blocked = {i for i in matched if gate[i]}
                skipped_total += len(blocked)
                for i in blocked:
                    skip_reasons[gate[i]] = skip_reasons.get(gate[i], 0) + 1

                chosen = set()
                if per_case:
                    clean = frame.copy()
                    for case_id, bytrack in per_case.items():
                        ranked = sorted(bytrack.items(), key=lambda kv: -kv[1][0])
                        persons = []
                        for t, (sc, bi) in ranked[:TOPK]:
                            x0, y0, x1, y1 = boxes[bi]
                            m = scfg.CROP_MARGIN
                            if m > 0:
                                dx, dy = int((x1 - x0) * m), int((y1 - y0) * m)
                                cy0, cy1 = max(0, y0 - dy), min(fh_, y1 + dy)
                                cx0, cx1 = max(0, x0 - dx), min(fw_, x1 + dx)
                            else:
                                cy0, cy1, cx0, cx1 = y0, y1, x0, x1
                            persons.append({
                                "track_id": f"track-{t}",
                                "score": sc,
                                "box": boxes[bi],
                                "crop": clean[cy0:cy1, cx0:cx1].copy(),
                            })
                            chosen.add(bi)
                        reporter.offer(case_id, cam.name_, now_utc, persons,
                                       clean, now)
                    if scfg.LOG_MATCH_DETAIL and nframe % 30 == 0:
                        best = max(v[0] for d in per_case.values()
                                   for v in d.values())
                        print(f"[match] {cam.name_} case={list(per_case)} "
                              f"best={best:.3f}", flush=True)

                for i, (x0, y0, x1, y1) in enumerate(boxes):
                    hit = i in chosen
                    part = i in blocked
                    col = ((0, 0, 255) if hit else
                           (0, 165, 255) if part else (140, 140, 140))
                    cv2.rectangle(frame, (x0, y0), (x1, y1), col,
                                  3 if hit else 2 if part else 1)
                    if probs[i] is not None:
                        tag = (f"MATCH {scores[i]:.2f} " if hit else
                               f"SKIP({gate[i]}) {scores[i]:.2f} " if part else
                               f"{scores[i]:.2f} ")
                        cv2.putText(frame, tag + attr_text(probs[i]),
                                    (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, col, 2 if hit else 1, cv2.LINE_AA)

                t = time.time()
                stt = pool.state[cam.name_]
                stt["fps"] = (0.9 * stt["fps"]
                              + 0.1 / max(1e-6, t - stt["tprev"]))
                stt["tprev"] = t
                cv2.putText(frame, f"[{cam.name_}] {len(chosen)} match "
                                   f"{stt['fps']:.1f} fps", (10, 24),
                            _FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"[{cam.name_}] {len(chosen)} match "
                                   f"{stt['fps']:.1f} fps", (10, 24),
                            _FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                maxw = frame.shape[1] - 20
                y = 48
                if conds:
                    for c in conds:
                        head = f"SEARCH case {c.case_id}"
                        if c.case_number:
                            head += f" ({c.case_number})"
                        y = draw_lines(frame, [head], 10, y, 0.5, (0, 255, 255))
                        y = draw_lines(frame, wrap_text(c.prompt, maxw, 0.55),
                                       10, y, 0.55, (0, 255, 255))
                        y += 4
                elif snap.targets:
                    draw_lines(frame, ["NO CASE for this camera (monitor only)"],
                               10, y, 0.5, (160, 160, 160))
                else:
                    draw_lines(frame, ["NO ACTIVE CASE (monitor only)"],
                               10, y, 0.5, (160, 160, 160))
                sender.submit(cam.name_, frame, len(chosen))

            nframe += len(work)
            if nframe >= STAT_EVERY:
                live = pool.live()
                per = " ".join(f"{k.replace('camera-', '')}:{s['fps']:.1f}"
                               for k, s in pool.state.items() if s["fps"] > 0.05)
                hit = 1.0 - (par_crops_total / max(1, persons_total))
                back = ""
                if sync is not None:
                    cur = sync.current()
                    back = (f" | target={len(cur.targets)}"
                            f" mq={'on' if (mq and mq.connected) else 'off'}"
                            f" sync={sync.ok}+{sync.not_modified}/{sync.failed}"
                            + (f"({sync.last_error})" if sync.failed else "")
                            + f" ev={reporter.sent}/{reporter.failed}"
                            + (f" 흐림{reporter.blurry_skipped}"
                               if reporter.blurry_skipped else "")
                            + (f"+drop{reporter.dropped}" if reporter.dropped else "")
                            + (f" 보관{reporter.archived}" if reporter.archived else ""))
                skip = ""
                if skipped_total:
                    why = " ".join(f"{k}{v}" for k, v in
                                   sorted(skip_reasons.items(), key=lambda kv: -kv[1]))
                    skip = f" | skip={skipped_total}({why})"
                print(f"[run] fps {per} | live={live} "
                      f"persons/f={persons_total / max(1, nframe):.1f} "
                      f"PAR캐시적중={hit * 100:.0f}% ({par_crops_total}crop/"
                      f"{par_calls}call) | send={sender.sent} drop={sender.dropped}"
                      f"{skip}{back}", flush=True)
                nframe = par_calls = par_crops_total = persons_total = 0
                skipped_total = 0
                skip_reasons.clear()
    except KeyboardInterrupt:
        pass
    finally:
        pool.stop_all()
        sender.stop()
        for t in (mq, sync, reporter):
            if t is not None:
                t.stop()
        if reporter is not None:
            print(f"[report] 등록 {reporter.sent}건 실패 {reporter.failed}건 "
                  f"버림 {reporter.dropped}건 보관 {reporter.archived}건 "
                  f"({reporter.bytes_sent / 1024 / 1024:.1f}MB, "
                  f"중복억제 {reporter.suppressed}프레임)", flush=True)
        print("stopped", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
