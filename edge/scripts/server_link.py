from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import cv2
import requests

import server_config as cfg

_RETRY_STATUS = (429, 500, 502, 503, 504)


def utc_now():
    return datetime.now(timezone.utc)


def rfc3339(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _headers(json_body=False):
    h = {"Accept": "application/json"}
    key = cfg.device_key()
    if key:
        h[cfg.DEVICE_KEY_HEADER] = key
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _url(path):
    return cfg.API_BASE.rstrip("/") + path


def code_to_path(code):
    p = cfg.CAMERA_CODE_TO_PATH.get(code)
    if p:
        return p
    return code if (cfg.CAMERA_CODE_PASSTHROUGH and code) else None


def path_to_code(path):
    c = cfg.PATH_TO_CAMERA_CODE.get(path)
    if c:
        return c
    return path if cfg.CAMERA_CODE_PASSTHROUGH else None


class Target:
    __slots__ = ("case_id", "case_number", "condition_id", "prompt", "attrs",
                 "camera_paths")

    def __init__(self, case_id, case_number, condition_id, prompt, attrs,
                 camera_paths):
        self.case_id = case_id
        self.case_number = case_number
        self.condition_id = condition_id
        self.prompt = prompt or ""
        self.attrs = attrs or (None, None, None, None)
        self.camera_paths = set(camera_paths)

    def has_attrs(self):
        return any(v is not None for v in self.attrs)

    def wants(self, camera_path):
        return self.has_attrs() and camera_path in self.camera_paths


class Snapshot:
    __slots__ = ("targets", "etag", "source")

    def __init__(self, targets=(), etag=None, source="server"):
        self.targets = list(targets)
        self.etag = etag
        self.source = source

    def for_camera(self, camera_path, all_cameras=False):
        if all_cameras:
            return [t for t in self.targets if t.has_attrs()]
        return [t for t in self.targets if t.wants(camera_path)]

    def camera_paths(self):
        out = []
        for t in self.targets:
            for p in t.camera_paths:
                if p not in out:
                    out.append(p)
        return out

    def key(self):
        return tuple(sorted((t.case_id, t.condition_id, t.prompt,
                             tuple(sorted(t.camera_paths))) for t in self.targets))

    def describe(self, labels=None):
        if not self.targets:
            return "검색 대상 없음 (분석 중지)"
        lines = []
        for t in self.targets:
            cams = ",".join(sorted(t.camera_paths)) or "카메라 없음"
            line = (f"  case {t.case_id}({t.case_number or '-'}) "
                    f"cond {t.condition_id} [{cams}] '{t.prompt}'")
            if labels:
                genders, colors, sleeves = labels
                g, u, d, s = t.attrs
                f = lambda a, i: a[i] if i is not None else "-"
                line += (f"\n      -> gender={f(genders, g)} upper={f(colors, u)}"
                         f" lower={f(colors, d)} sleeve={f(sleeves, s)}")
            if not t.has_attrs():
                line += "  ⚠ 인식 가능한 속성 없음(아무도 매칭 안 됨)"
            lines.append(line)
        return f"검색 대상 {len(self.targets)}건\n" + "\n".join(lines)


EMPTY = Snapshot()


class TargetSync(threading.Thread):
    def __init__(self, parse_query, labels=None):
        super().__init__(daemon=True)
        self.parse_query = parse_query
        self.labels = labels
        self._lock = threading.Lock()
        self._snap = EMPTY
        self._etag = None
        self._stopev = threading.Event()
        self._wake = threading.Event()
        self.ok = 0
        self.not_modified = 0
        self.failed = 0
        self.last_error = ""
        self.auth_fail = 0
        self.ever_ok = False

    def current(self):
        with self._lock:
            return self._snap

    def refresh_now(self):
        self._wake.set()

    def refresh_and_wait(self, timeout):
        before = self.ok + self.not_modified
        self._wake.set()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ok + self.not_modified > before:
                return True
            if self._stopev.is_set():
                return False
            time.sleep(0.1)
        return False

    def stop(self):
        self._stopev.set()
        self._wake.set()

    def _apply(self, snap):
        with self._lock:
            old = self._snap
            self._snap = snap
        if old.key() != snap.key():
            src = "" if snap.source == "server" else f" (source={snap.source})"
            print(f"[target] {snap.describe(self.labels)}{src}", flush=True)

    def _build(self, payload, etag):
        data = payload.get("data") if isinstance(payload, dict) else payload
        targets = []
        unknown = set()
        for case in data or []:
            case_id = case.get("caseId")
            paths = []
            for c in case.get("cameras") or []:
                code = c.get("cameraCode") if isinstance(c, dict) else c
                p = code_to_path(code)
                if p:
                    paths.append(p)
                elif code:
                    unknown.add(code)
            for cond in case.get("searchConditions") or []:
                prompt = (cond.get("prompt") or "").strip()
                targets.append(Target(
                    case_id=case_id, case_number=case.get("caseNumber"),
                    condition_id=cond.get("conditionId"), prompt=prompt,
                    attrs=self.parse_query(prompt) if prompt else None,
                    camera_paths=paths))
        if unknown:
            print(f"[target] ⚠ 모르는 cameraCode {sorted(unknown)} — 무시 "
                  f"(매핑: {sorted(cfg.CAMERA_CODE_TO_PATH)})", flush=True)
        return Snapshot(targets, etag, "server")

    def run(self):
        url = _url(cfg.SEARCH_TARGETS_PATH)
        bi = 0
        print(f"[target] 동기화 시작 {url} ({cfg.POLL_INTERVAL:.0f}초 주기, ETag)",
              flush=True)
        while not self._stopev.is_set():
            delay = cfg.POLL_INTERVAL
            try:
                h = _headers()
                if self._etag:
                    h["If-None-Match"] = self._etag
                r = requests.get(url, headers=h, timeout=cfg.API_TIMEOUT)
                if r.status_code == 200:
                    self.ok += 1
                    self.ever_ok = True
                    self.auth_fail = 0
                    bi = 0
                    self._etag = r.headers.get("ETag")
                    self._apply(self._build(r.json(), self._etag))
                elif r.status_code == 304:
                    self.not_modified += 1
                    self.ever_ok = True
                    bi = 0
                elif r.status_code in (401, 403):
                    self.auth_fail += 1
                    self.failed += 1
                    self.last_error = f"{r.status_code}"
                    if self.auth_fail == 1:
                        print(f"[target] ⚠ {r.status_code} 인증 실패 — Device Key 확인"
                              f" ({cfg.DEVICE_KEY_FILE})", flush=True)
                    if self.auth_fail >= cfg.POLL_MAX_AUTH_FAIL:
                        delay = 30.0
                else:
                    self.failed += 1
                    self.last_error = f"HTTP {r.status_code}"
                    if self.failed == 1 or self.failed % 12 == 0:
                        print(f"[target] ⚠ 서버 오류 HTTP {r.status_code} "
                              f"(누적 {self.failed}회) — 재시도", flush=True)
                    delay = cfg.POLL_BACKOFF[min(bi, len(cfg.POLL_BACKOFF) - 1)]
                    bi += 1
            except Exception as e:
                self.failed += 1
                self.last_error = type(e).__name__
                if self.failed == 1 or self.failed % 20 == 0:
                    print(f"[target] 서버 응답 없음 ({self.last_error}) — 재시도",
                          flush=True)
                delay = cfg.POLL_BACKOFF[min(bi, len(cfg.POLL_BACKOFF) - 1)]
                bi += 1
            self._wake.wait(delay)
            self._wake.clear()


class DetectionReporter(threading.Thread):
    def __init__(self, sync=None):
        super().__init__(daemon=True)
        self.sync = sync
        self._q = queue.Queue(maxsize=cfg.QUEUE_MAX)
        self._stopev = threading.Event()
        self._lock = threading.Lock()
        self._tracks = {}
        self._last_send = {}
        self._seq = self._load_seq()
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.suppressed = 0
        self.archived = 0
        self.blurry_skipped = 0
        self.bytes_sent = 0

    def _load_seq(self):
        try:
            with open(cfg.SEQ_FILE, encoding="utf-8") as f:
                return int(json.load(f).get("seq", 0))
        except Exception:
            return 0

    def _save_seq(self):
        try:
            os.makedirs(os.path.dirname(cfg.SEQ_FILE), exist_ok=True)
            tmp = cfg.SEQ_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"seq": self._seq}, f)
            os.replace(tmp, cfg.SEQ_FILE)
        except OSError:
            pass

    def _next_event_id(self, camera_code, dt):
        self._seq += 1
        self._save_seq()
        return f"{camera_code}-{dt.strftime('%Y%m%d')}-{self._seq:06d}"

    @staticmethod
    def sharpness(crop):
        if crop is None or crop.size == 0:
            return 0.0
        h, w = crop.shape[:2]
        m = max(h, w)
        if m > 160:
            s = 160.0 / m
            crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))),
                              interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    @staticmethod
    def _quality(score, box, frame_h, sharp=None):
        h = max(1, box[3] - box[1])
        size = min(1.0, h / max(1.0, frame_h * cfg.BEST_REF_HEIGHT_RATIO))
        sw = cfg.BEST_SHARP_WEIGHT if sharp is not None else 0.0
        base = 1.0 - cfg.BEST_SIZE_WEIGHT - sw
        q = float(score) * max(0.0, base) + size * cfg.BEST_SIZE_WEIGHT
        if sharp is not None:
            q += min(1.0, sharp / max(1.0, cfg.BEST_SHARP_REF)) * sw
        return q

    def offer(self, case_id, camera_path, detected_at, persons, frame, now):
        frame_h = frame.shape[0] if frame is not None else 1
        with self._lock:
            self._forget(now)
            for p in persons:
                key = (case_id, camera_path, p["track_id"])
                st = self._tracks.get(key)
                if st is not None:
                    st["seen"] = now
                sharp = self.sharpness(p.get("crop"))
                if cfg.MIN_SHARPNESS > 0 and sharp < cfg.MIN_SHARPNESS:
                    self.blurry_skipped += 1
                    continue
                q = self._quality(p["score"], p["box"], frame_h, sharp)
                if st is None:
                    self._tracks[key] = {"best_q": q, "best": p, "frame": frame,
                                         "opened": now, "seen": now,
                                         "sent_at": None,
                                         "detected_at": detected_at}
                    continue
                if st["sent_at"] is not None:
                    if cfg.REFRESH_SEC > 0 and now - st["sent_at"] >= cfg.REFRESH_SEC:
                        st.update(best_q=q, best=p, frame=frame, opened=now,
                                  sent_at=None, detected_at=detected_at)
                    else:
                        self.suppressed += 1
                    continue
                self.suppressed += 1
                if q > st["best_q"]:
                    st.update(best_q=q, best=p, frame=frame,
                              detected_at=detected_at)
            self._flush_ready(now)

    def _forget(self, now):
        for k in [k for k, s in self._tracks.items()
                  if now - s["seen"] > cfg.TRACK_FORGET_SEC]:
            del self._tracks[k]

    def _flush_ready(self, now):
        for (case_id, cam_path, _track), st in self._tracks.items():
            if st["sent_at"] is not None:
                continue
            if now - st["opened"] < cfg.BEST_WINDOW_SEC:
                continue
            if now - self._last_send.get(cam_path, 0.0) < cfg.MIN_SEND_GAP_SEC:
                continue
            self._last_send[cam_path] = now
            st["sent_at"] = now
            item = (case_id, cam_path, st["detected_at"], [st["best"]], st["frame"])
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped += 1
                if self.dropped in (1, 10) or self.dropped % 50 == 0:
                    print(f"[report] ⚠ 큐 가득 — {self.dropped}건 버림 "
                          f"(sent={self.sent} failed={self.failed})", flush=True)
            st["frame"] = None
            st["best"] = dict(st["best"], crop=None)

    def stop(self):
        self._stopev.set()
        self._save_seq()

    @staticmethod
    def _jpeg(img):
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), cfg.JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        return buf.tobytes()

    def _request_upload_urls(self, case_id, camera_code, event_id, persons):
        body = {
            "caseId": case_id,
            "cameraCode": camera_code,
            "eventId": event_id,
            "frame": {"contentType": cfg.CONTENT_TYPE},
            "detections": [{"trackId": p["track_id"],
                            "contentType": cfg.CONTENT_TYPE} for p in persons],
        }
        r = requests.post(_url(cfg.UPLOAD_URLS_PATH), headers=_headers(True),
                          data=json.dumps(body), timeout=cfg.API_TIMEOUT)
        if r.status_code not in (200, 201):
            return None, f"upload-urls HTTP {r.status_code} {r.text[:160]}", r.status_code
        data = r.json()
        return (data.get("data") or data), None, r.status_code

    def _put_image(self, upload_url, content_type, blob):
        r = requests.put(upload_url, data=blob,
                         headers={"Content-Type": content_type},
                         timeout=cfg.UPLOAD_TIMEOUT)
        if r.status_code not in (200, 201, 204):
            return f"PUT HTTP {r.status_code} {r.text[:120]}"
        return None

    def _register(self, case_id, camera_code, event_id, detected_at,
                  frame_key, persons, crop_keys):
        dets = []
        for p in persons:
            x0, y0, x1, y1 = p["box"]
            w, h = max(1, int(x1 - x0)), max(1, int(y1 - y0))
            dets.append({
                "trackId": p["track_id"],
                "similarity": round(min(1.0, max(0.0, float(p["score"]))), 4),
                "cropObjectKey": crop_keys[p["track_id"]],
                "boundingBox": {"x": max(0, int(x0)), "y": max(0, int(y0)),
                                "width": w, "height": h},
            })
        body = {"caseId": case_id, "cameraCode": camera_code, "eventId": event_id,
                "detectedAt": rfc3339(detected_at), "frameObjectKey": frame_key,
                "detections": dets}
        r = requests.post(_url(cfg.CANDIDATE_EVENTS_PATH), headers=_headers(True),
                          data=json.dumps(body), timeout=cfg.API_TIMEOUT)
        return r

    def _send_once(self, case_id, cam_path, detected_at, persons, frame,
                   event_id):
        camera_code = path_to_code(cam_path)
        if not camera_code:
            return False, False, f"cameraCode 매핑 없음({cam_path})"

        urls, err, status = self._request_upload_urls(case_id, camera_code,
                                                      event_id, persons)
        if err:
            return False, status in _RETRY_STATUS, err

        frame_up = urls.get("frame") or {}
        by_track = {d.get("trackId"): d for d in (urls.get("detections") or [])}

        blob = self._jpeg(frame)
        e = self._put_image(frame_up.get("uploadUrl"),
                            frame_up.get("contentType", cfg.CONTENT_TYPE), blob)
        if e:
            return False, True, f"frame {e}"
        total = len(blob)
        crop_keys = {}
        for p in persons:
            up = by_track.get(p["track_id"])
            if not up:
                return False, False, f"trackId {p['track_id']} 업로드 URL 없음"
            cblob = self._jpeg(p["crop"])
            e = self._put_image(up.get("uploadUrl"),
                                up.get("contentType", cfg.CONTENT_TYPE), cblob)
            if e:
                return False, True, f"crop {e}"
            total += len(cblob)
            crop_keys[p["track_id"]] = up.get("objectKey")

        r = self._register(case_id, camera_code, event_id, detected_at,
                           frame_up.get("objectKey"), persons, crop_keys)
        if r.status_code in (200, 201):
            self.bytes_sent += total
            return True, False, f"HTTP {r.status_code}"
        if r.status_code in (404, 422) and self.sync is not None:
            self.sync.refresh_now()
            return False, False, f"HTTP {r.status_code} 재동기화 요청"
        return False, r.status_code in _RETRY_STATUS, f"candidate-events HTTP {r.status_code} {r.text[:160]}"

    def _archive(self, case_id, cam_path, event_id, detected_at, persons,
                 frame, reason):
        if cfg.SAVE_DETECTIONS == "off":
            return
        try:
            base = os.path.join(cfg.SAVE_DETECTIONS_DIR, str(event_id))
            os.makedirs(base, exist_ok=True)
            meta = {"caseId": case_id, "cameraCode": path_to_code(cam_path),
                    "cameraPath": cam_path, "eventId": event_id,
                    "detectedAt": rfc3339(detected_at), "reason": reason,
                    "detections": [{"trackId": p["track_id"],
                                    "similarity": round(float(p["score"]), 4),
                                    "box": list(p["box"])} for p in persons]}
            with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            if frame is not None:
                with open(os.path.join(base, "frame.jpg"), "wb") as f:
                    f.write(self._jpeg(frame))
            for i, p in enumerate(persons):
                if p.get("crop") is not None:
                    with open(os.path.join(base, f"crop{i}.jpg"), "wb") as f:
                        f.write(self._jpeg(p["crop"]))
            self.archived += 1
            self._prune()
        except Exception:
            pass

    @staticmethod
    def _prune():
        try:
            root = cfg.SAVE_DETECTIONS_DIR
            dirs = [os.path.join(root, d) for d in os.listdir(root)]
            dirs = [d for d in dirs if os.path.isdir(d)]
            if len(dirs) <= cfg.SAVE_MAX_FILES:
                return
            dirs.sort(key=os.path.getmtime)
            for d in dirs[:len(dirs) - cfg.SAVE_MAX_FILES]:
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)
        except OSError:
            pass

    def run(self):
        if not cfg.REPORT_ENABLED:
            print("[report] REPORT_ENABLED=False — 서버 전송 없음", flush=True)
            return
        print(f"[report] 전송 준비 {cfg.API_BASE} "
              f"(presigned 업로드 -> candidate-events, "
              f"트랙당 1건 / 최고사진 선별 {cfg.BEST_WINDOW_SEC:.1f}초)", flush=True)
        while not self._stopev.is_set():
            with self._lock:
                self._flush_ready(time.time())
            try:
                item = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                case_id, cam_path, detected_at, persons, frame = item
                code = path_to_code(cam_path) or "CAM"
                event_id = self._next_event_id(code, detected_at)
                ok = False
                msg = ""
                for i, delay in enumerate((0.0,) + cfg.RETRY_DELAYS):
                    if delay and self._stopev.wait(delay):
                        break
                    ok, retry, msg = self._send_once(case_id, cam_path,
                                                     detected_at, persons,
                                                     frame, event_id)
                    if ok or not retry:
                        break
                if ok:
                    self.sent += 1
                    if cfg.LOG_MATCH_DETAIL:
                        print(f"[report] 등록 완료 {event_id} case={case_id} "
                              f"{code} {len(persons)}명", flush=True)
                    if cfg.SAVE_DETECTIONS == "always":
                        self._archive(case_id, cam_path, event_id, detected_at,
                                      persons, frame, "sent")
                else:
                    self.failed += 1
                    print(f"[report] 전송 실패 {event_id}: {msg}", flush=True)
                    if cfg.SAVE_DETECTIONS in ("onfail", "always"):
                        self._archive(case_id, cam_path, event_id, detected_at,
                                      persons, frame, msg)
            except Exception as e:
                self.failed += 1
                try:
                    print(f"[report] 처리 실패({type(e).__name__}) - 건너뜀",
                          flush=True)
                except Exception:
                    pass
        self._save_seq()
