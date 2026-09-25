import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_env_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k.startswith("export "):
                    k = k[7:].strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except OSError:
        pass


_load_env_file(os.path.join(ROOT, ".env"))


def _env(key, default):
    v = os.environ.get(key, "").strip()
    return v if v else default


API_BASE = _env("YOPAR_API_BASE", "")

PING_PATH = "/api/v1/device/media-server/ping"
SEARCH_TARGETS_PATH = "/api/v1/device/search-targets"
UPLOAD_URLS_PATH = "/api/v1/device/candidate-event-upload-urls"
CANDIDATE_EVENTS_PATH = "/api/v1/device/candidate-events"

API_TIMEOUT = 8.0
UPLOAD_TIMEOUT = 20.0

DEVICE_KEY_FILE = os.path.join(ROOT, "devicekey.txt")
DEVICE_KEY_HEADER = "X-Device-Key"

_KEY_CACHE = {"mtime": None, "value": ""}


def device_key():
    env = os.environ.get("YOPAR_DEVICE_KEY", "").strip()
    if env:
        return env
    try:
        mt = os.path.getmtime(DEVICE_KEY_FILE)
    except OSError:
        return _KEY_CACHE["value"]
    if _KEY_CACHE["mtime"] != mt:
        try:
            raw = open(DEVICE_KEY_FILE, encoding="utf-8").read()
        except OSError:
            return _KEY_CACHE["value"]
        _KEY_CACHE["mtime"] = mt
        val = ""
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for sep in ("=", ":"):
                if sep in s:
                    head, _, tail = s.partition(sep)
                    if head.strip().lower().replace("-", "").replace("_", "") in (
                            "devicekey", "xdevicekey", "key"):
                        s = tail.strip()
                    break
            val = s.strip().strip('"').strip("'")
            break
        _KEY_CACHE["value"] = val
    return _KEY_CACHE["value"]


POLL_INTERVAL = 5.0
POLL_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)
POLL_MAX_AUTH_FAIL = 5


RTSP_USER = _env("YOPAR_RTSP_USER", "")
RTSP_PASS = _env("YOPAR_RTSP_PASS", "")


def rtsp_url(host, port, path):
    from urllib.parse import quote
    if RTSP_USER:
        cred = quote(RTSP_USER, safe="") + ":" + quote(RTSP_PASS, safe="") + "@"
    else:
        cred = ""
    return f"rtsp://{cred}{host}:{port}/{path}"


MQ_ENABLED = True
MQ_HOST = _env("YOPAR_MQ_HOST", "")
MQ_PORT = int(_env("YOPAR_MQ_PORT", "5672"))
MQ_USER = _env("YOPAR_MQ_USER", "")
MQ_PASS = _env("YOPAR_MQ_PASS", "")
MQ_VHOST = _env("YOPAR_MQ_VHOST", "/")

MQ_QUEUE = "search.target.realtime.queue"
MQ_EXCHANGE = "search.target.exchange"
MQ_ROUTING_KEY = "search.target.updated"

MQ_RECONNECT_DELAYS = (2.0, 4.0, 8.0, 15.0, 30.0)
MQ_COMMAND_MEMORY = 512
MQ_SYNC_TIMEOUT = 15.0

SEARCH_ALL_CAMERAS = False

CAMERA_CODE_TO_PATH = {
}
PATH_TO_CAMERA_CODE = {v: k for k, v in CAMERA_CODE_TO_PATH.items()}

CAMERA_CODE_PASSTHROUGH = True

REPORT_ENABLED = True
JPEG_QUALITY = 85
CONTENT_TYPE = "image/jpeg"

QUEUE_MAX = 16
RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
SEQ_FILE = os.path.join(ROOT, "runtime", "event_seq.json")

BEST_WINDOW_SEC = 2.0
BEST_SIZE_WEIGHT = 0.2
BEST_REF_HEIGHT_RATIO = 0.5

BEST_SHARP_WEIGHT = 0.35
BEST_SHARP_REF = 900.0

MIN_SHARPNESS = 300.0

CROP_MARGIN = 0.06
REFRESH_SEC = 0.0
TRACK_FORGET_SEC = 30.0
MIN_SEND_GAP_SEC = 1.0

SAVE_DETECTIONS = "onfail"
SAVE_DETECTIONS_DIR = os.path.join(ROOT, "output", "detections")
SAVE_MAX_FILES = 300

LOG_MATCH_DETAIL = True

LOG_SEEN = False
LOG_SEEN_EVERY = 45
