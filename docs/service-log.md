# Service Log

What the YOPAR real-time service does to run four cameras on a Jetson Orin
Nano, with the reason and, where one was recorded, the measurement behind each
measure; followed by the attribute model it deploys. The
[README](../README.md) gives the usage and a summary.

## Conventions

- **Settings** are the constants in the code (`edge/scripts/`, `train/`),
  given with the file where they are defined.
- **Recorded measurements** are numbers written by the authors in the code
  comments of the service before it was merged into this repository
  (`yopar-edge-service`, last commit `32d0cae`); those comments are no longer
  in the code, and this log is where the numbers are kept. They were not
  measured again. Each one is marked *(recorded)*.
- **Model results** come from the files under `results/`;
  `tools/summarize.py` writes `results/summary/tables.md`, and the tables in
  the last section are copied from it unchanged. Values from training logs
  are shown as printed (3 decimals), other values with 4 decimals.

## Target and pipeline

- **Device.** Jetson Orin Nano; the CPU and the GPU share 7.4 GB of RAM
  *(recorded)*.
- **Input.** Four RTSP streams (`camera-01` … `camera-04`) from a Raspberry Pi 5
  media server on port 8554 (`PI5_IP`, `RTSP_PORT` in `capture_core.py`).
- **Per loop.** The newest frame of each camera → one batched YOLO11s person
  detection → attributes (gender, upper color, lower color, sleeve) for the
  person crops → match against the search targets of the central server →
  evidence upload and candidate event for matched persons
  (`jetson_par_sender.py`, `server_link.py`).

## 1. Inference runtime

- **No PyTorch on the device.** YOLO and PAR run as ONNX models on onnxruntime;
  YOLO pre- and post-processing (letterbox, decoding of the person channel of
  the `(B, 84, 8400)` output, NMS with `cv2.dnn.NMSBoxes`) is written with
  NumPy and OpenCV (`onnx_yolo.py`). Reason: not loading PyTorch saves one CUDA
  context, about 860 MB *(recorded)*, which on the shared 7.4 GB is the largest
  saving *(recorded)*.
- **Providers.** `trt` = TensorRT (fp16) → CUDA → CPU, `cuda` = CUDA (fp32) →
  CPU (`_PROV` in `capture_core.py`). A session is created with the first
  available provider and falls back to the next one on failure. The choice is
  `FORCE_PROVIDER` if set, else the verdict of `prepare.py`
  (`models/provider_verdict.json`), else `trt` for both models.
- **TensorRT engine cache.** fp16 engines are cached in `edge/models/trt_cache/`
  (a TensorRT timing cache is enabled as well); the first start builds one engine per batch
  size and takes a few minutes, later starts load them *(recorded)*. The engines
  are compiled for the GPU they were built on and cannot be copied to another
  device.
- **Memory.** CUDA arena `kSameAsRequested`, no maximum cuDNN workspace, CPU
  memory arena off, optional GPU memory limit `PAR_GPU_MEM_MB` (0 = none;
  320 is the value suggested by the CPU warning). Reason: with shared RAM the
  default arena can make CUBLAS allocations fail *(recorded)*.
- **Warm-up.** Every batch size of both models is run once at start
  (`warmup`), so the first frames do not pay the engine build or load.

## 2. TensorRT fp16 check (`scripts/prepare.py`)

Run once per device before the service; `--check` skips the build.

1. **RAM guard.** If the available RAM (`free -m`) is below 3000 MB it warns and
   asks; without a terminal it stops. Reason: with too little memory TensorRT
   skips tactics and a slower engine is stored in the cache *(recorded)*.
2. **Build.** Both models, all batch sizes.
3. **YOLO check.** TensorRT fp16 and CUDA fp32 on the sample image
   `edge/samples/bus.jpg`, for which `prepare.py` prints 4 persons as the
   expected count: fp16 is accepted when both
   find the same number of persons and every box has IoU ≥ 0.90 with its
   closest counterpart.
4. **PAR check.** On up to 8 person crops: fp16 is accepted when no head
   changes its top label and no match score crosses the matching threshold
   (0.25) for the query male / white upper / black lower / short sleeve.
5. **Speed.** Each batch size, 20 calls, TensorRT and CUDA; printed and written
   to `output/prepare_report.txt`. No result of this benchmark is kept in the
   repository.
6. **Verdict.** `models/provider_verdict.json` holds `trt` or `cuda` per model;
   the service reads it at start.

`edge/samples/bus.jpg` is not in the repository; the script stops when it is
missing.

## 3. Batching and pre-processing

- **One YOLO call for all cameras.** Frames of the cameras with a new frame are
  stacked into one batch (at most 4), so the GPU is called once per loop
  instead of once per camera.
- **Fixed batch sizes.** YOLO uses batch sizes 1, 2, 4 and PAR 1, 2, 4, 8; a
  batch is padded to the next size and the padded outputs are dropped. Reason:
  TensorRT builds a separate engine per batch size; fixed sizes also avoid
  re-allocation on the CUDA provider *(recorded)*.
- **Reused buffers.** Input arrays are allocated once per batch size (zeros,
  so padding never holds inf/NaN); the letterbox canvas is reused.
- **No extra copies.** BGR pixels are written straight into the CHW input (no
  `cvtColor` or `transpose` copy); the PAR normalization `(x/255 − mean)/std` is
  folded into one multiply and one add, done in place.
- **Detection.** Input 640 (512 is faster but misses distant persons
  *(recorded)*), confidence 0.35, NMS IoU 0.5, at most 20 detections; boxes
  narrower than 8 or lower than 16 pixels are dropped before PAR.

## 4. Attribute cache and PAR budget (`AttrTracker`)

- Per camera, each detected box is matched to a remembered box with IoU above
  0.5 (`PAR_CACHE_IOU`); if the remembered attributes are at most 12 frames old
  (`PAR_CACHE_TTL`) they are reused instead of running PAR; older attributes
  are computed again. An entry is dropped 36 frames (3 × TTL) after its
  attributes were last computed. Reason: clothing does not change between
  frames, and re-running after the TTL corrects detection jitter. The comment
  in `capture_core.py` calls this the largest optimization in that file and
  expects PAR calls to drop to about 1/TTL while a person stays in view
  *(recorded; an expectation, not a measurement)*.
- **Budget.** At most 8 crops per PAR call over all cameras (`PAR_MAX_CROPS`, a
  latency cap); larger boxes (closer persons) first, cameras taken in turn so
  that one camera cannot use the whole budget; only cache misses are sent.
- PAR is not run for a camera that has no search target assigned.
- The statistics line prints the cache hit rate (1 − crops / persons).

## 5. Full-body gate (`fullbody_reject`)

Only persons whose whole body is in the picture become candidates. Reason: a
searcher cannot check the clothing of a half-visible person, and the lower
color of such a crop is not reliable *(recorded)*.

| check | setting | reason / measurement |
| --- | --- | --- |
| box within 6 px of the frame border → `cut` | `FULLBODY_EDGE_MARGIN = 6` | most frequent case: the legs are cut when a person passes close to the camera *(recorded)* |
| height < 120 px → `small` | `FULLBODY_MIN_HEIGHT = 120` | the face is not resolved below this *(recorded)* |
| width < 70 px → `sliver` | `FULLBODY_MIN_WIDTH = 70` | all 79 labelled full-body crops pass; their smallest width is 71 px *(recorded)* |
| height/width < 1.8 → `upper` | `FULLBODY_MIN_RATIO = 1.8` | an upper-body-only box has a ratio of 1.0–1.5 *(recorded)* |
| height/width > 4.5 → `thin` | `FULLBODY_MAX_RATIO = 4.5` | lowered from 5.5, at which vertical strips occluded by walls or door frames passed (a forearm-only crop was judged full-body); 4.5 drops 2 of the 79 crops *(recorded)* |

The 79 crops were checked by eye as full-body; their ranges are height/width
1.99–5.49, width 71–282 px, height 310–632 px *(recorded)*. The gate uses box
geometry only; a person occluded inside the frame by a pillar, a car or another
person can still pass *(recorded)*. Among the top 3 persons at or above the
matching threshold, those rejected by the gate are drawn with their reason and
counted per reason instead of being dropped silently.

## 6. Sleeve threshold

`sleeve` is reported as `long` only when its probability is at least 0.95
(`SLEEVE_LONG_MIN`), instead of the larger of the two probabilities. Reason: the
model over-estimates long sleeves; with the larger probability 32 short sleeves
were read as long against 3 long as short, and in a track-level 5-fold
cross-validation the accuracy went from 84.9% to 90.7% with the threshold
*(recorded)*. The threshold is used for the attribute text shown on the
screen; the match score uses the probabilities and is not affected, and a
recalibration of the probabilities was not validated *(recorded)*.

## 7. Matching

- **Score.** Product of the probabilities of the attributes named in the search
  target (`score_probs`).
- **Threshold 0.25** (`MATCH_THRESHOLD`), chosen on the 79 labelled crops with
  22 real clothing combinations as queries, from a precision/recall curve
  *(recorded)*:

| threshold | precision | recall | F1 |
| --- | --- | --- | --- |
| 0.15 | 51.9% | 65.0% | 0.577 |
| **0.25** (used) | 56.9% | 62.4% | 0.596 |

- With four attributes the precision stays near 52% even at its optimum, since
  a per-attribute accuracy of about 85% is multiplied four times
  (0.85⁴ ≈ 0.52) *(recorded)*.
- For each search target, tracked persons with a score ≥ 0.25 that pass the
  full-body gate are ranked, and the top 3 are passed on for sending.
- A target is active when it is in the server's search-target list and assigned
  to the camera; `searchStart`/`searchEnd` are not used, since that window
  belongs to the recorded-video system. Only assigned cameras are searched,
  because the server rejects candidates from other cameras
  (`422 CAMERA_NOT_SELECTED`).
- The search text is parsed in Korean and English (`parse_query`).

## 8. Evidence selection and sending (`server_link.py`, `server_config.py`)

- **One event per track.** A person standing for 5 minutes gives thousands of
  matching frames *(recorded)*; each track keeps its best crop over a 2 s window
  (`BEST_WINDOW_SEC`) and is sent once (`REFRESH_SEC = 0`); a track seen again
  after 30 s is a new event (`TRACK_FORGET_SEC`).
- **Best crop.** Quality = score × 0.45 + size × 0.2 + min(1, sharpness / 900) ×
  0.35, with size = min(1, box height / (0.5 × frame height)). Sharpness is the
  variance of the Laplacian on the crop scaled to a long side of 160 px, so that
  it does not grow with the crop size. Reason: blurred crops with a high score
  were chosen over sharp ones. On the 79 crops sharpness varies up to 12×, has a
  correlation of −0.25 with the box height, and has median 849, 10th percentile
  517 and 90th percentile 1413 *(recorded)*.
- **Blur filter.** Crops with sharpness below 300 (`MIN_SHARPNESS`) are not
  used; 300 is below the recorded 10th percentile of 517 *(recorded)*.
- **Crop margin.** 6% of the box on each side is added to the uploaded crop so
  that heads and feet are not cut; the reported box stays the detected box.
- **Rate and queue.** Sends from one camera are at least 1 s apart
  (`MIN_SEND_GAP_SEC`); the send queue holds 16 items because they hold images.
- **Upload.** Upload URLs from the server → PUT of the JPEG (quality 85) to the
  presigned URL → candidate event, retried after 1, 2, 4 and 8 s with the same
  event id. Registering the event is retried for 429, 500, 502, 503 and 504; a
  failed image PUT is retried for any status; an exception (for example a
  timeout) is not retried.
- **Event ids** (`{cameraCode}-{YYYYMMDD}-{seq:06d}`) are saved to
  `runtime/event_seq.json` after every event. Saving every 20 events let a
  stopped process reuse ids and caused `409 EVENT_ID_CONFLICT` *(recorded)*.
- Evidence of uploads that end with an error response is kept in
  `output/detections/` (at most 300 events).

## 9. Keeping the GPU loop free

- **Camera threads.** One thread per camera keeps only the newest frame
  (`CAP_PROP_BUFFERSIZE = 1`), retries 3 s after a failed open and 2 s after a
  lost stream, and uses low-latency RTSP options (TCP, no buffering, no
  reordering).
- **Viewer stream.** JPEG encoding and TCP sending to the monitoring PC run in
  their own thread (`FrameSender`), which keeps only the newest frame per
  camera; a slow or disconnected PC drops old frames instead of blocking the
  loop. Frames are scaled to a width of at most 960 and encoded with quality 75.
- **Server threads.** Target synchronisation and event sending run in their
  own threads with timeouts of 8 s (API) and 20 s (upload); the service keeps
  monitoring when the server is down.
- **Cameras opened on demand.** Only `camera-01` … `camera-04` and the cameras
  assigned by the server are open (`CameraPool.reconcile`).
- Console output replaces characters it cannot encode, since an encoding error
  in a print would stop a thread and the uploads with it *(recorded)*.

## 10. Server synchronisation

- **Polling** every 5 s with `If-None-Match` (an unchanged list costs a 304);
  back-off 1, 2, 4, 8, 15 s after errors; after 5 authentication failures the
  interval becomes 30 s.
- **RabbitMQ** (`mq_listener.py`) is used to react at once: a message triggers
  a full resynchronisation, which must succeed before the message is
  acknowledged (otherwise it is re-queued and the listener waits 2 s); repeated command ids are
  ignored (last 512). Without RabbitMQ the service works by polling alone.
- **Device key** in the `X-Device-Key` header, from `YOPAR_DEVICE_KEY` or
  `devicekey.txt`; the file is read again when it changes, so the key can be
  replaced without a restart.

## 11. Guards and diagnostics

- **One instance.** `run_yopar.sh` refuses to start a second instance. Reason:
  when onnxruntime cannot get the GPU it falls back to the CPU without an error,
  and two instances at once are the most frequent cause *(recorded)*.
- **CPU warning.** After warm-up, if neither TensorRT nor CUDA is active, a
  warning with a checklist is printed. On the CPU, PAR takes about 500 ms
  instead of 4 ms *(recorded)*.
- **Statistics** every 60 frames: frames per second per camera, persons per
  frame, cache hit rate, gate rejections per reason (as counted above),
  synchronisation and upload counters.

## 12. Attribute model

The service uses `color_par_v4_multi_resnet50_sleeve.onnx` (v4). Versions v1–v5
were trained with `train/`; the kept scripts (v1, v2, v4, v5) use a 256×128
input, an ImageNet-pretrained backbone, Adam (learning rate 3e-4, weight decay
1e-4), batch size 64, 10% of the data for validation with `random.Random(0)`
(v1 by image, v2–v5 by person) and keep the epoch with the best mean
validation accuracy.

| version | data | backbone | heads | epochs | notes |
| --- | --- | --- | --- | --- | --- |
| v1 | Market-1501 | resnet18 | gender, upper (8), lower (9) | 20 | |
| v2 | PETA | resnet50 | gender, upper (11), lower (11) | 25 | class-weighted color losses |
| v3 | PETA + Market | resnet50 (also `swin_t`) | gender, upper, lower | 25 | script not kept |
| v4 | PETA + Market | resnet50 | as v3 + sleeve (2) | 25 | class-weighted color and sleeve losses |
| v5 | PETA + Market | resnet50 | as v4 | 40 | + ColorJitter, RandomErasing, weighted sampler, cosine schedule |

**L-a Data split per training run** (`results/summary/tables.md`)

| version | log | images | train | val | persons | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | `v1_market_resnet18.log` | 12936 | 11643 | 1293 | 751 | 20 |
| v2 | `v2_peta_resnet50.log` | 18986 | 17381 | 1605 | 8691 | 25 |
| v3 | `v3_multi_resnet50.log` | 31922 | 28558 | 3364 | 9442 | 25 |
| v3 (swin_t) | `v3_multi_swin_t.log` | 31922 | 28558 | 3364 | 9442 | 25 |
| v5 | `v5_multi_resnet50_aug.log` | 31487 | 28128 | 3359 | – | 40 |

**L-b Epochs with the best printed validation mean** (`results/summary/tables.md`)

| version | best mean | epoch | gender | upper | lower | sleeve |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.944 | 19 | 0.950 | 0.949 | 0.933 | – |
| v2 | 0.777 | 20, 25 | 0.892 / 0.885 | 0.703 / 0.725 | 0.735 / 0.720 | – / – |
| v3 | 0.768 | 22 | 0.885 | 0.703 | 0.717 | – |
| v3 (swin_t) | 0.786 | 17 | 0.864 | 0.751 | 0.744 | – |
| v5 | 0.832 | 40 | 0.888 | 0.736 | 0.740 | 0.963 |

The validation parts differ between versions (L-a); v2 prints the same mean at
epochs 20 and 25, so the saved epoch is not known from the log.

**L-c v4 on the validation split (3359 images)** (`results/summary/tables.md`)

| head | accuracy | macro-F1 | mA | top-2 accuracy |
| --- | --- | --- | --- | --- |
| gender | 0.8815 | 0.8801 | 0.8808 | – |
| upper | 0.7264 | 0.6473 | 0.8069 | 0.8604 |
| lower | 0.7210 | 0.6290 | 0.7653 | 0.8776 |
| sleeve | 0.9574 | 0.9574 | 0.9577 | – |
| mean of 4 | 0.8216 | – | – | – |
| all 4 correct | 0.4701 | – | – | – |

**L-d v4 per-class metrics, upper color (validation split)** (`results/summary/tables.md`)

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| black | 1015 | 0.7477 | 0.8966 | 0.8154 |
| blue | 307 | 0.6988 | 0.5668 | 0.6259 |
| brown | 125 | 0.5789 | 0.7920 | 0.6689 |
| green | 210 | 0.7672 | 0.6905 | 0.7268 |
| gray | 568 | 0.7224 | 0.4489 | 0.5537 |
| orange | 27 | 0.5938 | 0.7037 | 0.6441 |
| pink | 22 | 0.6667 | 0.2727 | 0.3871 |
| purple | 101 | 0.5692 | 0.3663 | 0.4458 |
| red | 174 | 0.6857 | 0.8276 | 0.7500 |
| white | 710 | 0.7572 | 0.8169 | 0.7859 |
| yellow | 100 | 0.7245 | 0.7100 | 0.7172 |

**L-e v4 per-class metrics, lower color (validation split)** (`results/summary/tables.md`)

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| black | 1513 | 0.8400 | 0.8777 | 0.8584 |
| blue | 551 | 0.7268 | 0.7241 | 0.7255 |
| brown | 250 | 0.5362 | 0.5040 | 0.5196 |
| green | 55 | 0.5273 | 0.5273 | 0.5273 |
| gray | 712 | 0.5694 | 0.5126 | 0.5395 |
| orange | 4 | – | 0.0000 | – |
| pink | 43 | 0.5625 | 0.8372 | 0.6729 |
| purple | 3 | 0.0000 | 0.0000 | – |
| red | 5 | 0.3571 | 1.0000 | 0.5263 |
| white | 204 | 0.6131 | 0.5980 | 0.6055 |
| yellow | 19 | 0.7500 | 0.6316 | 0.6857 |

**L-f Own test photos (15 images, as recorded; counts = value × 15)** (`results/summary/tables.md`)

| version | setup | gender | upper | lower | sleeve | heads | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Market, resnet18 | 11/15 | 8/15 | 11/15 | – | 3 | 0.6667 |
| v2 | PETA, resnet50 (11 colors) | 14/15 | 7/15 | 13/15 | – | 3 | 0.7556 |
| v3 | PETA+Market | 13/15 | 8/15 | 15/15 | – | 3 | 0.8000 |
| v4 | PETA+Market + sleeve | 13/15 | 12/15 | 13/15 | 15/15 | 4 | 0.8833 |
| v5 | v4 + strong augmentation/sampler | 12/15 | 10/15 | 13/15 | 15/15 | 4 | 0.8333 |

- `eval.py` crops each photo to the largest person box of `yolo11n.pt`
  (confidence 0.25). The 15 photos (9 men, 6 women) are not in the repository.
- v4 has the highest mean (0.8833) and the highest upper-color count (12/15).
- On the v4 validation split the upper color gray has a recall of 0.4489; 126
  of its 568 images are predicted black and 104 white.

## Not in the repository

`edge/samples/bus.jpg` (needed by `prepare.py`), `edge/pylibs/` (a local copy
of onnxruntime; added to the import path when present), the desktop launcher
that calls `run_yopar.sh`, the monitoring viewer that receives the frame
stream (`VIEW_PORT = 5006`), the datasets, the 15 test photos and the 79
labelled crops.
