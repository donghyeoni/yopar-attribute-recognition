# Experiment Log

Settings, result tables and observations of the pedestrian attribute
recognition (PAR) models v1–v5, and the structure and settings of the Jetson
service in `edge/`. The [README](../README.md) summarises the results; every
result value in it appears in this log.

## Conventions

- **Source of every number.** The files under `results/`; no model was
  trained or evaluated again for this document.
  - `results/train_logs/*.log`: console output of the training runs of v1,
    v2, v3, v3 with a `swin_t` backbone and v5. No console output of the v4
    run was kept.
  - `results/v4_val/`: `val_metrics.json` and the confusion matrices
    `cm_*.npy` written by `train/metrics_val.py` for the v4 weights.
  - `results/test15.csv`: accuracies of v1–v5 on 15 own test photos as
    recorded with `eval.py`, which prints them with 2 decimals. The photos and
    their labels are not in the repository.
- **Result tables.** `tools/summarize.py` reads these files and writes
  `results/summary/tables.md` and three figures; the tables below are copied
  unchanged from `tables.md`.
- **Precision.** Values from the training logs are shown as printed (3
  decimals). Values from `val_metrics.json` and derived means are shown with 4
  decimals. On 15 photos each printed 2-decimal accuracy corresponds to exactly
  one count `k/15`; L-f shows these counts and the mean over the heads computed
  from them. `–` marks a value that the source file does not contain (for example
  a count a log does not print), a head the model does not have, or a metric
  that is undefined for the class (precision without predictions, F1 when
  precision and recall are both 0).
- **Metrics** (`train/metrics_val.py`). Accuracy per head; macro-F1 is the mean
  of the per-class F1 over classes where it is defined; mA is the mean of the
  average per-class recall and the average per-class true-negative rate; top-2
  accuracy counts a prediction as correct when the true color is among the two
  highest scores. "all 4 correct" is the fraction of images with all four
  heads correct.

## Data

- **Market-1501** (`data/Market-1501-v15.09.15/bounding_box_train`) with the
  attribute annotations `data/Market-1501_Attribute/market_attribute.mat`
  (`train` split). Images of identity `0000` and `-1` are skipped. Upper colors
  (8): black, white, red, purple, yellow, gray, blue, green; lower colors (9):
  black, white, pink, purple, yellow, gray, blue, green, brown; the color is the
  field with the largest value.
- **PETA** (`data/PETA dataset/*/archive/Label.txt`). Gender from
  `personalMale`/`personalFemale`, colors from `upperBody<Color>` and
  `lowerBody<Color>` with 11 colors (Black, Blue, Brown, Green, Grey, Orange,
  Pink, Purple, Red, White, Yellow), sleeve from
  `upperBodyShortSleeve`/`upperBodyLongSleeve`. Images without one of the
  required labels are skipped.
- **Combined (v3–v5).** Market colors are mapped onto the 11 PETA colors;
  Market sleeve is taken from the `up` field (1 = long, 2 = short). v4 and v5
  also require a sleeve label, which is why they have fewer images than v3
  (L-a).
- The datasets are not in the repository.

## Models and training

Common to the scripts of v1, v2, v4 and v5: torchvision backbone with ImageNet weights, the
classifier replaced by one linear layer per head; input resized to 256×128,
random horizontal flip in training, ImageNet mean/std normalization; Adam,
learning rate 3e-4, weight decay 1e-4, batch size 64; 10% of the data held out
for validation with `random.Random(0)`; after each epoch the validation
accuracy of every head and their mean are printed, and the weights of the epoch
with the highest mean (the later epoch on a tie) are saved.

| version | script | data | backbone | heads | epochs | validation split | loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | `train/v1_market_resnet18.py` | Market | resnet18 | gender, upper (8), lower (9) | 20 | by image | cross-entropy |
| v2 | `train/v2_peta_resnet50.py` | PETA | resnet50 | gender, upper (11), lower (11) | 25 | by person | class-weighted cross-entropy for the colors |
| v3 | earlier version of `train/v4_multi_resnet50_sleeve.py` (not kept) | PETA + Market | resnet50 (also `swin_t`) | gender, upper, lower | 25 | by person | – |
| v4 | `train/v4_multi_resnet50_sleeve.py` | PETA + Market | resnet50 | gender, upper (11), lower (11), sleeve (2) | 25 | by person | class-weighted cross-entropy for colors and sleeve |
| v5 | `train/v5_multi_resnet50_aug.py` | PETA + Market | resnet50 | as v4 | 40 | by person | square root of the v4 class weights |

- Class weights are `N / (C · n_c)` over the training part (`n_c` images of
  class `c`, `C` classes).
- v5 adds, relative to v4: `ColorJitter` (brightness, contrast and saturation
  0.3, hue 0) and `RandomErasing` (p = 0.3, scale 0.02–0.15); a
  `WeightedRandomSampler` with per-image weight
  `1/√n_upper · 1/√n_lower · 1/√n_gender`; cosine learning-rate schedule over
  40 epochs.
- v1 splits the images at random, so images of one identity can be in both
  parts; v2–v5 split the person identities, so no identity is in both.
- The v3 weights were produced by `train/v4_multi_resnet50_sleeve.py` before
  the sleeve head was added; that version of the script is not kept, so the
  v3 row shows only what its log (`v3_multi_resnet50.log`,
  `v3_multi_swin_t.log`) prints.

## Results

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

- v2 prints the same mean (0.777) at epochs 20 and 25. The script saves an
  epoch when its unrounded mean is at least the best so far, so the printed
  values do not show which of the two was saved.
- The validation parts differ between versions (L-a), so the validation
  accuracies of different versions are not measured on the same images. v1
  has 3 heads with 8 and 9 colors and an image-level split.

**L-c v4 on the validation split (3359 images)** (`results/summary/tables.md`)

| head | accuracy | macro-F1 | mA | top-2 accuracy |
| --- | --- | --- | --- | --- |
| gender | 0.8815 | 0.8801 | 0.8808 | – |
| upper | 0.7264 | 0.6473 | 0.8069 | 0.8604 |
| lower | 0.7210 | 0.6290 | 0.7653 | 0.8776 |
| sleeve | 0.9574 | 0.9574 | 0.9577 | – |
| mean of 4 | 0.8216 | – | – | – |
| all 4 correct | 0.4701 | – | – | – |

`train/metrics_val.py` rebuilds the v4 validation split with the same code and
seed as the training script and evaluates the saved v4 weights; its 3359
validation images equal the validation size printed by v5 (L-a).

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

Observations (L-c–L-e):

- Sleeve has the highest accuracy of the four heads (0.9574) and the upper
  and lower colors the lowest (0.7264, 0.7210). The top-2 accuracy of the
  colors is 0.8604 (upper) and 0.8776 (lower).
- Upper color: gray has a recall of 0.4489 over 568 images; in the confusion
  matrix 126 gray images are predicted black and 104 white. pink (22 images)
  has the lowest F1 (0.3871).
- Lower color: orange (4 images) and purple (3 images) have no correct
  prediction; red has 5 images.

**L-f Own test photos (15 images, as recorded; counts = value × 15)** (`results/summary/tables.md`)

| version | setup | gender | upper | lower | sleeve | heads | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Market, resnet18 | 11/15 | 8/15 | 11/15 | – | 3 | 0.6667 |
| v2 | PETA, resnet50 (11 colors) | 14/15 | 7/15 | 13/15 | – | 3 | 0.7556 |
| v3 | PETA+Market | 13/15 | 8/15 | 15/15 | – | 3 | 0.8000 |
| v4 | PETA+Market + sleeve | 13/15 | 12/15 | 13/15 | 15/15 | 4 | 0.8833 |
| v5 | v4 + strong augmentation/sampler | 12/15 | 10/15 | 13/15 | 15/15 | 4 | 0.8333 |

- `eval.py` rotates each photo by its EXIF orientation, crops it to the
  largest person box found by YOLO (`yolo11n.pt`, confidence 0.25; the whole
  photo when no person is found) and compares the PAR prediction with the
  label file. The 15 photos were recorded as 9 men and 6 women.
- v4 has the highest mean (0.8833); its upper color is 12/15 against 7/15 to
  10/15 for the other versions. v1–v3 have 3 heads, so their mean is over 3
  heads and v4–v5 over 4.
- v4 is the version deployed on the Jetson (`edge/`).

## Jetson service (`edge/`)

Runs on the real-time AI device of the system (Jetson Orin Nano): RTSP streams
of 4 cameras → person detection → attribute recognition → matching against the
search targets registered on the central server → candidate events with the
person's box and evidence images sent to the server. No measurement of the
service is part of this repository.

| file | role |
| --- | --- |
| `scripts/jetson_par_sender.py` | entry point: camera pool, detection, attributes, matching, sending |
| `scripts/capture_core.py` | RTSP capture, loading of the YOLO and PAR ONNX models, attribute tracking, full-body gate |
| `scripts/onnx_yolo.py` | YOLO11 person detection with onnxruntime and OpenCV (resize, NMS), without PyTorch |
| `scripts/server_config.py` | backend settings read from `edge/.env` and `devicekey.txt` |
| `scripts/server_link.py` | synchronisation of the search targets and upload of detections |
| `scripts/mq_listener.py` | RabbitMQ consumer for notifications; polling is used as well |
| `scripts/prepare.py` | TensorRT engine build; compares TensorRT fp16 with CUDA fp32 outputs for YOLO and PAR and stores the choice in `models/provider_verdict.json` |
| `run_yopar.sh` | wrapper that refuses to start a second instance, then runs the entry point |

Settings in the code:

- **Models** (`edge/models/`, not in the repository): `yolo11s.onnx` and
  `color_par_v4_multi_resnet50_sleeve.onnx`; the TensorRT engines are cached in
  `edge/models/trt_cache/`.
- **Detection** (`capture_core.py`): input 640, confidence 0.35, NMS IoU 0.5,
  at most 20 detections; batch sizes 1, 2 or 4.
- **Attributes**: at most 8 crops per batch, batch sizes 1, 2, 4 or 8; a
  detected box reuses the attributes of a tracked box with IoU above 0.5 when
  they are at most 12 frames old; sleeve is `long` when its probability is at
  least 0.95.
- **Full-body gate**: box height ≥ 120 and width ≥ 70 pixels, height/width
  between 1.8 and 4.5, and more than 6 pixels from every image border.
- **Matching** (`jetson_par_sender.py`): for each search target, tracked
  persons with a match score of at least 0.25 that pass the full-body gate
  are ranked by score and the top 3 are passed to `server_link.py`.
- **Sending** (`server_link.py`, settings in `server_config.py`): crops with a
  sharpness below 300 are dropped; each track keeps its best crop over a 2 s
  window and is sent once (`REFRESH_SEC = 0`); sends from one camera are at
  least 1 s apart.
- **Server**: polling every 5 s with back-off 1, 2, 4, 8, 15 s; device key in
  the `X-Device-Key` header.
- `prepare.py` reads the sample image `edge/samples/bus.jpg`, which is not in
  the repository.
- `edge/requirements.txt` lists the packages installed with pip; the service
  also imports `cv2`, `numpy` and `requests`.
