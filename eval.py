"""색상(+소매) PAR 평가 (팔레트·헤드수 무관). 체크포인트 메타를 읽어
Market(8/9색·3헤드)든 PETA/통합(11색)이든, 소매 헤드 유무든 자동 대응.

test_image/ 사진을 YOLO로 사람 crop(EXIF 회전 보정) → 추론 → test_labels.csv와 대조.
CSV에 'sleeve' 열이 있으면 소매도 채점(short/long).

실행: python eval.py --weights weights/color_par_v4_multi_resnet50_sleeve.pt
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class Net(nn.Module):
    def __init__(self, backbone, n_g, n_u, n_d, n_s=0):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights=None)
        if hasattr(net, "fc"):
            d = net.fc.in_features; net.fc = nn.Identity()
        else:
            d = net.head.in_features; net.head = nn.Identity()
        self.backbone = net
        self.gender = nn.Linear(d, n_g)
        self.upper = nn.Linear(d, n_u)
        self.lower = nn.Linear(d, n_d)
        self.has_sleeve = n_s > 0
        if self.has_sleeve:
            self.sleeve = nn.Linear(d, n_s)

    def forward(self, x):
        f = self.backbone(x)
        out = [self.gender(f), self.upper(f), self.lower(f)]
        if self.has_sleeve:
            out.append(self.sleeve(f))
        return out


def load_upright_bgr(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="test_image")
    ap.add_argument("--csv", default="test_image/test_labels.csv")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.weights, map_location="cpu")
    GEN, UP, LO = ck["genders"], ck["upper_colors"], ck["lower_colors"]
    SLV = ck.get("sleeves")
    model = Net(ck["backbone"], len(GEN), len(UP), len(LO), len(SLV) if SLV else 0)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    H, W = ck.get("input_hw", (256, 128))
    tfm = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(_MEAN, _STD)])

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    has_slv_gt = SLV is not None and rows and "sleeve" in rows[0]
    cg = cu = cd = cs = tot = 0
    print(f"[{os.path.basename(args.weights)}]  (sleeve head: {'O' if SLV else 'X'})")
    for r in rows:
        path = os.path.join(args.dir, r["filename"])
        if not os.path.isfile(path):
            continue
        img = load_upright_bgr(path)
        res = yolo.predict(img, classes=[0], conf=0.25, verbose=False, device=device)
        xyxy = res[0].boxes.xyxy.detach().cpu().numpy().astype(int)
        if len(xyxy):
            a = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            x0, y0, x1, y1 = xyxy[a.argmax()]
            crop = img[max(0, y0):y1, max(0, x0):x1]
        else:
            crop = img
        x = tfm(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(x)
        gi, ui, di = int(outs[0].argmax()), int(outs[1].argmax()), int(outs[2].argmax())
        pred = [GEN[gi], UP[ui], LO[di]]
        true = [r["gender"].strip(), r["upper"].strip(), r["lower"].strip()]
        if SLV:
            si = int(outs[3].argmax())
            pred.append(SLV[si])
            true.append(r.get("sleeve", "").strip())
        mg, muu, md = pred[0] == true[0], pred[1] == true[1], pred[2] == true[2]
        cg += mg; cu += muu; cd += md; tot += 1
        line = f"  {r['filename']:9} P{tuple(pred)}  T{tuple(true)}  {'G' if mg else '.'}{'U' if muu else '.'}{'L' if md else '.'}"
        if has_slv_gt:
            ms = pred[3] == true[3]; cs += ms; line += "S" if ms else "."
        print(line)

    print(f"  -> gender {cg/tot:.2f}  upper {cu/tot:.2f}  lower {cd/tot:.2f}"
          + (f"  sleeve {cs/tot:.2f}" if has_slv_gt else "") + f"  (n={tot})")


if __name__ == "__main__":
    main()
