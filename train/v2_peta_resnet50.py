"""색상 PAR 학습 (PETA, 11색 + red 하의 포함). 성별/상의색/하의색 다중헤드.

PETA 원본(각 subset의 archive/Label.txt)을 파싱해 학습.
색 11종: black blue brown green gray orange pink purple red white yellow
(하의도 red 포함 → "빨간 바지" 검색 가능)

실행(venv):
  python train/v2_peta_resnet50.py
  python train/v2_peta_resnet50.py --backbone resnet18 --epochs 30
결과: weights/color_par_peta.pt
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

GENDERS = ["male", "female"]
# PETA 색 토큰 -> 표준(소문자, grey->gray)
COLOR_TOK = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
             "Purple", "Red", "White", "Yellow"]
COLORS = ["black", "blue", "brown", "green", "gray", "orange", "pink",
          "purple", "red", "white", "yellow"]
UPPER_COLORS = COLORS
LOWER_COLORS = COLORS
INPUT_HW = (256, 128)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
PETA_ROOT = "data/PETA dataset"


class ColorPARNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=False):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.gender = nn.Linear(self.feat_dim, len(GENDERS))
        self.upper = nn.Linear(self.feat_dim, len(UPPER_COLORS))
        self.lower = nn.Linear(self.feat_dim, len(LOWER_COLORS))

    def forward(self, x):
        f = self.backbone(x)
        return self.gender(f), self.upper(f), self.lower(f)


def parse_attrs(tokens):
    g = 0 if "personalMale" in tokens else (1 if "personalFemale" in tokens else None)
    up = next((i for i, c in enumerate(COLOR_TOK) if "upperBody" + c in tokens), None)
    dn = next((i for i, c in enumerate(COLOR_TOK) if "lowerBody" + c in tokens), None)
    return g, up, dn


def build_items():
    """(path, gender, up, down, group) 리스트. 색·성별 모두 있는 것만."""
    items = []
    for label_txt in glob.glob(os.path.join(PETA_ROOT, "*", "archive", "Label.txt")):
        subset = label_txt.split(os.sep)[-3]
        arch = os.path.dirname(label_txt)
        id2tok = {}
        with open(label_txt, encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split()
                if p:
                    id2tok[p[0]] = set(p[1:])
        for img in glob.glob(os.path.join(arch, "*")):
            ext = os.path.splitext(img)[1].lower()
            if ext not in (".bmp", ".png", ".jpg", ".jpeg"):
                continue
            pid = os.path.basename(img).split("_")[0]
            tok = id2tok.get(pid)
            if not tok:
                continue
            g, up, dn = parse_attrs(tok)
            if g is None or up is None or dn is None:
                continue
            items.append((img, g, up, dn, f"{subset}/{pid}"))
    return items


class PETAColor(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, g, u, d, _ = self.items[i]
        return self.tfm(Image.open(f).convert("RGB")), g, u, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--out", default="weights/color_par_peta.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = build_items()
    print(f"[peta] 학습가능 이미지 {len(items)} / device {device}")
    if not items:
        print("PETA 데이터를 못 찾음. data/PETA dataset/ 확인.")
        return

    # 인물(group) 단위 val 분할 → 누수 방지
    groups = sorted({it[4] for it in items})
    random.Random(0).shuffle(groups)
    n_val_g = max(1, int(len(groups) * args.val_split))
    val_g = set(groups[:n_val_g])
    tr_items = [it for it in items if it[4] not in val_g]
    va_items = [it for it in items if it[4] in val_g]
    print(f"[peta] train {len(tr_items)} / val {len(va_items)} (인물 {len(groups)}명)")

    H, W = INPUT_HW
    tfm_tr = T.Compose([T.Resize((H, W)), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(_MEAN, _STD)])
    tfm_va = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    dl_tr = DataLoader(PETAColor(tr_items, tfm_tr), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(PETAColor(va_items, tfm_va), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ColorPARNet(args.backbone, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 색 불균형 보정: 클래스 가중(역빈도). 희귀색(red 하의 등) 인식↑
    def cls_w(idx, n):
        cnt = np.bincount([it[idx] for it in tr_items], minlength=n).astype(float)
        cnt[cnt == 0] = 1
        return torch.tensor(cnt.sum() / (n * cnt), dtype=torch.float32, device=device)

    lossf_g = nn.CrossEntropyLoss()
    lossf_u = nn.CrossEntropyLoss(weight=cls_w(2, len(UPPER_COLORS)))
    lossf_d = nn.CrossEntropyLoss(weight=cls_w(3, len(LOWER_COLORS)))

    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        for x, g, u, d in dl_tr:
            x, g, u, d = x.to(device), g.to(device), u.to(device), d.to(device)
            pg, pu, pd = model(x)
            loss = lossf_g(pg, g) + lossf_u(pu, u) + lossf_d(pd, d)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        cg = cu = cd = tot = 0
        with torch.no_grad():
            for x, g, u, d in dl_va:
                pg, pu, pd = model(x.to(device))
                cg += (pg.argmax(1).cpu() == g).sum().item()
                cu += (pu.argmax(1).cpu() == u).sum().item()
                cd += (pd.argmax(1).cpu() == d).sum().item()
                tot += g.size(0)
        ag, au, ad = cg / tot, cu / tot, cd / tot
        mean = (ag + au + ad) / 3
        print(f"  ep {ep+1:3d}  gender {ag:.3f}  upper {au:.3f}  lower {ad:.3f}  mean {mean:.3f}")
        if mean >= best:
            best = mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": args.backbone,
                "genders": GENDERS, "upper_colors": UPPER_COLORS,
                "lower_colors": LOWER_COLORS, "input_hw": list(INPUT_HW)}, args.out)
    print(f"[peta] 완료. best mean_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
