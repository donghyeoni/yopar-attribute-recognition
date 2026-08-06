"""색상+소매 PAR 학습 (v4) — PETA + Market-1501 통합(11색), 4헤드.
헤드: 성별 / 상의색(11) / 하의색(11) / 소매(short,long).
백본 선택(resnet50 / swin_t). 인물 단위 val 분할, 클래스 가중.

주의: weights/color_par_v3_multi_resnet50.*는 이 파일에 소매 헤드를 추가하기 전
(3헤드) 버전의 결과물이라 지금 이 스크립트로는 재현되지 않는다.

실행:
  python train/v4_multi_resnet50_sleeve.py --backbone resnet50 --out weights/color_par_v4_multi_resnet50_sleeve.pt
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
COLORS = ["black", "blue", "brown", "green", "gray", "orange", "pink",
          "purple", "red", "white", "yellow"]
SLEEVES = ["short", "long"]
CIDX = {c: i for i, c in enumerate(COLORS)}
INPUT_HW = (256, 128)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

PETA_ROOT = "data/PETA dataset"
PETA_TOK = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
            "Purple", "Red", "White", "Yellow"]

MK_ROOT = "data/Market-1501-v15.09.15/bounding_box_train"
MK_ANN = "data/Market-1501_Attribute/market_attribute.mat"
MK_UP = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
MK_DN = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]
MK_UP_F = ["upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen"]
MK_DN_F = ["downblack", "downwhite", "downpink", "downpurple", "downyellow",
           "downgray", "downblue", "downgreen", "downbrown"]


class ColorPARNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=False):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        if hasattr(net, "fc"):
            d = net.fc.in_features; net.fc = nn.Identity()
        else:
            d = net.head.in_features; net.head = nn.Identity()
        self.backbone = net
        self.gender = nn.Linear(d, len(GENDERS))
        self.upper = nn.Linear(d, len(COLORS))
        self.lower = nn.Linear(d, len(COLORS))
        self.sleeve = nn.Linear(d, len(SLEEVES))

    def forward(self, x):
        f = self.backbone(x)
        return self.gender(f), self.upper(f), self.lower(f), self.sleeve(f)


def build_peta():
    items = []
    for lab in glob.glob(os.path.join(PETA_ROOT, "*", "archive", "Label.txt")):
        subset = lab.split(os.sep)[-3]
        arch = os.path.dirname(lab)
        id2 = {}
        for line in open(lab, encoding="utf-8", errors="ignore"):
            p = line.split()
            if p:
                id2[p[0]] = set(p[1:])
        for img in glob.glob(os.path.join(arch, "*")):
            if os.path.splitext(img)[1].lower() not in (".bmp", ".png", ".jpg", ".jpeg"):
                continue
            tok = id2.get(os.path.basename(img).split("_")[0])
            if not tok:
                continue
            g = 0 if "personalMale" in tok else (1 if "personalFemale" in tok else None)
            up = next((i for i, t in enumerate(PETA_TOK) if "upperBody" + t in tok), None)
            dn = next((i for i, t in enumerate(PETA_TOK) if "lowerBody" + t in tok), None)
            sl = (0 if "upperBodyShortSleeve" in tok else
                  (1 if "upperBodyLongSleeve" in tok else None))
            if None in (g, up, dn, sl):
                continue
            items.append((img, g, up, dn, sl,
                          f"peta:{subset}/{os.path.basename(img).split('_')[0]}"))
    return items


def build_market():
    from scipy.io import loadmat
    if not os.path.isdir(MK_ROOT):
        return []
    s = getattr(loadmat(MK_ANN, struct_as_record=False, squeeze_me=True)["market_attribute"], "train")
    ids = [str(x).zfill(4) for x in np.atleast_1d(s.image_index)]

    def col(n):
        return np.atleast_1d(getattr(s, n)).astype(int)

    gender = col("gender")
    up = np.stack([col(f) for f in MK_UP_F], axis=1)
    dn = np.stack([col(f) for f in MK_DN_F], axis=1)
    uplen = col("up")   # 1=long, 2=short
    lab = {}
    for i, pid in enumerate(ids):
        sl = 0 if int(uplen[i]) == 2 else 1     # short=0, long=1
        lab[pid] = (int(gender[i] - 1),
                    CIDX[MK_UP[int(up[i].argmax())]],
                    CIDX[MK_DN[int(dn[i].argmax())]], sl)
    items = []
    for img in glob.glob(os.path.join(MK_ROOT, "*.jpg")):
        pid = os.path.basename(img).split("_")[0]
        if pid in ("0000", "-1") or pid not in lab:
            continue
        g, u, d, sl = lab[pid]
        items.append((img, g, u, d, sl, f"market:{pid}"))
    return items


class DS(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, g, u, d, s, _ = self.items[i]
        return self.tfm(Image.open(f).convert("RGB")), g, u, d, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--out", default="weights/color_par_v4_multi_resnet50_sleeve.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = build_peta() + build_market()
    print(f"[multi+sleeve] 이미지 {len(items)} / backbone {args.backbone} / {device}")

    groups = sorted({it[5] for it in items})
    random.Random(0).shuffle(groups)
    val_g = set(groups[:max(1, int(len(groups) * args.val_split))])
    tr = [it for it in items if it[5] not in val_g]
    va = [it for it in items if it[5] in val_g]
    print(f"[multi+sleeve] train {len(tr)} / val {len(va)} (인물 {len(groups)})")

    H, W = INPUT_HW
    tfm_tr = T.Compose([T.Resize((H, W)), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(_MEAN, _STD)])
    tfm_va = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    dl_tr = DataLoader(DS(tr, tfm_tr), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(DS(va, tfm_va), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = ColorPARNet(args.backbone, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def cls_w(idx, n):
        cnt = np.bincount([it[idx] for it in tr], minlength=n).astype(float)
        cnt[cnt == 0] = 1
        return torch.tensor(cnt.sum() / (n * cnt), dtype=torch.float32, device=device)

    lg = nn.CrossEntropyLoss()
    lu = nn.CrossEntropyLoss(weight=cls_w(2, len(COLORS)))
    ll = nn.CrossEntropyLoss(weight=cls_w(3, len(COLORS)))
    ls = nn.CrossEntropyLoss(weight=cls_w(4, len(SLEEVES)))

    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        for x, g, u, d, s in dl_tr:
            x, g, u, d, s = (x.to(device), g.to(device), u.to(device),
                             d.to(device), s.to(device))
            pg, pu, pd, ps = model(x)
            loss = lg(pg, g) + lu(pu, u) + ll(pd, d) + ls(ps, s)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        cg = cu = cd = cs = tot = 0
        with torch.no_grad():
            for x, g, u, d, s in dl_va:
                pg, pu, pd, ps = model(x.to(device))
                cg += (pg.argmax(1).cpu() == g).sum().item()
                cu += (pu.argmax(1).cpu() == u).sum().item()
                cd += (pd.argmax(1).cpu() == d).sum().item()
                cs += (ps.argmax(1).cpu() == s).sum().item()
                tot += g.size(0)
        ag, au, ad, asl = cg / tot, cu / tot, cd / tot, cs / tot
        m = (ag + au + ad + asl) / 4
        print(f"  ep {ep+1:3d}  gen {ag:.3f}  up {au:.3f}  low {ad:.3f}  slv {asl:.3f}  mean {m:.3f}")
        if m >= best:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": args.backbone,
                "genders": GENDERS, "upper_colors": COLORS, "lower_colors": COLORS,
                "sleeves": SLEEVES, "input_hw": list(INPUT_HW)}, args.out)
    print(f"[multi+sleeve] 완료. best mean_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
