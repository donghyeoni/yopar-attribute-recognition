"""색상 PAR 학습 (독립 실행). Market-1501 attribute로 성별/상의색/하의색 학습.
결과 weights/color_par.pt 를 Jetson 추론에 사용.

실행(venv):
  python train/v1_market_resnet18.py
  python train/v1_market_resnet18.py --backbone resnet50 --epochs 30
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
UPPER_COLORS = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
LOWER_COLORS = ["black", "white", "pink", "purple", "yellow", "gray", "blue",
                "green", "brown"]
UP_FIELDS = ["upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray",
             "upblue", "upgreen"]
DOWN_FIELDS = ["downblack", "downwhite", "downpink", "downpurple", "downyellow",
               "downgray", "downblue", "downgreen", "downbrown"]
INPUT_HW = (256, 128)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ColorPARNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=False):
        super().__init__()
        ctor = getattr(torchvision.models, backbone)
        net = ctor(weights="DEFAULT" if pretrained else None)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.gender = nn.Linear(self.feat_dim, len(GENDERS))
        self.upper = nn.Linear(self.feat_dim, len(UPPER_COLORS))
        self.lower = nn.Linear(self.feat_dim, len(LOWER_COLORS))

    def forward(self, x):
        f = self.backbone(x)
        return self.gender(f), self.upper(f), self.lower(f)


def load_labels(ann, split="train"):
    from scipy.io import loadmat
    m = loadmat(ann, struct_as_record=False, squeeze_me=True)
    s = getattr(m["market_attribute"], split)
    ids = [str(x).zfill(4) for x in np.atleast_1d(s.image_index)]

    def col(name):
        return np.atleast_1d(getattr(s, name)).astype(int)

    gender = col("gender")
    up = np.stack([col(f) for f in UP_FIELDS], axis=1)
    down = np.stack([col(f) for f in DOWN_FIELDS], axis=1)
    return {pid: (int(gender[i] - 1), int(up[i].argmax()), int(down[i].argmax()))
            for i, pid in enumerate(ids)}


class MarketColor(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, (g, u, d) = self.items[i]
        return self.tfm(Image.open(f).convert("RGB")), g, u, d


def build_items(img_dir, labels):
    items = []
    for f in glob.glob(os.path.join(img_dir, "*.jpg")):
        pid = os.path.basename(f).split("_")[0]
        if pid in ("0000", "-1") or pid not in labels:
            continue
        items.append((f, labels[pid]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Market-1501-v15.09.15")
    ap.add_argument("--ann", default="data/Market-1501_Attribute/market_attribute.mat")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--out", default="weights/color_par.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = os.path.join(args.root, "bounding_box_train")
    labels = load_labels(args.ann, "train")
    items = build_items(img_dir, labels)
    print(f"[train] identity {len(labels)} / images {len(items)} / device {device}")

    H, W = INPUT_HW
    tfm_tr = T.Compose([T.Resize((H, W)), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(_MEAN, _STD)])
    tfm_va = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(_MEAN, _STD)])

    random.Random(0).shuffle(items)
    n_val = max(1, int(len(items) * args.val_split))
    va = MarketColor(items[:n_val], tfm_va)
    tr = MarketColor(items[n_val:], tfm_tr)
    print(f"[train] train {len(tr)} / val {len(va)}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = ColorPARNet(args.backbone, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        for x, g, u, d in dl_tr:
            x, g, u, d = (x.to(device), g.to(device), u.to(device), d.to(device))
            pg, pu, pd = model(x)
            loss = lossf(pg, g) + lossf(pu, u) + lossf(pd, d)
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
        print(f"  ep {ep + 1:3d}  gender {ag:.3f}  upper {au:.3f}  lower {ad:.3f}  mean {mean:.3f}")
        if mean >= best:
            best = mean
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": args.backbone,
                "genders": GENDERS, "upper_colors": UPPER_COLORS,
                "lower_colors": LOWER_COLORS, "input_hw": list(INPUT_HW)}, args.out)
    print(f"[train] 완료. best mean_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
