"""v5 학습 — v4와 같은 데이터(PETA+Market), 학습 레시피만 개선.

v4 대비 변경점:
  1) 색 증강: ColorJitter(brightness/contrast/saturation) — hue는 라벨이므로 건드리지 않음
     + RandomErasing(가림 강건성)
  2) 희귀색·여성 오버샘플링: WeightedRandomSampler (희귀 상/하의색, 여성 비중↑)
  3) 40 epoch + cosine LR 스케줄
  4) 헤드 4개 유지(성별/상의색/하의색/소매)

실행: python train/v5_multi_resnet50_aug.py
결과: weights/color_par_v5_multi_resnet50_aug.pt
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from v4_multi_resnet50_sleeve import (ColorPARNet, build_peta, build_market,
                                      COLORS, GENDERS, SLEEVES, INPUT_HW, _MEAN, _STD)


class DS(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, g, u, d, s, _ = self.items[i]
        return self.tfm(Image.open(f).convert("RGB")), g, u, d, s


def make_sampler(items):
    """희귀 색상 + 여성 비중을 올리는 샘플 가중치."""
    n_c = len(COLORS)
    up_cnt = np.bincount([it[2] for it in items], minlength=n_c).astype(float)
    dn_cnt = np.bincount([it[3] for it in items], minlength=n_c).astype(float)
    g_cnt = np.bincount([it[1] for it in items], minlength=2).astype(float)
    up_cnt[up_cnt == 0] = 1; dn_cnt[dn_cnt == 0] = 1; g_cnt[g_cnt == 0] = 1
    # 역빈도 기반(제곱근으로 완화) 가중치 결합
    w_up = 1.0 / np.sqrt(up_cnt)
    w_dn = 1.0 / np.sqrt(dn_cnt)
    w_g = 1.0 / np.sqrt(g_cnt)
    w = np.array([w_up[it[2]] * w_dn[it[3]] * w_g[it[1]] for it in items])
    w = w / w.mean()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(items), replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--out", default="weights/color_par_v5_multi_resnet50_aug.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = build_peta() + build_market()
    groups = sorted({it[5] for it in items})
    random.Random(0).shuffle(groups)
    val_g = set(groups[:max(1, int(len(groups) * args.val_split))])
    tr = [it for it in items if it[5] not in val_g]
    va = [it for it in items if it[5] in val_g]
    print(f"[v5] images {len(items)} / train {len(tr)} / val {len(va)} / {device}")

    H, W = INPUT_HW
    # 색 증강: hue는 제외(라벨이 색). 밝기/대비/채도만 흔든다.
    tfm_tr = T.Compose([
        T.Resize((H, W)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    tfm_va = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(_MEAN, _STD)])

    dl_tr = DataLoader(DS(tr, tfm_tr), batch_size=args.batch,
                       sampler=make_sampler(tr), num_workers=args.workers,
                       pin_memory=True)
    dl_va = DataLoader(DS(va, tfm_va), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = ColorPARNet(args.backbone, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def cls_w(idx, n):
        cnt = np.bincount([it[idx] for it in tr], minlength=n).astype(float)
        cnt[cnt == 0] = 1
        return torch.tensor(cnt.sum() / (n * cnt), dtype=torch.float32, device=device)

    # 샘플러가 이미 균형을 잡으므로 손실 가중은 약하게(sqrt)
    def soft_w(idx, n):
        w = cls_w(idx, n)
        return w.sqrt()

    lg = nn.CrossEntropyLoss()
    lu = nn.CrossEntropyLoss(weight=soft_w(2, len(COLORS)))
    ll = nn.CrossEntropyLoss(weight=soft_w(3, len(COLORS)))
    ls = nn.CrossEntropyLoss(weight=soft_w(4, len(SLEEVES)))

    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        for x, g, u, d, s in dl_tr:
            x, g, u, d, s = (x.to(device), g.to(device), u.to(device),
                             d.to(device), s.to(device))
            pg, pu, pd, ps = model(x)
            loss = lg(pg, g) + lu(pu, u) + ll(pd, d) + ls(ps, s)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

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
    print(f"[v5] 완료. best mean_acc={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
