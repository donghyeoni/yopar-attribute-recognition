import json
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_multi_resnet50_sleeve as v4

OUT = "results/v4_val"
WEIGHTS = "weights/color_par_v4_multi_resnet50_sleeve.pt"


def per_class_metrics(cm):
    n = cm.shape[0]
    total = cm.sum()
    rows, tprs, tnrs = [], [], []
    for c in range(n):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if (tp + fp) and (tp + fn) and (prec + rec) else float("nan"))
        rows.append({"precision": prec, "recall": rec, "f1": f1, "support": int(cm[c].sum())})
        if tp + fn:
            tprs.append(rec)
            tnrs.append(tn / (tn + fp) if tn + fp else float("nan"))
    f1s = [r["f1"] for r in rows if not np.isnan(r["f1"])]
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    mA = float((np.mean(tprs) + np.nanmean(tnrs)) / 2) if tprs else float("nan")
    return rows, macro_f1, mA


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = v4.build_peta() + v4.build_market()
    groups = sorted({it[5] for it in items})
    random.Random(0).shuffle(groups)
    val_g = set(groups[: max(1, int(len(groups) * 0.1))])
    va = [it for it in items if it[5] in val_g]
    print(f"[metrics] 전체 {len(items)} / val {len(va)} (인물 {len(groups)}) / {device}")

    ck = torch.load(WEIGHTS, map_location="cpu")
    model = v4.ColorPARNet(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    H, W = v4.INPUT_HW
    tfm = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(v4._MEAN, v4._STD)])
    dl = DataLoader(v4.DS(va, tfm), batch_size=128, shuffle=False, num_workers=0)

    T_, P_, TOP2 = [[] for _ in range(4)], [[] for _ in range(4)], [[], []]
    with torch.no_grad():
        for x, g, u, d, s in dl:
            outs = model(x.to(device))
            for hi, gt in enumerate((g, u, d, s)):
                T_[hi].append(gt.numpy())
                P_[hi].append(outs[hi].argmax(1).cpu().numpy())
            for k, hi in enumerate((1, 2)):
                top2 = outs[hi].topk(2, dim=1).indices.cpu().numpy()
                TOP2[k].append(top2)
    T_ = [np.concatenate(a) for a in T_]
    P_ = [np.concatenate(a) for a in P_]
    TOP2 = [np.concatenate(a) for a in TOP2]

    heads = ["gender", "upper", "lower", "sleeve"]
    palettes = [v4.GENDERS, v4.COLORS, v4.COLORS, v4.SLEEVES]

    res = {"n_val": len(va), "weights": os.path.basename(WEIGHTS), "heads": {}}
    exact = np.ones(len(T_[0]), dtype=bool)
    for hi, (name, pal) in enumerate(zip(heads, palettes)):
        acc = float((T_[hi] == P_[hi]).mean())
        exact &= T_[hi] == P_[hi]
        cm = np.zeros((len(pal), len(pal)), dtype=int)
        for t, p in zip(T_[hi], P_[hi]):
            cm[t, p] += 1
        rows, macro_f1, mA = per_class_metrics(cm)
        res["heads"][name] = {
            "accuracy": acc, "macro_f1": macro_f1, "mA": mA,
            "classes": {c: rows[i] for i, c in enumerate(pal)},
        }
        np.save(os.path.join(OUT, f"cm_{name}.npy"), cm)
        print(f"  {name:6s}  acc {acc:.3f}  macro-F1 {macro_f1:.3f}  mA {mA:.3f}")

    res["mean_accuracy"] = float(np.mean([res["heads"][h]["accuracy"] for h in heads]))
    res["exact_match_4attr"] = float(exact.mean())
    for k, name in enumerate(("upper", "lower")):
        hi = k + 1
        top2acc = float(np.mean([t in row for t, row in zip(T_[hi], TOP2[k])]))
        res["heads"][name]["top2_accuracy"] = top2acc
        print(f"  {name} top-2 acc {top2acc:.3f}")
    print(f"  mean acc {res['mean_accuracy']:.3f}   exact match(4속성) {res['exact_match_4attr']:.3f}")

    def clean(o):
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    with open(os.path.join(OUT, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=2)
    print(f"[metrics] 저장 -> {OUT}/val_metrics.json, cm_*.npy")


if __name__ == "__main__":
    main()
