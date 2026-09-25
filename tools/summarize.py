import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
OUT = os.path.join(RESULTS, "summary")

LOGS = [("v1", "v1_market_resnet18"), ("v2", "v2_peta_resnet50"),
        ("v3", "v3_multi_resnet50"), ("v3 (swin_t)", "v3_multi_swin_t"),
        ("v5", "v5_multi_resnet50_aug")]
HEADS = ["gender", "upper", "lower", "sleeve"]
COLORS = ["#E45756", "#F58518", "#54A24B", "#B279A2", "#4C78A8"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def f4(x):
    return f"{x:.4f}"


def parse_log(name):
    text = open(os.path.join(RESULTS, "train_logs", f"{name}.log"),
                encoding="utf-8").read()
    info = {}
    m = re.search(r"(?:이미지|images) (\d+)", text)
    info["images"] = int(m.group(1))
    m = re.search(r"train (\d+) / val (\d+)(?: \(인물 (\d+)명\))?", text)
    info["train"], info["val"] = int(m.group(1)), int(m.group(2))
    info["persons"] = int(m.group(3)) if m.group(3) else None
    m = re.search(r"identity (\d+)", text)
    if m:
        info["persons"] = int(m.group(1))
    epochs = []
    for line in text.splitlines():
        m = re.match(r"\s*ep\s+(\d+)\s+(.*)", line)
        if m:
            vals = dict(re.findall(r"(\w+) ([0-9.]+)", m.group(2)))
            epochs.append((int(m.group(1)), vals))
    info["epochs"] = epochs
    info["best"] = re.search(r"best mean_acc=([0-9.]+)", text).group(1)
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    logs = {label: parse_log(name) for label, name in LOGS}
    sections = []

    rows = []
    for label, name in LOGS:
        i = logs[label]
        rows.append([label, f"`{name}.log`", i["images"], i["train"], i["val"],
                     "–" if i["persons"] is None else i["persons"], len(i["epochs"])])
    sections.append(("L-a Data split per training run", md_table(
        ["version", "log", "images", "train", "val", "persons", "epochs"], rows)))

    rows = []
    for label, _ in LOGS:
        i = logs[label]
        tied = [e for e in i["epochs"] if e[1]["mean"] == i["best"]]

        def get(*ks):
            return " / ".join(next((v[k] for k in ks if k in v), "–") for _, v in tied)
        rows.append([label, i["best"], ", ".join(str(e) for e, _ in tied),
                     get("gender", "gen"), get("upper", "up"), get("lower", "low"),
                     get("slv")])
    sections.append(("L-b Epochs with the best printed validation mean",
                     md_table(["version", "best mean", "epoch", "gender", "upper",
                               "lower", "sleeve"], rows)))

    v4 = json.load(open(os.path.join(RESULTS, "v4_val", "val_metrics.json"),
                        encoding="utf-8"))
    rows = []
    for h in HEADS:
        d = v4["heads"][h]
        rows.append([h, f4(d["accuracy"]), f4(d["macro_f1"]), f4(d["mA"]),
                     f4(d["top2_accuracy"]) if "top2_accuracy" in d else "–"])
    rows.append(["mean of 4", f4(v4["mean_accuracy"]), "–", "–", "–"])
    rows.append(["all 4 correct", f4(v4["exact_match_4attr"]), "–", "–", "–"])
    sections.append((f"L-c v4 on the validation split ({v4['n_val']} images)",
                     md_table(["head", "accuracy", "macro-F1", "mA", "top-2 accuracy"],
                              rows)))

    for h in ("upper", "lower"):
        cls = v4["heads"][h]["classes"]
        rows = [[c, v["support"], *(("–" if v[k] is None else f4(v[k]))
                                    for k in ("precision", "recall", "f1"))]
                for c, v in cls.items()]
        sections.append((f"L-{'d' if h == 'upper' else 'e'} v4 per-class metrics, "
                         f"{h} color (validation split)",
                         md_table(["class", "support", "precision", "recall", "F1"], rows)))

    test = list(csv.DictReader(open(os.path.join(RESULTS, "test15.csv"),
                                    encoding="utf-8")))
    rows, means = [], []
    for r in test:
        n = int(r["images"])
        cnt = [round(float(r[h]) * n) if r[h] else None for h in HEADS]
        used = [c for c in cnt if c is not None]
        mean = sum(used) / (n * len(used))
        means.append(mean)
        rows.append([r["version"], r["setup"],
                     *(("–" if c is None else f"{c}/{n}") for c in cnt),
                     len(used), f4(mean)])
    sections.append(("L-f Own test photos (15 images, as recorded; counts = value × 15)",
                     md_table(["version", "setup", *HEADS, "heads", "mean"], rows)))

    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8",
              newline="\n") as f:
        for title, table in sections:
            f.write(f"### {title}\n\n{table}\n\n")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (label, _), c in zip(LOGS, COLORS):
        e = logs[label]["epochs"]
        ax.plot([x for x, _ in e], [float(v["mean"]) for _, v in e], color=c,
                lw=1.4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mean accuracy")
    ax.set_title("Validation mean accuracy per epoch (training logs)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "val_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(test))
    width = 0.2
    for k, (h, c) in enumerate(zip(HEADS, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"])):
        vals = [round(float(r[h]) * int(r["images"])) / int(r["images"]) if r[h] else 0
                for r in test]
        ax.bar(x + (k - 1.5) * width, vals, width, color=c, label=h, zorder=3)
    ax.plot(x, means, "k_", ms=22, mew=2, label="mean", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["version"] for r in test])
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("accuracy (15 images)")
    ax.set_title("Own test photos per version", fontsize=10)
    ax.legend(fontsize=7, ncol=5, loc="upper center")
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "test15.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, h in zip(axes, ("upper", "lower")):
        cm = np.load(os.path.join(RESULTS, "v4_val", f"cm_{h}.npy"))
        labels = list(v4["heads"][h]["classes"])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                            color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"v4 {h} color (validation, row-normalized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_confusion.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
