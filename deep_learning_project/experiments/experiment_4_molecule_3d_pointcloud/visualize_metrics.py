#!/usr/bin/env python3
"""Create a PNG metrics dashboard for experiment 4."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    venv_python = project_root / ".venv" / "bin" / "python"
    already_retried = os.environ.get("VISUALIZE_METRICS_RETRIED_VENV") == "1"
    if venv_python.exists() and not already_retried:
        os.environ["VISUALIZE_METRICS_RETRIED_VENV"] = "1"
        os.execv(str(venv_python), [str(venv_python), str(script_path), *sys.argv[1:]])
    raise ModuleNotFoundError(
        "Pillow is required to generate the PNG. Run this script with "
        "`.venv/bin/python experiments/experiment_4_molecule_3d_pointcloud/visualize_metrics.py` "
        "or install Pillow into your current Python environment."
    )


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "metrics.json"
OUTPUT_PATH = ROOT / "metrics_visualization.png"

WIDTH = 1440
HEIGHT = 1040
MARGIN = 60

COLORS = {
    "ink": "#17202a",
    "muted": "#5c6670",
    "grid": "#d9dee5",
    "panel": "#ffffff",
    "panel_alt": "#f9fafb",
    "bg": "#f6f7f9",
    "train": "#2563eb",
    "val": "#059669",
    "test": "#dc2626",
    "amber": "#d97706",
    "teal": "#0f766e",
    "pink": "#be185d",
    "purple": "#7c3aed",
    "border": "#cfd6df",
    "white": "#ffffff",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {
    "title": font(34, True),
    "subtitle": font(18, True),
    "body": font(16),
    "panel": font(20, True),
    "small": font(12),
    "small_bold": font(13, True),
    "metric": font(28, True),
    "count": font(24, True),
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    fill: str = COLORS["ink"],
    font_key: str = "body",
    anchor: str = "la",
) -> None:
    draw.text(xy, body, fill=fill, font=FONTS[font_key], anchor=anchor)


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    fill: str,
    outline: str | None = None,
    radius: int = 8,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str) -> None:
    rounded_rect(draw, (x, y, x + w, y + h), COLORS["panel"], COLORS["border"], radius=8)
    draw_text(draw, (x + 22, y + 26), title, font_key="panel")


def scale(value: float, vmin: float, vmax: float, low: float, high: float) -> float:
    if vmax == vmin:
        return (low + high) / 2
    return low + (value - vmin) * (high - low) / (vmax - vmin)


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int = 3) -> None:
    draw.line(points, fill=color, width=width, joint="curve")


def draw_final_metric_bars(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 60, 145, 640, 300
    panel(draw, x, y, w, h, "Final Metrics by Split")
    metrics_to_plot = [
        ("Acc", "accuracy"),
        ("Bal", "balanced_accuracy"),
        ("Prec", "precision"),
        ("Rec", "recall"),
        ("Spec", "specificity"),
        ("F1", "f1"),
        ("ROC", "roc_auc"),
        ("PR", "pr_auc"),
    ]
    splits = ["train", "val", "test"]
    colors = [COLORS["train"], COLORS["val"], COLORS["test"]]
    plot_x, plot_y, plot_w, plot_h = x + 62, y + 62, w - 90, h - 120

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        ty = plot_y + plot_h - tick * plot_h
        draw.line((plot_x, ty, plot_x + plot_w, ty), fill=COLORS["grid"], width=1)
        draw_text(draw, (plot_x - 12, ty), f"{tick:.2f}", fill=COLORS["muted"], font_key="small", anchor="ra")

    group_w = plot_w / len(metrics_to_plot)
    bar_w = group_w / 4
    for i, (label, metric) in enumerate(metrics_to_plot):
        base_x = plot_x + i * group_w + group_w * 0.18
        for j, split in enumerate(splits):
            value = float(metrics[split][metric])
            bh = value * plot_h
            bx = base_x + j * bar_w
            by = plot_y + plot_h - bh
            rounded_rect(draw, (bx, by, bx + bar_w * 0.85, plot_y + plot_h), colors[j], radius=2)
        draw_text(
            draw,
            (plot_x + i * group_w + group_w / 2, plot_y + plot_h + 16),
            label,
            fill=COLORS["muted"],
            font_key="small",
            anchor="ma",
        )

    legend_x = x + w - 210
    for j, split in enumerate(splits):
        lx = legend_x + j * 68
        rounded_rect(draw, (lx, y + 21, lx + 13, y + 34), colors[j], radius=2)
        draw_text(draw, (lx + 18, y + 21), split, fill=COLORS["muted"], font_key="small")


def draw_learning_curves(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 740, 145, 640, 300
    panel(draw, x, y, w, h, "Validation Learning Curves")
    history = metrics["history"]
    series = {
        "ROC-AUC": ("val_roc_auc", COLORS["train"]),
        "PR-AUC": ("val_pr_auc", COLORS["teal"]),
        "F1": ("val_f1", COLORS["pink"]),
        "Balanced Acc.": ("val_balanced_accuracy", COLORS["amber"]),
    }
    plot_x, plot_y, plot_w, plot_h = x + 62, y + 62, w - 100, h - 120

    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ty = scale(tick, 0.5, 0.95, plot_y + plot_h, plot_y)
        draw.line((plot_x, ty, plot_x + plot_w, ty), fill=COLORS["grid"], width=1)
        draw_text(draw, (plot_x - 12, ty), f"{tick:.1f}", fill=COLORS["muted"], font_key="small", anchor="ra")

    epochs = [int(item["epoch"]) for item in history]
    for _, (key, color) in series.items():
        pts = [
            (
                scale(int(item["epoch"]), min(epochs), max(epochs), plot_x, plot_x + plot_w),
                scale(float(item[key]), 0.5, 0.95, plot_y + plot_h, plot_y),
            )
            for item in history
        ]
        draw_polyline(draw, pts, color)
        for px, py in pts:
            draw.ellipse((px - 3.5, py - 3.5, px + 3.5, py + 3.5), fill=color)

    for epoch in [1, 4, 8, 12]:
        tx = scale(epoch, min(epochs), max(epochs), plot_x, plot_x + plot_w)
        draw_text(draw, (tx, plot_y + plot_h + 16), str(epoch), fill=COLORS["muted"], font_key="small", anchor="ma")
    draw_text(draw, (plot_x + plot_w / 2, y + h - 30), "Epoch", fill=COLORS["muted"], font_key="small", anchor="ma")

    legend_x = x + 358
    for i, (label, (_, color)) in enumerate(series.items()):
        lx = legend_x + (i % 2) * 120
        ly = y + 24 + (i // 2) * 20
        draw.line((lx, ly - 4, lx + 22, ly - 4), fill=color, width=3)
        draw_text(draw, (lx + 28, ly - 11), label, fill=COLORS["muted"], font_key="small")


def draw_loss(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 60, 490, 420, 260
    panel(draw, x, y, w, h, "Training Loss")
    history = metrics["history"]
    losses = [float(item["train_loss"]) for item in history]
    epochs = [int(item["epoch"]) for item in history]
    ymin = min(losses) * 0.98
    ymax = max(losses) * 1.02
    plot_x, plot_y, plot_w, plot_h = x + 62, y + 60, w - 95, h - 110

    for tick in [ymin, (ymin + ymax) / 2, ymax]:
        ty = scale(tick, ymin, ymax, plot_y + plot_h, plot_y)
        draw.line((plot_x, ty, plot_x + plot_w, ty), fill=COLORS["grid"], width=1)
        draw_text(draw, (plot_x - 12, ty), f"{tick:.3f}", fill=COLORS["muted"], font_key="small", anchor="ra")

    pts = [
        (
            scale(epoch, min(epochs), max(epochs), plot_x, plot_x + plot_w),
            scale(loss, ymin, ymax, plot_y + plot_h, plot_y),
        )
        for epoch, loss in zip(epochs, losses)
    ]
    draw_polyline(draw, pts, COLORS["purple"], width=4)
    draw_text(draw, (plot_x, y + h - 38), f"Start {losses[0]:.3f}", fill=COLORS["muted"], font_key="small")
    draw_text(draw, (plot_x + plot_w, y + h - 38), f"End {losses[-1]:.3f}", fill=COLORS["muted"], font_key="small", anchor="ra")


def draw_confusion(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 510, 490, 420, 260
    panel(draw, x, y, w, h, "Test Confusion Matrix")
    test = metrics["test"]
    cells = [
        ("TN", int(test["tn"]), x + 78, y + 75, COLORS["teal"]),
        ("FP", int(test["fp"]), x + 220, y + 75, COLORS["amber"]),
        ("FN", int(test["fn"]), x + 78, y + 158, COLORS["amber"]),
        ("TP", int(test["tp"]), x + 220, y + 158, COLORS["teal"]),
    ]
    max_count = max(count for _, count, *_ in cells)
    for label, count, cx, cy, color in cells:
        alpha = int(255 * (0.22 + 0.58 * (count / max_count)))
        fill = Image.new("RGBA", (122, 64), color + f"{alpha:02x}")
        draw.bitmap((cx, cy), fill)
        draw.rounded_rectangle((cx, cy, cx + 122, cy + 64), radius=6, outline=COLORS["white"], width=2)
        draw_text(draw, (cx + 61, cy + 12), label, font_key="small_bold", anchor="ma")
        draw_text(draw, (cx + 61, cy + 35), str(count), font_key="count", anchor="ma")
    draw_text(draw, (x + 141, y + 222), "Predicted inactive", fill=COLORS["muted"], font_key="small", anchor="ma")
    draw_text(draw, (x + 282, y + 222), "Predicted active", fill=COLORS["muted"], font_key="small", anchor="ma")
    draw_text(draw, (x + 20, y + 110), "Actual inactive", fill=COLORS["muted"], font_key="small")
    draw_text(draw, (x + 20, y + 193), "Actual active", fill=COLORS["muted"], font_key="small")


def draw_data_summary(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 960, 490, 420, 260
    panel(draw, x, y, w, h, "Dataset and Split Summary")
    active = int(metrics["n_active"])
    inactive = int(metrics["n_inactive"])
    total = int(metrics["n_rows"])
    bar_x, bar_y, bar_w, bar_h = x + 32, y + 72, w - 64, 32
    active_w = bar_w * active / total
    rounded_rect(draw, (bar_x, bar_y, bar_x + active_w, bar_y + bar_h), COLORS["teal"], radius=5)
    rounded_rect(draw, (bar_x + active_w, bar_y, bar_x + bar_w, bar_y + bar_h), COLORS["amber"], radius=5)
    draw_text(draw, (bar_x, bar_y + 55), f"Active: {active} ({active / total:.1%})", fill=COLORS["teal"], font_key="small_bold")
    draw_text(draw, (bar_x + bar_w, bar_y + 55), f"Inactive: {inactive} ({inactive / total:.1%})", fill=COLORS["amber"], font_key="small_bold", anchor="ra")

    split_summary = metrics["split_summary"]
    splits = [
        ("train", split_summary["train_rows"], COLORS["train"]),
        ("val", split_summary["val_rows"], COLORS["val"]),
        ("test", split_summary["test_rows"], COLORS["test"]),
    ]
    sx, sy = bar_x, y + 168
    for label, count, color in splits:
        bw = bar_w * count / total
        rounded_rect(draw, (sx, sy, sx + bw, sy + 30), color, radius=4)
        draw_text(draw, (sx + bw / 2, sy + 8), f"{label} {count}", fill=COLORS["white"], font_key="small_bold", anchor="ma")
        sx += bw
    draw_text(draw, (bar_x, y + 226), f"Total examples: {total}", fill=COLORS["muted"], font_key="small")
    draw_text(draw, (bar_x + bar_w, y + 226), f"Conformer failures: {int(metrics['n_conformer_failures'])}", fill=COLORS["muted"], font_key="small", anchor="ra")


def draw_key_findings(draw: ImageDraw.ImageDraw, metrics: dict) -> None:
    x, y, w, h = 60, 795, 1320, 180
    test = metrics["test"]
    val = metrics["val"]
    panel(draw, x, y, w, h, "Key Readout")
    cards = [
        ("Test ROC-AUC", fmt(test["roc_auc"]), "Ranking quality is moderate.", COLORS["train"]),
        ("Test PR-AUC", fmt(test["pr_auc"]), "High on active-heavy data.", COLORS["teal"]),
        ("Test F1", fmt(test["f1"]), "Positive detection is strong.", COLORS["pink"]),
        ("Test Specificity", pct(test["specificity"]), "Inactive separation is weak.", COLORS["amber"]),
        ("Best Val ROC-AUC", fmt(max(item["val_roc_auc"] for item in metrics["history"])), f"Final val ROC-AUC: {fmt(val['roc_auc'])}.", COLORS["purple"]),
    ]
    card_w = (w - 70) / 5
    for i, (label, value, note, color) in enumerate(cards):
        cx = x + 22 + i * (card_w + 8)
        cy = y + 58
        rounded_rect(draw, (cx, cy, cx + card_w, cy + 86), COLORS["panel_alt"], "#dce2ea", radius=6)
        draw_text(draw, (cx + 16, cy + 15), label, fill=COLORS["muted"], font_key="small_bold")
        draw_text(draw, (cx + 16, cy + 39), value, fill=color, font_key="metric")
        draw_text(draw, (cx + 16, cy + 70), note, fill=COLORS["muted"], font_key="small")


def draw_dashboard(metrics: dict) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    experiment = metrics["experiment"].replace("_", " ")
    question = metrics["question"]

    draw_text(draw, (MARGIN, 35), "Metrics Dashboard", font_key="title")
    draw_text(draw, (MARGIN, 82), experiment, fill=COLORS["muted"], font_key="subtitle")
    draw_text(draw, (MARGIN, 112), question, fill=COLORS["muted"], font_key="body")

    draw_final_metric_bars(draw, metrics)
    draw_learning_curves(draw, metrics)
    draw_loss(draw, metrics)
    draw_confusion(draw, metrics)
    draw_data_summary(draw, metrics)
    draw_key_findings(draw, metrics)
    return image


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    image = draw_dashboard(metrics)
    image.save(OUTPUT_PATH, "PNG")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
