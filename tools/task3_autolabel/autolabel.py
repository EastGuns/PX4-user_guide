#!/usr/bin/env python3
"""Task3 auto-labeling script.

Extracts frames from drone videos (downward-facing camera) and produces
bounding-box labels for the colored circular ground targets and the small
objects placed on them. Output follows the YOLO text format (one ``.txt``
per frame, ``class cx cy w h`` normalized to [0, 1]).

Input files are never modified. Every output goes under ``--out`` (default
``labels_out/`` next to the current directory):

    <out>/images/<video_stem>_<frame:06d>.jpg
    <out>/labels/<video_stem>_<frame:06d>.txt
    <out>/classes.txt
    <out>/preview/<video_stem>_<frame:06d>.jpg   (only with --preview)

Usage:
    python3 autolabel.py VIDEO_OR_DIR [VIDEO_OR_DIR ...] --out labels_out --fps 2

Only numpy and opencv-python are required:
    pip install numpy opencv-python-headless
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# --------------------------------------------------------------------------
# Class definitions. Order = class id in YOLO labels. Adjust to the official
# Task3 label rules once they are known (see README.md).
# --------------------------------------------------------------------------
CLASSES = [
    "red_circle",     # 0
    "green_circle",   # 1
    "blue_circle",    # 2
    "object",         # 3  small block placed on a circle
]

# HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255). Red wraps around 0.
# The targets are fully saturated paint (S ~ 255), while pavers, shadows and
# the grey board stay below S ~ 225 and V ~ 120, so high S/V floors reject them.
HSV_RANGES = {
    "red_circle": [((0, 200, 110), (8, 255, 255)), ((170, 200, 110), (179, 255, 255))],
    "green_circle": [((45, 200, 110), (80, 255, 255))],
    "blue_circle": [((100, 200, 100), (130, 255, 255))],
}


@dataclass
class Box:
    cls: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0

    def to_yolo(self, w: int, h: int) -> str:
        cx = (self.x1 + self.x2) / 2.0 / w
        cy = (self.y1 + self.y2) / 2.0 / h
        bw = (self.x2 - self.x1) / w
        bh = (self.y2 - self.y1) / h
        return f"{self.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def color_mask(hsv: np.ndarray, name: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[name]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_circles(frame: np.ndarray, min_area_frac: float, min_circularity: float) -> list[tuple[Box, np.ndarray]]:
    """Return (box, filled_mask) for every colored circle found."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    out: list[tuple[Box, np.ndarray]] = []
    for name in ("red_circle", "green_circle", "blue_circle"):
        cls = CLASSES.index(name)
        mask = color_mask(hsv, name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_frac * w * h:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            # Convexity + circularity of the hull: partially occluded circles
            # (objects on top, image border) still pass because we use the hull.
            perim = cv2.arcLength(hull, True)
            circularity = 4 * np.pi * hull_area / (perim * perim) if perim > 0 else 0
            solidity = area / hull_area
            x, y, bw, bh = cv2.boundingRect(hull)
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            # A circle cut by the frame edge is less round; relax (not drop) the test.
            need = min_circularity * (0.75 if touches_border else 1.0)
            if circularity < need or solidity < 0.80:
                continue
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
            out.append((Box(cls, x, y, x + bw, y + bh, float(circularity)), filled))
    return out


def detect_objects(frame: np.ndarray, circle_mask: np.ndarray, circle_color_mask: np.ndarray,
                   min_area: int, max_area_frac: float) -> list[Box]:
    """Objects = regions inside the circle hull that are NOT the circle color."""
    # Erode by ~6 % of the circle radius so the rim and anti-aliased edge are ignored.
    radius = int(np.sqrt(np.count_nonzero(circle_mask) / np.pi))
    k = max(5, 2 * int(0.06 * radius) + 1)
    inner = cv2.erode(circle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    holes = cv2.bitwise_and(inner, cv2.bitwise_not(circle_color_mask))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    circle_area = int(np.count_nonzero(circle_mask))
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area_frac * circle_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append(Box(CLASSES.index("object"), x, y, x + bw, y + bh))
    return boxes


def apply_masks(frame: np.ndarray, masks: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Black out fixed obstructions (e.g. landing gear in the corners) before detection."""
    if not masks:
        return frame
    h, w = frame.shape[:2]
    frame = frame.copy()
    for x1, y1, x2, y2 in masks:
        frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
    return frame


def label_frame(frame: np.ndarray, args: argparse.Namespace) -> list[Box]:
    frame = apply_masks(frame, args.mask)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    boxes: list[Box] = []
    for box, filled in detect_circles(frame, args.min_circle_area, args.min_circularity):
        boxes.append(box)
        if args.objects:
            cmask = color_mask(hsv, CLASSES[box.cls])
            boxes.extend(detect_objects(frame, filled, cmask, args.min_object_area, args.max_object_frac))
    return boxes


def draw(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}
    vis = frame.copy()
    for b in boxes:
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), colors.get(b.cls, (255, 255, 255)), 2)
        cv2.putText(vis, CLASSES[b.cls], (b.x1, max(12, b.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(b.cls, (255, 255, 255)), 1, cv2.LINE_AA)
    return vis


def iter_videos(inputs: list[str]):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS)
        elif p.suffix.lower() in VIDEO_EXTS:
            yield p
        else:
            print(f"skip (not a video): {p}", file=sys.stderr)


def process_video(video: Path, args: argparse.Namespace) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}", file=sys.stderr)
        return {"video": str(video), "error": "cannot open"}
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / args.fps))) if args.fps > 0 else 1
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    stats = {"video": str(video), "frames_seen": 0, "frames_written": 0, "boxes": 0,
             "per_class": {c: 0 for c in CLASSES}}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            stats["frames_seen"] += 1
            boxes = label_frame(frame, args)
            if boxes or args.keep_empty:
                stem = f"{video.stem}_{idx:06d}"
                h, w = frame.shape[:2]
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / f"{stem}.txt").write_text(
                    "".join(b.to_yolo(w, h) + "\n" for b in boxes), encoding="utf-8")
                if args.preview:
                    cv2.imwrite(str(out / "preview" / f"{stem}.jpg"), draw(frame, boxes))
                stats["frames_written"] += 1
                stats["boxes"] += len(boxes)
                for b in boxes:
                    stats["per_class"][CLASSES[b.cls]] += 1
        idx += 1
    cap.release()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video files or directories (searched recursively)")
    ap.add_argument("--out", default="labels_out", help="output directory (created; inputs are never touched)")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second to sample (0 = every frame)")
    ap.add_argument("--objects", action="store_true", help="also label small objects placed on the circles")
    ap.add_argument("--preview", action="store_true", help="write annotated preview images for QA")
    ap.add_argument("--keep-empty", action="store_true", help="also write frames with no detections")
    ap.add_argument("--mask", nargs="*", default=[], metavar="X1,Y1,X2,Y2",
                    help="normalized rectangles to ignore, e.g. landing gear: --mask 0,0,0.06,0.07 0.94,0,1,0.07")
    ap.add_argument("--min-circle-area", type=float, default=0.002,
                    help="min circle area as fraction of the frame (default 0.002)")
    ap.add_argument("--min-circularity", type=float, default=0.6)
    ap.add_argument("--min-object-area", type=int, default=40, help="min object area in pixels")
    ap.add_argument("--max-object-frac", type=float, default=0.25,
                    help="max object area as fraction of its circle")
    args = ap.parse_args(argv)
    try:
        args.mask = [tuple(float(v) for v in m.split(",")) for m in args.mask]
        assert all(len(m) == 4 for m in args.mask)
    except (ValueError, AssertionError):
        ap.error("--mask entries must be X1,Y1,X2,Y2 with values in [0, 1]")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    summary = [process_video(v, args) for v in iter_videos(args.inputs)]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
