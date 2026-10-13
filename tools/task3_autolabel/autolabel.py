#!/usr/bin/env python3
"""Task3 auto-labeling script.

Extracts frames from drone videos (downward-facing camera) and produces
bounding-box labels for the colored circular pads ("Rag") and the colored
cans ("Tin") placed on them, using the same six classes as the Task3 YOLO
detector (``tin_detector.hef`` / ``tin_detective.py``). Output follows the YOLO text format (one ``.txt``
per frame, ``class cx cy w h`` normalized to [0, 1]).

Input files are never modified. Every output goes under ``--out`` (default
``labels_out/`` next to the current directory):

    <out>/images/<video_stem>_<frame:06d>.jpg
    <out>/labels/<video_stem>_<frame:06d>.txt
    <out>/classes.txt
    <out>/data.yaml                               (Ultralytics-style, same class order)
    <out>/preview/<video_stem>_<frame:06d>.jpg   (only with --preview)

Usage:
    python3 autolabel.py VIDEO_OR_DIR [VIDEO_OR_DIR ...] --out labels_out --fps 2 --tins

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
# Class definitions. Order = class id in YOLO labels and MUST match data.yaml
# of the Task3 detector (NAMES in tin_detective.py). Rag = colored pad on the
# ground, Tin = colored can that is picked up / delivered.
# --------------------------------------------------------------------------
CLASSES = [
    "Blue_Rag",    # 0
    "Blue_Tin",    # 1
    "Green_Rag",   # 2
    "Green_Tin",   # 3
    "Red_Rag",     # 4
    "Red_Tin",     # 5
]
COLORS = ("Red", "Green", "Blue")

# HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255). Red wraps around 0.
# The pads are fully saturated paint (S ~ 255), while pavers, shadows and the
# grey board stay below S ~ 225 and V ~ 120, so high S/V floors reject them.
HSV_RANGES = {
    "Red": [((0, 200, 110), (8, 255, 255)), ((170, 200, 110), (179, 255, 255))],
    "Green": [((45, 200, 110), (80, 255, 255))],
    "Blue": [((100, 200, 100), (130, 255, 255))],
}
# Hue bands used to name a tin from its pixels (no V floor: dark/shaded cans).
TIN_HUE = {"Red": [(0, 12), (165, 179)], "Green": [(40, 90), ], "Blue": [(95, 135)]}


def cls_id(color: str, kind: str) -> int:
    return CLASSES.index(f"{color}_{kind}")


def tin_color(hsv: np.ndarray, mask: np.ndarray) -> str | None:
    """Name the can color by voting its saturated pixels into the hue bands.

    A vote (not a median) because red hue wraps around 0/179: the median of
    a red can's pixels can land in the middle of the range and read as blue.
    """
    # Shrink the blob so the anti-aliased rim of the pad does not vote.
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    px = hsv[core > 0] if np.count_nonzero(core) >= 10 else hsv[mask > 0]
    px = px[px[:, 1] >= 100]
    if len(px) < 10:
        return None
    hue = px[:, 0]
    votes = {color: int(sum(((hue >= lo) & (hue <= hi)).sum() for lo, hi in bands))
             for color, bands in TIN_HUE.items()}
    color, n = max(votes.items(), key=lambda kv: kv[1])
    return color if n >= 0.4 * len(px) else None


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


def detect_pads(frame: np.ndarray, args: argparse.Namespace) -> tuple[list[tuple[Box, np.ndarray]], list[Box]]:
    """Return ([(pad_box, filled_mask)], [free-standing tin boxes])."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    pads: list[tuple[Box, np.ndarray]] = []
    tins: list[Box] = []
    min_pad = args.min_pad_area * w * h
    max_tin = args.max_tin_frac * w * h
    for color in COLORS:
        mask = color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_pad:
                # Small saturated blob of a target color lying on the board or
                # ground = a can that is not on a pad (visible if it is bright).
                if args.free_tins and args.min_tin_area <= area <= max_tin:
                    x, y, bw, bh = cv2.boundingRect(c)
                    ratio = bw / max(bh, 1)
                    if 0.4 <= ratio <= 2.5:
                        tins.append(Box(cls_id(color, "Tin"), x, y, x + bw, y + bh))
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
            need = args.min_circularity * (0.75 if touches_border else 1.0)
            if circularity < need or solidity < 0.80:
                continue
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
            pads.append((Box(cls_id(color, "Rag"), x, y, x + bw, y + bh, float(circularity)), filled))
    return pads, tins


def hue_in_band(hsv: np.ndarray, color: str) -> np.ndarray:
    h = hsv[:, :, 0]
    m = np.zeros(h.shape, bool)
    for lo, hi in TIN_HUE[color]:
        m |= (h >= lo) & (h <= hi)
    return m


def detect_tins_on_pad(hsv: np.ndarray, circle_mask: np.ndarray, circle_color_mask: np.ndarray,
                       pad_color: str, min_area: int, max_area_frac: float,
                       shadow: "ShadowEstimate") -> list[Box]:
    """Tins on a pad = regions inside the pad hull that are NOT the pad color.

    A can casts a shadow on the pad, and the shadow is as dark as a dark can,
    so the hole is can + shadow. Pixels whose hue differs from the pad's are
    certainly the can; the rest (pad hue, dark) is shadow unless the can has
    the pad's color. For a same-color can the shadow direction learned from
    the other cans in this video says which end of the hole is the can.
    """
    # Erode by ~6 % of the circle radius so the rim and anti-aliased edge are ignored.
    radius = int(np.sqrt(np.count_nonzero(circle_mask) / np.pi))
    k = max(5, 2 * int(0.06 * radius) + 1)
    inner = cv2.erode(circle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    holes = cv2.bitwise_and(inner, cv2.bitwise_not(circle_color_mask))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    circle_area = int(np.count_nonzero(circle_mask))
    pad_hue = hue_in_band(hsv, pad_color)
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    deferred = []   # same-color cans, resolved once the shadow direction is known
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area_frac * circle_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        perim = cv2.arcLength(c, True)
        if not 0.3 <= bw / max(bh, 1) <= 3.0 or (perim > 0 and 4 * np.pi * area / (perim * perim) < 0.4):
            continue  # elongated or ragged = leg shadow / board seam crossing the pad, not a can
        blob = np.zeros_like(holes)
        cv2.drawContours(blob, [c], -1, 255, thickness=-1)
        ring = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & ~inner
        if np.count_nonzero(ring) > 0.15 * perim:
            continue  # hole touches the pad rim = cut-off edge, not a can
        color = tin_color(hsv, blob)
        if color is None:
            continue
        # Pixels that are not pad-colored are the can for sure.
        distinct = (blob > 0) & ~pad_hue & (hsv[:, :, 1] >= 100)
        distinct_u8 = cv2.morphologyEx(distinct.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n_distinct = int(np.count_nonzero(distinct_u8))
        if color != pad_color and n_distinct >= max(min_area // 2, 10):
            can = largest_component(distinct_u8)
            rest = (blob > 0) & (can == 0)
            cy_c, cx_c = np.argwhere(can > 0).mean(axis=0)
            if np.count_nonzero(rest) >= 0.3 * np.count_nonzero(can):
                ry, rx = np.argwhere(rest).mean(axis=0)
                shadow.add(rx - cx_c, ry - cy_c)
            x, y, bw, bh = cv2.boundingRect(can)
            boxes.append(Box(cls_id(color, "Tin"), x, y, x + bw, y + bh))
        else:
            deferred.append((color, blob))
    for color, blob in deferred:
        x, y, bw, bh = cv2.boundingRect(blob)
        d = shadow.direction()
        if d is not None and max(bw, bh) > 1.25 * min(bw, bh):
            # Keep the end of the blob that faces away from the shadow, one can-width long.
            pts = np.argwhere(blob > 0)[:, ::-1].astype(np.float32)        # (x, y)
            proj = pts @ d
            side = float(min(bw, bh))
            keep = pts[proj <= proj.min() + side]
            x, y = int(keep[:, 0].min()), int(keep[:, 1].min())
            bw, bh = int(keep[:, 0].max()) - x + 1, int(keep[:, 1].max()) - y + 1
        boxes.append(Box(cls_id(color, "Tin"), x, y, x + bw, y + bh))
    return boxes


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255


class ShadowEstimate:
    """Running unit vector from a can to its shadow (constant within a video: one sun)."""

    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.n = 0

    def add(self, dx: float, dy: float) -> None:
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return
        # Exponential average so a long video follows a slowly moving sun.
        w = 0.1 if self.n else 1.0
        self.vx = (1 - w) * self.vx + w * dx / norm
        self.vy = (1 - w) * self.vy + w * dy / norm
        self.n += 1

    def direction(self) -> np.ndarray | None:
        norm = float(np.hypot(self.vx, self.vy))
        if self.n == 0 or norm < 0.3:
            return None
        return np.array([self.vx / norm, self.vy / norm], np.float32)


def merge_pads(pads: list[tuple[Box, np.ndarray]]) -> list[tuple[Box, np.ndarray]]:
    """Union same-color pad fragments whose boxes overlap or nearly touch (a pad split by a leg shadow)."""
    merged: list[tuple[Box, np.ndarray]] = []
    for box, filled in sorted(pads, key=lambda p: -(p[0].x2 - p[0].x1) * (p[0].y2 - p[0].y1)):
        for i, (m, mf) in enumerate(merged):
            pad = 0.2 * max(m.x2 - m.x1, m.y2 - m.y1)   # tolerance: 20 % of the larger fragment
            if m.cls == box.cls and box.x1 < m.x2 + pad and box.x2 > m.x1 - pad \
                    and box.y1 < m.y2 + pad and box.y2 > m.y1 - pad:
                merged[i] = (Box(m.cls, min(m.x1, box.x1), min(m.y1, box.y1),
                                 max(m.x2, box.x2), max(m.y2, box.y2), max(m.score, box.score)),
                             mf | filled)
                break
        else:
            merged.append((box, filled))
    return merged


def apply_masks(frame: np.ndarray, masks: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Black out fixed obstructions (e.g. landing gear in the corners) before detection."""
    if not masks:
        return frame
    h, w = frame.shape[:2]
    frame = frame.copy()
    for x1, y1, x2, y2 in masks:
        frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
    return frame


def label_frame(frame: np.ndarray, args: argparse.Namespace, shadow: ShadowEstimate) -> list[Box]:
    frame = apply_masks(frame, args.mask)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    pads, free_tins = detect_pads(frame, args)
    pads = merge_pads(pads)
    boxes: list[Box] = []
    for box, filled in pads:
        boxes.append(box)
        if args.tins:
            pad_color = CLASSES[box.cls].split("_")[0]
            cmask = color_mask(hsv, pad_color)
            boxes.extend(detect_tins_on_pad(hsv, filled, cmask, pad_color, args.min_tin_area,
                                            args.max_tin_frac_pad, shadow))
    if args.free_tins:
        pad_union = np.zeros(frame.shape[:2], np.uint8)
        for _, filled in pads:
            pad_union |= filled
        for t in free_tins:
            if pad_union[(t.y1 + t.y2) // 2, (t.x1 + t.x2) // 2] == 0:
                boxes.append(t)
    return boxes


def draw(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
    vis = frame.copy()
    for b in boxes:
        name = CLASSES[b.cls]
        col = colors[name.split("_")[0]]
        thick = 2 if name.endswith("_Rag") else 1
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), col, thick)
        cv2.putText(vis, name, (b.x1, max(12, b.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
    shadow = ShadowEstimate()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            stats["frames_seen"] += 1
            boxes = label_frame(frame, args, shadow)
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
    ap.add_argument("--tins", action="store_true", help="also label the cans (*_Tin) sitting on pads; otherwise pads (*_Rag) only")
    ap.add_argument("--free-tins", action="store_true",
                    help="also label cans NOT on a pad (small saturated blobs). Off by default: red pavers in sunlight get labeled Red_Tin")
    ap.add_argument("--preview", action="store_true", help="write annotated preview images for QA")
    ap.add_argument("--keep-empty", action="store_true", help="also write frames with no detections")
    ap.add_argument("--mask", nargs="*", default=[], metavar="X1,Y1,X2,Y2",
                    help="normalized rectangles to ignore, e.g. landing gear: --mask 0,0,0.1,0.1 0.9,0,1,0.1")
    ap.add_argument("--min-pad-area", type=float, default=0.002,
                    help="min pad (Rag) area as fraction of the frame (default 0.002)")
    ap.add_argument("--min-circularity", type=float, default=0.6)
    ap.add_argument("--min-tin-area", type=int, default=40, help="min tin area in pixels")
    ap.add_argument("--max-tin-frac-pad", type=float, default=0.25,
                    help="max tin area as fraction of the pad it sits on")
    ap.add_argument("--max-tin-frac", type=float, default=0.004,
                    help="max area of a tin not on a pad, as fraction of the frame")
    args = ap.parse_args(argv)
    try:
        args.mask = [tuple(float(v) for v in m.split(",")) for m in args.mask]
        assert all(len(m) == 4 for m in args.mask)
    except (ValueError, AssertionError):
        ap.error("--mask entries must be X1,Y1,X2,Y2 with values in [0, 1]")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\nnc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    summary = [process_video(v, args) for v in iter_videos(args.inputs)]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
