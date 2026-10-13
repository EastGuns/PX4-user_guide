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
# The pads are fully saturated paint (S >= 244 in the middle of the frame), while
# pavers, shadows and blue-ish stains on the board stay below S ~ 220, so the
# strict mask (S >= 235) seeds pads. Toward the frame edge the lens desaturates a
# pad to S ~ 205, and the drone's shadow darkens it to V ~ 80, so once a pad is
# seeded its edge and shadowed parts are recovered with the relaxed floors
# below, but only inside the pad's own enclosing circle.
# A pad in the drone's shadow drops to V ~ 80, so once a pad is found with the
# strict floor its shadowed part is recovered with SHADOW_V_FLOOR, but only
# inside the pad's own enclosing circle (the shadowed board is blue-ish too).
HSV_RANGES = {
    "Red": [((0, 235, 110), (8, 255, 255)), ((170, 235, 110), (179, 255, 255))],
    "Green": [((45, 235, 110), (80, 255, 255))],
    "Blue": [((106, 235, 110), (130, 255, 255))],   # blue-ish board stains sit at H ~ 100
}
SHADOW_V_FLOOR = 60
RELAXED_S_FLOOR = 200

# The landing gear is red and pokes into the top corners when the camera tilts.
# The persistent part is also auto-masked (static_color_mask); this covers the rest.
DEFAULT_MASK = ("0,0,0.15,0.2", "0.85,0,1,0.2")
# The gear foot covers at most ~1 % of the frame (it is a fixed part of the airframe);
# a pad-colored blob in a masked corner that is larger than this is a pad.
GEAR_MAX_FRAC = 0.02
# The landing legs are black strips that run in from the frame border and end at a foot.
LEG_V_MAX = 40          # a leg pixel is darker than this (V)
LEG_MIN_FRAC = 0.003    # a leg is a dark component of at least this fraction of the frame

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


def color_mask(hsv: np.ndarray, name: str, v_floor: int | None = None, fill_holes: bool = True,
               s_floor: int | None = None) -> np.ndarray:
    """Pixels of one pad color. With fill_holes the mask is closed so a pad reads as one
    blob; without it, small cans on the pad stay as holes (needed to find them)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[name]:
        if v_floor is not None:
            lo = (lo[0], lo[1], v_floor)
        if s_floor is not None:
            lo = (lo[0], s_floor, lo[2])
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def detect_pads(frame: np.ndarray, args: argparse.Namespace, ignore: np.ndarray | None = None,
                gear_colors: set[str] | None = None) -> tuple[list[tuple[Box, np.ndarray]], list[Box]]:
    """Return ([(pad_box, filled_mask)], [free-standing tin boxes]).

    `ignore` marks regions holding the airframe (corner masks + auto mask): pixels
    there are never pulled into a pad as its shadowed part or as a split-off fragment,
    so the red landing-gear foot next to a red pad does not stretch the pad's box.
    The strict seed is left alone, so a pad that merely reaches into a corner stays whole.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    pads: list[tuple[Box, np.ndarray]] = []
    tins: list[Box] = []
    min_pad = args.min_pad_area * w * h
    max_tin = args.max_tin_frac * w * h
    neutral = (((hsv[:, :, 1] < 80) | (hsv[:, :, 2] < 60)).astype(np.uint8)) * 255
    for color in COLORS:
        mask = color_mask(hsv, color)
        relaxed = color_mask(hsv, color, SHADOW_V_FLOOR, s_floor=RELAXED_S_FLOOR)
        if ignore is not None and (not gear_colors or color in gear_colors):
            relaxed[ignore > 0] = 0   # a green pad fragment under the red foot is still pad
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            grown = None
            if area >= min_pad:
                # Cut thin bridges to blue-ish shadowed board before sizing the pad,
                # then recover the shadowed part of the pad: darker same-hue pixels,
                # but only inside the enclosing circle of what the strict mask found.
                blob = np.zeros_like(mask)
                cv2.drawContours(blob, [c], -1, 255, thickness=-1)
                k = max(7, int(0.08 * np.sqrt(area))) | 1
                core = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                core = largest_component(core) if np.count_nonzero(core) else blob
                cc, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                seed = max(cc, key=cv2.contourArea) if cc else c
                (ex, ey), er = cv2.minEnclosingCircle(seed)
                sx, sy, sw, sh = cv2.boundingRect(seed)
                if sx <= 1 or sy <= 1 or sx + sw >= w - 1 or sy + sh >= h - 1:
                    # Cut by the frame edge: the enclosing circle of the visible part is
                    # too small, so take the circle its outline actually lies on.
                    fit = fit_circle(cv2.convexHull(seed), w, h)
                    if fit is not None and fit[3] <= 0.04 and fit[2] <= max(w, h):
                        ex, ey, er = fit[:3]
                region = np.zeros_like(mask)
                cv2.circle(region, (int(ex), int(ey)), int(er * 1.05), 255, thickness=-1)
                grown = largest_component((relaxed & region) | core)
                # The drone's leg can cut a pad in two: pull in same-color fragments
                # that sit just outside the seed's enclosing circle (no bridge needed).
                near = np.zeros_like(mask)
                cv2.circle(near, (int(ex), int(ey)), int(er * 1.3), 255, thickness=-1)
                n_lab, labels, st, _ = cv2.connectedComponentsWithStats(relaxed & near, connectivity=8)
                for i in range(1, n_lab):
                    if st[i, cv2.CC_STAT_AREA] >= 0.02 * area and not np.any(grown[labels == i]):
                        grown[labels == i] = 255
                gc, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if gc:
                    # Fragments stay disconnected (the leg lies between them): one hull over all.
                    c = cv2.convexHull(np.vstack(gc))
                    grown_area = float(np.count_nonzero(grown))
                    if area < 0.45 * grown_area:
                        continue  # mostly relaxed pixels = a dim blotch, not a pad with a bright core
                    area = grown_area
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
            # The leg (black/grey) can lie across the pad: pixels it hides count as pad
            # when judging how well the blob fills its hull and its box.
            hull_fill = np.zeros_like(mask)
            cv2.drawContours(hull_fill, [hull], -1, 255, thickness=-1)
            occluded = hull_fill & neutral
            area = float(np.count_nonzero((grown if grown is not None else hull_fill & mask) | occluded))
            # Convexity + circularity of the hull: partially occluded circles
            # (objects on top, image border) still pass because we use the hull.
            perim = cv2.arcLength(hull, True)
            circularity = 4 * np.pi * hull_area / (perim * perim) if perim > 0 else 0
            solidity = area / hull_area
            x, y, bw, bh = cv2.boundingRect(hull)
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            if solidity < 0.80:
                continue
            # A circle cut by the frame edge is less round; relax the test, and a thin
            # arc at the edge (circularity far lower) passes if its outline really is
            # a circle of a plausible radius: a paving block or a stain cut by the edge
            # is not.
            need = args.min_circularity * (0.75 if touches_border else 1.0)
            if circularity < need and not (touches_border and circular_segment(hull, bw, bh, w, h)):
                continue
            # A disc fills ~78 % of its box; a blotch that passed the hull tests fills far less.
            if area / float(bw * bh) < 0.55:
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
                       shadow: "ShadowEstimate", ignore: np.ndarray | None = None) -> list[Box]:
    """Tins on a pad = small blobs inside the pad hull that are either of another
    hue than the pad or much darker than the pad's paint.

    A can casts a shadow on the pad, and the shadow is as dark as a dark can,
    so a blob is can + shadow. Pixels whose hue differs from the pad's are
    certainly the can; the rest (pad hue, dark) is shadow unless the can has
    the pad's color. For a same-color can the shadow direction learned from
    the other cans in this video says which end of the blob is the can.

    Brightness is judged relative to the pad's own sunlit paint, not with a
    fixed threshold: the defocused halo of a landing leg and washed-out paint
    near the frame edge are a bit darker and less saturated than the paint but
    far brighter than any can, so they are neither pad nor hole and never glue
    a can to a large region. The pad under the drone's or the leg's shadow is
    a large dark same-hue region; it is removed, but a compact blob inside it
    that is darker still is a can standing in that shadow.
    """
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    # Erode by ~6 % of the circle radius so the rim and anti-aliased edge are ignored.
    # The frame edge is not a rim: a pad cut by the frame keeps its pixels up to the edge,
    # so a can cut by the frame can still be found.
    circle_area = circle_area_px(circle_mask)
    radius = int(np.sqrt(circle_area / np.pi))
    k = max(5, 2 * int(0.06 * radius) + 1)
    padded = cv2.copyMakeBorder(circle_mask, k, k, k, k, cv2.BORDER_CONSTANT, value=255)
    inner = cv2.erode(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[k:-k, k:-k] > 0
    grey = s_ch < 80
    # The leg = a large grey region inside the circle; a can's grey lid is a small one.
    n_grey, grey_labels, grey_stats, _ = cv2.connectedComponentsWithStats(
        (grey & (circle_mask > 0)).astype(np.uint8), connectivity=8)
    leg = np.zeros(grey.shape, bool)
    for i in range(1, n_grey):
        if grey_stats[i, cv2.CC_STAT_AREA] >= 0.02 * circle_area:
            leg |= grey_labels == i
    pad_hue = hue_in_band(hsv, pad_color)
    strict = inner & (circle_color_mask > 0)
    pad_v = float(np.median(v_ch[strict])) if np.count_nonzero(strict) >= 50 else 200.0
    if ignore is not None:
        inner &= ignore == 0   # the gear foot over a pad is neither a can nor a shadow
    distinct_px = inner & ~grey & ~pad_hue & (s_ch >= 100)
    dark_same = inner & ~grey & pad_hue & (v_ch < 0.55 * pad_v)
    holes = ((distinct_px | dark_same).astype(np.uint8)) * 255
    # Large dark same-hue regions are the pad in shadow (drone, leg): removed, but a
    # compact blob inside that is darker than the shadow is a can standing in it.
    in_shadow_blobs: list[np.ndarray] = []
    n_lab, labels, st, _ = cv2.connectedComponentsWithStats(dark_same.astype(np.uint8), connectivity=8)
    for i in range(1, n_lab):
        if st[i, cv2.CC_STAT_AREA] < 0.05 * circle_area:
            continue
        comp = labels == i
        holes[comp] = 0
        v_b = float(np.median(v_ch[comp]))
        sub = cv2.morphologyEx((comp & (v_ch < 0.6 * v_b)).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sub_contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        in_shadow_blobs.extend(sub_contours)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rim = (~inner).astype(np.uint8) * 255
    if ignore is not None:
        rim[ignore > 0] = 0
    # A can is a fixed fraction of a pad (both are real objects): a blob far smaller
    # than that is a speck of dirt, however round.
    min_area = max(min_area, int(0.003 * circle_area))
    h, w = hsv.shape[:2]
    boxes = []
    deferred = []   # same-color cans, resolved once the shadow direction is known
    for c, in_shadow in [(c, False) for c in contours] + [(c, True) for c in in_shadow_blobs]:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area_frac * circle_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        perim = cv2.arcLength(c, True)
        # Shape on the convex hull: a dark can's outline is ragged (JPEG noise at V ~ 20),
        # which drags the raw circularity below the threshold, while the hull of a can,
        # or of a can plus its shadow, is still a compact blob. A leg shadow or board
        # seam crossing the pad stays elongated on its hull as well.
        hull = cv2.convexHull(c)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        circ = 4 * np.pi * hull_area / (hull_perim * hull_perim) if hull_perim > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0
        blob = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(blob, [c], -1, 255, thickness=-1)
        if not 0.3 <= bw / max(bh, 1) <= 3.0 or circ < 0.4 or solidity < 0.5:
            # Elongated or ragged = leg shadow / board seam crossing the pad, not a can.
            # But a can of another hue glued to such a region is still a can: take the
            # compact distinct-hue components inside it on their own.
            for part in distinct_components(((blob > 0) & distinct_px), min_area, rim):
                color = tin_color(hsv, part)
                if color is not None and color != pad_color:
                    x, y, bw, bh = cv2.boundingRect(part)
                    boxes.append(Box(cls_id(color, "Tin"), x, y, x + bw, y + bh))
            continue
        dilated = cv2.dilate(blob, ring_kernel)
        # A hole touching the pad's rim is the pad's own cut-off edge, not a can. The
        # frame edge is not a rim (inner reaches it), and neither is the gear foot's
        # outline (a can half hidden under the foot is still a can).
        if np.count_nonzero(dilated & rim) > 0.15 * perim:
            continue
        # Pixels that are not pad-colored are the can for sure; they also name its color
        # (a blue can half hidden by the leg is blue, however much shadow is in the blob).
        distinct_u8 = cv2.morphologyEx(((blob > 0) & distinct_px).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n_distinct = int(np.count_nonzero(distinct_u8))
        color = tin_color(hsv, distinct_u8) if n_distinct >= max(min_area // 2, 10) else None
        if color is not None and color != pad_color:
            # Its visible part is the can, even when the frame edge cuts it.
            can = largest_component(distinct_u8)
            rest = (blob > 0) & (can == 0)
            cy_c, cx_c = np.argwhere(can > 0).mean(axis=0)
            if np.count_nonzero(rest) >= 0.3 * np.count_nonzero(can):
                ry, rx = np.argwhere(rest).mean(axis=0)
                shadow.add(rx - cx_c, ry - cy_c)
            x, y, bw, bh = cv2.boundingRect(can)
            boxes.append(Box(cls_id(color, "Tin"), x, y, x + bw, y + bh))
            continue
        color = tin_color(hsv, blob)
        if color is None:
            continue
        # No distinct pixels: a can of the pad's own color (dark against the paint) or a
        # shadow sliver next to the leg. Only a clearly round blob that does not lean on
        # the leg counts: next to the leg a dark half-disc may be the shadow of a hidden
        # can. Inside a shadow region the blob must also be solid, since shadow gradients
        # make ragged blobs.
        contact = (dilated > 0) & leg
        touches_leg = int(np.count_nonzero(contact))
        if touches_leg > 0.1 * perim:
            # A dark blob against the leg on the side the shadows fall to may be the
            # shadow of a can hidden behind the leg; on the sunward side it is a can.
            d = shadow.direction()
            if d is None:
                continue
            cy_b, cx_b = np.argwhere(blob > 0).mean(axis=0)
            cy_l, cx_l = np.argwhere(contact).mean(axis=0)
            to_leg = np.array([cx_l - cx_b, cy_l - cy_b], np.float32)
            if float(to_leg @ d) < 0.5 * float(np.hypot(*to_leg)):
                continue  # the leg is not in the shadow direction: blob sits on the shadow side
        if circ < 0.6 or (in_shadow and solidity < 0.8):
            continue
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


def distinct_components(mask: np.ndarray, min_area: int, rim: np.ndarray) -> list[np.ndarray]:
    """Compact connected components of a distinct-hue mask that do not touch the pad rim."""
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area, hull_perim = cv2.contourArea(hull), cv2.arcLength(hull, True)
        circ = 4 * np.pi * hull_area / (hull_perim * hull_perim) if hull_perim > 0 else 0
        if circ < 0.5 or area < 0.6 * hull_area:
            continue
        part = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(part, [c], -1, 255, thickness=-1)
        ring = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & rim
        if np.count_nonzero(ring) > 0.15 * cv2.arcLength(c, True):
            continue
        parts.append(part)
    return parts


def fit_circle(hull: np.ndarray, w: int, h: int) -> tuple[float, float, float, float] | None:
    """Least-squares circle through the hull outline, ignoring points on the frame edge.
    Returns (cx, cy, r, rms residual / r), or None when there is too little outline."""
    outline = np.zeros((h, w), np.uint8)
    cv2.drawContours(outline, [hull], -1, 255, 1)
    outline[:3, :] = 0
    outline[-3:, :] = 0
    outline[:, :3] = 0
    outline[:, -3:] = 0
    pts = np.argwhere(outline > 0)[:, ::-1].astype(np.float64)   # (x, y)
    if len(pts) < 30:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(a, -(x * x + y * y), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx * cx + cy * cy - sol[2]
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    residual = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))) / r
    return float(cx), float(cy), r, residual


def circular_segment(hull: np.ndarray, bw: int, bh: int, w: int, h: int) -> bool:
    """Is this hull (cut by the frame edge) a segment of a circle? Fit a circle to the
    hull outline away from the frame edge and check the fit and the segment's thickness."""
    fit = fit_circle(hull, w, h)
    if fit is None:
        return False
    r, residual = fit[2], fit[3]
    # A plausible pad: radius no larger than the frame, and the visible segment at least
    # 12 % of the radius thick (a hairline sliver at the edge is not worth a label).
    return residual <= 0.04 and r <= max(w, h) and min(bw, bh) >= 0.12 * r


def circle_area_px(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


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


def drop_nested_pads(pads: list[tuple[Box, np.ndarray]]) -> list[tuple[Box, np.ndarray]]:
    """A small 'pad' whose center lies on a bigger pad is a can seen up close; the tin pass labels it."""
    keep = []
    for box, filled in pads:
        cx, cy = (box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2
        area = (box.x2 - box.x1) * (box.y2 - box.y1)
        nested = any(o is not box and of[cy, cx] > 0 and area < 0.3 * (o.x2 - o.x1) * (o.y2 - o.y1)
                     for o, of in pads)
        if not nested:
            keep.append((box, filled))
    return keep


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


def ignore_region(masks: list[tuple[float, float, float, float]], auto_mask: np.ndarray | None,
                  w: int, h: int) -> np.ndarray:
    """Ignored pixels: manual rectangles (normalized) + the auto mask."""
    region = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in masks:
        region[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 255
    if auto_mask is not None:
        region |= auto_mask
    return region


def mask_coverage(box: Box, region: np.ndarray) -> float:
    """Fraction of the box that lies inside the ignored region."""
    sub = region[box.y1:box.y2, box.x1:box.x2]
    return float(np.count_nonzero(sub)) / max(sub.size, 1)


def leg_mask(hsv: np.ndarray) -> np.ndarray:
    """Pixels of the landing legs: big black components that touch the frame border."""
    h, w = hsv.shape[:2]
    dark = (hsv[..., 2] < LEG_V_MAX).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        if area >= LEG_MIN_FRAC * w * h and (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            keep[i] = True
    return keep[lab]


def ends_leg(filled: np.ndarray, leg: np.ndarray) -> bool:
    """True when a leg ends at this blob: the blob touches leg pixels along a good part
    of its outline, and the leg around it lies on one side of the blob only (a pad the
    leg crosses has leg on both sides)."""
    h, w = filled.shape
    ys, xs = np.nonzero(filled)
    if not len(xs):
        return False
    ring = (cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0) & (filled == 0)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(len(c) for c in contours)
    if np.count_nonzero(ring & leg) < 0.3 * perimeter:
        return False
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    m = max(x2 - x1, y2 - y1)
    X1, X2, Y1, Y2 = max(0, x1 - m), min(w, x2 + m), max(0, y1 - m), min(h, y2 + m)
    ly, lx = np.nonzero(leg[Y1:Y2, X1:X2])
    if len(lx) < 300:
        return False
    pts = np.stack([lx + X1, ly + Y1], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
    axis = evecs[:, int(np.argmax(evals))]
    proj = (pts - mean) @ axis
    center = float((np.array([xs.mean(), ys.mean()], np.float32) - mean) @ axis)
    return min(float((proj > center).mean()), float((proj < center).mean())) <= 0.1


def label_frame(frame: np.ndarray, args: argparse.Namespace, shadow: ShadowEstimate,
                auto_mask: np.ndarray | None = None, gear_colors: set[str] | None = None) -> list[Box]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    ignore = ignore_region(args.mask, auto_mask, w, h) if (args.mask or auto_mask is not None) else None
    pads, free_tins = detect_pads(frame, args, ignore, gear_colors)
    # A detection that sits mostly inside an ignored region (landing gear in a top corner)
    # is dropped; a pad that merely reaches into the corner keeps its whole box. This
    # comes before merging, or the gear foot next to a red pad would be merged into it.
    # The gear foot is a small blob (it is a fixed part of the airframe), so a large
    # detection in the corner is a pad the camera tilted into it, foot or no foot; and
    # a detection of a color the airframe does not have (a green pad under the red foot)
    # is a pad as well.
    if ignore is not None:
        gear_max = GEAR_MAX_FRAC * w * h
        leg = leg_mask(hsv) if pads else None

        def is_gear(box: Box, px: int, filled: np.ndarray | None = None) -> bool:
            color = CLASSES[box.cls].split("_")[0]
            if gear_colors and color not in gear_colors:
                return False
            cov = mask_coverage(box, ignore)
            if cov >= 0.7 and px < gear_max:
                return True
            # A foot that swung out of its corner (the legs sway in flight): a small blob
            # of the gear color that still reaches into the corner and sits where a leg
            # ends. A pad the leg merely crosses has leg on both sides of it.
            return (filled is not None and leg is not None and cov >= 0.3
                    and px < 0.5 * gear_max and ends_leg(filled, leg))

        pads = [(b, f) for b, f in pads if not is_gear(b, int(np.count_nonzero(f)), f)]
        free_tins = [t for t in free_tins if not is_gear(t, (t.x2 - t.x1) * (t.y2 - t.y1))]
    pads = merge_pads(pads)
    pads = drop_nested_pads(pads)
    boxes: list[Box] = []
    for box, filled in pads:
        boxes.append(box)
        if args.tins:
            pad_color = CLASSES[box.cls].split("_")[0]
            cmask = color_mask(hsv, pad_color, fill_holes=False)
            boxes.extend(detect_tins_on_pad(hsv, filled, cmask, pad_color, args.min_tin_area,
                                            args.max_tin_frac_pad, shadow, ignore))
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


def static_color_mask(video: Path, step: int, max_frames: int = 60) -> tuple[np.ndarray | None, set[str]]:
    """Pixels that are pad-colored in most sampled frames belong to the airframe
    (red landing gear in the corners), not to the ground: the ground moves.
    Returns (mask, colors of the airframe parts found)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, set()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(step, total // max_frames if total else step)
    acc = None
    n = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            masks = {color: (color_mask(hsv, color) > 0).astype(np.float32) for color in COLORS}
            acc = masks if acc is None else {color: acc[color] + masks[color] for color in COLORS}
            n += 1
        idx += 1
    cap.release()
    if acc is None or n < 5:
        return None, set()
    static = np.zeros(next(iter(acc.values())).shape, np.uint8)
    colors = set()
    for color in COLORS:
        part = ((acc[color] / n) >= 0.5).astype(np.uint8) * 255
        if np.count_nonzero(part):
            colors.add(color)
            static |= part
    if not colors:
        return None, set()
    return cv2.dilate(static, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), colors


def process_video(video: Path, args: argparse.Namespace) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"cannot open {video}", file=sys.stderr)
        return {"video": str(video), "error": "cannot open"}
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / args.fps))) if args.fps > 0 else 1
    auto_mask, gear_colors = (None, set()) if args.no_auto_mask else static_color_mask(video, step)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    stats = {"video": str(video), "frames_seen": 0, "frames_written": 0, "boxes": 0,
             "per_class": {c: 0 for c in CLASSES},
             "auto_mask_frac": 0.0 if auto_mask is None else round(float(np.count_nonzero(auto_mask)) / auto_mask.size, 4)}
    shadow = ShadowEstimate()
    if args.tins:
        # Learn the shadow direction from a few frames spread over the video before
        # labeling, so the first frames can split a same-color can from its shadow too.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for pos in np.linspace(0, max(total - 1, 0), 8)[1:-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if ok:
                label_frame(frame, args, shadow, auto_mask, gear_colors)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            stats["frames_seen"] += 1
            boxes = label_frame(frame, args, shadow, auto_mask, gear_colors)
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
    ap.add_argument("--no-auto-mask", action="store_true",
                    help="do not auto-detect static airframe parts (landing gear) that stay pad-colored across frames")
    ap.add_argument("--mask", nargs="*", default=list(DEFAULT_MASK), metavar="X1,Y1,X2,Y2",
                    help="normalized rectangles to ignore. Default: the two top corners, where the landing "
                         "gear shows (%(default)s). Pass 'none' to disable")
    ap.add_argument("--min-pad-area", type=float, default=0.002,
                    help="min pad (Rag) area as fraction of the frame (default 0.002)")
    ap.add_argument("--min-circularity", type=float, default=0.6)
    ap.add_argument("--min-tin-area", type=int, default=40,
                    help="min tin area in pixels (blobs under 0.3 %% of the pad area are dropped as well)")
    ap.add_argument("--max-tin-frac-pad", type=float, default=0.25,
                    help="max tin area as fraction of the pad it sits on")
    ap.add_argument("--max-tin-frac", type=float, default=0.004,
                    help="max area of a tin not on a pad, as fraction of the frame")
    args = ap.parse_args(argv)
    if args.mask == ["none"]:
        args.mask = []
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
