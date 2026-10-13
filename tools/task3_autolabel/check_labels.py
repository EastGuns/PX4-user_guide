#!/usr/bin/env python3
"""Sanity check for autolabel.py output (Task3 sample layout: 3 pads, 2 cans per pad).

Flags, per frame: the same pad color twice, a can outside every pad, a fully
visible pad with a can count other than --cans-per-pad, and a fully visible
pad whose box is far from square. Use it to pick frames to review by eye.

    python3 check_labels.py labels_out [--cans-per-pad 2] [--width 1280 --height 720]
"""
import argparse, collections, glob, os

CL = ["Blue_Rag", "Blue_Tin", "Green_Rag", "Green_Tin", "Red_Rag", "Red_Tin"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="autolabel.py output directory (contains labels/)")
    ap.add_argument("--cans-per-pad", type=int, default=2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.out, "labels", "*.txt")))
    bad = {}
    for f in files:
        rows = [list(map(float, l.split())) for l in open(f, encoding="utf-8") if l.strip()]
        pads = [r for r in rows if int(r[0]) % 2 == 0]
        tins = [r for r in rows if int(r[0]) % 2 == 1]
        probs = []
        for c, n in collections.Counter(int(r[0]) for r in pads).items():
            if n > 1:
                probs.append(f"{CL[c]} x{n}")

        def inside(t, p):
            return abs(t[1] - p[1]) <= p[3] / 2 and abs(t[2] - p[2]) <= p[4] / 2

        for t in tins:
            if not any(inside(t, p) for p in pads):
                probs.append(f"{CL[int(t[0])]} outside pad")
        for p in pads:
            full = p[1] - p[3] / 2 > 0.01 and p[1] + p[3] / 2 < 0.99 and p[2] - p[4] / 2 > 0.01 and p[2] + p[4] / 2 < 0.99
            if not full:
                continue
            n = sum(1 for t in tins if inside(t, p))
            if n != a.cans_per_pad:
                probs.append(f"{CL[int(p[0])]} has {n} cans")
            aspect = (p[3] * a.width) / max(p[4] * a.height, 1e-6)
            if not 0.5 <= aspect <= 2.0:
                probs.append(f"{CL[int(p[0])]} odd aspect {aspect:.2f}")
        if probs:
            bad[os.path.basename(f)] = probs
    print(f"frames: {len(files)}  frames with issues: {len(bad)}")
    for name, probs in bad.items():
        print(name, probs)


if __name__ == "__main__":
    main()
