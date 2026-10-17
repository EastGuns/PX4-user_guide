#!/usr/bin/env python3
"""Hailo 양자화 캘리브레이션 이미지 세트를 만든다 (배포와 같은 640x640 letterbox, RGB 순서는 hailomz 가 처리).

  python3 tools/task3_train/make_calib.py dataset_split/images/train --out calib --n 1024 --video 영상.mp4

학습 프레임(1280x720)을 tin_detective.py 와 같은 방식(긴 변 640, 114 회색 패딩, 가운데 정렬)으로 letterbox 한
640x640 JPEG 를 저장한다. --video 를 주면 라벨링에 안 쓰인 중간 프레임도 섞어 n 장을 채운다.
"""
import argparse, glob, os, random
import cv2, numpy as np

def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    c = np.full((s, s, 3), 114, np.uint8); top = (s - nh) // 2; left = (s - nw) // 2
    c[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', help='학습 이미지 폴더')
    ap.add_argument('--out', default='calib')
    ap.add_argument('--n', type=int, default=1024, help='DFC 고급 최적화에는 1024장 이상 필요')
    ap.add_argument('--video', default=None, help='추가 프레임을 뽑을 영상')
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, '*.jpg')) + glob.glob(os.path.join(a.images, '*.png')))
    random.Random(0).shuffle(files)
    k = 0
    for f in files[:a.n]:
        img = cv2.imread(f)
        if img is None: continue
        cv2.imwrite(os.path.join(a.out, f'calib_{k:04d}.jpg'), letterbox(img, a.imgsz), [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
    if a.video and k < a.n:
        cap = cv2.VideoCapture(a.video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), a.n - k).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, img = cap.read()
            if not ok: continue
            cv2.imwrite(os.path.join(a.out, f'calib_{k:04d}.jpg'), letterbox(img, a.imgsz), [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
    print(f'캘리브레이션 이미지 {k} 장 → {a.out}')

if __name__ == '__main__':
    main()
