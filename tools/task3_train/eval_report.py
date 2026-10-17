#!/usr/bin/env python3
"""val 세트로 지표 표(클래스별 P/R/mAP50/mAP50-95), 혼동행렬, 예측 vs 라벨 미리보기 그림을 만든다.

  python3 tools/task3_train/eval_report.py runs/task3/weights/best.pt dataset_split/data.yaml --out report
"""
import argparse, glob, os, random, shutil
import cv2, numpy as np, yaml
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights'); ap.add_argument('data')
    ap.add_argument('--out', default='report')
    ap.add_argument('--conf', type=float, default=0.4, help='미리보기용 conf (tin_detective YOLO_CONF)')
    ap.add_argument('--n-preview', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    d = yaml.safe_load(open(a.data)); root = d.get('path', '.')
    imgs_all = sorted(glob.glob(os.path.join(root, d['val'], '*.jpg')))
    m = model.val(data=a.data, imgsz=640, batch=8, device='cpu', plots=True, project=os.path.abspath(a.out), name='val', exist_ok=True, verbose=False)
    names = model.names
    rows = ['| 클래스 | 이미지 | 인스턴스 | P | R | mAP50 | mAP50-95 |', '|---|---|---|---|---|---|---|']
    # 라벨 파일에서 클래스별 인스턴스/이미지 수
    nt_c = np.zeros(len(names), int); nt_i = np.zeros(len(names), int)
    for p_ in imgs_all:
        lab = p_.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        seen = set()
        if os.path.exists(lab):
            for l in open(lab):
                if l.strip(): c_ = int(l.split()[0]); nt_c[c_] += 1; seen.add(c_)
        for c_ in seen: nt_i[c_] += 1
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        ni = int(nt_i[int(c)]) if nt_i is not None else '-'; nc_ = int(nt_c[int(c)]) if nt_c is not None else '-'
        rows.append(f'| {names[int(c)]} | {ni} | {nc_} | {p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |')
    mp, mr, map50, map_ = m.box.mean_results()
    rows.append(f'| **all** | {len(imgs_all)} | {int(nt_c.sum()) if nt_c is not None else "-"} | {mp:.3f} | {mr:.3f} | {map50:.3f} | {map_:.3f} |')
    with open(os.path.join(a.out, 'metrics.md'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))
    for fn in ('confusion_matrix.png', 'confusion_matrix_normalized.png', 'PR_curve.png', 'F1_curve.png'):
        src = os.path.join(a.out, 'val', fn)
        if os.path.exists(src): shutil.copy2(src, os.path.join(a.out, fn))

    # 예측(실선) vs 라벨(점선) 미리보기
    imgs = list(imgs_all)
    random.Random(1).shuffle(imgs); imgs = sorted(imgs[:a.n_preview])
    tiles = []
    for p in imgs:
        img = cv2.imread(p); H, W = img.shape[:2]
        lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(lab):
            for l in open(lab):
                c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
                for k in range(x1, x2, 8): cv2.line(img, (k, y1), (min(k + 4, x2), y1), (255, 255, 0), 1); cv2.line(img, (k, y2), (min(k + 4, x2), y2), (255, 255, 0), 1)
                for k in range(y1, y2, 8): cv2.line(img, (x1, k), (x1, min(k + 4, y2)), (255, 255, 0), 1); cv2.line(img, (x2, k), (x2, min(k + 4, y2)), (255, 255, 0), 1)
        r = model.predict(img, imgsz=640, conf=a.conf, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()); nm = names[int(b.cls[0])]
            col = {'Blue': (255, 128, 0), 'Green': (0, 200, 0), 'Red': (0, 0, 255)}[nm.split('_')[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f'{nm} {float(b.conf[0]):.2f}', (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(img, os.path.basename(p).rsplit('_', 1)[-1].split('.')[0], (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    if tiles:
        cols = 3; rows_ = (len(tiles) + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        grid = np.zeros((rows_ * h, cols * w, 3), np.uint8)
        for i, t in enumerate(tiles):
            grid[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
        cv2.imwrite(os.path.join(a.out, 'preview_val_pred.jpg'), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('미리보기(실선=예측, 점선 노랑=라벨):', os.path.join(a.out, 'preview_val_pred.jpg'))

if __name__ == '__main__':
    main()
