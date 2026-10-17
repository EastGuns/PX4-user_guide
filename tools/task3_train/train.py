#!/usr/bin/env python3
"""Task3 tin/rag 검출기 전이학습 (Ultralytics YOLO11n 기본, YOLOv8n 도 가능. imgsz 640).

  python3 tools/task3_train/train.py dataset_split/data.yaml --epochs 60 --out runs/task3

설정 근거:
- 배포(tin_detective.py)가 640 letterbox 로 추론하므로 imgsz 640. 캔이 6~20 px 라 더 줄이지 않는다.
- 색이 곧 클래스(Red/Green/Blue)라 hue 증강(hsv_h)은 거의 끈다. 밝기·채도 증강은 그늘/햇빛 대응용으로 둔다.
- 하방 카메라라 상하·좌우 뒤집기 모두 자연스러운 장면이므로 flipud/fliplr 를 켠다.
- 시작 가중치는 기본 COCO 사전학습 yolo11n.pt (파이의 tin_detector.hef 가 YOLO11n). 기존 tin 모델 .pt 가 있으면 --weights 로 준다.
"""
import argparse, os, shutil
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', help='data.yaml (prepare_split.py 출력)')
    ap.add_argument('--weights', default='yolo11n.pt', help='시작 가중치 (.pt): yolo11n.pt(기본, 파이에서 쓰는 세대) / yolov8n.pt. 기존 tin 모델로 이어 학습하려면 그 .pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='cpu', help="'cpu' 또는 '0'(GPU)")
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='runs/task3', help='결과 폴더 (project/name)')
    ap.add_argument('--patience', type=int, default=0, help='조기 종료 (0 = 끔)')
    a = ap.parse_args()

    project, name = os.path.split(a.out.rstrip('/'))
    model = YOLO(a.weights)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        project=project or '.', name=name, exist_ok=True,
        patience=a.patience if a.patience > 0 else a.epochs + 1,
        optimizer='auto', cos_lr=True, close_mosaic=10,
        hsv_h=0.005, hsv_s=0.5, hsv_v=0.4,        # hue 거의 고정 (색 = 클래스)
        degrees=10.0, translate=0.1, scale=0.5,
        flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
        plots=True, verbose=True, seed=0, deterministic=True,
    )
    best = os.path.join(a.out, 'weights', 'best.pt')
    print('best.pt:', best if os.path.exists(best) else '없음')

if __name__ == '__main__':
    main()
