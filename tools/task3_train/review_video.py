#!/usr/bin/env python3
"""학습된 모델로 영상 전체를 추론해 검토 영상을 만든다 (박스 + 클래스/점수, 왼쪽 아래 프레임 번호, 원본 속도).

  python3 tools/task3_train/review_video.py runs/task3/weights/best.pt 영상.mp4 --out review.mp4 [--conf 0.4] [--stream-sim]

--stream-sim: 배포 환경(cam264 = 640x360 재인코딩)을 흉내 내어 프레임을 640x360 으로 줄인 뒤 추론한다.
              (letterbox 결과는 1280x720 원본과 같지만 downscale 보간 차이를 함께 본다.)
추론 설정은 tin_detective.py 와 같이 imgsz 640, conf 기본 0.4(YOLO_CONF), iou 0.7.
"""
import argparse, json, os
import cv2, numpy as np
from ultralytics import YOLO

COLORS = {'Blue': (255, 128, 0), 'Green': (0, 200, 0), 'Red': (0, 0, 255)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights'); ap.add_argument('video')
    ap.add_argument('--out', default='review.mp4')
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--stream-sim', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--log', default=None, help='프레임별 검출 JSONL 저장')
    a = ap.parse_args()

    model = YOLO(a.weights); names = model.names
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = a.out + '.raw.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    logf = open(a.log, 'w') if a.log else None
    i = 0; counts = {}
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and i >= a.max_frames): break
        src = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) if a.stream_sim else frame
        r = model.predict(src, imgsz=a.imgsz, conf=a.conf, iou=a.iou, verbose=False)[0]
        sx, sy = W / src.shape[1], H / src.shape[0]
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (b.xyxy[0].tolist()); c = int(b.cls[0]); s = float(b.conf[0])
            x1, x2 = x1 * sx, x2 * sx; y1, y2 = y1 * sy, y2 * sy
            nm = names[c]; col = COLORS.get(nm.split('_')[0], (255, 255, 255))
            th = 1 if nm.endswith('_Tin') else 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
            cv2.putText(frame, f'{nm} {s:.2f}', (int(x1), max(12, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            dets.append({'cls': nm, 'conf': round(s, 3), 'xyxy': [round(v, 1) for v in (x1, y1, x2, y2)]})
            counts[nm] = counts.get(nm, 0) + 1
        cv2.putText(frame, f'frame {i:06d}', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f'frame {i:06d}', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        if logf: logf.write(json.dumps({'frame': i, 'dets': dets}) + '\n')
        i += 1
        if i % 500 == 0: print(f'{i} 프레임...', flush=True)
    vw.release(); cap.release()
    if logf: logf.close()
    # 브라우저 재생용 H.264 로 재인코딩
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{a.out}"')
    if rc == 0: os.remove(tmp)
    else: os.rename(tmp, a.out)
    print(f'{i} 프레임 → {a.out}  검출 수: {counts}')

if __name__ == '__main__':
    main()
