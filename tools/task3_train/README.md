# Task3 tin/rag 검출기 학습 (`tools/task3_train`)

`tools/task3_autolabel` 이 만든 라벨로 YOLO11n(파이의 `tin_detector.hef` 세대, YOLOv8n 도 선택 가능)을 전이학습하고, 드론(라즈베리파이 + Hailo)의
`tin_detective.py` 가 쓰는 `tin_detector.hef` 로 내보내기까지의 재현 스크립트.
클래스 6개와 순서는 `tin_detective.py` 의 `NAMES` 와 같다: `Blue_Rag, Blue_Tin, Green_Rag, Green_Tin, Red_Rag, Red_Tin`.

## 설치
```bash
pip install -r tools/task3_train/requirements.txt      # GPU 면 torch 는 CUDA 빌드로
```

## 순서
```bash
# 0) 라벨 (task3_labels_2fps.zip 을 푼 폴더: images/ labels/ data.yaml)
# 1) 시간 구간 train/val 분할 (연속 프레임이라 무작위 분할은 누수)
python3 tools/task3_train/prepare_split.py task3_labels_2fps --out dataset_split           # 5구간, 구간마다 뒤 20% = val
# 2) 전이학습 (COCO 사전학습 yolo11n.pt 에서 시작. v8 이면 --weights yolov8n.pt, 기존 tin 모델에서 이어가려면 --weights 그.pt)
python3 tools/task3_train/train.py dataset_split/data.yaml --epochs 100 --out runs/task3    # GPU: --device 0 --batch 16
# 3) 검증 리포트: 클래스별 P/R/mAP 표, 혼동행렬, 예측(실선) vs 라벨(점선) 미리보기
python3 tools/task3_train/eval_report.py runs/task3/weights/best.pt dataset_split/data.yaml --out report
# 4) 검토 영상: 원본 영상 전체 추론, 박스 + 왼쪽 아래 프레임 번호, 원본 속도
python3 tools/task3_train/review_video.py runs/task3/weights/best.pt 영상.mp4 --out review.mp4 --conf 0.4
# 5) ONNX (opset 11, NMS 없음, 1x3x640x640) + Hailo end node 확인(model.23 = YOLO11 / model.22 = v8) + .pt 와 출력 대조
python3 tools/task3_train/export_onnx.py runs/task3/weights/best.pt --out task3_yolo11n.onnx
# 6) 양자화 캘리브레이션 이미지 (배포와 같은 640x640 letterbox)
python3 tools/task3_train/make_calib.py dataset_split/images/train --out calib --n 256 --video 영상.mp4
# 7) Hailo .hef (DFC + hailo_model_zoo 가 있는 PC/도커에서)  → hailo/README.md
bash tools/task3_train/hailo/compile.sh task3_yolo11n.onnx calib hailo10h     # 레시피(yolov11n/yolov8n)는 ONNX 에서 자동 판별
```

## 학습 설정 근거 (`train.py`)
| 항목 | 값 | 이유 |
| --- | --- | --- |
| 모델 | YOLO11n | 파이에서 돌던 `tin_detector.hef` 와 같은 세대. Hailo 모델주 master 에 `yolov11n` 레시피(온칩 NMS, hailo10h 지원)가 있다. `--weights yolov8n.pt` 로 v8n 도 같은 절차 |
| imgsz | 640 | 배포(`YOLO_IMGSZ=640` letterbox)와 동일. 캔이 6~20 px 라 줄이지 않음 |
| hsv_h | 0.005 | 색이 곧 클래스(Red/Green/Blue)라 hue 를 거의 안 흔듦. 밝기·채도 증강은 그늘/햇빛용으로 유지 |
| flipud / fliplr | 0.5 / 0.5 | 하방 카메라라 어느 방향이든 자연스러움 |
| mosaic / mixup | 1.0 / 0 | 기본. 마지막 10 에폭은 mosaic 끔 |
| 분할 | 5 구간 × 뒤 20% | 비행 단계(고도·배경)마다 val 이 생기고 train 과 시간이 겹치지 않음 (경계 1프레임 버림) |

캠 스트림(cam264)은 640x360 이고 학습 프레임은 1280x720 이지만, 둘 다 640 letterbox 를 거치면 같은 640x360 내용이 된다.
`review_video.py --stream-sim` 으로 640x360 축소 입력도 확인할 수 있다.

## 결과물
- `runs/task3/weights/best.pt`, `task3_yolo11n.onnx`, `calib/`, `report/metrics.md`·`confusion_matrix.png`·`preview_val_pred.jpg`, `review.mp4`
- `.hef` 는 Hailo DFC 가 있는 곳에서 `hailo/compile.sh` 로. 칩(Hailo-10H → DFC 5.x/모델주 master)과 파이의 HailoRT 버전에 맞는 DFC 여야 한다 (`hailo/README.md`).
