#!/usr/bin/env bash
# Task3 YOLOv8n ONNX → Hailo .hef  (Hailo Dataflow Compiler + hailo_model_zoo 가 설치된 환경에서 실행)
#
#   bash tools/task3_train/hailo/compile.sh task3_yolov8n.onnx calib/ hailo8l
#
# 인자:  1) ONNX (export_onnx.py 출력)   2) 캘리브레이션 이미지 폴더 (make_calib.py 출력)   3) hw-arch: hailo8 | hailo8l
#
# ★ hef 는 칩(hailo8 / hailo8l) 과 파이의 HailoRT 버전에 맞는 DFC 로 컴파일해야 로드된다.
#    DFC 3.27 ↔ HailoRT 4.17, 3.28 ↔ 4.18, 3.29 ↔ 4.19, 3.30 ↔ 4.20, 3.31 ↔ 4.21 (모델주 v2.11 ~ v2.15).
#    파이에서:  hailortcli fw-control identify   → Device Architecture 와 Firmware Version 확인.
set -euo pipefail
ONNX=${1:?ONNX 경로}; CALIB=${2:?캘리브레이션 이미지 폴더}; ARCH=${3:-hailo8l}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$(pwd)/hef_out}; mkdir -p "$OUT"
NAME=tin_detector

# 1) 모델주 레시피로 컴파일. --classes 6 이 NMS 설정의 클래스 수를 바꾸고, 정규화(/255)·sigmoid·NMS 를 hef 안에 넣는다.
#    end node 는 export_onnx.py 가 출력한 6개 Conv (box 3 + cls 3). 모델주가 자동으로 찾지만 명시해 둔다.
hailomz compile yolov8n \
  --ckpt "$ONNX" \
  --hw-arch "$ARCH" \
  --calib-path "$CALIB" \
  --classes 6 \
  --start-node-names images \
  --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv \
                   /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv \
                   /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv \
  --performance

# hailomz 는 현재 폴더에 yolov8n.hef 를 쓴다
mv -f yolov8n.hef "$OUT/$NAME.hef"
echo "hef: $OUT/$NAME.hef"

# 2) 확인: 입력 640x640x3 UINT8, 출력 1개 (NMS, 6 클래스 x 5 x 100)
hailortcli parse-hef "$OUT/$NAME.hef" || true
