# Hailo .hef 컴파일 / 파이 교체 절차

`tin_detective.py` 의 `HailoYOLO` 는 hef 출력이 **온칩 NMS 결과**(클래스별 `[ymin, xmin, ymax, xmax, score]` 정규화
좌표, `HailoInfer(..., input_type="UINT8", output_type="FLOAT32")`) 라고 가정한다. 그래서 hef 는 모델주(hailomz)
`yolov11n` 레시피(YOLO11n. v8n 이면 `yolov8n`)처럼 정규화(/255)·sigmoid·`nms_postprocess(meta_arch=yolov8)` 를 모두 포함해야 하고 클래스 수는 6 이다.
YOLO11 도 모델주에서 `meta_arch=yolov8` NMS 디코더를 쓰므로 출력 형식은 v8 과 같다. 다른 점은 end node(Detect 헤드가 `model.23`)와 디코더 레이어 번호뿐이고, `compile.sh` 가 ONNX 를 보고 자동으로 고른다.
클래스 순서는 `data.yaml` = `NAMES = ['Blue_Rag','Blue_Tin','Green_Rag','Green_Tin','Red_Rag','Red_Tin']` 이다.

## 필요한 것
- Hailo Developer Zone 에서 받은 **Hailo AI Software Suite** docker (DFC + 모델주 포함) 또는 DFC whl + `hailo_model_zoo` 설치 (x86 리눅스, GPU 없어도 됨. 양자화·컴파일 10~30 분).
- 파이의 HailoRT 버전과 짝이 맞는 DFC: `hailortcli fw-control identify` 로 칩과 펌웨어 버전 확인.
  - **Hailo-10H (라즈베리파이 AI HAT+ 2, 이 프로젝트의 칩)**: DFC **5.x** + `hailo_model_zoo` **master(v5.x)** 브랜치. HailoRT 도 5.x 이고
    DFC 와 같은 버전으로 맞춘다(예: 파이 HailoRT 5.4.0 → DFC 5.4.0, 모델주 v5.1.0). 모델주 README 의 호환 안내: master 는 Hailo-10/15 전용.
  - Hailo-8/8L: DFC 3.x + 모델주 v2.x, HailoRT 4.x (3.30↔4.20, 3.31↔4.21).

## 컴파일
```bash
# ONNX 와 캘리브레이션 이미지는 학습 쪽에서 만들어 둔 것 (export_onnx.py, make_calib.py)
bash tools/task3_train/hailo/compile.sh task3_yolo11n.onnx calib/ hailo10h    # Hailo-8L 이면 hailo8l (DFC 3.x, 모델주 v2.x 에는 yolov11n 레시피가 v2.14 부터 있음)
```
직접 DFC 를 쓰려면 `yolov11n_task3.alls` + `yolov11n_task3_nms_config.json`(v8 은 `yolov8n_*`)을 같은 폴더에 두고
`hailo parser onnx ... --end-node-names (compile.sh 의 6개)` → `hailo optimize --calib-set-path ... --model-script yolov11n_task3.alls` → `hailo compiler` 순서.

## 파이에서 확인·교체
```bash
hailortcli parse-hef tin_detector.hef          # 입력 640x640x3 UINT8, 출력 1개 (nms, 6 classes), 아키텍처 HAILO10H 인지
hailortcli run tin_detector.hef                # 로드/추론 되는지 (fps 출력)
cp ~/models/tin_detector.hef ~/models/tin_detector.hef.bak_$(date +%m%d)   # 기존 모델 백업
cp tin_detector.hef ~/models/tin_detector.hef  # HEF_PATH 기본값
python3 ~/tin_ws/scripts/tin_detective.py      # [검출] 로그에 Blue_Tin/Red_Rag 등이 뜨는지, 추론 ms 확인
```
`YOLO_CONF`(기본 0.4)는 노드가 다시 거르므로 hef 의 `nms_scores_th 0.2` 는 그대로 두면 된다.
