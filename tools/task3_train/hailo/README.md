# Hailo .hef 컴파일 / 파이 교체 절차

`tin_detective.py` 의 `HailoYOLO` 는 hef 출력이 **온칩 NMS 결과**(클래스별 `[ymin, xmin, ymax, xmax, score]` 정규화
좌표, `HailoInfer(..., input_type="UINT8", output_type="FLOAT32")`) 라고 가정한다. 그래서 hef 는 모델주(hailomz)
yolov8n 레시피처럼 정규화(/255)·sigmoid·`nms_postprocess(meta_arch=yolov8)` 를 모두 포함해야 하고 클래스 수는 6 이다.
클래스 순서는 `data.yaml` = `NAMES = ['Blue_Rag','Blue_Tin','Green_Rag','Green_Tin','Red_Rag','Red_Tin']` 이다.

## 필요한 것
- Hailo Developer Zone 에서 받은 **Hailo AI Software Suite** docker (DFC + 모델주 포함) 또는 DFC whl + `hailo_model_zoo` 설치 (x86 리눅스, GPU 없어도 됨. 양자화·컴파일 10~30 분).
- 파이의 HailoRT 버전과 짝이 맞는 DFC: `hailortcli fw-control identify` 로 칩(hailo8 / hailo8l)과 펌웨어 버전 확인.
  DFC 3.27↔HailoRT 4.17, 3.28↔4.18, 3.29↔4.19, 3.30↔4.20, 3.31↔4.21.

## 컴파일
```bash
# ONNX 와 캘리브레이션 이미지는 학습 쪽에서 만들어 둔 것 (export_onnx.py, make_calib.py)
bash tools/task3_train/hailo/compile.sh task3_yolov8n.onnx calib/ hailo8l     # 또는 hailo8
```
직접 DFC 를 쓰려면 `yolov8n_task3.alls` + `yolov8n_task3_nms_config.json` 을 같은 폴더에 두고
`hailo parser onnx ... --end-node-names (compile.sh 의 6개)` → `hailo optimize --calib-set-path ... --model-script yolov8n_task3.alls` → `hailo compiler` 순서.

## 파이에서 확인·교체
```bash
hailortcli parse-hef tin_detector.hef          # 입력 640x640x3 UINT8, 출력 1개 (nms, 6 classes) 인지
hailortcli run tin_detector.hef                # 로드/추론 되는지 (fps 출력)
cp ~/models/tin_detector.hef ~/models/tin_detector.hef.bak_$(date +%m%d)   # 기존 모델 백업
cp tin_detector.hef ~/models/tin_detector.hef  # HEF_PATH 기본값
python3 ~/tin_ws/scripts/tin_detective.py      # [검출] 로그에 Blue_Tin/Red_Rag 등이 뜨는지, 추론 ms 확인
```
`YOLO_CONF`(기본 0.4)는 노드가 다시 거르므로 hef 의 `nms_scores_th 0.2` 는 그대로 두면 된다.
