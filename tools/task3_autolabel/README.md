# Task3 자동 라벨링 (`autolabel.py`)

드론 하향 카메라 영상에서 프레임을 뽑아 **색상 원형 타겟(빨강/초록/파랑)** 과 그 위에
놓인 **소형 물체**의 바운딩 박스를 자동으로 만들어 줍니다. 결과는 YOLO 텍스트 형식이며,
원본 영상은 절대 수정하지 않고 모든 출력은 `--out` 디렉터리에만 씁니다.

## 설치

```bash
pip install -r tools/task3_autolabel/requirements.txt
```

## 사용법

```bash
# 영상 하나 또는 디렉터리(하위 폴더까지 검색)를 입력으로 받습니다.
python3 tools/task3_autolabel/autolabel.py data/videos --out labels_out --fps 2 --objects --preview

# 기체의 랜딩기어가 화면 모서리에 보이면 그 영역을 무시합니다 (정규화 좌표 x1,y1,x2,y2).
python3 tools/task3_autolabel/autolabel.py flight.mp4 --out labels_out --mask 0,0,0.06,0.07 0.94,0,1,0.07
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--out` | `labels_out` | 출력 디렉터리 (없으면 생성) |
| `--fps` | `2` | 초당 샘플링할 프레임 수. `0`이면 모든 프레임 |
| `--objects` | 꺼짐 | 원 위의 소형 물체(`object` 클래스)도 라벨링 |
| `--preview` | 꺼짐 | 검수용으로 박스를 그린 이미지를 `preview/`에 저장 |
| `--keep-empty` | 꺼짐 | 검출이 없는 프레임도 빈 라벨 파일과 함께 저장 |
| `--mask` | 없음 | 무시할 고정 영역(정규화 사각형, 여러 개 가능) |
| `--min-circle-area` | `0.002` | 프레임 면적 대비 최소 원 면적 |
| `--min-circularity` | `0.6` | 원형도 임계값 (화면 가장자리에 걸린 원은 자동 완화) |
| `--min-object-area` | `40` | 물체 최소 면적(픽셀) |
| `--max-object-frac` | `0.25` | 원 면적 대비 물체 최대 비율 |

## 출력 구조

```
labels_out/
├── classes.txt            # 클래스 이름 (줄 번호 = 클래스 id)
├── summary.json           # 영상별 프레임 수 / 박스 수 / 클래스별 개수
├── images/<영상이름>_<프레임번호>.jpg
├── labels/<영상이름>_<프레임번호>.txt   # "class cx cy w h" (0~1 정규화)
└── preview/...            # --preview 사용 시
```

## 클래스 (잠정)

| id | 이름 | 설명 |
| --- | --- | --- |
| 0 | `red_circle` | 빨간 원형 타겟 |
| 1 | `green_circle` | 초록 원형 타겟 |
| 2 | `blue_circle` | 파란 원형 타겟 |
| 3 | `object` | 원 위에 놓인 소형 물체 (`--objects` 사용 시) |

공식 Task3 라벨 규칙이 정해지면 `autolabel.py` 상단의 `CLASSES` 와 `HSV_RANGES` 를 맞춰
수정하면 됩니다. 물체를 색상별로 나누고 싶다면 `detect_objects` 에서 물체 영역의 평균
HSV 로 클래스를 분기하면 됩니다.

## 동작 원리

1. OpenCV 로 영상을 열고 `--fps` 간격으로 프레임을 샘플링합니다.
2. HSV 색 공간에서 채도·명도가 높은 빨강/초록/파랑 영역만 남깁니다. 타겟은 완전 포화
   페인트(S≈255)라서 바닥 보도블록·그림자(S≤225, V≤120)와 분리됩니다.
3. 각 영역의 볼록 껍질(convex hull)로 원형도와 충실도(solidity)를 검사해 원만 남깁니다.
   물체가 올려져 있거나 프레임에 잘린 원도 껍질 기준이라 통과합니다.
4. `--objects` 를 켜면 원 내부에서 원 색이 아닌 구멍을 찾아 `object` 로 라벨링합니다.
5. 결과를 YOLO 텍스트로 쓰고, `summary.json` 에 통계를 남깁니다.

자동 라벨은 초안입니다. `--preview` 이미지로 검수한 뒤 라벨링 툴(예: LabelImg, CVAT)에서
YOLO 형식으로 불러와 보정하는 것을 권장합니다.
