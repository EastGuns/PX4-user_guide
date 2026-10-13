# Task3 자동 라벨링 (`autolabel.py`)

드론 하향 카메라 영상에서 프레임을 뽑아 **색상 판(Rag)** 과 그 위에 놓인 **캔(Tin)** 의
바운딩 박스를 자동으로 만들어 줍니다. 클래스는 Task3 검출기(`tin_detector.hef`,
`tin_detective.py` 의 `NAMES`)와 같은 6개이며 순서도 `data.yaml` 과 동일합니다.
결과는 YOLO 텍스트 형식이고, 원본 영상은 절대 수정하지 않고 모든 출력은 `--out`
디렉터리에만 씁니다.

## 설치

```bash
pip install -r tools/task3_autolabel/requirements.txt
```

## 사용법

```bash
# 영상 하나 또는 디렉터리(하위 폴더까지 검색)를 입력으로 받습니다.
python3 tools/task3_autolabel/autolabel.py data/videos --out labels_out --fps 2 --tins --preview

# 기체의 랜딩기어가 화면 모서리에 보이면 그 영역을 무시합니다 (정규화 좌표 x1,y1,x2,y2).
python3 tools/task3_autolabel/autolabel.py flight.mp4 --out labels_out --tins --mask 0,0,0.1,0.1 0.9,0,1,0.1
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--out` | `labels_out` | 출력 디렉터리 (없으면 생성) |
| `--fps` | `2` | 초당 샘플링할 프레임 수. `0`이면 모든 프레임 |
| `--tins` | 꺼짐 | 판 위에 놓인 캔(`*_Tin`)도 라벨링. 끄면 판(`*_Rag`)만 |
| `--free-tins` | 꺼짐 | 판 밖(보드·바닥)에 있는 캔도 라벨링. 햇빛 받은 붉은 보도블록이 `Red_Tin` 으로 잡힐 수 있어 기본은 꺼짐 |
| `--preview` | 꺼짐 | 검수용으로 박스를 그린 이미지를 `preview/`에 저장 |
| `--keep-empty` | 꺼짐 | 검출이 없는 프레임도 빈 라벨 파일과 함께 저장 |
| `--mask` | 없음 | 무시할 고정 영역(정규화 사각형, 여러 개 가능) |
| `--min-pad-area` | `0.002` | 프레임 면적 대비 최소 판 면적 |
| `--min-circularity` | `0.6` | 판 원형도 임계값 (화면 가장자리에 걸린 판은 자동 완화) |
| `--min-tin-area` | `40` | 캔 최소 면적(픽셀) |
| `--max-tin-frac-pad` | `0.25` | 판 면적 대비 캔 최대 비율 |
| `--max-tin-frac` | `0.004` | `--free-tins` 일 때 프레임 면적 대비 캔 최대 비율 |

## 출력 구조

```
labels_out/
├── classes.txt            # 클래스 이름 (줄 번호 = 클래스 id)
├── data.yaml              # Ultralytics 형식 (names 순서 동일, train/val 은 둘 다 images/)
├── summary.json           # 영상별 프레임 수 / 박스 수 / 클래스별 개수
├── images/<영상이름>_<프레임번호>.jpg
├── labels/<영상이름>_<프레임번호>.txt   # "class cx cy w h" (0~1 정규화)
└── preview/...            # --preview 사용 시
```

## 클래스

| id | 이름 | 설명 |
| --- | --- | --- |
| 0 | `Blue_Rag` | 파란 판 |
| 1 | `Blue_Tin` | 파란 캔 |
| 2 | `Green_Rag` | 초록 판 |
| 3 | `Green_Tin` | 초록 캔 |
| 4 | `Red_Rag` | 빨간 판 |
| 5 | `Red_Tin` | 빨간 캔 |

클래스 순서는 `autolabel.py` 상단 `CLASSES` 에 있고, 학습용 `data.yaml` 의 `names` 와 반드시 같아야 합니다.

## 동작 원리

1. OpenCV 로 영상을 열고 `--fps` 간격으로 프레임을 샘플링합니다.
2. HSV 색 공간에서 채도·명도가 높은 빨강/초록/파랑 영역만 남깁니다. 판은 완전 포화
   페인트(S≈255)라서 바닥 보도블록·그림자(S≤225, V≤120)와 분리됩니다.
3. 각 영역의 볼록 껍질(convex hull)로 원형도와 충실도(solidity)를 검사해 판만 남깁니다.
   캔이 올려져 있거나 프레임에 잘린 판도 껍질 기준이라 통과하고, 다리 그림자로 쪼개진
   같은 색 조각은 하나의 박스로 합칩니다.
4. `--tins` 를 켜면 판 내부에서 판 색이 아닌 둥근 구멍을 찾고, 그 픽셀의 색상(hue) 투표로
   캔 색을 정해 `Red_Tin` / `Green_Tin` / `Blue_Tin` 을 붙입니다. 어두운 캔도 hue 로 구분됩니다.
   구멍에는 캔 그림자도 같이 잡히므로, 판과 색이 다른 캔은 판 색이 아닌 픽셀만 박스로 삼고,
   판과 같은 색 캔은 영상에서 학습한 그림자 방향(해 방향)의 반대쪽 끝 한 캔 폭만 박스로 삼습니다.
5. 결과를 YOLO 텍스트로 쓰고, `summary.json` 에 통계를 남깁니다.

## 한계 (검수 필요)

- 판 밖에 놓인 캔은 기본으로 라벨링하지 않습니다 (`--free-tins` 로 켤 수 있지만 붉은 보도블록 오검출 주의).
- 그림자가 판을 가로지르면 그 부분이 캔으로 잡힐 수 있습니다. 둥글지 않은 구멍은 걸러내지만 완벽하지 않습니다.
- 판과 같은 색 캔의 그림자 분리는 같은 영상에 다른 색 캔이 한 번이라도 보여야 (그림자 방향을 알 수 있어야) 동작합니다.
- 랜딩기어처럼 화면에 항상 보이는 빨간 부품은 `--mask` 로 제외해야 합니다.

자동 라벨은 초안입니다. `--preview` 이미지로 검수한 뒤 라벨링 툴(예: LabelImg, CVAT)에서
YOLO 형식으로 불러와 보정하는 것을 권장합니다.
