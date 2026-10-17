#!/usr/bin/env python3
"""학습된 best.pt 를 Hailo 모델주(hailomz) 가 받는 ONNX 로 내보낸다.

  python3 tools/task3_train/export_onnx.py runs/task3/weights/best.pt --out task3_yolov8n.onnx

- opset 11, NMS 미포함, 고정 입력 1x3x640x640, 모델주 yolov8 레시피는 head 의 6개 Conv 출력
  (/model.N/cv2.{0,1,2}/cv2.*.2/Conv = box 회귀, /model.N/cv3.{0,1,2}/cv3.*.2/Conv = 클래스, N=22 YOLOv8 / 23 YOLO11) 를
  end node 로 잘라서 쓴다. 내보낸 뒤 그 노드 이름이 있는지 확인해 hailo/compile.sh 가 쓰는 목록으로 출력한다.
- 같은 입력으로 .pt 와 ONNX 의 출력이 일치하는지(onnxruntime) 검사한다.
"""
import argparse, os, shutil, sys
import numpy as np

import re
# Hailo 모델주 레시피의 end node: Detect 헤드의 6개 Conv (box 회귀 cv2.x + 클래스 cv3.x).
# 헤드 인덱스는 YOLOv8 = model.22, YOLO11 = model.23 이라 ONNX 에서 찾아 쓴다.
def expected_end_nodes(head):
    return [f'/model.{head}/cv{k}.{i}/cv{k}.{i}.2/Conv' for i in range(3) for k in (2, 3)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('--out', default=None, help='출력 .onnx 경로 (기본: weights 옆 best.onnx)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--opset', type=int, default=11)
    a = ap.parse_args()

    from ultralytics import YOLO
    import onnx, onnxruntime as ort, torch

    model = YOLO(a.weights)
    path = model.export(format='onnx', imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=False, nms=False, batch=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        shutil.move(path, a.out); path = a.out
    m = onnx.load(path); onnx.checker.check_model(m)
    names = {n.name for n in m.graph.node}
    heads = sorted({int(h) for n in names for h in re.findall(r'^/model\.(\d+)/cv[23]\.\d/cv[23]\.\d\.2/Conv$', n)})
    head = heads[-1] if heads else 22
    EXPECTED_END_NODES = expected_end_nodes(head)
    missing = [n for n in EXPECTED_END_NODES if n not in names]
    print('ONNX:', path)
    print('입력:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print('출력:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    if missing:
        convs = sorted(n.name for n in m.graph.node if n.op_type == 'Conv' and 'model.22' in n.name)
        print('⚠ 예상 end node 가 없음:', missing, '\n  model.22 Conv 목록:', convs)
    else:
        print(f'Hailo end nodes OK (Detect 헤드 model.{head} → 모델주 레시피 {"yolov11n" if head == 23 else "yolov8n"}):', ' '.join(EXPECTED_END_NODES))
        with open(os.path.splitext(path)[0] + '.end_nodes.txt', 'w') as f:
            f.write(' '.join(EXPECTED_END_NODES) + '\n')

    # .pt 와 ONNX 출력 비교
    x = np.random.RandomState(0).rand(1, 3, a.imgsz, a.imgsz).astype(np.float32)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        y_pt = model.model(torch.from_numpy(x))
        y_pt = (y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt).numpy()
    diff = float(np.abs(y_onnx - y_pt).max())
    print(f'.pt vs ONNX 최대 차이: {diff:.2e}  shape {y_onnx.shape} (= 1 x (4+클래스) x anchors, 박스는 픽셀 단위)')
    if not np.allclose(y_onnx, y_pt, rtol=1e-3, atol=2e-2):
        sys.exit('⚠ ONNX 출력이 .pt 와 다릅니다')

if __name__ == '__main__':
    main()
