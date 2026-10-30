#!/usr/bin/env python3
"""Task3 학습 작업 한 번 (task3_train_server.py 가 실행한다. 직접 실행해도 된다).

  python task3_train_job.py --dataset "My First Project.v8i.yolov11.zip" --out train_runs/runs/20261030_2350_v8 \
                            [--video 원본영상.mp4] [--epochs 100] [--batch 8]

순서: zip 풀기·검사 → YOLO11n 학습 → valid/test 평가 → (점검 영상) → ONNX → 보정 이미지 1024장 → Hailo-10H 번들 zip.
진행 상황은 <out>/status.json 에 쓰고, 결과는 <out>/REPORT.md 와 그 폴더의 파일들이다.

레시피는 tools/task3_train/train.py (PR #2) 와 같다: COCO 사전학습 yolo11n.pt, imgsz 640, hsv_h 0.005 (색 = 클래스),
flipud/fliplr 0.5, 100 에폭, batch 8. 클래스 순서는 드론의 tin_detective.py NAMES 와 같아야 한다.
"""
import argparse, json, os, random, re, shutil, subprocess, sys, time, traceback, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
HAILO_DIR = os.path.join(HERE, 'hailo')
NAMES = ['Blue_Rag', 'Blue_Tin', 'Green_Rag', 'Green_Tin', 'Red_Rag', 'Red_Tin']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
PHASES = ['데이터 확인', '학습', '평가', '점검 영상', 'ONNX 변환', '보정 이미지 1024장', 'Hailo 번들']


class Status:
    """status.json 을 통째로 다시 쓴다. 윈도우에서 서버가 읽는 순간 os.replace 가 막힐 수 있어 몇 번 다시 시도한다."""

    def __init__(self, out, epochs):
        self.path = os.path.join(out, 'status.json')
        self.d = {'state': 'running', 'phase': 0, 'phases': [{'name': n, 'state': 'todo', 'note': ''} for n in PHASES],
                  'message': '', 'epoch': 0, 'epochs': epochs, 'batch_i': 0, 'batch_n': 0, 'sub': None, 'history': [],
                  'dataset': None, 'result': None, 'error': None, 'device': None,
                  'started': time.time(), 'updated': time.time(), 'train_started': None}
        self._last = 0.0
        self.save()

    def save(self, throttle=0.0):
        now = time.time()
        if throttle and now - self._last < throttle:
            return
        self._last = now; self.d['updated'] = now
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, ensure_ascii=False)
        for _ in range(40):
            try:
                os.replace(tmp, self.path); return
            except PermissionError:
                time.sleep(0.05)

    def phase(self, i, note='', msg=None):
        for k in range(i):
            if self.d['phases'][k]['state'] in ('todo', 'run'):
                self.d['phases'][k]['state'] = 'done'
        self.d['phase'] = i; self.d['phases'][i]['state'] = 'run'; self.d['phases'][i]['note'] = note
        self.d['sub'] = None
        self.d['message'] = msg if msg is not None else PHASES[i] + ' 중'
        print(f'== [{i + 1}/{len(PHASES)}] {PHASES[i]} {note}', flush=True)
        self.save()

    def done(self, i, note=None, state='done'):
        self.d['phases'][i]['state'] = state
        if note is not None:
            self.d['phases'][i]['note'] = note
        self.save()


def norm(n):
    return re.sub(r'[\s\-]+', '_', str(n).strip()).lower()


def unpack(zip_path, dst):
    """zip 을 dst 에 푼다 (../ 경로와 __MACOSX 는 버림)."""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename.replace('\\', '/')
            if name.endswith('/'):
                continue
            parts = [p for p in name.split('/') if p not in ('', '.', '..')]
            if not parts or parts[0] == '__MACOSX':
                continue
            target = os.path.join(dst, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, 'wb') as out:
                shutil.copyfileobj(src, out, 1 << 20)


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def label_of(img):
    """Ultralytics 와 같은 규칙: 경로의 마지막 images 폴더를 labels 로, 확장자를 .txt 로."""
    parts = os.path.normpath(img).split(os.sep)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'; break
    parts[-1] = os.path.splitext(parts[-1])[0] + '.txt'
    return os.sep.join(parts)


def source_key(img):
    """Roboflow 증강 사본(frame_00035_png.rf.<hash>.jpg)을 원본 단위로 묶는 키."""
    b = os.path.basename(img)
    return b.split('.rf.')[0] if '.rf.' in b else os.path.splitext(b)[0]


def find_splits(root):
    """(train, val, test) 이미지 목록과 설명. Roboflow(train/valid/test), Ultralytics(images/train), 분할 없는 zip(images/) 을 받는다."""
    def first(*cands):
        for c in cands:
            imgs = list_images(os.path.join(root, *c))
            if imgs:
                return imgs
        return []
    train = first(('train', 'images'), ('images', 'train'))
    val = first(('valid', 'images'), ('val', 'images'), ('images', 'valid'), ('images', 'val'))
    test = first(('test', 'images'), ('images', 'test'))
    how = 'zip 안의 train/valid/test 분할 그대로'
    warn = []
    if train and not val:
        # 증강 사본이 train/val 양쪽에 섞이지 않도록 원본 단위로 10% 를 val 로 뗀다.
        keys = sorted({source_key(p) for p in train}); random.Random(0).shuffle(keys)
        vk = set(keys[:max(1, len(keys) // 10)])
        val = [p for p in train if source_key(p) in vk]; train = [p for p in train if source_key(p) not in vk]
        how = 'valid 가 없어 train 의 10%(원본 단위)를 valid 로 뗌'; warn.append(how)
    if not train:
        flat = list_images(os.path.join(root, 'images'))
        if flat:
            # 라벨러가 내보낸 zip 처럼 분할이 없는 연속 프레임: 5구간으로 나눠 구간마다 뒤 20% 를 valid (무작위 분할은 누수).
            seg = max(1, len(flat) // 5); train, val = [], []
            for s in range(0, len(flat), seg):
                chunk = flat[s:s + seg]; cut = int(round(len(chunk) * 0.8))
                train += chunk[:cut]; val += chunk[cut:]
            how = '분할이 없어 시간 순 5구간 × 뒤 20% 를 valid 로 나눔'; warn.append(how)
    return train, val, test, how, warn


def count_labels(imgs):
    per = [0] * len(NAMES); boxes = 0; empty = 0; bad = 0
    for p in imgs:
        lp = label_of(p)
        if not os.path.exists(lp):
            empty += 1; continue
        n = 0
        with open(lp, encoding='utf-8', errors='replace') as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                try:
                    c = int(float(t[0]))
                except ValueError:
                    bad += 1; continue
                if 0 <= c < len(NAMES):
                    per[c] += 1; n += 1
                else:
                    bad += 1
        boxes += n; empty += (n == 0)
    return {'images': len(imgs), 'boxes': boxes, 'empty': empty, 'bad_rows': bad, 'per_class': dict(zip(NAMES, per))}


def remap_labels(imgs, mapping):
    dropped = 0
    for p in imgs:
        lp = label_of(p)
        if not os.path.exists(lp):
            continue
        rows = []
        with open(lp, encoding='utf-8', errors='replace') as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                try:
                    t[0] = str(mapping[int(float(t[0]))])
                except (ValueError, KeyError):
                    dropped += 1; continue   # 숫자가 아니거나 클래스 범위를 벗어난 줄은 버린다
                rows.append(' '.join(t))
        with open(lp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows) + ('\n' if rows else ''))
    return dropped


def prepare(a, st):
    import yaml
    st.phase(0, msg='zip 푸는 중')
    root = os.path.join(a.out, 'dataset')
    if os.path.isdir(root):
        shutil.rmtree(root)
    unpack(a.dataset, root)
    yamls = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower() in ('data.yaml', 'data.yml'):
                yamls.append(os.path.join(d, f))
    if not yamls:
        raise RuntimeError('zip 안에 data.yaml 이 없습니다. Roboflow 에서 "YOLOv11" 형식으로 내보낸 zip 인지 확인하세요.')
    yp = min(yamls, key=lambda p: p.count(os.sep)); droot = os.path.dirname(yp)
    with open(yp, encoding='utf-8') as f:
        y = yaml.safe_load(f) or {}
    names = y.get('names') or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    if not names:
        raise RuntimeError('data.yaml 에 names(클래스 목록)가 없습니다.')
    lookup = {norm(n): i for i, n in enumerate(NAMES)}
    unknown = [n for n in names if norm(n) not in lookup]
    if unknown:
        raise RuntimeError(f'드론 노드가 모르는 클래스 이름이 있습니다: {unknown}. 클래스는 {NAMES} 여섯 개여야 합니다.')
    mapping = {i: lookup[norm(n)] for i, n in enumerate(names)}
    train, val, test, how, warn = find_splits(droot)
    if not train or not val:
        raise RuntimeError('학습/검증 이미지를 찾지 못했습니다. zip 안에 train/images, valid/images 폴더가 있어야 합니다.')
    remapped = any(k != v for k, v in mapping.items())
    if remapped:
        dropped = remap_labels(train + val + test, mapping)
        warn.append(f'zip 의 클래스 순서 {names} 를 드론 순서로 바꿔 라벨 번호를 고쳐 씀')
        if dropped:
            warn.append(f'읽지 못한 라벨 줄 {dropped}개는 버림')
    lists = {}
    for k, imgs in (('train', train), ('val', val), ('test', test)):
        if imgs:
            lists[k] = os.path.join(a.out, f'{k}.txt')
            with open(lists[k], 'w', encoding='utf-8') as f:
                f.write('\n'.join(os.path.abspath(p) for p in imgs) + '\n')
    data_yaml = os.path.join(a.out, 'data.yaml')
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'path': os.path.abspath(a.out), **{k: os.path.abspath(v) for k, v in lists.items()},
                        'nc': len(NAMES), 'names': NAMES}, f, allow_unicode=True, sort_keys=False)
    counts = {k: count_labels(v) for k, v in (('train', train), ('valid', val), ('test', test)) if v}
    rf = y.get('roboflow') or {}
    ver = rf.get('version')
    tag = f'v{ver}' if ver not in (None, '') else (re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(a.dataset))[0]).strip('_')[:24] or 'run')
    if counts['train']['boxes'] == 0:
        raise RuntimeError('train 라벨에 박스가 하나도 없습니다.')
    for k, c in counts.items():
        if c['bad_rows']:
            warn.append(f'{k}: 읽지 못한 라벨 줄 {c["bad_rows"]}개는 무시')
    st.d['dataset'] = {'file': re.sub(r'^\d+_', '', os.path.basename(a.dataset)), 'tag': tag, 'roboflow': {k: rf.get(k) for k in ('workspace', 'project', 'version') if rf.get(k) is not None},
                       'names_in_zip': names, 'remapped': remapped, 'split': how, 'counts': counts, 'warnings': warn}
    note = ' / '.join(f'{k} {c["images"]}장' for k, c in counts.items())
    st.done(0, note)
    return data_yaml, lists, tag, train


def pick_device(want):
    import torch
    if want and want != 'auto':
        return want, (torch.cuda.get_device_name(0) if want not in ('cpu',) and torch.cuda.is_available() else 'CPU')
    if torch.cuda.is_available():
        return '0', torch.cuda.get_device_name(0)
    return 'cpu', 'CPU'


def train(a, st, data_yaml, device):
    from ultralytics import YOLO
    st.phase(1, msg='학습 준비 중 (처음이면 yolo11n.pt 를 내려받습니다)')
    model = YOLO(a.weights)
    cnt = {'i': 0}

    def on_epoch_start(tr):
        cnt['i'] = 0
        try:
            st.d['batch_n'] = len(tr.train_loader)
        except Exception:
            pass
        st.d['epoch'] = tr.epoch; st.d['epochs'] = tr.epochs
        if st.d['train_started'] is None:
            st.d['train_started'] = time.time()
        st.d['message'] = f'학습 중 · 에폭 {tr.epoch + 1}/{tr.epochs}'
        st.save()

    def on_batch_end(tr):
        cnt['i'] += 1; st.d['batch_i'] = cnt['i']
        st.save(throttle=1.0)

    def on_fit_epoch_end(tr):
        # ultralytics 8.4 는 마지막 에폭 뒤 final_eval(best.pt) 에서 이 콜백을 한 번 더 부른다. 새 에폭이 아니라서 건너뛴다
        if tr.epoch + 1 > tr.epochs or (st.d['history'] and st.d['history'][-1]['epoch'] >= tr.epoch + 1):
            return
        m = tr.metrics or {}
        def num(v):
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                return None
        row = {'epoch': tr.epoch + 1, 'P': num(m.get('metrics/precision(B)')), 'R': num(m.get('metrics/recall(B)')),
               'mAP50': num(m.get('metrics/mAP50(B)')), 'mAP50_95': num(m.get('metrics/mAP50-95(B)')), 't': time.time()}
        st.d['history'].append(row); st.d['epoch'] = tr.epoch + 1; st.d['batch_i'] = st.d['batch_n']
        best = max((h['mAP50'] or 0) for h in st.d['history'])
        st.d['message'] = f'학습 중 · 에폭 {tr.epoch + 1}/{tr.epochs} · mAP50 {row["mAP50"] or 0:.3f} (최고 {best:.3f})'
        st.save()

    model.add_callback('on_train_epoch_start', on_epoch_start)
    model.add_callback('on_train_batch_end', on_batch_end)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    kw = dict(data=data_yaml, epochs=a.epochs, imgsz=640, batch=a.batch, device=device, workers=a.workers,
              project=os.path.abspath(a.out), name='train', exist_ok=True,
              patience=a.epochs + 1, optimizer='auto', cos_lr=True, close_mosaic=min(10, a.epochs),
              hsv_h=0.005, hsv_s=0.5, hsv_v=0.4,          # hue 거의 고정 (색 = 클래스)
              degrees=10.0, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0,
              plots=True, verbose=True, seed=0, deterministic=True)
    if a.fraction < 1.0:
        kw['fraction'] = a.fraction
    model.train(**kw)
    wdir = os.path.join(a.out, 'train', 'weights')
    best = os.path.join(wdir, 'best.pt') if os.path.exists(os.path.join(wdir, 'best.pt')) else os.path.join(wdir, 'last.pt')
    if not os.path.exists(best):
        raise RuntimeError('학습은 끝났는데 best.pt 가 없습니다.')
    h = st.d['history']
    st.done(1, f'{len(h)} 에폭, 최고 mAP50 {max((r["mAP50"] or 0) for r in h):.3f}' if h else '')
    return best


def evaluate(a, st, best, data_yaml, lists, device):
    from ultralytics import YOLO
    st.phase(2)
    out = {}
    for split in ('val', 'test'):
        if split not in lists:
            continue
        model = YOLO(best)   # val 뒤 같은 객체를 다시 쓰면 ultralytics 8.4 에서 오류가 나는 경우가 있어 매번 새로 읽는다
        m = model.val(data=data_yaml, split=split, imgsz=640, batch=8, device=device, plots=True,
                      project=os.path.abspath(a.out), name=f'eval_{split}', exist_ok=True, verbose=False)
        key = 'valid' if split == 'val' else 'test'
        cnt = st.d['dataset']['counts'].get(key, {}).get('per_class', {})
        rows = []
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            nm = NAMES[int(c)]
            rows.append({'cls': nm, 'n': cnt.get(nm), 'P': round(float(p), 3), 'R': round(float(r), 3), 'mAP50': round(float(ap50), 3), 'mAP50_95': round(float(ap), 3)})
        mp, mr, m50, m5095 = m.box.mean_results()
        out[key] = {'images': st.d['dataset']['counts'][key]['images'], 'boxes': st.d['dataset']['counts'][key]['boxes'], 'rows': rows,
                    'all': {'P': round(float(mp), 3), 'R': round(float(mr), 3), 'mAP50': round(float(m50), 3), 'mAP50_95': round(float(m5095), 3)}}
        cm = os.path.join(a.out, f'eval_{split}', 'confusion_matrix.png')
        if os.path.exists(cm):
            shutil.copy2(cm, os.path.join(a.out, f'confusion_matrix_{key}.png'))
    note = ' / '.join(f'{k} mAP50 {v["all"]["mAP50"]:.3f}' for k, v in out.items())
    st.d['result'] = {'eval': out}
    st.done(2, note)
    return out


def review_video(a, st, best, dst):
    import cv2
    from ultralytics import YOLO
    st.phase(3, msg='점검 영상 만드는 중')
    colors = {'Blue': (255, 128, 0), 'Green': (0, 200, 0), 'Red': (0, 0, 255)}
    model = YOLO(best); names = model.names
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError('점검 영상을 열지 못했습니다 (코덱 확인)')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    raw = dst + '.raw.mp4'
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    i = 0; counts = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=640, conf=0.4, iou=0.7, verbose=False)[0]     # tin_detective 와 같은 conf 0.4
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist()); nm = names[int(b.cls[0])]; s = float(b.conf[0])
            col = colors.get(nm.split('_')[0], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1 if nm.endswith('_Tin') else 2)
            cv2.putText(frame, f'{nm} {s:.2f}', (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            counts[nm] = counts.get(nm, 0) + 1
        for th, c in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(frame, f'frame {i:06d}', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, th, cv2.LINE_AA)
        vw.write(frame); i += 1
        if i % 25 == 0:
            st.d['sub'] = [i, total]; st.d['message'] = f'점검 영상 {i}/{total or "?"} 프레임'; st.save(throttle=1.0)
    vw.release(); cap.release()
    ff = shutil.which('ffmpeg')
    if ff and subprocess.run([ff, '-y', '-loglevel', 'error', '-i', raw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                              '-movflags', '+faststart', dst]).returncode == 0:
        os.remove(raw)
    else:
        os.replace(raw, dst)   # ffmpeg 가 없으면 mp4v 그대로 (VLC·윈도우 미디어 플레이어로 재생)
    st.done(3, f'{i} 프레임')
    return dst


def export_onnx(a, st, best, dst):
    from ultralytics import YOLO
    st.phase(4)
    model = YOLO(best)
    path = model.export(format='onnx', imgsz=640, opset=11, simplify=True, dynamic=False, nms=False, batch=1)
    shutil.move(str(path), dst)
    note = ''
    try:
        import onnx
        m = onnx.load(dst); onnx.checker.check_model(m)
        names = {n.name for n in m.graph.node}
        heads = sorted({int(h) for n in names for h in re.findall(r'^/model\.(\d+)/cv[23]\.\d/cv[23]\.\d\.2/Conv$', n)})
        head = heads[-1] if heads else None
        ends = [f'/model.{head}/cv{k}.{i}/cv{k}.{i}.2/Conv' for i in range(3) for k in (2, 3)] if head else []
        missing = [n for n in ends if n not in names]
        if head is None or missing:
            raise RuntimeError(f'Hailo end node 를 ONNX 에서 찾지 못했습니다: {missing or "Detect 헤드 없음"}')
        note = f'end node OK (model.{head} = {"YOLO11" if head == 23 else "YOLOv8"})'
        with open(os.path.splitext(dst)[0] + '.end_nodes.txt', 'w', encoding='utf-8') as f:
            f.write(' '.join(ends) + '\n')
    except ImportError:
        note = 'onnx 패키지가 없어 end node 확인은 건너뜀'
    try:
        import numpy as np, onnxruntime as ort, torch
        x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
        sess = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
        y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        pm = YOLO(best).model.float().eval()
        with torch.no_grad():
            y_pt = pm(torch.from_numpy(x))
        y_pt = (y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt).numpy()
        diff = float(np.abs(y_onnx - y_pt).max())
        if not np.allclose(y_onnx, y_pt, rtol=1e-3, atol=2e-2):
            raise RuntimeError(f'ONNX 출력이 .pt 와 다릅니다 (최대 차이 {diff:.3g})')
        note += f', .pt 와 출력 일치 (최대 차이 {diff:.1e})'
    except ImportError:
        note += ', onnxruntime 이 없어 출력 대조는 건너뜀'
    st.done(4, note.strip(', '))
    return dst


def letterbox(img, s=640):
    import cv2, numpy as np
    h, w = img.shape[:2]; r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    c = np.full((s, s, 3), 114, np.uint8); top = (s - nh) // 2; left = (s - nw) // 2
    c[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return c


def make_calib(a, st, train_imgs, dst, n=1024):
    """배포와 같은 640x640 letterbox(114 회색). DFC 는 1024장 미만이면 최적화 단계를 낮추므로 꼭 1024장을 채운다."""
    import cv2, numpy as np
    st.phase(5)
    os.makedirs(dst, exist_ok=True)
    files = list(train_imgs); random.Random(0).shuffle(files)
    k = 0
    def put(img):
        nonlocal k
        p = os.path.join(dst, f'calib_{k:04d}.jpg')
        if not cv2.imwrite(p, letterbox(img), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f'보정 이미지를 쓰지 못했습니다: {p}')
        k += 1
        if k % 64 == 0:
            st.d['sub'] = [k, n]; st.save(throttle=1.0)
    for f in files[:n]:
        img = cv2.imread(f)
        if img is not None:
            put(img)
    from_video = 0
    if a.video and k < n:
        cap = cv2.VideoCapture(a.video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(total - 1, 0), n - k).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, img = cap.read()
            if ok:
                put(img); from_video += 1
        cap.release()
    flips = 0
    j = 0
    while k < n and files:
        img = cv2.imread(files[j % len(files)])
        if img is not None:
            put(cv2.flip(img, (-1, 0, 1)[(j // len(files)) % 3])); flips += 1   # 뒤집은 학습 프레임으로 마저 채움
        j += 1
    st.done(5, f'{k}장 (학습 {k - from_video - flips} + 영상 {from_video} + 뒤집기 {flips})')
    return k


def make_bundle(a, st, tag, onnx_path, calib_dir, report, dst):
    st.phase(6)
    onnx_name = os.path.basename(onnx_path)
    readme = (f'Task3 Hailo-10H 컴파일 번들 ({tag})\n\n'
              'Ubuntu PC (Hailo DFC 5.x + hailo_model_zoo 설치)에서:\n'
              f'  unzip {os.path.basename(dst)} -d bundle_{tag} && cd bundle_{tag}\n'
              f'  bash hailo/compile.sh {onnx_name} calib1024 hailo10h\n'
              '  -> hef_out/tin_detector.hef\n\n'
              '파이에서 교체 (기존 hef 백업 후):\n'
              '  cp ~/models/tin_detector.hef ~/models/tin_detector.hef.bak_$(date +%m%d)\n'
              '  cp tin_detector.hef ~/models/tin_detector.hef\n\n'
              'DFC 버전은 파이의 HailoRT 버전과 맞아야 합니다 (파이에서 hailortcli fw-control identify).\n'
              '클래스 순서: ' + ', '.join(NAMES) + ' (tin_detective.py NAMES 와 같음)\n')
    files = [(onnx_path, onnx_name), (report, 'REPORT.md')]
    end_nodes = os.path.splitext(onnx_path)[0] + '.end_nodes.txt'
    if os.path.exists(end_nodes):
        files.append((end_nodes, os.path.basename(end_nodes)))
    calib = sorted(os.listdir(calib_dir))
    tmp = dst + '.part'
    with zipfile.ZipFile(tmp, 'w') as z:
        z.writestr('README_컴파일.txt', readme, compress_type=zipfile.ZIP_DEFLATED)
        for src, arc in files:
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
        if os.path.isdir(HAILO_DIR):
            for f in sorted(os.listdir(HAILO_DIR)):
                p = os.path.join(HAILO_DIR, f)
                if os.path.isfile(p) and not f.startswith('.'):
                    with open(p, 'rb') as fh:
                        data = fh.read().replace(b'\r\n', b'\n')   # 윈도우에서 CRLF 로 받은 compile.sh 도 리눅스에서 돌게
                    zi = zipfile.ZipInfo('hailo/' + f, time.localtime()[:6]); zi.compress_type = zipfile.ZIP_DEFLATED
                    zi.external_attr = (0o755 if f.endswith('.sh') else 0o644) << 16
                    z.writestr(zi, data)
        for i, f in enumerate(calib):
            z.write(os.path.join(calib_dir, f), 'calib1024/' + f, compress_type=zipfile.ZIP_STORED)
            if i % 128 == 0:
                st.d['sub'] = [i, len(calib)]; st.save(throttle=1.0)
    os.replace(tmp, dst)
    st.done(6, f'{os.path.getsize(dst) / 1e6:.0f} MB')
    return dst


def write_report(a, st, tag, files):
    ds = st.d['dataset']; ev = (st.d.get('result') or {}).get('eval', {})
    L = [f'# Task3 YOLO11n 학습 결과 ({tag})', '']
    L.append(f'- 데이터: {ds["file"]} ({ds["split"]})')
    for k, c in ds['counts'].items():
        L.append(f'  - {k}: 이미지 {c["images"]}장, 박스 {c["boxes"]}개 ' + ', '.join(f'{n} {v}' for n, v in c['per_class'].items()))
    for w in ds['warnings']:
        L.append(f'  - 주의: {w}')
    h = st.d['history']; t0 = st.d.get('train_started') or st.d['started']
    took = (h[-1]['t'] - t0) / 60 if h else 0
    L.append(f'- 학습: {a.weights} 에서 {len(h)} 에폭, batch {a.batch}, imgsz 640, 장치 {st.d["device"]}, 학습 시간 {took:.1f}분')
    L.append(f'- 레시피: hsv_h 0.005 (색 = 클래스), hsv_s 0.5, hsv_v 0.4, degrees 10, flipud/fliplr 0.5, mosaic 1.0 (마지막 {min(10, a.epochs)} 에폭 끔)')
    for k, v in ev.items():
        L += ['', f'## {k} (이미지 {v["images"]}장, 박스 {v["boxes"]}개)', '', '| 클래스 | 박스 | P | R | mAP50 | mAP50-95 |', '|---|---|---|---|---|---|']
        for r in v['rows']:
            L.append(f'| {r["cls"]} | {r["n"] if r["n"] is not None else "-"} | {r["P"]:.3f} | {r["R"]:.3f} | {r["mAP50"]:.3f} | {r["mAP50_95"]:.3f} |')
        x = v['all']; L.append(f'| **전체** | {v["boxes"]} | {x["P"]:.3f} | {x["R"]:.3f} | {x["mAP50"]:.3f} | {x["mAP50_95"]:.3f} |')
    L += ['', '## 파일', '']
    for k, (f, what) in files.items():
        L.append(f'- `{f}`: {what}')
    L += ['', '## 다음 단계', '',
          f'1. `task3_hailo_bundle_{tag}.zip` 을 Ubuntu Hailo PC 로 옮겨 풀고 `bash hailo/compile.sh task3_yolo11n_{tag}.onnx calib1024 hailo10h`',
          '2. 나온 `hef_out/tin_detector.hef` 를 파이의 `~/models/tin_detector.hef` 로 교체 (기존 파일 백업)', '']
    p = os.path.join(a.out, 'REPORT.md')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='Roboflow YOLOv11 내보내기 zip (또는 라벨러가 저장한 YOLO zip)')
    ap.add_argument('--out', required=True, help='이 학습의 결과 폴더')
    ap.add_argument('--video', default=None, help='점검 영상과 보정 이미지 보충에 쓸 원본 영상 (선택)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='auto', help="auto(있으면 GPU) / 0 / cpu")
    ap.add_argument('--weights', default='yolo11n.pt', help='시작 가중치. 드론 hef 세대가 YOLO11n 이라 기본 yolo11n.pt')
    ap.add_argument('--fraction', type=float, default=1.0, help=argparse.SUPPRESS)   # 시험용: train 일부만
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = Status(a.out, a.epochs)
    try:
        data_yaml, lists, tag, train_imgs = prepare(a, st)
        device, devname = pick_device(a.device)
        st.d['device'] = devname; st.save()
        best = train(a, st, data_yaml, device)
        pt = os.path.join(a.out, f'task3_yolo11n_{tag}.pt'); shutil.copy2(best, pt)
        evaluate(a, st, best, data_yaml, lists, device)
        files = {'pt': (os.path.basename(pt), '학습된 가중치 (best)')}
        if a.video:
            try:
                rv = review_video(a, st, best, os.path.join(a.out, f'review_{tag}.mp4'))
                files['review'] = (os.path.basename(rv), '점검 영상 (박스 + 왼쪽 아래 프레임 번호, 원본 속도, conf 0.4)')
            except Exception as e:   # 점검 영상은 덤이라 실패해도 ONNX·번들은 계속 만든다
                traceback.print_exc()
                st.done(3, f'건너뜀: {e}', state='skip')
        else:
            st.done(3, '영상 없음', state='skip')
        onnx_path = export_onnx(a, st, best, os.path.join(a.out, f'task3_yolo11n_{tag}.onnx'))
        files['onnx'] = (os.path.basename(onnx_path), 'ONNX (opset 11, NMS 없음, 1x3x640x640)')
        calib = os.path.join(a.out, 'calib1024'); make_calib(a, st, train_imgs, calib)
        files['calib'] = ('calib1024/', 'Hailo 양자화 보정 이미지 1024장 (640 letterbox)')
        files['bundle'] = (f'task3_hailo_bundle_{tag}.zip', 'Ubuntu Hailo PC 로 옮길 번들 (ONNX + calib1024 + hailo/compile.sh)')
        files['report'] = ('REPORT.md', '이 보고서')
        report = write_report(a, st, tag, files)
        bundle = make_bundle(a, st, tag, onnx_path, calib, report, os.path.join(a.out, f'task3_hailo_bundle_{tag}.zip'))
        st.d['result'].update({'tag': tag, 'files': {k: v[0] for k, v in files.items()},
                               'compile_cmd': f'bash hailo/compile.sh {os.path.basename(onnx_path)} calib1024 hailo10h'})
        ev = st.d['result']['eval']
        st.d['state'] = 'done'; st.d['sub'] = None
        st.d['message'] = '끝났습니다. ' + ' · '.join(f'{k} mAP50 {v["all"]["mAP50"]:.3f}' for k, v in ev.items())
        st.save()
        print('완료:', bundle, flush=True)
    except BaseException as e:   # 오류도 페이지에 보이도록 status 에 남긴다
        traceback.print_exc()
        i = st.d['phase']; st.d['phases'][i]['state'] = 'error'
        st.d['state'] = 'error'; st.d['error'] = f'{type(e).__name__}: {e}'
        st.d['message'] = f'{PHASES[i]} 단계에서 멈춤'
        st.save()
        sys.exit(1)


if __name__ == '__main__':
    main()
