#!/usr/bin/env python3
"""autolabel.py 출력(images/, labels/, data.yaml)을 시간 구간 기준으로 train/val 로 나눈다.

연속 프레임을 무작위로 섞어 나누면 거의 같은 프레임이 train 과 val 양쪽에 들어가(누수)
val 지표가 부풀려진다. 그래서 프레임을 이름(= 프레임 번호) 순으로 정렬한 뒤
`--segments` 개의 연속 구간으로 자르고, 각 구간의 뒤쪽 `--val-frac` 만큼을 val 로 쓴다.
(비행 단계마다 고도·배경이 다르므로 한 구간만 통째로 빼는 것보다 모든 단계가 val 에 들어간다.)

원본 폴더는 건드리지 않고 `--out` 아래에 images/{train,val}, labels/{train,val} 로 복사(또는 심볼릭 링크)한다.

사용:
  python3 tools/task3_train/prepare_split.py task3_labels_2fps --out dataset_split
"""
import argparse, os, re, shutil, sys, yaml

def frame_key(name):
    m = re.findall(r'(\d+)', name)
    return (name.rsplit('_', 1)[0], int(m[-1]) if m else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='autolabel 출력 폴더 (images/, labels/, data.yaml)')
    ap.add_argument('--out', default='dataset_split')
    ap.add_argument('--segments', type=int, default=5, help='시간 구간 수')
    ap.add_argument('--val-frac', type=float, default=0.2, help='각 구간에서 val 로 뺄 비율')
    ap.add_argument('--gap', type=int, default=1, help='train/val 경계에서 버릴 프레임 수(누수 완화)')
    ap.add_argument('--link', action='store_true', help='복사 대신 심볼릭 링크')
    a = ap.parse_args()

    src_img = os.path.join(a.src, 'images'); src_lab = os.path.join(a.src, 'labels')
    names = sorted((f for f in os.listdir(src_img) if f.lower().endswith(('.jpg', '.png'))), key=frame_key)
    if not names:
        sys.exit(f'이미지 없음: {src_img}')
    with open(os.path.join(a.src, 'data.yaml')) as f:
        meta = yaml.safe_load(f)

    train, val = [], []
    n = len(names); seg = n / a.segments
    for s in range(a.segments):
        lo, hi = int(round(s * seg)), int(round((s + 1) * seg))
        chunk = names[lo:hi]
        k = int(round(len(chunk) * a.val_frac))
        cut = len(chunk) - k
        train += chunk[:max(0, cut - a.gap)]
        val += chunk[cut:]
    for split, lst in (('train', train), ('val', val)):
        di = os.path.join(a.out, 'images', split); dl = os.path.join(a.out, 'labels', split)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for nm in lst:
            lab = os.path.splitext(nm)[0] + '.txt'
            for s_, d_ in ((os.path.join(src_img, nm), os.path.join(di, nm)),
                           (os.path.join(src_lab, lab), os.path.join(dl, lab))):
                if not os.path.exists(s_):
                    continue
                if os.path.lexists(d_):
                    os.remove(d_)
                if a.link:
                    os.symlink(os.path.abspath(s_), d_)
                else:
                    shutil.copy2(s_, d_)
    out_yaml = {'path': os.path.abspath(a.out), 'train': 'images/train', 'val': 'images/val',
                'nc': len(meta['names']), 'names': list(meta['names'])}
    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(a.out, 'split.txt'), 'w') as f:
        f.write('# train\n' + '\n'.join(train) + '\n# val\n' + '\n'.join(val) + '\n')
    print(f'총 {n} 프레임 → train {len(train)} / val {len(val)} (구간 {a.segments}, 경계 gap {a.gap})')
    print('data.yaml:', os.path.join(a.out, 'data.yaml'))

if __name__ == '__main__':
    main()
