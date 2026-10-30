#!/usr/bin/env python3
"""Task3 클릭 라벨러의 "PC에서 학습" 패널을 켜는 로컬 학습 서버 (윈도우는 학습서버_시작.bat 을 더블클릭).

  python task3_train_server.py [--port 8765] [--work 결과폴더] [--no-browser]

- 이 PC 의 브라우저에만 열린다 (127.0.0.1). 라벨러 페이지를 http://127.0.0.1:8765 로 열어 주고,
  페이지에서 고른 Roboflow zip 을 받아 task3_train_job.py 를 따로 실행해 학습한다 (GPU 가 있으면 GPU).
- 결과는 <work>/runs/<날짜_시각_버전>/ 에 쌓인다: REPORT.md, task3_yolo11n_<버전>.pt/.onnx, calib1024/,
  task3_hailo_bundle_<버전>.zip, (점검 영상). 서버 창을 닫으면 서버와 진행 중인 학습이 함께 끝난다.
- 필요한 것: 파이썬 3.9+, ultralytics (GPU 학습은 CUDA 용 torch). 서버 자체는 표준 라이브러리만 쓴다.
"""
import argparse, http.server, json, os, re, shutil, signal, socket, subprocess, sys, threading, time, urllib.parse, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.normpath(os.path.join(HERE, '..', 'click_labeler.html'))
JOB = os.path.join(HERE, 'task3_train_job.py')
API_VERSION = 1
WIN = os.name == 'nt'


class App:
    def __init__(self, work, port):
        self.work = os.path.abspath(work); self.port = port
        self.uploads = os.path.join(self.work, 'uploads'); self.runs = os.path.join(self.work, 'runs')
        os.makedirs(self.uploads, exist_ok=True); os.makedirs(self.runs, exist_ok=True)
        self.lock = threading.Lock()
        self.proc = None; self.run = None; self.stopped = False; self.log = None
        self.env = None
        self.last_status = {}
        runs = self.list_runs()
        self.run = runs[0] if runs else None   # 서버를 다시 켜도 마지막 결과가 보이게

    def list_runs(self):
        return sorted((d for d in os.listdir(self.runs) if os.path.isfile(os.path.join(self.runs, d, 'status.json'))), reverse=True)

    def check_env(self):
        code = ('import json,sys\nr={"python":sys.version.split()[0]}\n'
                'try:\n import torch;r["torch"]=torch.__version__;r["cuda"]=bool(torch.cuda.is_available());r["gpu"]=torch.cuda.get_device_name(0) if r["cuda"] else None\n'
                'except Exception as e: r["torch_error"]=str(e)[:200]\n'
                'try:\n import ultralytics;r["ultralytics"]=ultralytics.__version__\n'
                'except Exception as e: r["ultralytics_error"]=str(e)[:200]\n'
                'print(json.dumps(r))')
        try:
            out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=180, env=child_env()).stdout.strip().splitlines()
            self.env = json.loads(out[-1]) if out else {'error': '환경 확인 실패'}
        except Exception as e:
            self.env = {'error': str(e)}
        e = self.env
        if e.get('ultralytics'):
            dev = f'GPU {e["gpu"]}' if e.get('cuda') else 'GPU 없음 → CPU 로 학습 (매우 느림)'
            print(f'환경: python {e.get("python")} · torch {e.get("torch")} · ultralytics {e.get("ultralytics")} · {dev}', flush=True)
        else:
            print('⚠ ultralytics 를 불러오지 못했습니다:', e.get('ultralytics_error') or e.get('torch_error') or e.get('error'), flush=True)
            print('   설치: pip install ultralytics   (GPU 용 torch 는 https://pytorch.org 의 안내대로 CUDA 빌드로)', flush=True)

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def run_dir(self, run):
        if not run or not re.fullmatch(r'[\w\-.]+', run):
            return None
        d = os.path.join(self.runs, run)
        return d if os.path.isdir(d) else None

    def status(self):
        with self.lock:
            run = self.run
            alive = self.busy()
            code = self.proc.poll() if self.proc is not None else None
        if not run:
            return {'state': 'idle', 'run': None}
        d = os.path.join(self.runs, run); p = os.path.join(d, 'status.json')
        st = self.last_status.get(run) or {}
        try:
            with open(p, encoding='utf-8') as f:
                st = json.load(f)
            self.last_status[run] = st
        except Exception:
            pass   # 쓰는 중이면 직전 것을 쓴다
        st = dict(st)
        st['run'] = run; st['run_dir'] = d; st['alive'] = alive
        if st.get('state') == 'running' and not alive:
            if self.stopped and run == self.run:
                st['state'] = 'stopped'; st['message'] = '중지했습니다.'
            else:
                st['state'] = 'error'
                st['error'] = st.get('error') or (f'학습 프로세스가 끝났습니다 (종료 코드 {code}). 로그를 확인하세요.' if code is not None else '서버가 다시 켜져서 이전 학습은 끊겼습니다.')
        st['log'] = tail(os.path.join(d, 'log.txt'))
        return st

    def start(self, ds, vid, epochs, batch, fraction):
        with self.lock:
            if self.busy():
                raise ValueError('이미 학습 중입니다.')
            dsp = os.path.join(self.uploads, ds)
            if not re.fullmatch(r'[\w\-. ()\[\]가-힣]+', ds) or not os.path.isfile(dsp):
                raise ValueError('올린 데이터 zip 을 찾지 못했습니다. 다시 골라 주세요.')
            vp = None
            if vid:
                vp = os.path.join(self.uploads, vid)
                if not re.fullmatch(r'[\w\-. ()\[\]가-힣]+', vid) or not os.path.isfile(vp):
                    raise ValueError('올린 점검 영상을 찾지 못했습니다.')
            m = re.search(r'\.v(\d+)i?\.', ds)
            run = time.strftime('%Y%m%d_%H%M%S') + (f'_v{m.group(1)}' if m else '')
            d = os.path.join(self.runs, run); os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, '-u', JOB, '--dataset', dsp, '--out', d, '--epochs', str(epochs), '--batch', str(batch)]
            if vp:
                cmd += ['--video', vp]
            if fraction and 0 < fraction < 1:
                cmd += ['--fraction', str(fraction)]
            self.log = open(os.path.join(d, 'log.txt'), 'wb')
            kw = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {'start_new_session': True}
            # 작업 폴더를 결과 폴더 위로 두면 yolo11n.pt 를 한 번만 내려받고 다음 학습에서 다시 쓴다
            self.proc = subprocess.Popen(cmd, cwd=self.work, stdout=self.log, stderr=subprocess.STDOUT, env=child_env(), **kw)
            self.run = run; self.stopped = False
            print(f'학습 시작: {run}  (에폭 {epochs}, batch {batch}, 데이터 {ds}{", 영상 " + vid if vid else ""})', flush=True)
            return run

    def stop(self):
        with self.lock:
            if not self.busy():
                return False
            self.stopped = True; pid = self.proc.pid
        if WIN:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)   # 데이터로더 자식 프로세스까지
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print('학습을 중지했습니다.', flush=True)
        return True


def child_env():
    e = dict(os.environ)
    e.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # 아나콘다 base 에서 OpenMP 중복 로드로 죽는 것 방지
    e['PYTHONUTF8'] = '1'; e['PYTHONIOENCODING'] = 'utf-8'; e['PYTHONUNBUFFERED'] = '1'
    return e


def tail(path, n=40, maxbytes=64 * 1024):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - maxbytes)); data = f.read()
    except OSError:
        return []
    lines = []
    for raw in data.decode('utf-8', 'replace').split('\n'):
        seg = raw.split('\r'); s = next((x for x in reversed(seg) if x.strip()), '')   # 진행 막대(\r)는 마지막 상태만
        s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s).rstrip()
        if s:
            lines.append(s)
    return lines[-n:]


def safe_name(name, size):
    base = os.path.basename(name.replace('\\', '/')) or 'upload'
    base = re.sub(r'[^\w\-. ()\[\]가-힣]', '_', base)[:120]
    return f'{int(size)}_{base}'


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'Task3Trainer/1'
    app = None

    def log_message(self, fmt, *args):
        pass   # 폴링 요청까지 창에 찍히면 시끄럽다

    # --- 보안: 이 PC 의 이 주소에서 연 페이지만 받는다 (DNS rebinding·다른 사이트 요청 차단) ---
    def allowed_hosts(self):
        p = self.app.port
        return {f'127.0.0.1:{p}', f'localhost:{p}'}

    def guard(self, post=False):
        if self.headers.get('Host', '') not in self.allowed_hosts():
            self.send_json({'error': 'host not allowed'}, 403); return False
        if post:
            origin = self.headers.get('Origin')
            if self.headers.get('X-T3') != '1' or (origin and origin.replace('http://', '') not in self.allowed_hosts()):
                self.send_json({'error': 'origin not allowed'}, 403); return False
        return True

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n).decode('utf-8') or '{}') if n else {}

    def do_GET(self):
        if not self.guard():
            return
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ('/', '/index.html', '/click_labeler.html'):
            try:
                with open(PAGE, 'rb') as f:
                    body = f.read()
            except OSError:
                self.send_json({'error': f'라벨러 파일이 없습니다: {PAGE}'}, 404); return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body); return
        if u.path == '/api/ping':
            self.send_json({'ok': True, 'api': API_VERSION, 'env': self.app.env, 'busy': self.app.busy(), 'work': self.app.work}); return
        if u.path == '/api/status':
            self.send_json(self.app.status()); return
        if u.path == '/api/has':
            name = safe_name(q.get('name', [''])[0], q.get('size', ['0'])[0]); p = os.path.join(self.app.uploads, name)
            ok = os.path.isfile(p) and os.path.getsize(p) == int(q.get('size', ['0'])[0] or 0)
            self.send_json({'id': name if ok else None}); return
        if u.path == '/api/file':
            d = self.app.run_dir(q.get('run', [''])[0]); rel = q.get('f', [''])[0]
            p = os.path.normpath(os.path.join(d, rel)) if d and rel else None
            if not p or not p.startswith(d + os.sep) or not os.path.isfile(p):
                self.send_json({'error': 'file not found'}, 404); return
            size = os.path.getsize(p); fn = os.path.basename(p)
            ctype = {'.md': 'text/markdown; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.mp4': 'video/mp4', '.txt': 'text/plain; charset=utf-8'}.get(os.path.splitext(fn)[1].lower(), 'application/octet-stream')
            self.send_response(200)
            self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(size))
            if q.get('inline', ['0'])[0] != '1':
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + urllib.parse.quote(fn))
            self.end_headers()
            with open(p, 'rb') as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
            return
        self.send_json({'error': 'not found'}, 404)

    def do_POST(self):
        if not self.guard(post=True):
            return
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == '/api/upload':
                size = int(self.headers.get('Content-Length') or 0)
                if size <= 0:
                    raise ValueError('빈 파일입니다.')
                name = safe_name(q.get('name', [''])[0], size); p = os.path.join(self.app.uploads, name); tmp = p + '.part'
                left = size
                with open(tmp, 'wb') as f:
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk); left -= len(chunk)
                if left:
                    os.remove(tmp); raise ValueError('업로드가 중간에 끊겼습니다.')
                os.replace(tmp, p)
                self.send_json({'id': name}); return
            if u.path == '/api/train':
                j = self.read_json()
                epochs = max(1, min(1000, int(j.get('epochs') or 100))); batch = max(1, min(128, int(j.get('batch') or 8)))
                run = self.app.start(str(j.get('dataset') or ''), str(j.get('video') or '') or None, epochs, batch, float(j.get('fraction') or 0))
                self.send_json({'run': run}); return
            if u.path == '/api/stop':
                self.send_json({'stopped': self.app.stop()}); return
            if u.path == '/api/open':
                j = self.read_json(); d = self.app.run_dir(str(j.get('run') or ''))
                if not d:
                    raise ValueError('결과 폴더가 없습니다.')
                if WIN:
                    os.startfile(d)   # 탐색기로 열기
                else:
                    subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.send_json({'ok': True, 'path': d}); return
        except ValueError as e:
            self.send_json({'error': str(e)}, 400); return
        except Exception as e:
            self.send_json({'error': f'{type(e).__name__}: {e}'}, 500); return
        self.send_json({'error': 'not found'}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--work', default=os.path.join(os.path.dirname(PAGE), 'train_runs'), help='업로드와 학습 결과를 둘 폴더')
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    srv = None
    for port in range(a.port, a.port + 11):
        try:
            srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler); break
        except OSError:
            continue
    if srv is None:
        sys.exit(f'포트 {a.port}~{a.port + 10} 이 모두 사용 중입니다.')
    app = App(a.work, port); Handler.app = app; srv.daemon_threads = True
    url = f'http://127.0.0.1:{port}/'
    print(f'Task3 학습 서버: {url}   (이 창을 닫으면 서버와 진행 중인 학습이 끝납니다)', flush=True)
    print(f'결과 폴더: {app.runs}', flush=True)
    threading.Thread(target=app.check_env, daemon=True).start()
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == '__main__':
    main()
