#!/usr/bin/env bash
# Motion Guard - CCTV Imou motion detection -> Telegram, untuk STB HG680P (Armbian/Debian/Ubuntu arm64)
set -e
[ "$(id -u)" -eq 0 ] || { echo "Jalankan sebagai root (sudo bash install.sh)"; exit 1; }
APP=/opt/cctv-motion
PORT=8080

echo "=============================================="
echo "   MOTION GUARD - Installer untuk HG680P"
echo "=============================================="

echo "[1/5] Install paket (ffmpeg, python, flask, numpy, pillow)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ffmpeg python3 python3-venv python3-pip python3-flask python3-numpy python3-pil ca-certificates curl
apt-get install -y python3-waitress || echo "waitress tidak tersedia, pakai server bawaan Flask"

echo "[2/5] Atur zona waktu & NTP..."
if [ "$(timedatectl show -p Timezone --value 2>/dev/null)" = "Etc/UTC" ]; then
  timedatectl set-timezone Asia/Jakarta || true
fi
timedatectl set-ntp true || true

echo "[3/5] Menulis aplikasi ke $APP ..."
id cctv >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin -d "$APP" cctv
mkdir -p "$APP/data/snaps"

cat > "$APP/app.py" <<'PYEOF'
#!/usr/bin/env python3
import os, io, json, time, uuid, glob, hmac, threading, subprocess, collections, datetime
import urllib.request, urllib.parse
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_from_directory, Response, abort

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SNAP = os.path.join(DATA, 'snaps')
CFG_FILE = os.path.join(DATA, 'config.json')
EVT_FILE = os.path.join(DATA, 'events.jsonl')
os.makedirs(SNAP, exist_ok=True)
W, H, FPS = 640, 360, 2
START = time.time()
lock = threading.Lock()
cams = {}
seen = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_cfg():
    d = {'token': '', 'chat_id': '', 'password': '', 'armed': True, 'cameras': []}
    try:
        with open(CFG_FILE) as f:
            d.update(json.load(f))
    except Exception:
        pass
    return d


cfg = load_cfg()


def save_cfg():
    tmp = CFG_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, CFG_FILE)


events = collections.deque(maxlen=500)
try:
    with open(EVT_FILE) as f:
        for l in f.readlines()[-500:]:
            events.append(json.loads(l))
    with open(EVT_FILE, 'w') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')
except Exception:
    pass


COCO = {'person': (0, 'Orang'), 'bicycle': (1, 'Sepeda'), 'car': (2, 'Mobil'), 'motorcycle': (3, 'Motor')}
DEF_OBJS = ['person', 'motorcycle']
MODEL = os.path.join(BASE, 'models', 'yolov8n.onnx')


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


class Detector:
    def __init__(s):
        s.ok, s.err, s.lock = False, '', threading.Lock()
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 3
            s.sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
            i = s.sess.get_inputs()[0]
            s.name = i.name
            s.S = i.shape[2] if isinstance(i.shape[2], int) else 416
            s.ok = True
            log('AI siap, ukuran input', s.S)
        except Exception as e:
            s.err = str(e)
            log('AI TIDAK AKTIF:', e)

    def detect(s, fr, objs, conf):
        ids = np.array([COCO[o][0] for o in objs])
        r = min(s.S / W, s.S / H)
        nw, nh = int(W * r), int(H * r)
        im = Image.fromarray(fr)
        if (nw, nh) != (W, H):
            im = im.resize((nw, nh))
        px, py = (s.S - nw) // 2, (s.S - nh) // 2
        cv = np.full((s.S, s.S, 3), 114, np.uint8)
        cv[py:py + nh, px:px + nw] = np.asarray(im)
        x = (cv.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with s.lock:
            out = s.sess.run(None, {s.name: x})[0][0]
        if out.shape[0] < out.shape[1]:
            out = out.T
        sc = out[:, 4 + ids]
        cf = sc.max(1)
        k = cf >= conf
        if not k.any():
            return []
        b, cf, cl = out[k, :4], cf[k], ids[sc[k].argmax(1)]
        xy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        xy = (xy - np.array([px, py, px, py])) / r
        res = []
        for i in cf.argsort()[::-1]:
            bx = tuple(int(v) for v in xy[i])
            if all(iou(bx, q[2]) < 0.5 for q in res):
                name = next(v[1] for v in COCO.values() if v[0] == cl[i])
                res.append((name, float(cf[i]), bx))
        return res


AI = Detector()


def tg(method, fields=None, files=None, timeout=30):
    token = cfg.get('token')
    if not token:
        return {'ok': False, 'description': 'Token belum diisi'}
    url = f'https://api.telegram.org/bot{token}/{method}'
    try:
        if files:
            b = uuid.uuid4().hex
            body = b''
            for k, v in (fields or {}).items():
                body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, (name, data, ctype) in files.items():
                body += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                         f'Content-Type: {ctype}\r\n\r\n').encode() + data + b'\r\n'
            body += f'--{b}--\r\n'.encode()
            req = urllib.request.Request(url, body, {'Content-Type': f'multipart/form-data; boundary={b}'})
        else:
            req = urllib.request.Request(url, json.dumps(fields or {}).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {'ok': False, 'description': str(e)}
    except Exception as e:
        return {'ok': False, 'description': str(e)}


def today0():
    return datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def add_event(cam, snap, pct, what='Gerakan'):
    e = {'t': int(time.time()), 'cam': cam.id, 'name': cam.c['name'], 'snap': snap, 'pct': round(pct, 1), 'what': what}
    with lock:
        events.append(e)
        with open(EVT_FILE, 'a') as f:
            f.write(json.dumps(e) + '\n')
    return e


def prune(keep=300):
    for f in sorted(glob.glob(SNAP + '/*.jpg'))[:-keep]:
        try:
            os.remove(f)
        except OSError:
            pass


def notify(cam, jpg, e):
    cid = cfg['chat_id']
    cap = (f"🚨 {e.get('what', 'Gerakan')} terdeteksi\n📷 {cam.c['name']}\n"
           f"🕒 {time.strftime('%d/%m/%Y %H:%M:%S')}")
    if cam.c.get('photo', True):
        tg('sendPhoto', {'chat_id': cid, 'caption': cap}, {'photo': ('motion.jpg', jpg, 'image/jpeg')})
    else:
        tg('sendMessage', {'chat_id': cid, 'text': cap})
    if cam.c.get('clip'):
        p = f'/tmp/clip_{cam.id}.mp4'
        try:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-rtsp_transport', 'tcp', '-i', cam.url(),
                            '-t', '8', '-an', '-c', 'copy', '-movflags', '+faststart', p],
                           timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(p, 'rb') as f:
                tg('sendVideo', {'chat_id': cid, 'caption': f"🎥 {cam.c['name']} (8 detik)"},
                   {'video': ('clip.mp4', f.read(), 'video/mp4')}, timeout=90)
            os.remove(p)
        except Exception as ex:
            log('clip error', ex)


class Cam(threading.Thread):
    def __init__(s, c):
        super().__init__(daemon=True)
        s.c, s.id = c, c['id']
        s.stop, s.proc, s.jpg = False, None, None
        s.last, s.online, s.last_motion, s.level, s.cool = 0, False, 0, 0.0, 0
        s.busy, s.nextchk = False, 0

    def url(s):
        c, q = s.c, urllib.parse.quote
        return (f"rtsp://{q(c.get('user', 'admin'), safe='')}:{q(c.get('pass', ''), safe='')}"
                f"@{c['ip']}:{c.get('port', 554)}/cam/realmonitor?channel=1&subtype={1 if c.get('sub', True) else 0}")

    def halt(s):
        s.stop = True
        try:
            s.proc.kill()
        except Exception:
            pass

    def run(s):
        n = W * H * 3
        while not s.stop:
            cmd = ['ffmpeg', '-loglevel', 'error', '-rtsp_transport', 'tcp', '-i', s.url(), '-an',
                   '-vf', f'fps={FPS},scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
            try:
                s.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except Exception as e:
                log('ffmpeg error', e)
                time.sleep(10)
                continue
            s.last = time.time()
            bg, hits, warm = None, 0, 0
            while not s.stop:
                buf = s.proc.stdout.read(n)
                if len(buf) < n:
                    break
                t = time.time()
                s.last, s.online = t, True
                fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                im = Image.fromarray(fr)
                b = io.BytesIO()
                im.save(b, 'JPEG', quality=70)
                s.jpg = b.getvalue()
                g = fr.reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3, 4), dtype=np.float32)
                if bg is None:
                    bg, warm = g, 0
                    continue
                d = np.abs(g - bg) > (40 - s.c.get('sens', 50) * 0.35)
                bg = bg * 0.85 + g * 0.15
                warm += 1
                s.level = round(float(d.mean()) * 100, 2)
                if warm < 5:
                    continue
                hits = hits + 1 if s.level >= s.c.get('area', 0.8) else 0
                if (hits >= 2 and cfg.get('armed') and not s.busy and t > s.nextchk
                        and t - s.cool > s.c.get('cooldown', 30)):
                    hits = 0
                    ys, xs = np.where(d)
                    box = (int(xs.min()) * 4, int(ys.min()) * 4, int(xs.max()) * 4 + 4, int(ys.max()) * 4 + 4)
                    objs = s.c.get('objs', DEF_OBJS)
                    if objs and AI.ok:
                        s.busy = True
                        threading.Thread(target=s.verify, args=(im, box, objs), daemon=True).start()
                    else:
                        s.cool = s.last_motion = t
                        s.trigger(im, [('Gerakan', 0, box)], 'Gerakan')
            s.online = False
            try:
                s.proc.kill()
            except Exception:
                pass
            if not s.stop:
                time.sleep(5)

    def verify(s, im, box, objs):
        try:
            dets = AI.detect(np.asarray(im), objs, s.c.get('conf', 45) / 100)
        except Exception as e:
            log('AI error, kirim sebagai gerakan biasa:', e)
            dets = [('Gerakan', 0, box)]
        try:
            if dets:
                s.cool = s.last_motion = time.time()
                s.trigger(im, dets, ', '.join(sorted({d[0] for d in dets})))
            else:
                s.nextchk = time.time() + 3
                log('Gerakan diabaikan (bukan orang/motor):', s.c['name'])
        finally:
            s.busy = False

    def trigger(s, im, dets, what):
        im = im.copy()
        dr = ImageDraw.Draw(im)
        for nm, cf, bx in dets:
            dr.rectangle(bx, outline=(255, 59, 92), width=3)
            if cf:
                dr.text((bx[0] + 4, max(0, bx[1] - 12)), f'{nm} {int(cf * 100)}%', fill=(255, 255, 0))
        dr.text((8, 6), f"{s.c['name']}  {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(255, 255, 255))
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=85)
        jpg = b.getvalue()
        name = f"{int(time.time())}_{s.id}.jpg"
        with open(os.path.join(SNAP, name), 'wb') as f:
            f.write(jpg)
        e = add_event(s, name, s.level, what)
        prune()
        log('TERDETEKSI', s.c['name'], what)
        if cfg.get('token') and cfg.get('chat_id'):
            threading.Thread(target=notify, args=(s, jpg, e), daemon=True).start()


def start_cams():
    for c in cfg['cameras']:
        if c.get('enabled', True):
            cams[c['id']] = Cam(c)
            cams[c['id']].start()


def restart():
    for c in list(cams.values()):
        c.halt()
    cams.clear()
    start_cams()


def watchdog():
    while True:
        time.sleep(5)
        for c in list(cams.values()):
            age = time.time() - c.last
            if age > 12:
                c.online = False
            if age > 25:
                try:
                    c.proc.kill()
                except Exception:
                    pass


def status_text():
    on = sum(1 for c in cams.values() if c.online)
    n = sum(1 for e in events if e['t'] >= today0())
    return (f"Status: {'🟢 AKTIF' if cfg.get('armed') else '⚪ NONAKTIF'}\nKamera online: {on}/{len(cfg['cameras'])}\n"
            f"Gerakan hari ini: {n}\n\nPerintah: /on /off /snap /status")


def reply(t):
    cid = cfg['chat_id']
    if t == '/on':
        cfg['armed'] = True
        save_cfg()
        msg = '🟢 Deteksi gerak AKTIF'
    elif t == '/off':
        cfg['armed'] = False
        save_cfg()
        msg = '⚪ Deteksi gerak dimatikan'
    elif t == '/snap':
        sent = 0
        for c in list(cams.values()):
            if c.jpg:
                tg('sendPhoto', {'chat_id': cid, 'caption': c.c['name']}, {'photo': ('s.jpg', c.jpg, 'image/jpeg')})
                sent += 1
        if sent:
            return
        msg = 'Belum ada gambar dari kamera.'
    else:
        msg = status_text()
    tg('sendMessage', {'chat_id': cid, 'text': msg})


def poll():
    off = 0
    while True:
        if not cfg.get('token'):
            time.sleep(5)
            continue
        r = tg('getUpdates', {'offset': off, 'timeout': 25}, timeout=40)
        if not r.get('ok'):
            time.sleep(5)
            continue
        for u in r['result']:
            off = u['update_id'] + 1
            m = u.get('message') or {}
            cid = str(m.get('chat', {}).get('id', ''))
            if not cid:
                continue
            seen[cid] = m['chat'].get('first_name') or m['chat'].get('title') or cid
            txt = (m.get('text') or '').strip().lower().split('@')[0]
            if not cfg.get('chat_id'):
                tg('sendMessage', {'chat_id': cid, 'text': f'Chat ID kamu: {cid}\nMasukkan di Dashboard > Pengaturan.'})
            elif cid == str(cfg['chat_id']):
                reply(txt)


def sysinfo():
    r = {}
    try:
        r['temp'] = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except Exception:
        r['temp'] = None
    try:
        m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
        r['mem'] = round(100 - m['MemAvailable'] * 100 / m['MemTotal'])
    except Exception:
        r['mem'] = None
    r['load'] = round(os.getloadavg()[0], 2)
    r['up'] = int(float(open('/proc/uptime').read().split()[0]))
    return r


app = Flask(__name__)


@app.before_request
def auth():
    pw = cfg.get('password')
    if pw:
        a = request.authorization
        if not a or not hmac.compare_digest(a.password or '', pw):
            return Response('Login diperlukan', 401, {'WWW-Authenticate': 'Basic realm="Motion Guard"'})


@app.get('/')
def index():
    return send_from_directory(BASE, 'index.html')


@app.get('/snaps/<path:f>')
def snaps(f):
    return send_from_directory(SNAP, f, max_age=86400)


@app.get('/api/live/<cid>')
def live(cid):
    c = cams.get(cid)
    if not c or not c.jpg:
        abort(404)
    return Response(c.jpg, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})


def pub(c):
    r = {k: c.get(k) for k in ('id', 'name', 'ip', 'port', 'user', 'sub', 'sens', 'area', 'cooldown', 'photo', 'clip', 'enabled')}
    r['objs'] = c.get('objs', DEF_OBJS)
    r['conf'] = c.get('conf', 45)
    cm = cams.get(c['id'])
    r['online'] = bool(cm and cm.online)
    r['level'] = cm.level if cm else 0
    return r


@app.get('/api/state')
def state():
    t0 = today0()
    hours = [0] * 24
    today = 0
    for e in events:
        if e['t'] >= t0:
            today += 1
            hours[datetime.datetime.fromtimestamp(e['t']).hour] += 1
    return jsonify(armed=cfg.get('armed', True), cams=[pub(c) for c in cfg['cameras']], today=today, hours=hours,
                   events=list(events)[::-1][:24], sys=sysinfo(), token_set=bool(cfg.get('token')),
                   chat_id=cfg.get('chat_id', ''), pw_set=bool(cfg.get('password')), ai=AI.ok)


@app.post('/api/arm')
def arm():
    cfg['armed'] = bool(request.get_json(force=True).get('armed'))
    save_cfg()
    return jsonify(ok=True)


@app.post('/api/settings')
def settings():
    j = request.get_json(force=True)
    if j.get('token'):
        cfg['token'] = j['token'].strip()
    cfg['chat_id'] = str(j.get('chat_id', '')).strip()
    if j.get('password'):
        cfg['password'] = '' if j['password'] == '-' else j['password']
    save_cfg()
    return jsonify(ok=True)


@app.post('/api/test')
def test():
    if not cfg.get('chat_id'):
        return jsonify(ok=False, description='Chat ID belum diisi')
    return jsonify(tg('sendMessage', {'chat_id': cfg['chat_id'], 'text': '✅ Motion Guard terhubung!\n' + status_text()}))


@app.get('/api/chats')
def chats():
    return jsonify(seen)


@app.post('/api/camera')
def cam_save():
    j = request.get_json(force=True)
    cid = j.get('id') or uuid.uuid4().hex[:6]
    old = next((c for c in cfg['cameras'] if c['id'] == cid), None)
    c = {'id': cid, 'name': j.get('name') or 'Kamera', 'ip': j['ip'].strip(), 'port': int(j.get('port') or 554),
         'user': j.get('user') or 'admin', 'pass': j.get('pass') or (old or {}).get('pass', ''),
         'sub': bool(j.get('sub', True)), 'sens': int(j.get('sens', 50)), 'area': float(j.get('area', 0.8)),
         'cooldown': int(j.get('cooldown', 30)), 'photo': bool(j.get('photo', True)),
         'clip': bool(j.get('clip', False)), 'enabled': bool(j.get('enabled', True)),
         'objs': [o for o in j.get('objs', DEF_OBJS) if o in COCO], 'conf': int(j.get('conf', 45))}
    if old:
        old.update(c)
    else:
        cfg['cameras'].append(c)
    save_cfg()
    restart()
    return jsonify(ok=True)


@app.delete('/api/camera/<cid>')
def cam_del(cid):
    cfg['cameras'] = [c for c in cfg['cameras'] if c['id'] != cid]
    save_cfg()
    restart()
    return jsonify(ok=True)


if __name__ == '__main__':
    start_cams()
    threading.Thread(target=poll, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    port = int(os.environ.get('PORT', 8080))
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        app.run(host='0.0.0.0', port=port, threaded=True)
PYEOF

cat > "$APP/index.html" <<'HTMLEOF'
<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion Guard</title>
<style>
:root{--bg:#0a0e1a;--card:#121829;--line:#222c45;--tx:#e8ecf8;--mut:#8591b0;--a:#6366f1;--b:#22d3ee;--ok:#22c55e;--bad:#ef4444}
*{box-sizing:border-box;margin:0}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:radial-gradient(900px 500px at 10% -10%,#312e81 0,transparent 60%),radial-gradient(800px 500px at 100% 0,#0e7490 0,transparent 55%),var(--bg);color:var(--tx);min-height:100vh;padding:20px;max-width:1300px;margin:auto}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;gap:12px;flex-wrap:wrap}
.brand{display:flex;gap:12px;align-items:center}
.logo{width:44px;height:44px;border-radius:14px;background:linear-gradient(135deg,var(--a),var(--b));display:grid;place-items:center;font-size:22px;box-shadow:0 0 24px #6366f180}
h1{font-size:20px}header p,.mut,small{color:var(--mut);font-size:12px}
.hr{display:flex;align-items:center;gap:10px;background:var(--card);border:1px solid var(--line);padding:8px 12px;border-radius:14px}
.sw{position:relative;width:46px;height:26px}.sw input{display:none}
.sw span{position:absolute;inset:0;background:#2a3350;border-radius:20px;cursor:pointer;transition:.2s}
.sw span:after{content:"";position:absolute;width:20px;height:20px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s}
.sw input:checked+span{background:linear-gradient(90deg,var(--ok),#16a34a)}.sw input:checked+span:after{left:23px}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:#1a2238;color:var(--tx);padding:7px 12px;border-radius:10px}
button:hover{border-color:var(--a)}
.pri{background:linear-gradient(135deg,var(--a),var(--b));border:0;font-weight:600;color:#fff}
.ic{padding:6px 10px;font-size:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:20px}
.card,.stat{background:rgba(18,24,41,.85);border:1px solid var(--line);border-radius:18px;padding:16px;backdrop-filter:blur(8px)}
.stat b{display:block;font-size:26px;margin-top:6px;background:linear-gradient(90deg,#fff,var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
main{display:grid;grid-template-columns:1fr 340px;gap:20px}
@media(max-width:900px){main{grid-template-columns:1fr}}
h2,h3{font-size:15px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.cam{background:var(--card);border:1px solid var(--line);border-radius:18px;overflow:hidden}
.view{position:relative;aspect-ratio:16/9;background:#05070d url() center/cover}
.view img{width:100%;height:100%;object-fit:cover;display:block}
.view:before{content:"Menunggu sinyal…";position:absolute;inset:0;display:grid;place-items:center;color:var(--mut);font-size:13px}
.view img{position:relative}
.pill{position:absolute;top:10px;left:10px;font-size:11px;font-weight:700;padding:4px 9px;border-radius:20px;z-index:2;backdrop-filter:blur(6px)}
.pill.ok{background:#22c55e30;color:#4ade80}.pill.off{background:#ef444430;color:#f87171}
.ov{position:absolute;bottom:0;left:0;right:0;padding:18px 12px 8px;background:linear-gradient(transparent,#000a);font-weight:600;z-index:2}
.cb{padding:12px}.lv{height:6px;background:#1e2742;border-radius:6px;overflow:hidden;margin-bottom:10px}
.lv i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--ok),#facc15,var(--bad));transition:width .4s}
.acts{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--mut)}
.acts button{padding:4px 9px;font-size:12px;margin-left:6px}.del:hover{border-color:var(--bad);color:var(--bad)}
.empty{grid-column:1/-1;text-align:center;padding:50px 20px;border:2px dashed var(--line);border-radius:18px}
.empty p{margin:8px 0 16px}
aside{display:flex;flex-direction:column;gap:16px}
.chart{display:flex;gap:3px;align-items:flex-end;height:90px}
.bar{flex:1;height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;font-size:9px;color:var(--mut)}
.bar i{width:100%;background:#2d3a63;border-radius:3px 3px 0 0;min-height:2px}.bar.now i{background:linear-gradient(var(--b),var(--a))}
.bar span{height:12px}
#evts{max-height:460px;overflow:auto;display:flex;flex-direction:column;gap:8px}
.ev{display:flex;gap:10px;align-items:center;text-decoration:none;color:inherit;padding:6px;border-radius:12px;border:1px solid transparent}
.ev:hover{background:#1a2238;border-color:var(--line)}
.ev img{width:84px;height:48px;object-fit:cover;border-radius:8px;background:#05070d}
.ev b{font-size:13px;display:block}
dialog{max-height:92vh;overflow:auto;background:#111729;color:var(--tx);border:1px solid var(--line);border-radius:20px;padding:22px;width:min(440px,92vw)}
dialog::backdrop{background:#000a;backdrop-filter:blur(4px)}
dialog h3{font-size:17px}
label{display:block;font-size:12px;color:var(--mut);margin:10px 0 4px}
input[type=text],input[type=password],input[type=number],select{width:100%;padding:9px 11px;background:#0a0e1a;border:1px solid var(--line);border-radius:10px;color:var(--tx);font:inherit}
input[type=range]{width:100%}
.row{display:flex;gap:10px}.row>*{flex:1}
.chk{display:flex;gap:14px;margin-top:12px;font-size:13px;flex-wrap:wrap}.chk label{margin:0;color:var(--tx);display:flex;gap:6px;align-items:center}
.fb{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}
.hint{font-size:11px;color:var(--mut);margin-top:6px;line-height:1.5}
#toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1e2742;border:1px solid var(--a);padding:10px 18px;border-radius:12px;display:none;z-index:99}
</style></head><body>
<header>
 <div class="brand"><div class="logo">◉</div><div><h1>Motion Guard</h1><p>STB HG680P · Armbian · Imou → Telegram</p></div></div>
 <div class="hr"><label class="sw" style="margin:0"><input type="checkbox" id="armed"><span></span></label><b id="armtxt">AKTIF</b><button class="ic" id="btnSet" title="Pengaturan">⚙</button></div>
</header>
<div class="stats">
 <div class="stat"><small>Gerakan hari ini</small><b id="sToday">0</b></div>
 <div class="stat"><small>Kamera online</small><b id="sCam">0 / 0</b></div>
 <div class="stat"><small>Gerakan terakhir</small><b id="sLast">—</b></div>
 <div class="stat"><small>Kesehatan STB</small><b id="sSys">—</b></div>
</div>
<main>
 <section><h2>Kamera <button class="pri" onclick="openCam()">+ Tambah</button></h2><div class="cams" id="cams"></div></section>
 <aside>
  <div class="card"><h3>Aktivitas per jam <small>hari ini</small></h3><div class="chart" id="chart"></div></div>
  <div class="card"><h3>Riwayat gerakan</h3><div id="evts"></div></div>
 </aside>
</main>

<dialog id="dCam"><h3>Kamera Imou</h3>
 <input type="hidden" id="f_id">
 <label>Nama</label><input type="text" id="f_name" placeholder="Teras depan">
 <div class="row"><div><label>IP kamera</label><input type="text" id="f_ip" placeholder="192.168.1.20"></div><div style="max-width:90px"><label>Port</label><input type="number" id="f_port" value="554"></div></div>
 <div class="row"><div><label>Username</label><input type="text" id="f_user" value="admin"></div><div><label>Password / Safety Code</label><input type="password" id="f_pass" placeholder="(kosong = tidak diubah)"></div></div>
 <label>Stream</label><select id="f_sub"><option value="1">Sub-stream (ringan, disarankan)</option><option value="0">Main-stream (berat untuk STB)</option></select>
 <label>Sensitivitas: <span id="v_sens"></span></label><input type="range" id="f_sens" min="1" max="100" oninput="v_sens.textContent=this.value">
 <div class="row"><div><label>Min. area gerak (%)</label><input type="number" id="f_area" step="0.1" value="0.8"></div><div><label>Jeda notifikasi (detik)</label><input type="number" id="f_cd" value="30"></div></div>
 <div class="chk"><label><input type="checkbox" id="f_photo" checked>Kirim foto</label><label><input type="checkbox" id="f_clip">Kirim klip 8 dtk</label><label><input type="checkbox" id="f_en" checked>Aktif</label></div>
 <label>Deteksi objek AI (kosongkan semua = kirim semua gerakan)</label>
 <div class="chk" style="margin-top:4px"><label><input type="checkbox" class="ob" value="person">🧍 Orang</label><label><input type="checkbox" class="ob" value="motorcycle">🏍 Motor</label><label><input type="checkbox" class="ob" value="car">🚗 Mobil</label><label><input type="checkbox" class="ob" value="bicycle">🚲 Sepeda</label></div>
 <label>Keyakinan AI</label><select id="f_conf"><option value="30">Sensitif (30%)</option><option value="45">Normal (45%)</option><option value="60">Ketat (60%)</option></select>
 <p class="hint">Password = <b>Safety Code</b> di stiker bawah kamera (atau password yang Anda set di Imou Life). Pastikan RTSP/ONVIF aktif bila ada opsinya. Untuk klip video, set kamera ke H.264.</p>
 <div class="fb"><button onclick="dCam.close()">Batal</button><button class="pri" onclick="saveCam()">Simpan</button></div>
</dialog>

<dialog id="dSet"><h3>Pengaturan Telegram</h3>
 <label>Bot Token (dari @BotFather)</label><input type="password" id="s_tok">
 <label>Chat ID</label><div class="row"><input type="text" id="s_chat"><button style="flex:0 0 auto" onclick="getChat()">Ambil otomatis</button></div>
 <p class="hint">Kirim pesan apa saja ke bot Anda dulu, lalu klik “Ambil otomatis”. Perintah bot: /status /snap /on /off</p>
 <label>Password dashboard (opsional)</label><input type="password" id="s_pw" placeholder="kosong = tidak diubah, ketik - untuk hapus">
 <p class="hint">Login: username apa saja, password sesuai isian ini.</p>
 <div class="fb"><button onclick="testTg()">Tes Telegram</button><button onclick="dSet.close()">Tutup</button><button class="pri" onclick="saveSet()">Simpan</button></div>
</dialog>
<div id="toast"></div>
<script>
const $=s=>document.querySelector(s),api=(u,m='GET',b)=>fetch(u,{method:m,headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined}).then(r=>r.json());
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ago=t=>{if(!t)return'—';const d=Date.now()/1e3-t;return d<60?'baru saja':d<3600?Math.floor(d/60)+' mnt lalu':d<86400?Math.floor(d/3600)+' jam lalu':Math.floor(d/86400)+' hari lalu'};
const hm=t=>new Date(t*1e3).toLocaleTimeString('id-ID',{hour:'2-digit',minute:'2-digit'});
const toast=m=>{const t=$('#toast');t.textContent=m;t.style.display='block';setTimeout(()=>t.style.display='none',3200)};
let S,sig='',esig='';
const card=c=>`<div class="cam"><div class="view"><img class="live" data-id="${c.id}" alt=""><span id="p${c.id}" class="pill off">…</span><div class="ov">${esc(c.name)}</div></div><div class="cb"><div class="lv"><i id="l${c.id}"></i></div><div class="acts"><span>${esc(c.ip)} · ${c.sub?'sub':'main'}</span><span><button onclick="openCam('${c.id}')">Edit</button><button class="del" onclick="delCam('${c.id}')">Hapus</button></span></div></div></div>`;
async function load(){
 try{S=await api('/api/state')}catch(e){return}
 $('#armed').checked=S.armed;$('#armtxt').textContent=S.armed?'AKTIF':'NONAKTIF';
 $('#sToday').textContent=S.today;$('#sCam').textContent=S.cams.filter(c=>c.online).length+' / '+S.cams.length;
 $('#sLast').textContent=ago(S.events[0]&&S.events[0].t);
 $('#sSys').textContent=(S.sys.temp!=null?S.sys.temp.toFixed(0)+'°C':'–')+' · RAM '+(S.sys.mem??'–')+'%'+(S.ai?' · AI ✓':' · AI ✗');
 const ns=JSON.stringify(S.cams.map(c=>[c.id,c.name,c.ip,c.sub]));
 if(ns!==sig){sig=ns;$('#cams').innerHTML=S.cams.length?S.cams.map(card).join(''):'<div class="empty"><div style="font-size:40px">📷</div><p>Belum ada kamera. Tambahkan kamera Imou pertama Anda.</p><button class="pri" onclick="openCam()">+ Tambah kamera</button></div>'}
 S.cams.forEach(c=>{const p=$('#p'+c.id);if(!p)return;p.className='pill '+(c.online?'ok':'off');p.textContent=c.enabled?(c.online?'● ONLINE':'OFFLINE'):'NONAKTIF';$('#l'+c.id).style.width=Math.min(100,c.level*12)+'%'});
 const es=JSON.stringify(S.events.map(e=>e.t));
 if(es!==esig){esig=es;$('#evts').innerHTML=S.events.length?S.events.map(e=>`<a class="ev" href="/snaps/${e.snap}" target="_blank"><img loading="lazy" src="/snaps/${e.snap}"><div><b>${esc(e.name)}</b><small>${hm(e.t)} · ${esc(e.what||'Gerakan')}</small></div></a>`).join(''):'<p class="mut">Belum ada gerakan terdeteksi.</p>'}
 const mx=Math.max(1,...S.hours),h=new Date().getHours();
 $('#chart').innerHTML=S.hours.map((v,i)=>`<div class="bar${i==h?' now':''}" title="${i}:00 — ${v}"><i style="height:${v/mx*100}%"></i><span>${i%6==0?i:''}</span></div>`).join('');
}
setInterval(()=>document.querySelectorAll('img.live').forEach(el=>{const i=new Image();i.onload=()=>el.src=i.src;i.src='/api/live/'+el.dataset.id+'?'+Date.now()}),1500);
$('#armed').onchange=async e=>{await api('/api/arm','POST',{armed:e.target.checked});toast(e.target.checked?'Deteksi diaktifkan':'Deteksi dimatikan');load()};
function openCam(id){
 const c=(S&&S.cams.find(x=>x.id===id))||{};
 $('#f_id').value=c.id||'';$('#f_name').value=c.name||'';$('#f_ip').value=c.ip||'';$('#f_port').value=c.port||554;$('#f_user').value=c.user||'admin';$('#f_pass').value='';
 $('#f_sub').value=c.sub===false?'0':'1';$('#f_sens').value=c.sens||50;$('#v_sens').textContent=c.sens||50;$('#f_area').value=c.area||0.8;$('#f_cd').value=c.cooldown||30;
 $('#f_photo').checked=c.photo!==false;$('#f_clip').checked=!!c.clip;$('#f_en').checked=c.enabled!==false;
 const ob=c.objs||['person','motorcycle'];document.querySelectorAll('.ob').forEach(x=>x.checked=ob.includes(x.value));$('#f_conf').value=c.conf||45;
 $('#dCam').showModal();
}
async function saveCam(){
 if(!$('#f_ip').value.trim())return toast('IP kamera wajib diisi');
 await api('/api/camera','POST',{id:$('#f_id').value,name:$('#f_name').value,ip:$('#f_ip').value,port:$('#f_port').value,user:$('#f_user').value,pass:$('#f_pass').value,sub:$('#f_sub').value==='1',sens:$('#f_sens').value,area:$('#f_area').value,cooldown:$('#f_cd').value,photo:$('#f_photo').checked,clip:$('#f_clip').checked,enabled:$('#f_en').checked,objs:[...document.querySelectorAll('.ob:checked')].map(x=>x.value),conf:$('#f_conf').value});
 $('#dCam').close();toast('Kamera disimpan, menyambung…');setTimeout(load,800);
}
async function delCam(id){if(confirm('Hapus kamera ini?')){await api('/api/camera/'+id,'DELETE');load()}}
$('#btnSet').onclick=()=>{$('#s_tok').value='';$('#s_tok').placeholder=S&&S.token_set?'•••••• tersimpan (kosong = tidak diubah)':'123456:ABC...';$('#s_chat').value=S?S.chat_id:'';$('#s_pw').value='';$('#dSet').showModal()};
async function saveSet(){await api('/api/settings','POST',{token:$('#s_tok').value,chat_id:$('#s_chat').value,password:$('#s_pw').value});$('#dSet').close();toast('Pengaturan disimpan');load()}
async function getChat(){const c=await api('/api/chats'),k=Object.keys(c);if(k.length){$('#s_chat').value=k[0];toast('Chat ID: '+k[0]+' ('+c[k[0]]+')')}else toast('Belum ada pesan. Simpan token, kirim pesan ke bot, coba lagi.')}
async function testTg(){await saveSet();const r=await api('/api/test','POST');toast(r.ok?'✅ Pesan tes terkirim':'❌ '+(r.description||'gagal'))}
load();setInterval(load,3000);
</script></body></html>
HTMLEOF

echo "[AI] Menyiapkan deteksi objek (orang & motor)..."
PY=/usr/bin/python3
mkdir -p "$APP/models"
ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "x86_64" ]; then
  if [ ! -x "$APP/venv/bin/python" ]; then
    python3 -m venv --system-site-packages "$APP/venv" </dev/null || true
  fi
  if [ -x "$APP/venv/bin/python" ] && "$APP/venv/bin/pip" install --no-cache-dir onnxruntime </dev/null; then
    PY="$APP/venv/bin/python"
  else
    echo "!! Gagal memasang onnxruntime. AI nonaktif, semua gerakan akan dikirim."
  fi
else
  echo "!! Arsitektur $ARCH tidak didukung onnxruntime. AI nonaktif, semua gerakan akan dikirim."
fi

if [ "$PY" != "/usr/bin/python3" ]; then
  MODEL="$APP/models/yolov8n.onnx"
  check_model() {
    "$PY" - "$1" <<'PYC'
import sys, numpy as np, onnxruntime as ort
s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
i = s.get_inputs()[0]
S = i.shape[2] if isinstance(i.shape[2], int) else 320
o = s.run(None, {i.name: np.zeros((1, 3, S, S), np.float32)})[0]
assert o.ndim == 3 and 84 in o.shape[1:], o.shape
PYC
  }
  if [ -s "$MODEL" ] && check_model "$MODEL" </dev/null; then
    echo "Model AI sudah ada dan valid."
  else
    rm -f "$MODEL"
    for U in \
      "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx" \
      "https://huggingface.co/Quazim0t0/yolov8-onnx/resolve/main/yolov8n.onnx" \
      "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"; do
      echo "Mengunduh model: $U"
      if curl -fL --retry 2 -o "$MODEL.tmp" "$U" </dev/null && check_model "$MODEL.tmp" </dev/null; then
        mv "$MODEL.tmp" "$MODEL"
        echo "Model AI siap."
        break
      fi
      rm -f "$MODEL.tmp"
    done
    [ -s "$MODEL" ] || echo "!! Model gagal diunduh/valid. AI nonaktif sampai file yolov8n.onnx ada di $MODEL"
  fi
fi

echo "[4/5] Membuat service systemd..."
cat > /etc/systemd/system/cctv-motion.service <<SVCEOF
[Unit]
Description=Motion Guard - CCTV Motion Detection to Telegram
After=network-online.target
Wants=network-online.target

[Service]
User=cctv
WorkingDirectory=/opt/cctv-motion
ExecStart=$PY /opt/cctv-motion/app.py
Restart=always
RestartSec=5
Nice=5

[Install]
WantedBy=multi-user.target
SVCEOF

chown -R cctv:cctv "$APP"
chmod 700 "$APP/data"

echo "[5/5] Menjalankan service..."
systemctl daemon-reload
systemctl enable cctv-motion >/dev/null 2>&1
systemctl restart cctv-motion
sleep 2

IP=$(hostname -I | awk '{print $1}')
echo
echo "=============================================="
echo " SELESAI!  Buka dashboard di browser:"
echo "   http://$IP:$PORT"
echo
echo " Perintah berguna:"
echo "   systemctl status cctv-motion      (cek status)"
echo "   journalctl -u cctv-motion -f      (lihat log)"
echo "   systemctl restart cctv-motion     (restart)"
echo "=============================================="
