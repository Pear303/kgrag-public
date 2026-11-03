# -*- coding: utf-8 -*-
"""KG-RAG 服务公共工具：无窗口启动、端口/进程检查、.env 读取。

只依赖标准库；既能被 service_control.py 调用，也能被 setup_wizard.py 复用。
"""
import ctypes
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATE_NO_WINDOW = 0x08000000
DEFAULT_API_PORT = 8001
DEFAULT_UI_PORT = 7860
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_NEO4J_PORT = 7690
DEFAULT_NEO4J_SERVICE = "neo4j-pump"
DEFAULT_LOG_DIR = ROOT / "logs" / "service"
WEBUI_DIR = ROOT / "webui"
WEBUI_SRC = WEBUI_DIR / "src"
WEBUI_DIST = WEBUI_DIR / "dist"
WEBUI_INDEX = WEBUI_DIST / "index.html"
WEBUI_TITLE = "KGRAG 工业故障推理引擎"

DEFAULT_REQUIREMENTS = [
    ("ragas", "ragas"),
    ("llama_index", "llama-index"),
    ("langchain_core", "langchain-core"),
    ("langchain_ollama", "langchain-ollama"),
    ("langchain_openai", "langchain-openai"),
    ("openai", "openai"),
    ("ollama", "ollama"),
    ("neo4j", "neo4j"),
    ("networkx", "networkx"),
    ("pyvis", "pyvis"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("jieba", "jieba"),
    ("nltk", "nltk"),
    ("datasets", "datasets"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("gradio", "gradio"),
    ("requests", "requests"),
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
    ("pdfplumber", "pdfplumber"),
    ("docx", "python-docx"),
    ("pytest", "pytest"),
]


def decode_bytes(data):
    if not data:
        return ""
    for enc in ("utf-8", "mbcs", "gbk"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", "replace")


def run_hidden(cmd, timeout=30, cwd=None, env=None):
    """运行命令并隐藏窗口，返回 (rc, stdout+stderr 文本)。"""
    try:
        p = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd or ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
            timeout=timeout,
        )
        return p.returncode, decode_bytes(p.stdout)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 127, "%s: %s" % (type(e).__name__, e)


def read_env_file(path=None):
    """读取 KEY=VALUE；只用于取值，不打印任何值。"""
    p = Path(path or (ROOT / ".env"))
    out = {}
    if not p.exists():
        return out
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def get_setting(key, default=None, env_map=None):
    val = os.environ.get(key)
    if val not in (None, ""):
        return val
    if env_map is None:
        env_map = read_env_file()
    return env_map.get(key, default)


def env_summary_keys(env_map):
    """只返回键名，便于诊断而不泄漏值。"""
    return sorted(env_map.keys())


def _version_tuple(text):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def _python_probe(path):
    rc, out = run_hidden(
        [path, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
        timeout=10,
    )
    if rc != 0:
        return None
    return _version_tuple(out)


def find_python():
    """优先项目约定的 3.10，其次 PATH/当前解释器。返回 (path, version_text)。"""
    candidates = []
    local_app = os.environ.get("LOCALAPPDATA", "")
    for c in (
        os.environ.get("KGRAG_PYTHON"),
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(Path(local_app) / "Programs" / "Python" / "Python310" / "python.exe")
        if local_app else None,
        r"C:\Program Files\Python310\python.exe",
        shutil.which("python"),
        str(Path(sys.executable).with_name("python.exe")),
        sys.executable,
    ):
        if c and c not in candidates:
            candidates.append(c)
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        ver = _python_probe(path)
        if ver and (3, 10) <= ver[:2] < (3, 13):
            return path, ".".join(str(x) for x in ver)
    return None, None


def find_pythonw(python_path=None):
    if python_path:
        cand = Path(python_path).with_name("pythonw.exe")
        if cand.exists():
            return str(cand)
    for c in (shutil.which("pythonw"),):
        if c and Path(c).exists():
            return c
    return None


def find_ollama():
    cands = [
        os.environ.get("OLLAMA_EXE"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
        shutil.which("ollama"),
    ]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


def port_open(port, host="127.0.0.1", timeout=0.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def wait_port(port, timeout=60, host="127.0.0.1", interval=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port, host):
            return True
        time.sleep(interval)
    return False


def http_json(url, timeout=4):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kgrag-service/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            import json
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def api_health(port):
    return http_json("http://127.0.0.1:%d/health/lite" % int(port), timeout=4)


def ui_alive(port):
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:%d/" % int(port),
            headers={"User-Agent": "kgrag-service/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models():
    d = http_json("http://127.0.0.1:%d/api/tags" % DEFAULT_OLLAMA_PORT, timeout=4)
    if not d:
        return None
    return [m.get("name", "") for m in (d.get("models") or [])]


def service_state(name=DEFAULT_NEO4J_SERVICE):
    rc, out = run_hidden(["sc", "query", name], timeout=15)
    if rc != 0:
        return "MISSING"
    m = re.search(r"STATE\s*:\s*\d+\s+(\w+)", out)
    return m.group(1).upper() if m else "UNKNOWN"


def start_windows_service(name=DEFAULT_NEO4J_SERVICE):
    return run_hidden(["net", "start", name], timeout=90)


def pid_alive(pid):
    if not pid:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    try:
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        kernel32.CloseHandle(h)
        return bool(ok) and code.value == STILL_ACTIVE
    except Exception:
        return False


def pid_on_port(port):
    rc, out = run_hidden(["netstat", "-ano", "-p", "tcp"], timeout=20)
    if rc != 0:
        return None
    pat = re.compile(r":%d\s+\S+\s+LISTENING\s+(\d+)\s*$" % int(port), re.M)
    m = pat.search(out)
    return int(m.group(1)) if m else None


def process_cmdline(pid):
    if not pid:
        return ""
    ps = (
        "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d')"
        ".CommandLine" % int(pid)
    )
    rc, out = run_hidden(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        timeout=20,
    )
    return out.strip() if rc == 0 else ""


def process_parent(pid):
    if not pid:
        return None
    ps = (
        "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d')"
        ".ParentProcessId" % int(pid)
    )
    rc, out = run_hidden(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        timeout=20,
    )
    if rc != 0:
        return None
    m = re.search(r"\d+", out)
    return int(m.group(0)) if m else None


def process_image(pid):
    if not pid:
        return ""
    rc, out = run_hidden(
        ["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"],
        timeout=15,
    )
    if rc != 0 or not out:
        return ""
    m = re.match(r'"([^"]+)"', out.strip())
    return m.group(1) if m else ""


def kill_tree(pid, timeout=20):
    rc, out = run_hidden(["taskkill", "/PID", str(int(pid)), "/T", "/F"], timeout=timeout)
    return rc, out


def tail_text(path, n=30, max_chars=6000):
    p = Path(path)
    if not p.exists():
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])[-max_chars:]
    except Exception as e:
        return "<read log failed: %s>" % e


def free_gb(path=None):
    try:
        target = Path(path or ROOT)
        if not target.exists():
            target = target.parent
        return shutil.disk_usage(str(target)).free / (1024 ** 3)
    except Exception:
        return None


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_log_path(log_dir, prefix):
    ensure_dir(log_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / ("%s_%s.log" % (prefix, stamp))
    prune_logs(log_dir, prefix, keep=5)
    return path


def prune_logs(log_dir, prefix, keep=5):
    try:
        files = sorted(
            Path(log_dir).glob("%s_*.log" % prefix),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[keep:]:
            old.unlink()
    except Exception:
        pass


def _newest_mtime(root, skip_dirs=("node_modules", ".git")):
    newest = 0.0
    newest_path = ""
    root = Path(root)
    if not root.exists():
        return newest, newest_path
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            p = Path(cur) / name
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > newest:
                newest, newest_path = m, str(p)
    return newest, newest_path


def webui_build_info():
    """新版前端的构建状态：dist 是否存在、源码是否比 dist 新。"""
    info = {
        "dist": str(WEBUI_DIST),
        "index": str(WEBUI_INDEX),
        "exists": WEBUI_INDEX.exists(),
        "stale": None,
        "src_newest": None,
        "dist_newest": None,
        "build_sha8": None,
    }
    if not info["exists"]:
        return info
    src_m, src_p = _newest_mtime(WEBUI_SRC)
    dist_m, dist_p = _newest_mtime(WEBUI_DIST)
    info.update({
        "src_newest": src_m,
        "src_newest_path": src_p,
        "dist_newest": dist_m,
        "dist_newest_path": dist_p,
        "stale": bool(src_m and dist_m and src_m > dist_m),
        "build_sha8": hashlib.sha256(WEBUI_INDEX.read_bytes()).hexdigest()[:8],
    })
    return info


def webui_served(port):
    """请求 API 的 /ui/，确认返回的是新版前端而不是旧 Gradio。"""
    url = "http://127.0.0.1:%d/ui/" % int(port)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kgrag-service/1.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            body = r.read().decode("utf-8", "replace")
            ok = (r.status == 200 and WEBUI_TITLE in body
                  and '<div id="app">' in body)
            return {"ok": ok, "status": r.status, "url": url,
                    "title_ok": WEBUI_TITLE in body,
                    "app_mount_ok": '<div id="app">' in body}
    except Exception as e:
        return {"ok": False, "status": None, "url": url,
                "error": "%s: %s" % (type(e).__name__, str(e)[:120])}


def find_model(models, wanted):
    """Ollama 模型名比较：m 与 m:latest 视为同一个。"""
    if not wanted:
        return False
    w = wanted if ":" in wanted else wanted + ":latest"
    for m in models or []:
        mm = m if ":" in m else m + ":latest"
        if mm == w:
            return True
    return False


def parse_bolt_uri(uri, default_port=DEFAULT_NEO4J_PORT):
    s = (uri or "").strip()
    m = re.match(r"^(?:bolt|neo4j)(?:\+s)?://([^:/]+)(?::(\d+))?", s)
    if not m:
        return "localhost", int(default_port)
    return m.group(1), int(m.group(2) or default_port)


def start_ollama_hidden(log_dir):
    """启动 ollama serve（无窗口），返回 (pid, log_path)。"""
    exe = find_ollama()
    if not exe:
        raise RuntimeError(
            "未找到 Ollama。请先运行 python scripts/setup_wizard.py --install-ollama，"
            "或从 https://ollama.com/download 安装。"
        )
    path = new_log_path(log_dir, "ollama")
    with open(path, "ab") as fh:
        p = subprocess.Popen(
            [exe, "serve"],
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
        )
    return p.pid, path
