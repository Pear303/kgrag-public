# -*- coding: utf-8 -*-
"""KG-RAG 安装/自检入口。

作用：
  1) 检查 Python / pip 依赖 / .env / Neo4j / 图谱与缓存 / Ollama / 模型 / 端口；
  2) 可安装 Python 依赖、通过 winget 安装 Ollama、拉取本地模型；
  3) 不自动安装或覆盖 Neo4j，不写图库；发现缺失只报告并给命令。

用法：
    python scripts/setup_wizard.py                 # 菜单
    python scripts/setup_wizard.py --check         # 只检查（免费）
    python scripts/setup_wizard.py --install-deps  # 安装 requirements.txt
    python scripts/setup_wizard.py --install-ollama
    python scripts/setup_wizard.py --pull-models
    python scripts/setup_wizard.py --all           # 依赖 + Ollama + 模型
    python scripts/setup_wizard.py --start         # 无窗口启动
"""
import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from service_common import (  # noqa: E402
    DEFAULT_API_PORT,
    DEFAULT_LOG_DIR,
    DEFAULT_NEO4J_SERVICE,
    DEFAULT_OLLAMA_PORT,
    DEFAULT_REQUIREMENTS,
    DEFAULT_UI_PORT,
    ROOT,
    api_health,
    ensure_dir,
    find_model,
    find_ollama,
    find_python,
    free_gb,
    get_setting,
    ollama_models,
    parse_bolt_uri,
    pid_on_port,
    port_open,
    read_env_file,
    run_hidden,
    service_state,
    start_ollama_hidden,
    ui_alive,
    wait_port,
    WEBUI_DIR,
    WEBUI_INDEX,
    webui_build_info,
    webui_served,
)

RECOMMENDED_LLM = "qwen3.5:4b"
RECOMMENDED_EMB = "qwen3-embedding:0.6b"
NEED_GB = 8.0

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class Report:
    def __init__(self):
        self.ok = []
        self.warn = []
        self.fail = []
        self.section = ""

    def add(self, state, name, detail=""):
        line = name + (("  (" + detail + ")") if detail else "")
        if state == "OK":
            self.ok.append(line)
            print("[OK]    " + line)
        elif state == "WARN":
            self.warn.append(line)
            print("[WARN]  " + line)
        else:
            self.fail.append(line)
            print("[FAIL]  " + line)

    def summary(self):
        print("-" * 66)
        print("汇总：%d OK / %d WARN / %d FAIL"
              % (len(self.ok), len(self.warn), len(self.fail)))
        return len(self.fail)


def _env_map_or_defaults():
    env_map = read_env_file()
    if not env_map:
        print("[WARN]  未找到 .env；模型名使用推荐值，Neo4j 密码必须自行配置。")
    return env_map


def _file_size(path):
    try:
        return Path(path).stat().st_size / (1024 ** 2)
    except Exception:
        return None


def check_platform(rep):
    print("=" * 66)
    print("KG-RAG 安装/运行环境检查")
    print("=" * 66)
    if sys.platform == "win32":
        rep.add("OK", "Windows 平台", "%s %s" % (platform.release(), platform.version()))
    else:
        rep.add("FAIL", "非 Windows 平台", "本项目当前面向 Windows 部署")
    if platform.machine().lower() in ("amd64", "x86_64"):
        rep.add("OK", "64 位系统")
    else:
        rep.add("WARN", "非 64 位系统", platform.machine())


def check_python(rep):
    print("-" * 66)
    print("Python 与依赖")
    py, ver = find_python()
    if py:
        rep.add("OK", "Python %s" % ver, py)
    else:
        rep.add("FAIL", "未找到 Python 3.10–3.12",
                "运行 python scripts/setup_wizard.py 或设置 KGRAG_PYTHON")
        return
    missing = []
    for mod, pkg in DEFAULT_REQUIREMENTS:
        try:
            found = importlib.util.find_spec(mod) is not None
        except Exception:
            found = False
        if not found:
            missing.append(pkg)
    if not missing:
        rep.add("OK", "Python 依赖齐全", "%d 个包" % len(DEFAULT_REQUIREMENTS))
    else:
        rep.add("FAIL", "缺少 Python 依赖", ", ".join(missing))
        print("        -> 运行: python scripts/setup_wizard.py --install-deps")

    try:
        import importlib.metadata as md
        req = ROOT / "requirements.txt"
        pins = []
        if req.exists():
            for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.split("#", 1)[0].strip()
                if "==" in line:
                    name, _, ver = line.partition("==")
                    pins.append((name.strip(), ver.strip()))
        bad = []
        for name, ver in pins:
            try:
                got = md.version(name)
            except Exception:
                continue
            if got != ver:
                bad.append("%s %s→%s" % (name, ver, got))
        if bad:
            rep.add("WARN", "依赖版本与 requirements.txt 不一致", ", ".join(bad[:6]))
        elif pins:
            rep.add("OK", "依赖版本与 requirements.txt 一致", "%d 个钉版本" % len(pins))
    except Exception:
        pass


def check_env(rep):
    print("-" * 66)
    print(".env 配置（只显示键名/值的存在性）")
    p = ROOT / ".env"
    if not p.exists():
        rep.add("FAIL", ".env 不存在", "复制 .env.example 为 .env 并填写")
        return {}
    rep.add("OK", ".env 存在", str(p))
    env = read_env_file()
    required = [
        "LLM_PROVIDER", "EMBEDDING_PROVIDER",
        "LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL", "LOCAL_EMBEDDING_MODEL",
        "NEO4J_PASSWORD", "SERVING_NEO4J_URI",
        "SERVING_MANUAL", "SERVING_EXTRACT", "SERVING_ROUND",
        "SERVING_PORT",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        rep.add("FAIL", "缺少关键配置键", ", ".join(missing))
    else:
        rep.add("OK", "关键配置键齐全", "%d 项" % len(required))

    provider = env.get("LLM_PROVIDER", "")
    if provider == "local":
        rep.add("OK", "答案模型 = local")
    else:
        rep.add("FAIL", "LLM_PROVIDER=%r，生产要求 local" % provider)

    base = env.get("LOCAL_LLM_BASE_URL", "")
    host = base.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
    if host in ("localhost", "127.0.0.1"):
        rep.add("OK", "本地模型地址指向本机", base)
    else:
        rep.add("FAIL", "本地模型地址不是本机", base)

    preset = env.get("SERVING_PRESET_META")
    if preset:
        if Path(preset).exists():
            rep.add("OK", "SERVING_PRESET_META", Path(preset).name)
        else:
            rep.add("WARN", "SERVING_PRESET_META 不存在", preset)

    manual = env.get("SERVING_MANUAL", "")
    if manual:
        mp = ROOT / "data" / "manuals" / manual
        size = _file_size(mp)
        if size is None:
            rep.add("FAIL", "手册不存在", str(mp))
        else:
            rep.add("OK", "手册", "%s (%.1f MB)" % (manual, size))
    extract = env.get("SERVING_EXTRACT", "")
    if extract:
        ep = ROOT / "reports" / "kg_visual" / extract
        size = _file_size(ep)
        if size is None:
            rep.add("FAIL", "图谱抽取件不存在", str(ep))
        else:
            rep.add("OK", "图谱抽取件", "%s (%.1f MB)" % (extract, size))
    return env


def check_data(rep, env):
    print("-" * 66)
    print("知识库缓存（微观/宏观/Vector）")
    round_name = env.get("SERVING_ROUND", "")
    cache = ROOT / ".sisyphus" / "evidence" / "cache"
    if not round_name:
        rep.add("FAIL", "未配置 SERVING_ROUND")
        return
    comm = cache / ("communities_%s.json" % round_name)
    if comm.exists():
        n = None
        try:
            n = len(json.loads(comm.read_text(encoding="utf-8")))
        except Exception:
            pass
        rep.add("OK", "社区缓存", "%s (%s 个)" % (comm.name, n if n is not None else "?"))
    else:
        rep.add("WARN", "社区缓存缺失（宏观查询不可用）", str(comm))
    chunk = cache / ("chunk_index_%s.json" % round_name)
    if chunk.exists():
        rep.add("OK", "分块索引", chunk.name)
    else:
        rep.add("WARN", "分块索引缺失（宏观证据回填可能不可用）", str(chunk))
    vec = cache / ("vector_index_%s" % round_name)
    if vec.exists():
        rep.add("OK", "Vector 索引缓存", vec.name)
    else:
        rep.add("WARN", "Vector 索引缓存缺失（首次会重建/对比变慢）", str(vec))


def check_neo4j(rep, env):
    print("-" * 66)
    print("Neo4j 与图谱")
    service = os.environ.get("NEO4J_SERVICE") or DEFAULT_NEO4J_SERVICE
    st = service_state(service)
    if st == "RUNNING":
        rep.add("OK", "服务 %s" % service, "RUNNING")
    elif st == "MISSING":
        rep.add("FAIL", "服务 %s 不存在" % service,
                "需安装 Neo4j 5.x 或改用已有实例")
    else:
        rep.add("WARN", "服务 %s 状态 %s" % (service, st), "启动器会尝试 net start")

    uri = env.get("SERVING_NEO4J_URI", "bolt://localhost:7690")
    host, port = parse_bolt_uri(uri)
    if not port_open(port):
        rep.add("FAIL", "Neo4j 端口不可达", uri)
        return
    rep.add("OK", "Neo4j 端口", uri)

    pwd = env.get("NEO4J_PASSWORD", "")
    user = env.get("NEO4J_USER", "neo4j")
    if not pwd:
        rep.add("FAIL", "NEO4J_PASSWORD 未配置")
        return
    if pwd in ("change-me", "your-neo4j-password"):
        rep.add("WARN", "NEO4J_PASSWORD 仍是模板占位值", "请改成实际密码")
    try:
        from neo4j import GraphDatabase
    except Exception:
        rep.add("WARN", "未安装 neo4j 驱动，跳过图谱计数")
        return
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    try:
        drv = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=6)
        drv.verify_connectivity()
        with drv.session() as s:
            total = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            manual = env.get("SERVING_MANUAL", "")
            own = None
            if manual:
                own = s.run(
                    "MATCH (n:Entity {source_manual: $m}) RETURN count(n) AS c",
                    m=manual,
                ).single()["c"]
        drv.close()
        if total > 0:
            rep.add("OK", "图谱实体", "全库 %d / 本手册 %s" % (total, own if own is not None else "?"))
        else:
            rep.add("FAIL", "图谱为空", "需要先建库（scripts/build_kb.py）")
        if own == 0:
            rep.add("WARN", "当前手册在图中没有实体", "检查 SERVING_MANUAL 是否与建库时一致")
    except Exception as e:
        rep.add("FAIL", "Neo4j 连接/查询失败", str(e)[:160])


def check_ollama(rep, env):
    print("-" * 66)
    print("Ollama 与本地模型")
    models = ollama_models()
    if models is None:
        rep.add("FAIL", "Ollama 未运行", "http://127.0.0.1:%d" % DEFAULT_OLLAMA_PORT)
        return
    rep.add("OK", "Ollama 在线", "%d 个模型" % len(models))
    for key, fallback in (("LOCAL_LLM_MODEL", RECOMMENDED_LLM),
                          ("LOCAL_EMBEDDING_MODEL", RECOMMENDED_EMB)):
        want = env.get(key) or fallback
        if find_model(models, want):
            rep.add("OK", key, want)
        else:
            rep.add("FAIL", "缺少模型 %s" % want, "python scripts/setup_wizard.py --pull-models")
    if (env.get("LOCAL_LLM_MODEL") or "").startswith("qwen3.5"):
        if env.get("LOCAL_THINKING_OFF") != "1":
            rep.add("WARN", "LOCAL_THINKING_OFF 未设为 1",
                    "qwen3.5 思考段会吃生成预算")
        if not env.get("LOCAL_NUM_CTX"):
            rep.add("WARN", "LOCAL_NUM_CTX 未设置",
                    "qwen3.5:4b 原生上下文很大，需显式限制")
    free = free_gb(Path(os.environ.get("USERPROFILE", str(ROOT))) / ".ollama")
    if free is not None:
        if free >= NEED_GB:
            rep.add("OK", "模型盘剩余空间", "%.1f GB" % free)
        else:
            rep.add("WARN", "模型盘剩余空间不足", "%.1f GB（建议 >= %.0f GB）" % (free, NEED_GB))


def check_webui(rep, env):
    print("-" * 66)
    print("新版前端（webui/dist，由 API 的 /ui/ 提供）")
    bi = webui_build_info()
    if not bi.get("exists"):
        rep.add("FAIL", "webui/dist 未构建", "运行 python scripts/setup_wizard.py --build-webui")
        return
    rep.add("OK", "构建产物存在", "build=%s" % bi.get("build_sha8"))
    if bi.get("stale"):
        rep.add("FAIL", "webui/src 比 dist 新", "源码改了没构建；运行 python scripts/setup_wizard.py --build-webui")
    else:
        rep.add("OK", "构建产物不旧于源码")
    if not (WEBUI_DIR / "node_modules").exists():
        rep.add("WARN", "webui/node_modules 不存在", "重新构建需要先 npm install")
    try:
        api_port = int(os.environ.get("SERVING_PORT") or env.get("SERVING_PORT") or DEFAULT_API_PORT)
    except Exception:
        api_port = DEFAULT_API_PORT
    if port_open(api_port):
        s = webui_served(api_port)
        if s.get("ok"):
            rep.add("OK", "/ui/ 返回新版前端", "title + #app 校验通过")
        else:
            rep.add("FAIL", "/ui/ 不是新版前端", str(s)[:160])
    else:
        rep.add("WARN", "API 未运行，暂时无法核验 /ui/", "启动后再跑一次 --check")


def check_ports(rep, env):
    print("-" * 66)
    print("服务端口")
    try:
        api_port = int(os.environ.get("SERVING_PORT") or env.get("SERVING_PORT") or DEFAULT_API_PORT)
        ui_port = int(env.get("UI_PORT") or DEFAULT_UI_PORT)
    except Exception:
        api_port, ui_port = DEFAULT_API_PORT, DEFAULT_UI_PORT
    if port_open(api_port):
        h = api_health(api_port)
        if h and h.get("ready") is True:
            rep.add("OK", "API :%d 已运行" % api_port,
                    "pid=%s round=%s" % (pid_on_port(api_port), h.get("round")))
        else:
            rep.add("FAIL", "API :%d 被占用但不是 KG-RAG" % api_port)
    else:
        rep.add("OK", "API :%d 空闲" % api_port)
    if port_open(ui_port):
        rep.add("WARN", "旧 Gradio :%d 在跑" % ui_port,
                "新版请用 /ui/；可运行 python scripts/service_control.py stop 清理")
    else:
        rep.add("OK", "旧 Gradio :%d 未启动" % ui_port, "预期")


def check_tools(rep):
    print("-" * 66)
    print("安装/构建工具")
    if shutil.which("winget"):
        rep.add("OK", "winget 可用")
    else:
        rep.add("WARN", "winget 不可用", "Ollama 需手动安装")
    for name in ("node", "npm"):
        exe = shutil.which(name)
        if not exe:
            rep.add("WARN", "%s 不可用" % name, "构建新版前端需要 Node/npm")
            continue
        rc, out = run_hidden([exe, "-v"], timeout=15)
        rep.add("OK", name, out.strip()[:40] if rc == 0 else "version failed")
    rc, out = run_hidden(["java", "-version"], timeout=15)
    if rc == 0 and "version" in out.lower():
        first = out.strip().splitlines()[0] if out.strip() else "java"
        rep.add("OK", "Java", first[:90])
    else:
        rep.add("WARN", "未检测到 Java", "Neo4j 5.x 需要 Java 17/21")


def check_all(rep=None):
    rep = rep or Report()
    rep.section = "运行环境"
    check_platform(rep)
    rep.section = "Python 与依赖"
    check_python(rep)
    rep.section = "配置（.env）"
    env = check_env(rep)
    if env:
        rep.section = "知识库缓存"
        check_data(rep, env)
        rep.section = "Neo4j 与图谱"
        check_neo4j(rep, env)
        rep.section = "Ollama 与本地模型"
        check_ollama(rep, env)
        rep.section = "新版前端"
        check_webui(rep, env)
        rep.section = "服务端口"
        check_ports(rep, env)
    rep.section = "安装/构建工具"
    check_tools(rep)
    n_fail = rep.summary()
    if n_fail:
        print("有 FAIL 项；先修复，或至少确认它们不影响你当前要用的功能。")
    else:
        print("环境就绪。")
    return n_fail


def init_env(args=None):
    print("=" * 66)
    print("生成 .env 模板")
    dst = ROOT / ".env"
    src = ROOT / ".env.example"
    if dst.exists():
        print("[OK]   .env 已存在，不覆盖: %s" % dst)
        return 0
    if not src.exists():
        print("[FAIL] .env.example 不存在")
        return 1
    if getattr(args, "dry_run", False):
        print("[dry-run] copy %s -> %s" % (src, dst))
        return 0
    shutil.copyfile(src, dst)
    print("[OK]   已生成 %s" % dst)
    print("       请修改 NEO4J_PASSWORD，并按实际手册修改 SERVING_*。")
    return 0


def install_deps(args):
    print("=" * 66)
    print("安装 Python 依赖")
    py, ver = find_python()
    if not py:
        print("[FAIL] 未找到 Python 3.10–3.12；先安装 Python 3.10。")
        return 1
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("[FAIL] requirements.txt 不存在")
        return 1
    cmd = [py, "-m", "pip", "install", "-r", str(req),
           "--disable-pip-version-check", "--no-input"]
    if args.index_url:
        cmd += ["--index-url", args.index_url]
    print("[..]   %s (%s)" % (py, ver))
    print("[..]   " + " ".join(cmd))
    if args.dry_run:
        print("[dry-run] 未执行 pip")
        return 0
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc == 0:
        print("[OK]   依赖安装完成")
    else:
        print("[FAIL] pip 返回 %d；检查网络或加 --index-url <镜像>" % rc)
    return rc


def install_ollama(args):
    print("=" * 66)
    print("安装 Ollama")
    exe = find_ollama()
    if exe:
        print("[OK]   已安装: %s" % exe)
        return 0
    winget = shutil.which("winget")
    if not winget:
        print("[FAIL] 未找到 winget；请从 https://ollama.com/download/windows 手动安装。")
        return 1
    cmd = [winget, "install", "--id", "Ollama.Ollama", "-e", "--silent",
           "--accept-package-agreements", "--accept-source-agreements"]
    print("[..]   " + " ".join(cmd))
    if args.dry_run:
        print("[dry-run] 未执行 winget")
        return 0
    rc = subprocess.call(cmd)
    exe = find_ollama()
    if rc == 0 and exe:
        print("[OK]   Ollama 已安装: %s" % exe)
        return 0
    print("[WARN] winget 返回 %d；未在常见路径找到 ollama.exe。" % rc)
    print("       请从 https://ollama.com/download/windows 手动安装。")
    return 0 if exe else 1


def _ensure_ollama_running():
    models = ollama_models()
    if models is not None:
        return True
    if port_open(DEFAULT_OLLAMA_PORT):
        deadline = time.time() + 30
        while time.time() < deadline:
            models = ollama_models()
            if models is not None:
                return True
            time.sleep(1)
        return False
    exe = find_ollama()
    if not exe:
        return False
    pid, path = start_ollama_hidden(DEFAULT_LOG_DIR)
    print("[..]   Ollama 已无窗口启动 pid=%s log=%s" % (pid, path))
    return wait_port(DEFAULT_OLLAMA_PORT, timeout=60)


def pull_models(args):
    print("=" * 66)
    print("下载本地模型")
    env = read_env_file()
    wanted = [
        env.get("LOCAL_LLM_MODEL") or RECOMMENDED_LLM,
        env.get("LOCAL_EMBEDDING_MODEL") or RECOMMENDED_EMB,
    ]
    if not _ensure_ollama_running():
        print("[FAIL] Ollama 未运行且无法启动；先安装/启动 Ollama。")
        return 1
    exe = find_ollama()
    if not exe:
        print("[FAIL] 找到 Ollama 服务但找不到 ollama.exe 命令行。")
        return 1
    models = ollama_models() or []
    missing = [m for m in wanted if not find_model(models, m)]
    if not missing:
        print("[OK]   模型已齐: %s" % ", ".join(wanted))
        return 0
    free = free_gb(Path(os.environ.get("USERPROFILE", str(ROOT))) / ".ollama")
    print("[..]   待下载: %s" % ", ".join(missing))
    if free is not None:
        print("[..]   模型盘剩余 %.1f GB（本次预计需数 GB）" % free)
        if free < NEED_GB:
            print("[WARN] 剩余空间偏小，可能失败。")
    if args.dry_run:
        for m in missing:
            print("[dry-run] %s pull %s" % (exe, m))
        return 0
    for m in missing:
        print("[..]   ollama pull %s" % m, flush=True)
        rc = subprocess.call([exe, "pull", m])
        if rc != 0:
            print("[FAIL] 拉取 %s 失败（rc=%d）" % (m, rc))
            return rc
    models = ollama_models() or []
    left = [m for m in wanted if not find_model(models, m)]
    if left:
        print("[FAIL] 拉取后仍缺: %s" % ", ".join(left))
        return 1
    print("[OK]   模型已齐: %s" % ", ".join(wanted))
    return 0


def build_webui(args):
    print("=" * 66)
    print("构建新版前端（webui/dist）")
    pkg = WEBUI_DIR / "package.json"
    if not pkg.exists():
        print("[FAIL] webui/package.json 不存在")
        return 1
    npm = shutil.which("npm")
    if not npm:
        print("[FAIL] 未找到 npm；请安装 Node.js 18+。")
        return 1
    lock = WEBUI_DIR / "package-lock.json"
    install_cmd = [npm, "ci" if lock.exists() else "install"]
    build_cmd = [npm, "run", "build"]
    print("[..]   cwd=%s" % WEBUI_DIR)
    print("[..]   " + " ".join(install_cmd))
    print("[..]   " + " ".join(build_cmd))
    if args.dry_run:
        print("[dry-run] 未执行 npm")
        return 0
    if not (WEBUI_DIR / "node_modules").exists():
        rc = subprocess.call(install_cmd, cwd=str(WEBUI_DIR))
        if rc != 0:
            print("[FAIL] npm install 返回 %d" % rc)
            return rc
    rc = subprocess.call(build_cmd, cwd=str(WEBUI_DIR))
    if rc != 0:
        print("[FAIL] npm run build 返回 %d" % rc)
        return rc
    bi = webui_build_info()
    if not bi.get("exists"):
        print("[FAIL] 构建后仍找不到 webui/dist/index.html")
        return 1
    if bi.get("stale"):
        print("[WARN] 构建后源码仍比 dist 新，请检查系统时间/输出目录")
        return 1
    print("[OK]   新版前端已构建: build=%s" % bi.get("build_sha8"))
    return 0


def _run_control(action, args):
    py, _ = find_python()
    if not py:
        print("[FAIL] 未找到 Python 3.10–3.12")
        return 1
    cmd = [py, "-X", "utf8", str(ROOT / "scripts" / "service_control.py"), action]
    if getattr(args, "no_browser", False):
        cmd.append("--no-browser")
    return subprocess.call(cmd, cwd=str(ROOT))


def install_all(args):
    if not (ROOT / ".env").exists():
        init_env(args)
    n_fail = check_all()
    if n_fail and not args.yes:
        ans = input("\n检查有 FAIL；仍继续安装 Python 依赖/Ollama/模型？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 1
    rc = install_deps(args)
    if rc != 0:
        return rc
    rc = install_ollama(args)
    if rc != 0:
        return rc
    rc = pull_models(args)
    if rc != 0:
        return rc
    print("\n" + "=" * 66)
    check_all()
    if getattr(args, "start", False):
        print("\n启动系统（无窗口）...")
        return _run_control("start", args)
    print("\n依赖/模型处理完成。运行 python scripts/service_control.py start 启动。")
    return 0


def menu(args):
    while True:
        print("")
        print("=" * 66)
        print(" KG-RAG 安装 / 自检入口")
        print("=" * 66)
        print(" 1) 检查环境（免费）")
        print(" 2) 安装/修复 Python 依赖")
        print(" 3) 安装 Ollama（winget）")
        print(" 4) 下载本地模型")
        print(" 5) 全部执行（依赖 + Ollama + 模型）")
        print(" 6) 无窗口启动系统")
        print(" 7) 停止 API/WebUI")
        print(" 8) 生成 .env 模板（不覆盖）")
        print(" 9) 构建/更新新版前端（webui/dist）")
        print(" 0) 退出")
        try:
            choice = input("\n选择 [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if choice == "1":
            check_all()
        elif choice == "2":
            install_deps(args)
        elif choice == "3":
            install_ollama(args)
        elif choice == "4":
            pull_models(args)
        elif choice == "5":
            install_all(args)
        elif choice == "6":
            _run_control("start", args)
        elif choice == "7":
            _run_control("stop", args)
        elif choice == "8":
            init_env(args)
        elif choice == "9":
            build_webui(args)
        elif choice == "0":
            return 0
        else:
            print("未知选项")
        try:
            input("\n按回车返回菜单...")
        except (EOFError, KeyboardInterrupt):
            return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="setup_wizard.py",
                                description="KG-RAG 安装/自检入口")
    p.add_argument("--check", action="store_true", help="只检查环境")
    p.add_argument("--init-env", action="store_true", help="从 .env.example 生成 .env（不覆盖）")
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--install-ollama", action="store_true")
    p.add_argument("--pull-models", action="store_true")
    p.add_argument("--build-webui", action="store_true", help="构建 webui/dist 新版前端")
    p.add_argument("--all", action="store_true")
    p.add_argument("--start", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--yes", action="store_true", help="跳过确认")
    p.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    p.add_argument("--index-url", default=None, help="pip 镜像地址")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    if args.check:
        return 1 if check_all() else 0
    if args.init_env:
        return init_env(args)
    if args.install_deps:
        return install_deps(args)
    if args.install_ollama:
        return install_ollama(args)
    if args.pull_models:
        return pull_models(args)
    if args.build_webui:
        return build_webui(args)
    if args.all:
        return install_all(args)
    if args.start:
        return _run_control("start", args)
    if args.stop:
        return _run_control("stop", args)
    if sys.stdin is not None and sys.stdin.isatty():
        return menu(args)
    return 1 if check_all() else 0


if __name__ == "__main__":
    sys.exit(main())
