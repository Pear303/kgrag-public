# -*- coding: utf-8 -*-
"""KG-RAG 服务控制：无窗口启动 / 停止 / 状态。

用法：
    python scripts/service_control.py start
    python scripts/service_control.py stop
    python scripts/service_control.py status
    python scripts/service_control.py start --dry-run
    python scripts/service_control.py restart-ollama --yes

启动时只拉起本项目的 API/WebUI（CREATE_NO_WINDOW），Ollama 若未运行也以无窗口
方式拉起；Neo4j 只检查/尝试 `net start neo4j-pump`，不安装、不写库。
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from service_common import (  # noqa: E402
    CREATE_NO_WINDOW,
    DEFAULT_API_PORT,
    DEFAULT_LOG_DIR,
    DEFAULT_NEO4J_SERVICE,
    DEFAULT_OLLAMA_PORT,
    DEFAULT_UI_PORT,
    ROOT,
    api_health,
    ensure_dir,
    find_ollama,
    find_python,
    get_setting,
    kill_tree,
    new_log_path,
    ollama_models,
    parse_bolt_uri,
    pid_alive,
    pid_on_port,
    port_open,
    process_cmdline,
    process_image,
    process_parent,
    read_env_file,
    run_hidden,
    service_state,
    start_ollama_hidden,
    start_windows_service,
    tail_text,
    ui_alive,
    wait_port,
    find_model,
    WEBUI_INDEX,
    WEBUI_TITLE,
    webui_build_info,
    webui_served,
)

_STATE = {"log_path": None}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(msg)
    path = _STATE.get("log_path")
    if path:
        try:
            ensure_dir(Path(path).parent)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
    if sys.stdout is not None:
        try:
            print(line, flush=True)
        except Exception:
            pass


def is_gui():
    return Path(sys.executable).name.lower() == "pythonw.exe"


def notify_error(title, message, force=False):
    if not force and not is_gui():
        return
    text = str(message)[:1800]
    try:
        ctypes.windll.user32.MessageBoxW(0, text, str(title)[:80], 0x10)
    except Exception:
        pass


def state_path(args):
    return Path(args.state_dir) / "state.json"


def load_state(args):
    import json
    p = state_path(args)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(args, state):
    import json
    p = state_path(args)
    ensure_dir(p.parent)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _port(args_value, key, default):
    if args_value is not None:
        return int(args_value)
    try:
        return int(get_setting(key, str(default)))
    except Exception:
        return int(default)


def _health_ready(h):
    return bool(h) and h.get("ready") is True


def _wait_api(port, timeout, proc=None):
    deadline = time.time() + int(timeout)
    last = None
    while time.time() < deadline:
        h = api_health(port)
        if _health_ready(h):
            return h
        last = h
        if proc is not None and proc.poll() is not None:
            return None
        time.sleep(2)
    return last if _health_ready(last) else None


def _wait_ui(port, timeout, proc=None):
    deadline = time.time() + int(timeout)
    while time.time() < deadline:
        if ui_alive(port):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(1)
    return False


def _start_child(name, cmd, log_dir, env):
    path = new_log_path(log_dir, name)
    with open(path, "ab") as fh:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
        )
    return p, path


def check_webui_build():
    info = webui_build_info()
    if not info["exists"]:
        raise RuntimeError(
            "新版前端未构建：%s 不存在。\n"
            "请运行 python scripts/setup_wizard.py --build-webui 或 (cd webui && npm install && npm run build)。"
            % WEBUI_INDEX
        )
    if info["stale"]:
        raise RuntimeError(
            "webui/src 比 webui/dist 新：源码改了但没重新构建，启动会继续用旧页面。\n"
            "请运行 python scripts/setup_wizard.py --build-webui 或 (cd webui && npm run build)。"
        )
    log("[OK]   新版前端构建: build=%s dist=%s"
        % (info.get("build_sha8"), info.get("dist_newest_path")))
    return info


def check_webui_served(api_port):
    s = webui_served(api_port)
    if not s.get("ok"):
        raise RuntimeError(
            "API 已启动，但 /ui/ 返回的不是新版前端（title/mount 校验失败）。\n"
            "请检查 webui/dist 与 api_server.py 的 /ui 挂载。详情: %s"
            % json.dumps(s, ensure_ascii=False)
        )
    log("[OK]   新版前端 /ui/ 已就绪（title + #app 校验通过）")
    return s


def ensure_neo4j(args, env_map):
    uri = get_setting("SERVING_NEO4J_URI", "bolt://localhost:7690", env_map)
    _host, port = parse_bolt_uri(uri)
    if port_open(port):
        log("[OK]   Neo4j 已连通 (%s)" % uri)
        return
    service = get_setting("NEO4J_SERVICE", DEFAULT_NEO4J_SERVICE, env_map)
    st = service_state(service)
    log("[..]   Neo4j 端口 %d 未开，服务 %s 状态=%s" % (port, service, st))
    if st == "MISSING":
        raise RuntimeError(
            "Neo4j 服务 %s 不存在，且 %s 不可达。\n"
            "请先安装 Neo4j 5.x 并启动服务（或把 SERVING_NEO4J_URI 指到已有实例）。\n"
            "本启动器不会自动安装/覆盖数据库。" % (service, uri)
        )
    if st != "RUNNING":
        rc, out = start_windows_service(service)
        log("[..]   net start %s rc=%s" % (service, rc))
        if rc != 0:
            raise RuntimeError(
                "Neo4j 服务启动失败（可能需要管理员权限）。\n%s\n"
                "请手动以管理员运行: net start %s" % (out.strip()[:500], service)
            )
    if not wait_port(port, timeout=120):
        raise RuntimeError("Neo4j 服务已启动但 %s 在 120 秒内仍不可达。" % uri)
    log("[OK]   Neo4j 已连通 (%s)" % uri)


def ensure_ollama(args, log_dir):
    env_map = read_env_file()
    models = ollama_models()
    started_by_us = False
    start_log = None
    if models is None and port_open(DEFAULT_OLLAMA_PORT):
        deadline = time.time() + 30
        while time.time() < deadline and models is None:
            time.sleep(2)
            models = ollama_models()
        if models is None:
            log("[WARN] Ollama 端口已开但 /api/tags 暂不可读，按已运行处理（不重复启动）")
            return {"started_by_us": False, "pid": pid_on_port(DEFAULT_OLLAMA_PORT)}
    if models is None:
        pid, path = start_ollama_hidden(log_dir)
        started_by_us = True
        start_log = str(path)
        log("[..]   Ollama 未运行，已无窗口启动 pid=%s log=%s" % (pid, path))
        deadline = time.time() + 90
        while time.time() < deadline:
            models = ollama_models()
            if models is not None:
                break
            time.sleep(2)
        if models is None:
            raise RuntimeError(
                "Ollama 启动后 90 秒内 /api/tags 仍不可用。\n日志: %s\n%s"
                % (path, tail_text(path, 25))
            )
    log("[OK]   Ollama 已在线 (%d 个模型)" % len(models))
    missing = []
    for key in ("LOCAL_LLM_MODEL", "LOCAL_EMBEDDING_MODEL"):
        want = get_setting(key, "", env_map)
        if want and not find_model(models, want):
            missing.append(want)
    if missing:
        raise RuntimeError(
            "Ollama 缺少本地模型: %s\n请先运行 python scripts/setup_wizard.py --pull-models。"
            % ", ".join(missing)
        )
    entry = {"started_by_us": started_by_us,
             "pid": pid_on_port(DEFAULT_OLLAMA_PORT)}
    if start_log:
        entry["log"] = start_log
    return entry


def start_api(args, env_map, log_dir, api_port):
    if port_open(api_port):
        h = api_health(api_port)
        if _health_ready(h):
            log("[OK]   API 已在运行 (: %d)，保持不动" % api_port)
            return {"started_by_us": False, "pid": pid_on_port(api_port),
                    "port": api_port, "health": h}
        raise RuntimeError(
            "端口 %d 已被占用，但不是可用的 KG-RAG API；为避免误杀，未做任何处理。"
            % api_port
        )
    python, ver = find_python()
    if not python:
        raise RuntimeError(
            "未找到 Python 3.10–3.12。请先运行 python scripts/setup_wizard.py --install-deps，"
            "或设置环境变量 KGRAG_PYTHON 指向 python.exe。"
        )
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "KGRAG_PYTHON": python,
        "SERVING_PORT": str(api_port),
    })
    proc, path = _start_child("api", [python, "-u", "api_server.py"], log_dir, env)
    log("[..]   API 已无窗口启动 pid=%d python=%s (%s) log=%s"
        % (proc.pid, python, ver, path))
    h = _wait_api(api_port, args.wait, proc)
    if not h:
        raise RuntimeError(
            "API 在 %s 秒内未就绪。\n日志: %s\n%s"
            % (args.wait, path, tail_text(path, 35))
        )
    log("[OK]   API 就绪 :%d" % api_port)
    return {"started_by_us": True, "pid": proc.pid, "port": api_port,
            "log": str(path), "python": python, "health": h}


def start_legacy_gradio(args, log_dir, api_port, ui_port):
    if port_open(ui_port):
        if ui_alive(ui_port):
            log("[OK]   旧 Gradio 已在运行 (: %d)，保持不动" % ui_port)
            return {"mode": "legacy_gradio", "started_by_us": False,
                    "pid": pid_on_port(ui_port), "port": ui_port}
        raise RuntimeError(
            "端口 %d 已被占用，但不是可用的旧 Gradio；为避免误杀，未做任何处理。"
            % ui_port
        )
    python, ver = find_python()
    if not python:
        raise RuntimeError("未找到 Python 3.10–3.12，无法启动 WebUI。")
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "KGRAG_PYTHON": python,
        "UI_PORT": str(ui_port),
        "KGRAG_API_BASE": "http://127.0.0.1:%d" % api_port,
    })
    proc, path = _start_child("gradio_legacy", [python, "-u", "frontend/app.py"], log_dir, env)
    log("[..]   旧 Gradio 已无窗口启动 pid=%d log=%s（仅调试用）" % (proc.pid, path))
    if not _wait_ui(ui_port, 120, proc):
        raise RuntimeError(
            "旧 Gradio 在 120 秒内未就绪。\n日志: %s\n%s"
            % (path, tail_text(path, 35))
        )
    log("[OK]   旧 Gradio 就绪 :%d" % ui_port)
    return {"mode": "legacy_gradio", "started_by_us": True, "pid": proc.pid,
            "port": ui_port, "log": str(path), "python": python}


def cmd_start(args):
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    _STATE["log_path"] = str(log_dir / "launcher.log")
    env_map = read_env_file()
    api_port = _port(args.api_port, "SERVING_PORT", DEFAULT_API_PORT)
    ui_port = _port(args.ui_port, "UI_PORT", DEFAULT_UI_PORT)
    log("=" * 60)
    log("KG-RAG 无窗口启动  API=:%d  新版前端=/ui/" % api_port)
    log("项目: %s" % ROOT)
    if args.legacy_gradio:
        log("[WARN] --legacy-gradio：本次会启动旧 Gradio :%d（仅调试用）" % ui_port)
    if args.dry_run:
        python, ver = find_python()
        log("[plan] Python: %s (%s)" % (python, ver))
        log("[plan] Neo4j : %s -> %s"
            % (get_setting("SERVING_NEO4J_URI", "bolt://localhost:7690", env_map),
               "open" if port_open(parse_bolt_uri(
                   get_setting("SERVING_NEO4J_URI", "bolt://localhost:7690", env_map))[1])
               else "closed"))
        log("[plan] Ollama: %s" % ("open" if ollama_models() is not None else "closed"))
        log("[plan] API   : %s" % ("open" if port_open(api_port) else "closed"))
        if args.legacy_gradio:
            log("[plan] 前端  : 旧 Gradio :%d（显式 legacy）" % ui_port)
        else:
            bi = webui_build_info()
            log("[plan] 前端  : webui/dist build=%s exists=%s stale=%s"
                % (bi.get("build_sha8"), bi.get("exists"), bi.get("stale")))
            if port_open(api_port):
                log("[plan] /ui/  : %s" % json.dumps(webui_served(api_port), ensure_ascii=False))
        log("[plan] dry-run：未启动任何进程")
        return 0

    state = load_state(args)
    state.update({"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "api_port": api_port})
    try:
        if not (ROOT / ".env").exists():
            raise RuntimeError("缺少 .env；请先复制 .env.example 为 .env 并填写配置。")
        if get_setting("LLM_PROVIDER", "", env_map) != "local":
            raise RuntimeError(
                "生产答案必须走本地模型，但 LLM_PROVIDER 不是 local；"
                "请检查 .env（这是作者定死的约束）。"
            )
        build_info = None
        if not args.no_ui and not args.legacy_gradio:
            build_info = check_webui_build()
        if not args.no_neo4j:
            ensure_neo4j(args, env_map)
        if not args.no_ollama:
            state["ollama"] = ensure_ollama(args, log_dir)
            save_state(args, state)
        state["api"] = start_api(args, env_map, log_dir, api_port)
        save_state(args, state)

        if args.legacy_gradio:
            state["ui"] = start_legacy_gradio(args, log_dir, api_port, ui_port)
            state["ui_port"] = ui_port
            ui_url = "http://127.0.0.1:%d/" % ui_port
        else:
            if not args.no_ui:
                check_webui_served(api_port)
                info = build_info or webui_build_info()
                state["ui"] = {
                    "mode": "webui",
                    "served_by": "api",
                    "port": api_port,
                    "url": "http://127.0.0.1:%d/ui/" % api_port,
                    "build_sha8": info.get("build_sha8"),
                    "dist_newest": info.get("dist_newest"),
                    "started_by_us": False,
                }
                state["ui_port"] = api_port
            ui_url = "http://127.0.0.1:%d/ui/" % api_port
        save_state(args, state)
        if not args.no_browser and not args.no_ui:
            webbrowser.open(ui_url)
        log("[OK]   启动完成：API http://127.0.0.1:%d/docs  前端 %s"
            % (api_port, ui_url))
        log("日志目录: %s" % log_dir)
        return 0
    except Exception as e:
        log("[FAIL] %s" % e)
        save_state(args, state)
        notify_error("KG-RAG 启动失败", "%s\n\n日志: %s" % (e, log_dir))
        return 1


def _verified_owner(pid, port, script_name):
    if not pid_alive(pid):
        return False
    if pid_on_port(port) == int(pid):
        return True
    cmd = process_cmdline(pid).lower()
    if script_name.lower() in cmd and str(ROOT).lower() in cmd.replace("/", "\\"):
        return True
    return False


def _kill_verified(pid, port, script_name, label):
    if not pid:
        return False
    if not _verified_owner(pid, port, script_name):
        log("[WARN] %s pid=%s 未通过归属核验，未杀" % (label, pid))
        return False
    rc, out = kill_tree(pid)
    log("[OK]   已停止 %s pid=%s rc=%s" % (label, pid, rc))
    parent = process_parent(pid)
    if parent and parent != pid:
        image = process_image(parent).lower()
        pcmd = process_cmdline(parent).lower()
        if image == "cmd.exe" and script_name.lower() in pcmd:
            kill_tree(parent)
            log("[OK]   已关闭 %s 的父命令行 pid=%s" % (label, parent))
    return rc == 0


def cmd_stop(args):
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    _STATE["log_path"] = str(log_dir / "launcher.log")
    env_map = read_env_file()
    api_port = _port(args.api_port, "SERVING_PORT", DEFAULT_API_PORT)
    ui_port = _port(args.ui_port, "UI_PORT", DEFAULT_UI_PORT)
    log("=" * 60)
    log("KG-RAG 停止  API=:%d（新版前端随 API；旧 Gradio :%d 若在跑会一并停）"
        % (api_port, ui_port))
    state = load_state(args)
    killed = False
    for key, port, script, label in (
        ("api", api_port, "api_server.py", "API"),
        ("ui", ui_port, "frontend/app.py", "旧 Gradio"),
    ):
        entry = state.get(key) or {}
        pid = entry.get("pid")
        if pid and _kill_verified(pid, port, script, label):
            killed = True
    legacy_ok = (getattr(args, "legacy", True)
                 and api_port == DEFAULT_API_PORT and ui_port == DEFAULT_UI_PORT)
    if legacy_ok:
        for title in ("KG-RAG API*", "KG-RAG WebUI*"):
            rc, out = run_hidden(
                ["taskkill", "/FI", "WINDOWTITLE eq %s" % title, "/T", "/F"],
                timeout=20,
            )
            if rc == 0:
                killed = True
                log("[OK]   已按窗口标题停止 %s" % title)
    for port, script, label in (
        (api_port, "api_server.py", "API"),
        (ui_port, "frontend/app.py", "旧 Gradio"),
    ):
        owner = pid_on_port(port)
        if owner and _kill_verified(owner, port, script, label):
            killed = True
    if getattr(args, "ollama", False):
        owner = pid_on_port(DEFAULT_OLLAMA_PORT)
        image = process_image(owner).lower() if owner else ""
        if owner and "ollama" in image:
            kill_tree(owner)
            wait_port(DEFAULT_OLLAMA_PORT, timeout=15)
            log("[OK]   已停止 Ollama pid=%s" % owner)
            killed = True
        elif owner:
            log("[WARN] 11434 的进程 %s 不是 ollama.exe，未杀" % owner)
    p = state_path(args)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    if not killed:
        log("[OK]   没有发现本项目的运行中实例")
    log("[OK]   停止完成（Neo4j 保持运行；Ollama 需加 --ollama 才停）")
    return 0


def cmd_status(args):
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    _STATE["log_path"] = str(log_dir / "launcher.log")
    env_map = read_env_file()
    api_port = _port(args.api_port, "SERVING_PORT", DEFAULT_API_PORT)
    ui_port = _port(args.ui_port, "UI_PORT", DEFAULT_UI_PORT)
    uri = get_setting("SERVING_NEO4J_URI", "bolt://localhost:7690", env_map)
    host, neo_port = parse_bolt_uri(uri)
    log("=" * 60)
    log("KG-RAG 状态")
    log("项目      : %s" % ROOT)
    python, pyver = find_python()
    log("Python    : %s (%s)" % (python, pyver))
    log("Neo4j     : %s -> %s" % (uri, "open" if port_open(neo_port) else "closed"))
    log("Ollama    : %s" % ("open" if ollama_models() is not None else "closed"))
    log("API :%d   : %s" % (api_port, "open" if port_open(api_port) else "closed"))
    if port_open(api_port):
        h = api_health(api_port) or {}
        log("  health  : ready=%s manual=%s round=%s entities=%s communities=%s vector=%s"
            % (h.get("ready"), h.get("manual"), h.get("round"),
               h.get("graph_entities"), h.get("communities"), h.get("vector_ready")))
    bi = webui_build_info()
    log("新版前端  : %s | build=%s exists=%s stale=%s"
        % ("http://127.0.0.1:%d/ui/" % api_port, bi.get("build_sha8"),
           bi.get("exists"), bi.get("stale")))
    if port_open(api_port):
        log("  /ui/    : %s" % json.dumps(webui_served(api_port), ensure_ascii=False))
    if port_open(ui_port):
        log("[WARN] 旧 Gradio :%d 在跑（新版请用 /ui/；需要清理可运行 python scripts/service_control.py stop）" % ui_port)
    else:
        log("旧 Gradio :%d closed（正常，默认不再启动）" % ui_port)
    models = ollama_models()
    if models is not None:
        for key in ("LOCAL_LLM_MODEL", "LOCAL_EMBEDDING_MODEL"):
            want = get_setting(key, "", env_map)
            log("  %-18s %s (%s)" % (key + ":", want,
                                    "present" if find_model(models, want) else "MISSING"))
    state = load_state(args)
    if state:
        log("state     : %s" % state_path(args))
        for key in ("api", "ui", "ollama"):
            e = state.get(key) or {}
            if e:
                log("  %-7s pid=%s port=%s mode=%s build=%s started_by_us=%s"
                    % (key, e.get("pid"), e.get("port", ""), e.get("mode", ""),
                       e.get("build_sha8", ""), e.get("started_by_us")))
    log("日志目录  : %s" % log_dir)
    return 0


def cmd_restart_ollama(args):
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)
    _STATE["log_path"] = str(log_dir / "launcher.log")
    owner = pid_on_port(DEFAULT_OLLAMA_PORT)
    image = process_image(owner).lower() if owner else ""
    if owner and "ollama" not in image:
        log("[FAIL] 11434 的进程 %s 不是 ollama.exe，拒绝重启" % owner)
        return 1
    if owner:
        if not args.yes:
            log("当前 Ollama pid=%s。重启会短暂中断正在进行的生成；确认请加 --yes。" % owner)
            return 2
        kill_tree(owner)
        wait_port(DEFAULT_OLLAMA_PORT, timeout=20)
        log("[OK]   已停止 Ollama pid=%s" % owner)
    try:
        entry = ensure_ollama(args, log_dir)
        log("[OK]   Ollama 已无窗口启动 pid=%s" % entry.get("pid"))
        return 0
    except Exception as e:
        log("[FAIL] %s" % e)
        return 1


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", default=str(DEFAULT_LOG_DIR))
    common.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    common.add_argument("--gui-errors", action="store_true")

    p = argparse.ArgumentParser(
        prog="service_control.py",
        description="KG-RAG 无窗口服务控制",
    )
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("start", parents=[common],
                       help="无窗口启动 Neo4j/Ollama/API（新版前端在 /ui）")
    s.add_argument("--api-port", type=int, default=None)
    s.add_argument("--ui-port", type=int, default=None)
    s.add_argument("--no-ui", action="store_true", help="不校验/不打开前端")
    s.add_argument("--legacy-gradio", action="store_true",
                   help="显式启动旧 Gradio（默认不启动；仅调试）")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--no-neo4j", action="store_true")
    s.add_argument("--no-ollama", action="store_true")
    s.add_argument("--wait", type=int, default=240, help="API 就绪等待秒数")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("stop", parents=[common], help="停止 API/WebUI")
    s.add_argument("--api-port", type=int, default=None)
    s.add_argument("--ui-port", type=int, default=None)
    s.add_argument("--ollama", action="store_true", help="一并停止 Ollama")
    s.add_argument("--no-legacy", dest="legacy", action="store_false",
                   help="不按旧窗口标题停止")
    s.set_defaults(func=cmd_stop, legacy=True)

    s = sub.add_parser("status", parents=[common], help="查看运行状态")
    s.add_argument("--api-port", type=int, default=None)
    s.add_argument("--ui-port", type=int, default=None)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("restart-ollama", parents=[common],
                       help="把现有 Ollama 重启为无窗口模式")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_restart_ollama)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
