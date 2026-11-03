# -*- coding: utf-8 -*-
"""环境自检脚本 —— 检查 Python/依赖/Neo4j/Ollama/.env 就绪状态
用法：python scripts/check_env.py
输出：每项 [OK] / [MISSING]，最后汇总 + 修复指引。"""
import os
import sys
import io
import json

out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout = out

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ok = []
warn = []
fail = []


def report(name, status, detail=""):
    tag = {"OK": "OK", "WARN": "WARN", "FAIL": "FAIL"}[status]
    mark = {"OK": "[OK]    ", "WARN": "[WARN]  ", "FAIL": "[FAIL]  "}[status]
    line = f"{mark}{name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if status == "OK":
        ok.append(name)
    elif status == "WARN":
        warn.append(name)
    else:
        fail.append(name)


# 1. Python 版本
print("=" * 62)
print("KG-RAG 环境自检")
print("=" * 62)
pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
if sys.version_info >= (3, 10):
    report(f"Python {pyver}", "OK")
else:
    report(f"Python {pyver}（需 >=3.10）", "FAIL", "请安装 Python 3.10+")


# 2. 依赖（requirements.txt 顶层包）
print("-" * 62)
print("依赖检查（requirements.txt 顶层）")
print("-" * 62)
REQ_TOP = ["ragas", "llama_index", "langchain_core", "langchain_ollama",
           "langchain_openai", "openai", "ollama", "neo4j", "networkx",
           "pyvis", "numpy", "pandas", "jieba", "nltk", "datasets",
           "fastapi", "uvicorn", "gradio", "requests", "dotenv", "pydantic"]
missing = []
for pkg in REQ_TOP:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if not missing:
    report("全部依赖已安装", "OK", f"{len(REQ_TOP)} 个包")
else:
    report(f"缺失 {len(missing)} 个依赖", "FAIL", ", ".join(missing))
    print("  -> 运行: pip install -r requirements.txt")


# 3. .env 是否存在
print("-" * 62)
print(".env 配置")
print("-" * 62)
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    report(".env 文件存在", "OK")
    # 检查关键 key
    env = {}
    try:
        with open(env_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if "=" in ln and not ln.startswith("#"):
                    k, _, v = ln.partition("=")
                    env[k.strip()] = v.strip()
    except Exception:
        pass
    needs = ["API_BASE_URL_1", "API_MODEL_1", "NEO4J_EVAL_URI",
             "LOCAL_EMBEDDING_MODEL"]
    miss_key = [k for k in needs if k not in env]
    if miss_key:
        report("缺少关键配置项", "WARN", ", ".join(miss_key))
    else:
        report("关键配置项齐全", "OK")
else:
    report(".env 文件", "FAIL", "请复制 .env.example 为 .env 并填写 API key")


# 4. Neo4j
print("-" * 62)
print("Neo4j")
print("-" * 62)
neo_uri = None
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    neo_uri = os.getenv("SERVING_NEO4J_URI") or os.getenv("NEO4J_EVAL_URI") or "bolt://localhost:7688"
except Exception:
    neo_uri = "bolt://localhost:7688"
try:
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(neo_uri, auth=("neo4j", os.environ.get('NEO4J_PASSWORD', '')))
    drv.verify_connectivity()
    drv.close()
    report(f"Neo4j 连通（{neo_uri}）", "OK")
except Exception as e:
    report(f"Neo4j 不可达（{neo_uri}）", "FAIL", str(e)[:60])
    print("  -> 确认 Neo4j 服务已启动（服务名 neo4j-pump）")


# 5. Ollama
print("-" * 62)
print("Ollama / Embedding")
print("-" * 62)
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code == 200:
        models = [m.get("name", "") for m in r.json().get("models", [])]
        report("Ollama 服务在线", "OK", f"{len(models)} 个模型")
        embed_model = os.getenv("LOCAL_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        if any(embed_model in m for m in models):
            report(f"embedding 模型 {embed_model}", "OK")
        else:
            report(f"embedding 模型 {embed_model}", "WARN", "需 ollama pull")
    else:
        report("Ollama 服务", "FAIL", f"HTTP {r.status_code}")
except Exception:
    report("Ollama 服务（localhost:11434）", "FAIL", "未运行")
    print("  -> 启动 Ollama: ollama serve")


# 6. API 可达性（可选，不阻塞）
print("-" * 62)
print("LLM API（deepseek 代理）")
print("-" * 62)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    base = os.getenv("API_BASE_URL_1", "")
    key = os.getenv("DEEPSEEK_API_KEY_1", "")
    if base and key:
        import requests
        r = requests.post(base + "/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": os.getenv("API_MODEL_1", "deepseek-v4-flash"),
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 5},
                          timeout=30)
        report(f"API 可达（{base.split('//')[-1]}）", "OK" if r.status_code == 200 else "WARN",
               f"HTTP {r.status_code}")
    else:
        report("API key 未配置", "WARN")
except Exception as e:
    report("API 可达性", "WARN", str(e)[:50])


# 汇总
print("=" * 62)
print(f"汇总：{len(ok)} 项 OK / {len(warn)} 项 WARN / {len(fail)} 项 FAIL")
if fail:
    print("有 FAIL 项，先修复后再继续。")
elif warn:
    print("有 WARN 项（不阻塞核心功能），建议补齐。")
else:
    print("环境就绪，可以启动系统或复现评测。")
print("=" * 62)
sys.exit(1 if fail else 0)