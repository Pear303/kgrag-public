# KG-RAG 工业设备故障诊断问答系统

面向工业设备维修手册的知识图谱检索增强生成（KG-RAG）问答系统。系统把手册整理为可检索、可追溯的知识库，先检索手册依据，再由本地小模型生成回答；依据不足时明确拒答。

## 主要能力

- **手册入库**：Markdown / PDF / DOCX 手册解析、分块、表格与报警号结构识别。
- **知识图谱**：LLM 抽取实体与关系，规则清洗后写入 Neo4j（`:Entity` / `:REL`），实体带向量索引，支持多跳路径检索。
- **社区摘要**：Louvain 社区发现 + 社区描述向量匹配，支持宏观问题。
- **混合检索**：图谱路径检索、社区召回、滑动窗口向量检索，按槽位装配上下文。
- **本地生成**：生产答案固定由本机 Ollama 模型生成，不依赖外部大模型。
- **拒答护栏**：超纲/无关问题识别，依据不足时拒答。
- **Web 界面与 REST API**：Vue3 前端 + FastAPI 后端，支持流式回答、手册管理、推理链可视化。
- **评测链路**：RAGAS 四指标、配对统计、自助法置信区间。

## 架构

```
维修手册 (md/pdf/docx)
        │
        ▼
  解析与分块 ──► 表格/报警号结构识别
        │
        ▼
  LLM 实体关系抽取 ──► 规则清洗
        │
        ▼
  Neo4j 图谱（:Entity / :REL + 向量索引）
        │
        ├──► Louvain 社区发现 ──► 社区摘要
        │
        ▼
  查询路由与上下文装配
        ├── 图谱多跳路径
        ├── 社区召回
        └── 滑动窗口向量检索
        │
        ▼
  本机 Ollama 生成 ──► 答案 + 依据 + 推理链
```

## 目录结构

```
.
├── api_server.py            # FastAPI 服务与手册管理
├── serve.py                 # 问答服务层（图谱/社区/向量 + 生成）
├── main.py                  # 建库与评测管道入口
├── run_eval.py              # 全链路评测入口
├── config/                  # 配置与 LLM/Embedding 工厂
├── backend/                 # 解析、抽取、图谱、检索、评测等核心模块
├── pipeline/                # 建库管道阶段与编排
├── frontend/                # 旧版 Gradio 调试界面
├── webui/                   # Vue3 前端（构建到 webui/dist）
├── scripts/                 # 建库、环境检查、服务控制
├── tests/                   # 单元测试
├── lib/ static/             # 前端静态依赖
└── requirements.txt
```

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10–3.12 | |
| Neo4j | 5.x | 图谱存储，需启用向量索引 |
| Ollama | 最新 | 本地生成与向量模型 |
| Node.js | 18+ | 仅构建前端时需要 |

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux/macOS
```

至少修改：

- `NEO4J_URI` / `SERVING_NEO4J_URI`：建图库与查询库地址
- `NEO4J_PASSWORD`
- `LOCAL_LLM_MODEL` / `LOCAL_EMBEDDING_MODEL`
- `SERVING_MANUAL` / `SERVING_EXTRACT` / `SERVING_ROUND`

拉取本地模型：

```bash
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b
```

### 3. 准备手册

把手册放到 `data/manuals/`，例如 `data/manuals/your_manual.md`。

### 4. 构建知识库

先建一个隔离的 Neo4j 实例（示例端口 7691），再运行：

```bash
python scripts/build_kb.py \
  --manual data/manuals/your_manual.md \
  --uri bolt://localhost:7691
```

若已有 LLM 抽取产物，可加 `--extract reports/kg_visual/your_extract.json` 一并导入。

> `build_kb.py` 不给默认目标库；7688/7690/7692 被视为受保护实例，需显式 `--allow-protected` 才会写入。

### 5. 构建前端并启动服务

```bash
cd webui
npm ci
npm run build
cd ..

python api_server.py
```

打开 `http://127.0.0.1:8001/ui/`；API 文档在 `http://127.0.0.1:8001/docs`。

### 6. 环境自检

```bash
python scripts/check_env.py
```

## 评测

```bash
python run_eval.py --doc your_manual.md --qa-count 20
```

评测使用 RAGAS 0.4.3 的四项无参考指标（ faithfulness、answer relevancy、context precision、context recall），并支持配对差值、自助法置信区间与符号检验。评测生成器与生产生成器可独立配置，但生产答案固定使用本机模型。

## 配置说明

| 变量 | 作用 |
|------|------|
| `LLM_PROVIDER` | `local`（默认）或 `online` |
| `EMBEDDING_PROVIDER` | `local`（默认）或 `online` |
| `LOCAL_LLM_BASE_URL` | Ollama 地址，默认 `http://localhost:11434` |
| `NEO4J_URI` | 建图/上传库 |
| `SERVING_NEO4J_URI` | 问答查询库 |
| `API_TOKEN` | 可选；设置后除健康检查与静态页外要求 `X-API-Token` |
| `FROZEN_API_HOSTS` | 可选；逗号分隔的禁用主机黑名单 |

## 设计约束

- 生产答案必须由本机模型生成；外部模型只用于建库抽取、出题与评测判分。
- 默认只针对单一手册构建图谱；跨手册结论需要独立验证。
- 手册、抽取产物、评测数据与运行缓存不随仓库分发，需自行准备。

## 说明

本项目复现并扩展了一篇公开的 KG-RAG 工程实践文章的 L1 骨架，补齐了数据管道、检索装配、评测与 Web 服务。仓库仅包含源码与配置模板，不包含任何维修手册原文、密钥或实验数据。
