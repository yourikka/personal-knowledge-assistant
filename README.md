# 个人知识库智能整理助手

这是一个面向个人知识管理的多 Agent MVP。它把网页、PDF、Markdown、图片和纯文本统一接入入库流水线，自动完成解析、清洗、分类、摘要、关联和问答，并额外支持生图 Agent。

## 一句话做什么

你平时收藏的网页、PDF、笔记、图片自动入库，按主题分类、摘要、打标签、建立关联，支持自然语言检索和问答。

## 当前实现范围

- 入库主链路使用 `LangGraph StateGraph` 编排：采集、解析、清洗、切块、分类打标、摘要、持久化、知识关联
- 检索问答和生图作为独立 Agent 能力接入 API 层
- `SQLite` 落元数据、原文、摘要、标签、chunk、关联关系、会话历史和长期记忆
- `Chroma` 可选接入；向量默认可切到智谱 `embedding-3`，未配置时回退到本地哈希向量检索
- 网页采集优先走 `urllib + BeautifulSoup`，可选启用 `Playwright`
- 图片 OCR 优先走 `pytesseract`，本机未安装 Tesseract 时自动降级
- PDF 优先走 `PyPDF2`，不可用时回退为文本解码
- 分类、摘要、问答 Agent 默认优先调用 `gpt-5.4`
- 生图 Agent 默认调用 `gpt-image-2`
- 支持第三方 OpenAI 兼容 API，只需要提供兼容的 `base_url + api_key`
- 检索问答使用 chunk 级增强 RAG：高质量切块、查询改写、多路召回、重排、MMR 去冗余、上下文压缩、引用回答
- 检索问答支持记忆系统：按会话召回相关记忆，回答后自动提炼用户偏好、目标、决策和稳定事实

## 项目结构

- `app/main.py`：FastAPI 入口
- `app/config.py`：环境配置
- `app/db.py`：SQLite 仓储
- `app/models.py`：请求、响应和流水线状态模型
- `app/pipeline/orchestrator.py`：LangGraph 编排器
- `app/pipeline/agents/`：各 Agent 实现
- `app/services/parser_utils.py`：网页、PDF、Markdown、图片解析
- `app/services/chunking.py`：文档切块策略，保留标题路径、字符范围和 overlap
- `app/services/text_utils.py`：文本清洗、标签、摘要、向量工具
- `app/services/embedding_service.py`：智谱 `embedding-3` / 本地回退向量生成
- `app/services/memory_service.py`：会话记忆召回、格式化、提炼和向量索引
- `app/services/vector_store.py`：Chroma / 本地向量检索适配
- `app/static/`：知识入库、文档内容查看、切片、检索和文档管理 Web 前端
- `evals/`：离线 RAG 评测样例
- `scripts/eval_rag.py`：RAG 检索与回答质量评测脚本
- `docs/spec.md`：项目完整产品与技术规格说明
- `docs/improvement-spec.md`：项目增强路线与技术升级规格说明
- `docs/ui-behavior.md`：前端交互约定，约束文档列表区、阅读页和滚动行为

## 运行方式

1. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 复制环境变量

```bash
cp .env.example .env
```

3. 启动服务

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

说明：

- 根路径 `/` 直接提供 Web 工作台，区块按内容入库、处理流水线、文档内容、切片结果、分类摘要、知识关联和检索问答组织。
- 启动服务后，用浏览器访问 `http://127.0.0.1:8010/` 即可使用。

## 主要接口

### 健康检查

```bash
curl http://127.0.0.1:8010/health
```

### 文档入库

```bash
curl -X POST http://127.0.0.1:8010/api/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "markdown",
    "source": "# LangGraph\n\nLangGraph 适合做多节点 Agent 编排。",
    "title": "LangGraph 笔记"
  }'
```

### 查看文档切片

```bash
curl http://127.0.0.1:8010/api/knowledge/documents/{document_id}/chunks
```

### 删除文档

```bash
curl -X DELETE http://127.0.0.1:8010/api/knowledge/documents/{document_id}
```

如果运行环境不方便发送 `DELETE` 请求，也可以使用兼容入口：

```bash
curl -X POST http://127.0.0.1:8010/api/knowledge/documents/{document_id}/delete
```

### 文件上传入库

```bash
curl -X POST http://127.0.0.1:8010/api/knowledge/upload \
  -F "file=@/absolute/path/to/note.md" \
  -F "source_type=markdown" \
  -F "title=我的笔记"
```

### 问答检索

```bash
curl -X POST http://127.0.0.1:8010/api/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我收藏的内容里哪些和 Agent 编排有关？",
    "top_k": 3,
    "session_id": "demo-session"
  }'
```

### 生图

```bash
curl -X POST http://127.0.0.1:8010/api/images/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一个极简的个人知识库控制台界面插画，强调多 Agent 流水线和知识网络",
    "size": "1024x1024",
    "quality": "high"
  }'
```

## Agent 对应职责

1. 数据采集 Agent：读取 URL、本地文件或内联文本，做黑名单校验和去重。
2. 格式解析 Agent：按 HTML、PDF、Markdown、图片等格式提纯文本并提取元数据。
3. 内容清洗 Agent：去脚本、去广告噪声、修正常见乱码、统一空白格式。
4. 文章切块 Agent：按标题、段落、句子边界生成 chunk，保存字符范围和标题路径。
5. 分类打标 Agent：按技术、生活、学习三大主题分类，并产出 3 到 5 个标签。
6. 摘要提取 Agent：抽取核心句，生成 100 到 200 字摘要。
7. 知识关联 Agent：计算相似度，建立双向链接。
8. 检索问答 Agent：支持自然语言检索，返回摘要、chunk 引用和来源。
9. 生图 Agent：按知识主题或检索结果生成配图、封面或概念图。

## LangGraph 编排

入库主链路不是手写顺序调用，而是 `StateGraph`：

```text
agent_acquisition
  ├─ duplicate -> END
  └─ parse -> agent_parser -> agent_cleaning -> agent_chunking -> agent_classification -> agent_summary -> persist -> agent_linking -> END
```

节点之间传递 `PipelineState`，重复文档会在采集节点后通过条件边直接结束，不会重复解析、摘要或写库。

## RAG 策略

在线问答和离线知识关联共用同一套 chunk 级 `RAGService`：

- 高质量切块：优先保留 Markdown 标题路径、段落和句子边界，超长内容用带 overlap 的滑窗兜底。
- 查询改写：结合会话历史和关键词生成多个检索表达。
- 多路召回：以 chunk 为候选，同时使用向量召回、关键词召回、标签和摘要信号。
- 混合重排：综合语义分、词项重叠、召回信号、标签命中和新近度。
- MMR 选择：减少重复 chunk，优先保留互补信息。
- 上下文压缩：按字符预算组装标题、来源、分类、标签、摘要、标题路径和 chunk 原文。
- 引用回答：问答 Agent 要求模型只基于上下文回答，并使用 `[1]`、`[2]` 引用来源；接口引用会返回 `chunk_id`、`chunk_index`、`char_start`、`char_end`。

`LinkingAgent` 也复用这套 RAG 召回结果来建立双向链接，关联边会保留召回信号，便于后续做图谱展示和排障。

## 记忆系统

记忆系统只接在检索问答链路，不影响入库、分类、摘要和切块这些可重复执行的文档处理节点。

- 短期记忆：`chat_turns` 保存多轮会话历史，用于查询改写和回答上下文。
- 长期记忆：`memory_records` 保存用户偏好、长期目标、项目决策和稳定事实。
- 记忆召回：`MemoryService` 使用向量检索和关键词检索召回当前问题相关记忆。
- 记忆写回：`QueryAgent` 在回答后提炼最多 3 条高价值记忆，写入 SQLite 并加入向量索引。
- 记忆边界：文档事实仍必须来自 RAG 上下文引用；记忆只辅助理解用户偏好和项目背景。

## 设计说明

- 这是一个可运行 MVP，不是最终生产版。
- 当前已接 OpenAI，分类、摘要、问答 Agent 优先使用 `gpt-5.4`；未配置 `OPENAI_API_KEY` 时自动回退本地规则。
- 生图 Agent 使用 `gpt-image-2`；未配置 `OPENAI_API_KEY` 时会返回可解释的空结果而不是直接崩溃。
- 基础版 Agent 已补充运行统计、来源元数据、清洗前后统计和知识关联图谱结构。

## 当前已完成能力说明

- URL、Markdown、纯文本、本地 PDF、本地图片五类入口已经打通
- 支持直接上传文件走入库流程
- 支持查询文档列表、文档详情、关联结果和自然语言问答
- 支持调用生图 Agent 生成知识库封面图或概念图
- 支持会话历史和长期记忆写入 SQLite，用于多轮上下文增强
- OCR、Playwright、Chroma 都是可选能力，不会阻塞基础功能启动
- 支持 `python scripts/smoke_test.py` 做基础回归测试
- 支持 `python scripts/eval_rag.py` 做离线 RAG 质量评测

## OpenAI 配置

在 `.env` 里至少补齐：

```env
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://你的第三方兼容接口/v1
OPENAI_TEXT_MODEL=gpt-5.4
OPENAI_IMAGE_MODEL=gpt-image-2
EMBEDDING_PROVIDER=zhipu
EMBEDDING_API_KEY=你的智谱key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=2048
EMBEDDING_TIMEOUT_SECONDS=60
EMBEDDING_PATH=/embeddings
OPENAI_TEXT_TIMEOUT_SECONDS=60
OPENAI_IMAGE_TIMEOUT_SECONDS=180
OPENAI_CHAT_COMPLETIONS_PATH=/chat/completions
OPENAI_IMAGE_GENERATIONS_PATH=/images/generations
RAG_REWRITE_ENABLED=true
RAG_MULTI_QUERY_LIMIT=4
RAG_CANDIDATE_MULTIPLIER=5
RAG_MMR_LAMBDA=0.72
RAG_CONTEXT_CHAR_BUDGET=6000
MEMORY_ENABLED=true
MEMORY_TOP_K=5
MEMORY_MIN_SCORE=0.12
MEMORY_WRITE_LIMIT=3
MEMORY_MAX_CONTENT_CHARS=500
MEMORY_BOOTSTRAP_LIMIT=1000
CHUNK_TARGET_CHARS=900
CHUNK_OVERLAP_CHARS=160
CHUNK_MIN_CHARS=180
CHUNK_MAX_CHARS=1400
```

说明：

- 文本类 Agent 统一走 `gpt-5.4`
- 生图 Agent 走 `gpt-image-2`
- 向量检索推荐走智谱 `embedding-3`
- 当前实现走的是 OpenAI 兼容 HTTP 接口：`/chat/completions` 和 `/images/generations`
- 向量接口单独走智谱官方兼容路径：`/embeddings`
- 也就是说你用第三方 API 时，不需要额外安装 OpenAI 官方 SDK
- 生图通常比文本慢得多，所以单独提供了 `OPENAI_IMAGE_TIMEOUT_SECONDS`
- 如果没配 `EMBEDDING_PROVIDER=zhipu` 或没配 `EMBEDDING_API_KEY`，系统会回退到本地 hash embedding
- 记忆系统默认开启；如需关闭，可设置 `MEMORY_ENABLED=false`
- 如果第三方平台路径不是标准 OpenAI 路径，可以改：
  - `OPENAI_CHAT_COMPLETIONS_PATH`
  - `OPENAI_IMAGE_GENERATIONS_PATH`
- 如果你强行写成 `gpt5.4` 或 `image2`，那不是标准模型 ID，请求会失败
