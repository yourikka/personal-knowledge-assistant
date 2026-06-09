# 个人知识库智能整理助手 Spec

## 文档信息

- 项目名称：个人知识库智能整理助手
- 文档类型：产品与技术规格说明
- 当前阶段：MVP 已实现，进入增强迭代阶段
- 适用代码库：`personal-knowledge-assistant`
- 关联文档：
  - [README.md](/home/rikka/.marscode/personal-knowledge-assistant/README.md)
  - [docs/ui-behavior.md](/home/rikka/.marscode/personal-knowledge-assistant/docs/ui-behavior.md)
  - [docs/improvement-spec.md](/home/rikka/.marscode/personal-knowledge-assistant/docs/improvement-spec.md)

## 1. 背景与目标

### 1.1 问题定义

个人知识收集通常分散在网页收藏、Markdown 笔记、PDF、截图、图片和临时文本里。原始资料有几个典型问题：

- 来源分散，无法统一检索
- 文档结构不一致，难以直接比较和复用
- 同类内容重复收藏，缺少去重和关联
- 收藏之后缺少二次整理，无法形成持续可用的知识库
- 用户提问时，很难快速定位到最相关的原文片段

### 1.2 产品目标

本项目的目标是把异构内容统一接入一个可检索、可问答、可追溯来源的个人知识库，并通过多 Agent 流水线完成自动整理。

核心目标：

- 支持多种内容源统一入库
- 自动完成解析、清洗、切块、分类、摘要和知识关联
- 支持基于 chunk 的 RAG 问答，并返回引用来源
- 支持会话记忆与长期记忆，提升多轮问答连续性
- 提供可直接操作的 Web 工作台

### 1.3 非目标

以下内容不属于当前阶段必须实现的目标：

- 企业级多租户权限系统
- 复杂团队协作流程
- 大规模分布式调度和多节点部署
- 完整知识图谱实体抽取与图查询引擎
- 高并发生产环境 SLA 保证

## 2. 用户与使用场景

### 2.1 目标用户

- 重度收藏网页、论文和教程的开发者
- 习惯积累个人笔记和学习资料的学生或研究者
- 需要维护个人资料库并进行问答检索的内容工作者

### 2.2 核心场景

#### 场景 A：快速入库

用户输入 URL、上传 PDF、上传截图、粘贴 Markdown 或纯文本，系统自动完成：

1. 采集与去重
2. 格式解析
3. 内容清洗
4. 切块
5. 分类打标
6. 摘要提取
7. 相似文档关联

#### 场景 B：知识检索问答

用户围绕已收藏资料发起自然语言问题，例如：

- “我收藏的内容里哪些和 LangGraph 有关？”
- “这篇文章的核心结论是什么？”
- “我以前记录过关于 RAG 重排的资料吗？”

系统返回：

- 简洁回答
- 引用的原始 chunk
- 对应文档标题、来源和摘要
- 相关记忆（如果有）

#### 场景 C：持续记忆

用户在多轮会话中表达偏好、长期目标和项目背景，例如：

- “记住：以后默认用中文简洁回答”
- “这个项目以后优先走 LangGraph”

系统自动将高价值信息提炼为长期记忆，并在后续问答中辅助理解用户意图。

#### 场景 D：知识再表达

用户基于入库知识生成配图或概念图，用于文章封面、思维导图素材或项目展示。

## 3. 功能范围

### 3.1 已实现功能

- 文档入库：URL、Markdown、纯文本、本地 PDF、本地图片
- 文件上传接口
- LangGraph 编排的入库流水线
- chunk 级 RAG 检索与问答
- 长期记忆与会话记忆
- 文档列表、详情、切片、删除
- 知识关联
- 生图 Agent
- Web 控制台

### 3.2 本阶段必须保证的能力

- 输入内容可以稳定入库
- 文档可以稳定生成切片、摘要、分类和标签
- 问答必须返回来源引用
- 删除文档后，元数据和向量索引应同步清理
- 问答记忆必须可召回、可写回、可删除

### 3.3 下一阶段推荐增强

- 分层召回与 rerank
- 知识图谱增强
- 自动评测系统
- 异步任务编排
- 增量重建索引

## 4. 系统总览

### 4.1 逻辑架构

系统由四个层次组成：

1. 接入层
   - FastAPI API
   - Web 前端

2. 编排层
   - LangGraph `StateGraph`
   - QueryAgent / ImageGenerationAgent

3. 能力层
   - Acquisition / Parser / Cleaning / Chunking / Classification / Summary / Linking / Query / Memory / Image Generation
   - RAGService / VectorStore / EmbeddingService / OpenAIService

4. 存储层
   - SQLite：元数据、chunk、链接、聊天记录、长期记忆
   - Chroma：文档 chunk 和记忆向量

### 4.2 当前入库图

```text
agent_acquisition
  ├─ duplicate -> END
  └─ parse -> agent_parser -> agent_cleaning -> agent_chunking -> agent_classification -> agent_summary -> persist -> agent_linking -> END
```

### 4.3 当前问答链路

```text
用户问题
-> QueryAgent
-> MemoryService.retrieve
-> RAGService.retrieve
-> LLM 组合回答
-> 保存 chat_turns
-> MemoryService.learn_from_turn
```

## 5. Agent 规格

本节定义每个 Agent 的输入、输出、关键行为和约束。

### 5.1 AcquisitionAgent

职责：

- 接收 `IngestRequest`
- 识别输入类型
- 获取原始内容
- 生成 `document_id`
- 基于 fingerprint 去重

输入：

- `source_type`
- `source`
- `title`
- `metadata`

输出：

- `PipelineState.raw_bytes`
- `PipelineState.source_uri`
- `PipelineState.fingerprint`
- `PipelineState.duplicate_of`

约束：

- URL 必须经过黑名单检查
- 文件大小不能超过 `MAX_SOURCE_BYTES`
- 命中重复后必须终止后续流程

### 5.2 ParserAgent

职责：

- 将 HTML/PDF/Markdown/图片等内容转换成纯文本
- 提取标题、来源、时间等元数据

输出：

- `parsed_text`
- `title`
- `metadata`

约束：

- 保留尽可能多的来源元信息
- 失败时要给出可解释错误，而不是 silent failure

### 5.3 CleaningAgent

职责：

- 清理脚本、广告、噪声、异常空白和常见乱码

输出：

- `cleaned_text`

约束：

- 不应篡改正文核心语义
- 清洗后文本应适合切块和摘要

### 5.4 ChunkingAgent

职责：

- 将清洗后的文档按标题层级、段落边界和句子边界切块
- 在必要时使用 overlap 滑窗兜底

输出：

- `chunks[]`
  - `id`
  - `chunk_index`
  - `text`
  - `char_start`
  - `char_end`
  - `metadata.heading_path`

约束：

- 尽量避免跨章节混块
- 尽量让 chunk 保持语义完整
- chunk 不能过短也不能过长

### 5.5 ClassificationAgent

职责：

- 输出主题分类和标签

当前分类域：

- `技术`
- `生活`
- `学习`

输出：

- `category`
- `confidence`
- `tags`

约束：

- 标签数量为 3 到 5 个
- `confidence` 范围必须在 0 到 1
- 模型不可用时要回退到本地规则

### 5.6 SummaryAgent

职责：

- 为文档生成 100 到 200 字的中文摘要

输出：

- `summary`

约束：

- 保留核心观点、关键对象和主要结论
- 不允许出现模板化废话

### 5.7 LinkingAgent

职责：

- 基于文档内容和摘要召回相似文档
- 建立双向链接
- 输出简化知识图结构

输出：

- `related`
- `graph`

约束：

- 使用统一的 RAG 召回逻辑
- 低于阈值的关联必须丢弃

### 5.8 QueryAgent

职责：

- 基于文档向量、关键词和记忆完成问答
- 生成带引用的答案

输出：

- `answer`
- `references`
- `memories`
- `logs`

约束：

- 回答必须优先基于 RAG 上下文
- 记忆只能辅助用户意图理解，不能替代文档证据
- 如果上下文不足，要明确说明不足

### 5.9 MemoryService

职责：

- 管理短期与长期记忆
- 召回当前问题相关记忆
- 从聊天轮次中提炼高价值记忆

长期记忆类型：

- `preference`
- `goal`
- `decision`
- `fact`
- `project`

约束：

- 记忆必须支持按 `session_id` 隔离
- 只保存稳定且未来可复用的信息
- 需要可删除

### 5.10 ImageGenerationAgent

职责：

- 将用户需求改写为高质量生图提示词
- 调用图像模型生成图片

输出：

- `prompt`
- `revised_prompt`
- `model`
- `image_b64` 或 `image_url`

约束：

- 原始意图不能被错误篡改
- 改写失败时回退原始提示词

## 6. 数据模型

### 6.1 documents

用途：

- 存储文档主记录

关键字段：

- `id`
- `fingerprint`
- `source_type`
- `source_uri`
- `title`
- `raw_text`
- `cleaned_text`
- `summary`
- `category`
- `confidence`
- `tags_json`
- `metadata_json`
- `created_at`
- `updated_at`

### 6.2 document_chunks

用途：

- 存储文档切片

关键字段：

- `id`
- `document_id`
- `chunk_index`
- `text`
- `char_start`
- `char_end`
- `metadata_json`

### 6.3 document_links

用途：

- 存储文档间相似关联

关键字段：

- `source_id`
- `target_id`
- `score`

### 6.4 chat_turns

用途：

- 存储多轮问答历史

关键字段：

- `session_id`
- `role`
- `content`
- `created_at`

### 6.5 memory_records

用途：

- 存储长期记忆

关键字段：

- `id`
- `session_id`
- `kind`
- `content`
- `importance`
- `tags_json`
- `metadata_json`
- `created_at`
- `updated_at`

## 7. 检索与问答规格

### 7.1 检索策略

当前检索是 chunk 级混合检索，包含：

- 多查询改写
- 向量检索
- 关键词检索
- 标签和摘要信号
- 新近度信号
- MMR 去重

### 7.2 召回顺序

1. 查询改写
2. 多路候选召回
3. 混合打分重排
4. MMR 选取
5. 上下文预算压缩
6. LLM 生成答案

### 7.3 回答规则

- 回答语言默认中文
- 回答应简洁、可读
- 文档事实必须带 `[1]`、`[2]` 这类引用
- 记忆类信息可使用 `[M1]`、`[M2]`
- 没有足够上下文时必须显式说明

### 7.4 记忆参与规则

- 优先作为用户偏好和背景补充
- 不允许用记忆替代原文证据
- 当知识库没有相关文档但有记忆时，可退化为记忆回答

## 8. API 规格

### 8.1 核心接口

- `GET /health`
- `POST /api/knowledge/ingest`
- `POST /api/knowledge/upload`
- `POST /api/knowledge/query`
- `POST /api/jobs/documents/{document_id}/enrich`
- `GET /api/knowledge/documents`
- `GET /api/knowledge/documents/{document_id}`
- `GET /api/knowledge/documents/{document_id}/chunks`
- `GET /api/knowledge/documents/{document_id}/graph`
- `DELETE /api/knowledge/documents/{document_id}`
- `POST /api/knowledge/documents/{document_id}/delete`
- `POST /api/knowledge/reindex`
- `POST /api/images/generate`
- `GET /api/memories`
- `PATCH /api/memories/{memory_id}`
- `DELETE /api/memories/{memory_id}`

### 8.2 入库接口要求

- 请求必须包含 `source_type` 与 `source`
- 成功后返回：
  - `document_id`
  - `duplicate`
  - `title`
  - `category`
  - `tags`
  - `summary`
  - `related`
  - `graph`
  - `logs`

### 8.3 问答接口要求

- 请求：
  - `query`
  - `top_k`
  - `session_id`
- 响应：
  - `answer`
  - `references`
  - `memories`
  - `session_id`
  - `logs`

## 9. Web 工作台规格

### 9.1 页面目标

Web 工作台不是营销页，而是直接可操作的知识整理控制台。

### 9.2 主区块

- 内容入库
- 处理流水线
- 文档内容
- 切片结果
- 检索问答
- 知识关联
- 运行日志
- 生成配图

### 9.3 文档内容区约束

- 只显示文档卡片概览
- 区块独立滚动
- 点击卡片进入阅读页

### 9.4 阅读页约束

- 居中弹层
- 正文独立滚动
- 支持原文/清洗后视图切换

## 10. 配置规格

### 10.1 基础配置

- `APP_HOST`
- `APP_PORT`
- `SQLITE_PATH`
- `CHROMA_DIR`

### 10.2 模型配置

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_TEXT_MODEL`
- `OPENAI_IMAGE_MODEL`
- `OPENAI_CHAT_COMPLETIONS_PATH`
- `OPENAI_IMAGE_GENERATIONS_PATH`

### 10.3 向量配置

- `EMBEDDING_PROVIDER`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- `EMBEDDING_PATH`

### 10.4 RAG 配置

- `RAG_REWRITE_ENABLED`
- `RAG_MULTI_QUERY_LIMIT`
- `RAG_CANDIDATE_MULTIPLIER`
- `RAG_MMR_LAMBDA`
- `RAG_CONTEXT_CHAR_BUDGET`
- `RAG_MIN_SCORE`
- `RAG_RECENT_BOOST`
- `RAG_TAG_BOOST`

### 10.5 Chunk 配置

- `CHUNK_TARGET_CHARS`
- `CHUNK_OVERLAP_CHARS`
- `CHUNK_MIN_CHARS`
- `CHUNK_MAX_CHARS`

### 10.6 Memory 配置

- `MEMORY_ENABLED`
- `MEMORY_TOP_K`
- `MEMORY_MIN_SCORE`
- `MEMORY_WRITE_LIMIT`
- `MEMORY_MAX_CONTENT_CHARS`
- `MEMORY_BOOTSTRAP_LIMIT`

## 11. 非功能性要求

### 11.1 正确性

- 文档入库必须可重放
- 删除文档必须同步删除向量项
- 问答必须可追溯到引用片段

### 11.2 可解释性

- 流水线各步骤要保留日志
- 检索结果要返回引用和来源
- 记忆结果应可枚举和删除

### 11.3 可扩展性

- Agent 能力应通过 service 边界拆分
- RAG 策略应支持后续升级到分层检索和 rerank
- memory、vector、llm provider 应可替换

### 11.4 性能

当前阶段不设硬 SLA，但要求：

- 小体量文档入库可在交互式时间内完成
- 问答请求在合理 `top_k` 范围内可返回结果
- 生图请求允许单独较长超时

## 12. 安全与边界

- URL 输入必须经过黑名单过滤
- 不默认抓取内网地址
- 不在代码库中保存真实密钥
- 兼容第三方 OpenAI 接口，但必须允许超时和错误回传

## 13. 测试要求

### 13.1 当前必须覆盖

- 删除文档后，SQLite 与向量索引同步删除
- EmbeddingService 在本地与外部 provider 间的回退逻辑
- MemoryService 的写入、召回和 session 隔离
- 没有文档时的记忆回答退化路径

### 13.2 推荐后续覆盖

- 各类输入源的解析测试
- chunk 边界质量测试
- RAG 回答引用一致性测试
- 前端关键工作流的 API 联调测试

## 14. 里程碑建议

### M1：当前 MVP

- 多源入库
- 基础 RAG
- Web 工作台
- 生图
- 记忆系统

### M2：质量增强

- 分层检索
- rerank
- 自动评测基线
- 更细粒度的记忆压缩

### M3：高级知识能力

- 实体关系抽取
- 图谱增强检索
- 增量重建索引
- 异步任务队列

## 15. 验收标准

满足以下条件视为当前版本达标：

- 用户可以输入 URL、文本或上传文件完成入库
- 文档能生成摘要、分类、标签与 chunk
- 问答能返回可追溯引用
- 长期记忆能被召回并参与回答
- 文档删除后不会残留脏向量索引
- Web 端可完成主流程：入库、浏览、问答、删除、查看切片、任务管理和记忆编辑

## 16. 开放问题

以下问题留待后续版本处理：

- 是否要把长期记忆拆成全局记忆与项目记忆
- 是否引入 cross-encoder 或 LLM rerank
- 是否为不同文档类型设计不同切块策略
- 是否需要把任务队列扩展为可持久化多进程 Worker
- 是否需要为记忆管理增加批量合并、冲突解决和过期策略配置
