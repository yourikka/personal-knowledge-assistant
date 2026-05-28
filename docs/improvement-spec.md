# 个人知识库智能整理助手增强方案 Spec

## 文档信息

- 文档名称：项目完善与技术增强规格说明
- 适用范围：`personal-knowledge-assistant` 后续增强迭代
- 目标读者：项目维护者、后端工程师、Agent/RAG 方向开发者
- 关联文档：
  - [docs/spec.md](/home/rikka/.marscode/personal-knowledge-assistant/docs/spec.md)
  - [README.md](/home/rikka/.marscode/personal-knowledge-assistant/README.md)

## 1. 文档目的

这份文档不重复描述当前系统“已经有什么”，而是回答另一个问题：

- 这个项目下一步应该往哪里增强
- 哪些增强项真正有技术深度
- 每个增强项落地时要改哪些模块
- 做到什么程度算完成

目标是把项目从“能跑的个人知识库 MVP”，提升为“具有明确技术壁垒的知识系统”。

## 2. 当前系统基线

当前系统已经具备以下基础能力：

- 基于 `LangGraph StateGraph` 的入库流水线
- 多源内容采集与解析
- 高质量 chunk 切块
- 分类、摘要、知识关联
- chunk 级 RAG 检索问答
- 会话记忆和长期记忆
- FastAPI + Web 工作台
- SQLite 元数据存储 + Chroma 向量检索

当前系统的短板主要在以下几个方面：

- 检索仍以 chunk 级为主，缺少分层召回
- 关联仍偏“相似文档”，缺少结构化知识关系
- 记忆已经可用，但没有分层、压缩、冲突管理
- 缺少系统性评测
- 入库流程仍是同步执行，扩展性有限

## 3. 设计原则

后续增强必须遵守以下原则：

### 3.1 不破坏当前主链路

- 现有 `ingest -> chunk -> classify -> summary -> persist -> linking` 主链路必须保持可运行
- 新能力优先通过新 service、新表或新配置项接入
- 避免把实验性逻辑直接塞进现有 Agent 主体

### 3.2 可渐进上线

- 每个增强项都必须能独立启停
- 能通过配置项灰度启用
- 失败时要有明确回退路径

### 3.3 可测试

- 每个增强项都必须带最小验收测试
- 至少覆盖核心数据流、降级逻辑和边界条件

### 3.4 可解释

- 检索、重排、记忆、图谱增强都必须保留日志与中间信号
- 问答输出不能因为增强而失去可追溯性

## 4. 增强优先级总览

### 4.1 第一梯队

第一梯队是最值得优先做的增强，技术含量高，并且能直接拉高项目质量上限。

1. 分层检索 + 重排
2. 知识图谱增强问答
3. 记忆分层

### 4.2 第二梯队

第二梯队偏系统工程和质量建设，技术难度也高，但建立在第一梯队之后更合理。

1. 高质量多模态解析
2. 自动评测系统
3. 异步任务系统

### 4.3 第三梯队

第三梯队更偏高级工程能力和个性化体验。

1. 增量索引与实时更新
2. Agent 自反思
3. 个性化检索

## 5. 第一梯队增强规格

## 5.1 分层检索 + 重排

### 5.1.1 目标

将当前单层 chunk 检索升级为：

```text
document -> section -> chunk
```

并在召回后增加更强的 rerank 层，提升问答的相关性、覆盖率和引用精度。

### 5.1.2 当前问题

当前 `RAGService` 的候选单位主要是 chunk，存在几个问题：

- 长文档中高层语义缺失，chunk 容易碎
- 单纯 chunk 召回可能漏掉“相关文档但 chunk 不明显命中”的情况
- 现在的重排主要依赖启发式分数，缺少更强的语义排序层

### 5.1.3 目标能力

- 先召回文档级候选
- 再在候选文档内做 section 级召回
- 最后进入 chunk 级细粒度引用定位
- 在最终候选上接入 cross-encoder rerank 或 LLM rerank

### 5.1.4 建议实现

新增数据层：

- `document_sections` 表
  - `id`
  - `document_id`
  - `section_index`
  - `heading`
  - `heading_path_json`
  - `text`
  - `char_start`
  - `char_end`
  - `metadata_json`

向量索引层新增三类对象：

- `kind=document`
- `kind=section`
- `kind=chunk`

新增服务：

- `HierarchicalRetrievalService`
  - `retrieve_documents()`
  - `retrieve_sections()`
  - `retrieve_chunks()`
  - `merge_candidates()`

对现有模块的影响：

- [app/services/rag_service.py](/home/rikka/.marscode/personal-knowledge-assistant/app/services/rag_service.py)
  从单层召回迁移为可配置的分层召回
- [app/services/vector_store.py](/home/rikka/.marscode/personal-knowledge-assistant/app/services/vector_store.py)
  增加 `section` 类型索引
- `ChunkingAgent`
  需要额外生成 section 结构，而不只是 chunk

### 5.1.5 重排策略

至少分两层：

1. 轻量启发式重排
   - 语义相似度
   - 关键词重合
   - 标签命中
   - 新近度
   - 标题路径命中

2. 深度 rerank
   - 可选 cross-encoder
   - 可选 LLM rerank

建议新增配置项：

- `RAG_HIERARCHICAL_ENABLED`
- `RAG_DOCUMENT_TOP_K`
- `RAG_SECTION_TOP_K`
- `RAG_CHUNK_TOP_K`
- `RAG_RERANK_ENABLED`
- `RAG_RERANK_PROVIDER`
- `RAG_RERANK_TOP_K`

### 5.1.6 验收标准

- 长文档问答时，引用 chunk 的标题路径命中率明显提升
- 对跨段落主题问题，召回覆盖率优于当前版本
- 可以通过配置项关闭分层检索，回退到现有 chunk 模式

## 5.2 知识图谱增强问答

### 5.2.1 目标

将当前“相似文档链接”增强为可用于问答的结构化关系网络。

### 5.2.2 当前问题

当前 `LinkingAgent` 的关联更多是相似性，而不是显式关系。系统无法直接回答：

- “谁提出了这个方法”
- “这个概念和哪些主题相关”
- “哪几篇内容围绕同一个实体展开”

### 5.2.3 目标能力

- 从文档中抽取实体
- 抽取实体间关系
- 进行实体归一化和消歧
- 存储为图结构
- 在问答检索时把图关系作为辅助信号

### 5.2.4 建议实现

新增表：

- `graph_entities`
  - `id`
  - `name`
  - `entity_type`
  - `aliases_json`
  - `metadata_json`

- `graph_edges`
  - `id`
  - `source_entity_id`
  - `target_entity_id`
  - `relation`
  - `confidence`
  - `evidence_document_id`
  - `evidence_chunk_id`
  - `metadata_json`

- `document_entities`
  - `document_id`
  - `entity_id`
  - `mention_count`
  - `first_chunk_id`

新增 Agent 或 Service：

- `GraphExtractionService`
- `EntityNormalizationService`
- `GraphRetrievalService`

### 5.2.5 处理流程

1. 文档入库后，从摘要和 chunk 中抽取实体与关系
2. 做别名归并和实体消歧
3. 写入图结构表
4. QueryAgent 在召回时，先从 query 提取实体
5. 用图结构扩展候选文档或候选 chunk

### 5.2.6 验收标准

- 系统能输出实体关系证据而不只是相似文档
- 图谱增强召回可以单独启停
- 至少支持 `人物/技术/组织/概念` 四类实体

## 5.3 记忆分层

### 5.3.1 目标

把当前单层长期记忆扩展为多层记忆系统。

### 5.3.2 当前问题

当前 `MemoryService` 已支持长期记忆，但仍是统一存储，缺少：

- 短期/长期边界
- 项目级记忆与全局偏好分离
- 可遗忘机制
- 冲突检测
- 重要性衰减

### 5.3.3 目标记忆结构

分为四层：

1. 短期会话记忆
   - 当前 session 的最近轮次
2. 长期用户偏好记忆
   - 语言、回答风格、输出习惯
3. 项目事实记忆
   - 项目约束、技术栈、设计决策
4. 可遗忘记忆
   - 时间敏感、重要性低的信息

### 5.3.4 建议实现

扩展 `memory_records`：

- `scope`
  - `session`
  - `user`
  - `project`
- `ttl_seconds`
- `last_accessed_at`
- `conflict_key`
- `status`
  - `active`
  - `superseded`
  - `expired`

新增机制：

- `MemoryCompressionService`
- `MemoryConflictResolver`
- `MemoryDecayScheduler`

### 5.3.5 检索优先级

建议优先级：

1. session memory
2. project memory
3. user preference memory
4. low-importance memory

### 5.3.6 验收标准

- 记忆召回时能区分作用域
- 冲突记忆不会同时以有效状态参与回答
- 低重要性、长时间未访问的记忆可以衰减

## 6. 第二梯队增强规格

## 6.1 高质量多模态解析

### 6.1.1 目标

让 PDF、图片等复杂内容不仅提取文本，还尽量恢复结构。

### 6.1.2 增强点

- PDF 标题层级恢复
- 表格抽取
- 图片区域 OCR
- 图文块顺序恢复
- 图片内容摘要

### 6.1.3 需要改动的模块

- [app/services/parser_utils.py](/home/rikka/.marscode/personal-knowledge-assistant/app/services/parser_utils.py)
- `ParserAgent`
- `ChunkingAgent`

### 6.1.4 验收标准

- PDF 中标题与正文分离效果优于纯文本串联
- 表格内容不会被全部压扁成乱码

## 6.2 自动评测系统

### 6.2.1 目标

建立一套可重复运行的 RAG 和 Agent 评测基线。

### 6.2.2 评测维度

- 检索命中率
- 引用准确率
- 回答一致性
- 摘要保真度
- 分类准确率
- 记忆命中有效性

### 6.2.3 建议实现

新增目录：

- `evals/`
  - `rag_cases.jsonl`
  - `summary_cases.jsonl`
  - `classification_cases.jsonl`

新增脚本：

- `scripts/eval_rag.py`
- `scripts/eval_summary.py`
- `scripts/eval_classification.py`

新增结果输出：

- `eval_runs`
- `eval_metrics`

### 6.2.4 验收标准

- 可以一条命令跑完整套离线评测
- 可以比较本次变更和上一次基线结果

## 6.3 异步任务系统

### 6.3.1 目标

把同步入库链路升级为任务驱动的异步流水线。

### 6.3.2 当前问题

当前入库流程同步执行，导致：

- 大文件或 OCR 请求阻塞接口
- 无法精细展示任务状态
- 无法重试或取消

### 6.3.3 目标能力

- 提交任务后立即返回 `job_id`
- 后台异步执行解析、embedding、摘要、linking
- 支持失败重试、取消、幂等

### 6.3.4 建议实现

新增表：

- `jobs`
- `job_events`

新增服务：

- `JobScheduler`
- `JobWorker`
- `JobStateStore`

建议优先使用单机队列模式，避免过早引入外部 MQ。

### 6.3.5 验收标准

- 大文件上传后接口能快速返回
- 前端可查询任务状态
- 失败任务可重试

## 7. 第三梯队增强规格

## 7.1 增量索引与实时更新

### 7.1.1 目标

避免文档修改后全量重建索引。

### 7.1.2 需要实现的能力

- 基于 fingerprint 或内容 diff 判断变化范围
- 仅重算受影响的 sections/chunks
- 仅更新受影响的 links、memory、graph edges

### 7.1.3 验收标准

- 单文档变更不会触发全库 rebuild

## 7.2 Agent 自反思

### 7.2.1 目标

让分类、摘要、问答具备首次输出后的自检能力。

### 7.2.2 适用场景

- 摘要超字数
- 分类置信度不合理
- 回答缺少引用
- 记忆抽取不稳定

### 7.2.3 建议实现

新增统一 `SelfCheckService`，按任务类型调用：

- `check_summary()`
- `check_classification()`
- `check_answer()`

### 7.2.4 验收标准

- 自检失败时能够自动重试一次或回退

## 7.3 个性化检索

### 7.3.1 目标

根据用户行为动态调整检索权重。

### 7.3.2 可用信号

- 常问主题
- 常点文档
- 常用标签
- session 历史
- memory 命中偏好

### 7.3.3 建议实现

新增表：

- `user_query_profiles`
- `document_click_events`
- `query_feedback`

在 `RAGService._rerank()` 中加入个性化加权。

### 7.3.4 验收标准

- 相同 query 在不同用户或不同 profile 下可产生不同排序

## 8. 推荐实施顺序

为了控制风险，建议按以下顺序推进：

1. 分层检索 + rerank
2. 自动评测系统
3. 记忆分层
4. 知识图谱增强
5. 异步任务系统
6. 增量索引
7. Agent 自反思
8. 个性化检索

原因：

- 分层检索和评测系统先做，能尽快建立“质量提升”和“质量验证”的闭环
- 记忆分层和图谱增强建立在更稳的 RAG 基础上
- 异步任务和增量索引是系统工程升级，适合在功能稳定后做

## 9. 交付要求

每个增强项在交付时必须带：

- 设计说明
- 数据结构变更
- 配置项
- 最小测试集合
- 回退策略
- 验收结论

## 10. 最终目标

项目完成增强后，应从“个人资料整理工具”升级为“具备持续学习、结构化理解、可评测、可扩展的个人知识操作系统”。
