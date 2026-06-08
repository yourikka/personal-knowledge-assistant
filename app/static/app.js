const state = {
  documents: [],
  selectedDocumentId: null,
  selectedDocument: null,
  contentView: "cleaned",
  queryRunId: 0,
  queryAbortController: null,
};

const els = {
  healthPill: document.getElementById("health-pill"),
  docCountPill: document.getElementById("doc-count-pill"),
  recentDocsCount: document.getElementById("recent-docs-count"),
  runtimeMode: document.getElementById("runtime-mode"),
  lastDocumentId: document.getElementById("last-document-id"),
  chunkCount: document.getElementById("chunk-count"),
  pipelineSteps: document.getElementById("pipeline-steps"),
  logConsole: document.getElementById("log-console"),
  documentList: document.getElementById("document-list"),
  documentDetail: document.getElementById("document-detail"),
  contentInsight: document.getElementById("content-insight"),
  contentSearchInput: document.getElementById("content-search-input"),
  readerOverlay: document.getElementById("reader-overlay"),
  readerTitle: document.getElementById("reader-title"),
  readerMeta: document.getElementById("reader-meta"),
  readerBody: document.getElementById("reader-body"),
  closeReader: document.getElementById("close-reader"),
  chunksList: document.getElementById("chunks-list"),
  relatedList: document.getElementById("related-list"),
  ingestForm: document.getElementById("ingest-form"),
  ingestResult: document.getElementById("ingest-result"),
  sourceType: document.getElementById("source-type"),
  sourceText: document.getElementById("source-text"),
  sourceFile: document.getElementById("source-file"),
  fileInputWrap: document.getElementById("file-input-wrap"),
  queryForm: document.getElementById("query-form"),
  queryInput: document.getElementById("query-input"),
  answerContent: document.getElementById("answer-content"),
  memoriesList: document.getElementById("memories-list"),
  referencesList: document.getElementById("references-list"),
  imageForm: document.getElementById("image-form"),
  imageStage: document.getElementById("image-stage"),
};

const pipelineStepMap = {
  acquisition: "acquisition",
  parser: "parser",
  cleaning: "cleaning",
  chunking: "chunking",
  classification: "classification",
  summary: "summary",
  linking: "linking",
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "string" ? payload : payload.detail || JSON.stringify(payload);
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function streamApi(url, payload, handlers = {}, options = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: options.signal,
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const block of events) {
      dispatchStreamEvent(block, handlers);
    }
  }

  if (buffer.trim()) {
    dispatchStreamEvent(buffer, handlers);
  }
}

function dispatchStreamEvent(block, handlers) {
  const lines = block.split("\n");
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
  const data = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (!data) {
    return;
  }

  let payload;
  try {
    payload = JSON.parse(data);
  } catch {
    payload = data;
  }
  if (event === "error") {
    throw new Error(payload.error || "流式检索失败。");
  }
  handlers[event]?.(payload);
}

function appendLogs(prefix, logs) {
  const lines = Array.isArray(logs) ? logs : [String(logs)];
  lines.forEach((line) => {
    const item = document.createElement("div");
    item.className = "log-line";
    item.textContent = `${prefix}: ${line}`;
    els.logConsole.prepend(item);
  });
}

function resetPipeline() {
  els.pipelineSteps.querySelectorAll("li").forEach((item) => {
    item.classList.remove("done", "error");
  });
}

function markPipelineFromLogs(logs = []) {
  resetPipeline();
  logs.forEach((line) => {
    const key = String(line).split(":")[0];
    const step = pipelineStepMap[key];
    if (!step) {
      return;
    }
    els.pipelineSteps.querySelector(`[data-step="${step}"]`)?.classList.add("done");
  });
}

function markPipelineError() {
  els.pipelineSteps.querySelectorAll("li:not(.done)").forEach((item, index) => {
    if (index === 0) {
      item.classList.add("error");
    }
  });
}

function renderDocumentList() {
  if (!state.documents.length) {
    els.documentList.innerHTML = '<div class="muted-row">还没有文档。</div>';
    return;
  }

  els.documentList.innerHTML = state.documents
    .map((doc) => `
      <div class="document-row ${doc.id === state.selectedDocumentId ? "active" : ""}">
        <button class="document-item ${doc.id === state.selectedDocumentId ? "active" : ""}" type="button" data-id="${doc.id}">
          <span class="document-title">${escapeHtml(doc.title)}</span>
          <span class="document-meta">${escapeHtml(doc.category)} · ${escapeHtml((doc.tags || []).join(", "))}</span>
        </button>
        <button
          class="document-delete"
          type="button"
          data-delete-id="${doc.id}"
          aria-label="删除 ${escapeHtml(doc.title)}"
          title="删除文档"
        >
          删除
        </button>
      </div>
    `)
    .join("");

  els.documentList.querySelectorAll("[data-id]").forEach((node) => {
    node.addEventListener("click", () => loadDocument(node.dataset.id));
  });
  els.documentList.querySelectorAll("[data-delete-id]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      void handleDeleteDocument(node.dataset.deleteId);
    });
  });
}

function updateDocumentCounters() {
  const count = state.documents.length;
  els.docCountPill.textContent = `docs: ${count}`;
  els.recentDocsCount.textContent = String(count);
}

function clearDocumentWorkspace(message = "选择或入库文档后显示摘要、标签和来源。") {
  state.selectedDocument = null;
  els.lastDocumentId.textContent = "等待文档";
  els.documentDetail.innerHTML = `<div class="placeholder">${escapeHtml(message)}</div>`;
  els.contentInsight.innerHTML = '<div class="placeholder">这里会按统一格式展示已入库文档，区块内部可独立滚动，点击任意文档进入阅读页。</div>';
  els.contentSearchInput.value = "";
  closeReader();
  els.relatedList.innerHTML = '<div class="placeholder">入库后显示相似内容和双向链接。</div>';
  els.chunksList.innerHTML = '<div class="placeholder">文档入库后，这里显示真实 chunk 文本、标题路径和字符范围。</div>';
  els.chunkCount.textContent = "0 chunks";
}

function renderDocumentDetail(document) {
  if (!document) {
    els.documentDetail.innerHTML = '<div class="placeholder">选择或入库文档后显示摘要、标签和来源。</div>';
    els.relatedList.innerHTML = '<div class="placeholder">入库后显示相似内容和双向链接。</div>';
    return;
  }

  const tags = (document.tags || []).map((tag) => `<span class="detail-chip">${escapeHtml(tag)}</span>`).join("");
  els.documentDetail.innerHTML = `
    <div class="detail-title">${escapeHtml(document.title)}</div>
    <div class="detail-summary">${escapeHtml(document.summary || "暂无摘要")}</div>
    <div class="detail-chips">
      <span class="detail-chip">${escapeHtml(document.category)}</span>
      <span class="detail-chip">confidence ${Number(document.confidence || 0).toFixed(2)}</span>
      <span class="detail-chip">${escapeHtml(document.source_type)}</span>
    </div>
    <div class="detail-tags">${tags || '<span class="placeholder">暂无标签</span>'}</div>
    <div class="badge-row">
      <span class="badge">${escapeHtml(document.source_uri)}</span>
    </div>
  `;

  renderRelated(document.related || []);
  renderContentPanel(document);
}

function renderContentPanel(document) {
  renderContentCards();
  if (!els.readerOverlay.classList.contains("hidden")) {
    renderReader();
  }
}

function renderContentCards() {
  const keyword = els.contentSearchInput.value.trim().toLowerCase();
  const documents = state.documents.filter((item) => {
    if (!keyword) {
      return true;
    }
    const haystack = [item.title, item.summary, item.category, ...(item.tags || [])].join(" ").toLowerCase();
    return haystack.includes(keyword);
  });

  if (!documents.length) {
    els.contentInsight.innerHTML = '<div class="placeholder">没有匹配的文档。</div>';
    return;
  }

  els.contentInsight.innerHTML = documents
    .map((item) => {
      const selected = item.id === state.selectedDocumentId;
      const tags = (item.tags || []).map((tag) => `<span class="detail-chip">${escapeHtml(tag)}</span>`).join("");
      return `
        <article
          class="content-card ${selected ? "active" : ""}"
          tabindex="0"
          role="button"
          aria-label="打开 ${escapeHtml(item.title)} 阅读页"
          data-content-id="${item.id}"
        >
          <div class="content-summary">
            <div>
              <div class="detail-title">${escapeHtml(item.title)}</div>
              <div class="detail-summary">${escapeHtml(item.summary || "暂无摘要")}</div>
            </div>
            <div class="content-facts">
              <span class="detail-chip">${escapeHtml(item.category)}</span>
              <span class="detail-chip">confidence ${Number(item.confidence || 0).toFixed(2)}</span>
              <span class="detail-chip">${escapeHtml(item.source_type)}</span>
              <span class="detail-chip">${Number((item.related || []).length)} related</span>
            </div>
          </div>
          <div class="detail-tags">${tags || '<span class="placeholder">暂无标签</span>'}</div>
          <div class="content-source">${escapeHtml(item.source_uri)}</div>
        </article>
      `;
    })
    .join("");

  els.contentInsight.querySelectorAll("[data-content-id]").forEach((node) => {
    node.addEventListener("click", () => void openReaderFromCard(node.dataset.contentId));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void openReaderFromCard(node.dataset.contentId);
      }
    });
  });
}

function setContentView(view) {
  state.contentView = view;
  document.querySelectorAll("[data-content-view]").forEach((button) => {
    const active = button.dataset.contentView === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  if (!els.readerOverlay.classList.contains("hidden")) {
    renderReader();
  }
}

function renderSearchableText(target, text, query) {
  const needle = query.trim();
  if (!needle) {
    target.textContent = text;
    return;
  }

  const lowerText = text.toLowerCase();
  const lowerNeedle = needle.toLowerCase();
  const parts = [];
  let cursor = 0;
  let index = lowerText.indexOf(lowerNeedle);

  while (index !== -1) {
    parts.push(escapeHtml(text.slice(cursor, index)));
    parts.push(`<mark>${escapeHtml(text.slice(index, index + needle.length))}</mark>`);
    cursor = index + needle.length;
    index = lowerText.indexOf(lowerNeedle, cursor);
  }

  parts.push(escapeHtml(text.slice(cursor)));
  target.innerHTML = parts.join("");
  target.querySelector("mark")?.scrollIntoView({ block: "center" });
}

function getCurrentDocumentText() {
  if (!state.selectedDocument) {
    return "暂无内容。";
  }
  const text = state.contentView === "raw" ? state.selectedDocument.raw_text : state.selectedDocument.cleaned_text;
  return text?.trim() || "暂无内容。";
}

function openReader() {
  if (!state.selectedDocument) {
    window.alert("请先选择一篇文档。");
    return;
  }
  renderReader();
  els.readerOverlay.classList.remove("hidden");
  els.readerOverlay.setAttribute("aria-hidden", "false");
  els.readerBody.focus();
}

async function openReaderFromCard(documentId) {
  if (!documentId) {
    return;
  }
  if (state.selectedDocumentId !== documentId || !state.selectedDocument) {
    await loadDocument(documentId);
  }
  openReader();
}

function closeReader() {
  els.readerOverlay.classList.add("hidden");
  els.readerOverlay.setAttribute("aria-hidden", "true");
}

function renderReader() {
  const document = state.selectedDocument;
  if (!document) {
    els.readerTitle.textContent = "文档阅读";
    els.readerMeta.textContent = "未选择文档";
    els.readerBody.textContent = "暂无内容。";
    return;
  }

  els.readerTitle.textContent = document.title || "文档阅读";
  els.readerMeta.textContent = `${document.category} · ${document.source_type} · ${document.source_uri}`;
  renderSearchableText(els.readerBody, getCurrentDocumentText(), els.contentSearchInput.value);
}

function renderRelated(related) {
  if (!related.length) {
    els.relatedList.innerHTML = '<div class="placeholder">暂无关联内容。</div>';
    return;
  }

  els.relatedList.innerHTML = related
    .map((item) => `
      <button class="related-item" type="button" data-document-id="${escapeHtml(item.target_id)}">
        <strong>${escapeHtml(item.title || item.target_id)}</strong>
        <span class="related-meta">score ${Number(item.score || 0).toFixed(4)}</span>
      </button>
    `)
    .join("");

  els.relatedList.querySelectorAll("[data-document-id]").forEach((node) => {
    node.addEventListener("click", () => loadDocument(node.dataset.documentId));
  });
}

function renderIngestResult(result) {
  els.lastDocumentId.textContent = result.document_id ? `doc ${result.document_id.slice(0, 8)}` : "等待文档";
  els.ingestResult.innerHTML = `
    <div class="summary-title">${escapeHtml(result.title)}</div>
    <div class="detail-summary">${escapeHtml(result.summary || "暂无摘要")}</div>
    <div class="badge-row">
      <span class="badge">${escapeHtml(result.category)}</span>
      <span class="badge">${result.duplicate ? "duplicate" : "new"}</span>
      <span class="badge">${(result.tags || []).map(escapeHtml).join(", ") || "untagged"}</span>
    </div>
  `;
}

async function loadChunks(documentId) {
  try {
    const chunks = await api(`/api/knowledge/documents/${documentId}/chunks`);
    renderChunks(chunks);
  } catch (error) {
    els.chunksList.innerHTML = `<div class="placeholder">切片加载失败：${escapeHtml(error.message)}</div>`;
    els.chunkCount.textContent = "0 chunks";
  }
}

function renderChunks(chunks) {
  els.chunkCount.textContent = `${chunks.length} chunks`;
  if (!chunks.length) {
    els.chunksList.innerHTML = '<div class="placeholder">暂无切片。</div>';
    return;
  }

  els.chunksList.innerHTML = chunks
    .map((chunk) => {
      const headingPath = chunk.metadata?.heading_path?.join(" > ") || "无标题路径";
      return `
        <article class="chunk-item">
          <div class="chunk-title">chunk #${chunk.chunk_index}</div>
          <div class="chunk-meta">${escapeHtml(headingPath)} · chars ${chunk.char_start}-${chunk.char_end}</div>
          <div class="chunk-text">${escapeHtml(chunk.text)}</div>
        </article>
      `;
    })
    .join("");
}

function renderReferences(references) {
  if (!references.length) {
    els.referencesList.innerHTML = '<div class="placeholder">暂无引用。</div>';
    return;
  }

  els.referencesList.innerHTML = references
    .map((ref) => `
      <button class="reference-item" type="button" data-document-id="${escapeHtml(ref.id)}">
        <strong>${escapeHtml(ref.title)}</strong>
        <span class="reference-meta">
          chunk #${ref.chunk_index ?? "-"} · ${escapeHtml((ref.heading_path || []).join(" > ") || "无标题路径")}
        </span>
        <span class="reference-meta">${escapeHtml(ref.source_uri)}</span>
      </button>
    `)
    .join("");

  els.referencesList.querySelectorAll("[data-document-id]").forEach((node) => {
    node.addEventListener("click", () => loadDocument(node.dataset.documentId));
  });
}

function renderMemories(memories) {
  if (!memories.length) {
    els.memoriesList.innerHTML = '<div class="placeholder">暂无相关记忆。</div>';
    return;
  }

  els.memoriesList.innerHTML = memories
    .map((memory) => `
      <article class="reference-item memory-item">
        <strong>${escapeHtml(memory.kind)} · score ${Number(memory.score || 0).toFixed(4)}</strong>
        <span class="reference-meta">${escapeHtml(memory.content || "")}</span>
        <span class="reference-meta">${escapeHtml((memory.tags || []).join(", ") || "untagged")}</span>
      </article>
    `)
    .join("");
}

function renderImageResult(result) {
  const imageSrc = result.image_b64 ? `data:image/png;base64,${result.image_b64}` : result.image_url;
  if (!imageSrc) {
    els.imageStage.innerHTML = '<div class="placeholder">接口返回了空图片。</div>';
    return;
  }

  els.imageStage.innerHTML = `
    <div class="image-preview">
      <img src="${imageSrc}" alt="生成的知识库主题图片">
      <div class="image-meta">${escapeHtml(result.revised_prompt || result.prompt)}</div>
    </div>
  `;
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    els.healthPill.textContent = `system: ${health.status}`;
    const indexItems = health.vector_store?.local_items ?? 0;
    const docs = health.repository?.documents ?? 0;
    els.runtimeMode.textContent = `${health.chroma_enabled ? "FastAPI + Chroma" : "FastAPI + Local Vector"} · ${docs} docs · ${indexItems} indexed`;
  } catch (error) {
    els.healthPill.textContent = "system: offline";
    appendLogs("health", error.message);
  }
}

async function refreshDocuments() {
  try {
    state.documents = await api("/api/knowledge/documents?limit=12");
    updateDocumentCounters();
    renderDocumentList();
    if (!state.documents.length) {
      state.selectedDocumentId = null;
      clearDocumentWorkspace("当前没有可查看的文档。");
      return;
    }

    const selectedStillExists = state.documents.some((item) => item.id === state.selectedDocumentId);
    if (!state.selectedDocumentId || !selectedStillExists) {
      await loadDocument(state.documents[0].id);
    }
  } catch (error) {
    appendLogs("documents", error.message);
    els.documentList.innerHTML = '<div class="muted-row">文档加载失败。</div>';
  }
}

async function loadDocument(documentId) {
  try {
    const document = await api(`/api/knowledge/documents/${documentId}`);
    state.selectedDocumentId = documentId;
    state.selectedDocument = document;
    els.lastDocumentId.textContent = `doc ${documentId.slice(0, 8)}`;
    renderDocumentList();
    renderDocumentDetail(document);
    await loadChunks(documentId);
  } catch (error) {
    appendLogs("document", error.message);
  }
}

async function handleDeleteDocument(documentId) {
  const document = state.documents.find((item) => item.id === documentId);
  const title = document?.title || documentId;
  const confirmed = window.confirm(`确定删除文档「${title}」吗？这会同时删除它的切片和关联索引。`);
  if (!confirmed) {
    return;
  }

  try {
    let result;
    try {
      result = await api(`/api/knowledge/documents/${documentId}`, { method: "DELETE" });
    } catch (error) {
      if (error.status !== 405) {
        throw error;
      }
      result = await api(`/api/knowledge/documents/${documentId}/delete`, { method: "POST" });
    }
    appendLogs("delete", [`已删除 ${documentId}`, `chunks ${result.deleted_chunk_ids.length}`]);

    const wasSelected = state.selectedDocumentId === documentId;
    state.documents = state.documents.filter((item) => item.id !== documentId);
    if (wasSelected) {
      state.selectedDocumentId = null;
    }

    renderDocumentList();
    updateDocumentCounters();

    if (wasSelected) {
      if (state.documents.length) {
        await loadDocument(state.documents[0].id);
      } else {
        clearDocumentWorkspace("当前没有可查看的文档。");
      }
    }
  } catch (error) {
    appendLogs("delete", error.message);
    window.alert(`删除失败：${error.message}`);
  }
}

async function handleIngest(event) {
  event.preventDefault();
  resetPipeline();
  const formData = new FormData(els.ingestForm);
  const sourceType = formData.get("source_type");
  const title = formData.get("title") || "";
  try {
    let result;
    if (sourceType === "pdf" || sourceType === "image") {
      const file = els.sourceFile.files[0];
      if (!file) {
        throw new Error("请先选择文件。");
      }
      const uploadData = new FormData();
      uploadData.append("file", file);
      uploadData.append("source_type", sourceType);
      uploadData.append("title", title);
      result = await api("/api/knowledge/upload", { method: "POST", body: uploadData });
    } else {
      result = await api("/api/knowledge/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: sourceType,
          source: formData.get("source"),
          title: title || null,
          metadata: {},
        }),
      });
    }

    renderIngestResult(result);
    markPipelineFromLogs(result.logs || []);
    appendLogs("ingest", result.logs || []);
    await refreshDocuments();
    if (result.document_id) {
      await loadDocument(result.document_id);
    }
    document.getElementById("pipeline-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    markPipelineError();
    appendLogs("ingest", error.message);
    els.ingestResult.innerHTML = `<div class="placeholder">入库失败：${escapeHtml(error.message)}</div>`;
  }
}

async function handleQuery(event) {
  event.preventDefault();
  const formData = new FormData(els.queryForm);
  const query = String(formData.get("query") || "").trim();
  if (!query) {
    els.answerContent.textContent = "请输入问题。";
    return;
  }
  const request = {
    query,
    top_k: Number(formData.get("top_k") || 3),
    session_id: formData.get("session_id") || null,
  };
  state.queryRunId += 1;
  const runId = state.queryRunId;
  state.queryAbortController?.abort();
  const controller = new AbortController();
  state.queryAbortController = controller;
  const submitButton = els.queryForm.querySelector('button[type="submit"]');
  const previousButtonText = submitButton?.textContent || "提问";
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "检索中";
  }
  const isCurrentRun = () => state.queryRunId === runId;

  try {
    els.answerContent.textContent = "正在检索...";
    renderMemories([]);
    renderReferences([]);
    let receivedDelta = false;
    await streamApi("/api/knowledge/query/stream", request, {
      delta: (text) => {
        if (!isCurrentRun()) {
          return;
        }
        if (!receivedDelta) {
          els.answerContent.textContent = "";
          receivedDelta = true;
        }
        els.answerContent.textContent += text;
      },
      references: (references) => {
        if (isCurrentRun()) {
          renderReferences(references || []);
        }
      },
      memories: (memories) => {
        if (isCurrentRun()) {
          renderMemories(memories || []);
        }
      },
      status: (message) => {
        if (isCurrentRun() && !receivedDelta) {
          els.answerContent.textContent = message || "正在检索...";
        }
      },
      logs: (logs) => {
        if (isCurrentRun()) {
          appendLogs("query", logs || []);
        }
      },
    }, { signal: controller.signal });
    if (!isCurrentRun()) {
      return;
    }
    if (!els.answerContent.textContent.trim()) {
      els.answerContent.textContent = "暂无答案。";
    }
  } catch (error) {
    if (error.name === "AbortError" || !isCurrentRun()) {
      return;
    }
    try {
      const result = await api("/api/knowledge/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      els.answerContent.textContent = result.answer || "暂无答案。";
      renderMemories(result.memories || []);
      renderReferences(result.references || []);
      appendLogs("query", ["流式检索失败，已切换普通检索。", ...(result.logs || [])]);
    } catch (fallbackError) {
      appendLogs("query", [error.message, fallbackError.message]);
      els.answerContent.textContent = `检索失败：${fallbackError.message}`;
      renderMemories([]);
      renderReferences([]);
    }
  } finally {
    if (isCurrentRun()) {
      state.queryAbortController = null;
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = previousButtonText;
      }
    }
  }
}

async function handleImage(event) {
  event.preventDefault();
  const formData = new FormData(els.imageForm);
  try {
    const result = await api("/api/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: formData.get("prompt"),
        size: formData.get("size"),
        quality: formData.get("quality"),
      }),
    });
    renderImageResult(result);
    appendLogs("image", result.logs || []);
  } catch (error) {
    appendLogs("image", error.message);
    els.imageStage.innerHTML = `<div class="placeholder">生图失败：${escapeHtml(error.message)}</div>`;
  }
}

function updateSourceMode() {
  const fileMode = els.sourceType.value === "pdf" || els.sourceType.value === "image";
  els.fileInputWrap.classList.toggle("hidden", !fileMode);
  els.sourceText.closest("label").classList.toggle("hidden", fileMode);
}

function bindQuickActions() {
  document.querySelectorAll("[data-scroll-target]").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById(button.dataset.scrollTarget)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  document.querySelectorAll("[data-content-view]").forEach((button) => {
    button.addEventListener("click", () => setContentView(button.dataset.contentView));
  });

  els.contentSearchInput.addEventListener("input", () => {
    renderContentCards();
    if (!els.readerOverlay.classList.contains("hidden")) {
      renderReader();
    }
  });
  els.closeReader.addEventListener("click", closeReader);
  els.readerOverlay.addEventListener("click", (event) => {
    if (event.target === els.readerOverlay) {
      closeReader();
    }
  });

  document.getElementById("refresh-documents").addEventListener("click", refreshDocuments);
  document.getElementById("use-sample-ingest").addEventListener("click", () => {
    els.sourceType.value = "markdown";
    updateSourceMode();
    els.ingestForm.title.value = "Chunk RAG 流水线示例";
    els.ingestForm.source.value = [
      "# Chunk RAG 流水线示例",
      "",
      "个人知识库的核心使用逻辑是把内容先入库，再经过采集、解析、清洗、切片、分类、摘要和关联。",
      "",
      "## 切片策略",
      "",
      "系统会优先保留标题、段落和句子边界，并为长文本生成带 overlap 的稳定 chunk，后续检索会直接召回 chunk。",
      "",
      "## 问答阶段",
      "",
      "问答不是直接读全文，而是通过向量和关键词召回相关 chunk，再生成带引用的答案。",
    ].join("\n");
  });
}

function bindGlobalHotkeys() {
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      els.queryInput.focus();
      document.getElementById("query-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (event.key === "Escape" && !els.readerOverlay.classList.contains("hidden")) {
      closeReader();
    }
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function boot() {
  els.ingestForm.addEventListener("submit", handleIngest);
  els.queryForm.addEventListener("submit", handleQuery);
  els.imageForm.addEventListener("submit", handleImage);
  els.sourceType.addEventListener("change", updateSourceMode);
  updateSourceMode();
  resetPipeline();
  bindQuickActions();
  bindGlobalHotkeys();
  await refreshHealth();
  await refreshDocuments();
}

boot();
