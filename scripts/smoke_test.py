from __future__ import annotations

import os
import sys
import uuid

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.main import pipeline
from app.models import IngestRequest


def main() -> None:
    unique = uuid.uuid4().hex[:8]
    request = IngestRequest(
        source_type="markdown",
        source=(
            f"# LangGraph Smoke {unique}\n\n"
            "这是一段用于回归测试的内容，覆盖采集、解析、清洗、分类、摘要、持久化和关联。"
            "\n\n## RAG 切块\n\n"
            "系统会在清洗后按标题、段落和句子边界切成稳定 chunk，并把 chunk 写入 SQLite 和向量索引。"
        ),
        title=f"LangGraph Smoke {unique}",
    )
    first = pipeline.ingest(request)
    assert first["duplicate"] is False, first
    assert "LangGraph" in " ".join(first["logs"]), first["logs"]
    assert first["document_id"], first
    assert first["summary"], first
    assert isinstance(first["graph"], dict), first
    assert any(log.startswith("chunking:") for log in first["logs"]), first["logs"]

    duplicate = pipeline.ingest(request)
    assert duplicate["duplicate"] is True, duplicate

    answer = pipeline.query("LangGraph 编排测试覆盖了什么？", 3, f"smoke-{unique}")
    assert answer["answer"], answer
    assert isinstance(answer["references"], list), answer
    assert any(log.startswith("rag:") for log in answer["logs"]), answer["logs"]
    if answer["references"]:
        assert "signals" in answer["references"][0], answer["references"][0]
        assert "chunk_id" in answer["references"][0], answer["references"][0]
        assert "chunk_index" in answer["references"][0], answer["references"][0]

    image = pipeline.generate_image("个人知识库 LangGraph smoke test cover", "1024x1024", "high")
    assert "logs" in image, image

    print("smoke-ok")
    print({"document_id": first["document_id"], "references": len(answer["references"]), "image_has_b64": bool(image["image_b64"])})


if __name__ == "__main__":
    main()
