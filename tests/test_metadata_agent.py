from __future__ import annotations

from app.config import Settings
from app.models import IngestRequest, PipelineState
from app.pipeline.agents.metadata_agent import MetadataAgent
from app.services.openai_client import OpenAIService
from app.services.self_check_service import SelfCheckService


class FakeOpenAIService(OpenAIService):
    def __init__(self, result=None, error: Exception | None = None) -> None:
        super().__init__(Settings(openai_api_key="test-key"))
        self.result = result
        self.error = error
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def generate_json(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def build_state() -> PipelineState:
    text = (
        "技术：LangGraph。"
        "LangGraph 用于构建多 Agent 编排流程，可以把采集、解析、清洗、切片、分类和摘要拆成节点。"
        "系统需要保留分类、标签和摘要，便于后续检索和关联。"
    )
    return PipelineState(
        request=IngestRequest(source_type="text", source=text, title="LangGraph 元数据"),
        cleaned_text=text,
        title="LangGraph 元数据",
    )


def test_metadata_agent_uses_single_model_call_for_category_tags_and_summary():
    openai = FakeOpenAIService(
        result={
            "category": "技术",
            "confidence": 0.93,
            "tags": ["LangGraph", "Agent编排", "元数据"],
            "summary": "本文围绕 LangGraph 元数据生成展开，说明系统会把采集、解析、清洗、切片、分类和摘要拆成节点，并保留分类、标签和摘要用于后续检索与关联。",
        }
    )
    agent = MetadataAgent(openai, SelfCheckService(Settings(openai_api_key="")))

    state = agent.run(build_state())

    assert openai.calls == 1
    assert state.category == "技术"
    assert state.confidence == 0.93
    assert state.tags == ["LangGraph", "Agent编排", "元数据"]
    assert "LangGraph 元数据生成" in state.summary
    assert any("metadata: 已完成分类、标签和摘要生成。 使用gpt-5.4。" in log for log in state.logs)


def test_metadata_agent_falls_back_when_model_fails():
    openai = FakeOpenAIService(error=RuntimeError("provider timeout"))
    agent = MetadataAgent(openai, SelfCheckService(Settings(openai_api_key="")))

    state = agent.run(build_state())

    assert openai.calls == 1
    assert state.category
    assert state.tags
    assert state.summary
    assert any("模型生成失败" in log and "provider timeout" in log for log in state.logs)
    assert any("使用本地规则回退" in log for log in state.logs)


def test_metadata_agent_local_mode_skips_model_call():
    openai = FakeOpenAIService(error=RuntimeError("should not call provider"))
    agent = MetadataAgent(openai, SelfCheckService(Settings(openai_api_key="")))

    state = agent.run(build_state(), use_model=False)

    assert openai.calls == 0
    assert state.category
    assert state.tags
    assert state.summary
    assert any("使用本地规则回退" in log for log in state.logs)


def test_metadata_agent_local_tags_filter_fragments_and_use_title():
    text = (
        "技术：LangGraph。"
        "LangGraph 用于构建 Agent 工作流。"
        "短碎片 la an ng 不应成为标签。"
        "RAG 检索和 Chroma 向量索引用于提升知识库问答质量。"
    )
    state = PipelineState(
        request=IngestRequest(source_type="text", source=text, title="LangGraph RAG 知识库"),
        cleaned_text=text,
        title="LangGraph RAG 知识库",
    )
    agent = MetadataAgent(FakeOpenAIService(error=RuntimeError("should not call provider")))

    result = agent.run(state, use_model=False)

    assert "langgraph" in [tag.lower() for tag in result.tags]
    assert all(tag.lower() not in {"la", "an", "ng"} for tag in result.tags)
    assert all(len(tag) >= 2 for tag in result.tags)
    assert result.summary
