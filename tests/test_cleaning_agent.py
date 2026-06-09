from __future__ import annotations

from app.models import IngestRequest, PipelineState
from app.pipeline.agents.cleaning_agent import CleaningAgent


def test_cleaning_agent_removes_html_ads_and_repairs_text():
    raw = """
    <script>alert("x")</script>
    <style>.ad{display:none}</style>
    #   LangGraph   使用笔记

    免责声明：本文仅代表作者观点。点击这里领取优惠券！！！

    技术：  LangGraph
    组织：OpenAI

    这  是   一段   很   乱 的 文本，，，，里面 有     很多    空格。

    <div>这里混入 HTML 标签 <b>重点</b> 和 &nbsp; 实体。</div>

    更多精彩内容请关注公众号｜扫码关注｜广告位招租

    ç¨‹åºå‘˜ 常见乱码 mixed with English    words.
    """
    state = PipelineState(
        request=IngestRequest(source_type="text", source=raw, title="清洗质量测试"),
        parsed_text=raw,
        title="清洗质量测试",
    )

    result = CleaningAgent().run(state)
    cleaned = result.cleaned_text

    assert "<script" not in cleaned
    assert "<style" not in cleaned
    assert "<div" not in cleaned
    assert "&nbsp;" not in cleaned
    assert "免责声明" not in cleaned
    assert "关注公众号" not in cleaned
    assert "广告位招租" not in cleaned
    assert "，，" not in cleaned
    assert "程序员" in cleaned
    assert "LangGraph 使用笔记" in cleaned
    assert result.metadata["cleaning"]["removed_chars"] > 0
