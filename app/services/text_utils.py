from __future__ import annotations

import hashlib
import html
import math
import re
from collections import Counter


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "一个",
    "一些",
    "我们",
    "你们",
    "他们",
    "这些",
    "那些",
    "可以",
    "以及",
    "如果",
    "因为",
    "所以",
    "然后",
    "就是",
    "适合",
    "可以把",
    "这个",
    "那个",
    "如果结合",
    "它可以把",
    "你可以用",
    "工作流",
    "一般",
    "怎么",
}

AD_PATTERNS = [
    r"广告",
    r"扫码",
    r"关注公众号",
    r"更多精彩",
    r"免责声明",
    r"优惠券",
    r"点击下载",
    r"点击这里",
    r"转载注明",
    r"广告位招租",
]

HTML_BLOCK_RE = re.compile(r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>\n]+>")
MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "Â": "",
}
CP1252_REVERSE = {
    "€": 0x80,
    "‚": 0x82,
    "ƒ": 0x83,
    "„": 0x84,
    "…": 0x85,
    "†": 0x86,
    "‡": 0x87,
    "ˆ": 0x88,
    "‰": 0x89,
    "Š": 0x8A,
    "‹": 0x8B,
    "Œ": 0x8C,
    "Ž": 0x8E,
    "‘": 0x91,
    "’": 0x92,
    "“": 0x93,
    "”": 0x94,
    "•": 0x95,
    "–": 0x96,
    "—": 0x97,
    "˜": 0x98,
    "™": 0x99,
    "š": 0x9A,
    "›": 0x9B,
    "œ": 0x9C,
    "ž": 0x9E,
    "Ÿ": 0x9F,
}
MOJIBAKE_SPAN_RE = re.compile(r"[\u00a0-\u00ff€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ\x80-\x9f]{2,}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_document_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def fix_mojibake(text: str) -> str:
    repaired = text.replace("\u00a0", " ").replace("\ufeff", "")
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    repaired = _repair_utf8_mojibake(repaired)
    return repaired


def strip_html(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = HTML_BLOCK_RE.sub("\n", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return cleaned.replace("\u00a0", " ")


def remove_noise(text: str) -> str:
    cleaned = strip_html(text)
    lines = []
    for line in cleaned.splitlines():
        normalized_line = normalize_whitespace(line)
        if not normalized_line:
            lines.append("")
            continue
        if _is_noise_line(normalized_line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    for pattern in AD_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"(?i)(cookie|subscribe|newsletter|sign up|all rights reserved)", " ", cleaned)
    cleaned = re.sub(r"(上一篇|下一篇|相关阅读|推荐阅读)", " ", cleaned)
    cleaned = _normalize_punctuation(cleaned)
    return normalize_document_text(cleaned)


def _repair_utf8_mojibake(text: str) -> str:
    try:
        candidate = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
    except UnicodeError:
        return text
    if not candidate.strip():
        return text
    if _text_quality(candidate) > _text_quality(text) + 0.08:
        return candidate
    return MOJIBAKE_SPAN_RE.sub(_repair_mojibake_span, text)


def _repair_mojibake_span(match: re.Match[str]) -> str:
    span = match.group(0)
    raw = bytearray()
    for char in span:
        if char in CP1252_REVERSE:
            raw.append(CP1252_REVERSE[char])
        elif ord(char) <= 255:
            raw.append(ord(char))
        else:
            return span
    try:
        candidate = raw.decode("utf-8")
    except UnicodeDecodeError:
        return span
    if _text_quality(candidate) > _text_quality(span) + 0.08:
        return candidate
    return span


def _text_quality(text: str) -> float:
    useful = sum(1 for char in text if "\u4e00" <= char <= "\u9fff" or char.isascii())
    suspicious = sum(1 for char in text if char in {"�", "Â", "Ã", "¢", "€", "œ", "˜", "™"})
    return (useful - suspicious * 3) / max(len(text), 1)


def _is_noise_line(line: str) -> bool:
    lowered = line.lower()
    if re.search(r"(?i)(cookie|subscribe|newsletter|all rights reserved)", lowered):
        return True
    matches = sum(1 for pattern in AD_PATTERNS if re.search(pattern, line, flags=re.IGNORECASE))
    if matches >= 1 and len(line) <= 80:
        return True
    if re.fullmatch(r"(上一篇|下一篇|相关阅读|推荐阅读|赞|收藏|分享)[\s\S]{0,20}", line):
        return True
    return False


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r";{2,}", ";", text)
    return text


def text_stats(text: str) -> dict[str, int]:
    tokens = tokenize(text)
    return {
        "chars": len(text),
        "tokens": len(tokens),
        "lines": len([line for line in text.splitlines() if line.strip()]),
    }


def tokenize(text: str) -> list[str]:
    items = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,8}", text.lower())
    normalized = []
    for item in items:
        token = item.strip()
        if not token or token in STOPWORDS or len(token) < 2:
            continue
        normalized.append(token)
        if len(token) >= 4:
            for size in (2, 3, 4):
                for index in range(0, len(token) - size + 1):
                    piece = token[index : index + size]
                    if piece not in STOPWORDS:
                        normalized.append(piece)
    return normalized


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    counts = Counter(tokenize(text))
    if not counts:
        return []
    keywords = []
    for token, _ in counts.most_common(limit * 3):
        if re.fullmatch(r"[A-Za-z0-9_\-]+", token):
            keywords.append(token)
        elif 2 <= len(token) <= 6 and "可以" not in token and "适合" not in token:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def overlap_score(left_text: str, right_text: str) -> float:
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]


def summarize_text(text: str, min_chars: int = 100, max_chars: int = 200) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    keywords = set(extract_keywords(text, limit=8))
    ranked = []
    for sentence in sentences:
        score = sum(1 for token in tokenize(sentence) if token in keywords)
        ranked.append((score, sentence))
    ranked.sort(key=lambda item: (-item[0], len(item[1])))

    result = []
    total = 0
    for _, sentence in ranked:
        if sentence in result:
            continue
        result.append(sentence)
        total += len(sentence)
        if total >= min_chars:
            break

    summary = " ".join(result)
    summary = normalize_whitespace(summary)
    summary = re.sub(r"(\b\w+\b)( \1\b)+", r"\1", summary, flags=re.IGNORECASE)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def classify_text(text: str) -> tuple[str, float]:
    category_keywords = {
        "技术": ["python", "go", "agent", "rag", "数据库", "后端", "前端", "编程", "算法", "架构", "代码"],
        "学习": ["学习", "课程", "笔记", "方法", "复盘", "考试", "读书", "知识", "训练"],
        "生活": ["生活", "健康", "旅行", "饮食", "运动", "家庭", "情绪", "效率"],
    }
    lowered = text.lower()
    scores: dict[str, int] = {}
    for category, words in category_keywords.items():
        scores[category] = sum(1 for word in words if word.lower() in lowered)

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    confidence = 0.45 if best_score == 0 else min(0.95, 0.55 + best_score * 0.1)
    return best_category, round(confidence, 2)


def make_hash_embedding(text: str, dims: int = 128) -> list[float]:
    vector = [0.0] * dims
    counts = Counter(tokenize(text))
    if not counts:
        return vector

    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
