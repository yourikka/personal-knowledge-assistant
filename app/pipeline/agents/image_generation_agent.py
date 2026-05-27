from __future__ import annotations

import os
import uuid

from app.config import Settings
from app.services.openai_client import OpenAIService


class ImageGenerationAgent:
    def __init__(self, settings: Settings, openai_service: OpenAIService) -> None:
        self.settings = settings
        self.openai_service = openai_service

    def run(self, prompt: str, size: str, quality: str) -> dict:
        logs = ["image-generation: 已开始执行生图任务。"]
        if not self.openai_service.enabled():
            logs.append("image-generation: 未配置 OPENAI_API_KEY，无法调用 gpt-image-2。")
            return {
                "prompt": prompt,
                "revised_prompt": prompt,
                "model": self.settings.openai_image_model,
                "image_b64": None,
                "image_url": None,
                "logs": logs,
            }

        final_prompt = self._rewrite_prompt(prompt, logs)
        try:
            image_result = self.openai_service.generate_image(prompt=final_prompt, size=size, quality=quality)
        except Exception as error:
            logs.append(f"image-generation: gpt-image-2 调用失败，原因：{error}")
            return {
                "prompt": prompt,
                "revised_prompt": final_prompt,
                "model": self.settings.openai_image_model,
                "image_b64": None,
                "image_url": None,
                "logs": logs,
            }
        if not image_result:
            logs.append("image-generation: gpt-image-2 未返回图片结果。")
            return {
                "prompt": prompt,
                "revised_prompt": final_prompt,
                "model": self.settings.openai_image_model,
                "image_b64": None,
                "image_url": None,
                "logs": logs,
            }

        if image_result.get("image_b64"):
            output_dir = os.path.join("data", "generated-images")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.png")
            self.openai_service.save_base64_image(image_result["image_b64"], output_path)
            logs.append(f"image-generation: 已保存图片到 {output_path}。")

        logs.append("image-generation: 已完成 gpt-image-2 生图。")
        return {
            "prompt": prompt,
            "revised_prompt": image_result.get("revised_prompt", final_prompt),
            "model": image_result.get("model", self.settings.openai_image_model),
            "image_b64": image_result.get("image_b64"),
            "image_url": image_result.get("image_url"),
            "logs": logs,
        }

    def _rewrite_prompt(self, prompt: str, logs: list[str]) -> str:
        result = self.openai_service.generate_json(
            system_prompt=(
                "你是生图提示词优化助手，只能输出 JSON。"
                "返回字段固定为 image_prompt。"
                "你的任务是把用户的中文需求改写成适合图像模型生成的高质量提示词。"
                "要保留用户原意，不要增加用户未要求的主体或场景设定。"
                "需要明确主体、构图、镜头视角、材质或风格、光线、色彩和画面重点。"
                "如果用户没有指定风格，可以补足为自然、清晰、可执行的视觉描述。"
                "不要写解释，不要写多版本，不要输出 Markdown。"
            ),
            user_prompt=(
                "请把下面的生图需求改写成单条高质量提示词。\n\n"
                f"用户原始需求:\n{prompt}"
            ),
        )
        image_prompt = ""
        if result:
            image_prompt = str(result.get("image_prompt") or "").strip()
        if image_prompt:
            logs.append("image-generation: 已完成提示词优化。")
            return image_prompt
        logs.append("image-generation: 提示词优化失败，已回退原始提示词。")
        return prompt
