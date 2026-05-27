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

        try:
            image_result = self.openai_service.generate_image(prompt=prompt, size=size, quality=quality)
        except Exception as error:
            logs.append(f"image-generation: gpt-image-2 调用失败，原因：{error}")
            return {
                "prompt": prompt,
                "revised_prompt": prompt,
                "model": self.settings.openai_image_model,
                "image_b64": None,
                "image_url": None,
                "logs": logs,
            }
        if not image_result:
            logs.append("image-generation: gpt-image-2 未返回图片结果。")
            return {
                "prompt": prompt,
                "revised_prompt": prompt,
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
            "revised_prompt": image_result.get("revised_prompt", prompt),
            "model": image_result.get("model", self.settings.openai_image_model),
            "image_b64": image_result.get("image_b64"),
            "image_url": image_result.get("image_url"),
            "logs": logs,
        }
