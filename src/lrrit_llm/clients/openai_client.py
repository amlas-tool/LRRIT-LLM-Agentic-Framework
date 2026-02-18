from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class OpenAIChatClient:
    """
    Minimal wrapper expected by agents: .complete(prompt: str) -> str
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",         # default model
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a careful, evidence-grounded evaluator."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
