#!/usr/bin/env python3
"""OpenAI Python SDK smoke against a Ferrum or vLLM-compatible endpoint."""
import os

from openai import OpenAI

base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
api_key = os.environ.get("OPENAI_API_KEY", "dummy-key-not-checked")
model = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-0.6B")

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
