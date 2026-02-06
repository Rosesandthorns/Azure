import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class ModelConfig:
    provider: str
    endpoint_url: str
    model_name: str
    timeout_s: int
    max_tokens: int
    temperature: float


class ModelInterface:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class LocalHTTPModel(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _fallback_response(self, messages: List[Dict[str, str]]) -> str:
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if not last_user:
            return "I am an artificial system. How can I help?"
        return (
            "I am an artificial system running locally. "
            "I will answer briefly: "
            f"{last_user['content'][:300]}"
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if not self.config.endpoint_url:
            return self._fallback_response(messages)
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        try:
            response = requests.post(
                self.config.endpoint_url,
                json=payload,
                timeout=self.config.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self.logger.warning("Model request failed: %s", exc)
            return self._fallback_response(messages)
        try:
            data = response.json()
        except json.JSONDecodeError:
            self.logger.warning("Model response not JSON, using fallback.")
            return self._fallback_response(messages)
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        return self._fallback_response(messages)


def build_model_interface(config: Dict[str, object]) -> ModelInterface:
    model_config = ModelConfig(
        provider=str(config.get("provider", "local_http")),
        endpoint_url=str(config.get("endpoint_url", "")),
        model_name=str(config.get("model_name", "local-model")),
        timeout_s=int(config.get("timeout_s", 30)),
        max_tokens=int(config.get("max_tokens", 512)),
        temperature=float(config.get("temperature", 0.7)),
    )
    return LocalHTTPModel(model_config)
