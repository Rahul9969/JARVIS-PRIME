"""
JARVIS-PRIME Cognitive Core — Multi-Provider LLM Router
=========================================================

Provides a unified interface to multiple free LLM providers with
automatic failover, rate limiting, and token tracking.

Provider Stack (priority order):
    1. Groq     — Llama 3.3 70B (free, 14.4K req/day, blazing fast)
    2. Gemini   — Gemini 2.5 Flash (free, 1.5K req/day, 1M context)
    3. Ollama   — Qwen 2.5 7B local (free, unlimited, ~8 tok/s CPU)
    4. Mock     — Template responses (no API needed, for testing)

Usage:
    core = CognitiveCore()
    response = await core.generate("Explain Casimir force engineering")
    response = await core.generate_structured("Decompose this goal", schema={...})
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx


class ProviderStatus(str, Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ProviderMetrics:
    """Track provider usage and performance."""
    requests_made: int = 0
    tokens_used: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: float = 0.0
    rate_limit_reset: float = 0.0
    status: ProviderStatus = ProviderStatus.AVAILABLE

    def record_request(self, tokens: int, latency_ms: float, success: bool) -> None:
        self.requests_made += 1
        self.tokens_used += tokens
        self.last_request_time = time.time()
        alpha = 0.2
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        if not success:
            self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests_made": self.requests_made,
            "tokens_used": self.tokens_used,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "status": self.status.value,
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.metrics = ProviderMetrics()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text from prompt."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        ...


class GroqProvider(LLMProvider):
    """
    Groq cloud inference — extremely fast, free tier.
    Uses custom LPU hardware for low-latency inference.

    Free tier: 30 req/min, 14,400 req/day.
    Models: llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, etc.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
    ):
        super().__init__("groq", model)
        self.api_key = api_key or os.environ.get("JARVIS_GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1"
        self._client: httpx.AsyncClient | None = None

    def is_available(self) -> bool:
        if not self.api_key:
            self.metrics.status = ProviderStatus.DISABLED
            return False
        if self.metrics.status == ProviderStatus.RATE_LIMITED:
            if time.time() > self.metrics.rate_limit_reset:
                self.metrics.status = ProviderStatus.AVAILABLE
            else:
                return False
        return self.metrics.status == ProviderStatus.AVAILABLE

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        client = await self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            latency = (time.monotonic() - start) * 1000

            if response.status_code == 429:
                self.metrics.status = ProviderStatus.RATE_LIMITED
                self.metrics.rate_limit_reset = time.time() + 60
                self.metrics.record_request(0, latency, False)
                raise RateLimitError(f"Groq rate limited. Retry after 60s.")

            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", len(content) // 4)
            self.metrics.record_request(tokens, latency, True)
            return content

        except httpx.HTTPStatusError as e:
            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(0, latency, False)
            if e.response.status_code == 429:
                self.metrics.status = ProviderStatus.RATE_LIMITED
                self.metrics.rate_limit_reset = time.time() + 60
            raise
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(0, latency, False)
            raise


class GeminiProvider(LLMProvider):
    """
    Google Gemini via AI Studio — free tier with generous limits.
    1,500 req/day for Flash models, 1M+ token context.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
    ):
        super().__init__("gemini", model)
        self.api_key = api_key or os.environ.get("JARVIS_GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._client: httpx.AsyncClient | None = None

    def is_available(self) -> bool:
        if not self.api_key:
            self.metrics.status = ProviderStatus.DISABLED
            return False
        if self.metrics.status == ProviderStatus.RATE_LIMITED:
            if time.time() > self.metrics.rate_limit_reset:
                self.metrics.status = ProviderStatus.AVAILABLE
            else:
                return False
        return self.metrics.status == ProviderStatus.AVAILABLE

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        client = await self._get_client()

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        start = time.monotonic()
        try:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    },
                },
            )
            latency = (time.monotonic() - start) * 1000

            if response.status_code == 429:
                self.metrics.status = ProviderStatus.RATE_LIMITED
                self.metrics.rate_limit_reset = time.time() + 60
                self.metrics.record_request(0, latency, False)
                raise RateLimitError("Gemini rate limited.")

            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response format
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts)
            else:
                content = ""

            tokens = data.get("usageMetadata", {}).get("totalTokenCount", len(content) // 4)
            self.metrics.record_request(tokens, latency, True)
            return content

        except httpx.HTTPStatusError as e:
            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(0, latency, False)
            raise
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(0, latency, False)
            raise


class OllamaProvider(LLMProvider):
    """
    Ollama local inference — unlimited, private, works offline.
    Runs Qwen 2.5 7B Q4_K_M at ~8 tok/s on Ryzen 7 CPU.
    """

    def __init__(
        self,
        base_url: str = "",
        model: str = "qwen2.5:7b",
    ):
        super().__init__("ollama", model)
        self.base_url = base_url or os.environ.get(
            "JARVIS_OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._client: httpx.AsyncClient | None = None
        self._checked = False
        self._is_running = False

    def is_available(self) -> bool:
        if not self._checked:
            # Lazy check — don't block on init
            return True  # Optimistic; will fail gracefully
        return self._is_running

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)  # Long timeout for CPU inference
        return self._client

    async def _check_availability(self) -> bool:
        """Check if Ollama is running."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
            self._checked = True
            self._is_running = resp.status_code == 200
            return self._is_running
        except Exception:
            self._checked = True
            self._is_running = False
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if not self._checked:
            await self._check_availability()
        if not self._is_running:
            raise ConnectionError("Ollama is not running")

        client = await self._get_client()
        start = time.monotonic()

        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            latency = (time.monotonic() - start) * 1000
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "")
            tokens = data.get("eval_count", len(content) // 4)
            self.metrics.record_request(tokens, latency, True)
            return content

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(0, latency, False)
            raise


class MockProvider(LLMProvider):
    """
    Mock provider for testing without any API keys.
    Returns structured template responses.
    """

    def __init__(self):
        super().__init__("mock", "mock-v1")

    def is_available(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        await asyncio.sleep(0.05)  # Simulate latency

        prompt_lower = prompt.lower()

        # Task decomposition
        if "decompose" in prompt_lower or "subtask" in prompt_lower:
            return json.dumps({
                "tasks": [
                    {
                        "id": "task-001",
                        "domain": "exotic_physics",
                        "type": "full_analysis",
                        "description": "Physics analysis from query",
                        "priority": "high",
                    },
                    {
                        "id": "task-002",
                        "domain": "scientific",
                        "type": "research_summary",
                        "description": "Research context from query",
                        "priority": "medium",
                    },
                ],
                "cross_domain_opportunities": ["Physics-AI synthesis possible"],
            })

        # Synthesis
        if "synthesize" in prompt_lower or "combine" in prompt_lower:
            return (
                "## Synthesized Analysis\n\n"
                "The cross-domain analysis reveals several key insights:\n\n"
                "1. **Primary Finding**: Results are consistent across domains.\n"
                "2. **Cross-Domain Insight**: Connections identified between physics and AI research.\n"
                "3. **Recommendation**: Further investigation warranted on frontier intersections.\n"
            )

        # Validation
        if "validate" in prompt_lower or "quality" in prompt_lower:
            return json.dumps({
                "quality_score": 0.85,
                "completeness": 0.9,
                "is_sufficient": True,
                "suggestions": ["Consider additional parameter sweeps"],
            })

        # Default
        self.metrics.record_request(len(prompt) // 4, 50, True)
        return (
            f"[Mock LLM Response]\n\n"
            f"Analysis of: {prompt[:200]}...\n\n"
            f"This is a mock response. Connect a real LLM provider "
            f"(Groq/Gemini/Ollama) for production-quality reasoning.\n\n"
            f"Key points:\n"
            f"- Query understood and categorized\n"
            f"- Domain agents would be engaged for detailed analysis\n"
            f"- Cross-domain synthesis opportunities identified\n"
        )


class RateLimitError(Exception):
    """Raised when a provider hits rate limits."""
    pass


class CognitiveCore:
    """
    Multi-Provider LLM Router with automatic failover.

    Priority: Groq -> Gemini -> Ollama -> Mock
    Features:
    - Automatic failover on errors/rate limits
    - Per-provider metrics tracking
    - Structured JSON output generation
    - System prompt management
    - Token budget tracking
    """

    SYSTEM_PROMPT = (
        "You are JARVIS-PRIME, an omnidisciplinary autonomous intelligence system. "
        "You operate at the frontier of human knowledge across physics, AI, biology, "
        "quantum computing, and more. You reason from first principles, identify gaps "
        "in existing research, and produce mathematically rigorous analysis. "
        "You never say 'I cannot' — you say 'here is the research pathway to make this possible.' "
        "Respond with precision, depth, and actionable insights."
    )

    def __init__(
        self,
        groq_api_key: str = "",
        gemini_api_key: str = "",
        ollama_base_url: str = "",
    ):
        self.providers: list[LLMProvider] = [
            GroqProvider(api_key=groq_api_key),
            GeminiProvider(api_key=gemini_api_key),
            OllamaProvider(base_url=ollama_base_url),
            MockProvider(),
        ]
        self._total_tokens = 0
        self._total_requests = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        preferred_provider: str | None = None,
    ) -> str:
        """
        Generate text using the best available provider.

        Tries providers in priority order, fails over on errors.
        """
        sys_prompt = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT

        # If preferred provider specified, try it first
        if preferred_provider:
            for p in self.providers:
                if p.name == preferred_provider and p.is_available():
                    try:
                        result = await p.generate(prompt, sys_prompt, temperature, max_tokens)
                        self._total_requests += 1
                        return result
                    except Exception:
                        pass

        # Standard failover chain
        last_error = None
        for provider in self.providers:
            if not provider.is_available():
                continue
            try:
                result = await provider.generate(prompt, sys_prompt, temperature, max_tokens)
                self._total_requests += 1
                return result
            except Exception as e:
                last_error = e
                continue

        # All providers failed — should never happen since MockProvider is always available
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def generate_structured(
        self,
        prompt: str,
        schema_hint: str = "",
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output from LLM.

        Adds JSON formatting instructions and parses the result.
        """
        json_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.\n"
        )
        if schema_hint:
            json_prompt += f"Expected format: {schema_hint}\n"

        response = await self.generate(
            json_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # Try to extract JSON from response
        return self._parse_json(response)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: return as wrapped text
        return {"raw_response": text}

    def get_active_provider(self) -> str:
        """Get the name of the first available provider."""
        for p in self.providers:
            if p.is_available():
                return p.name
        return "none"

    def stats(self) -> dict[str, Any]:
        """Get statistics for all providers."""
        return {
            "active_provider": self.get_active_provider(),
            "total_requests": self._total_requests,
            "providers": {
                p.name: {
                    "model": p.model,
                    **p.metrics.to_dict(),
                }
                for p in self.providers
            },
        }

    async def close(self) -> None:
        """Close all HTTP clients."""
        for p in self.providers:
            if hasattr(p, '_client') and p._client and not p._client.is_closed:
                await p._client.aclose()
