"""LLM client using the standard OpenAI Chat Completions interface.

The endpoint is fully configurable (``base_url`` + ``api_key`` + ``model``) so
any OpenAI-compatible service works: OpenAI, Azure OpenAI, DeepSeek, local
vLLM / Ollama / LM Studio, etc.

Configuration sources (in priority order):
  1. Environment variables: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
  2. config/llm_config.json

If no usable API key is found and ``allow_mock_fallback`` is true, a
``MockLLM`` is used so the full attack-defense pipeline runs offline. The mock
behaves like a model that aggregates evidence from the retrieved contexts:
it extracts short answer candidates from each context and returns the
majority (ties broken by retrieval rank). This makes ASR / RobustAccuracy
meaningful even without a real LLM.
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Optional

from core.config import load_json


class LLMClient:
    """OpenAI-compatible chat client with a deterministic mock fallback."""

    def __init__(self, config_path: Optional[str] = None,
                 allow_mock_fallback: bool = True,
                 max_output_tokens: int = 150,
                 temperature: float = 0.0):
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = None
        self.model_name: str = "mock"
        self.is_mock = False

        # Resolve config path from default.yaml if not given.
        if config_path is None:
            config_path = "config/llm_config.json"
        cfg = load_json(config_path) if _file_exists(config_path) else {}

        # Env vars override file config.
        api_key = os.environ.get("OPENAI_API_KEY") or _first_key(cfg)
        base_url = os.environ.get("OPENAI_BASE_URL") or cfg.get("base_url")
        model = os.environ.get("OPENAI_MODEL") or cfg.get("model_info", {}).get("name", "gpt-3.5-turbo")
        params = cfg.get("params", {})
        self.max_output_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", params.get("max_output_tokens", max_output_tokens)))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", params.get("temperature", temperature)))

        has_key = bool(api_key) and api_key not in ("YOUR_API_KEY_HERE", "", None)

        if has_key:
            try:
                from openai import OpenAI  # standard SDK
                self._client = OpenAI(api_key=api_key, base_url=base_url)
                self.model_name = model
                self.is_mock = False
                return
            except Exception as e:  # pragma: no cover - depends on env
                print(f"[LLM] openai client init failed ({e!r}); falling back to mock.")

        if not allow_mock_fallback:
            raise RuntimeError(
                "No OpenAI-compatible API key configured and mock fallback disabled. "
                "Set OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL or fill config/llm_config.json."
            )
        print("[LLM] No API key configured - using MockLLM (offline evidence-aggregation model).")
        self._mock = MockLLM()
        self.model_name = "mock"
        self.is_mock = True

    def query(self, prompt: str) -> str:
        """Return the model's text response for a single prompt."""
        if self.is_mock:
            return self._mock.query(prompt)
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover - network/runtime errors
            print(f"[LLM] query failed ({e!r}); returning empty string.")
            return ""


class MockLLM:
    """Offline heuristic LLM.

    Parses the MULTIPLE_PROMPT to recover the contexts and the question, then
    extracts short answer candidates from each context and returns the
    majority-vote candidate (ties broken by retrieval rank). This mirrors how
    a real RAG generator is misled: whichever answer dominates the retrieved
    context wins.
    """

    _CONTEXT_RE = re.compile(r"Contexts:\s*(.*?)\s*Query:\s*(.*?)\s*Answer:", re.DOTALL)

    def query(self, prompt: str) -> str:
        m = self._CONTEXT_RE.search(prompt)
        if not m:
            # Fallback: return the first short line of the prompt.
            return prompt.strip().split("\n")[0][:64]
        context_block, question = m.group(1).strip(), m.group(2).strip()
        contexts = [c.strip() for c in context_block.split("\n") if c.strip()]
        if not contexts:
            return "I don't know"

        candidates: list[str] = []
        for ctx in contexts:
            cand = _extract_answer_candidate(ctx, question)
            if cand:
                candidates.append(cand)

        if not candidates:
            return "I don't know"

        # Majority vote, ties broken by earliest (highest-ranked) occurrence.
        counts = Counter(_norm_key(c) for c in candidates)
        best_key = counts.most_common(1)[0][0]
        for c in candidates:  # preserve original surface form of the winner
            if _norm_key(c) == best_key:
                return c
        return candidates[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_exists(config_path: str) -> bool:
    from core.config import resolve_path
    return resolve_path(config_path).exists()


def _first_key(cfg: dict) -> Optional[str]:
    info = cfg.get("api_key_info", {})
    keys = info.get("api_keys", [])
    pos = int(info.get("api_key_use", 0))
    if 0 <= pos < len(keys):
        return keys[pos]
    return None


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip().lower())


# Patterns that commonly introduce a factual answer in a passage.
# Each captures either a capitalized phrase or a number.
_CAP_OR_NUM = r"([A-Z][\w&'\.\- ]{0,40}?|\d{1,6}(?:\.\d+)?)"
_ANSWER_PATTERNS = [
    # "X is/are/was/were <answer>"
    re.compile(r"\b(?:is|are|was|were|equals?|means|refers to)\s+" + _CAP_OR_NUM +
               r"(?:[.,;]|\b(?:in|of|and|which|that|for|with|located)\b|$)", re.IGNORECASE),
    # "comprising/contains/total of <number>"
    re.compile(r"\b(?:comprising|contains|consists? of|totals? to|total of|count of|number of)\s+"
               r"(?:a total of\s+)?(\d{1,6}(?:[\.,]\d+)?)", re.IGNORECASE),
    # "<number> episodes/members/..."
    re.compile(r"\b(\d{1,6})\s+(?:episodes|members|seasons|countries|states|cities|people|years|days|times)\b", re.IGNORECASE),
    # "the capital of ... is <Capitalized>"
    re.compile(r"\bthe capital of\b[^.]{0,40}?\bis\s+([A-Z][\w ]{0,40}?)(?:[.,;]|$)", re.IGNORECASE),
    # "the answer is <answer>"
    re.compile(r"\bthe answer is\s+" + _CAP_OR_NUM + r"(?:[.,;]|$)", re.IGNORECASE),
]


def _extract_answer_candidate(ctx: str, question: str) -> str:
    """Extract the most salient short answer span from a single context."""
    qlower = question.lower().strip()
    is_yesno = bool(re.match(r"^(is|are|was|were|do|does|did|can|could|will|"
                             r"would|should|has|have|had)\b", qlower))

    # Yes/no questions: look for an explicit yes/no assertion.
    if is_yesno:
        m = re.search(r"\b(?:is|are|was|were)\s+(yes|no|true|false)\b", ctx, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        m = re.search(r"\bthe answer is\s+(yes|no)\b", ctx, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    # Prefer explicit answer patterns first.
    for pat in _ANSWER_PATTERNS:
        m = pat.search(ctx)
        if m:
            cand = m.group(1).strip().strip("\"'.,;").strip()
            if cand:
                return cand

    # Otherwise, look for a capitalized phrase or number near a question keyword.
    qkeywords = [w for w in re.findall(r"[A-Za-z]{4,}", question)]
    sents = re.split(r"(?<=[.!?\n])\s+", ctx)
    for sent in sents:
        if any(kw.lower() in sent.lower() for kw in qkeywords):
            # numbers
            nums = re.findall(r"\b\d{1,6}(?:[\.,]\d+)?\b", sent)
            if nums:
                return nums[0]
            # capitalized multi-word phrase
            caps = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3}\b", sent)
            if caps:
                return caps[0]
    return ""
