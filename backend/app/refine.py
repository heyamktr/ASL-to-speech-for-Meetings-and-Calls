"""LLM overlay: turn recognized ASL gloss words into a natural English sentence.

The sign classifier emits isolated WLASL words in the order they were signed —
ASL "gloss", with no articles, no copula, no inflection, and topic-comment order
(e.g. ``ME STORE GO FINISH`` -> "I went to the store."). This module sends that
word list to Gemini (Google Gen AI SDK) with a gloss->English system prompt and
returns one clean, punctuated sentence for the caption + text-to-speech layer.

Design mirrors ``tts.py``: the LLM is optional and best-effort. It never raises
for a normal availability problem — every failure path falls back to a
deterministic local join (capitalize + join + period) so the caption/voice
pipeline keeps working even with no API key or no ``google-genai`` package
installed.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RefineRequest(BaseModel):
    words: list[str] = Field(
        ..., min_length=1, description="ASL gloss words in the order they were signed"
    )


class RefineResponse(BaseModel):
    sentence: str
    source: str  # "llm" | "fallback"


SYSTEM_PROMPT = (
    "You convert American Sign Language (ASL) gloss into natural, written English "
    "for a live speech caption during a video call.\n\n"
    "You receive a short list of individual signs recognized in the order they were "
    'signed. ASL gloss omits articles, the verb "to be", and inflections, and uses '
    'topic-comment order (e.g. "ME STORE GO FINISH" -> "I went to the store.").\n\n'
    "Rules:\n"
    "- Output ONE natural English sentence (or a short question) conveying the signs.\n"
    '- Add articles, the correct form of "to be", verb tense/agreement, '
    "capitalization, and ending punctuation so it reads naturally when spoken aloud.\n"
    "- Reorder words only as needed for correct English grammar.\n"
    "- Use ONLY the meaning present in the given signs. Do not invent names, facts, "
    "or details that were not signed.\n"
    '- If the signs clearly form a question, end with "?".\n'
    "- Output only the sentence — no quotes, no preamble, no explanation."
)

# Prior turns steer the model toward terse, sign-faithful rewrites. These are
# few-shot examples (not a final-turn prefill), so the "model" turns are allowed.
FEW_SHOT: list[tuple[str, str]] = [
    ("you help me", "Can you help me?"),
    ("me name what", "What is your name?"),
    ("thank you", "Thank you."),
    ("me deaf you hear", "I am deaf and you are hearing."),
    ("me go home now", "I am going home now."),
]


def _local_fallback(words: list[str]) -> str:
    """Deterministic offline rewrite: join, capitalize, add a period."""
    text = " ".join(w.strip() for w in words if w.strip())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


def refine_gloss(
    words: list[str],
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    max_words: int,
) -> tuple[str, str]:
    """Rewrite gloss ``words`` into ``(sentence, source)``.

    ``source`` is ``"llm"`` when Gemini produced the sentence, ``"fallback"`` when
    the deterministic local join was used (no key, missing SDK, empty/failed
    response). This function never raises for an availability problem.

    NOTE: the Gemini call is blocking; invoke this from a threadpool inside an
    async handler (``starlette.concurrency.run_in_threadpool``), as ``main.py`` does.
    """
    cleaned = [w.strip() for w in words if w and w.strip()]
    cleaned = cleaned[-max_words:]  # cap runaway input before it reaches the model
    if not cleaned:
        return "", "fallback"

    if not api_key:
        return _local_fallback(cleaned), "fallback"

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("google-genai SDK unavailable, using local fallback: %s", exc)
        return _local_fallback(cleaned), "fallback"

    contents: list = []
    for gloss, english in FEW_SHOT:
        contents.append(types.Content(role="user", parts=[types.Part(text=gloss)]))
        contents.append(types.Content(role="model", parts=[types.Part(text=english)]))
    contents.append(
        types.Content(role="user", parts=[types.Part(text=" ".join(cleaned))])
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
            ),
        )
        text = (response.text or "").strip()
        # The model occasionally wraps the sentence in quotes despite the prompt.
        text = text.strip('"').strip()
        if not text:
            return _local_fallback(cleaned), "fallback"
        return text, "llm"
    except Exception as exc:
        logger.warning("LLM refine failed, using local fallback: %s", exc)
        return _local_fallback(cleaned), "fallback"
