# src/llm_optional.py
import os, re
from typing import Optional, Tuple

# 1) Prefer Streamlit secrets (local & cloud), fallback to env var
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

from openai import OpenAI

_client: Optional[OpenAI] = None
def _client_ok() -> bool:
    """Initialize the OpenAI client once"""
    global _client
    if not OPENAI_API_KEY:
        return False
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return True

def have_llm() -> bool:
    return _client_ok()

def _one_line(text: str, max_chars: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) > max_chars:
        t = t[:max_chars-3].rstrip() + "..."
    if t and t[-1] not in ".!?":
        t += "."
    return t

def _chat(prompt: str, max_tokens: int) -> Tuple[Optional[str], Optional[str]]:
    if not _client_ok():
        return None, "Missing OPENAI_API_KEY (use .streamlit/secrets.toml or env var)"
    try:
        res = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        txt = (res.choices[0].message.content or "").strip()
        return (txt if txt else None, None)
    except Exception as e:
        return None, str(e)

def llm_summarize(subject: str, body: str, label: str) -> Tuple[Optional[str], Optional[str]]:
    prompt = (
        "Summarize the sender's intent in ONE short line (<=25 words). "
        "Be specific. Use the body for context. Do not repeat the subject.\n\n"
        f"Priority: {label}\nSubject: {subject}\nBody:\n{body}\n\n"
        "Return ONLY the one-line summary:"
    )
    txt, err = _chat(prompt, max_tokens=60)
    return (_one_line(txt, 180) if txt else None, err)

def llm_reply(subject: str, body: str, label: str) -> Tuple[Optional[str], Optional[str]]:
    prompt = (
        "Write a concise, professional email reply (<120 words).\n"
        "• Acknowledge the issue/request\n"
        "• Set expectations (no invented facts)\n"
        "• List 2–4 short bullet-point next steps\n"
        "Tone: clear, helpful, and polite.\n\n"
        f"Priority: {label}\nSubject: {subject}\nBody:\n{body}\n\n"
        "Return ONLY the email body (no headings):"
    )
    txt, err = _chat(prompt, max_tokens=220)
    return (txt.strip() if txt else None, err)
