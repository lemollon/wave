# lc/chains.py — simple LangChain wrappers (works even if you just call OpenAI directly)

from typing import List
import json

try:
    from langchain_core.prompts import ChatPromptTemplate
    _LC_OK = True
except Exception:
    _LC_OK = False

def market_summary_prompt(niche: str, city: str, data: dict) -> str:
    """Return a prompt string (works with or without LangChain installed)."""
    return (
        f"Summarize what's trending for {niche} in {city} in 5 bullets. "
        f"Then propose 3 ride-the-wave posts. Data:\n{json.dumps(data)[:5000]}"
    )
