# lc/orchestrator.py — minimal LangChain/LangGraph orchestrator
# If LangChain/LangGraph/OpenAI are not installed or API key missing, callers should catch exceptions.

import os
import pandas as pd

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    LC_DEPS = True
except Exception:
    LC_DEPS = False

def _env(k, d=""):
    return os.getenv(k, d)

def _llm(temp=0.3):
    if not LC_DEPS:
        raise RuntimeError("LangChain/LangGraph not installed")
    key = _env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing for LangChain")
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp, api_key=key)

# --------- Lead research graph ----------
# State: {"leads": DataFrame, "city": str, "state": str, "notes": str}

def lc_lead_research(df: pd.DataFrame, city: str, state: str) -> pd.DataFrame:
    """Enrich 'Why' reasons with LLM and normalize Score (0-100)."""
    if not LC_DEPS:
        raise RuntimeError("LangChain/LangGraph not installed")

    llm = _llm(0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You help a local pro pick partnership targets. Be specific and brief."),
        ("human", "Given this business:\n{row}\nCity: {city} {state}\n"
                  "Return a one-line reason why it's a strong feeder partner.")
    ])

    def enrich(state):
        leads = state["leads"]
        out = leads.copy()
        why_extra = []
        for _, r in out.iterrows():
            rowtxt = f"Name={r.get('Name')}, Rating={r.get('Rating')}, Reviews={r.get('Reviews')}, Address={r.get('Address')}"
            msg = prompt.format_messages(row=rowtxt, city=state["city"], state=state["state"])
            try:
                reason = llm.invoke(msg).content.strip()
            except Exception as e:
                reason = f"(AI unavailable: {e})"
            why_extra.append(reason)
        out["Why"] = out["Why"].fillna("").astype(str) + (" · " if out["Why"].notna().any() else "") + pd.Series(why_extra)
        # normalize score gently
        out["Score"] = out["Score"].clip(lower=0, upper=100)
        return {"leads": out, "city": state["city"], "state": state["state"], "notes": ""}

    g = StateGraph(dict)
    g.add_node("enrich", enrich)
    g.set_entry_point("enrich")
    g.add_edge("enrich", END)
    app = g.compile()

    result = app.invoke({"leads": df, "city": city, "state": state, "notes": ""})
    return result["leads"]

# --------- Playbook generator ----------
def lc_playbook(persona: str, target: str) -> str:
    if not LC_DEPS:
        raise RuntimeError("LangChain not installed")
    llm = _llm(0.4)
    prompt = ChatPromptTemplate.from_template(
        "Create a short outreach playbook for a {persona} targeting {target}. "
        "Include: 3 talk-tracks, 3 subject lines, and 3 CTAs. Keep it concise, bullet style."
    )
    return llm.invoke(prompt.format_messages(persona=persona, target=target)).content
