# app.py — Wave (Trends → Leads → Outreach → Weekly Report)
# Full single-file Streamlit app with robust suggestion chips, cleanup,
# “Industry-first” guidance, and optional Pro stubs (LangChain/LangGraph/CrewAI).

import os
import io
import re
import json
import textwrap
import datetime as dt
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests

# ---------- Optional libs with graceful fallbacks ----------
DOCX_OK = True
try:
    from docx import Document
except Exception:
    DOCX_OK = False

MAPS_OK = True
try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    MAPS_OK = False

# OpenAI v1 SDK import
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Pro libs (optional)
LC_OK = LG_OK = LCO_OK = False
try:
    from langchain_core.prompts import ChatPromptTemplate
    LC_OK = True
except Exception:
    LC_OK = False

try:
    from langgraph.graph import StateGraph, END
    LG_OK = True
except Exception:
    LG_OK = False

try:
    from langchain_openai import ChatOpenAI
    LCO_OK = True
except Exception:
    LCO_OK = False

CREW_OK = False
try:
    from crewai import Agent, Task, Crew
    CREW_OK = True
except Exception:
    CREW_OK = False


# -------------------- ENV / helpers --------------------
def _env(k: str, d: str = "") -> str:
    """
    Look up a value from OS env; if absent, fall back to Streamlit secrets.
    This lets the app work on Render (env vars) and Streamlit Cloud (secrets).
    """
    v = os.getenv(k)
    if not v:
        try:
            v = st.secrets.get(k, d)  # safe even if secrets isn't configured
        except Exception:
            v = d
    return v


OPENAI_API_KEY = _env("OPENAI_API_KEY", "")

# Safe client init for Streamlit + Render
if OpenAI and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
else:
    client = None


def llm_ok() -> bool:
    return client is not None


def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4) -> str:
    """Small wrapper around OpenAI chat; returns '' if not configured."""
    if not llm_ok():
        return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(AI unavailable: {e})"


def build_docx_bytes(title: str, body_md: str) -> bytes:
    if not DOCX_OK:
        return b""
    doc = Document()
    doc.add_heading(title, level=1)
    for para in body_md.split("\n\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def inject_css():
    st.markdown(
        """
        <style>
          :root { --card-bg:#0e1117; --card-border:#2b2f36; }
          .kpi-card {border:1px solid var(--card-border); border-radius:14px; padding:16px 18px; background:var(--card-bg);}
          .kpi-label {font-size:.80rem; opacity:.85; letter-spacing:.2px}
          .kpi-value {font-weight:800; font-size:1.25rem; margin-top:4px}
          .stDataFrame { border:1px solid var(--card-border); border-radius:12px; }
          .stAlert { border-radius:12px; }
          .stButton>button { border-radius:10px; padding:.5rem .9rem; font-weight:600 }
          .chip { display:inline-block; padding:6px 10px; border:1px solid #2b2f36; border-radius:20px; margin:4px 6px 0 0; background:#11151c; cursor:pointer; font-size:0.9rem;}
          .chip:hover { background:#151a22; }
          .mini-banner {background:#0f1520; border:1px solid #2b2f36; border-radius:10px; padding:10px 12px; font-size:.92rem; margin:-6px 0 10px 0;}
          .pill { display:inline-block; padding:2px 8px; border:1px solid #2b2f36; border-radius:999px; margin-left:6px; font-size:.8rem; opacity:.9;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_banner():
    st.markdown(
        """
        <div class="mini-banner">
          <strong>Tip:</strong> Start with your <em>Industry</em> on <u>Trend Rider</u>. That selection powers Lead Finder, Outreach, and the Weekly Report.
          You can <b>combine industries</b> (comma-separated). To start fresh, use <b>Clear all context</b> in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------- Warm-up button (Render/Cloud cold starts) -------------
def warm_up_render(url: str, timeout: int = 10) -> str:
    if not url:
        return "No wake URL set. Add RENDER_WAKE_URL to your environment."
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if 200 <= r.status_code < 400:
            return f"Warmed! ({r.status_code})"
        r = requests.get(url, timeout=timeout)
        return f"Warmed! ({r.status_code})" if 200 <= r.status_code < 400 else f"Service responded: {r.status_code}"
    except Exception as e:
        return f"Warm-up failed: {e}"


# --------------------- Suggestion helpers --------------------
def extract_strings(raw: str, max_items: int = 20) -> List[str]:
    """
    Robustly extract a list of strings from LLM responses.
    Handles:
      - ```json ...``` fenced blocks
      - JSON arrays / objects
      - newline/comma separated lists
      - strips quotes/brackets, dedupes, and trims
    """
    if not raw:
        return []

    txt = raw.strip()

    # Strip triple fences
    txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
    txt = re.sub(r"```$", "", txt)

    # Remove leading 'json' label if present
    txt = re.sub(r"^\s*json\s*", "", txt, flags=re.IGNORECASE)

    # Try JSON load first
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed if str(x).strip()]
            return list(dict.fromkeys(items))[:max_items]
        elif isinstance(parsed, dict):
            # if dict with 'items' or 'keywords' etc.
            for k in ("items", "keywords", "values", "suggestions"):
                if k in parsed and isinstance(parsed[k], list):
                    items = [str(x).strip() for x in parsed[k] if str(x).strip()]
                    return list(dict.fromkeys(items))[:max_items]
    except Exception:
        pass

    # Fallback: strip [] and quotes, then split by commas/newlines
    txt = txt.replace("[", " ").replace("]", " ")
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")
    tokens = re.split(r"[\n,]+", txt)
    items = []
    for t in tokens:
        t = t.strip().strip('"').strip("'")
        # Drop trailing artifacts like ) or .
        t = re.sub(r"[)\.]+$", "", t).strip()
        if t and not t.lower().startswith(("suggested", "keywords", "subreddits", "feeder", "category")):
            items.append(t)
    # Dedup preserve order
    return list(dict.fromkeys(items))[:max_items]


def list_to_csv_str(values: List[str]) -> str:
    return ", ".join([v for v in values if v])


def csv_str_to_list(s: str) -> List[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def sync_text_and_list(text_key: str, list_key: str):
    """
    Ensure the text input and the session list stay in sync.
    """
    tl = csv_str_to_list(st.session_state.get(text_key, ""))
    cur = st.session_state.get(list_key, [])
    if tl != cur:
        st.session_state[list_key] = tl


def render_chip_row(items: List[str], add_to: str, text_key: str, label: str):
    """
    Render items as clickable chips. On click, add to list and update text input.
    `add_to` is the session key for the target list; `text_key` is the text input state key.
    """
    st.caption(label)
    cols = st.columns(6)
    # Slight grid, but we'll just loop and place inline
    hit_add_all = st.button("Add all", key=f"addall-{add_to}")
    hit_clear = st.button("Clear", key=f"clear-{add_to}")

    if hit_add_all:
        new = list(dict.fromkeys((st.session_state.get(add_to, []) or []) + items))
        st.session_state[add_to] = new
        st.session_state[text_key] = list_to_csv_str(new)

    if hit_clear:
        st.session_state[add_to] = []
        st.session_state[text_key] = ""

    # Chips
    chip_html = []
    for i in items:
        safe_key = f"chip-{add_to}-{hash(i)}"
        # HTML button-like chip rendered as form button
        clicked = st.button(i, key=safe_key)
        if clicked:
            new = list(dict.fromkeys((st.session_state.get(add_to, []) or []) + [i]))
            st.session_state[add_to] = new
            st.session_state[text_key] = list_to_csv_str(new)


# --------------------- Trends (inline) --------------------
def google_trends_rising(keywords: List[str], geo="US", timeframe="now 7-d") -> Dict:
    try:
        from pytrends.request import TrendReq
    except Exception:
        return {"source": "google_trends", "error": "pytrends not installed", "rising": [], "iot": {}}
    pytrends = TrendReq(hl="en-US", tz=360)
    rising_all, iot_map = [], {}
    for kw in keywords[:6]:
        try:
            pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
            iot = pytrends.interest_over_time()
            if not iot.empty:
                ser = iot[kw].reset_index().rename(columns={kw: "interest", "date": "ts"})
                ser["keyword"] = kw
                iot_map[kw] = ser.to_dict(orient="records")
            rq = pytrends.related_queries()
            if kw in rq and rq[kw].get("rising") is not None:
                rising_df = rq[kw]["rising"].head(10)
                for _, row in rising_df.iterrows():
                    rising_all.append({
                        "keyword": kw,
                        "query": row["query"],
                        "value": int(row.get("value", 0)),
                        "link": f"https://www.google.com/search?q={row['query'].replace(' ', '+')}",
                    })
        except Exception as e:
            rising_all.append({"keyword": kw, "query": None, "value": 0, "error": str(e)})
    return {"source": "google_trends", "rising": rising_all, "iot": iot_map}


def reddit_hot_or_top(subreddits: List[str], mode: str = "hot", limit: int = 15) -> Dict:
    """
    Attempts PRAW if credentials exist, else uses public JSON endpoint with raw_json=1 and retry.
    """
    # Try praw
    try:
        import praw  # noqa
        cid = _env("REDDIT_CLIENT_ID")
        csec = _env("REDDIT_CLIENT_SECRET")
        ua = _env("REDDIT_USER_AGENT", "wave/1.0 by <user>")
        if cid and csec and ua:
            reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua, check_for_async=False)
            reddit.read_only = True
            posts = []
            for sub in (subreddits or [])[:6]:
                try:
                    sr = reddit.subreddit(sub)
                    it = sr.top(limit=limit, time_filter="week") if mode == "top" else sr.hot(limit=limit)
                    for p in it:
                        posts.append({
                            "subreddit": sub,
                            "title": getattr(p, "title", None),
                            "score": int(getattr(p, "score", 0) or 0),
                            "url": f"https://www.reddit.com{getattr(p,'permalink','')}",
                            "created_utc": int(getattr(p, "created_utc", 0) or 0)
                        })
                except Exception as e:
                    posts.append({"subreddit": sub, "title": None, "score": 0, "error": str(e)})
            posts = sorted(posts, key=lambda x: x.get("score", 0), reverse=True)
            st.session_state["last_reddit_source"] = "praw"
            return {"source": "reddit", "posts": posts}
    except Exception:
        pass

    # Fallback: public JSON endpoint
    headers = {
        "User-Agent": "WavePilotPro/1.0 (https://wave; contact: owner@example.com)"
    }
    posts = []
    for sub in (subreddits or [])[:6]:
        try:
            url = f"https://www.reddit.com/r/{sub}/{ 'top' if mode=='top' else 'hot'}.json?limit={limit}&raw_json=1"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                # one quick retry
                r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            for c in data.get("data", {}).get("children", []):
                d = c.get("data", {})
                posts.append({
                    "subreddit": sub,
                    "title": d.get("title"),
                    "score": int(d.get("score", 0) or 0),
                    "url": "https://www.reddit.com" + d.get("permalink", ""),
                    "created_utc": int(d.get("created_utc", 0) or 0)
                })
        except Exception as e:
            posts.append({"subreddit": sub, "title": None, "score": 0, "error": str(e)})
    posts = sorted(posts, key=lambda x: x.get("score", 0), reverse=True)
    st.session_state["last_reddit_source"] = "reddit"
    return {"source": "reddit", "posts": posts}


def youtube_search(api_key: str, query: str, max_results: int = 10) -> Dict:
    if not api_key:
        return {"source": "youtube", "items": [], "error": "missing key"}
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part": "snippet", "q": query, "type": "video", "order": "date",
                  "maxResults": max_results, "key": api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        items = []
        for it in r.json().get("items", []):
            sn = it.get("snippet", {})
            items.append({
                "title": sn.get("title"),
                "channel": sn.get("channelTitle"),
                "publishedAt": sn.get("publishedAt"),
                "url": f"https://www.youtube.com/watch?v={it['id'].get('videoId')}"
            })
        return {"source": "youtube", "items": items}
    except Exception as e:
        return {"source": "youtube", "items": [], "error": str(e)}


def gather_trends(niche_keywords: List[str], city="", state="", subs: Optional[List[str]] = None,
                  geo="US", timeframe="now 7-d", youtube_api_key: Optional[str] = None,
                  reddit_mode: str = "hot") -> Dict:
    subs = subs or ["SmallBusiness", "Marketing"]
    kw = [k for k in niche_keywords if k][:6]
    if city and state:
        kw.append(f"{city} {state}")
    gt = google_trends_rising(kw, geo=geo, timeframe=timeframe)
    rd = reddit_hot_or_top(subs, mode=reddit_mode)
    yt = youtube_search(youtube_api_key or _env("YOUTUBE_API_KEY", ""), " | ".join(kw) or "local business", 10)
    return {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "inputs": {"keywords": kw, "geo": geo, "timeframe": timeframe, "city": city, "state": state, "subs": subs},
        "google_trends": gt,
        "reddit": rd,
        "youtube": yt
    }


# ------------------ Google Places (v1 Text Search) ----------------
def search_places_optional(query: str, city: str, state: str, limit: int = 12, api_key: str = "") -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    text_url = "https://places.googleapis.com/v1/places:searchText"
    loc = ", ".join([x for x in [city.strip(), state.strip()] if x])
    q = f"{query} in {loc}" if loc else query
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.nationalPhoneNumber,"
            "places.websiteUri,places.googleMapsUri"
        ),
    }
    body = {"textQuery": q, "maxResultCount": limit}
    r = requests.post(text_url, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    places = r.json().get("places", []) or []
    rows = []
    for p in places[:limit]:
        name = (p.get("displayName") or {}).get("text") or p.get("name", "").split("/")[-1]
        locd = p.get("location") or {}
        rows.append({
            "Name": name,
            "Rating": float(p.get("rating", 0) or 0),
            "Reviews": int(p.get("userRatingCount", 0) or 0),
            "Phone": p.get("nationalPhoneNumber", ""),
            "Website": p.get("websiteUri", ""),
            "Address": p.get("formattedAddress", ""),
            "Lat": locd.get("latitude"),
            "Lng": locd.get("longitude"),
            "MapsUrl": p.get("googleMapsUri", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["__quality"] = df["Rating"].fillna(0) * (1 + (df["Reviews"].fillna(0) / 1000))
        df = df.sort_values("__quality", ascending=False).drop(columns="__quality")
    return df


# ----------------- Streamlit app setup -------------------
st.set_page_config(page_title="Wave — AI Growth Team (Pro)", page_icon="🌊", layout="wide")
inject_css()

# Sidebar: theme + Warm-up + Pro toggles + Clear
st.sidebar.markdown("### Appearance")
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
if theme == "Light":
    st.markdown("<style>body{background:#f6f7fb;color:#111}</style>", unsafe_allow_html=True)

st.sidebar.markdown("### Keep service awake")
wake_url = _env("RENDER_WAKE_URL", "")
if st.sidebar.button("🔥 Warm up the AI"):
    msg = warm_up_render(wake_url)
    (st.sidebar.success if msg.startswith("Warmed") else st.sidebar.warning)(msg)

st.sidebar.markdown("### Clear context")
if st.sidebar.button("Clear all context"):
    for k in ["trend_data_cache", "lead_data_cache", "niche_keywords_list", "subreddits_list", "feeder_cats_list",
              "trend_niche", "trend_subs", "last_reddit_source"]:
        st.session_state.pop(k, None)
    st.sidebar.success("Cleared app context.")

st.sidebar.markdown("### Pro toggles (optional)")
use_langchain = st.sidebar.toggle("LangChain Enricher", value=False)
use_langgraph = st.sidebar.toggle("LangGraph Orchestrator", value=False)
use_crewai = st.sidebar.toggle("CrewAI Growth Crew", value=False)
autogpt_url = st.sidebar.text_input("AutoGPT Webhook URL (optional)", _env("AUTOGPT_URL", ""))

# Session vars (plus suggestion lists)
st.session_state.setdefault("trend_data_cache", None)
st.session_state.setdefault("lead_data_cache", None)
st.session_state.setdefault("reddit_mode", "hot")
st.session_state.setdefault("out_persona", "Local Professional")
st.session_state.setdefault("niche_keywords_list", [])
st.session_state.setdefault("subreddits_list", [])
st.session_state.setdefault("feeder_cats
