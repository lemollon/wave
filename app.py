# app.py — WavePilot Pro (single file)
# Trends → Leads → Outreach → Weekly Report + (optional) LangChain/LangGraph/CrewAI stubs.
# Works on Render Free (core requirements). Pro libs are optional (requirements-pro.txt).

import os
import io
import json
import textwrap
import time
import datetime as dt
from typing import Dict, List, Optional

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

# Pro libs (optional) — import flags
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
def _secrets_file_exists() -> bool:
    """Return True only if a Streamlit secrets.toml actually exists on disk."""
    paths = [
        "/app/.streamlit/secrets.toml",
        os.path.expanduser("~/.streamlit/secrets.toml"),
        "/root/.streamlit/secrets.toml",
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                return True
        except Exception:
            pass
    return False


def _env(k: str, d: str = "") -> str:
    """
    Look up a value from OS env; if absent, fall back to Streamlit secrets
    ONLY if a secrets.toml file truly exists (prevents the red secrets banner).
    """
    v = os.getenv(k)
    if v:
        return v
    if _secrets_file_exists():
        try:
            return st.secrets.get(k, d)
        except Exception:
            return d
    return d


def _env_any(keys: List[str], default: str = "") -> str:
    """Return the first found env value among aliases."""
    for k in keys:
        v = _env(k)
        if v:
            return v
    return default


def to_json(obj) -> str:
    """Safe JSON that turns pandas/numpy/Datetime objects into strings."""
    def _default(o):
        try:
            if hasattr(o, "isoformat"):
                return o.isoformat()
        except Exception:
            pass
        return str(o)
    return json.dumps(obj, default=_default)


# Be generous in finding your key (but do NOT print it)
OPENAI_API_KEY = (
    _env("OPENAI_API_KEY")
    or _env("OPENAI_API_TOKEN")
    or _env("OPENAI")
    or _env("OPENAI_SECRET")
    or ""
)

# YouTube & Places keys (allow common aliases, but prefer canonical names)
YOUTUBE_API_KEY = _env_any(["YOUTUBE_API_KEY", "GOOGLE_YOUTUBE", "googl_youtube", "YOUTUBE_KEY"])
GOOGLE_PLACES_API_KEY = _env_any(["GOOGLE_PLACES_API_KEY", "GOOGLE_MAPS_API_KEY", "GOOGLE_API_KEY", "PLACES_API_KEY"])

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


def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4, model: Optional[str] = None) -> str:
    """Small wrapper around OpenAI chat; returns '' if not configured."""
    if not llm_ok():
        return ""
    try:
        r = client.chat.completions.create(
            model=model or "gpt-4o-mini",
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
          .chip {display:inline-block; padding:4px 10px; border-radius:14px; background:#1d2431; margin:3px; font-size:.85rem;}
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


def _safe_show_df(df: pd.DataFrame, preferred_cols: List[str], **kwargs):
    """Render only columns that actually exist; avoid KeyError."""
    if df is None or df.empty:
        st.info("No data to display.")
        return
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        st.dataframe(df.head(20), **kwargs)
    else:
        st.dataframe(df[cols].head(20), **kwargs)


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
                        "query": row.get("query"),
                        "value": int(row.get("value", 0) or 0),
                        "link": f"https://www.google.com/search?q={(row.get('query') or '').replace(' ', '+')}",
                    })
        except Exception as e:
            rising_all.append({"keyword": kw, "query": None, "value": 0, "error": str(e)})
    return {"source": "google_trends", "rising": rising_all, "iot": iot_map}


# -------- Reddit with multi-tier fallback (real only; AI summary separate) --------
def _reddit_ai_summary(subreddits: List[str], mode: str, keywords: List[str], city: str, state: str) -> str:
    """If Reddit APIs block us, generate an AI 'what's trending in reddit-like communities' summary (CLEARLY labeled)."""
    if not llm_ok():
        return "(LLM unavailable)"
    prompt = (
        "Reddit APIs are unavailable, but we still need actionable insight for marketing.\n"
        f"Industry keywords: {', '.join(keywords[:10]) or 'n/a'}\n"
        f"Subreddits we would check: {', '.join(subreddits[:8]) or 'n/a'}\n"
        f"Location: {city}, {state}\n\n"
        "Based on general internet/consumer trends (not specific posts), summarize:\n"
        "1) 5 topics likely trending in these communities\n"
        "2) why each matters for lead-gen this week\n"
        "3) 3 post ideas with hooks + CTA\n"
        "Keep the tone sales-forward, punchy, and localized."
    )
    return llm(prompt, system="Be useful, honest, and never claim specific Reddit posts.", temp=0.5)


def _reddit_fallback_json(subreddits: List[str], mode: str = "hot", limit: int = 15) -> Dict:
    """
    Fetch basic posts via Reddit's public JSON if PRAW/keys fail.
    Hardened with raw_json=1, realistic UA, and one retry on 429.
    """
    posts = []
    mode = "top" if mode == "top" else "hot"
    ua = _env("REDDIT_PUBLIC_UA", "WavePilotPro/1.0 (https://wavepilot.example; contact admin@example.com)")
    headers = {"User-Agent": ua}
    for sub in (subreddits or [])[:6]:
        try:
            url = f"https://www.reddit.com/r/{sub}/{mode}.json?limit={min(limit,25)}&raw_json=1"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                time.sleep(1.2)  # brief backoff
                r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            children = (data.get("data") or {}).get("children", [])
            for ch in children:
                d = (ch.get("data") or {})
                posts.append({
                    "subreddit": sub,
                    "title": d.get("title"),
                    "score": int(d.get("score", 0) or 0),
                    "url": "https://www.reddit.com" + (d.get("permalink") or ""),
                    "error": "",
                })
        except Exception as e:
            posts.append({"subreddit": sub, "title": None, "score": 0, "url": "", "error": f"fallback: {e}"})
    posts = sorted(posts, key=lambda x: x.get("score", 0), reverse=True)
    return {"source": "reddit_fallback", "posts": posts}


def reddit_hot_or_top(subreddits: List[str], mode: str = "hot", limit: int = 15) -> Dict:
    try:
        import praw
    except Exception:
        # No PRAW? go to fallback JSON
        return _reddit_fallback_json(subreddits, mode=mode, limit=limit)
    cid = _env("REDDIT_CLIENT_ID")
    csec = _env("REDDIT_CLIENT_SECRET")
    ua = _env("REDDIT_USER_AGENT", "WavePilotPro/1.0 (https://wavepilot.example; contact admin@example.com)")
    # If keys are missing, use fallback JSON
    if not (cid and csec and ua):
        return _reddit_fallback_json(subreddits, mode=mode, limit=limit)
    try:
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
                        "url": f"https://www.reddit.com{getattr(p, 'permalink', '')}",
                        "created_utc": int(getattr(p, "created_utc", 0) or 0),
                        "error": "",
                    })
            except Exception as e:
                posts.append({"subreddit": sub, "title": None, "score": 0, "url": "", "error": str(e)})
        posts = sorted(posts, key=lambda x: x.get("score", 0), reverse=True)
        return {"source": "reddit", "posts": posts}
    except Exception:
        return _reddit_fallback_json(subreddits, mode=mode, limit=limit)


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
            sn = it.get("snippet", {}) or {}
            vid = (it.get("id") or {}).get("videoId", "")
            items.append({
                "title": sn.get("title"),
                "channel": sn.get("channelTitle"),
                "publishedAt": sn.get("publishedAt"),
                "url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
            })
        return {"source": "youtube", "items": items}
    except Exception as e:
        return {"source": "youtube", "items": [], "error": str(e)}


def gather_trends(niche_keywords: List[str], city="", state="", subs: Optional[List[str]] = None,
                  geo="US", timeframe="now 7-d", youtube_api_key: Optional[str] = None,
                  reddit_mode: str = "hot") -> Dict:
    subs = subs or ["SmallBusiness", "Marketing"]
    kw = [k for k in niche_keywords if k][:10]
    if city and state:
        kw.append(f"{city} {state}")
    gt = google_trends_rising(kw, geo=geo, timeframe=timeframe)
    rd = reddit_hot_or_top(subs, mode=reddit_mode)
    yt = youtube_search(youtube_api_key or YOUTUBE_API_KEY, " | ".join(kw) or "local business", 10)
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
        name = (p.get("displayName") or {}).get("text") or (p.get("name", "").split("/")[-1] if p.get("name") else "")
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


# ----------------- AI Assist helpers (suggestions, ranking, polish) -------------------
INDUSTRY_CHOICES = [
    "Real Estate", "Mortgage", "Roofing", "Plumbing", "HVAC", "Electrician", "Landscaping",
    "Dentist", "Med Spa", "Chiropractor", "Gym / Fitness", "Restaurant", "Auto Repair",
    "E-commerce", "SaaS", "Moving Company", "Home Builder", "College Admissions", "Insurance"
]


def ai_suggest_keywords(industry: str, city: str, state: str) -> List[str]:
    if not llm_ok():
        return []
    prompt = (
        f"Suggest 8–12 high-intent, localizable niche keywords for the {industry} industry "
        f"in {city}, {state}. Return a JSON array of short keyword strings only."
    )
    txt = llm(prompt, system="You help marketers find money keywords.")
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Fallback: comma-split
    return [x.strip() for x in txt.split(",") if x.strip()][:12]


def ai_suggest_subreddits(industry: str) -> List[str]:
    if not llm_ok():
        return []
    prompt = (
        f"Suggest 5–8 subreddits relevant to {industry} buyers or operators. "
        "Return a JSON array of subreddit names (without r/)."
    )
    txt = llm(prompt, system="You know Reddit communities broadly.")
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [str(x).replace("r/", "").strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [x.replace("r/", "").strip() for x in txt.split(",") if x.strip()][:8]


def ai_suggest_feeders(industry: str, city: str) -> List[str]:
    if not llm_ok():
        return []
    prompt = (
        f"For {industry} in {city}, list 6–10 feeder business categories that meet ideal customers early. "
        "Examples for real estate include: apartment complexes, movers, mortgage brokers, home builders, storage facilities, schools. "
        "Return a JSON array of category strings."
    )
    txt = llm(prompt, system="You suggest practical partner categories.")
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in txt.split(",") if x.strip()][:10]


def ai_rank_opportunity(items: List[dict], kind: str, context: str) -> List[int]:
    """
    Ask AI to score each item 1-100 for revenue opportunity. Return list of scores (same length).
    kind: 'trends' | 'reddit' | 'youtube'
    """
    if not llm_ok() or not items:
        return [50] * len(items)
    short_items = items[:20]  # limit tokens
    prompt = (
        f"Score each {kind} item 1-100 for near-term lead-gen opportunity. "
        f"Context: {context}. Return JSON array of integers, length {len(short_items)}."
        "\nItems:\n" + to_json(short_items)
    )
    txt = llm(prompt, system="Be decisive and sales-focused. Higher=more opportunity.", temp=0.2)
    try:
        arr = json.loads(txt)
        if isinstance(arr, list) and len(arr) == len(short_items):
            scores = [int(x) if str(x).isdigit() else 50 for x in arr]
            # If fewer than items, pad; if more, trim
            if len(scores) < len(items):
                scores += [50] * (len(items) - len(scores))
            return scores[:len(items)]
    except Exception:
        pass
    return [50] * len(items)


def ai_polish_copy(text: str, tone: str = "Salesy") -> str:
    if not llm_ok() or not text.strip():
        return text
    s = (
        "You are a conversion-focused copy chief. Polish the draft to be crisp, persuasive, and high-converting. "
        "Preserve factual claims, add urgency sparingly, and avoid fluff."
    )
    p = f"Tone: {tone}. Rewrite the following keeping the core offer and CTAs clear:\n\n{text.strip()}"
    return llm(p, system=s, temp=0.5)


# ----------------- Streamlit app setup -------------------
st.set_page_config(page_title="WavePilot — AI Growth Team (Pro)", page_icon="🌊", layout="wide")
inject_css()

# Sidebar: theme + Warm-up + Pro (always on)
st.sidebar.markdown("### Appearance")
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
if theme == "Light":
    st.markdown("<style>body{background:#f6f7fb;color:#111}</style>", unsafe_allow_html=True)

st.sidebar.markdown("### Keep service awake")
wake_url = _env("RENDER_WAKE_URL", "")
if st.sidebar.button("🔥 Warm up the AI"):
    msg = warm_up_render(wake_url)
    (st.sidebar.success if msg.startswith("Warmed") else st.sidebar.warning)(msg)

# Pro features forced ON (no toggles)
use_langchain = True
use_langgraph = True
use_crewai   = True
st.sidebar.markdown("### Pro (always on)")
autogpt_url = st.sidebar.text_input("AutoGPT Webhook URL (optional)", _env("AUTOGPT_URL", ""))

# Session vars
st.session_state.setdefault("trend_data_cache", None)
st.session_state.setdefault("lead_data_cache", None)
st.session_state.setdefault("reddit_mode", "hot")
st.session_state.setdefault("out_persona", "Local Professional")

# NEW: store suggestions
st.session_state.setdefault("suggested_keywords", [])
st.session_state.setdefault("suggested_subs", [])
st.session_state.setdefault("suggested_feeders", [])

# NEW: store last reddit source for diagnostics
st.session_state.setdefault("last_reddit_source", "")

st.title("🌊 WavePilot — AI Growth Team (Pro)")
st.caption("Trends → Leads → Outreach (+ LangChain, LangGraph, CrewAI).")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report", "Pro Lab", "Competitor Sniffer"]
)

# ===================== TREND RIDER =====================
with tab1:
    st.subheader("Trend Rider — ride what's hot")

    # Industry-first controls + AI suggestions
    ctop1, ctop2, ctop3 = st.columns([1.2, 1, 1])
    industry_choice = ctop1.selectbox("Industry (quick pick)", INDUSTRY_CHOICES, index=0)
    industry_free = ctop2.text_input("Or type your industry", "", placeholder="e.g., Real Estate Investor")
    industry = industry_free.strip() or industry_choice

    rank_ai = ctop3.toggle("AI: Rank by Opportunity", value=False, help="Let AI rank tables by near-term lead-gen impact.")

    ccity, cstate = st.columns(2)
    city_prefill = ccity.text_input("City", "Katy", key="trend_city2")
    state_prefill = cstate.text_input("State", "TX", key="trend_state2")

    # AI suggestion buttons (before the main form so we can inject values)
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("💡 Suggest keywords"):
        st.session_state.suggested_keywords = ai_suggest_keywords(industry, city_prefill, state_prefill)
        st.success(f"Suggested {len(st.session_state.suggested_keywords)} keywords.")
    if sc2.button("💬 Suggest subreddits"):
        st.session_state.suggested_subs = ai_suggest_subreddits(industry)
        st.success(f"Suggested {len(st.session_state.suggested_subs)} subreddits.")
    if sc3.button("🧲 Suggest feeder categories (for Lead Finder)"):
        st.session_state.suggested_feeders = ai_suggest_feeders(industry, city_prefill)
        st.success(f"Suggested {len(st.session_state.suggested_feeders)} feeder categories.")

    # Show suggestions as chips for visibility
    if st.session_state.suggested_keywords:
        st.markdown("**Suggested keywords:** " + " ".join([f"<span class='chip'>{k}</span>" for k in st.session_state.suggested_keywords]), unsafe_allow_html=True)
    if st.session_state.suggested_subs:
        st.markdown("**Suggested subreddits:** " + " ".join([f"<span class='chip'>{s}</span>" for s in st.session_state.suggested_subs]), unsafe_allow_html=True)

    with st.expander("What this does", expanded=True):
        st.markdown(
            "- **Google Trends** rising queries around your niche\n"
            "- **Reddit** hot/top threads for topic hooks (real-only; AI summary if blocked)\n"
            "- **YouTube** fresh videos for zeitgeist\n"
            "- An **AI market summary** and salesy post ideas\n"
        )
    st.selectbox("Reddit ranking", ["hot", "top"], index=0, key="reddit_mode")

    # Form with proper submit button
    trend_form = st.form(key="trend_form", clear_on_submit=False)
    with trend_form:
        # Seed inputs with suggestions if present
        default_kw = ", ".join(st.session_state.suggested_keywords[:8]) or "real estate, mortgage, school districts"
        default_subs = ", ".join(st.session_state.suggested_subs[:6]) or "RealEstate, Austin, personalfinance"

        # Use session_state to avoid overwriting the user's manual edits repeatedly
        if "trend_niche" not in st.session_state:
            st.session_state["trend_niche"] = default_kw
        if "trend_subs" not in st.session_state:
            st.session_state["trend_subs"] = default_subs

        niche = st.text_input("Niche keywords (comma-separated)", st.session_state["trend_niche"], key="trend_niche")
        subs = st.text_input("Reddit subs (comma-separated)", st.session_state["trend_subs"], key="trend_subs")

        timeframe = st.selectbox("Google Trends timeframe", ["now 7-d", "now 1-d", "now 30-d", "today 3-m"], index=0)
        submitted = trend_form.form_submit_button("Fetch trends")

    # Run data pulls
    if submitted:
        keywords = [s.strip() for s in st.session_state["trend_niche"].split(",") if s.strip()]
        sub_list = [s.strip().replace("r/", "") for s in st.session_state["trend_subs"].split(",") if s.strip()]
        data = gather_trends(
            niche_keywords=keywords, city=city_prefill, state=state_prefill, subs=sub_list,
            timeframe=timeframe, youtube_api_key=YOUTUBE_API_KEY, reddit_mode=st.session_state.reddit_mode
        )
        st.session_state.trend_data_cache = data
        st.session_state.last_reddit_source = data.get("reddit", {}).get("source", "")

    data = st.session_state.trend_data_cache
    if not data:
        st.info("Enter your niche/city and click **Fetch trends**.")
    else:
        rising = pd.DataFrame(data.get("google_trends", {}).get("rising", []))
        st.markdown("### Google Trends — Rising Queries")
        if rising.empty:
            st.info("No rising queries or pytrends missing.")
        else:
            if rank_ai:
                items = rising[["keyword", "query", "value"]].fillna("").to_dict(orient="records")
                scores = ai_rank_opportunity(items, "trends", f"{industry} in {city_prefill}, {state_prefill}")
                rising = rising.copy()
                rising["AI_Opportunity"] = scores[:len(rising)]
                rising = rising.sort_values("AI_Opportunity", ascending=False)
            _safe_show_df(rising, ["keyword", "query", "value", "AI_Opportunity", "link"], use_container_width=True)

        st.markdown("### Reddit — Hot Posts")
        rd = data.get("reddit", {})
        posts = pd.DataFrame(rd.get("posts", []))
        if posts.empty or len([p for p in rd.get("posts", []) if p.get("title")]) == 0:
            st.warning("Reddit: real posts unavailable (API blocked or no data).")
            # AI fallback summary (CLEARLY LABELED)
            ai_summary = _reddit_ai_summary(
                subreddits=[s.strip() for s in st.session_state["trend_subs"].split(",") if s.strip()],
                mode=st.session_state.reddit_mode,
                keywords=[s.strip() for s in st.session_state["trend_niche"].split(",") if s.strip()],
                city=city_prefill, state=state_prefill,
            )
            st.markdown("#### Reddit AI fallback (insights, not real posts)")
            st.info(ai_summary)
        else:
            if rank_ai:
                items = posts[["title", "score", "subreddit"]].fillna("").to_dict(orient="records")
                scores = ai_rank_opportunity(items, "reddit", f"{industry} in {city_prefill}, {state_prefill}")
                posts = posts.copy()
                posts["AI_Opportunity"] = scores[:len(posts)]
                posts = posts.sort_values("AI_Opportunity", ascending=False)
            preferred = ["subreddit", "title", "score", "AI_Opportunity", "url", "error"]
            _safe_show_df(posts, preferred, use_container_width=True)

        st.markdown("### YouTube — Fresh Videos (optional)")
        yt = data.get("youtube", {})
        vids = pd.DataFrame(yt.get("items", []))
        if vids.empty:
            if yt.get("error"):
                st.caption(f"YouTube: {yt['error']}")
            else:
                st.caption("Add YOUTUBE_API_KEY to show videos.")
        else:
            if rank_ai:
                items = vids[["title", "channel", "publishedAt"]].fillna("").to_dict(orient="records")
                scores = ai_rank_opportunity(items, "youtube", f"{industry} in {city_prefill}, {state_prefill}")
                vids = vids.copy()
                vids["AI_Opportunity"] = scores[:len(vids)]
                vids = vids.sort_values("AI_Opportunity", ascending=False)
            _safe_show_df(vids, ["title", "channel", "publishedAt", "AI_Opportunity", "url"], use_container_width=True)

        st.markdown("### AI Market Summary (salesy)")
        sample = {
            "industry": industry,
            "city": city_prefill,
            "state": state_prefill,
            "trending_queries": rising.head(8).to_dict(orient="records") if not rising.empty else [],
            "reddit_source": st.session_state.last_reddit_source,
            "reddit_top": posts.head(8).to_dict(orient="records") if not posts.empty else [],
        }
        summary = llm(
            system="You are a sales-forward SMB strategist. Be punchy and opportunity-oriented.",
            prompt=(f"Summarize 5 bullets of what's trending for {industry} in {city_prefill}, {state_prefill}. "
                    f"Then propose 3 ride-the-wave post ideas with hooks + CTAs.\nData:\n{to_json(sample)}")
        ) or "Add OPENAI_API_KEY to enable AI-written summaries."
        st.info(summary)

        # Source integrity line
        st.caption(
            f"Source integrity — Google Trends ✅ · Reddit ({st.session_state.last_reddit_source or 'n/a'}) · "
            f"YouTube {'✅' if vids is not None else '—'}"
        )


# ===================== LEAD FINDER =====================
with tab2:
    st.subheader("Lead Finder — Nearby partners & feeder businesses")
    with st.expander("How this helps", expanded=True):
        st.markdown(
            "- We surface **feeder businesses** that meet your future customers first.\n"
            "- You get **phone, website, directions** to start a partnership.\n"
            "- We compute an **Actionability Score** and explain **why** each lead is hot.\n"
            "- Typical targets for real estate: *apartment complex, movers, mortgage broker, home builder*.\n"
        )

    # Suggest feeders (from Trend tab suggestions)
    if st.session_state.suggested_feeders:
        st.markdown("**AI-suggested feeder categories:** " + " ".join([f"<span class='chip'>{k}</span>" for k in st.session_state.suggested_feeders]), unsafe_allow_html=True)

    # Form with submit
    lead_form = st.form(key="lead_form", clear_on_submit=False)
    with lead_form:
        cat = st.text_input(
            "Place type / query",
            st.session_state.suggested_feeders[0] if st.session_state.suggested_feeders else "apartment complex",
            key="lead_cat"
        )
        city2 = st.text_input("City", "Katy", key="lead_city")
        state2 = st.text_input("State", "TX", key="lead_state")
        limit = st.slider("How many?", 5, 30, 12, key="lead_limit")
        go = lead_form.form_submit_button("Search")

    def actionability_score(row, query: str):
        score, reasons = 0, []
        r = float(row.get("Rating") or 0)
        n = int(row.get("Reviews") or 0)
        if r >= 4.4:
            score += 35; reasons.append("High rating (≥4.4)")
        elif r >= 4.0:
            score += 20; reasons.append("Solid rating (≥4.0)")
        if n >= 200:
            score += 35; reasons.append("Strong review volume (≥200)")
        elif n >= 50:
            score += 20; reasons.append("Decent review volume (≥50)")
        if "apartment" in query.lower():
            score += 20; reasons.append("Feeder: high tenant churn → frequent moves")
        if row.get("Website"):
            score += 10; reasons.append("Website present")
        if row.get("Phone"):
            score += 10; reasons.append("Phone present")
        return min(score, 100), reasons

    if go:
        base_df = search_places_optional(cat, city2, state2, limit=limit, api_key=GOOGLE_PLACES_API_KEY)
        st.session_state.lead_data_cache = base_df if base_df is not None else False

    df = st.session_state.lead_data_cache
    if df is None:
        st.info("Enter a query (e.g., **apartment complex**, **moving company**, **mortgage broker**, **home builder**).")
    elif df is False or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning("No results (check GOOGLE_PLACES_API_KEY or try a nearby city/another query).")
    else:
        scored = df.copy()
        scored["Score"] = 0
        scored["Why"] = ""
        for i, row in scored.iterrows():
            s, rs = actionability_score(row, st.session_state.get("lead_cat", ""))
            scored.at[i, "Score"] = int(s)
            scored.at[i, "Why"] = " · ".join(rs)

        c1, c2, c3 = st.columns(3)
        with c1: kpi("Shown", str(len(scored)))
        with c2: kpi("Avg rating", f"{scored['Rating'].mean():.2f}" if len(scored) else "–")
        with c3: kpi("Median reviews", f"{int(scored['Reviews'].median())}" if len(scored) else "–")

        dff = scored.reset_index(drop=True).copy()
        st.markdown("#### Ranked leads")
        show_cols = ["Name", "Score", "Why", "Rating", "Reviews", "Phone", "Website", "Address"]
        _safe_show_df(dff, show_cols, use_container_width=True)

        # Select + details panel
        if "Name" in dff.columns and len(dff) > 0:
            name_choice = st.selectbox("Select a business", list(dff["Name"].astype(str).unique()))
        else:
            st.warning("No 'Name' column in the results.")
            name_choice = None

        if name_choice:
            row = dff[dff["Name"].astype(str) == str(name_choice)].iloc[0].to_dict()
            st.markdown(f"##### 📍 {row.get('Name','')}")
            st.markdown(f"{row.get('Address','')}")
            st.markdown(
                f"- ⭐ **{row.get('Rating','—')}** ({row.get('Reviews','—')} reviews)\n"
                f"- 📞 {row.get('Phone','—')}\n"
                f"- 🌐 {row.get('Website','—')}\n"
                f"- 🧭 [Open in Google Maps]({row.get('MapsUrl','')})"
            )

        # map
        if MAPS_OK:
            st.markdown("#### Map")
            dfc = dff.dropna(subset=["Lat", "Lng"]).copy()
            if not dfc.empty:
                m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                for _, r in dfc.iterrows():
                    folium.Marker(
                        [r["Lat"], r["Lng"]],
                        tooltip=r["Name"],
                        popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews<br>"
                              f"<a href='{r['MapsUrl']}' target='_blank'>Open in Google Maps</a>",
                    ).add_to(m)
                st_folium(m, height=520)
            else:
                st.info("No coordinates to plot.")
        else:
            st.caption("Install folium + streamlit-folium to see the map.")

        st.download_button("⬇️ Export CSV", dff[show_cols].to_csv(index=False).encode("utf-8"),
                           "leads.csv", "text/csv")

        # One-click outreach
        st.markdown("#### One-click outreach")
        if name_choice:
            lead2 = dff[dff["Name"].astype(str) == str(name_choice)].iloc[0].to_dict()
            colA, colB = st.columns(2)
            with colA:
                if st.button("Draft email", use_container_width=True, key="btn_email"):
                    body = (llm(
                        system="You write sales-forward B2B outreach for local partnerships.",
                        prompt=(f"Draft a short email from a {st.session_state.get('out_persona','Local Professional')} "
                                f"to {lead2['Name']} ({lead2.get('Website','')}). "
                                f"Goal: propose a referral partnership. Include a clear CTA. "
                                f"Keep it 120–150 words. Make it salesy and specific to {st.session_state.get('lead_cat','')} in {city2}, {state2}.")
                    ) or
                    f"Hi {lead2['Name']} team,\n\nI’d love to explore a simple referral partnership. "
                    "We serve the same customers and can help each other win more business. "
                    "Could we schedule a 10-minute chat this week?\n\nBest,\n<Your Name>")
                    st.code(body)
            with colB:
                if st.button("Draft SMS", use_container_width=True, key="btn_sms"):
                    sms = (llm(
                        system="You write concise, salesy SMS for local B2B outreach.",
                        prompt=(f"Write a friendly 240-character text to {lead2['Name']} proposing a quick chat about referrals in {city2}. "
                                f"One clear CTA.")
                    ) or
                    f"Hi {lead2['Name']}! Quick idea: partner on referrals? 10-min chat this week?")
                    st.code(sms)

        # AutoGPT webhook (optional)
        if autogpt_url:
            if st.button("Arm AutoGPT: watch for new high-score leads", key="btn_autogpt"):
                payload = {"action": "lead_watch", "query": st.session_state.get("lead_cat",""), "city": city2, "state": state2, "threshold": 85}
                try:
                    r = requests.post(autogpt_url, json=payload, timeout=15)
                    st.success(f"Webhook sent: {r.status_code}")
                except Exception as e:
                    st.warning(f"Webhook failed: {e}")


# ===================== OUTREACH FACTORY =====================
with tab3:
    st.subheader("Outreach Factory — ready-to-send sequences")
    st.caption("AI-polished emails/SMS with fixed send dates. Export as TXT/DOCX.")

    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", st.session_state.get("out_persona", "Real Estate Agent"),
                                key="out_persona")
        target = st.text_input("Target audience", "Apartment complex managers in Katy, TX", key="out_target")
    with c2:
        tone = st.selectbox("Tone", ["Salesy", "Friendly", "Professional", "Urgent"], index=0, key="out_tone")
        touches = st.slider("Number of touches", 3, 6, 4, key="out_touches")

    # AI multi-step sequence (improved)
    if st.button("Generate AI Sequence", key="out_generate_ai"):
        base = [
            {"send_dt": str(dt.date.today()), "channel": "email", "objective": "intro + value", "cta": "quick call"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=2)), "channel": "sms", "objective": "light nudge", "cta": "reply yes"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=5)), "channel": "email", "objective": "case study or proof", "cta": "15-min slot"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=9)), "channel": "email", "objective": "last call / deadline", "cta": "book now"},
        ][:touches]

        prompt = (
            f"Create a {touches}-touch outreach sequence for a {persona} targeting {target}. "
            f"Tone: {tone} and sales-forward. Each step must include channel, subject (if email), and message body. "
            f"Return PLAIN TEXT. Steps metadata:\n{to_json(base)}"
        )
        polished = llm(prompt, system="You write high-converting outreach.", temp=0.5) or \
                   "\n".join([f"{s['send_dt']} • {s['channel']}: {s['objective']} (CTA: {s['cta']})" for s in base])

        st.markdown("### AI-Generated Sequence")
        st.markdown(polished)

        st.download_button("⬇️ Download TXT", polished.encode("utf-8"), "outreach.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                build_docx_bytes("Outreach Plan", polished),
                "outreach.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    st.divider()
    st.markdown("### Sales Copy Polisher")
    raw_copy = st.text_area("Paste any copy to improve (email, SMS, ad, landing section)", height=180)
    pol_tone = st.selectbox("Polish tone", ["Salesy", "Friendly", "Professional", "Luxury", "Urgent"], index=0)
    if st.button("Polish copy"):
        polished = ai_polish_copy(raw_copy, pol_tone)
        st.markdown("#### Polished")
        st.markdown(polished or "(no change)")


# ===================== WEEKLY REPORT =====================
with tab4:
    st.subheader("Weekly Output (PDF/DOCX-ready text)")
    st.caption("Combines Trend hooks + Leads + Outreach into a single narrative.")

    biz = st.text_input("Business type", "Real Estate Agent", key="report_biz")
    rcity = st.text_input("City for report", "Katy", key="report_city")
    date = st.date_input("Week of", dt.date.today(), key="report_date")

    if st.button("Build Weekly Report", key="report_build"):
        # Core report body (structured)
        report = textwrap.dedent(f"""
        # WavePilot Weekly Report — {biz}, {rcity}
        **Week of:** {date}

        ## 1) Lead Finder — where your next clients are
        - Pull feeder businesses (apartments, movers, mortgage brokers).
        - Sort by **Actionability Score**; call high-scoring leads first.
        - Use one-click outreach for fast wins.

        ## 2) Trend Rider — ride what's hot
        - Look for rising searches (Google Trends) & hot Reddit threads.
        - Publish 3 posts that hit the trending questions this week.

        ## 3) Outreach Factory — send this today
        - AI-generated multi-touch sequence ready (TXT/DOCX).
        - Keep it sales-forward, concrete, and localized.

        _WavePilot — AI Growth Team_
        """).strip()

        # Salesy executive summary
        exec_summary = llm(
            system="You are a revenue-first CMO. Be salesy and to-the-point.",
            prompt=(f"Write a short executive summary for {biz} in {rcity} that sells the plan above. "
                    f"End with 3 crisp focus actions for this week.")
        ) or "(AI summary unavailable)"

        st.markdown("### Executive Summary (salesy)")
        st.info(exec_summary)
        st.markdown("### Full Report (plain text)")
        st.text(report)

        st.download_button("⬇️ Download TXT", (exec_summary + "\n\n" + report).encode("utf-8"),
                           "weekly_report.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                build_docx_bytes("Weekly Report", exec_summary + "\n\n" + report),
                "weekly_report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ===================== PRO LAB (diagnostics + stubs) =====================
with tab5:
    st.subheader("Pro Lab — LangChain · LangGraph · CrewAI (optional)")

    # --- Diagnostics: see exactly what's missing ---
    with st.expander("Pro diagnostics", expanded=False):
        st.write({
            "LangChain core (LC_OK)": LC_OK,
            "LangChain OpenAI (LCO_OK)": LCO_OK,
            "LangGraph (LG_OK)": LG_OK,
            "CrewAI (CREW_OK)": CREW_OK,
            "OPENAI_API_KEY present": bool(OPENAI_API_KEY),
            "YOUTUBE_API_KEY present": bool(YOUTUBE_API_KEY),
            "GOOGLE_PLACES_API_KEY present": bool(GOOGLE_PLACES_API_KEY),
            "Last Reddit source": st.session_state.get("last_reddit_source", ""),
        })

    # LangChain Enricher
    st.markdown("### LangChain Enricher")
    if not (LC_OK and LCO_OK and OPENAI_API_KEY and use_langchain):
        st.warning("LangChain libraries or OPENAI_API_KEY missing.")
    else:
        leads_df = st.session_state.lead_data_cache
        if isinstance(leads_df, pd.DataFrame) and not leads_df.empty:
            top_n = st.slider("How many leads to enrich?", 3, 15, 5, key="lc_topn")
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
            prompt = ChatPromptTemplate.from_template(
                "Rate partner fit for referrals.\nLead: {lead}\nReturn JSON: fit(1-100), why, next_action."
            )
            enriched_rows = []
            for _, row in leads_df.head(top_n).iterrows():
                lead_json = row.to_dict()
                try:
                    msg = model.invoke(prompt.format_messages(lead=to_json(lead_json)))
                    txt = (msg.content or "").strip()
                    data = {}
                    try:
                        data = json.loads(txt)
                    except Exception:
                        start = txt.find("{"); end = txt.rfind("}")
                        if start != -1 and end != -1:
                            data = json.loads(txt[start:end+1])
                    enriched_rows.append({
                        "Name": row.get("Name"),
                        "LC_Fit": data.get("fit", 50),
                        "LC_Why": data.get("why", ""),
                        "LC_Next": data.get("next_action", ""),
                    })
                except Exception as e:
                    enriched_rows.append({"Name": row.get("Name"), "LC_Fit": 50, "LC_Why": f"error: {e}", "LC_Next": ""})
            st.dataframe(pd.DataFrame(enriched_rows), use_container_width=True)
        else:
            st.info("Run **Lead Finder** first so we have leads to enrich.")

    st.divider()

    # LangGraph Orchestrator
    st.markdown("### LangGraph Orchestrator")
    if not (LG_OK and LCO_OK and OPENAI_API_KEY and use_langgraph):
        st.warning("LangGraph libraries or OPENAI_API_KEY missing.")
    else:
        trend_data = st.session_state.trend_data_cache
        if not trend_data:
            st.info("Fetch trends in **Trend Rider** first.")
        else:
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

            try:
                g = StateGraph(dict)

                def decide(state: dict) -> dict:
                    rising = trend_data.get("google_trends", {}).get("rising", [])
                    hot = len([r for r in rising if (r or {}).get("value", 0) >= 100]) >= 3
                    state["intent"] = "content" if hot else "research"
                    return state

                def do_research(state: dict) -> dict:
                    msg = model.invoke(f"Summarize 5 bullet insights from:\n{to_json(trend_data)[:6000]}")
                    state["brief"] = (msg.content or "").strip()
                    return state

                def do_content(state: dict) -> dict:
                    msg = model.invoke(
                        "Draft 3 short post ideas with hooks + CTAs for a local business based on these trends:\n"
                        + to_json(trend_data)[:6000]
                    )
                    state["brief"] = (msg.content or "").strip()
                    return state

                g.add_node("decide", decide)
                g.add_node("research", do_research)
                g.add_node("content", do_content)
                g.set_entry_point("decide")

                def router(state: dict):
                    return state.get("intent", "research")

                g.add_conditional_edges("decide", router, {"research": "research", "content": "content"})
                g.add_edge("research", END)
                g.add_edge("content", END)

                app = g.compile()
                out = app.invoke({"intent": "", "brief": ""})
                st.info(f"Intent: **{out.get('intent','?')}**")
                st.markdown(out.get("brief", "(no output)"))

            except Exception as e:
                # Clean fallback if any LangGraph runtime error occurs
                rising = trend_data.get("google_trends", {}).get("rising", [])
                hot = len([r for r in rising if (r or {}).get("value", 0) >= 100]) >= 3
                intent = "content" if hot else "research"
                if intent == "research":
                    brief = llm(
                        f"Summarize 5 bullet insights from:\n{to_json(trend_data)[:6000]}",
                        system="You are a concise SMB strategist.",
                    ) or "(LLM unavailable)"
                else:
                    brief = llm(
                        "Draft 3 short post ideas with hooks + CTAs for a local business based on these trends:\n"
                        + to_json(trend_data)[:6000],
                        system="You write punchy social content.",
                    ) or "(LLM unavailable)"
                st.warning(f"LangGraph fallback used due to runtime error: {e}")
                st.info(f"Intent: **{intent}**")
                st.markdown(brief)

    st.divider()

    # CrewAI Growth Crew
    st.markdown("### CrewAI Growth Crew")
    if not (CREW_OK and OPENAI_API_KEY and use_crewai):
        st.warning("CrewAI not installed or OPENAI_API_KEY missing.")
    else:
        biz = st.text_input("Business", "Real Estate Agent", key="crew_biz")
        niche = st.text_input("Niche focus", "Relocation in Katy, TX", key="crew_niche")
        if st.button("Run Growth Crew", key="crew_run"):
            researcher = Agent(
                role="Market Researcher",
                goal="Find insights & angles that will convert for a local SMB.",
                backstory="You love concise facts and actionable takeaways.",
                allow_delegation=False,
                verbose=False,
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY),
            )
            writer = Agent(
                role="Copywriter",
                goal="Write a crisp one-pager a business owner can use today.",
                backstory="You are crisp, persuasive, and practical.",
                allow_delegation=False,
                verbose=False,
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY),
            )
            t1 = Task(
                description=f"Summarize 5 bullet insights for {biz} ({niche}). Provide sources if relevant.",
                agent=researcher,
                expected_output="A JSON or bullet list with 5 concise insights and optional source links."
            )
            t2 = Task(
                description="Write a 1-page brief with headline, 3 bullets, 1 CTA. Make it print-friendly.",
                agent=writer,
                expected_output="A markdown one-pager: headline, three bullets, and a clear CTA."
            )
            crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
            try:
                result = crew.kickoff()
            except Exception as e:
                result = f"(CrewAI runtime error: {e})"
            st.markdown("#### One-pager")
            st.markdown(str(result))

# ===================== COMPETITOR SNIFFER =====================
with tab6:
    st.subheader("Competitor Sniffer — see what's resonating")
    c1, c2 = st.columns(2)
    comp = c1.text_input("Competitor / Brand name", "Zillow")
    comp_city = c2.text_input("City (optional)", "Katy")
    if st.button("Analyze competitor"):
        # Use YouTube + Reddit to fetch surface signals
        yt = youtube_search(YOUTUBE_API_KEY, f"{comp} {comp_city}" if comp_city else comp, max_results=8)
        rd = reddit_hot_or_top([comp.replace(" ", "")], mode="top", limit=8)
        vids = pd.DataFrame(yt.get("items", []))
        posts = pd.DataFrame(rd.get("posts", []))

        st.markdown("#### YouTube (recent)")
        _safe_show_df(vids, ["title", "channel", "publishedAt", "url"], use_container_width=True)
        st.markdown("#### Reddit (last week)")
        _safe_show_df(posts, ["subreddit", "title", "score", "url", "error"], use_container_width=True)

        # SWOT-style AI analysis
        swot = llm(
            system="You are a sharp GTM strategist. Be sales-forward and tactical.",
            prompt=(
                f"Analyze {comp} (context city: {comp_city}). Based on the YouTube titles and Reddit headlines below, "
                f"write a mini SWOT and 5 actionable opportunities for our client to exploit this week.\n\n"
                f"YouTube:\n{to_json(vids.to_dict(orient='records') if not vids.empty else [])}\n\n"
                f"Reddit:\n{to_json(posts.to_dict(orient='records') if not posts.empty else [])}"
            ),
            temp=0.5
        ) or "(AI analysis unavailable)"
        st.markdown("#### AI SWOT + Opportunities")
        st.info(swot)


# If OPENAI_API_KEY is missing, show a gentle hint (does not stop the app)
if client is None and not OPENAI_API_KEY:
    st.caption("Set OPENAI_API_KEY in your environment to enable AI features.")
