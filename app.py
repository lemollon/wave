# app.py — WavePilot (Streamlit front end)
# Works with: tools/trends.py, tools/places.py, services/crew_api.py
# No local setup required beyond Streamlit Cloud + Secrets.

from __future__ import annotations
import os, io, json, textwrap, datetime as dt
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import requests

# ----- Optional libs (guarded) -----
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

OPENAI_OK = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_OK = False

# ----- Local modules -----
from tools.trends import gather_trends
from tools.places import search_places_optional

# ==========================
# Helpers / configuration
# ==========================

def get_secret(name: str, default: str = "") -> str:
    # Read Streamlit secrets first, then env as a fallback
    try:
        v = st.secrets[name]
        if isinstance(v, (int, float)): v = str(v)
        return v.strip()
    except Exception:
        return os.getenv(name, default).strip()

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY", "")
GOOGLE_PLACES_API_KEY = get_secret("GOOGLE_PLACES_API_KEY", "")
REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = get_secret("REDDIT_USER_AGENT", "wavepilot/0.1 by user")
CREW_API_URL = get_secret("CREW_API_URL", "")  # e.g., https://wave-xxxx.onrender.com/run_crew

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OPENAI_OK) else None

def llm_on() -> bool: return client is not None

def llm(prompt: str, system: str = "You are a concise SMB growth strategist.", temp: float = 0.4) -> str:
    if not llm_on(): return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temp,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(AI unavailable: {e})"

def build_docx_bytes(title: str, body_md: str) -> bytes:
    if not DOCX_OK: return b""
    doc = Document()
    doc.add_heading(title, level=1)
    for para in body_md.split("\n\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    return buf.read()

def soft_caption(msg: str):
    st.markdown(f"<div style='opacity:.75;font-size:.9rem'>{msg}</div>", unsafe_allow_html=True)

# Session safe containers
if "trend_data" not in st.session_state: st.session_state.trend_data = None
if "leads_df" not in st.session_state: st.session_state.leads_df = None
if "outreach_txt" not in st.session_state: st.session_state.outreach_txt = ""
if "crew_last" not in st.session_state: st.session_state.crew_last = None

# Page config
st.set_page_config(page_title="WavePilot — AI Growth Team", page_icon="🌊", layout="wide")

# Header
st.title("🌊 WavePilot — AI Growth Team for Local Businesses")
st.caption(
    "Find **what’s trending**, discover **high-value local partners**, and ship **ready-to-send outreach**. "
    "Powered by free data sources (Google Trends, Reddit) + optional Google Places. "
    "Crew endpoint: "
    f"{'**set**' if CREW_API_URL else ':red[missing — add CREW_API_URL to Secrets]'}"
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report", "Crew Engine"
])

# ==========================
# Trend Rider
# ==========================
with tab1:
    st.subheader("Trend Rider — Ride what’s hot (free sources)")
    c1, c2, c3 = st.columns(3)
    with c1:
        niche = st.text_input("Niche keywords (comma-separated)", "real estate, mortgage, school districts")
    with c2:
        city = st.text_input("City", "Austin")
    with c3:
        state = st.text_input("State", "TX")

    c4, c5 = st.columns(2)
    with c4:
        subs = st.text_input("Reddit subs (comma-separated)", "Austin, personalfinance, RealEstate")
    with c5:
        timeframe = st.selectbox("Google Trends timeframe", ["now 7-d","now 1-d","now 30-d","today 3-m"], index=0)

    if st.button("Fetch trends"):
        keywords = [s.strip() for s in niche.split(",") if s.strip()]
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]
        st.session_state.trend_data = gather_trends(
            niche_keywords=keywords, city=city, state=state,
            subs=sub_list, timeframe=timeframe, youtube_api_key=YOUTUBE_API_KEY
        )

    data = st.session_state.trend_data
    if not data:
        soft_caption("Run a search to populate this section.")
    else:
        st.write("**Inputs:**", data.get("inputs", {}))

        # Google Trends
        st.markdown("### Google Trends — Rising Queries")
        rising = pd.DataFrame(data.get("google_trends", {}).get("rising", []))
        if not rising.empty:
            st.dataframe(rising[["keyword","query","value","link"]], use_container_width=True)
        else:
            st.info("No rising queries (pytrends may be missing or no spikes).")

        # Reddit
        st.markdown("### Reddit — Hot Posts")
        posts = pd.DataFrame(data.get("reddit", {}).get("posts", []))
        if not posts.empty:
            st.dataframe(posts[["subreddit","title","score","url"]].head(20), use_container_width=True)
        else:
            err = data.get("reddit", {}).get("error")
            st.info(err or "No Reddit results.")

        # YouTube (optional)
        st.markdown("### YouTube — Fresh Videos (optional)")
        vids = pd.DataFrame(data.get("youtube", {}).get("items", []))
        if not vids.empty:
            st.dataframe(vids[["title","channel","publishedAt","url"]], use_container_width=True)
        else:
            yerr = data.get("youtube", {}).get("error")
            st.caption(yerr or "No YouTube results (YOUTUBE_API_KEY optional).")

        # AI summary
        st.markdown("### AI Market Summary")
        sample = {
            "trending_queries": rising.head(8).to_dict("records") if not rising.empty else [],
            "reddit_top": posts.head(8).to_dict("records") if not posts.empty else []
        }
        summary = llm(
            system="You are a concise SMB strategist.",
            prompt=f"Summarize in ~5 bullets what is trending for {city}, {state} in niche {niche}. "
                   f"Then propose 3 ride-the-wave post ideas. Data:\n{json.dumps(sample)}"
        ) or "Add OPENAI_API_KEY in Secrets for AI summaries."
        st.info(summary)

# ==========================
# Lead Finder
# ==========================
with tab2:
    st.subheader("Lead Finder — Nearby partners & feeder businesses")
    st.caption("Uses **Google Places (New)**. If no key is set, this tab still loads without results.")
    c1, c2, c3 = st.columns(3)
    with c1:
        q = st.text_input("Place type / query", "apartment complex")
    with c2:
        city2 = st.text_input("City", "Austin")
    with c3:
        state2 = st.text_input("State", "TX")
    limit = st.slider("How many results?", 5, 30, 12)

    if st.button("Search places"):
        if not GOOGLE_PLACES_API_KEY:
            st.warning("Add GOOGLE_PLACES_API_KEY in Secrets to enable this search.")
        df = search_places_optional(q, city2, state2, limit=limit, api_key=GOOGLE_PLACES_API_KEY)
        st.session_state.leads_df = df

    df = st.session_state.leads_df
    if df is None or df.empty:
        soft_caption("No results yet. Try a query (or add GOOGLE_PLACES_API_KEY).")
    else:
        # KPIs
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Results", len(df))
        with c2:
            st.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
        with c3:
            st.metric("Median Reviews", int(df['Reviews'].median()))

        # Map
        st.markdown("#### Map")
        if MAPS_OK:
            dfc = df.dropna(subset=["Lat","Lng"])
            if not dfc.empty:
                m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                for _, r in dfc.iterrows():
                    folium.Marker(
                        [r["Lat"], r["Lng"]],
                        tooltip=r["Name"],
                        popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews"
                    ).add_to(m)
                st_folium(m, height=480, width=None)
            else:
                st.info("No coordinates to plot.")
        else:
            st.caption("Install folium + streamlit-folium to see a map.")

        # Table
        st.markdown("#### Table")
        st.dataframe(df[["Name","Rating","Reviews","Address"]], use_container_width=True)

# ==========================
# Outreach Factory
# ==========================
with tab3:
    st.subheader("Outreach Factory — Ready-to-send sequences")
    st.caption("Uses AI if OPENAI_API_KEY is set; otherwise falls back to a clean default.")
    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", "Real Estate Agent")
        target = st.text_input("Target audience", "Apartment complex managers in Austin, TX")
    with c2:
        tone = st.selectbox("Tone", ["Friendly","Professional","Hype"], index=0)
        touches = st.slider("Number of touches", 3, 6, 3)

    if st.button("Generate sequence"):
        base = [
            {"send_dt": str(dt.date.today()), "channel": "email", "subject": "Quick hello 👋",
             "body": f"Hi there — I’m a {persona} in {target}. Could we collaborate?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=2)), "channel": "sms", "subject": "",
             "body": "Hey! Just checking in — open to a quick chat this week?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=7)), "channel": "email",
             "subject": "Ready when you are", "body":"Happy to help with referrals and co-marketing. What works?"}
        ][:touches]

        prompt = (
            f"Polish this outreach for a {persona} to contact {target}. "
            f"Keep SAME dates/channels. Tone: {tone}. Return PLAIN TEXT, not JSON.\n"
            f"Steps JSON:\n{json.dumps(base)}"
        )
        polished = llm(prompt, system="You write high-converting SMB outreach.") \
                   or "\n".join([f"{s['send_dt']} [{s['channel']}]: {s.get('subject','')}\n{s['body']}" for s in base])

        st.session_state.outreach_txt = polished

    if st.session_state.outreach_txt:
        st.markdown("### AI-Polished Copy")
        st.markdown(st.session_state.outreach_txt)

        st.download_button("⬇️ Download TXT", st.session_state.outreach_txt.encode("utf-8"),
                           "outreach.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Outreach Plan", st.session_state.outreach_txt),
                               "outreach.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            soft_caption("Install `python-docx` to enable DOCX export.")

# ==========================
# Weekly Report
# ==========================
with tab4:
    st.subheader("Weekly Report — Combine trends, leads & outreach")
    biz = st.text_input("Business type", "Real Estate Agent")
    r_city = st.text_input("City for report", "Austin, TX")
    wk = st.date_input("Week of", dt.date.today())

    if st.button("Build report"):
        pieces = []
        pieces.append(f"# WavePilot Weekly Report — {biz}, {r_city}\n**Week of:** {wk}\n")

        # Trend summary from session (if exists)
        d = st.session_state.trend_data
        if d:
            rising = pd.DataFrame(d.get("google_trends", {}).get("rising", []))
            posts = pd.DataFrame(d.get("reddit", {}).get("posts", []))
            pieces.append("## Trend Rider — What’s hot")
            if not rising.empty:
                top = rising.head(5)[["keyword","query","value"]].to_markdown(index=False)
                pieces.append("**Google rising queries (top 5):**\n\n" + top)
            if not posts.empty:
                top_r = posts.head(5)[["subreddit","title","score"]].to_markdown(index=False)
                pieces.append("\n**Reddit hot topics (top 5):**\n\n" + top_r)
        else:
            pieces.append("## Trend Rider — (Run a trends search to populate)")

        # Leads from session (if exists)
        df = st.session_state.leads_df
        if df is not None and not df.empty:
            pieces.append("\n## Lead Finder — Local partners")
            sample = df.head(10)[["Name","Rating","Reviews","Address"]].to_markdown(index=False)
            pieces.append(sample)
        else:
            pieces.append("\n## Lead Finder — (Run a places search to populate)")

        # Outreach from session (if exists)
        if st.session_state.outreach_txt:
            pieces.append("\n## Outreach — Ready to send\n")
            pieces.append(st.session_state.outreach_txt)

        report_txt = "\n\n".join(pieces)
        st.text(report_txt)

        st.download_button("⬇️ Download TXT", report_txt.encode("utf-8"),
                           "weekly_report.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Weekly Report", report_txt),
                               "weekly_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ==========================
# Crew Engine (Render API)
# ==========================
with tab5:
    st.subheader("Crew Engine — Connect to your Render API")
    st.caption("This pings your FastAPI backend (services/crew_api.py). Add `CREW_API_URL` in Secrets.")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Check health"):
            if not CREW_API_URL:
                st.error("Missing CREW_API_URL in Secrets.")
            else:
                # infer base by stripping /run_crew
                base = CREW_API_URL.rsplit("/", 1)[0] if "/run_crew" in CREW_API_URL else CREW_API_URL
                try:
                    r = requests.get(base + "/healthz", timeout=20)
                    st.write("Status:", r.status_code, r.text)
                except Exception as e:
                    st.error(f"Health check failed: {e}")

    with c2:
        demo_niche = st.text_input("Niche", "Real Estate Agent")
        demo_city  = st.text_input("City", "Austin, TX")
        if st.button("Run Crew demo"):
            if not CREW_API_URL:
                st.error("Missing CREW_API_URL in Secrets.")
            else:
                # Build small payload using session data if available
                d = st.session_state.trend_data or {}
                leads_df = st.session_state.leads_df or pd.DataFrame()
                lead_list = leads_df.head(5).to_dict("records") if not leads_df.empty else []
                payload = {"niche": demo_niche, "city": demo_city, "trends": d, "leads": lead_list}
                try:
                    r = requests.post(CREW_API_URL, json=payload, timeout=60)
                    st.write("HTTP:", r.status_code)
                    out = r.json()
                    st.session_state.crew_last = out
                    st.json(out)
                except Exception as e:
                    st.error(f"Crew request failed: {e}")

    if st.session_state.crew_last:
        st.markdown("### Crew Output (summary)")
        data = st.session_state.crew_last.get("data", {})
        if data:
            st.markdown("- " + "\n- ".join(data.get("trend_summary", [])))
            if data.get("top_leads"):
                st.markdown("**Top leads (scored):**")
                st.dataframe(pd.DataFrame(data["top_leads"]), use_container_width=True)
            if data.get("content_plan"):
                st.markdown("**7-Day Content Plan:**")
                st.markdown("\n".join([f"- {s}" for s in data["content_plan"]]))
            if data.get("outreach_sequence"):
                st.markdown("**Outreach Sequence:**")
                st.markdown("\n".join([f"- {s['send_dt']} [{s['channel']}] {s.get('subject','')}" for s in data["outreach_sequence"]]))
        else:
            st.caption("No data field in response.")
