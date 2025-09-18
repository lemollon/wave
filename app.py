# app.py — WavePilot v0.2
# Front end that shows:
# 1) Trend Rider (Google Trends + Reddit) — free sources
# 2) Lead Finder (optional Google Places)
# 3) Outreach Factory (LangChain prompts; OpenAI optional)
# 4) Weekly Report export (TXT/DOCX)
#
# Notes:
# - Every widget has a unique key to avoid StreamlitDuplicateElementId errors.
# - Graceful fallbacks if APIs/keys are missing.
# - No JSON surfaces in the UI; AI output is plain text.

import os, io, json, textwrap, datetime as dt
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import altair as alt

# ---- optional libs ----
DOCX_OK = True
try:
    from docx import Document
except Exception:
    DOCX_OK = False

try:
    import folium
    from streamlit_folium import st_folium
    MAPS_OK = True
except Exception:
    MAPS_OK = False

# OpenAI optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- local tools ----
from tools.trends import gather_trends
from tools.places import search_places_optional


# ================= Helpers =================
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

def llm_ok() -> bool:
    return client is not None

def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4) -> str:
    """Small wrapper around OpenAI. Returns empty string on failure."""
    if not llm_ok():
        return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temp,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(AI unavailable: {e})"

def build_docx_bytes(title: str, body_text: str) -> bytes:
    """Create a DOCX (no JSON formatting)."""
    if not DOCX_OK:
        return b""
    doc = Document()
    doc.add_heading(title, level=1)
    for para in body_text.split("\n\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()

def inject_css():
    st.markdown("""
    <style>
      .kpi-card {border:1px solid #E5E7EB;border-radius:12px;padding:12px 16px;background:#F8FAFC}
      .kpi-label {font-size:.85rem;opacity:.85}
      .kpi-value {font-weight:800;font-size:1.2rem;margin-top:2px;word-break:break-word}
      .helpbox {background:#0f172a0d;border:1px solid #33415526;border-radius:10px;padding:10px 12px;margin:.25rem 0}
    </style>
    """, unsafe_allow_html=True)

def kpi(label: str, value: str):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def explain(title: str, body: str):
    st.markdown(f"#### {title}")
    st.markdown(f"<div class='helpbox'>{body}</div>", unsafe_allow_html=True)


# ================= Page config =================
st.set_page_config(page_title="WavePilot — AI Growth Team", page_icon="🌊", layout="wide")
inject_css()

st.title("🌊 WavePilot — AI Growth Team for Local Businesses")
st.caption("Uses **LangChain tools**, **CrewAI roles** (optional service), and an optional **AutoGPT-style watcher**. "
           "Free trend intel via Google Trends + Reddit. OpenAI is optional for nicer copy.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report"
])

# ================= TREND RIDER =================
with tab1:
    st.subheader("Trend Rider — Ride what's hot (free data)")
    explain("How this helps",
            "We look at **Google search spikes** and **hot Reddit posts** in your niche and city. "
            "You get ready-to-post hooks tied to what people care about *this week*.")

    with st.form("trend_form"):
        niche = st.text_input("Niche keywords (comma-separated)",
                              "real estate, mortgage, school districts",
                              key="trend_niche")
        city  = st.text_input("City", "Austin", key="trend_city")
        state = st.text_input("State", "TX", key="trend_state")
        subs  = st.text_input("Reddit subs (comma-separated)",
                              "Austin, personalfinance, RealEstate",
                              key="trend_subs")
        timeframe = st.selectbox("Google Trends timeframe",
                                 ["now 7-d","now 1-d","now 30-d","today 3-m"],
                                 index=0, key="trend_timeframe")
        submitted = st.form_submit_button("Fetch trends", key="trend_submit")

    if submitted:
        keywords = [s.strip() for s in niche.split(",") if s.strip()]
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]

        data = gather_trends(
            niche_keywords=keywords,
            city=city, state=state, subs=sub_list,
            timeframe=timeframe,
            youtube_api_key=_env("YOUTUBE_API_KEY","")
        )

        st.write("**Inputs**:", data.get("inputs", {}))

        # Google Trends
        gt = data.get("google_trends", {})
        rising = pd.DataFrame(gt.get("rising", []))
        st.markdown("### Google Trends — Rising Queries")
        if not rising.empty:
            st.dataframe(rising[["keyword","query","value","link"]],
                         use_container_width=True)
        else:
            st.info("No rising queries (or pytrends missing).")

        # Reddit
        rd = data.get("reddit", {})
        posts = pd.DataFrame(rd.get("posts", []))
        st.markdown("### Reddit — Hot Posts")
        if not posts.empty:
            st.dataframe(posts[["subreddit","title","score","url"]].head(20),
                         use_container_width=True)
        else:
            st.info(rd.get("error","No Reddit results"))

        # Optional YouTube
        yt = data.get("youtube", {})
        vids = pd.DataFrame(yt.get("items", []))
        st.markdown("### YouTube — Fresh Videos (optional)")
        if not vids.empty:
            st.dataframe(vids[["title","channel","publishedAt","url"]],
                         use_container_width=True)
        else:
            st.caption(yt.get("error","No YouTube results (key optional)."))

        # AI summary (plain text, never JSON)
        st.markdown("### AI Market Summary")
        sample = {
            "trending_queries": rising.head(8).to_dict(orient="records") if not rising.empty else [],
            "reddit_top": posts.head(8).to_dict(orient="records") if not posts.empty else []
        }
        summary = llm(
            system="You are a concise SMB strategist.",
            prompt=(
                f"Summarize in ~5 bullets what is trending for {city}, {state} in niche {keywords}. "
                f"Then propose 3 ride-the-wave post ideas. "
                f"Keep it plain text (no JSON, no code blocks). Data:\n{json.dumps(sample)}"
            )
        ) or "Add OPENAI_API_KEY in your env/secrets for AI summaries."
        st.info(summary)


# ================= LEAD FINDER =================
with tab2:
    st.subheader("Lead Finder — Nearby partners & feeder businesses")
    explain("How this helps",
            "We use Google Places (New) to find **feeder businesses**—partners that interact with your future customers "
            "(e.g., apartment complexes, movers, mortgage brokers for real estate). "
            "Partner with them for warm referrals.")

    with st.form("lead_form"):
        cat   = st.text_input("Place type / query", "apartment complex", key="lead_cat")
        city2 = st.text_input("City", "Austin", key="lead_city")
        state2= st.text_input("State", "TX", key="lead_state")
        limit = st.slider("How many?", 5, 30, 12, key="lead_limit")
        go    = st.form_submit_button("Search", key="lead_submit")

    if go:
        df = search_places_optional(
            cat, city2, state2, limit=limit,
            api_key=_env("GOOGLE_PLACES_API_KEY","")
        )

        if df is None:
            st.warning("No results — add GOOGLE_PLACES_API_KEY to your .env or Streamlit secrets.")
        elif df.empty:
            st.warning("No places found. Try a broader query or nearby city.")
        else:
            # KPIs
            c1, c2, c3 = st.columns(3)
            with c1: kpi("Shown", str(len(df)))
            with c2: kpi("Avg rating", f"{df['Rating'].mean():.2f}" if len(df) else "-")
            with c3: kpi("Median reviews", f"{int(df['Reviews'].median())}" if len(df) else "-")

            # Map (optional)
            if MAPS_OK:
                st.markdown("#### Map")
                dfc = df.dropna(subset=["Lat","Lng"]).copy()
                if not dfc.empty:
                    m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                    for _, r in dfc.iterrows():
                        folium.Marker(
                            [r["Lat"], r["Lng"]],
                            tooltip=r["Name"],
                            popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews"
                        ).add_to(m)
                    st_folium(m, height=500)
                else:
                    st.info("No coordinates to plot.")
            else:
                st.caption("Install folium + streamlit-folium to see the map.")

            st.markdown("#### Table")
            st.dataframe(df[["Name","Rating","Reviews","Address"]], use_container_width=True)


# ================= OUTREACH FACTORY =================
with tab3:
    st.subheader("Outreach Factory — Ready-to-send sequences")
    explain("How this helps",
            "We take your target and generate a **3-touch sequence** (email + SMS) you can paste into Gmail, SMS, or CRM. "
            "Exports available as TXT/DOCX.")

    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", "Real Estate Agent", key="out_persona")
        target  = st.text_input("Target audience", "Apartment complex managers in Austin, TX", key="out_target")
    with c2:
        tone    = st.selectbox("Tone", ["Friendly","Professional","Hype"], index=0, key="out_tone")
        touches = st.slider("Number of touches", 3, 6, 3, key="out_touches")

    if st.button("Generate Sequence", key="out_generate"):
        base = [
            {"send_dt": str(dt.date.today()), "channel": "email",
             "subject": "Quick hello 👋",
             "body": f"Hi there — I’m a {persona} in {target}. Could we collaborate?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=2)), "channel": "sms",
             "subject": "", "body": "Hey! Just checking in — open to a quick chat this week?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=7)), "channel": "email",
             "subject": "Ready when you are",
             "body": "Happy to help with referrals and co-marketing. What works?"}
        ][:touches]

        prompt = (
            f"Polish this outreach for a {persona} to contact {target}. "
            f"Keep the SAME dates/channels. Tone: {tone}. "
            f"Return plain text paragraphs only (no JSON, no code blocks).\n"
            f"Steps JSON:\n{json.dumps(base)}"
        )
        polished = llm(prompt, system="You write high-converting SMB outreach.") \
                   or "\n".join([f"{s['send_dt']} {s['channel']}: {s['body']}" for s in base])

        st.markdown("### AI-Polished Copy")
        st.markdown(polished)

        st.download_button("⬇️ Download TXT", polished.encode("utf-8"),
                           "outreach.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Outreach Plan", polished),
                               "outreach.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.caption("Install python-docx for DOCX export.")


# ================= WEEKLY REPORT =================
with tab4:
    st.subheader("Weekly Output (TXT/DOCX)")
    explain("What this is",
            "One page your client (or you) can use this week: where to get leads, what to post, and what to send.")

    biz  = st.text_input("Business type", "Real Estate Agent", key="report_biz")
    cityr = st.text_input("City for report", "Austin, TX", key="report_city")
    dater = st.date_input("Week of", dt.date.today(), key="report_date")

    if st.button("Build Weekly Report", key="report_build"):
        report = textwrap.dedent(f"""
        # WavePilot Weekly Report — {biz}, {cityr}
        **Week of:** {dater}

        ## 1) Lead Finder — where your next clients are
        - Use the Lead Finder tab to pull feeder businesses (apartments, movers, mortgage brokers).
        - Partner for referrals. High-churn = hot.

        ## 2) Trend Rider — ride what's hot right now
        - Use the Trend tab. Look for “school districts”, “rates”, “moving to {cityr}”.
        - Post 3x with local hooks + hashtags.

        ## 3) Outreach Factory — send this today
        - 3-touch sequence ready in the Outreach tab (TXT/DOCX).
        - Keep it friendly, concrete, and localized.

        _WavePilot — AI Growth Team_
        """).strip()

        st.text(report)
        st.download_button("⬇️ Download TXT", report.encode("utf-8"),
                           "weekly_report.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Weekly Report", report),
                               "weekly_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
