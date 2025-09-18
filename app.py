# app.py — WavePilot v0.1
# Front end that shows:
# 1) Trend Rider (Google Trends + Reddit) — free sources
# 2) Lead Finder (optional Google Places)
# 3) Outreach Factory (LangChain prompts; OpenAI optional)
#
# CrewAI/AutoGPT are wired via modules; app can call local crew service later.

import os, io, json, textwrap, datetime as dt
from typing import List, Dict

import streamlit as st
import pandas as pd
import altair as alt

# exports
DOCX_OK = True
try:
    from docx import Document
except Exception:
    DOCX_OK = False

# optional maps/leads
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

from tools.trends import gather_trends
from tools.places import search_places_optional

# ------------- helpers -------------
def _env(name, default=""):
    return os.getenv(name, default)

OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

def llm_ok(): return client is not None

def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4) -> str:
    if not llm_ok(): return ""
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
        if para.strip():
            doc.add_paragraph(para.strip())
    buf = io.BytesIO(); doc.save(buf); buf.seek(0); return buf.read()

def inject_css():
    st.markdown("""
    <style>
    .kpi-card {border:1px solid #E5E7EB;border-radius:12px;padding:12px 16px;background:#F8FAFC}
    .kpi-label {font-size:.85rem;opacity:.85}
    .kpi-value {font-weight:800;font-size:1.2rem;margin-top:2px}
    </style>
    """, unsafe_allow_html=True)

def kpi(label, value):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------- page config -------------
st.set_page_config(page_title="WavePilot — AI Growth Team", page_icon="🌊", layout="wide")
inject_css()

st.title("🌊 WavePilot — AI Growth Team for Local Businesses")
st.caption("Uses **LangChain tools**, **CrewAI roles**, and an optional **AutoGPT watcher**. Free trend intel via Google Trends + Reddit.")

tab1, tab2, tab3, tab4 = st.tabs(["Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report"])

# ================= TREND RIDER =================
with tab1:
    st.subheader("Trend Rider — Ride what's hot (free data)")
    with st.form("trend_form"):
        niche = st.text_input("Niche keywords (comma-separated)", "real estate, mortgage, school districts")
        city  = st.text_input("City", "Austin")
        state = st.text_input("State", "TX")
        subs  = st.text_input("Reddit subs (comma-separated)", "Austin, personalfinance, RealEstate")
        timeframe = st.selectbox("Google Trends timeframe", ["now 7-d","now 1-d","now 30-d","today 3-m"], index=0)
        submitted = st.form_submit_button("Fetch trends")

    if submitted:
        keywords = [s.strip() for s in niche.split(",") if s.strip()]
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]
        data = gather_trends(
            niche_keywords=keywords, city=city, state=state, subs=sub_list,
            timeframe=timeframe, youtube_api_key=_env("YOUTUBE_API_KEY","")
        )

        st.write("**Inputs**:", data.get("inputs", {}))

        # Google Trends
        gt = data.get("google_trends", {})
        rising = pd.DataFrame(gt.get("rising", []))
        st.markdown("### Google Trends — Rising Queries")
        if not rising.empty:
            st.dataframe(rising[["keyword","query","value","link"]], use_container_width=True)
        else:
            st.info("No rising queries (or pytrends missing).")

        # Reddit
        rd = data.get("reddit", {})
        posts = pd.DataFrame(rd.get("posts", []))
        st.markdown("### Reddit — Hot Posts")
        if not posts.empty:
            st.dataframe(posts[["subreddit","title","score","url"]].head(20), use_container_width=True)
        else:
            st.info(rd.get("error","No Reddit results"))

        # Optional YouTube
        yt = data.get("youtube", {})
        vids = pd.DataFrame(yt.get("items", []))
        st.markdown("### YouTube — Fresh Videos (optional)")
        if not vids.empty:
            st.dataframe(vids[["title","channel","publishedAt","url"]], use_container_width=True)
        else:
            st.caption(yt.get("error","No YouTube results (key optional)."))

        # AI summary
        st.markdown("### AI Market Summary")
        sample = {
            "trending_queries": rising.head(8).to_dict(orient="records") if not rising.empty else [],
            "reddit_top": posts.head(8).to_dict(orient="records") if not posts.empty else []
        }
        summary = llm(
            system="You are a concise SMB strategist.",
            prompt=f"Summarize in ~5 bullets what is trending for {city}, {state} in niche {keywords}. "
                   f"Then propose 3 ride-the-wave post ideas. Data:\n{json.dumps(sample)}"
        ) or "Add OPENAI_API_KEY for AI summaries."
        st.info(summary)

# ================= LEAD FINDER =================
with tab2:
    st.subheader("Lead Finder — Nearby partners & feeder businesses")
    st.caption("Uses Google Places (New). Optional; you can run the app without setting the Places key.")

    with st.form("lead_form"):
        cat = st.text_input("Place type / query", "apartment complex")
        city2 = st.text_input("City", "Austin")
        state2 = st.text_input("State", "TX")
        limit = st.slider("How many?", 5, 30, 12)
        go = st.form_submit_button("Search")

    if go:
        df = search_places_optional(cat, city2, state2, limit=limit, api_key=_env("GOOGLE_PLACES_API_KEY",""))
        if df is None or df.empty:
            st.warning("No results (did you add GOOGLE_PLACES_API_KEY?).")
        else:
            # KPIs
            c1, c2, c3 = st.columns(3)
            with c1: kpi("Shown", str(len(df)))
            with c2: kpi("Avg rating", f"{df['Rating'].mean():.2f}")
            with c3: kpi("Median reviews", f"{int(df['Reviews'].median())}")

            # Map (optional)
            if MAPS_OK:
                st.markdown("#### Map")
                dfc = df.dropna(subset=["Lat","Lng"]).copy()
                if not dfc.empty:
                    m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                    for _, r in dfc.iterrows():
                        folium.Marker([r["Lat"], r["Lng"]],
                                      tooltip=r["Name"],
                                      popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews").add_to(m)
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
    st.caption("Uses LangChain prompts + OpenAI (optional). Exports to TXT/DOCX.")

    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", "Real Estate Agent")
        target  = st.text_input("Target audience", "Apartment complex managers in Austin, TX")
    with c2:
        tone    = st.selectbox("Tone", ["Friendly","Professional","Hype"], index=0)
        touches = st.slider("Number of touches", 3, 6, 3)

    if st.button("Generate Sequence"):
        base = [
            {"send_dt": str(dt.date.today()),             "channel": "email", "subject": "Quick hello 👋", "body": f"Hi there — I’m a {persona} in {target}. Could we collaborate?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=2)), "channel": "sms",   "subject": "",                "body": "Hey! Just checking in — open to a quick chat this week?"},
            {"send_dt": str(dt.date.today()+dt.timedelta(days=7)), "channel": "email", "subject": "Ready when you are", "body": "Happy to help with referrals and co-marketing. What works?"}
        ]
        base = base[:touches]

        prompt = (
            f"Polish this outreach for a {persona} to contact {target}. "
            f"Keep SAME dates/channels. Tone: {tone}. Return PLAIN TEXT, not JSON.\n"
            f"Steps JSON:\n{json.dumps(base)}"
        )
        polished = llm(prompt, system="You write high-converting SMB outreach.") or "\n".join([f"{s['send_dt']} {s['channel']}: {s['body']}" for s in base])

        st.markdown("### AI-Polished Copy")
        st.markdown(polished)

        st.download_button("⬇️ Download TXT", polished.encode("utf-8"), "outreach.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX", build_docx_bytes("Outreach Plan", polished),
                               "outreach.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.caption("Install python-docx for DOCX export.")

# ================= WEEKLY REPORT =================
with tab4:
    st.subheader("Weekly Output (PDF/DOCX-ready text)")
    st.caption("Combines Trend hooks + Leads + Outreach into a single narrative.")
    biz  = st.text_input("Business type", "Real Estate Agent")
    city = st.text_input("City for report", "Austin")
    date = st.date_input("Week of", dt.date.today())

    if st.button("Build Weekly Report"):
        report = textwrap.dedent(f"""
        # WavePilot Weekly Report — {biz}, {city}
        **Week of:** {date}

        ## 1) Lead Finder — where your next clients are
        - Use the Lead Finder tab to pull feeder businesses (apartments, movers, mortgage brokers).
        - Partner for referrals. High-churn = hot.

        ## 2) Trend Rider — ride what's hot right now
        - Use the Trend tab. Look for “school districts”, “rates”, “moving to {city}”.
        - Post 3x with local hooks + hashtags.

        ## 3) Outreach Factory — send this today
        - 3-touch sequence ready in the Outreach tab (TXT/DOCX).
        - Keep it friendly, concrete, and localized.

        _WavePilot — AI Growth Team_
        """).strip()
        st.text(report)
        st.download_button("⬇️ Download TXT", report.encode("utf-8"), "weekly_report.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX", build_docx_bytes("Weekly Report", report),
                               "weekly_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
