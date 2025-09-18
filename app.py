# app.py — WavePilot v0.1.2
# Tabs:
# 1) Trend Rider (Google Trends + Reddit + optional YouTube)
# 2) Lead Finder (Google Places New) — map + table
# 3) Outreach Factory (AI-polished emails/SMS + TXT/DOCX export)
# 4) Weekly Report (TXT/DOCX export)
#
# Fixes:
# - Persist results with st.session_state (no flash-then-disappear)
# - Hide debug JSON by default (toggle to show)
# - Reddit handled gracefully
# - Unique keys for all widgets
# - NEW: Backward-compatible gather_trends_safe() so you don't need to update tools/trends.py

import os, io, json, textwrap, datetime as dt
from typing import Dict

import streamlit as st
import pandas as pd
import altair as alt

# Exports
DOCX_OK = True
try:
    from docx import Document
except Exception:
    DOCX_OK = False

# Map
try:
    import folium
    from streamlit_folium import st_folium
    MAPS_OK = True
except Exception:
    MAPS_OK = False

# OpenAI (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from tools.trends import gather_trends   # your existing file is fine
from tools.places import search_places_optional  # your existing file is fine

# ----------------- helpers -----------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

def llm_ok() -> bool:
    return client is not None

def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4) -> str:
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

def build_docx_bytes(title: str, body_md: str) -> bytes:
    if not DOCX_OK: return b""
    doc = Document(); doc.add_heading(title, level=1)
    for para in body_md.split("\n\n"):
        if para.strip(): doc.add_paragraph(para.strip())
    buf = io.BytesIO(); doc.save(buf); buf.seek(0); return buf.read()

def inject_css():
    st.markdown("""
    <style>
    .kpi-card {border:1px solid #2b2f36;border-radius:12px;padding:12px 16px;background:#0e1117}
    .kpi-label {font-size:.85rem;opacity:.85}
    .kpi-value {font-weight:800;font-size:1.2rem;margin-top:2px}
    </style>
    """, unsafe_allow_html=True)

def kpi(label: str, value: str):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---- backward-compatible wrapper for gather_trends ----
def gather_trends_safe(**kwargs) -> Dict:
    """
    Try calling tools.trends.gather_trends with all kwargs.
    If the installed function doesn't accept a kw (e.g., reddit_mode),
    retry without it so older deployments don't crash.
    """
    try:
        return gather_trends(**kwargs)
    except TypeError:
        kwargs.pop("reddit_mode", None)
        return gather_trends(**kwargs)

# ----------------- page + state -----------------
st.set_page_config(page_title="WavePilot — AI Growth Team", page_icon="🌊", layout="wide")
inject_css()

st.session_state.setdefault("trend_data_cache", None)
st.session_state.setdefault("lead_data_cache", None)
st.session_state.setdefault("reddit_mode", "hot")

st.title("🌊 WavePilot — AI Growth Team for Local Businesses")
st.caption("Free trend intel (Google Trends + Reddit), optional local leads (Google Places), AI outreach & exports.")

tab1, tab2, tab3, tab4 = st.tabs(["Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report"])

# ================= TREND RIDER =================
with tab1:
    st.subheader("Trend Rider — ride what's hot (free data)")
    with st.expander("How this helps", expanded=False):
        st.write("Find rising topics and hot discussions so your content rides current demand.")

    show_debug = st.toggle("Show debug JSON (advanced)", value=False, key="trend_debug")
    st.selectbox("Reddit ranking", ["hot","top"], index=0, key="reddit_mode")

    with st.form("trend_form"):
        niche = st.text_input("Niche keywords (comma-separated)",
                              "real estate, mortgage, school districts", key="trend_niche")
        city  = st.text_input("City", "Katy", key="trend_city")
        state = st.text_input("State", "TX", key="trend_state")
        subs  = st.text_input("Reddit subs (comma-separated)",
                              "RealEstate, Austin, personalfinance", key="trend_subs")
        timeframe = st.selectbox("Google Trends timeframe",
                                 ["now 7-d","now 1-d","now 30-d","today 3-m"],
                                 index=0, key="trend_timeframe")
        submitted = st.form_submit_button("Fetch trends", key="trend_submit")

    if submitted:
        keywords = [s.strip() for s in niche.split(",") if s.strip()]
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]
        data = gather_trends_safe(
            niche_keywords=keywords,
            city=city,
            state=state,
            subs=sub_list,
            timeframe=timeframe,
            youtube_api_key=_env("YOUTUBE_API_KEY",""),
            reddit_mode=st.session_state.reddit_mode,   # removed automatically if your trends.py doesn't support it
        )
        st.session_state.trend_data_cache = data

    data = st.session_state.trend_data_cache
    if not data:
        st.info("Enter your niche/city and click **Fetch trends**.")
    else:
        if show_debug:
            st.json(data.get("inputs", {}))

        gt = data.get("google_trends", {})
        rising = pd.DataFrame(gt.get("rising", []))
        st.markdown("### Google Trends — Rising Queries")
        if not rising.empty:
            st.dataframe(rising[["keyword","query","value","link"]], use_container_width=True)
        else:
            st.info(gt.get("error","No rising queries (or pytrends missing)."))

        st.markdown("### Reddit — Hot Posts")
        rd = data.get("reddit", {})
        if err := rd.get("error"):
            st.warning(f"Reddit: {err} • Check REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT.")
        posts = pd.DataFrame(rd.get("posts", []))
        if not posts.empty:
            st.dataframe(posts[["subreddit","title","score","url"]].head(20), use_container_width=True)
        else:
            if not err:
                st.info("No Reddit posts found for those subs.")

        st.markdown("### YouTube — Fresh Videos (optional)")
        yt = data.get("youtube", {})
        vids = pd.DataFrame(yt.get("items", []))
        if not vids.empty:
            st.dataframe(vids[["title","channel","publishedAt","url"]], use_container_width=True)
        else:
            if yterr := yt.get("error"):
                st.caption(f"YouTube: {yterr}")
            else:
                st.caption("Add YOUTUBE_API_KEY to show this section.")

        st.markdown("### AI Market Summary")
        sample = {
            "trending_queries": rising.head(8).to_dict(orient="records") if not rising.empty else [],
            "reddit_top": posts.head(8).to_dict(orient="records") if not posts.empty else [],
        }
        summary = llm(
            system="You are a concise SMB strategist.",
            prompt=(f"Summarize in ~5 bullets what is trending for {city}, {state} in niche {niche}. "
                    f"Then propose 3 ride-the-wave post ideas. Data:\n{json.dumps(sample)}")
        ) or "Add OPENAI_API_KEY to enable AI-written summaries."
        st.info(summary)

# ================= LEAD FINDER =================
with tab2:
    st.subheader("Lead Finder — Nearby partners & feeder businesses")
    with st.expander("How this helps", expanded=True):
        st.markdown("We use Google Places (New) to find **feeder businesses**—partners that interact with your future customers (e.g., apartment complexes, movers, mortgage brokers for real estate). Partner with them for warm referrals.")

    with st.form("lead_form"):
        cat   = st.text_input("Place type / query", "apartment complex", key="lead_cat")
        city2 = st.text_input("City", "Katy", key="lead_city")
        state2= st.text_input("State", "TX", key="lead_state")
        limit = st.slider("How many?", 5, 30, 12, key="lead_limit")
        go    = st.form_submit_button("Search", key="lead_submit")

    if go:
        df = search_places_optional(cat, city2, state2, limit=limit, api_key=_env("GOOGLE_PLACES_API_KEY",""))
        st.session_state.lead_data_cache = df if df is not None else False

    df = st.session_state.lead_data_cache
    if df is None:
        st.info("Enter a query (e.g., **apartment complex**, **moving company**, **mortgage broker**, **home builder**).")
    elif df is False or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning("No results (check your GOOGLE_PLACES_API_KEY or try a nearby city/another query).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Shown", str(len(df)))
        with c2: kpi("Avg rating", f"{df['Rating'].mean():.2f}" if len(df) else "–")
        with c3: kpi("Median reviews", f"{int(df['Reviews'].median())}" if len(df) else "–")

        if MAPS_OK:
            st.markdown("#### Map")
            dfc = df.dropna(subset=["Lat","Lng"]).copy()
            if not dfc.empty:
                m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                for _, r in dfc.iterrows():
                    folium.Marker(
                        [r["Lat"], r["Lng"]],
                        tooltip=r["Name"],
                        popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews",
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
    st.subheader("Outreach Factory — ready-to-send sequences")
    st.caption("AI-polished emails/SMS with fixed send dates. Export as TXT/DOCX.")

    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", "Real Estate Agent", key="out_persona")
        target  = st.text_input("Target audience", "Apartment complex managers in Katy, TX", key="out_target")
    with c2:
        tone    = st.selectbox("Tone", ["Friendly","Professional","Hype"], index=0, key="out_tone")
        touches = st.slider("Number of touches", 3, 6, 3, key="out_touches")

    if st.button("Generate Sequence", key="out_generate"):
        base = [
            {"send_dt": str(dt.date.today()),                       "channel": "email", "subject": "Quick hello 👋", "body": f"Hi there — I’m a {persona} in {target}. Could we collaborate?"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=2)),"channel": "sms",   "subject": "",               "body": "Hey! Just checking in — open to a quick chat this week?"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=7)),"channel": "email", "subject": "Ready when you are", "body": "Happy to help with referrals and co-marketing. What works?"}
        ]
        base = base[:touches]

        prompt = (
            f"Polish this outreach for a {persona} to contact {target}. "
            f"Keep SAME dates and channels. Tone: {tone}. Return PLAIN TEXT (not JSON). "
            f"Here are the steps as JSON for your reference:\n{json.dumps(base)}"
        )
        polished = llm(prompt, system="You write high-converting SMB outreach.") or \
                   "\n".join([f"{s['send_dt']} • {s['channel']}: {s['subject']} {s['body']}".strip() for s in base])

        st.markdown("### AI-Polished Copy")
        st.markdown(polished)

        st.download_button("⬇️ Download TXT", polished.encode("utf-8"), "outreach.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Outreach Plan", polished),
                               "outreach.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.caption("Install `python-docx` for DOCX export.")

# ================= WEEKLY REPORT =================
with tab4:
    st.subheader("Weekly Output (PDF/DOCX-ready text)")
    st.caption("Combines Trend hooks + Leads + Outreach into a single narrative.")

    biz  = st.text_input("Business type", "Real Estate Agent", key="report_biz")
    rcity= st.text_input("City for report", "Katy", key="report_city")
    date = st.date_input("Week of", dt.date.today(), key="report_date")

    if st.button("Build Weekly Report", key="report_build"):
        report = textwrap.dedent(f"""
        # WavePilot Weekly Report — {biz}, {rcity}
        **Week of:** {date}

        ## 1) Lead Finder — where your next clients are
        - Use the Lead Finder tab to pull feeder businesses (apartments, movers, mortgage brokers).
        - Partner for referrals. High-churn = hot.

        ## 2) Trend Rider — ride what's hot right now
        - Use the Trend tab. Look for “school districts”, “rates”, “moving to {rcity}”.
        - Post 3x with local hooks + hashtags.

        ## 3) Outreach Factory — send this today
        - 3-touch sequence ready in the Outreach tab (TXT/DOCX).
        - Keep it friendly, concrete, and localized.

        _WavePilot — AI Growth Team_
        """).strip()

        st.text(report)
        st.download_button("⬇️ Download TXT", report.encode("utf-8"), "weekly_report.txt", "text/plain")
        if DOCX_OK:
            st.download_button("⬇️ Download DOCX",
                               build_docx_bytes("Weekly Report", report),
                               "weekly_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
