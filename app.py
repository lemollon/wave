# app.py — WavePilot v0.3 “Pro”
# Free features: Trends + Leads + Outreach + Weekly Report
# Pro (optional): LangChain/LangGraph orchestrator, CrewAI lead dossiers, AutoGPT webhooks

import os, io, json, textwrap, datetime as dt
from typing import Dict, List

import streamlit as st
import pandas as pd

# --- Optional libs (we degrade gracefully) ---
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

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Local modules
from tools.trends import gather_trends
from tools.places import search_places_optional

# Pro orchestrator (LangChain + LangGraph)
try:
    from lc.orchestrator import lc_lead_research, lc_playbook
    LC_OK = True
except Exception:
    LC_OK = False

# CrewAI growth crew (optional)
try:
    from crew.run import run_growth_crew
    CREW_OK = True
except Exception:
    CREW_OK = False

# ----------------- env & helpers -----------------
def _env(k: str, d: str = "") -> str:
    return os.getenv(k, d)

OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

def llm_ok() -> bool:
    return client is not None

def llm(prompt: str, system: str = "You are a helpful marketer.", temp: float = 0.4) -> str:
    if not llm_ok(): return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=temp,
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
      :root { --card-bg:#0e1117; --card-border:#2b2f36; }
      .kpi-card {border:1px solid var(--card-border); border-radius:14px; padding:16px 18px; background:var(--card-bg);}
      .kpi-label {font-size:.80rem; opacity:.85; letter-spacing:.2px}
      .kpi-value {font-weight:800; font-size:1.25rem; margin-top:4px}
      h2, h3, h4 { letter-spacing:.2px }
      .stDataFrame { border:1px solid var(--card-border); border-radius:12px; }
      .stAlert { border-radius:12px; }
      .stButton>button { border-radius:10px; padding:.5rem .9rem; font-weight:600 }
    </style>
    """, unsafe_allow_html=True)

def kpi(label: str, value: str):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Back-compat wrapper (if your trends.gather_trends lacks reddit_mode)
def gather_trends_safe(**kwargs) -> Dict:
    try:
        return gather_trends(**kwargs)
    except TypeError:
        kwargs.pop("reddit_mode", None)
        return gather_trends(**kwargs)

# ------------- Streamlit page/state -------------
st.set_page_config(page_title="WavePilot — AI Growth Team", page_icon="🌊", layout="wide")
inject_css()

st.session_state.setdefault("trend_data_cache", None)
st.session_state.setdefault("lead_data_cache", None)
st.session_state.setdefault("reddit_mode", "hot")
st.session_state.setdefault("out_persona", "Local Professional")

st.title("🌊 WavePilot — AI Growth Team for Local Businesses")
st.caption("Free trend intel (Google Trends + Reddit), actionable local leads (Google Places), AI outreach, and optional Pro automations.")

pro_enabled = st.sidebar.toggle("Enable Pro Orchestrator (LangChain/LangGraph)", value=False)
if pro_enabled and not LC_OK:
    st.sidebar.warning("LangChain/LangGraph not installed yet. Install requirements and redeploy.")
crew_enabled = st.sidebar.toggle("Enable Growth Crew (CrewAI)", value=False)
if crew_enabled and not CREW_OK:
    st.sidebar.warning("CrewAI not installed. Install requirements and redeploy.")
autogpt_url = st.sidebar.text_input("AutoGPT Webhook URL (optional)", _env("AUTOGPT_URL",""))

tab1, tab2, tab3, tab4 = st.tabs(["Trend Rider", "Lead Finder", "Outreach Factory", "Weekly Report"])

# ================= TREND RIDER =================
with tab1:
    st.subheader("Trend Rider — ride what's hot (free data)")
    with st.expander("How this helps", expanded=False):
        st.write("Spot rising topics and hot discussions in your niche & city so your content rides current demand.")

    show_debug = st.toggle("Show debug JSON (advanced)", value=False, key="trend_debug")
    st.selectbox("Reddit ranking", ["hot","top"], index=0, key="reddit_mode",
                 help="If you see 401, double-check Reddit keys & USER_AGENT in secrets.")

    with st.form("trend_form"):
        niche = st.text_input("Niche keywords (comma-separated)",
                              "real estate, mortgage, school districts", key="trend_niche")
        city  = st.text_input("City", "Katy", key="trend_city")
        state = st.text_input("State", "TX", key="trend_state")
        subs  = st.text_input("Reddit subs (comma-separated)",
                              "RealEstate, Austin, personalfinance", key="trend_subs")
        timeframe = st.selectbox("Google Trends timeframe",
                                 ["now 7-d","now 1-d","now 30-d","today 3-m"], index=0, key="trend_timeframe")
        submitted = st.form_submit_button("Fetch trends", key="trend_submit")

    if submitted:
        keywords = [s.strip() for s in niche.split(",") if s.strip()]
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]
        data = gather_trends_safe(
            niche_keywords=keywords, city=city, state=state, subs=sub_list,
            timeframe=timeframe, youtube_api_key=_env("YOUTUBE_API_KEY",""),
            reddit_mode=st.session_state.reddit_mode,  # ignored by older trends.py
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
        st.markdown(
            "- We surface **feeder businesses** that meet your future customers first.\n"
            "- You get **phone, website, and directions** to start a partnership.\n"
            "- We compute an **Actionability Score** and explain **why** each lead is hot.\n"
            "- **Pro Orchestrator** can enrich & score leads with LangChain (optional)."
        )

    with st.form("lead_form"):
        cat   = st.text_input("Place type / query", "apartment complex", key="lead_cat")
        city2 = st.text_input("City", "Katy", key="lead_city")
        state2= st.text_input("State", "TX", key="lead_state")
        limit = st.slider("How many?", 5, 30, 12, key="lead_limit")
        go    = st.form_submit_button("Search", key="lead_submit")

    def actionability_score(row, query: str):
        score, reasons = 0, []
        r = float(row.get("Rating") or 0)
        n = int(row.get("Reviews") or 0)
        if r >= 4.4: score += 35; reasons.append("High rating (≥4.4)")
        elif r >= 4.0: score += 20; reasons.append("Solid rating (≥4.0)")
        if n >= 200: score += 35; reasons.append("Strong review volume (≥200)")
        elif n >= 50: score += 20; reasons.append("Decent review volume (≥50)")
        if "apartment" in query.lower(): score += 20; reasons.append("Feeder: high tenant churn → frequent moves")
        if row.get("Website"): score += 10; reasons.append("Website present")
        if row.get("Phone"): score += 10; reasons.append("Phone present")
        return min(score, 100), reasons

    if go:
        base_df = search_places_optional(cat, city2, state2, limit=limit, api_key=_env("GOOGLE_PLACES_API_KEY",""))
        st.session_state.lead_data_cache = base_df if base_df is not None else False

    df = st.session_state.lead_data_cache
    if df is None:
        st.info("Enter a query (e.g., **apartment complex**, **moving company**, **mortgage broker**, **home builder**).")
    elif df is False or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning("No results (check your GOOGLE_PLACES_API_KEY or try a nearby city/another query).")
    else:
        # Local scoring
        scored = df.copy()
        scored["Score"] = 0
        scored["Why"] = ""
        for i, row in scored.iterrows():
            s, rs = actionability_score(row, cat)
            scored.at[i, "Score"] = int(s)
            scored.at[i, "Why"] = " · ".join(rs)

        # Pro enrichment (LangChain/LangGraph)
        if pro_enabled and LC_OK and st.toggle("Use Pro Orchestrator (enrich with LangChain)", value=False, key="lc_enrich"):
            try:
                enriched = lc_lead_research(scored, city2, state2)
                if isinstance(enriched, pd.DataFrame) and not enriched.empty:
                    scored = enriched
                    st.success("Pro enrichment complete (LangChain).")
            except Exception as e:
                st.warning(f"Pro enrichment failed: {e}")

        c1, c2, c3 = st.columns(3)
        with c1: kpi("Shown", str(len(scored)))
        with c2: kpi("Avg rating", f"{scored['Rating'].mean():.2f}" if len(scored) else "–")
        with c3: kpi("Median reviews", f"{int(scored['Reviews'].median())}" if len(scored) else "–")

        if MAPS_OK:
            st.markdown("#### Map")
            dfc = scored.dropna(subset=["Lat","Lng"]).copy()
            if not dfc.empty:
                m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12)
                for _, r in dfc.iterrows():
                    folium.Marker(
                        [r["Lat"], r["Lng"]],
                        tooltip=r["Name"],
                        popup=f"<b>{r['Name']}</b><br>{r['Address']}<br>{r['Rating']} ⭐ / {r['Reviews']} reviews<br>"
                              f"<a href='{r['MapsUrl']}' target='_blank'>Open in Google Maps</a>",
                    ).add_to(m)
                st_folium(m, height=500)
            else:
                st.info("No coordinates to plot.")
        else:
            st.caption("Install folium + streamlit-folium to see the map.")

        st.markdown("#### Ranked leads")
        show_cols = ["Name","Score","Why","Rating","Reviews","Phone","Website","Address"]
        st.dataframe(scored[show_cols], use_container_width=True)
        st.download_button("⬇️ Export CSV", scored.to_csv(index=False).encode("utf-8"),
                           "leads.csv", "text/csv")

        # CrewAI dossiers
        if crew_enabled and CREW_OK and st.toggle("Draft Growth Crew dossier for selected lead", value=False, key="crew_dossier"):
            pick = st.selectbox("Choose a lead", scored["Name"].tolist(), key="crew_pick")
            lead = scored[scored["Name"] == pick].iloc[0].to_dict()
            try:
                dossier = run_growth_crew(lead, persona=st.session_state.get("out_persona","Local Professional"))
                st.markdown("##### Dossier")
                st.markdown(dossier)
            except Exception as e:
                st.warning(f"CrewAI failed: {e}")

        # One-click outreach (AI-polished)
        st.markdown("#### One-click outreach")
        pick2 = st.selectbox("Choose a lead", scored["Name"].tolist(), key="lead_pick")
        lead2 = scored[scored["Name"] == pick2].iloc[0].to_dict()
        colA, colB = st.columns(2)
        with colA:
            if st.button("Draft email", use_container_width=True, key="btn_email"):
                body = (llm(
                    system="You write friendly B2B outreach for local partnerships.",
                    prompt=(f"Draft a short email from a {st.session_state.out_persona} "
                            f"to {lead2['Name']} ({lead2.get('Website','')}). "
                            f"Goal: propose a referral partnership. Include a clear CTA. "
                            f"Keep it 120–150 words.")
                ) or
                f"Hi {lead2['Name']} team,\n\nI’d love to explore a simple referral partnership. "
                "We serve the same customers and can help each other win more business. "
                "Could we schedule a 10-minute chat this week?\n\nBest,\n<Your Name>")
                st.code(body)
        with colB:
            if st.button("Draft SMS", use_container_width=True, key="btn_sms"):
                sms = (llm(
                    system="You write concise SMS for local B2B outreach.",
                    prompt=(f"Write a friendly 240-character text to {lead2['Name']} proposing a quick chat about referrals. "
                            f"One clear CTA.")
                ) or
                f"Hi {lead2['Name']}! Quick idea: partner on referrals? 10-min chat this week?")
                st.code(sms)

        st.markdown(
            f"- 📞 **Phone:** {lead2.get('Phone','—')}  \n"
            f"- 🌐 **Website:** {('['+lead2['Website']+']('+lead2['Website']+')') if lead2.get('Website') else '—'}  \n"
            f"- 🗺️ **Maps:** {('['+lead2.get('Address','Maps')+']('+lead2.get('MapsUrl','')+')') if lead2.get('MapsUrl') else '—'}"
        )

        # AutoGPT webhook (optional)
        if autogpt_url:
            if st.button("Arm AutoGPT: watch for new high-score leads", key="btn_autogpt"):
                import requests
                payload = {"action":"lead_watch","query":cat,"city":city2,"state":state2,"threshold":85}
                try:
                    r = requests.post(autogpt_url, json=payload, timeout=15)
                    st.success(f"Webhook sent: {r.status_code}")
                except Exception as e:
                    st.warning(f"Webhook failed: {e}")

# ================= OUTREACH FACTORY =================
with tab3:
    st.subheader("Outreach Factory — ready-to-send sequences")
    st.caption("AI-polished emails/SMS with fixed send dates. Export as TXT/DOCX.")

    c1, c2 = st.columns(2)
    with c1:
        persona = st.text_input("Your business type", st.session_state.get("out_persona","Real Estate Agent"), key="out_persona")
        target  = st.text_input("Target audience", "Apartment complex managers in Katy, TX", key="out_target")
    with c2:
        tone    = st.selectbox("Tone", ["Friendly","Professional","Hype"], index=0, key="out_tone")
        touches = st.slider("Number of touches", 3, 6, 3, key="out_touches")

    # Pro playbook (LangChain) to suggest angles
    if pro_enabled and LC_OK and st.toggle("Get AI Playbook (LangChain)", value=True, key="playbook_toggle"):
        try:
            playbook = lc_playbook(persona, target)
            st.markdown("##### AI Playbook")
            st.markdown(playbook)
        except Exception as e:
            st.warning(f"Playbook generation failed: {e}")

    if st.button("Generate Sequence", key="out_generate"):
        base = [
            {"send_dt": str(dt.date.today()),                       "channel": "email", "subject": "Quick hello 👋",
             "body": f"Hi there — I’m a {persona} in {target}. Could we collaborate?"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=2)),"channel": "sms",   "subject": "",
             "body": "Hey! Just checking in — open to a quick chat this week?"},
            {"send_dt": str(dt.date.today() + dt.timedelta(days=7)),"channel": "email", "subject": "Ready when you are",
             "body": "Happy to help with referrals and co-marketing. What works?"}
        ]
        base = base[:touches]

        prompt = (
            f"Polish this outreach for a {persona} to contact {target}. "
            f"Keep SAME dates and channels. Tone: {tone}. Return PLAIN TEXT (not JSON). "
            f"Here are the steps as JSON for reference:\n{json.dumps(base)}"
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
        - Pull feeder businesses (apartments, movers, mortgage brokers).
        - Sort by **Actionability Score**; call high-scoring leads first.
        - Use one-click outreach for fast wins.

        ## 2) Trend Rider — ride what's hot
        - Look for rising searches (Google Trends) & hot Reddit threads.
        - Publish 3 posts that hit the trending questions this week.

        ## 3) Outreach Factory — send this today
        - 3-touch sequence ready (TXT/DOCX).
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
