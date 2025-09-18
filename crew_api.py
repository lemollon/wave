# crew_api.py
# Start on Render with:
# uvicorn crew_api:app --host 0.0.0.0 --port $PORT

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "WavePilot Crew API"
APP_VERSION = "1.1.0"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crew_api")

# ---------------- FastAPI ----------------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Allow Streamlit or any frontend to call us (tighten origins if you have a fixed domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # e.g., ["https://yourapp.streamlit.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Schemas ----------------
class Lead(BaseModel):
    Name: Optional[str] = ""
    Rating: Optional[float] = None
    Reviews: Optional[int] = None
    Address: Optional[str] = ""
    Lat: Optional[float] = None
    Lng: Optional[float] = None

class CrewPayload(BaseModel):
    niche: str = Field(..., description="Business type (e.g., 'Real Estate Agent')")
    city: str = Field(..., description="City or 'City, ST' (e.g., 'Austin, TX')")
    trends: Dict[str, Any] = Field(default_factory=dict, description="Trends payload (Google/Reddit/YouTube)")
    leads: List[Lead] = Field(default_factory=list, description="Optional leads")

class CrewResponse(BaseModel):
    ok: bool
    data: Dict[str, Any]
    version: str = APP_VERSION

# ---------------- Helpers ----------------
def _short_city(city: str) -> str:
    return city.strip()

def _content_plan(niche: str, city: str, trends: Dict[str, Any]) -> List[str]:
    """Deterministic 7-day plan that incorporates top trend if present."""
    gt = trends.get("google_trends", {}).get("rising", [])
    rd = trends.get("reddit", {}).get("posts", [])
    top_kw = gt[0]["query"] if gt else "local market update"
    top_reddit = rd[0]["title"] if rd else f"{niche} tips for { _short_city(city) }"

    return [
        f"Mon — IG Reel: '{top_kw.title()}' explained for {_short_city(city)}",
        f"Tue — Short post: 3 quick wins for {_short_city(city)} {niche.lower()} clients",
        f"Wed — TikTok: '{top_reddit[:70]}…' with a 30s voiceover + B-roll",
        f"Thu — Story: Behind the scenes / FAQ (DM for the checklist)",
        f"Fri — Carousel: 'Top 5 neighborhoods in {_short_city(city)} for new {niche.lower()} clients'",
        "Sat — Community shoutout (tag local partner; ask a Q to boost replies)",
        "Sun — Email digest with one clear CTA (reply to get the PDF)"
    ]

def _outreach(niche: str, city: str, leads: List[Lead]) -> List[Dict[str, str]]:
    """Three-touch sequence with friendly, practical messaging."""
    target_hint = leads[0].Name if leads else "nearby partners"
    return [
        {
            "send_dt": "in 0 days",
            "channel": "email",
            "subject": f"Quick hello from a {niche} in {_short_city(city)}",
            "body": (
                f"Hi there — I help folks in {_short_city(city)} as a {niche}. "
                f"I’d love to collaborate with {target_hint} on something helpful for the community. "
                "Open to a 15-min chat this week?"
            ),
        },
        {
            "send_dt": "in 2 days",
            "channel": "sms",
            "subject": "",
            "body": (
                "Hey! Just checking back — happy to share a one-page guide we can co-brand. "
                "Would that be useful?"
            ),
        },
        {
            "send_dt": "in 7 days",
            "channel": "email",
            "subject": "Shall we try a small test?",
            "body": (
                "If you’re open, we can run a small pilot next week (no cost). "
                "If not, all good — thanks for considering!"
            ),
        },
    ]

def _score_leads(leads: List[Lead]) -> List[Dict[str, Any]]:
    """
    Transparent, deterministic 'OpportunityScore':
      - Fewer reviews = easier wedge (higher score)
      - Lower rating = more room to help (higher score)
      - Bounded 1..99
    """
    scored: List[Dict[str, Any]] = []
    for l in leads:
        base = 50
        if l.Reviews is not None:
            # Up to +40 points if reviews are very low
            base += max(0, 40 - min(40, int((l.Reviews or 0) / 10)))
        if l.Rating is not None:
            # Up to +20 points if rating is low
            base += max(0, 20 - int((l.Rating or 0) * 3))
        score = min(99, max(1, base))
        data = l.model_dump()
        data["OpportunityScore"] = score
        scored.append(data)
    scored.sort(key=lambda x: x.get("OpportunityScore", 0), reverse=True)
    return scored

# ---------------- Meta ----------------
@app.get("/", tags=["meta"])
def root():
    return {"name": APP_NAME, "version": APP_VERSION}

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok", "version": APP_VERSION}

@app.get("/ping", tags=["meta"])
def ping():
    return {"pong": True}

# ---------------- Crew Orchestrator ----------------
@app.post("/run_crew", response_model=CrewResponse, tags=["crew"])
def run_crew(p: CrewPayload, request: Request):
    """
    Orchestrates the plan:
      - Summarize the top trend signals
      - Score leads by 'OpportunityScore'
      - Produce a 7-day content plan
      - Generate a 3-touch outreach sequence
    Deterministic (no external LLM dependency).
    """
    try:
        ip = request.client.host if request.client else "?"
        log.info("Crew start niche=%s city=%s leads=%d ip=%s", p.niche, p.city, len(p.leads), ip)

        top_leads = _score_leads(p.leads) if p.leads else []
        content_plan = _content_plan(p.niche, p.city, p.trends or {})
        outreach = _outreach(p.niche, p.city, p.leads or [])

        gt = p.trends.get("google_trends", {}).get("rising", []) if p.trends else []
        rd = p.trends.get("reddit", {}).get("posts", []) if p.trends else []
        summary_lines = [
            f"Top Google query: '{gt[0]['query']}' (value {gt[0].get('value','?')})" if gt else "No Google spike detected.",
            f"Top Reddit topic: '{rd[0]['title']}' (score {rd[0].get('score','?')})" if rd else "No Reddit topic detected.",
        ]

        result = {
            "summary": f"Built a weekly plan for {p.niche} in {p.city}.",
            "trend_summary": summary_lines,
            "top_leads": top_leads[:10],
            "content_plan": content_plan,
            "outreach_sequence": outreach,
        }
        return CrewResponse(ok=True, data=result)
    except Exception as e:
        log.exception("Crew error: %s", e)
        raise HTTPException(status_code=500, detail=f"Crew error: {e}")

