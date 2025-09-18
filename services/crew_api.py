# services/crew_api.py — run with:
# uvicorn services.crew_api:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from crew.setup import run_wavepilot_crew

app = FastAPI()

class CrewPayload(BaseModel):
    niche: str
    city: str
    trends: Dict[str, Any]
    leads: List[Dict[str, Any]] = []

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/run_crew")
def run_crew(p: CrewPayload):
    out = run_wavepilot_crew(p.niche, p.city, p.trends, p.leads)
    return {"ok": True, "data": out}
