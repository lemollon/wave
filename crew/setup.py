# crew/setup.py — three-role crew with fallback if CrewAI is not installed

from typing import List, Dict
import json

try:
    from crewai import Agent, Task, Crew, Process
    _CREW_OK = True
except Exception:
    _CREW_OK = False

def run_wavepilot_crew(niche: str, city: str, trends: dict, leads: list) -> Dict:
    if not _CREW_OK:
        # Fallback: deterministic summary
        return {
            "summary": f"Crew (fallback) processed {len(leads)} leads for {niche} in {city}.",
            "content_plan": [
                "Post Mon: 'Top neighborhoods for families in {city}'",
                "Post Wed: 'Rates update and what it means for buyers'",
                "Post Fri: 'Open house highlight'"
            ],
            "outreach": [
                "Email apt manager: co-host move-in Q&A night",
                "SMS mover: cross-referrals",
                "Email mortgage broker: rate explainer co-branded"
            ]
        }

    trend_scout = Agent(
        role="TrendScout",
        goal="Summarize the biggest waves in the data.",
        backstory="You love data and trends.",
        allow_delegation=False,
    )
    planner = Agent(
        role="Planner",
        goal="Turn the trends into a 7-day content plan.",
        backstory="You schedule to match local attention windows.",
        allow_delegation=False,
    )
    copyboss = Agent(
        role="CopyBoss",
        goal="Draft outreach that gets replies.",
        backstory="You write concise, local-first messages.",
        allow_delegation=False,
    )

    t1 = Task(description=f"Summarize trends for {niche} in {city}:\n{json.dumps(trends)[:4000]}", agent=trend_scout)
    t2 = Task(description=f"Create 7-day calendar using these leads:\n{json.dumps(leads)[:4000]}", agent=planner)
    t3 = Task(description="Write a 3-touch outreach sequence to top 3 partner types.", agent=copyboss)

    crew = Crew(agents=[trend_scout, planner, copyboss], tasks=[t1, t2, t3], process=Process.sequential)
    result = crew.kickoff()
    return {"summary":"Crew complete","raw":str(result)}
