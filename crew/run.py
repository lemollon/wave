# crew/run.py — optional CrewAI growth crew
# If CrewAI is not installed, app will catch and show a friendly warning.

def run_growth_crew(lead: dict, persona: str = "Local Professional") -> str:
    """
    Returns a brief 'dossier' for the selected lead:
    - Who they are
    - Why they’re a good partner
    - First-email idea
    This is a lightweight stub unless crewai is installed.
    """
    try:
        from crewai import Agent, Task, Crew
    except Exception:
        # Fallback stub so UI still works
        name = lead.get("Name","(unknown)")
        why  = f"Good partner due to local visibility and target overlap. Rating {lead.get('Rating','?')} with {lead.get('Reviews','?')} reviews."
        email = (f"Subject: Quick partnership idea\n\nHi {name} team,\n"
                 f"I'm a {persona} nearby. I think we can refer clients to each other.\n"
                 "Open to a 10-minute chat this week?\n\nBest,\n<Your Name>")
        return f"**Lead:** {name}\n\n**Why now:** {why}\n\n**First email idea:**\n\n{email}"

    # If CrewAI is installed, run a tiny 2-agent crew
    name = lead.get("Name","(unknown)")
    context = f"{name} | Rating {lead.get('Rating')} | Reviews {lead.get('Reviews')} | Website {lead.get('Website','')}"
    researcher = Agent(name="Researcher", role="Find reasons to partner", goal="Summarize why this business is a good feeder partner.")
    copywriter = Agent(name="Copywriter", role="Write first email", goal="Draft a 120-150 word email to start a partnership.")
    t1 = Task(description=f"Analyze this business and city context:\n{context}", agent=researcher)
    t2 = Task(description=f"Use the research summary to draft email from a {persona}.", agent=copywriter)
    crew = Crew(agents=[researcher, copywriter], tasks=[t1, t2])
    out = crew.kickoff()
    return str(out)
