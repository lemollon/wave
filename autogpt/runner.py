# autogpt/runner.py — placeholder "watcher".
# In production, trigger via GitHub Actions cron to ping your crew_api.

import os, json, datetime as dt, requests
from tools.trends import gather_trends

def main():
    niche = os.getenv("NIGHTLY_NICHE","real estate")
    city  = os.getenv("NIGHTLY_CITY","Austin, TX")
    trends = gather_trends([niche], city=city.split(",")[0], state=city.split(",")[-1].strip())
    payload = {"niche": niche, "city": city, "trends": trends, "leads": []}
    crew_url = os.getenv("CREW_API_URL","http://localhost:8000/run_crew")
    try:
        r = requests.post(crew_url, json=payload, timeout=60)
        print("Crew result:", r.status_code, r.text[:300])
    except Exception as e:
        print("Error calling crew:", e)

if __name__ == "__main__":
    main()
