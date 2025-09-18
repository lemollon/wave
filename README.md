# wave
Intelligent Lead Generator
# WavePilot — AI Growth Team for Local Businesses

**Showcases**: LangChain tools, CrewAI multi-agent orchestration, (optional) AutoGPT-style nightly watcher.  
**Delivers**: free trend intel (Google Trends + Reddit), optional local lead finder (Google Places), ready-to-send outreach, weekly export.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys (Reddit is free; OpenAI optional)
# run Streamlit
streamlit run app.py

