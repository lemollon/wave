# tools/trends.py — Google Trends + Reddit (free) + optional YouTube

import os, time, datetime as dt
from typing import List, Dict, Optional
import pandas as pd
import requests

# Google Trends
try:
    from pytrends.request import TrendReq
    _PYTRENDS_OK = True
except Exception:
    _PYTRENDS_OK = False

# Reddit
try:
    import praw
    _PRAW_OK = True
except Exception:
    _PRAW_OK = False

def _env(k, d=""): return os.getenv(k, d)

def google_trends_rising(keywords: List[str], geo="US", timeframe="now 7-d") -> Dict:
    if not _PYTRENDS_OK:
        return {"source":"google_trends","error":"pytrends not installed","rising":[],"iot":{}}
    pytrends = TrendReq(hl="en-US", tz=360)
    rising_all, iot_map = [], {}
    for kw in keywords[:6]:
        try:
            pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
            # interest over time
            iot = pytrends.interest_over_time()
            if not iot.empty:
                ser = iot[kw].reset_index().rename(columns={kw:"interest","date":"ts"})
                ser["keyword"] = kw
                iot_map[kw] = ser.to_dict(orient="records")
            # related rising
            rq = pytrends.related_queries()
            if kw in rq and isinstance(rq[kw].get("rising"), pd.DataFrame):
                rising_df = rq[kw]["rising"].head(10)
                rising_all += [{
                    "keyword": kw,
                    "query": row["query"],
                    "value": int(row.get("value", 0)),
                    "link": f"https://www.google.com/search?q={row['query'].replace(' ','+')}"
                } for _, row in rising_df.iterrows()]
            time.sleep(1.0)
        except Exception as e:
            rising_all.append({"keyword": kw, "query": None, "value": 0, "error": str(e)})
    return {"source":"google_trends","rising":rising_all,"iot":iot_map}

def reddit_hot(subreddits: List[str], limit: int = 15) -> Dict:
    if not _PRAW_OK:
        return {"source":"reddit","error":"praw not installed","posts":[]}
    cid = _env("REDDIT_CLIENT_ID"); csec = _env("REDDIT_CLIENT_SECRET")
    ua = _env("REDDIT_USER_AGENT","wavepilot/0.1")
    if not (cid and csec):
        return {"source":"reddit","error":"Missing Reddit API keys","posts":[]}
    try:
        reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
        posts = []
        for sub in (subreddits or [])[:6]:
            for p in reddit.subreddit(sub).hot(limit=limit):
                posts.append({
                    "subreddit": sub,
                    "title": p.title,
                    "score": int(p.score),
                    "url": f"https://www.reddit.com{p.permalink}",
                    "created_utc": int(p.created_utc)
                })
            time.sleep(0.6)
        posts = sorted(posts, key=lambda x: x.get("score",0), reverse=True)
        return {"source":"reddit","posts":posts}
    except Exception as e:
        return {"source":"reddit","error":str(e),"posts":[]}

def youtube_search(api_key: str, query: str, max_results: int = 10) -> Dict:
    if not api_key: return {"source":"youtube","items":[],"error":"missing key"}
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part":"snippet","q":query,"type":"video","order":"date","maxResults":max_results,"key":api_key}
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        items=[]
        for it in r.json().get("items",[]):
            sn = it.get("snippet",{})
            items.append({"title":sn.get("title"),"channel":sn.get("channelTitle"),
                          "publishedAt":sn.get("publishedAt"),
                          "url":f"https://www.youtube.com/watch?v={it['id'].get('videoId')}"})
        return {"source":"youtube","items":items}
    except Exception as e:
        return {"source":"youtube","items":[],"error":str(e)}

def gather_trends(niche_keywords: List[str], city="", state="", subs=None, geo="US", timeframe="now 7-d", youtube_api_key=None) -> Dict:
    subs = subs or ["SmallBusiness","Marketing"]
    kw = [k for k in niche_keywords if k][:6]
    if city and state: kw.append(f"{city} {state}")
    gt = google_trends_rising(kw, geo=geo, timeframe=timeframe)
    rd = reddit_hot(subs)
    yt = youtube_search(youtube_api_key or _env("YOUTUBE_API_KEY",""), " | ".join(kw) or "local business", 10)
    return {"generated_at": dt.datetime.utcnow().isoformat(),
            "inputs":{"keywords":kw,"geo":geo,"timeframe":timeframe,"city":city,"state":state,"subs":subs},
            "google_trends":gt,"reddit":rd,"youtube":yt}
