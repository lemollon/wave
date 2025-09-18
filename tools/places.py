# tools/places.py — minimal Google Places (New) text search → DataFrame

import os, requests, pandas as pd

def search_places_optional(query: str, city: str, state: str, limit: int = 12, api_key: str = "") -> pd.DataFrame | None:
    if not api_key:
        return None
    text_url = "https://places.googleapis.com/v1/places:searchText"
    loc = ", ".join([x for x in [city.strip(), state.strip()] if x])
    q = f"{query} in {loc}" if loc else query
    headers = {"Content-Type":"application/json","X-Goog-Api-Key":api_key,
               "X-Goog-FieldMask":"places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount"}
    body = {"textQuery": q, "maxResultCount": limit}
    r = requests.post(text_url, headers=headers, json=body, timeout=20); r.raise_for_status()
    places = r.json().get("places", []) or []
    rows=[]
    for p in places[:limit]:
        name = (p.get("displayName") or {}).get("text") or p.get("name","").split("/")[-1]
        locd = p.get("location") or {}
        rows.append({
            "Name": name,
            "Rating": float(p.get("rating",0) or 0),
            "Reviews": int(p.get("userRatingCount",0) or 0),
            "Address": p.get("formattedAddress",""),
            "Lat": locd.get("latitude"),
            "Lng": locd.get("longitude")
        })
    return pd.DataFrame(rows)
