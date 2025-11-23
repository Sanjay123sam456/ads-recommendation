# app.py
import os
import time
import json
import logging
import urllib.parse
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from icecream import ic
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
# ---------------- CONFIG ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("ad-backend")

PORT = int(os.getenv("PORT", 80))
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
CSE_BASE = "https://www.googleapis.com/customsearch/v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")


CACHE_TTL = 300
DEFAULT_RADIUS_METERS = 2000
DEFAULT_LIMIT = 20

# ---------------- MODELS ----------------
class POI(BaseModel):
    name: str
    address: Optional[str]
    lat: float
    lon: float
    category: Optional[str]

class CSEItem(BaseModel):
    title: Optional[str]
    snippet: Optional[str]
    link: Optional[str]

class TopAd(BaseModel):
    title: str
    ad_text: str
    source_link: Optional[str]
    lat: Optional[float] = None
    lon: Optional[float] = None

# ---------------- UTILITIES ----------------
def build_osm_address(tags: Dict[str, Any]) -> Optional[str]:
    if not tags:
        return None
    parts = []
    for k in ["addr:housenumber","addr:street","addr:suburb","addr:city","addr:state","addr:postcode"]:
        v = tags.get(k)
        if v:
            parts.append(v)
    return ", ".join(parts) if parts else None

def fetch_pois_osm(lat: float, lon: float, radius: int = DEFAULT_RADIUS_METERS, limit: int = 10) -> List[POI]:
    query = f"""
    [out:json][timeout:20];
    (
      node(around:{radius},{lat},{lon})["amenity"];
      node(around:{radius},{lat},{lon})["shop"];
      node(around:{radius},{lat},{lon})["tourism"];
      node(around:{radius},{lat},{lon})["leisure"];
    );
    out center;
    """
    r = requests.post(OVERPASS_URL, data=query, timeout=25)
    r.raise_for_status()
    data = r.json()
    pois: List[POI] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        la = el.get("lat") or el.get("center", {}).get("lat")
        lo = el.get("lon") or el.get("center", {}).get("lon")
        if la is None or lo is None:
            continue
        cat = tags.get("shop") or tags.get("amenity") or tags.get("tourism") or tags.get("leisure")
        addr = build_osm_address(tags)
        try:
            pois.append(POI(name=name, address=addr, lat=float(la), lon=float(lo), category=cat))
        except Exception:
            continue
        if len(pois) >= limit:
            break
    return pois

def call_google_cse(query: str, num: int = 10, safe: bool = True) -> Dict[str, Any]:
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID must be set")
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num
    }
    if safe:
        params["safe"] = "active"
    url = CSE_BASE + "?" + urllib.parse.urlencode(params)
    resp = requests.get(url, timeout=12)
    resp.raise_for_status()
    return resp.json()

def extract_cse_item(it: Dict[str, Any]) -> CSEItem:
    return CSEItem(
        title = it.get("title"),
        snippet = it.get("snippet"),
        link = it.get("link") or it.get("formattedUrl")
    )

def dedupe_cse_items(items: List[CSEItem]) -> List[CSEItem]:
    seen = set()
    out = []
    for it in items:
        key = (it.title or "") + "|" + (it.link or "") + "|" + (it.snippet or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def simple_name_match(text: str, name: str) -> bool:
    if not text or not name:
        return False
    text_l = text.lower()
    tokens = [t for t in "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in name).split() if t and len(t)>=3]
    if not tokens:
        return False
    return any(t in text_l for t in tokens)

def extract_coords_from_google_maps_url(url: str) -> Optional[Dict[str, float]]:
    # google maps formats: .../@12.345678,98.765432,17z  OR ...?q=12.345678,98.765432
    try:
        if "google.com/maps" in url:
            # try @lat,lon
            import re
            m = re.search(r"/@(-?\d+\.\d+),(-?\d+\.\d+)", url)
            if m:
                return {"lat": float(m.group(1)), "lon": float(m.group(2))}
            # try q=lat,lon
            m2 = re.search(r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)", url)
            if m2:
                return {"lat": float(m2.group(1)), "lon": float(m2.group(2))}
    except Exception:
        return None
    return None

def looks_like_map_link(link: Optional[str]) -> bool:
    if not link: return False
    link = link.lower()
    return "google.com/maps" in link or "openstreetmap.org" in link or "/place/" in link or "maps.app.goo.gl" in link

def looks_like_online_marketplace(link: Optional[str]) -> bool:
    if not link: return False
    link = link.lower()
    marketplaces = ["amazon.", "flipkart.", "myntra.", "snapdeal.", "ajio.", "ebay.", "shopify.", "etsy."]
    return any(m in link for m in marketplaces)

# ---------------- LLM ----------------
def call_openrouter_generate(prompt: str, model: str = OPENROUTER_MODEL, timeout: int = 30) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that selects the top 5 most relevant ads and writes short ad copy tailored to the user's interest. Return only a JSON array of objects."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        content = msg.get("content") or first.get("text")
        return content or json.dumps(data)
    return json.dumps(data)

# ---------------- APP ----------------
app = FastAPI(title="Ads Recommender (coords-enabled)")
from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ads/recommend", response_model=List[TopAd])
def recommend_ads(
    lat: float,
    lon: float,
    interest: str = Query(..., description="User interest (e.g., electronics, food, fashion)"),
    radius: int = Query(DEFAULT_RADIUS_METERS),
    poi_limit: int = Query(6, description="Max POIs to fetch from OSM"),
    cse_per_poi: int = Query(10, description="How many Google CSE results per POI")
):
    # 1) fetch POIs
    try:
        pois = fetch_pois_osm(lat, lon, radius=radius, limit=poi_limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch POIs from OSM: {e}")

    ic("OSM POIs:", [p.dict() for p in pois])

    if not pois:
        # still continue but response will be empty
        return []

    # 2) collect google results per-poi (address-based queries)
    all_cse: List[CSEItem] = []
    for p in pois:
        location_term = p.address or p.name
        q = f"{interest} offers near {location_term}"
        try:
            data = call_google_cse(q, num=cse_per_poi)
        except Exception as e:
            logger.warning(f"CSE failure for q={q}: {e}")
            continue
        items = data.get("items", []) or []
        cse_items = [extract_cse_item(it) for it in items]
        all_cse.extend(cse_items)

    all_cse = dedupe_cse_items(all_cse)
    ic("Google CSE combined results (deduped):", [it.dict() for it in all_cse])

    # 3) build prompt
    pois_short = [{"name": p.name, "address": p.address or "", "lat": p.lat, "lon": p.lon, "category": p.category or ""} for p in pois]
    cse_short = [{"title": it.title or "", "snippet": it.snippet or "", "link": it.link or ""} for it in all_cse[:50]]

    prompt = (
        "User interest: " + interest + "\n\n"
        "Nearby POIs (name/address/lat/lon/category):\n" + json.dumps(pois_short, ensure_ascii=False, indent=2) + "\n\n"
        "Google search results (title/snippet/link):\n" + json.dumps(cse_short, ensure_ascii=False, indent=2) + "\n\n"
        "Task:\n"
        "1) Select the 5 most relevant search results for this user's interest and these POIs.\n"
        "2) For each selected result, produce a short ad with the following JSON structure:\n"
        "{\"title\":\"...\",\"ad_text\":\"...\",\"source_link\":\"...\"}\n"
        "3) Return only a JSON array of up to 5 objects in that exact structure. Do not add commentary."
    )

    # 4) call LLM
    try:
        llm_raw = call_openrouter_generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    ic("LLM raw output:", llm_raw)

    # 5) parse LLM JSON
    parsed = []
    try:
        txt = llm_raw.strip()
        start = txt.find('[')
        end = txt.rfind(']')
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(txt[start:end+1])
        else:
            parsed = json.loads(txt)
    except Exception as e:
        # fallback — return single raw blob as ad
        ic("LLM parse failed:", str(e))
        return [TopAd(title="LLM output (raw)", ad_text=llm_raw[:1000], source_link=None, lat=None, lon=None)]

    # 6) For each ad, try to map coordinates:
    out_ads: List[TopAd] = []
    for obj in parsed[:5]:
        title = (obj.get("title") or "").strip()
        ad_text = (obj.get("ad_text") or obj.get("ad") or obj.get("description") or "").strip()
        link = (obj.get("source_link") or obj.get("link") or "").strip() or None

        ad_lat = None
        ad_lon = None

        # Rule 1: match to POIs by tokens in title/ad_text/snippet
        combined_text = " ".join([title, ad_text]).lower()
        matched_poi = None
        for p in pois:
            if simple_name_match(combined_text, p.name):
                matched_poi = p
                break
        if matched_poi:
            ad_lat = matched_poi.lat
            ad_lon = matched_poi.lon
        else:
            # Rule 2: if link is a map link, try to extract coords
            if link and looks_like_map_link(link):
                coords = extract_coords_from_google_maps_url(link)
                if coords:
                    ad_lat = coords["lat"]
                    ad_lon = coords["lon"]
            # Rule 3: if link looks like a marketplace/online-only, explicitly set coords to None
            elif link and looks_like_online_marketplace(link):
                ad_lat = None
                ad_lon = None
            # else: leave None (unknown). If user wants aggressive heuristics we could try geocoding link/brand.

        out_ads.append(TopAd(title=title or "Untitled", ad_text=ad_text or "", source_link=link, lat=ad_lat, lon=ad_lon))

    return out_ads

# simple health
@app.get("/health")
def health():
    return {
        "ok": True,
        "google_search": {"key_set": bool(GOOGLE_SEARCH_API_KEY), "cse_set": bool(GOOGLE_CSE_ID)},
        "openrouter": bool(OPENROUTER_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
