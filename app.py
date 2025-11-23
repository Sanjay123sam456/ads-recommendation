# app.py -- Updated backend with Category-based Google Custom Search (/offers/search)
import os
import time
import json
import logging
import urllib.parse
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# OpenRouter LLM (optional)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o"

# GEOAPIFY KEY (primary POI source)
GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")

# GOOGLE CUSTOM SEARCH (JSON)
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
CSE_BASE = "https://www.googleapis.com/customsearch/v1"

CACHE_TTL = 300
PORT = int(os.getenv("PORT", 8000))
DEFAULT_RADIUS_METERS = 2000
DEFAULT_LIMIT = 20

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("ad-backend")

app = FastAPI(title="AI Ad Aware Backend (Category Search)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class POI(BaseModel):
    name: str
    address: Optional[str]
    lat: float
    lon: float
    category: Optional[str]

class OfferItem(BaseModel):
    poi_name: Optional[str]
    category: Optional[str]
    offer_title: str
    offer_body: str
    discount_percent: Optional[float]
    valid_till: Optional[str]

class OfferWithPOI(BaseModel):
    poi: POI
    offer: OfferItem

class GeneratedAd(BaseModel):
    poi_name: str
    address: Optional[str]
    category: Optional[str]
    offer_title: str
    offer_body: str
    discount_percent: Optional[float]
    ad_title: str
    ad_text: str
    cta_text: str

# ---------------- CACHE ----------------
class TTLCache:
    def __init__(self, ttl):
        self.ttl = ttl
        self.store: Dict[str, Dict[str, Any]] = {}

    def get(self, k: str):
        d = self.store.get(k)
        if not d:
            return None
        if time.time() - d["ts"] > self.ttl:
            try:
                del self.store[k]
            except KeyError:
                pass
            return None
        return d["val"]

    def set(self, k: str, v: Any):
        self.store[k] = {"val": v, "ts": time.time()}

poi_cache = TTLCache(CACHE_TTL)

# ---------------- OFFERS JSON ----------------
OFFERS_FILE = "offers.json"

def load_offers() -> List[OfferItem]:
    try:
        with open(OFFERS_FILE, "r", encoding="utf-8") as f:
            offers_raw = json.load(f)
            offers = [OfferItem(**o) for o in offers_raw]
            logger.info(f"Loaded {len(offers)} offers from {OFFERS_FILE}")
            return offers
    except Exception as e:
        logger.warning(f"Failed loading offers.json: {e}")
        return []

OFFERS_DB: List[OfferItem] = load_offers()
OFFERS_BY_NAME: Dict[str, List[OfferItem]] = {}
OFFERS_BY_CATEGORY: Dict[str, List[OfferItem]] = {}

for offer in OFFERS_DB:
    if offer.poi_name:
        OFFERS_BY_NAME.setdefault(offer.poi_name.strip().lower(), []).append(offer)
    if offer.category:
        OFFERS_BY_CATEGORY.setdefault(offer.category.strip().lower(), []).append(offer)

# ---------------- OSM fallback ----------------
def build_osm_address(tags: Dict[str, Any]) -> Optional[str]:
    if not tags:
        return None
    parts = []
    for k in ["addr:housenumber","addr:street","addr:suburb","addr:city","addr:state","addr:postcode"]:
        v = tags.get(k)
        if v:
            parts.append(v)
    return ", ".join(parts) if parts else None

def fetch_pois_osm(lat: float, lon: float, radius: int = DEFAULT_RADIUS_METERS, limit: int = DEFAULT_LIMIT) -> List[POI]:
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
    r = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=25)
    r.raise_for_status()
    data = r.json()
    pois: List[POI] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        la = el.get("lat") or el.get("center", {}).get("lat")
        lo = el.get("lon") or el.get("center", {}).get("lon")
        cat = tags.get("shop") or tags.get("amenity") or tags.get("tourism")
        addr = build_osm_address(tags)
        try:
            pois.append(POI(name=name, address=addr, lat=float(la), lon=float(lo), category=cat))
        except Exception:
            continue
        if len(pois) >= limit:
            break
    return pois

# ---------------- Geoapify primary ----------------
def fetch_pois_geoapify(lat: float, lon: float, radius: int = DEFAULT_RADIUS_METERS, limit: int = DEFAULT_LIMIT) -> List[POI]:
    if not GEOAPIFY_KEY:
        raise RuntimeError("GEOAPIFY_API_KEY missing")
    url = "https://api.geoapify.com/v2/places"
    params = {
        "apiKey": GEOAPIFY_KEY,
        "filter": f"circle:{lon},{lat},{radius}",
        "limit": limit,
        "categories": "commercial,service,entertainment,tourism,leisure,healthcare,education,food,shop"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    pois: List[POI] = []
    for f in data.get("features", []):
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None])
        lon2, lat2 = coords if len(coords) >= 2 else (None, None)
        if lat2 is None or lon2 is None:
            continue
        cats = props.get("categories") or []
        cat = cats[0] if isinstance(cats, list) and cats else (props.get("type") or "other")
        addr = props.get("formatted") or props.get("address_line1") or props.get("address_line2")
        pois.append(POI(name=props.get("name") or "Unknown place", address=addr, lat=float(lat2), lon=float(lon2), category=cat))
        if len(pois) >= limit:
            break
    return pois

def get_pois(lat: float, lon: float, radius: int = DEFAULT_RADIUS_METERS, limit: int = DEFAULT_LIMIT) -> List[POI]:
    key = f"{round(lat,6)}:{round(lon,6)}:{radius}:{limit}"
    cached = poi_cache.get(key)
    if cached:
        return cached
    try:
        pois = fetch_pois_geoapify(lat, lon, radius, limit)
        logger.info(f"Geoapify returned {len(pois)} POIs")
    except Exception as e:
        logger.warning(f"Geoapify failed: {e} -> falling back to OSM")
        pois = fetch_pois_osm(lat, lon, radius, limit)
    poi_cache.set(key, pois)
    return pois

# ---------------- Offer matching helpers ----------------
def find_offer_for_poi(poi: POI) -> Optional[OfferItem]:
    if poi.name:
        k = poi.name.strip().lower()
        if k in OFFERS_BY_NAME and OFFERS_BY_NAME[k]:
            return OFFERS_BY_NAME[k][0]
    if poi.category:
        cat = poi.category.strip().lower().split(".")[0]
        if cat in OFFERS_BY_CATEGORY and OFFERS_BY_CATEGORY[cat]:
            return OFFERS_BY_CATEGORY[cat][0]
    return None

def attach_offers_to_pois(pois: List[POI]) -> List[OfferWithPOI]:
    out: List[OfferWithPOI] = []
    for p in pois:
        o = find_offer_for_poi(p)
        if o:
            out.append(OfferWithPOI(poi=p, offer=o))
    return out

# ---------------- Google Custom Search helpers ----------------
def call_google_cse(query: str, num: int = 3, safe: bool = True) -> Dict[str, Any]:
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_ID not configured")
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
    # debug logs
    logger.info(f"CSE REQUEST URL: {resp.request.url}")
    logger.info(f"CSE STATUS: {resp.status_code}")
    logger.info(f"CSE RESPONSE (first 800 chars): {resp.text[:800]!s}")
    resp.raise_for_status()
    return resp.json()

def extract_cse_item(it: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "title": it.get("title"),
        "snippet": it.get("snippet"),
        "link": it.get("link") or it.get("formattedUrl")
    }

def simple_text_match(short_name: str, text: str) -> bool:
    if not short_name or not text:
        return False
    tokens = [t for t in "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in short_name).split() if t and len(t) >= 2]
    if not tokens:
        return False
    text_lower = text.lower()
    for t in tokens:
        if t in text_lower:
            return True
    return False

# ---------------- Category -> search phrase map ----------------
CATEGORY_QUERIES = {
    "food": ["restaurant offers near {lat} {lon}", "restaurant discounts near {lat} {lon}", "food delivery offers near {lat} {lon}"],
    "pharmacy": ["pharmacy offers near {lat} {lon}", "pharmacy discounts near {lat} {lon}", "medical store discounts near {lat} {lon}"],
    "electronics": ["electronics store discounts near {lat} {lon}", "mobile offers near {lat} {lon}", "tv deals near {lat} {lon}"],
    "grocery": ["grocery offers near {lat} {lon}", "supermarket discounts near {lat} {lon}", "grocery coupons near {lat} {lon}"],
    "fashion": ["clothing sale near {lat} {lon}", "fashion discounts near {lat} {lon}", "apparel offers near {lat} {lon}"],
    "fuel": ["petrol pump offers near {lat} {lon}", "fuel discount near {lat} {lon}"],
    "default": ["best offers near {lat} {lon}", "local discounts near {lat} {lon}", "deals near {lat} {lon}"]
}

# ---------------- New endpoint: /offers/search (Category-based) ----------------
@app.get("/offers/search")
def offers_search(
    lat: float,
    lon: float,
    category: Optional[str] = Query(None, description="Category e.g. food, pharmacy, electronics, grocery, fashion"),
    radius: int = Query(DEFAULT_RADIUS_METERS),
    limit: int = Query(6, description="Max POIs to try")
):
    """
    Category-based web search for offers using Google CSE.
    - Uses 'category' to build queries (Option B).
    - Uses POIs near lat/lon as seeds.
    - Returns list of {poi, cse_result, matched_query}
    """
    try:
        pois = get_pois(lat, lon, radius=radius, limit=limit)
    except Exception as e:
        logger.warning(f"offers_search: POI fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch POIs")

    if not pois:
        return []

    cat = (category or "default").strip().lower()
    patterns = CATEGORY_QUERIES.get(cat, CATEGORY_QUERIES["default"])

    results: List[Dict[str, Any]] = []
    seen_links = set()

    # For each POI, try a few category queries (replace lat/lon placeholders)
    for poi in pois:
        for pattern in patterns:
            q = pattern.format(lat=round(lat,4), lon=round(lon,4))
            try:
                data = call_google_cse(q, num=3)
                print(data)
            except Exception as e:
                logger.warning(f"CSE call failed for q='{q}': {e}")
                continue

            items = data.get("items", []) or []
            for it in items:
                link = it.get("link") or it.get("formattedUrl")
                if not link or link in seen_links:
                    continue
                combined = " ".join([it.get("title",""), it.get("snippet",""), str(link)])
                # match if POI name tokens appear in snippet/title OR category word appears
                if simple_text_match(poi.name, combined) or (poi.category and poi.category.lower() in combined.lower()):
                    seen_links.add(link)
                    results.append({
                        "poi": {"name": poi.name, "address": poi.address, "lat": poi.lat, "lon": poi.lon, "category": poi.category},
                        "cse": extract_cse_item(it),
                        "matched_query": q
                    })
                    # once matched for this POI, break to next POI
                    break
            # if matched (last appended), break patterns loop for this POI
            if any(r["poi"]["name"] == poi.name for r in results):
                break

    return results

# ---------------- Debug route to inspect raw CSE responses ----------------
@app.get("/debug/cse")
def debug_cse(q: Optional[str] = None, num: int = 5, safe: bool = True):
    """
    Debug endpoint: call CSE with a test query (or provided q) and return raw JSON.
    Example: /debug/cse?q=restaurant+gurgaon
    """
    if not q:
        q = "restaurant offers near 28.4595 77.0266"
    try:
        data = call_google_cse(q, num=num, safe=safe)
        items = data.get("items", [])[:num]
        return {"ok": True, "query": q, "count": len(items), "items": items}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------- Existing endpoints (kept) ----------------
@app.get("/places/nearby")
def places_nearby(lat: float, lon: float):
    return get_pois(lat, lon)

@app.get("/offers/nearby", response_model=List[OfferWithPOI])
def offers_nearby(lat: float, lon: float):
    pois = get_pois(lat, lon)
    return attach_offers_to_pois(pois)

@app.get("/ads/nearby", response_model=List[GeneratedAd])
def template_ads(lat: float, lon: float):
    pois = get_pois(lat, lon)
    items = attach_offers_to_pois(pois)
    out: List[GeneratedAd] = []
    for item in items:
        poi = item.poi; offer = item.offer
        out.append(GeneratedAd(
            poi_name=poi.name,
            address=poi.address,
            category=poi.category,
            offer_title=offer.offer_title,
            offer_body=offer.offer_body,
            discount_percent=offer.discount_percent,
            ad_title=f"{offer.offer_title} at {poi.name}",
            ad_text=f"{offer.offer_body} Visit {poi.name} ({poi.address or 'near you'}).",
            cta_text="View Offer"
        ))
    return out

@app.get("/ads/llm-nearby", response_model=List[GeneratedAd])
def llm_ads(lat: float, lon: float):
    pois = get_pois(lat, lon)
    items = attach_offers_to_pois(pois)
    result: List[GeneratedAd] = []
    for item in items:
        poi = item.poi; offer = item.offer
        prompt = f"""
Write a short catchy ad:

Place: {poi.name}
Address: {poi.address}
Category: {poi.category}
Offer: {offer.offer_title}
Details: {offer.offer_body}
Discount: {offer.discount_percent}

Return format:
Title:
Body:
CTA:
"""
        try:
            out = ""
            if OPENROUTER_API_KEY:
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": OPENROUTER_MODEL, "messages": [{"role":"user","content":prompt}]}
                r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    out = msg.get("content") or choices[0].get("text") or str(choices[0])
            if not out:
                out = f"Title: {offer.offer_title}\nBody: {offer.offer_body}\nCTA: View offer"
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            out = f"Title: {offer.offer_title}\nBody: {offer.offer_body}\nCTA: View offer"

        title = body = cta = ""
        for line in out.splitlines():
            if line.lower().startswith("title"):
                title = line.split(":",1)[1].strip()
            if line.lower().startswith("body"):
                body = line.split(":",1)[1].strip()
            if line.lower().startswith("cta"):
                cta = line.split(":",1)[1].strip()
        result.append(GeneratedAd(
            poi_name=poi.name,
            address=poi.address,
            category=poi.category,
            offer_title=offer.offer_title,
            offer_body=offer.offer_body,
            discount_percent=offer.discount_percent,
            ad_title=title or offer.offer_title,
            ad_text=body or offer.offer_body,
            cta_text=cta or "View offer"
        ))
    return result

@app.get("/health")
def health():
    return {
        "geoapify": bool(GEOAPIFY_KEY),
        "offers_loaded": len(OFFERS_DB),
        "openrouter": bool(OPENROUTER_API_KEY),
        "google_search": {"key_set": bool(GOOGLE_SEARCH_API_KEY), "cse_set": bool(GOOGLE_CSE_ID)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)
