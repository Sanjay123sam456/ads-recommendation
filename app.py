# app.py
import os
import json
import logging
import re
import urllib.parse
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, Query
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
GOOGLE_CSE_GL = os.getenv("GOOGLE_CSE_GL", "in")
GOOGLE_CSE_CR = os.getenv("GOOGLE_CSE_CR", "countryIN")
GOOGLE_CSE_LR = os.getenv("GOOGLE_CSE_LR", "lang_en")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_BASE = os.getenv("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")


CACHE_TTL = 300
DEFAULT_RADIUS_METERS = 2000
DEFAULT_LIMIT = 20
# ---------------- DEMO ADS FALLBACK ----------------
SAMPLE_ADS = [
    {
        "title": "Sample Electronics Offer – Demo",
        "ad_text": "Up to 30% off on gadgets near you. (Demo data shown due to API quota limit.)",
        "source_link": "https://example.com/electronics"
    },
    {
        "title": "Demo Fashion Sale – Nearby",
        "ad_text": "Trending clothing and accessories — demo preview when live data is unavailable.",
        "source_link": "https://example.com/fashion"
    },
    {
        "title": "Grocery Savings – Demo",
        "ad_text": "Daily essentials at discounted prices — demo ad for UI testing.",
        "source_link": "https://example.com/grocery"
    },
]


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
GENERIC_LOCALITY_WORDS = {
    "near", "nearby", "road", "nagar", "city", "sector", "market", "mall",
    "block", "street", "shop", "store", "area", "district", "state", "india"
}

OFFER_TEMPLATES = [
    "Flat 20% off on selected items",
    "Buy 1 Get 1 on weekend specials",
    "Up to 30% off on combo deals",
    "Flat Rs.200 off above Rs.999",
    "Free add-on with every purchase",
]

NON_LOCAL_TLDS = {
    ".ng", ".pk", ".bd", ".lk", ".np", ".us", ".uk", ".au", ".ca", ".za", ".br", ".de", ".fr", ".it"
}

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
        "num": num,
        "gl": GOOGLE_CSE_GL,
        "cr": GOOGLE_CSE_CR,
        "lr": GOOGLE_CSE_LR,
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

def extract_local_tokens_from_pois(pois: List[POI]) -> List[str]:
    tokens: set[str] = set()
    for p in pois:
        blob = f"{p.name or ''} {p.address or ''}".lower()
        for tok in re.findall(r"[a-z0-9]+", blob):
            if tok.isdigit():
                if len(tok) >= 2:
                    tokens.add(tok)
                continue
            if len(tok) < 4:
                continue
            if tok in GENERIC_LOCALITY_WORDS:
                continue
            tokens.add(tok)
    return sorted(tokens)

def is_non_local_link(link: Optional[str]) -> bool:
    if not link:
        return False
    lower = link.lower()
    return any(lower.endswith(tld) or f"{tld}/" in lower for tld in NON_LOCAL_TLDS)

def extract_offer_phrase(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"(flat\s*\d{1,3}%\s*off)",
        r"(up\s*to\s*\d{1,3}%\s*off)",
        r"(buy\s*1\s*get\s*1)",
        r"(bogo)",
        r"(flat\s*rs\.?\s*\d+\s*off)",
        r"(\d{1,3}%\s*off)",
    ]
    lower = text.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            phrase = m.group(1)
            return " ".join(phrase.split()).title()
    return None

def choose_offer_text(index: int) -> str:
    return OFFER_TEMPLATES[index % len(OFFER_TEMPLATES)]

def is_cse_item_local(item: CSEItem, local_tokens: List[str]) -> bool:
    if not local_tokens:
        return True
    if is_non_local_link(item.link):
        return False
    blob = f"{item.title or ''} {item.snippet or ''} {item.link or ''}".lower()
    matched = sum(1 for tok in local_tokens if tok in blob)
    return matched >= 2

def prioritize_local_cse(items: List[CSEItem], local_tokens: List[str], limit: int = 50) -> List[CSEItem]:
    local_items = [it for it in items if is_cse_item_local(it, local_tokens)]
    if local_items:
        return local_items[:limit]
    non_foreign = [it for it in items if not is_non_local_link(it.link)]
    if non_foreign:
        return non_foreign[:limit]
    return items[:limit]

def fetch_cse_without_pois(interest: str, lat: float, lon: float, cse_per_query: int = 10) -> List[CSEItem]:
    queries = [
        f"{interest} offers near {lat:.5f},{lon:.5f}",
        f"{interest} deals near me",
        f"best {interest} shops nearby",
    ]
    all_items: List[CSEItem] = []
    for q in queries:
        try:
            data = call_google_cse(q, num=cse_per_query)
        except Exception as e:
            logger.warning("CSE failure for fallback q=%s: %s", q, e)
            continue
        items = data.get("items", []) or []
        all_items.extend(extract_cse_item(it) for it in items)
    return dedupe_cse_items(all_items)

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

def build_fallback_ads(interest: str) -> List[TopAd]:
    interest_label = (interest or "local").strip().title() or "Local"
    return [
        TopAd(
            title=f"{interest_label} Offer - Demo",
            ad_text=f"Live provider data is temporarily unavailable. Showing demo {interest or 'local'} offer.",
            source_link="https://example.com/offers",
            lat=None,
            lon=None,
        ),
        TopAd(
            title=f"Top {interest_label} Picks - Demo",
            ad_text=f"Fallback recommendations for {interest or 'local'} while upstream APIs recover.",
            source_link="https://example.com/recommendations",
            lat=None,
            lon=None,
        ),
    ]

def build_ads_from_cse(items: List[CSEItem], interest: str, limit: int = 5) -> List[TopAd]:
    ads: List[TopAd] = []
    for idx, it in enumerate(items[:limit]):
        title = (it.title or "").strip() or f"{interest.title()} Offer"
        snippet_raw = (it.snippet or "").strip()
        offer = extract_offer_phrase(f"{it.title or ''} {snippet_raw}") or choose_offer_text(idx)
        snippet = snippet_raw or f"Nearby {interest} deal. Offer: {offer}."
        if "off" not in snippet.lower() and "buy 1 get 1" not in snippet.lower() and "bogo" not in snippet.lower():
            snippet = f"{snippet} Offer: {offer}."
        link = (it.link or "").strip() or None
        ads.append(TopAd(title=title, ad_text=snippet, source_link=link, lat=None, lon=None))
    return ads

def build_ads_from_pois(pois: List[POI], interest: str, limit: int = 5) -> List[TopAd]:
    ads: List[TopAd] = []
    for idx, p in enumerate(pois[:limit]):
        offer = choose_offer_text(idx)
        ads.append(
            TopAd(
                title=f"{p.name} - {interest.title()} Nearby",
                ad_text=f"Local {interest} option near you at {p.name}. Offer: {offer}.",
                source_link=None,
                lat=p.lat,
                lon=p.lon,
            )
        )
    return ads

# ---------------- LLM ----------------
def call_gemini_generate(prompt: str, model: str = GEMINI_MODEL, timeout: int = 30) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a helpful assistant that selects the top 5 most relevant ads and writes short ad copy "
                        "tailored to the user's interest. Return only a JSON array of objects."
                    )
                }
            ]
        },
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return json.dumps(data)

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text_chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join([t for t in text_chunks if t]).strip()
    return text or json.dumps(data)

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
    poi_limit: int = Query(4, description="Max POIs to fetch from OSM"),
    cse_per_poi: int = Query(3, description="How many Google CSE results per POI")
):
    # 1) fetch POIs
    pois: List[POI] = []
    try:
        pois = fetch_pois_osm(lat, lon, radius=radius, limit=poi_limit)
    except Exception as e:
        logger.warning("OSM fetch failed; continuing with CSE-only flow. Error: %s", e)

    ic("OSM POIs:", [p.dict() for p in pois])

    # 2) collect google results per-poi (address-based queries)
    all_cse: List[CSEItem] = []
    if pois:
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
    else:
        logger.warning("No POIs found; trying coordinate-based CSE queries.")
        all_cse = fetch_cse_without_pois(interest, lat, lon, cse_per_query=cse_per_poi)

    all_cse = dedupe_cse_items(all_cse)
    local_tokens = extract_local_tokens_from_pois(pois)
    all_cse = prioritize_local_cse(all_cse, local_tokens=local_tokens, limit=50)
    ic("Google CSE combined results (deduped):", [it.dict() for it in all_cse])
                # If Google CSE returned no results (quota exhausted / error), return demo ads
    if not all_cse:
        if pois:
            logger.warning("Google CSE returned no results. Using POI-based offline ads fallback.")
            return build_ads_from_pois(pois, interest=interest, limit=5)
        logger.warning("Google CSE returned no results. Using SAMPLE_ADS demo fallback.")
        return build_fallback_ads(interest)



    # 3) build prompt
    pois_short = [{"name": p.name, "address": p.address or "", "lat": p.lat, "lon": p.lon, "category": p.category or ""} for p in pois]
    cse_short = [{"title": it.title or "", "snippet": it.snippet or "", "link": it.link or ""} for it in all_cse[:50]]

    prompt = (
        "User interest: " + interest + "\n\n"
        "Nearby POIs (name/address/lat/lon/category):\n" + json.dumps(pois_short, ensure_ascii=False, indent=2) + "\n\n"
        "Google search results (title/snippet/link):\n" + json.dumps(cse_short, ensure_ascii=False, indent=2) + "\n\n"
        "Task:\n"
        "1) Select up to 5 results that are LOCAL to the given POIs/city only. Ignore foreign-country or unrelated city results.\n"
        "2) Prioritize OFFLINE nearby shops/restaurants/stores first, online options only after local offline options.\n"
        "3) Each ad_text must include a concrete offer phrase like 'Flat 20% Off', 'Buy 1 Get 1', or 'Up to 30% Off'.\n"
        "2) For each selected result, produce a short ad with the following JSON structure:\n"
        "{\"title\":\"...\",\"ad_text\":\"...\",\"source_link\":\"...\"}\n"
        "3) Return only a JSON array of up to 5 objects in that exact structure. Do not add commentary."
    )

    # 4) call LLM
    try:
        llm_raw = call_gemini_generate(prompt)
    except Exception as e:
        logger.warning("LLM generation failed; using CSE-only fallback. Error: %s", e)
        return build_ads_from_cse(all_cse, interest=interest, limit=5)

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
        if not isinstance(parsed, list):
            logger.warning("LLM output was not a JSON array; using CSE-only fallback.")
            return build_ads_from_cse(all_cse, interest=interest, limit=5)
    except Exception as e:
        # fallback — return single raw blob as ad
        logger.warning("LLM parse failed; using CSE-only fallback. Error: %s", e)
        return build_ads_from_cse(all_cse, interest=interest, limit=5)

    # 6) For each ad, try to map coordinates:
    out_ads: List[TopAd] = []
    for idx, obj in enumerate(parsed[:5]):
        if not isinstance(obj, dict):
            continue
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

        # If no location could be inferred from content/link, anchor to nearby POIs
        # so results remain local and render as offline map markers.
        if (ad_lat is None or ad_lon is None) and pois:
            poi = pois[idx % len(pois)]
            ad_lat = poi.lat
            ad_lon = poi.lon

        final_text = ad_text or f"Nearby {interest} offer. Offer: {choose_offer_text(idx)}."
        if "off" not in final_text.lower() and "buy 1 get 1" not in final_text.lower() and "bogo" not in final_text.lower():
            final_text = f"{final_text} Offer: {choose_offer_text(idx)}."

        out_ads.append(TopAd(title=title or "Untitled", ad_text=final_text, source_link=link, lat=ad_lat, lon=ad_lon))

    if not out_ads:
        logger.warning("LLM produced no usable ad objects; using CSE-only fallback.")
        fallback_ads = build_ads_from_cse(all_cse, interest=interest, limit=5)
        if pois:
            for i, ad in enumerate(fallback_ads):
                if i < len(pois):
                    ad.lat = pois[i].lat
                    ad.lon = pois[i].lon
        return fallback_ads

    # Ensure at least some results show as offline/local if POIs are available.
    if pois and not any(ad.lat is not None and ad.lon is not None for ad in out_ads):
        for i, ad in enumerate(out_ads):
            if i >= len(pois):
                break
            ad.lat = pois[i].lat
            ad.lon = pois[i].lon

    # Offline first: keep local/map-able ads before pure online ads.
    out_ads.sort(key=lambda a: 0 if (a.lat is not None and a.lon is not None) else 1)

    return out_ads

# simple health
@app.get("/health")
def health():
    return {
        "ok": True,
        "google_search": {"key_set": bool(GOOGLE_SEARCH_API_KEY), "cse_set": bool(GOOGLE_CSE_ID)},
        "gemini": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
