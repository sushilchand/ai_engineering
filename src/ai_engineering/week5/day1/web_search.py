import json
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_engineering.constants import APIFY_KEY


def web_search(query: str) -> str:
    if not APIFY_KEY:
        raise RuntimeError("APIFY_KEY is not configured")

    endpoint = "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items"
    url = f"{endpoint}?{urlencode({'token': APIFY_KEY})}"
    payload = json.dumps({"queries": query, "maxPagesPerQuery": 1}).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        try:
            with urlopen(
                request, timeout=60, context=ssl.create_default_context()
            ) as response:
                search_data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            with urlopen(
                request,
                timeout=60,
                context=ssl._create_unverified_context(),
            ) as response:
                search_data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Apify web search failed: {error}") from error

    results = [
        result for page in search_data for result in page.get("organicResults", [])
    ]
    if not results:
        return "No search results found."

    return "\n\n".join(
        f"{result.get('title', 'Untitled')}\n{result.get('url', '')}\n{result.get('description', '')}"
        for result in results[:5]
    )


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information and return the top results.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
