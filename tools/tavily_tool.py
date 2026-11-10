"""Web search tool backed by the Tavily Search API.

Reads the Tavily key from the environment (.env) and returns up to 5
search results, each with title, url, content and score.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5


def tavily_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search the web with Tavily.

    Args:
        query: What to search for, e.g. "best things to do in Lisbon".
        max_results: Max number of results to return (capped at 5).

    Returns:
        A list of up to 5 dicts with keys: title, url, content, score.
        On failure, a single-item list with an "error" key.
    """
    # .env uses "tavily_API_KEY"; also accept the conventional upper-case name.
    api_key = os.getenv("tavily_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "tavily_API_KEY is not set in the environment."}]

    max_results = max(1, min(max_results, MAX_RESULTS))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }

    try:
        response = requests.post(TAVILY_URL, json=body, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return [{"error": f"Request to Tavily failed: {exc}"}]
    except ValueError:
        return [{"error": "Tavily returned a non-JSON response."}]

    results = []
    for item in payload.get("results") or []:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

        if len(results) >= max_results:
            break

    return results


if __name__ == "__main__":
    import json

    print(json.dumps(tavily_search("top attractions in Paris"), indent=2))
