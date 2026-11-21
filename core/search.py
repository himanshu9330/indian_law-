"""
Web search fallback module using SerpAPI.
Restricts results specifically to India legal context (google.co.in, gl=in, hl=en).
Gracefully handles missing keys or connectivity issues.
"""

import os
from typing import Any, Optional
from core.config import SERPAPI_API_KEY


def search_web(
    query: str,
    api_key: Optional[str] = None,
    num_results: int = 3,
) -> dict[str, Any]:
    """
    Search Indian legal web sources via SerpAPI.
    Returns:
      {
        "web_context": str,
        "results": list[dict], # list of {'title', 'link', 'snippet', 'source'}
        "success": bool
      }
    """
    key = api_key or SERPAPI_API_KEY or os.getenv("SERPAPI_API_KEY")
    if not key or key.strip() in ("", "your_serpapi_api_key_here"):
        return {
            "web_context": "Web search skipped: SERPAPI_API_KEY not configured.",
            "results": [],
            "success": False,
        }

    try:
        from serpapi import GoogleSearch

        params = {
            "engine": "google",
            "q": query,
            "api_key": key.strip(),
            "num": num_results,
            "gl": "in",                      # Restrict result origin to India
            "hl": "en",                      # Interface in English
            "google_domain": "google.co.in", # Use Indian Google domain
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        web_context_lines = []
        structured_results = []

        for i, res in enumerate(organic_results):
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            link = res.get("link", "")
            source = res.get("source", "")

            web_context_lines.append(f"Source {i+1}: {title}\nSnippet: {snippet}")
            structured_results.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "source": source or "Web",
            })

        web_context = "\n\n".join(web_context_lines)
        if not web_context:
            web_context = "No specific Indian legal web results found."

        return {
            "web_context": web_context,
            "results": structured_results,
            "success": True,
        }
    except Exception as e:
        return {
            "web_context": f"Search error: {str(e)}",
            "results": [],
            "success": False,
        }
