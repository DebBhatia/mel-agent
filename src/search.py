"""
WEB SEARCH MODULE
==================
DuckDuckGo-powered web search for Mel.
Free, no API key required.

Install: pip install duckduckgo-search
"""

import logging
from typing import Optional

logger = logging.getLogger("search")


class WebSearch:
    """Performs web searches via DuckDuckGo (free, no key needed)."""

    @staticmethod
    async def search(query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web for query. Returns list of results with title, url, snippet.
        Falls back gracefully if duckduckgo-search is not installed.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return []

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("url", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                    })
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    @staticmethod
    async def news_search(query: str, max_results: int = 5) -> list[dict]:
        """Search DuckDuckGo News."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.news(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
                    })
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo news search failed: {e}")
            return []
