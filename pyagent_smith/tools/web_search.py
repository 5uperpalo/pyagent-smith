import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List
from urllib.parse import urlparse

from ddgs import DDGS
from langchain.tools import tool

# Suppress the harmless asyncio cleanup error that occurs when event loop closes
# before subprocess transport cleanup completes (common in Jupyter notebooks)
_original_stderr_write = sys.stderr.write
_filter_buffer = ""


def _filter_stderr_write(data):
    """Filter out the harmless asyncio cleanup error."""
    global _filter_buffer
    if isinstance(data, str):
        # Buffer recent writes to handle multi-line error messages
        _filter_buffer += data
        # Keep only last 2000 chars to avoid memory issues
        if len(_filter_buffer) > 2000:
            _filter_buffer = _filter_buffer[-2000:]

        # Check if this is the asyncio cleanup error (both patterns must be present)
        if (
            "Exception ignored in: <function BaseSubprocessTransport.__del__" in _filter_buffer
            and "RuntimeError: Event loop is closed" in _filter_buffer
        ):
            # Clear buffer and suppress this specific error - it's harmless
            _filter_buffer = ""
            return

        # If we haven't seen the error pattern in a while, clear old buffer
        if len(_filter_buffer) > 500 and "BaseSubprocessTransport" not in _filter_buffer[-500:]:
            _filter_buffer = _filter_buffer[-100:]

    _original_stderr_write(data)


# Only apply the filter if we're in an environment where this error is common (Jupyter)
try:
    import IPython
    if IPython.get_ipython() is not None:
        # Store original write function and replace with filtered version
        _original_stderr_write_func = sys.stderr.write
        # Replace stderr.write with our filtered version
        object.__setattr__(sys.stderr, 'write', _filter_stderr_write)
except (ImportError, AttributeError):
    pass


def _normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing protocol, www, and trailing slashes."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.rstrip("/")
        return f"{domain}{path}".lower()
    except Exception:
        return url.lower()


def _search_with_backend(query: str, max_results: int, backend: str) -> List[Dict]:
    """Search using a specific DuckDuckGo backend.

    Args:
        query: The search query string
        max_results: Maximum number of results to return
        backend: Backend to use ("bing", "brave", "duckduckgo", "auto")

    Returns:
        List of search result dictionaries
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, backend=backend))
        # Small delay to allow subprocess cleanup before event loop might close
        time.sleep(0.01)
        return results if results else []
    except Exception:
        # Return empty list on error to allow other backends to succeed
        return []


def _majority_vote_results(all_results: List[List[Dict]], max_results: int = 5) -> List[Dict]:
    """Apply majority voting to aggregate results from multiple search engines.

    Args:
        all_results: List of result lists from different search engines
        max_results: Maximum number of results to return

    Returns:
        Aggregated and ranked results based on majority voting
    """
    # Count occurrences of each URL across all search engines
    url_votes: Dict[str, int] = Counter()
    url_to_results: Dict[str, List[Dict]] = defaultdict(list)

    for engine_results in all_results:
        seen_urls = set()
        for result in engine_results:
            url = result.get("href", "")
            if url:
                normalized_url = _normalize_url(url)
                if normalized_url not in seen_urls:
                    url_votes[normalized_url] += 1
                    url_to_results[normalized_url].append(result)
                    seen_urls.add(normalized_url)

    # Sort by vote count (descending), then by first occurrence
    sorted_urls = sorted(
        url_votes.items(),
        key=lambda x: (x[1], -len(url_to_results[x[0]][0].get("title", ""))),
        reverse=True,
    )

    # Aggregate results: take the best result for each URL (highest vote count)
    aggregated_results = []
    seen_normalized = set()

    for normalized_url, vote_count in sorted_urls[: max_results * 2]:  # Get more candidates
        if normalized_url in seen_normalized:
            continue

        # Take the first result from the list (they're similar)
        best_result = url_to_results[normalized_url][0].copy()
        # Add vote count as metadata
        best_result["_vote_count"] = vote_count
        aggregated_results.append(best_result)
        seen_normalized.add(normalized_url)

        if len(aggregated_results) >= max_results:
            break

    return aggregated_results


@tool("web_search", return_direct=False)
def web_search_tool(
    query: str,
    max_results: int = 5,
) -> str:
    """Perform a web search using DuckDuckGo Search.

    Use this tool to search the web for information. The query parameter should be a search string,
    not a JSON object or schema. For example: query="rock climbing in Montserrat" not query="{{'location': 'Montserrat'}}".

    Args:
        query: The search query string (e.g., "rock climbing in Montserrat" or "Siurana climbing routes")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string containing search results with titles, URLs, and descriptions
    """
    try:
        # Single backend search (original behavior)
        with DDGS() as ddgs:
            aggregated_results = list(ddgs.text(query, max_results=max_results, backend="auto"))
        # Small delay to allow subprocess cleanup before event loop might close
        time.sleep(0.01)

        if not aggregated_results:
            return f"No results found for query: {query}"

        # Format results as a readable string
        formatted_results = []
        for idx, result in enumerate(aggregated_results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            href = result.get("href", "No URL")

            result_str = f"{idx}. {title}\n   URL: {href}\n   {body}"
            result_str += "\n"

            formatted_results.append(result_str)

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool("web_search_with_self_consistency", return_direct=False)
def web_search_tool_with_self_consistency(
    query: str,
    max_results: int = 5,
) -> str:
    """Perform a web search with self-consistency using multiple search engines.

    Use this tool to search the web for information using multiple backends for more reliable results.
    The query parameter should be a search string, not a JSON object or schema.
    For example: query="rock climbing in Montserrat" not query="{{'location': 'Montserrat'}}".

    Args:
        query: The search query string (e.g., "rock climbing in Montserrat" or "Siurana climbing routes")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string containing search results with titles, URLs, descriptions, and consensus scores
    """
    try:
        # Use multiple backends for self-consistency
        backends = ["bing", "brave", "duckduckgo"]
        all_results = []

        for backend in backends:
            results = _search_with_backend(query, max_results=max_results * 2, backend=backend)
            if results:
                all_results.append(results)

        num_engines = len(all_results)
        if not all_results:
            return f"No results found for query: {query}"

        # Apply majority voting
        aggregated_results = _majority_vote_results(all_results, max_results=max_results)

        if not aggregated_results:
            return f"No results found for query: {query}"

        # Format results as a readable string
        formatted_results = []
        for idx, result in enumerate(aggregated_results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            href = result.get("href", "No URL")
            vote_count = result.get("_vote_count", "")

            result_str = f"{idx}. {title}\n   URL: {href}\n   {body}"
            if vote_count:
                result_str += f"\n   (Consensus: {vote_count}/{num_engines} engines)"
            result_str += "\n"

            formatted_results.append(result_str)

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"
