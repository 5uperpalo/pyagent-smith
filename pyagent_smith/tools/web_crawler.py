"""Web crawler tool using Crawl4AI with Ollama LLM for extracting vendor information."""

import asyncio
import json
from typing import List

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, LLMConfig, LLMExtractionStrategy
from langchain.tools import tool

from pyagent_smith.tools.web_crawler_utils.location import LocationInfo
from pyagent_smith.llm import get_ollama_settings


async def _crawl_locations_async(url: str) -> List[dict]:
    """
    Asynchronously crawl a website and extract location information using Ollama.

    Args:
        url: The URL of the website to crawl

    Returns:
        List of vendor dictionaries with vendor_name and description
    """
    # Get Ollama settings
    settings = get_ollama_settings()
    base_url = settings.get("base_url")
    model = settings.get("model", "functiongemma")

    if not base_url:
        raise ValueError("Ollama not configured. Provide base_url via env vars or config.json")

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider=f"ollama/{model}",  # Model is part of provider string
            base_url=base_url,  # Pass base_url separately
        ),
        schema=LocationInfo.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "Extract all location names, descriptions from the https://www.thecrag.com/ page. "
            "Look for location names, descriptions on the page. "
            "For each location found, extract: location_name (REQUIRED - this is the location name), "
            "location_description (description of the location), "
            "Extract ALL locations listed on the page, even if some fields are missing. "
            "If you see any location names, extract them. Return a list of location objects."
        ),
        input_format="markdown",
        verbose=True,  # Enable verbose to see what's happening
    )

    # Initialize and run the crawler
    # Add delay to ensure JavaScript content loads
    # Note: Try with headless=False first to see if page blocks headless browsers
    # If that works, we can switch back to headless=True
    async with AsyncWebCrawler(headless=False, verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=llm_strategy,
                page_timeout=60000,  # 60 second timeout
                delay_before_return_html=10.0,  # Wait 10 seconds for JS to render
                # Don't use CSS selector initially - let's see all content first
                # If we get content but wrong area, we can add selector back
                # css_selector="main, [role='main'], .results, .supplier-list, .company-list",
                js_code="""
                // Wait for page to fully load
                await new Promise(resolve => {
                    if (document.readyState === 'complete') {
                        setTimeout(resolve, 5000);
                    } else {
                        window.addEventListener('load', () => {
                            setTimeout(resolve, 5000);
                        });
                        setTimeout(resolve, 15000); // Fallback timeout
                    }
                });

                // Try to scroll to trigger lazy loading and infinite scroll
                window.scrollTo(0, document.body.scrollHeight / 4);
                await new Promise(resolve => setTimeout(resolve, 3000));
                window.scrollTo(0, document.body.scrollHeight / 2);
                await new Promise(resolve => setTimeout(resolve, 3000));
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(resolve => setTimeout(resolve, 3000));

                // Check if we have any supplier/company listings
                const hasContent = document.body.innerText.length > 100;
                console.log('Page content length:', document.body.innerText.length);
                const hasListings = document.body.innerText.includes('supplier') ||
                    document.body.innerText.includes('company') ||
                    document.body.innerText.includes('vendor');
                console.log('Has supplier listings:', hasListings);
                """,
            ),
        )

        if not result.success:
            error_msg = result.error_message or "Unknown error"
            raise Exception(f"Crawler error: {error_msg}")

        # Check if we have extracted content
        if not result.extracted_content:
            # Check if we have markdown or html to debug
            has_markdown = bool(result.markdown)
            has_html = bool(result.html)
            markdown_preview = result.markdown[:2000] if result.markdown else "None"
            markdown_length = len(result.markdown) if result.markdown else 0
            html_length = len(result.html) if result.html else 0
            html_preview = result.html[:1000] if result.html else "None"
            raise Exception(
                f"No extracted content returned. "
                f"Crawl succeeded but extraction returned empty. "
                f"Has markdown: {has_markdown} (length: {markdown_length}), Has HTML: {has_html} (length: {html_length}). "
                f"Markdown preview (first 2000 chars): {markdown_preview}... "
                f"HTML preview (first 1000 chars): {html_preview}... "
                f"If markdown/HTML lengths are very small (1-10 chars), the page content is not being captured. "
                f"This might indicate: (1) Page blocks headless browsers, (2) Requires authentication, "
                f"(3) Content loaded via API calls, or (4) Anti-bot protection."
            )

        # Parse the extracted content
        try:
            extracted_data = json.loads(result.extracted_content)
        except json.JSONDecodeError as e:
            # Log what we got for debugging
            content_preview = result.extracted_content[:1000] if result.extracted_content else "None"
            raise Exception(
                f"Failed to parse extracted content as JSON. "
                f"Content preview (first 1000 chars): {content_preview}... "
                f"Error: {str(e)}. "
                f"Full content length: {len(result.extracted_content) if result.extracted_content else 0} chars."
            )

        # Handle both single object and list of objects
        if isinstance(extracted_data, dict):
            # If it's a single vendor object, wrap it in a list
            if "vendor_name" in extracted_data:
                extracted_data = [extracted_data]
            else:
                # If it's a wrapper object, try to find the list
                # Common wrapper keys: vendors, data, results, items
                extracted_data = (
                    extracted_data.get("vendors")
                    or extracted_data.get("data")
                    or extracted_data.get("results")
                    or extracted_data.get("items")
                    or []
                )
                if not isinstance(extracted_data, list):
                    # If still not a list, check if it's a single vendor wrapped in another key
                    for key in extracted_data.keys():
                        if isinstance(extracted_data[key], list):
                            extracted_data = extracted_data[key]
                            break
                    else:
                        extracted_data = []
        elif not isinstance(extracted_data, list):
            extracted_data = []

        # Filter to ensure we only return valid vendor objects
        valid_vendors = []
        for item in extracted_data:
            if isinstance(item, dict) and "vendor_name" in item:
                valid_vendors.append(item)

        # If no valid vendors found, raise an exception with debug info
        if not valid_vendors:
            # Try to understand what we got and what content was available
            debug_info = []
            debug_info.append(f"Extracted data type: {type(extracted_data)}")
            if isinstance(extracted_data, list):
                debug_info.append(f"List length: {len(extracted_data)}")
                if extracted_data:
                    debug_info.append(f"First item type: {type(extracted_data[0])}")
                    if isinstance(extracted_data[0], dict):
                        debug_info.append(f"First item keys: {list(extracted_data[0].keys())[:10]}")
            elif isinstance(extracted_data, dict):
                debug_info.append(f"Dict keys: {list(extracted_data.keys())[:10]}")
            content_len = len(result.extracted_content) if result.extracted_content else 0
            debug_info.append(f"Raw extracted content length: {content_len}")

            # Show what markdown content was available to the LLM
            markdown_preview = result.markdown[:3000] if result.markdown else "None"
            markdown_length = len(result.markdown) if result.markdown else 0
            debug_info.append(f"Markdown content length: {markdown_length}")

            error_msg = (
                f"No valid vendors found after parsing. "
                f"Debug info: {'; '.join(debug_info)}. "
                f"The LLM returned an empty list [], which means it didn't find any vendors matching the schema. "
                f"Markdown preview (first 3000 chars) sent to LLM: {markdown_preview}... "
                f"If the markdown doesn't contain vendor information, the page might require: "
                f"(1) Different CSS selectors to target the right content, "
                f"(2) Longer wait times for JavaScript to render, or "
                f"(3) The page content might be loaded dynamically via API calls."
            )
            raise Exception(error_msg)

        return valid_vendors


@tool("web_crawler_extract_locations", return_direct=False)
def web_crawler(url: str) -> str:
    """Extract location names and descriptions from a website.

    This tool crawls the specified URL and uses Ollama LLM to extract location
    information including location names and descriptions from the page content.

    Parameters:
    - url: The URL of the website listing locations to crawl

    Returns:
    - str: A formatted string listing all extracted locations with their names and descriptions,
           or an error message if extraction fails
    """
    try:
        # Run the async crawler
        locations = asyncio.run(_crawl_locations_async(url))

        if not locations:
            return (
                "No locations were found on the specified page. "
                "This could mean: (1) The page structure is different than expected, "
                "(2) The LLM extraction didn't find matching location information, "
                "(3) The page requires JavaScript rendering that didn't complete, or "
                "(4) The page content is protected/blocked. "
                "Try checking the page manually to see if location listings are visible."
            )

        # Format the results as a readable string
        formatted_results = []
        formatted_results.append(f"Found {len(locations)} location(s):\n")

        for idx, location in enumerate(locations, 1):
            location_name = location.get("location_name", "Unknown")
            formatted_results.append(f"{idx}. {location_name}")

            description = location.get("location_description")
            if description:
                formatted_results.append(f"   Description: {description}")

            formatted_results.append("")  # Empty line between locations

        return "\n".join(formatted_results)

    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        # Check if it's a Playwright browser installation error
        if "playwright" in error_msg.lower() or "chromium" in error_msg.lower() or "browser" in error_msg.lower():
            return (
                f"Web crawler error: {error_msg}\n\n"
                f"To fix this issue, install Playwright browsers by running:\n"
                f"  playwright install chromium\n\n"
                f"Or install all browsers:\n"
                f"  playwright install"
            )
        return f"Web crawler error: {error_msg}"
