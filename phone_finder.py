# Standalone module to find Company Phone Numbers using Company Name or Domain
# Integrates SearXNG search and URL validation.

import json
import re
import sys
import os
import argparse
import logging
import time
import requests
from urllib.parse import urlparse, quote_plus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Integrated URL Sanitization Logic (from url_validator.py) ---
def sanitize_url(url):
    """
    Clean and sanitize URLs, extracting just the domain without 'www.'.

    Args:
        url (str): The URL to sanitize

    Returns:
        str or None: The sanitized domain (e.g., 'google.com') or None if invalid/error.
                     Returns None instead of "null" string for better programmatic use.
    """
    if not isinstance(url, str) or not url.strip():
        logger.warning("Invalid URL input: empty or not a string.")
        return None

    # Ensure URL starts with a scheme for reliable parsing
    if not url.startswith('http://') and not url.startswith('https://'):
        if url.startswith('//'):
            url = 'https:' + url
        else:
            url = 'https://' + url

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain.lower().startswith('www.'):
            domain = domain[4:]

        domain = re.sub(r':\d+$', '', domain) # Remove port

        # Basic check for a valid domain structure
        if not domain or '.' not in domain or len(domain.split('.')[-1]) < 2:
            logger.warning(f"URL '{url}' resulted in invalid domain '{domain}'")
            return None

        return domain.lower()
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None

# --- Integrated SearXNG Client Logic (simplified from searx.py) ---
class SearXNGClient:
    """Handles communication with a SearXNG instance."""
    def __init__(self, base_url):
        if not base_url or not base_url.startswith(('http://', 'https://')):
             raise ValueError("Invalid SearXNG base URL provided.")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        })
        logger.debug(f"SearXNGClient initialized with base URL: {self.base_url}")

    def search(self, query, max_pages=3, timeout=10):
        """Search SearXNG and return simplified results."""
        all_results = []
        page = 1
        last_error = None
        logger.debug(f"Searching SearXNG for '{query}' up to {max_pages} pages.")
        while page <= max_pages:
            encoded_query = quote_plus(query)
            url = f"{self.base_url}/search?q={encoded_query}&format=json&pageno={page}"
            logger.debug(f"Fetching page {page}: {url}")
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()

                page_results = data.get('results', [])
                if page_results:
                    logger.debug(f"Got {len(page_results)} results from page {page}.")
                    for result in page_results:
                        simplified = {
                            'title': result.get('title', ''),
                            'url': result.get('url', ''),
                            'content': result.get('content', '')
                            # No score needed for phone finding
                        }
                        all_results.append(simplified)
                    page += 1
                    time.sleep(0.2) # Small polite delay
                else:
                    logger.debug(f"No more results found on page {page}.")
                    break
            except requests.exceptions.RequestException as e:
                last_error = f"SearXNG request failed page {page}: {e}"
                logger.error(last_error)
                break
            except json.JSONDecodeError as e:
                last_error = f"SearXNG JSON decode error page {page}: {e}"
                logger.error(last_error)
                break
            except Exception as e:
                last_error = f"Unexpected SearXNG error page {page}: {e}"
                logger.error(last_error)
                break # Stop fetching on unexpected error

        result_data = {"query": query, "num_results": len(all_results), "results": all_results}
        if last_error and not all_results:
            result_data["error"] = last_error

        logger.debug(f"SearXNG search completed. Total results: {len(all_results)}")
        return result_data

def search_web_standalone(query, searx_base_url, pages=3, max_retries=2):
    """Standalone web search function using the provided SearXNG instance URL."""
    if not searx_base_url:
        logger.error("No SearXNG base URL provided for search.")
        return json.dumps({"query": query, "num_results": 0, "results": [], "error": "Missing SearXNG URL"})

    try:
        client = SearXNGClient(searx_base_url)
    except ValueError as e:
        logger.error(f"Failed to initialize SearXNGClient: {e}")
        return json.dumps({"query": query, "num_results": 0, "results": [], "error": str(e)})

    attempt = 0
    results_dict = {}
    while attempt < max_retries:
        logger.debug(f"Search attempt {attempt + 1}/{max_retries} for query: '{query}'")
        results_dict = client.search(query, max_pages=pages)

        if results_dict["num_results"] > 0:
            logger.info(f"SearXNG search successful for '{query}' on attempt {attempt+1}")
            return json.dumps(results_dict)
        elif "error" in results_dict:
             logger.warning(f"SearXNG search error for '{query}' on attempt {attempt+1}: {results_dict['error']}")
        else:
            logger.warning(f"SearXNG search for '{query}' yielded 0 results on attempt {attempt+1}")

        attempt += 1
        if attempt < max_retries:
            delay = 1 * (2 ** (attempt - 1)) # Exponential backoff: 1s, 2s
            logger.info(f"Retrying SearXNG search after {delay:.1f}s...")
            time.sleep(delay)

    # If loop finishes without returning, it failed
    final_error = results_dict.get("error", "No search results found after multiple attempts.")
    no_results_data = {
        "query": query,
        "num_results": 0,
        "results": [],
        "error": final_error
    }
    logger.error(f"SearXNG search failed permanently for '{query}': {final_error}")
    return json.dumps(no_results_data)

# --- Core Phone Finder Logic ---
class PhoneNumberSearcher:
    def __init__(self, searx_base_url):
        if not searx_base_url:
            raise ValueError("SearXNG base URL must be provided.")
        self.searx_base_url = searx_base_url
        # Regex focused on common North American formats. Can be expanded.
        # Allows optional +1, separators like . - space, optional parens around area code.
        self.phone_pattern = re.compile(r'''
            (?:(?:\+?1[-.\s]?)?)            # Optional +1 and separator
            (?:(\()?\s*[2-9]\d{2}\s*(\))?|([2-9]\d{2})) # Area code: (ddd) or ddd (capturing groups help later if needed)
            [-.\s]?                         # Optional separator
            ([2-9]\d{2})                    # Exchange code (cannot start with 0 or 1)
            [-.\s]?                         # Optional separator
            (\d{4})                         # Subscriber number
            # Lookbehind/ahead could potentially exclude numbers within longer sequences, but keep simple for now
        ''', re.VERBOSE)


    def search_with_searx(self, query, pages=3):
        """Search using integrated standalone searx function."""
        results_json = search_web_standalone(query, self.searx_base_url, pages=pages)
        try:
            data = json.loads(results_json)
            return data.get("results", [])
        except json.JSONDecodeError:
            logger.error(f"Failed to decode search results JSON for query: {query}")
            return []

    def extract_phone_numbers(self, text):
        """Extract phone numbers using regex from a given text block."""
        if not text or not isinstance(text, str):
            return []
        
        found_numbers = []
        matches = self.phone_pattern.finditer(text)
        for match in matches:
            # Format the matched number immediately
            formatted = self.format_phone_number(match.group(0))
            if formatted: # Ensure formatting was successful
                 found_numbers.append(formatted)
        return found_numbers

    def format_phone_number(self, phone_number):
        """Format the phone number consistently to +1-XXX-XXX-XXXX."""
        if not phone_number: return None
        digits = re.sub(r'\D', '', phone_number) # Remove all non-digits

        if len(digits) == 10:
            # Assume US number, add +1
            return f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        elif len(digits) == 11 and digits.startswith('1'):
            # Already has US country code
            return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:11]}"
        elif len(digits) > 11 and digits.startswith('1'):
            # Potential number with extension - truncate? Or ignore? For now, ignore >11 digits.
             logger.debug(f"Ignoring likely invalid or extended number: {phone_number} -> {digits}")
             return None
        else:
            # Doesn't match expected US/Canada format length
            logger.debug(f"Ignoring number with unexpected length: {phone_number} -> {digits}")
            return None

    def is_likely_domain(self, input_text):
        """Check if input looks like a domain or URL."""
        # Basic check: contains a dot, doesn't have too many spaces
        if '.' in input_text and ' ' not in input_text.strip().split('.')[-1]:
             # Check if it parses as a URL with a netloc
            try:
                parsed = urlparse('https://' + input_text if not input_text.startswith(('http','//')) else input_text)
                if parsed.netloc:
                    # Further check TLD pattern if desired
                    return bool(re.search(r'\.[a-zA-Z]{2,}$', parsed.netloc))
            except Exception:
                return False
        return input_text.startswith(('http://', 'https://', '//'))


    def find_phone_numbers(self, company_input):
        """Find phone numbers for the given company name or domain."""
        target_name = company_input
        target_domain = None

        if self.is_likely_domain(company_input):
            logger.info(f"Input '{company_input}' detected as domain/URL.")
            sanitized = sanitize_url(company_input)
            if sanitized:
                target_domain = sanitized
                # Use a potentially cleaner name from the domain for the query
                target_name = target_domain.split('.')[0] # e.g., 'google' from 'google.com'
                logger.info(f"Sanitized domain: {target_domain}. Using '{target_name}' for search name.")
            else:
                logger.warning(f"Failed to sanitize domain '{company_input}', using original input as name.")
                target_name = company_input # Fallback to original input if sanitization fails
        else:
             logger.info(f"Input '{company_input}' treated as company name.")

        # Construct search query
        # Prioritize "contact" and official site if domain known
        if target_domain:
             queries = [
                 f'"{target_name}" contact phone number site:{target_domain}',
                 f'"{target_name}" phone number customer service',
                 f'"{target_name}" "contact us"',
             ]
        else:
            queries = [
                 f'"{target_name}" contact phone number',
                 f'"{target_name}" customer service phone',
                 f'"{target_name}" "contact us"',
            ]

        all_phone_numbers = set() # Use a set to automatically handle duplicates

        for query in queries:
            logger.info(f"Running query: {query}")
            results = self.search_with_searx(query)

            if not results:
                logger.warning(f"No results found for query: {query}")
                continue

            for result in results:
                # Extract from title and content
                text_to_scan = (result.get('title', '') or '') + " " + (result.get('content', '') or '')
                numbers_found = self.extract_phone_numbers(text_to_scan)
                if numbers_found:
                    logger.debug(f"Found numbers {numbers_found} in result for query '{query}'")
                    all_phone_numbers.update(numbers_found)

            # Optional: Add a small delay between different query types if needed
            # time.sleep(0.5)

        # Convert set back to list for ordered JSON output (sets are unordered)
        unique_phone_numbers = sorted(list(all_phone_numbers))

        logger.info(f"Found {len(unique_phone_numbers)} unique phone number(s).")
        return unique_phone_numbers # Return list directly

# --- Public Function Interface ---
def get_company_phone_numbers(company_input, searx_url):
    """
    Main function to get phone numbers for a company/domain.

    Args:
        company_input (str): Company name or domain/URL.
        searx_url (str): The base URL of the SearXNG instance.

    Returns:
        str: JSON string containing a list of found phone numbers under the key 'phone_numbers'.
             Example: '{"phone_numbers": ["+1-XXX-XXX-XXXX", "+1-YYY-YYY-YYYY"]}'
             Returns '{"phone_numbers": []}' if none found or on error.
    """
    try:
        searcher = PhoneNumberSearcher(searx_base_url=searx_url)
        phone_numbers = searcher.find_phone_numbers(company_input)
        # Return JSON list
        return json.dumps({"phone_numbers": phone_numbers})
    except ValueError as e:
        logger.error(f"Initialization error: {e}")
        return json.dumps({"phone_numbers": [], "error": str(e)})
    except Exception as e:
        logger.exception(f"An unexpected error occurred in get_company_phone_numbers: {e}")
        return json.dumps({"phone_numbers": [], "error": "An unexpected error occurred"})


# --- Command-Line Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find phone numbers for a company using SearXNG.")
    parser.add_argument("company_input", help="The company name or domain/URL.")
    parser.add_argument("--searx-url", required=True, help="Base URL of the SearXNG instance (e.g., http://localhost:8080)")
    parser.add_argument("--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Set the logging level.")

    args = parser.parse_args()

    # Update log level from args
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    logger.setLevel(log_level_map[args.log_level])
    logging.getLogger().setLevel(log_level_map[args.log_level]) # Set root logger level too

    print(f"Searching phone numbers for: '{args.company_input}'")
    print(f"Using SearXNG at: {args.searx_url}")

    start_time = time.time()
    result_json = get_company_phone_numbers(
        company_input=args.company_input,
        searx_url=args.searx_url
    )
    end_time = time.time()

    print("\n--- Result ---")
    # Pretty print the JSON result
    try:
        parsed_result = json.loads(result_json)
        print(json.dumps(parsed_result, indent=2))
        if "error" in parsed_result:
            print(f"\nError during search: {parsed_result['error']}")
        elif not parsed_result.get("phone_numbers"):
             print("\nNo phone numbers found.")

    except json.JSONDecodeError:
        print("Error: Failed to parse the final JSON result.")
        print(f"Raw output: {result_json}")

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
