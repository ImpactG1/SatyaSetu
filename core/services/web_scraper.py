"""
Web Scraper Service — Real-Time Source Verification
Searches the web for claims and scrapes content from real news sources
(like Google AI mode) for cross-referencing and source attribution.

Supports: DuckDuckGo search, direct URL scraping, multi-source extraction.
"""

import os
import re
import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ── Known Indian & International news domains for source identification ──
KNOWN_SOURCES = {
    # Indian News
    'hindustantimes.com': {'name': 'Hindustan Times', 'credibility': 8.0, 'type': 'mainstream'},
    'timesofindia.indiatimes.com': {'name': 'Times of India', 'credibility': 7.5, 'type': 'mainstream'},
    'ndtv.com': {'name': 'NDTV', 'credibility': 8.0, 'type': 'mainstream'},
    'thehindu.com': {'name': 'The Hindu', 'credibility': 8.5, 'type': 'mainstream'},
    'indianexpress.com': {'name': 'Indian Express', 'credibility': 8.0, 'type': 'mainstream'},
    'livemint.com': {'name': 'Livemint', 'credibility': 7.5, 'type': 'mainstream'},
    'news18.com': {'name': 'News18', 'credibility': 7.0, 'type': 'mainstream'},
    'indiatoday.in': {'name': 'India Today', 'credibility': 7.5, 'type': 'mainstream'},
    'deccanherald.com': {'name': 'Deccan Herald', 'credibility': 7.5, 'type': 'mainstream'},
    'tribuneindia.com': {'name': 'Tribune India', 'credibility': 7.5, 'type': 'mainstream'},
    'theprint.in': {'name': 'The Print', 'credibility': 7.5, 'type': 'mainstream'},
    'thewire.in': {'name': 'The Wire', 'credibility': 7.0, 'type': 'mainstream'},
    'scroll.in': {'name': 'Scroll.in', 'credibility': 7.0, 'type': 'mainstream'},

    # Indian Fact-checkers
    'altnews.in': {'name': 'AltNews', 'credibility': 9.0, 'type': 'fact_checker'},
    'boomlive.in': {'name': 'BOOM Live', 'credibility': 9.0, 'type': 'fact_checker'},
    'factly.in': {'name': 'Factly', 'credibility': 8.5, 'type': 'fact_checker'},
    'vishvasnews.com': {'name': 'Vishvas News', 'credibility': 8.5, 'type': 'fact_checker'},
    'thequint.com': {'name': 'The Quint', 'credibility': 7.5, 'type': 'mainstream'},
    'newschecker.in': {'name': 'Newschecker', 'credibility': 8.5, 'type': 'fact_checker'},

    # International
    'reuters.com': {'name': 'Reuters', 'credibility': 9.5, 'type': 'mainstream'},
    'apnews.com': {'name': 'Associated Press', 'credibility': 9.5, 'type': 'mainstream'},
    'bbc.com': {'name': 'BBC', 'credibility': 9.0, 'type': 'mainstream'},
    'bbc.co.uk': {'name': 'BBC', 'credibility': 9.0, 'type': 'mainstream'},
    'theguardian.com': {'name': 'The Guardian', 'credibility': 8.5, 'type': 'mainstream'},
    'nytimes.com': {'name': 'New York Times', 'credibility': 9.0, 'type': 'mainstream'},
    'washingtonpost.com': {'name': 'Washington Post', 'credibility': 8.5, 'type': 'mainstream'},
    'cnn.com': {'name': 'CNN', 'credibility': 7.5, 'type': 'mainstream'},
    'aljazeera.com': {'name': 'Al Jazeera', 'credibility': 8.0, 'type': 'mainstream'},
    'snopes.com': {'name': 'Snopes', 'credibility': 9.0, 'type': 'fact_checker'},
    'politifact.com': {'name': 'PolitiFact', 'credibility': 9.0, 'type': 'fact_checker'},
    'factcheck.org': {'name': 'FactCheck.org', 'credibility': 9.0, 'type': 'fact_checker'},
    'fullfact.org': {'name': 'Full Fact', 'credibility': 9.0, 'type': 'fact_checker'},
}

# Request headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


class WebSearchScraper:
    """
    Searches the web for a claim/query, scrapes top results,
    extracts article content, and returns structured source data
    with publisher names for cross-referencing.
    """

    def __init__(self, timeout: int = 6, max_results: int = 5):
        self.timeout = timeout
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search_and_scrape(self, query: str, include_fact_check: bool = True) -> Dict:
        """
        Main method: Search the web for a query/claim, scrape top results,
        and return structured source data.

        Returns:
            {
                'query': str,
                'sources_scraped': [
                    {
                        'source_name': 'Hindustan Times',
                        'source_domain': 'hindustantimes.com',
                        'source_type': 'mainstream',
                        'credibility': 8.0,
                        'title': 'Article title',
                        'snippet': 'First 500 chars...',
                        'full_text': 'Full article text...',
                        'url': 'https://...',
                        'relevance_score': 0.85,
                    }, ...
                ],
                'total_sources': int,
                'fact_checker_sources': int,
                'mainstream_sources': int,
                'source_names': ['Hindustan Times', 'NDTV', ...],
                'consensus': 'agreement' | 'conflicting' | 'insufficient',
                'summary': str,
            }
        """
        logger.info(f"Web search & scrape for: {query[:80]}")

        # Step 1: Search DuckDuckGo for the query
        search_urls = self._search_duckduckgo(query)

        # Also search specifically for fact-checks
        if include_fact_check:
            fc_urls = self._search_duckduckgo(f"{query} fact check")
            # Merge, dedup
            seen = set(search_urls)
            for u in fc_urls:
                if u not in seen:
                    search_urls.append(u)
                    seen.add(u)

        # Limit to max_results
        search_urls = search_urls[:self.max_results]

        if not search_urls:
            logger.warning("No search results found")
            return self._empty_result(query)

        # Step 2: Scrape each URL in parallel
        scraped_sources = self._scrape_urls_parallel(search_urls)

        # Step 3: Score relevance to the original query
        scored_sources = self._score_relevance(query, scraped_sources)

        # Step 4: Build result
        source_names = list({s['source_name'] for s in scored_sources})
        fact_checker_count = sum(1 for s in scored_sources if s['source_type'] == 'fact_checker')
        mainstream_count = sum(1 for s in scored_sources if s['source_type'] == 'mainstream')

        # Step 5: Assess consensus
        consensus = self._assess_consensus(scored_sources, query)

        result = {
            'query': query,
            'sources_scraped': scored_sources,
            'total_sources': len(scored_sources),
            'fact_checker_sources': fact_checker_count,
            'mainstream_sources': mainstream_count,
            'source_names': source_names,
            'consensus': consensus,
            'summary': self._build_source_summary(scored_sources, source_names),
        }

        logger.info(f"Scraped {len(scored_sources)} sources: {', '.join(source_names[:5])}")
        return result

    def scrape_single_url(self, url: str) -> Optional[Dict]:
        """
        Scrape a single URL and extract article data.
        Useful for verifying a specific source.
        """
        return self._scrape_url(url)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   SEARCH ENGINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _search_duckduckgo(self, query: str) -> List[str]:
        """
        Search DuckDuckGo HTML and extract result URLs.
        DuckDuckGo doesn't require an API key.
        """
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=8)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            urls = []

            # DuckDuckGo HTML results are in <a class="result__a"> tags
            for link in soup.select('a.result__a'):
                href = link.get('href', '')
                # DuckDuckGo wraps URLs in a redirect; extract the actual URL
                actual_url = self._extract_ddg_url(href)
                if actual_url and self._is_valid_news_url(actual_url):
                    urls.append(actual_url)

            # Also try result__url class
            if not urls:
                for link in soup.select('a.result__url'):
                    href = link.get('href', '')
                    actual_url = self._extract_ddg_url(href)
                    if actual_url and self._is_valid_news_url(actual_url):
                        urls.append(actual_url)

            # Fallback: try all links with result class
            if not urls:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    actual_url = self._extract_ddg_url(href)
                    if actual_url and self._is_valid_news_url(actual_url):
                        urls.append(actual_url)

            # Deduplicate while preserving order
            seen = set()
            unique_urls = []
            for u in urls:
                domain = urlparse(u).netloc.lower()
                if domain not in seen:
                    seen.add(domain)
                    unique_urls.append(u)

            logger.info(f"DuckDuckGo search found {len(unique_urls)} URLs for: {query[:50]}")
            return unique_urls

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _extract_ddg_url(self, href: str) -> Optional[str]:
        """Extract actual URL from DuckDuckGo redirect link."""
        if not href:
            return None

        # DuckDuckGo uses //duckduckgo.com/l/?uddg=ENCODED_URL&rut=...
        if 'uddg=' in href:
            from urllib.parse import unquote, parse_qs, urlparse as up
            parsed = up(href)
            params = parse_qs(parsed.query)
            if 'uddg' in params:
                return unquote(params['uddg'][0])

        # Direct URL
        if href.startswith('http'):
            return href

        # Protocol-relative
        if href.startswith('//'):
            return 'https:' + href

        return None

    def _is_valid_news_url(self, url: str) -> bool:
        """Filter out non-news URLs (social media, search engines, etc.)."""
        if not url or not url.startswith('http'):
            return False

        domain = urlparse(url).netloc.lower()

        # Skip non-news domains
        skip_domains = {
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'x.com', 'instagram.com', 'reddit.com', 'wikipedia.org',
            'amazon.com', 'flipkart.com', 'duckduckgo.com', 'bing.com',
            'yahoo.com', 'linkedin.com', 'pinterest.com', 'tiktok.com',
        }

        for skip in skip_domains:
            if skip in domain:
                return False

        return True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   SCRAPING ENGINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _scrape_urls_parallel(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs in parallel using ThreadPool."""
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(self._scrape_url, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
        return results

    def _scrape_url(self, url: str) -> Optional[Dict]:
        """
        Scrape a single URL and extract:
        - Title
        - Main article text
        - Publisher/source name
        - Meta information
        """
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()

            # Only process HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return None

            soup = BeautifulSoup(response.text, 'lxml')

            # Extract title
            title = self._extract_title(soup)

            # Extract main article text
            article_text = self._extract_article_text(soup)

            # Skip if no meaningful content
            if not article_text or len(article_text) < 100:
                return None

            # Identify the source
            domain = urlparse(url).netloc.lower().replace('www.', '')
            source_info = self._identify_source(domain)

            return {
                'source_name': source_info['name'],
                'source_domain': domain,
                'source_type': source_info['type'],
                'credibility': source_info['credibility'],
                'title': title or 'Untitled',
                'snippet': article_text[:500],
                'full_text': article_text[:3000],  # Cap at 3000 chars
                'url': url,
                'relevance_score': 0.0,  # Will be scored later
            }

        except requests.Timeout:
            logger.warning(f"Timeout scraping: {url}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Request error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the article title using multiple strategies."""
        # Try OpenGraph title first (most reliable for news)
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        # Try <title> tag
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            # Often has "Article Title | Publisher" — take the first part
            raw = title_tag.string.strip()
            for sep in [' | ', ' - ', ' — ', ' :: ', ' : ']:
                if sep in raw:
                    return raw.split(sep)[0].strip()
            return raw

        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return ''

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """
        Extract the main article body text using multiple strategies:
        1. <article> tag
        2. Common article class names
        3. Largest text block heuristic
        """
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header',
                                   'aside', 'iframe', 'form', 'noscript',
                                   'svg', 'button', 'input']):
            tag.decompose()

        # Strategy 1: <article> tag
        article = soup.find('article')
        if article:
            text = self._clean_text(article.get_text(separator=' ', strip=True))
            if len(text) > 200:
                return text

        # Strategy 2: Common article container classes
        article_selectors = [
            '.article-body', '.article-content', '.article__body',
            '.story-body', '.story-content', '.story__content',
            '.post-content', '.post-body', '.entry-content',
            '.content-body', '.main-content', '.article-text',
            '[itemprop="articleBody"]', '.detail-content',
            '#article-body', '#story-body', '.td-post-content',
        ]

        for selector in article_selectors:
            container = soup.select_one(selector)
            if container:
                text = self._clean_text(container.get_text(separator=' ', strip=True))
                if len(text) > 200:
                    return text

        # Strategy 3: Largest paragraph cluster
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Group nearby paragraphs and find the densest cluster
            texts = [p.get_text(strip=True) for p in paragraphs
                     if len(p.get_text(strip=True)) > 40]
            if texts:
                combined = ' '.join(texts)
                return self._clean_text(combined)

        # Fallback: body text
        body = soup.find('body')
        if body:
            return self._clean_text(body.get_text(separator=' ', strip=True))[:2000]

        return ''

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common noise patterns
        text = re.sub(r'(Advertisement|ADVERTISEMENT|Sponsored|SPONSORED|Read More|SHARE|Share this)', '', text)
        # Remove URLs in text
        text = re.sub(r'https?://\S+', '', text)
        return text.strip()

    def _identify_source(self, domain: str) -> Dict:
        """
        Identify the source from its domain.
        Returns name, type, and credibility score.
        """
        # Check known sources
        for known_domain, info in KNOWN_SOURCES.items():
            if known_domain in domain:
                return info

        # Unknown source — extract name from domain
        # e.g., 'economictimes.com' → 'Economictimes'
        name_parts = domain.replace('.com', '').replace('.in', '').replace('.co.uk', '')
        name_parts = name_parts.replace('.org', '').replace('.net', '')
        name = name_parts.replace('.', ' ').replace('-', ' ').title()

        return {
            'name': name,
            'credibility': 5.0,  # Unknown source default
            'type': 'unknown',
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   RELEVANCE & CONSENSUS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _score_relevance(self, query: str, sources: List[Dict]) -> List[Dict]:
        """Score each scraped source's relevance to the original query."""
        query_words = set(query.lower().split())

        for source in sources:
            text = f"{source['title']} {source['snippet']}".lower()
            text_words = set(text.split())

            # Word overlap score
            overlap = len(query_words & text_words)
            word_score = min(1.0, overlap / max(len(query_words), 1))

            # Known source bonus
            credibility_bonus = source['credibility'] / 20.0  # 0-0.5

            # Fact-checker bonus
            fc_bonus = 0.15 if source['source_type'] == 'fact_checker' else 0.0

            relevance = min(1.0, word_score * 0.6 + credibility_bonus + fc_bonus)
            source['relevance_score'] = round(relevance, 3)

        # Sort by relevance
        sources.sort(key=lambda s: s['relevance_score'], reverse=True)
        return sources

    def _assess_consensus(self, sources: List[Dict], query: str) -> str:
        """
        Assess whether sources agree or disagree about the claim.
        Returns: 'agreement', 'conflicting', or 'insufficient'.
        """
        if len(sources) < 2:
            return 'insufficient'

        # Simple heuristic: look for negation/denial keywords in texts
        query_lower = query.lower()
        supports = 0
        denies = 0

        deny_patterns = [
            r'\bfalse\b', r'\bfake\b', r'\bhoax\b', r'\bmisleading\b',
            r'\bno evidence\b', r'\bno such\b', r'\bunverified\b',
            r'\bdebunked\b', r'\bnot true\b', r'\bdid not\b',
            r'\bmisinformation\b', r'\bfact.?check\b.*\bfalse\b',
        ]

        support_patterns = [
            r'\bconfirmed\b', r'\bverified\b', r'\btrue\b',
            r'\bofficial\b.*\bsaid\b', r'\baccording to\b',
            r'\breported\b', r'\bannounced\b',
        ]

        for source in sources:
            text = source.get('full_text', '').lower()
            title = source.get('title', '').lower()
            combined = f"{title} {text}"

            deny_count = sum(1 for p in deny_patterns if re.search(p, combined))
            support_count = sum(1 for p in support_patterns if re.search(p, combined))

            if deny_count > support_count:
                denies += 1
            elif support_count > 0:
                supports += 1

        total = supports + denies
        if total == 0:
            return 'insufficient'
        if denies > supports * 1.5:
            return 'mostly_denied'
        if supports > denies * 1.5:
            return 'mostly_supported'
        if denies > 0 and supports > 0:
            return 'conflicting'
        return 'insufficient'

    def _build_source_summary(self, sources: List[Dict], source_names: List[str]) -> str:
        """Build a human-readable summary of scraped sources."""
        if not sources:
            return "No web sources could be found for this claim."

        parts = []
        parts.append(f"Searched and scraped {len(sources)} web sources")

        if source_names:
            top_names = source_names[:5]
            parts.append(f"including {', '.join(top_names)}")

        fc_sources = [s for s in sources if s['source_type'] == 'fact_checker']
        if fc_sources:
            fc_names = [s['source_name'] for s in fc_sources]
            parts.append(f"Fact-checkers: {', '.join(fc_names)}")

        return '. '.join(parts) + '.'

    def _empty_result(self, query: str) -> Dict:
        """Return an empty result structure."""
        return {
            'query': query,
            'sources_scraped': [],
            'total_sources': 0,
            'fact_checker_sources': 0,
            'mainstream_sources': 0,
            'source_names': [],
            'consensus': 'insufficient',
            'summary': 'No web sources could be found for this claim.',
        }
