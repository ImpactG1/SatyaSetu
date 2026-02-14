"""
API Integration Services
Google Fact Check API & NewsAPI integration
"""

import os
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from newsapi import NewsApiClient

logger = logging.getLogger(__name__)


class GoogleFactCheckService:
    """Integration with Google Fact Check Tools API"""
    
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        if not self.api_key:
            logger.warning("Google Fact Check API key not configured")
    
    def search_claims(self, query: str, language_code: str = 'en', max_results: int = 10) -> List[Dict]:
        """
        Search for fact-checked claims related to a query
        
        Args:
            query: Search query text
            language_code: Language code (default: 'en')
            max_results: Maximum number of results
            
        Returns:
            List of fact-check results
        """
        if not self.api_key:
            logger.error("Google Fact Check API key not available")
            return []
        
        try:
            params = {
                'key': self.api_key,
                'query': query,
                'languageCode': language_code,
                'pageSize': max_results
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            claims = data.get('claims', [])
            
            # Parse and structure the results
            results = []
            for claim in claims:
                claim_review = claim.get('claimReview', [{}])[0]
                
                result = {
                    'claim_text': claim.get('text', ''),
                    'claimant': claim.get('claimant', 'Unknown'),
                    'claim_date': claim.get('claimDate', ''),
                    'rating': claim_review.get('textualRating', 'Unrated'),
                    'publisher': claim_review.get('publisher', {}).get('name', 'Unknown'),
                    'url': claim_review.get('url', ''),
                    'title': claim_review.get('title', ''),
                    'language_code': claim_review.get('languageCode', language_code)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} fact-check results for query: {query}")
            return results
            
        except requests.RequestException as e:
            logger.error(f"Error fetching from Google Fact Check API: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Google Fact Check service: {e}")
            return []
    
    def check_url(self, url: str) -> List[Dict]:
        """
        Check if a specific URL has been fact-checked
        
        Args:
            url: URL to check
            
        Returns:
            List of fact-check results for the URL
        """
        if not self.api_key:
            return []
        
        try:
            params = {
                'key': self.api_key,
                'reviewPublisherSiteFilter': url
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            claims = data.get('claims', [])
            
            results = []
            for claim in claims:
                claim_review = claim.get('claimReview', [{}])[0]
                result = {
                    'claim_text': claim.get('text', ''),
                    'rating': claim_review.get('textualRating', 'Unrated'),
                    'url': claim_review.get('url', ''),
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking URL with Fact Check API: {e}")
            return []


class NewsAPIService:
    """Integration with NewsAPI for news aggregation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            self.client = None
        else:
            self.client = NewsApiClient(api_key=self.api_key)
    
    def get_top_headlines(self, category: Optional[str] = None, 
                          country: str = 'us', page_size: int = 20) -> List[Dict]:
        """
        Get top headlines from NewsAPI
        
        Args:
            category: News category (business, entertainment, health, science, sports, technology)
            country: Country code (default: 'us')
            page_size: Number of results (max 100)
            
        Returns:
            List of news articles
        """
        if not self.client:
            logger.error("NewsAPI client not initialized")
            return []
        
        try:
            response = self.client.get_top_headlines(
                category=category,
                country=country,
                page_size=page_size
            )
            
            articles = response.get('articles', [])
            
            # Structure the results
            results = []
            for article in articles:
                result = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'image_url': article.get('urlToImage', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown')
                }
                results.append(result)
            
            logger.info(f"Fetched {len(results)} articles from NewsAPI")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def search_everything(self, query: str, from_date: Optional[datetime] = None,
                          to_date: Optional[datetime] = None, 
                          sort_by: str = 'publishedAt', page_size: int = 20) -> List[Dict]:
        """
        Search all news articles matching a query
        
        Args:
            query: Search query
            from_date: Start date for articles
            to_date: End date for articles
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of results (max 100)
            
        Returns:
            List of news articles
        """
        if not self.client:
            logger.error("NewsAPI client not initialized")
            return []
        
        try:
            # Default to last 7 days if no date range specified
            if not from_date:
                from_date = datetime.now() - timedelta(days=7)
            
            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d') if from_date else None,
                to=to_date.strftime('%Y-%m-%d') if to_date else None,
                sort_by=sort_by,
                page_size=page_size,
                language='en'
            )
            
            articles = response.get('articles', [])
            
            results = []
            for article in articles:
                result = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'image_url': article.get('urlToImage', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown')
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} articles for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching NewsAPI: {e}")
            return []
    
    def get_sources(self, category: Optional[str] = None, 
                    language: str = 'en', country: str = 'us') -> List[Dict]:
        """
        Get available news sources
        
        Args:
            category: News category filter
            language: Language code
            country: Country code
            
        Returns:
            List of news sources
        """
        if not self.client:
            return []
        
        try:
            response = self.client.get_sources(
                category=category,
                language=language,
                country=country
            )
            
            sources = response.get('sources', [])
            return [{
                'id': s.get('id'),
                'name': s.get('name'),
                'description': s.get('description'),
                'url': s.get('url'),
                'category': s.get('category')
            } for s in sources]
            
        except Exception as e:
            logger.error(f"Error fetching sources from NewsAPI: {e}")
            return []


class MultiSourceAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.fact_check_service = GoogleFactCheckService()
        self.news_service = NewsAPIService()
    
    def aggregate_content(self, query: str, include_news: bool = True, 
                          include_fact_checks: bool = True) -> Dict:
        """
        Aggregate content from multiple sources
        
        Args:
            query: Search query
            include_news: Include NewsAPI results
            include_fact_checks: Include fact-check results
            
        Returns:
            Dictionary with aggregated results
        """
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'news_articles': [],
            'fact_checks': [],
            'total_sources': 0
        }
        
        if include_news:
            articles = self.news_service.search_everything(query)
            results['news_articles'] = articles
            results['total_sources'] += len(articles)
        
        if include_fact_checks:
            fact_checks = self.fact_check_service.search_claims(query)
            results['fact_checks'] = fact_checks
            results['total_sources'] += len(fact_checks)
        
        logger.info(f"Aggregated {results['total_sources']} sources for query: {query}")
        return results
    
    def detect_anomalies(self, articles: List[Dict]) -> Dict:
        """
        Detect anomalies in aggregated content
        
        Args:
            articles: List of articles to analyze
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not articles:
            return {'anomalies_detected': False, 'details': []}
        
        # Simple anomaly detection based on:
        # 1. Same story from multiple sources with conflicting information
        # 2. Unusual spike in coverage of a topic
        # 3. Sources with low credibility reporting unique claims
        
        anomalies = {
            'anomalies_detected': False,
            'details': [],
            'conflicting_reports': 0,
            'unique_claims': 0,
            'coverage_spike': False
        }
        
        # Group articles by similarity (simplified - in production use NLP)
        source_count = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            source_count[source] = source_count.get(source, 0) + 1
        
        # Check for coverage spike
        if len(articles) > 50:  # Arbitrary threshold
            anomalies['coverage_spike'] = True
            anomalies['anomalies_detected'] = True
            anomalies['details'].append({
                'type': 'coverage_spike',
                'message': f'Unusual volume of content detected: {len(articles)} articles'
            })
        
        return anomalies
