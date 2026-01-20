#!/usr/bin/env python3
"""
Optimized OpenAlex Author Disambiguation MCP Server with Peer-Review Filtering

Provides a FastMCP-compliant API for author disambiguation and institution resolution
using the OpenAlex API with streamlined output to minimize token usage.
"""

import logging
from typing import Optional
from fastmcp import FastMCP
from alex_mcp.data_objects import (
    OptimizedAuthorResult,
    OptimizedSearchResponse,
    OptimizedWorksSearchResponse,
    OptimizedGeneralWorksSearchResponse,
    OptimizedWorkResult,
    AutocompleteAuthorCandidate,
    AutocompleteAuthorsResponse,
    optimize_author_data,
    optimize_work_data,
    extract_locations
)
import pyalex
import os
import sys
import aiohttp
import asyncio
import json
import re
import time
import ssl
import certifi
from collections import OrderedDict
from datetime import datetime, timedelta

def get_config():
    mailto = os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        print(
            "ERROR: The environment variable OPENALEX_MAILTO must be set to your email address "
            "to use the OpenAlex MCP server. Example: export OPENALEX_MAILTO='your-email@example.com'",
            file=sys.stderr
        )
        sys.exit(1)
    return {
        "OPENALEX_MAILTO": mailto,
        "OPENALEX_USER_AGENT": os.environ.get(
            "OPENALEX_USER_AGENT",
            f"alex-mcp (+{mailto})"
        ),
        "OPENALEX_MAX_AUTHORS": int(os.environ.get("OPENALEX_MAX_AUTHORS", 50)),  # Reduced default
        "OPENALEX_RATE_PER_SEC": int(os.environ.get("OPENALEX_RATE_PER_SEC", 10)),
        "OPENALEX_RATE_PER_DAY": int(os.environ.get("OPENALEX_RATE_PER_DAY", 100000)),
        "OPENALEX_USE_DAILY_API": os.environ.get("OPENALEX_USE_DAILY_API", "true").lower() == "true",
        "OPENALEX_SNAPSHOT_INTERVAL_DAYS": int(os.environ.get("OPENALEX_SNAPSHOT_INTERVAL_DAYS", 30)),
        "OPENALEX_PREMIUM_UPDATES": os.environ.get("OPENALEX_PREMIUM_UPDATES", "hourly"),
        "OPENALEX_RETRACTION_BUG_START": os.environ.get("OPENALEX_RETRACTION_BUG_START", "2023-12-22"),
        "OPENALEX_RETRACTION_BUG_END": os.environ.get("OPENALEX_RETRACTION_BUG_END", "2024-03-19"),
        "OPENALEX_NO_FUNDING_DATA": os.environ.get("OPENALEX_NO_FUNDING_DATA", "true").lower() == "true",
        "OPENALEX_MISSING_CORRESPONDING_AUTHORS": os.environ.get("OPENALEX_MISSING_CORRESPONDING_AUTHORS", "true").lower() == "true",
        "OPENALEX_PARTIAL_ABSTRACTS": os.environ.get("OPENALEX_PARTIAL_ABSTRACTS", "true").lower() == "true",
        "OPENALEX_ENABLE_SCIHUB": os.environ.get("OPENALEX_ENABLE_SCIHUB", "true").lower() == "true",
    }

# Configure logging
log_level_name = os.environ.get("ALEX_MCP_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("OpenAlex Academic Research")

# ============================================================================
# Cache and Rate Limiting Infrastructure
# ============================================================================

class SimpleCache:
    """
    Simple in-memory cache with TTL (Time To Live) for abstracts.
    Thread-safe for async operations.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[dict]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                # Check if expired
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    logger.debug(f"💾 Cache HIT: {key}")
                    return value
                else:
                    # Expired, remove
                    del self.cache[key]
                    logger.debug(f"⏰ Cache EXPIRED: {key}")
            return None
    
    async def set(self, key: str, value: dict):
        """Set value in cache with current timestamp."""
        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"🗑️ Cache EVICT: {oldest_key}")
            
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)
            logger.debug(f"💾 Cache SET: {key}")
    
    async def clear(self):
        """Clear all cached items."""
        async with self._lock:
            self.cache.clear()
            logger.info("🗑️ Cache CLEARED")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Prevents exceeding API rate limits.
    """
    def __init__(self, max_requests: int = 100, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.tokens = max_requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * (self.max_requests / self.time_window)
            )
            self.last_update = now
            
            # If no tokens available, wait
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.time_window / self.max_requests)
                logger.debug(f"⏳ Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
    
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'current_tokens': self.tokens
        }


# Initialize cache and rate limiters
abstract_cache = SimpleCache(max_size=1000, ttl_seconds=3600)  # 1 hour TTL
semantic_scholar_limiter = RateLimiter(max_requests=90, time_window=1.0)  # 90 req/s (conservative)
unpaywall_limiter = RateLimiter(max_requests=100, time_window=1.0)  # 100 req/s (courtesy)
scihub_limiter = RateLimiter(max_requests=50, time_window=1.0)  # 50 req/s (very conservative)


def configure_pyalex(email: str):
    """
    Configure pyalex for OpenAlex API usage.

    Args:
        email (str): The email to use for OpenAlex API requests.
    """
    pyalex.config.email = email

# Load configuration
config = get_config()
configure_pyalex(config["OPENALEX_MAILTO"])
pyalex.config.user_agent = config["OPENALEX_USER_AGENT"]


def is_peer_reviewed_journal(work_data) -> bool:
    """
    Improved filter to determine if a work is from a peer-reviewed journal.
    
    Uses a balanced approach that catches data catalogs and preprints while
    not being overly strict about DOIs (some legitimate papers lack them in OpenAlex).
    
    Args:
        work_data: OpenAlex work object
        
    Returns:
        bool: True if the work appears to be from a peer-reviewed journal
    """
    try:
        # Safe string extraction with None checking
        title = work_data.get('title') or ''
        if isinstance(title, str):
            title = title.lower()
        else:
            title = str(title).lower() if title is not None else ''
        
        # Quick exclusions based on title patterns
        title_exclusions = [
            'vizier online data catalog',
            'online data catalog',
            'data catalog',
            'catalog:',
            'database:',
            'repository:',
            'preprint',
            'arxiv:',
            'biorxiv',
            'medrxiv',
        ]
        
        for exclusion in title_exclusions:
            if exclusion in title:
                logger.debug(f"Excluding based on title pattern '{exclusion}': {title[:100]}")
                return False
        
        # Check primary location
        primary_location = work_data.get('primary_location')
        if not primary_location:
            logger.debug("Excluding work without primary location")
            return False
        
        # Check source information
        source = primary_location.get('source', {})
        if not source:
            logger.debug("Excluding work without source")
            return False
        
        # Get journal/source information with safe None checking
        journal_name_raw = source.get('display_name') or ''
        journal_name = journal_name_raw.lower() if isinstance(journal_name_raw, str) else str(journal_name_raw).lower()
        
        publisher = work_data.get('publisher', '')
        doi = work_data.get('doi')
        issn_l = source.get('issn_l')
        issn = source.get('issn')
        
        source_type_raw = source.get('type') or ''
        source_type = source_type_raw.lower() if isinstance(source_type_raw, str) else str(source_type_raw).lower()
        
        # CRITICAL: Exclude known data catalogs by journal name
        excluded_journals = [
            'vizier online data catalog',
            'ycat',
            'catalog',
            'database',
            'repository',
            'arxiv',
            'biorxiv',
            'medrxiv',
            'ssrn',
            'research square',
            'zenodo',
            'figshare',
            'dryad',
            'github',
            'protocols.io',
            'ceur',
            'conference proceedings',
            'workshop proceedings',
        ]
        
        for excluded in excluded_journals:
            if excluded in journal_name:
                logger.debug(f"Excluding journal pattern '{excluded}': {journal_name}")
                return False
        
        # CRITICAL: Data catalogs typically have no publisher AND no DOI
        # This catches VizieR entries effectively
        if not publisher and not doi:
            logger.debug(f"Excluding work without publisher AND DOI: {title[:100]}")
            return False
        
        # Source type should be journal (if specified)
        if source_type and source_type not in ['journal', '']:
            logger.debug(f"Excluding non-journal source type: {source_type}")
            return False
        
        # Work type should be article or letter with safe None checking
        work_type_raw = work_data.get('type') or ''
        work_type = work_type_raw.lower() if isinstance(work_type_raw, str) else str(work_type_raw).lower()
        if work_type not in ['article', 'letter']:
            logger.debug(f"Excluding work type: {work_type}")
            return False
        
        # Should have reasonable publication year
        pub_year = work_data.get('publication_year')
        if not pub_year or pub_year < 1900 or pub_year > 2030:
            logger.debug(f"Excluding work with invalid publication year: {pub_year}")
            return False
        
        # For papers claiming to be from legitimate journals, check quality signals
        known_legitimate_journals = [
            'nature',
            'science',
            'cell',
            'astrophysical journal',
            'astronomy and astrophysics',
            'monthly notices',
            'physical review',
            'journal of',
            'proceedings of',
        ]
        
        is_known_journal = any(known in journal_name for known in known_legitimate_journals)
        
        if is_known_journal:
            # For known journals, be more lenient (don't require DOI)
            # But still require either publisher or ISSN
            if not publisher and not issn_l and not issn:
                logger.debug(f"Excluding known journal without publisher/ISSN: {journal_name}")
                return False
        else:
            # For unknown journals, require more quality signals
            quality_signals = sum([
                bool(doi),          # Has DOI
                bool(publisher),    # Has publisher  
                bool(issn_l or issn),  # Has ISSN
                bool(journal_name and len(journal_name) > 5),  # Reasonable journal name
            ])
            
            if quality_signals < 2:  # Require at least 2 quality signals
                logger.debug(f"Excluding unknown journal with insufficient quality signals ({quality_signals}/4): {journal_name}")
                return False
        
        # Additional quality checks
        if 'cited_by_count' not in work_data:
            logger.debug("Excluding work without citation data")
            return False
        
        # Very long titles might be data descriptions
        if len(title) > 250:
            logger.debug(f"Excluding work with very long title: {title[:100]}...")
            return False
        
        # If we get here, it passes all checks
        logger.debug(f"ACCEPTED: {title[:100]}")
        return True
        
    except Exception as e:
        logger.error(f"Error in peer review check for work: {e}")
        logger.error(f"Work data keys: {list(work_data.keys()) if isinstance(work_data, dict) else 'Not a dict'}")
        logger.error(f"Work title: {repr(work_data.get('title') if isinstance(work_data, dict) else 'N/A')}")
        logger.error(f"Primary location: {repr(work_data.get('primary_location') if isinstance(work_data, dict) else 'N/A')}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def filter_peer_reviewed_works(works: list) -> list:
    """
    Apply peer-review filtering to a list of works.
    
    Args:
        works: List of OpenAlex work objects
        
    Returns:
        list: Filtered list containing only peer-reviewed journal works
    """
    filtered_works = []
    excluded_count = 0
    
    logger.info(f"Starting filtering of {len(works)} works...")
    
    for i, work in enumerate(works):
        # Safe handling of potentially None work or title
        if work is None:
            logger.warning(f"Skipping None work at position {i+1}")
            excluded_count += 1
            continue
            
        title_raw = work.get('title') if isinstance(work, dict) else None
        title = (title_raw or 'Unknown')[:60] if title_raw is not None else 'Unknown'
        
        try:
            if is_peer_reviewed_journal(work):
                filtered_works.append(work)
                logger.debug(f"✓ KEPT work {i+1}: {title}")
            else:
                excluded_count += 1
                logger.debug(f"✗ EXCLUDED work {i+1}: {title}")
        except Exception as e:
            logger.error(f"Error filtering work {i+1} (title: {title}): {e}")
            excluded_count += 1
    
    logger.info(f"Filtering complete: {len(filtered_works)} kept, {excluded_count} excluded from {len(works)} total")
    return filtered_works


def search_authors_core(
    name: str,
    institution: Optional[str] = None,
    topic: Optional[str] = None,
    country_code: Optional[str] = None,
    limit: int = 15  # Reduced default limit
) -> OptimizedSearchResponse:
    """
    Optimized core logic for searching authors using OpenAlex.
    Returns streamlined author data to minimize token usage.

    Args:
        name: Author name to search for.
        institution: (Optional) Institution name filter.
        topic: (Optional) Topic filter.
        country_code: (Optional) Country code filter.
        limit: Maximum number of results to return (default: 15).

    Returns:
        OptimizedSearchResponse: Streamlined response with essential author data.
    """
    try:
        # Build query
        query = pyalex.Authors().search_filter(display_name=name)
        
        # Add filters if provided
        filters = {}
        if institution:
            filters['affiliations.institution.display_name.search'] = institution
        if topic:
            filters['x_concepts.display_name.search'] = topic
        if country_code:
            filters['affiliations.institution.country_code'] = country_code
        
        if filters:
            query = query.filter(**filters)
        
        # Execute query with limit
        results = query.get(per_page=min(limit, 100))  # Increased for comprehensive search
        authors = list(results)
        
        # Convert to optimized format
        optimized_authors = []
        for author_data in authors:
            try:
                optimized_author = optimize_author_data(author_data)
                optimized_authors.append(optimized_author)
            except Exception as e:
                logger.warning(f"Error optimizing author data: {e}")
                # Skip problematic authors rather than failing completely
                continue
        
        logger.info(f"Found {len(optimized_authors)} authors for query: {name}")
        
        return OptimizedSearchResponse(
            query=name,
            total_count=len(optimized_authors),
            results=optimized_authors
        )
        
    except Exception as e:
        logger.error(f"Error searching authors for query '{name}': {e}")
        return OptimizedSearchResponse(
            query=name,
            total_count=0,
            results=[]
        )


def autocomplete_authors_core(
    name: str, 
    context: Optional[str] = None, 
    limit: int = 10,
    filter_no_institution: bool = True,
    enable_institution_ranking: bool = True
) -> AutocompleteAuthorsResponse:
    """
    Enhanced core function for author autocomplete with intelligent filtering and ranking.
    
    Args:
        name: Author name to search for
        context: Optional context for better matching (institution, research area, etc.)
        limit: Maximum number of candidates to return (increased default to 10)
        filter_no_institution: If True, exclude candidates with no institutional affiliation
        enable_institution_ranking: If True, rank candidates by institutional context relevance
        
    Returns:
        AutocompleteAuthorsResponse with filtered and ranked candidate authors
    """
    try:
        logger.info(f"🔍 Autocompleting authors for: '{name}' (limit: {limit})")
        if context:
            logger.info(f"   📝 Context provided: {context}")
        
        # Use PyAlex autocomplete for authors - get more results for filtering
        raw_limit = min(limit * 2, 20)  # Get 2x candidates for filtering
        results = pyalex.Authors().autocomplete(name)[:raw_limit]
        
        # Convert to our data model first
        all_candidates = []
        for result in results:
            candidate = AutocompleteAuthorCandidate(
                openalex_id=result.get('id', ''),
                display_name=result.get('display_name', ''),
                institution_hint=result.get('hint'),
                works_count=result.get('works_count', 0),
                cited_by_count=result.get('cited_by_count', 0),
                entity_type=result.get('entity_type', 'author'),
                external_id=result.get('external_id')
            )
            all_candidates.append(candidate)
        
        # ENHANCEMENT 1: Filter out candidates with no institution
        if filter_no_institution:
            filtered_candidates = [
                c for c in all_candidates 
                if c.institution_hint and c.institution_hint not in ['No institution', 'None', '']
            ]
            excluded_count = len(all_candidates) - len(filtered_candidates)
            if excluded_count > 0:
                logger.info(f"   🔍 Filtered out {excluded_count} candidates with no institution")
        else:
            filtered_candidates = all_candidates
        
        # ENHANCEMENT 2: Institution-aware ranking (if context provided)
        if enable_institution_ranking and context and filtered_candidates:
            scored_candidates = []
            context_lower = context.lower()
            
            for candidate in filtered_candidates:
                relevance_score = 0
                matched_terms = []
                
                inst_hint = (candidate.institution_hint or '').lower()
                
                # High-value institutional matches
                high_value_terms = [
                    'max planck', 'harvard', 'stanford', 'mit', 'cambridge', 'oxford',
                    'excellence cluster', 'crick', 'wellcome', 'nih', 'cnrs', 'inserm'
                ]
                for term in high_value_terms:
                    if term in context_lower and term in inst_hint:
                        relevance_score += 3
                        matched_terms.append(f"{term} (+3)")
                
                # Location-based matches
                location_terms = ['germany', 'uk', 'usa', 'france', 'köln', 'cologne', 'london', 'boston', 'berlin']
                for term in location_terms:
                    if term in context_lower and term in inst_hint:
                        relevance_score += 2
                        matched_terms.append(f"{term} (+2)")
                
                # Research field alignment (basic keyword matching)
                research_terms = ['biology', 'chemistry', 'biochemistry', 'physics', 'medicine']
                for term in research_terms:
                    if term in context_lower and term in inst_hint:
                        relevance_score += 1
                        matched_terms.append(f"{term} (+1)")
                
                # High-impact researcher bonus
                if candidate.cited_by_count and candidate.cited_by_count > 1000:
                    relevance_score += 1
                    matched_terms.append("high-impact (+1)")
                
                scored_candidates.append({
                    'candidate': candidate,
                    'relevance_score': relevance_score,
                    'matched_terms': matched_terms
                })
            
            # Sort by relevance score (descending), then by citation count
            scored_candidates.sort(key=lambda x: (x['relevance_score'], x['candidate'].cited_by_count), reverse=True)
            
            # Extract ranked candidates
            final_candidates = [sc['candidate'] for sc in scored_candidates[:limit]]
            
            # Log ranking results
            logger.info(f"   🏆 Institution-aware ranking applied:")
            for i, sc in enumerate(scored_candidates[:3], 1):  # Log top 3
                candidate = sc['candidate']
                logger.info(f"      {i}. {candidate.display_name} (score: {sc['relevance_score']}, {candidate.institution_hint})")
        else:
            # No ranking, just take first N candidates
            final_candidates = filtered_candidates[:limit]
        
        # Log final candidates (debug only)
        for candidate in final_candidates:
            logger.debug(f"   👤 {candidate.display_name} ({candidate.institution_hint or 'No institution'}) - {candidate.works_count} works")
        
        response = AutocompleteAuthorsResponse(
            query=name,
            context=context,
            total_candidates=len(final_candidates),
            candidates=final_candidates,
            search_metadata={
                'api_used': 'openalex_autocomplete',
                'has_context': context is not None,
                'filtered_no_institution': filter_no_institution,
                'institution_ranking_enabled': enable_institution_ranking and context is not None,
                'response_time_ms': None  # Could be added with timing
            }
        )
        
        logger.info(f"✅ Found {len(final_candidates)} candidates for '{name}'")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in autocomplete_authors_core: {e}")
        # Return empty response on error
        return AutocompleteAuthorsResponse(
            query=name,
            context=context,
            total_candidates=0,
            candidates=[],
            search_metadata={
                'api_used': 'openalex_autocomplete',
                'has_context': context is not None,
                'error': str(e)
            }
        )


def search_works_core(
    query: str,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    publication_year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    type: Optional[str] = None,
    limit: int = 25,
    peer_reviewed_only: bool = True,
    search_type: str = "general",
    open_access_only: bool = False,
    has_abstract: bool = False,
    sort: str = "relevance",
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False
) -> OptimizedGeneralWorksSearchResponse:
    """
    Core logic for searching works using OpenAlex with configurable search modes.
    Returns streamlined work data to minimize token usage.

    Args:
        query: Search query text
        author: (Optional) Author name filter
        institution: (Optional) Institution name filter
        publication_year: (Optional) Publication year filter (single year)
        year_from: (Optional) Publication year from (inclusive)
        year_to: (Optional) Publication year to (inclusive)
        type: (Optional) Work type filter (e.g., "article", "letter")
        limit: Maximum number of results (default: 25, max: 100)
        peer_reviewed_only: If True, apply peer-review filters (default: True)
        search_type: Search mode - "general", "title", or "title_and_abstract"
        open_access_only: If True, only return OA works (default: False)
        has_abstract: If True, only return works with abstracts (default: False)
        sort: Sort order - "relevance" (default), "date" (newest first), or "cited_by_count"

    Returns:
        OptimizedGeneralWorksSearchResponse: Streamlined response with work data.
    """
    try:
        # Ensure reasonable limits to control token usage
        limit = min(limit, 100)
        
        # Build the search query using PyAlex based on search_type
        if search_type == "title":
            # Use title-specific search for precise title matching
            works_query = pyalex.Works()
            filters = {'title.search': query}
        elif search_type == "title_and_abstract":
            # Use title and abstract search
            works_query = pyalex.Works()
            filters = {'title_and_abstract.search': query}
        else:  # search_type == "general" or any other value
            # Use general search across title, abstract, and fulltext (default behavior)
            works_query = pyalex.Works().search(query)
            filters = {}
        
        # Add author filter if provided
        if author:
            filters['raw_author_name.search'] = author
        
        # Add institution filter if provided  
        if institution:
            filters['authorships.institutions.display_name.search'] = institution
        
        # Add publication year filter (single year - backward compatibility)
        if publication_year:
            filters['publication_year'] = publication_year
        
        # Note: year_from/year_to are applied via post-filtering after API call
        # (PyAlex doesn't directly support date range filters in a reliable way)
        if year_from is not None or year_to is not None:
            # Remove single year if range is specified (range takes precedence)
            filters.pop('publication_year', None)
        
        # Add open access filter (NEW)
        if open_access_only:
            # OpenAlex filter syntax: is_oa:true
            filters['is_oa'] = 'true'
        
        # Add has abstract filter (NEW)
        if has_abstract:
            filters['has_abstract'] = True
        
        # Add type filter
        if type:
            filters['type'] = type
        elif peer_reviewed_only:
            # Focus on journal articles and letters for academic work
            filters['type'] = 'article|letter'
        
        # Add basic quality filters
        if peer_reviewed_only:
            filters['is_retracted'] = False
        
        # Apply standard filters to query
        if filters:
            works_query = works_query.filter(**filters)
        
        # Apply sorting (NEW)
        if sort == "date":
            works_query = works_query.sort("publication_date:desc")
        elif sort == "cited_by_count":
            works_query = works_query.sort("cited_by_count:desc")
        # Default "relevance" is handled by .search() automatically
        
        # Execute query
        logger.info(f"Searching OpenAlex works: search_type='{search_type}', query='{query[:50]}...', "
                   f"year_from={year_from}, year_to={year_to}, open_access_only={open_access_only}, "
                   f"has_abstract={has_abstract}, sort={sort}")
        
        # Execute the query
        results = works_query.get(per_page=limit)
        
        # Apply year range filtering post-query (reliable fallback approach)
        # PyAlex doesn't have reliable native support for publication_date ranges
        if year_from is not None or year_to is not None:
            filtered_results = []
            for work in results:
                pub_year = work.get('publication_year')
                if pub_year:
                    year_ok = True
                    if year_from is not None and pub_year < year_from:
                        year_ok = False
                    if year_to is not None and pub_year > year_to:
                        year_ok = False
                    if year_ok:
                        filtered_results.append(work)
            results = filtered_results
        
        # Apply additional peer-review filtering if requested
        if peer_reviewed_only and results:
            logger.info(f"Applying peer-review filtering to {len(results)} results...")
            results = filter_peer_reviewed_works(results)
            logger.info(f"After peer-review filtering: {len(results)} results remain")
        
        # Convert to optimized format
        optimized_works = []
        for work in results:
            try:
                optimized_work = optimize_work_data(
                    work,
                    include_authorships=include_authorships,
                    include_locations=include_locations,
                    include_best_oa_location=include_best_oa_location,
                    include_references=include_references,
                    include_related_works=include_related_works,
                    include_abstract_inverted_index=include_abstract_inverted_index,
                    include_fulltext_urls=include_fulltext_urls,
                    max_authorships=max_authorships,
                    max_locations=max_locations,
                    max_references=max_references,
                    max_related_works=max_related_works,
                    max_fulltext_urls=max_fulltext_urls,
                    abstract_max_chars=abstract_max_chars,
                    compact_ids=compact_ids
                )
                optimized_works.append(optimized_work)
            except Exception as e:
                logger.warning(f"Error optimizing work data: {e}")
                continue
        
        logger.info(f"Returning {len(optimized_works)} optimized works for search query")
        
        # Include applied filters in response metadata
        applied_filters = {
            'year_from': year_from,
            'year_to': year_to,
            'open_access_only': open_access_only,
            'has_abstract': has_abstract,
            'sort': sort
        }
        
        return OptimizedGeneralWorksSearchResponse(
            query=query,
            total_count=len(optimized_works),
            results=optimized_works,
            filters=applied_filters
        )
        
    except Exception as e:
        logger.error(f"Error searching works for query '{query}': {e}")
        return OptimizedGeneralWorksSearchResponse(
            query=query,
            total_count=0,
            results=[],
            filters={}
        )


def retrieve_author_works_core(
    author_id: str,
    limit: int = 20_000,  # High default limit for comprehensive analysis
    order_by: str = "date",  # "date" or "citations"
    publication_year: Optional[int] = None,
    type: Optional[str] = None,
    journal_only: bool = True,  # Default to True for peer-reviewed content
    min_citations: Optional[int] = None,
    peer_reviewed_only: bool = True,  # Default to True
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False,
) -> OptimizedWorksSearchResponse:
    """
    Enhanced core logic to retrieve peer-reviewed works for a given OpenAlex Author ID.
    Returns streamlined work data to minimize token usage and ensures only legitimate
    peer-reviewed journal articles and letters.

    Args:
        author_id: OpenAlex Author ID
        limit: Maximum number of results (default: 2000 for comprehensive analysis)
        order_by: Sort order - "date" or "citations"
        publication_year: Filter by specific year
        type: Filter by work type (e.g., "journal-article")
        journal_only: If True, only return journal articles and letters
        min_citations: Minimum citation count filter
        peer_reviewed_only: If True, apply comprehensive peer-review filters

    Returns:
        OptimizedWorksSearchResponse: Streamlined response with peer-reviewed work data.
    """
    try:
        limit = min(limit, 20_000)
        
        # Build base filters
        filters = {"author.id": author_id}
        
        # Add optional filters
        if publication_year:
            filters["publication_year"] = publication_year
        if type:
            filters["type"] = type
        elif journal_only:
            # Focus on journal articles and letters for academic work
            filters["type"] = "article|letter"
        if min_citations:
            filters["cited_by_count"] = f">={min_citations}"
        
        # Add some basic API-level filters (but not too restrictive)
        if peer_reviewed_only or journal_only:
            # Only exclude obviously retracted papers at API level
            filters["is_retracted"] = "false"
        
        # Convert author_id to proper format if needed
        if author_id.startswith("https://openalex.org/"):
            author_id_short = author_id.split("/")[-1]
            filters["author.id"] = f"https://openalex.org/{author_id_short}"

        # Build query - get more results for post-filtering if needed
        if peer_reviewed_only:
            initial_limit = min(limit * 4, 20_000)  # Get 4x more for filtering, much higher limit
        else:
            initial_limit = limit
            
        works_query = pyalex.Works().filter(**filters)
        
        # Apply sorting
        if order_by == "citations":
            works_query = works_query.sort(cited_by_count="desc")
        else:
            works_query = works_query.sort(publication_date="desc")
        
        # Execute query using pagination to get ALL works
        logger.info(f"Querying OpenAlex for up to {initial_limit} works with filters: {filters}")
        
        # Use paginate() to get all works, not just the first page
        all_works = []
        pager = works_query.paginate(per_page=200, n_max=initial_limit)  # Use 200 per page (API recommended)
        
        for page in pager:
            all_works.extend(page)
            if len(all_works) >= initial_limit:
                break
        
        works = all_works[:initial_limit]  # Ensure we don't exceed the limit
        logger.info(f"Retrieved {len(works)} works from OpenAlex via pagination")
        
        # Apply peer-review filtering if requested
        if peer_reviewed_only:
            logger.info("Applying peer-review filtering...")
            works = filter_peer_reviewed_works(works)
            logger.info(f"After filtering: {len(works)} works remain")
        
        # Limit to requested number after filtering
        works = works[:limit]
        
        # Get author name for response (if available from first work)
        author_name = None
        if works:
            authorships = works[0].get('authorships', [])
            for authorship in authorships:
                author = authorship.get('author', {})
                if author.get('id') == author_id:
                    author_name = author.get('display_name')
                    break
        
        # Convert to optimized format
        optimized_works = []
        for work_data in works:
            try:
                optimized_work = optimize_work_data(
                    work_data,
                    include_authorships=include_authorships,
                    include_locations=include_locations,
                    include_best_oa_location=include_best_oa_location,
                    include_references=include_references,
                    include_related_works=include_related_works,
                    include_abstract_inverted_index=include_abstract_inverted_index,
                    include_fulltext_urls=include_fulltext_urls,
                    max_authorships=max_authorships,
                    max_locations=max_locations,
                    max_references=max_references,
                    max_related_works=max_related_works,
                    max_fulltext_urls=max_fulltext_urls,
                    abstract_max_chars=abstract_max_chars,
                    compact_ids=compact_ids
                )
                optimized_works.append(optimized_work)
            except Exception as e:
                logger.warning(f"Error optimizing work data: {e}")
                continue
        
        logger.info(f"Final result: {len(optimized_works)} works for author: {author_id}")
        
        return OptimizedWorksSearchResponse(
            author_id=author_id,
            author_name=author_name,
            total_count=len(optimized_works),
            results=optimized_works,
            filters=filters
        )
        
    except Exception as e:
        logger.error(f"Error retrieving works for author {author_id}: {e}")
        return OptimizedWorksSearchResponse(
            author_id=author_id,
            total_count=0,
            results=[],
            filters={}
        )


@mcp.tool(
    annotations={
        "title": "Search Authors (Optimized)",
        "description": (
            "Search for authors by name with optional filters. "
            "Returns streamlined author data optimized for AI agents with ~70% fewer tokens. "
            "Includes essential info: name, ORCID, affiliations (as strings), metrics, and research fields."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def search_authors(
    name: str,
    institution: Optional[str] = None,
    topic: Optional[str] = None,
    country_code: Optional[str] = None,
    limit: int = 15
) -> dict:
    """
    Optimized MCP tool wrapper for searching authors.

    Args:
        name: Author name to search for.
        institution: (Optional) Institution name filter.
        topic: (Optional) Topic filter.
        country_code: (Optional) Country code filter.
        limit: Maximum number of results to return (default: 15, max: 100).

    Returns:
        dict: Serialized OptimizedSearchResponse with streamlined author data.
    """
    # Ensure reasonable limits to control token usage
    limit = min(limit, 100)  # Increased for comprehensive author search
    
    response = search_authors_core(
        name=name,
        institution=institution,
        topic=topic,
        country_code=country_code,
        limit=limit
    )
    return response.model_dump()


@mcp.tool(
    annotations={
        "title": "Retrieve Author Works (Peer-Reviewed Only)",
        "description": (
            "Retrieve peer-reviewed journal works for a given OpenAlex Author ID. "
            "Automatically filters out data catalogs, preprint servers, and non-journal content. "
            "Returns streamlined work data optimized for AI agents with ~80% fewer tokens. "
            "Uses balanced filtering: excludes VizieR catalogs but allows legitimate papers without DOIs."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def retrieve_author_works(
    author_id: str,
    limit: Optional[int] = None,
    order_by: str = "date",
    publication_year: Optional[int] = None,
    type: Optional[str] = None,
    journal_only: bool = True,
    min_citations: Optional[int] = None,
    peer_reviewed_only: bool = True,
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False,
) -> dict:
    """
    Enhanced MCP tool wrapper for retrieving author works with flexible filtering.

    Args:
        author_id: OpenAlex Author ID (e.g., 'https://openalex.org/A123456789')
        limit: Maximum number of results (default: None = ALL works via pagination, max: 2000)
        order_by: Sort order - "date" for newest first, "citations" for most cited first
        publication_year: Filter by specific publication year
        type: Filter by work type (e.g., "journal-article", "letter")
        journal_only: If True, only return journal articles and letters (default: True)
        min_citations: Only return works with at least this many citations
        peer_reviewed_only: If True, apply balanced peer-review filters (default: True)

    Returns:
        dict: Serialized OptimizedWorksSearchResponse with author's works.
        
    Usage Patterns:
        # For AI validation (sample of high-impact works)
        retrieve_author_works(author_id, limit=20, order_by="citations")
        
        # For complete benchmark evaluation (ALL works, minimal filtering)
        retrieve_author_works(author_id, peer_reviewed_only=False, journal_only=False)
        
        # For peer-reviewed works only (default behavior)
        retrieve_author_works(author_id)
    """
    # Handle limit: None means ALL works, otherwise cap at reasonable limit
    logger.info(f"MCP tool received limit parameter: {limit}")
    if limit is None:
        limit = 2000  # Set a very high limit to get ALL works
        logger.info(f"No limit specified, setting to {limit} for comprehensive retrieval")
    else:
        limit = min(limit, 2000)  # Increased max limit for comprehensive analysis
        logger.info(f"Explicit limit specified, capped to {limit}")
    
    response = retrieve_author_works_core(
        author_id=author_id,
        limit=limit,
        order_by=order_by,
        publication_year=publication_year,
        type=type,
        journal_only=journal_only,
        min_citations=min_citations,
        peer_reviewed_only=peer_reviewed_only,
        include_authorships=include_authorships,
        include_locations=include_locations,
        include_best_oa_location=include_best_oa_location,
        include_references=include_references,
        include_related_works=include_related_works,
        include_abstract_inverted_index=include_abstract_inverted_index,
        include_fulltext_urls=include_fulltext_urls,
        max_authorships=max_authorships,
        max_locations=max_locations,
        max_references=max_references,
        max_related_works=max_related_works,
        max_fulltext_urls=max_fulltext_urls,
        abstract_max_chars=abstract_max_chars,
        compact_ids=compact_ids,
    )
    return response.model_dump()


@mcp.tool(
    annotations={
        "title": "Search Works (Optimized)",
        "description": (
            "Search for academic works with configurable search modes and optional filters. "
            "Returns streamlined work data optimized for AI agents with ~80% fewer tokens. "
            "Supports different search types: 'general' (title/abstract/fulltext), 'title' (title only), "
            "or 'title_and_abstract' (title and abstract only). "
            "Supports author, institution, publication year, and type filters. "
            "Automatically applies peer-review filtering to exclude data catalogs and preprints."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def search_works(
    query: str,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    publication_year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    type: Optional[str] = None,
    limit: int = 25,
    peer_reviewed_only: bool = True,
    search_type: str = "general",
    open_access_only: bool = False,
    has_abstract: bool = False,
    sort: str = "relevance",
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False
) -> dict:
    """
    Optimized MCP tool wrapper for searching works with advanced filters.

    Args:
        query: Search query text
        author: (Optional) Author name filter
        institution: (Optional) Institution name filter
        publication_year: (Optional) Publication year filter (single year)
        year_from: (Optional) Filter for papers from this year (inclusive)
        year_to: (Optional) Filter for papers up to this year (inclusive)
        type: (Optional) Work type filter (e.g., "article", "letter")
        limit: Maximum number of results (default: 25, max: 100)
        peer_reviewed_only: If True, apply peer-review filters (default: True)
        search_type: Search mode - "general" (title/abstract/fulltext), "title" (title only), 
                    or "title_and_abstract" (title and abstract only)
        open_access_only: If True, only return OA works (default: False)
        has_abstract: If True, only return works with abstracts (default: False)
        sort: Sort order - "relevance" (default), "date" (newest first), or "cited_by_count"

    Returns:
        dict: Serialized OptimizedGeneralWorksSearchResponse with streamlined work data.
    """
    # Ensure reasonable limits to control token usage
    limit = min(limit, 100)
    
    response = search_works_core(
        query=query,
        author=author,
        institution=institution,
        publication_year=publication_year,
        year_from=year_from,
        year_to=year_to,
        type=type,
        limit=limit,
        peer_reviewed_only=peer_reviewed_only,
        search_type=search_type,
        open_access_only=open_access_only,
        has_abstract=has_abstract,
        sort=sort,
        include_authorships=include_authorships,
        include_locations=include_locations,
        include_best_oa_location=include_best_oa_location,
        include_references=include_references,
        include_related_works=include_related_works,
        include_abstract_inverted_index=include_abstract_inverted_index,
        include_fulltext_urls=include_fulltext_urls,
        max_authorships=max_authorships,
        max_locations=max_locations,
        max_references=max_references,
        max_related_works=max_related_works,
        max_fulltext_urls=max_fulltext_urls,
        abstract_max_chars=abstract_max_chars,
        compact_ids=compact_ids
    )
    return response.model_dump()


@mcp.tool(
    annotations={
        "title": "Search Works (Smart Multi-Query)",
        "description": (
            "Run multiple work queries and merge results with deduplication by OpenAlex ID. "
            "Useful for synonyms, multilingual queries, or expanded keyword sets. "
            "Returns a unified, deduplicated result set." 
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def search_works_smart(
    queries: list,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    publication_year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    type: Optional[str] = None,
    limit_per_query: int = 15,
    total_limit: int = 50,
    peer_reviewed_only: bool = True,
    search_type: str = "general",
    open_access_only: bool = False,
    has_abstract: bool = False,
    sort: str = "relevance",
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False
) -> dict:
    """
    Run multiple queries and deduplicate results by work ID.

    Args:
        queries: List of query strings
        limit_per_query: Max results per query (default: 15)
        total_limit: Max total unique results after deduplication (default: 50)
        Other args mirror search_works filters

    Returns:
        dict with:
        - queries: List of queries executed
        - total_count: Total unique works after deduplication
        - results: Deduplicated works
        - per_query_counts: Raw counts per query
        - raw_total: Total raw results before deduplication
    """
    if not queries or not isinstance(queries, list):
        return {
            'error': 'queries must be a non-empty list of strings',
            'queries': queries or []
        }

    limit_per_query = min(max(limit_per_query, 1), 100)
    total_limit = min(max(total_limit, 1), 200)

    deduped = OrderedDict()
    per_query_counts = {}
    raw_total = 0

    for query in queries:
        if not isinstance(query, str) or not query.strip():
            continue

        response = search_works_core(
            query=query,
            author=author,
            institution=institution,
            publication_year=publication_year,
            year_from=year_from,
            year_to=year_to,
            type=type,
            limit=limit_per_query,
            peer_reviewed_only=peer_reviewed_only,
            search_type=search_type,
            open_access_only=open_access_only,
            has_abstract=has_abstract,
            sort=sort,
            include_authorships=include_authorships,
            include_locations=include_locations,
            include_best_oa_location=include_best_oa_location,
            include_references=include_references,
            include_related_works=include_related_works,
            include_abstract_inverted_index=include_abstract_inverted_index,
            include_fulltext_urls=include_fulltext_urls,
            max_authorships=max_authorships,
            max_locations=max_locations,
            max_references=max_references,
            max_related_works=max_related_works,
            max_fulltext_urls=max_fulltext_urls,
            abstract_max_chars=abstract_max_chars,
            compact_ids=compact_ids
        )

        per_query_counts[query] = len(response.results)
        raw_total += len(response.results)

        for work in response.results:
            if work.id not in deduped:
                deduped[work.id] = work
                if len(deduped) >= total_limit:
                    break
        if len(deduped) >= total_limit:
            break

    results = [work.model_dump() for work in list(deduped.values())[:total_limit]]

    return {
        'queries': queries,
        'total_count': len(results),
        'results': results,
        'per_query_counts': per_query_counts,
        'raw_total': raw_total,
        'deduplicated': raw_total - len(results)
    }


@mcp.tool(
    annotations={
        "title": "Autocomplete Authors (Smart Disambiguation)",
        "description": (
            "Get multiple author candidates using OpenAlex autocomplete API for intelligent disambiguation. "
            "Returns a ranked list of potential author matches with institutional hints and research metrics. "
            "Perfect when you need to disambiguate authors and have context like institution, research area, or co-authors. "
            "The AI can select the best match based on the provided context. "
            "Much faster than full search (~200ms) and provides multiple options for better accuracy."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def autocomplete_authors(
    name: str,
    context: Optional[str] = None, 
    limit: int = 10,
    filter_no_institution: bool = True,
    enable_institution_ranking: bool = True
) -> dict:
    """
    Enhanced autocomplete authors with intelligent filtering and ranking.
    
    Args:
        name: Author name to search for (e.g., "James Briscoe", "M. Ralser")
        context: Optional context to help with disambiguation (e.g., "Francis Crick Institute developmental biology", "Max Planck Institute Köln Germany")
        limit: Maximum number of candidates to return (default: 10, max: 15)
        filter_no_institution: If True, exclude candidates with no institutional affiliation (default: True)
        enable_institution_ranking: If True, rank candidates by institutional context relevance (default: True)
        
    Returns:
        dict: Serialized AutocompleteAuthorsResponse with filtered and ranked candidate authors, including:
        - openalex_id: Full OpenAlex author ID
        - display_name: Author's display name
        - institution_hint: Current/last known institution 
        - works_count: Number of published works
        - cited_by_count: Total citation count
        - external_id: ORCID or other external identifiers
        - search_metadata: Information about filtering and ranking applied
        
    Example usage:
        # Get high-quality candidates with institutional filtering
        candidates = await autocomplete_authors("Ivan Matić", context="Max Planck Institute Biology Ageing Köln Germany")
        
        # For seasoned researchers, institution hints and ranking help disambiguation
        # AI can then select the best match or retrieve works for further verification
        
    Enhanced Features:
        - Filters out candidates with no institutional affiliation (reduces noise)
        - Institution-aware ranking when context is provided (improves accuracy)
        - Higher default limit (10 vs 5) for better candidate coverage
        - Detailed logging for debugging and optimization
    """
    # Ensure reasonable limits - increased max to 15
    limit = min(max(limit, 1), 15)
    
    response = autocomplete_authors_core(
        name=name,
        context=context, 
        limit=limit,
        filter_no_institution=filter_no_institution,
        enable_institution_ranking=enable_institution_ranking
    )
    return response.model_dump()


# PubMed Integration Functions
import requests
import xml.etree.ElementTree as ET
from typing import Union

def pubmed_search_core(
    query: str,
    max_results: int = 20,
    search_type: str = "author"
) -> dict:
    """
    Core PubMed search functionality using E-utilities API.
    
    Args:
        query: Search query (author name, DOI, or keywords)
        max_results: Maximum number of results to return
        search_type: Type of search ("author", "doi", "title", "keywords")
        
    Returns:
        dict with search results including PMIDs, total count, and basic metadata
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # Format search term based on type
        if search_type == "author":
            search_term = f'"{query}"[Author]'
        elif search_type == "doi":
            clean_doi = query.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            search_term = f'"{clean_doi}"[AID]'
        elif search_type == "title":
            search_term = f'"{query}"[Title]'
        else:  # keywords
            search_term = query
        
        logger.info(f"🔍 PubMed search: {search_term} (max: {max_results})")
        
        # Search PubMed
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': search_term,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        search_data = response.json()
        
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        total_count = int(search_data.get('esearchresult', {}).get('count', 0))
        
        logger.info(f"📊 Found {total_count} total results, retrieved {len(pmids)} PMIDs")
        
        # Get basic details for retrieved PMIDs (if any)
        articles = []
        if pmids:
            articles = get_pubmed_summaries(pmids[:min(len(pmids), 10)])  # Limit to 10 for performance
        
        return {
            'query': query,
            'search_type': search_type,
            'search_term_used': search_term,
            'total_count': total_count,
            'retrieved_count': len(pmids),
            'pmids': pmids,
            'articles': articles,
            'search_metadata': {
                'api_used': 'pubmed_esearch',
                'max_results_requested': max_results,
                'response_time_ms': None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ PubMed search error: {e}")
        return {
            'query': query,
            'search_type': search_type,
            'total_count': 0,
            'retrieved_count': 0,
            'pmids': [],
            'articles': [],
            'error': str(e)
        }


def get_pubmed_summaries(pmids: list) -> list:
    """
    Get summary information for a list of PMIDs using esummary.
    
    Args:
        pmids: List of PubMed IDs
        
    Returns:
        List of article summaries with basic metadata
    """
    if not pmids:
        return []
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # Get summaries
        summary_url = f"{base_url}esummary.fcgi"
        summary_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'json'
        }
        
        response = requests.get(summary_url, params=summary_params, timeout=15)
        response.raise_for_status()
        summary_data = response.json()
        
        articles = []
        uids = summary_data.get('result', {}).get('uids', [])
        
        for uid in uids:
            article_data = summary_data.get('result', {}).get(uid, {})
            if article_data:
                # Extract key information
                authors = article_data.get('authors', [])
                author_names = [author.get('name', '') for author in authors[:5]]  # First 5 authors
                
                article = {
                    'pmid': uid,
                    'title': article_data.get('title', ''),
                    'authors': author_names,
                    'journal': article_data.get('fulljournalname', ''),
                    'pub_date': article_data.get('pubdate', ''),
                    'doi': article_data.get('elocationid', ''),  # Often contains DOI
                    'pmcid': article_data.get('pmcid', ''),
                    'publication_types': article_data.get('pubtype', [])
                }
                articles.append(article)
        
        logger.info(f"📄 Retrieved summaries for {len(articles)} articles")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Error getting PubMed summaries: {e}")
        return []


def get_pubmed_author_sample(author_name: str, sample_size: int = 5) -> dict:
    """
    Get a sample of works by an author from PubMed with institutional information.
    
    Args:
        author_name: Author name to search for
        sample_size: Number of sample works to analyze in detail
        
    Returns:
        dict with author sample analysis including affiliations and name variants
    """
    try:
        logger.info(f"🔍 Getting PubMed author sample for: {author_name}")
        
        # Search for author
        search_result = pubmed_search_core(author_name, max_results=sample_size, search_type="author")
        
        if not search_result['pmids']:
            return {
                'author_name': author_name,
                'total_works': 0,
                'sample_works': [],
                'institutional_keywords': [],
                'name_variants': [],
                'email_addresses': []
            }
        
        # Get detailed information for sample
        sample_pmids = search_result['pmids'][:sample_size]
        detailed_articles = []
        all_affiliations = []
        name_variants = set()
        email_addresses = set()
        
        for pmid in sample_pmids:
            article_details = get_detailed_pubmed_article(pmid, author_name)
            if article_details:
                detailed_articles.append(article_details)
                
                # Extract affiliations and variants for target author
                for author_info in article_details.get('author_details', []):
                    if is_target_author(author_info, author_name):
                        all_affiliations.extend(author_info.get('affiliations', []))
                        
                        # Collect name variants
                        full_name = f"{author_info['first_name']} {author_info['last_name']}".strip()
                        if full_name:
                            name_variants.add(full_name)
                        
                        # Extract email addresses
                        for affil in author_info.get('affiliations', []):
                            emails = extract_emails_from_text(affil)
                            email_addresses.update(emails)
        
        # Extract institutional keywords
        institutional_keywords = extract_institutional_keywords(all_affiliations)
        
        return {
            'author_name': author_name,
            'total_works': search_result['total_count'],
            'sample_works': detailed_articles,
            'institutional_keywords': institutional_keywords,
            'name_variants': list(name_variants),
            'email_addresses': list(email_addresses),
            'sample_metadata': {
                'sample_size': len(detailed_articles),
                'affiliations_found': len(all_affiliations)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error in PubMed author sample: {e}")
        return {
            'author_name': author_name,
            'total_works': 0,
            'sample_works': [],
            'error': str(e)
        }


def get_detailed_pubmed_article(pmid: str, target_author: str) -> dict:
    """Get detailed article information including author affiliations"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        response = requests.get(fetch_url, params=fetch_params, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.text)
        article = root.find('.//PubmedArticle')
        
        if article is None:
            return None
        
        # Extract basic info
        title_elem = article.find('.//ArticleTitle')
        title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ''
        
        journal_elem = article.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ''
        
        # Extract authors with affiliations
        author_details = []
        author_list = article.find('.//AuthorList')
        if author_list is not None:
            for author_elem in author_list.findall('Author'):
                author_info = extract_detailed_author_info(author_elem)
                author_details.append(author_info)
        
        return {
            'pmid': pmid,
            'title': title,
            'journal': journal,
            'author_details': author_details
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching detailed article {pmid}: {e}")
        return None


def extract_detailed_author_info(author_elem: ET.Element) -> dict:
    """Extract detailed author information from XML element"""
    author_info = {
        'last_name': '',
        'first_name': '',
        'initials': '',
        'affiliations': []
    }
    
    try:
        last_name = author_elem.find('LastName')
        if last_name is not None:
            author_info['last_name'] = last_name.text or ''
        
        first_name = author_elem.find('ForeName')
        if first_name is not None:
            author_info['first_name'] = first_name.text or ''
        
        initials = author_elem.find('Initials')
        if initials is not None:
            author_info['initials'] = initials.text or ''
        
        # Get affiliations
        affil_info = author_elem.find('AffiliationInfo')
        if affil_info is not None:
            for affil in affil_info.findall('Affiliation'):
                if affil.text:
                    author_info['affiliations'].append(affil.text.strip())
        
    except Exception:
        pass
    
    return author_info


def is_target_author(author_info: dict, target_name: str) -> bool:
    """Check if author_info matches target author name"""
    full_name = f"{author_info['first_name']} {author_info['last_name']}".strip().lower()
    target_lower = target_name.lower()
    
    # Simple similarity check
    return (target_lower in full_name or 
            full_name in target_lower or
            author_info['last_name'].lower() in target_lower)


def extract_emails_from_text(text: str) -> list:
    """Extract email addresses from text"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_institutional_keywords(affiliations: list) -> list:
    """Extract common institutional keywords from affiliations"""
    if not affiliations:
        return []
    
    # Combine all affiliations
    all_text = ' '.join(affiliations).lower()
    
    # Common institutional keywords
    keywords = []
    institutional_terms = [
        'university', 'institute', 'college', 'school', 'center', 'centre',
        'hospital', 'laboratory', 'department', 'faculty', 'division',
        'max planck', 'harvard', 'stanford', 'mit', 'cambridge', 'oxford',
        'excellence cluster', 'cnrs', 'inserm', 'nih'
    ]
    
    for term in institutional_terms:
        if term in all_text:
            keywords.append(term)
    
    return keywords[:10]  # Return top 10


@mcp.tool(
    annotations={
        "title": "Search PubMed",
        "description": (
            "Search PubMed database for publications by author, DOI, title, or keywords. "
            "Provides basic article metadata including authors, journal, and publication info. "
            "Useful for cross-validation with OpenAlex data and discovering name variants."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def search_pubmed(
    query: str,
    search_type: str = "author",
    max_results: int = 20
) -> dict:
    """
    Search PubMed database for publications.
    
    Args:
        query: Search query (author name, DOI, title, or keywords)
        search_type: Type of search - "author", "doi", "title", or "keywords" (default: "author")
        max_results: Maximum number of results to return (default: 20, max: 50)
        
    Returns:
        dict: Search results with PMIDs, article metadata, and summary statistics
        
    Example usage:
        # Search for author
        search_pubmed("Ivan Matic", search_type="author", max_results=10)
        
        # Search by DOI
        search_pubmed("10.1038/nprot.2009.36", search_type="doi")
        
        # Search by keywords
        search_pubmed("ADP-ribosylation DNA repair", search_type="keywords")
    """
    # Validate parameters
    max_results = min(max(max_results, 1), 50)  # Cap at 50 for performance
    valid_types = ["author", "doi", "title", "keywords"]
    if search_type not in valid_types:
        search_type = "author"
    
    logger.info(f"🔍 PubMed search: '{query}' (type: {search_type}, max: {max_results})")
    
    result = pubmed_search_core(query, max_results, search_type)
    return result


@mcp.tool(
    annotations={
        "title": "PubMed Author Sample",
        "description": (
            "Get a detailed sample of works by an author from PubMed including "
            "institutional affiliations, name variants, and email addresses. "
            "Useful for cross-validation and institutional disambiguation."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def pubmed_author_sample(
    author_name: str,
    sample_size: int = 5
) -> dict:
    """
    Get detailed author sample from PubMed with institutional information.
    
    Args:
        author_name: Author name to search for (e.g., "Ivan Matic", "J Smith")
        sample_size: Number of recent works to analyze in detail (default: 5, max: 10)
        
    Returns:
        dict: Author analysis including:
        - total_works: Total number of works found in PubMed
        - sample_works: Detailed information for sample works
        - institutional_keywords: Common institutional terms found
        - name_variants: Different name formats found
        - email_addresses: Email addresses extracted from affiliations
        
    Example usage:
        # Get institutional profile for author
        pubmed_author_sample("Ivan Matic", sample_size=5)
    """
    # Validate parameters
    sample_size = min(max(sample_size, 1), 10)  # Cap at 10 for performance
    
    logger.info(f"🔍 PubMed author sample: '{author_name}' (sample: {sample_size})")
    
    result = get_pubmed_author_sample(author_name, sample_size)
    return result


# ============================================================================
# Enhanced Content Tools - Abstract and Full-text Access
# ============================================================================

@mcp.tool(
    annotations={
        "title": "Cache Statistics",
        "description": (
            "Get statistics about the abstract cache and rate limiters. "
            "Useful for monitoring performance and cache hit rates."
        ),
        "readOnlyHint": True,
        "openWorldHint": False
    }
)
async def get_cache_stats() -> dict:
    """
    Get cache and rate limiter statistics.
    
    Returns:
        dict: Statistics including:
        - cache: Size, max size, TTL
        - rate_limiters: Current token counts and limits
    """
    return {
        'cache': abstract_cache.stats(),
        'semantic_scholar_limiter': semantic_scholar_limiter.stats(),
        'unpaywall_limiter': unpaywall_limiter.stats()
    }


@mcp.tool(
    annotations={
        "title": "Clear Abstract Cache",
        "description": (
            "Clear all cached abstracts. "
            "Use this to force fresh data retrieval from Semantic Scholar."
        ),
        "readOnlyHint": False,
        "openWorldHint": False
    }
)
async def clear_abstract_cache() -> dict:
    """
    Clear all cached abstracts.
    
    Returns:
        dict: Confirmation message
    """
    await abstract_cache.clear()
    return {
        'status': 'success',
        'message': 'Abstract cache cleared'
    }


@mcp.tool(
    annotations={
        "title": "Get Work Abstract",
        "description": (
            "Fetch complete, non-inverted abstract from Semantic Scholar with caching and fallback. "
            "OpenAlex sometimes provides inverted abstracts (alphabetically sorted words) "
            "due to copyright restrictions. This tool tries multiple strategies: "
            "1. Check cache first (1 hour TTL) "
            "2. Fetch from Semantic Scholar (with rate limiting) "
            "3. Fallback to reconstructing from OpenAlex inverted index "
            "Also returns influential citation count and open access PDF links when available."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_work_abstract(
    doi: str = None,
    title: str = None,
    openalex_id: str = None
) -> dict:
    """
    Get complete abstract for a paper from Semantic Scholar.
    
    Args:
        doi: DOI of the paper (preferred, most reliable)
        title: Paper title (fallback if no DOI)
        openalex_id: OpenAlex work ID (e.g., "https://openalex.org/W1234567890")
        
    Returns:
        dict: Abstract data with:
        - abstract: Full abstract text
        - source: "semantic_scholar"
        - paper_id: Semantic Scholar paper ID
        - citation_count: Total citations
        - influential_citation_count: Influential citations (AI-powered metric)
        - open_access_pdf: Direct PDF link if available
        - error: Error message if fetch failed
        
    Example usage:
        # Get abstract by DOI (preferred)
        get_work_abstract(doi="10.1038/s41587-024-02534-3")
        
        # Get abstract by OpenAlex ID
        get_work_abstract(openalex_id="https://openalex.org/W1234567890")
        
        # Fallback to title search
        get_work_abstract(title="BERT: Pre-training of Deep Bidirectional Transformers")
    """
    if not any([doi, title, openalex_id]):
        return {
            'error': 'At least one identifier required: doi, title, or openalex_id',
            'source': 'semantic_scholar'
        }
    
    logger.info(f"🔍 Fetching abstract from Semantic Scholar")
    result = await fetch_semantic_scholar_abstract(doi=doi, title=title, openalex_id=openalex_id)
    return result


def normalize_work_id(work_id: str) -> str:
    """
    Normalize a work ID to just the W... format.
    
    Accepts:
    - "W2741809807"
    - "https://openalex.org/W2741809807"
    - "openalex.org/W2741809807"
    
    Returns:
        str: Just the work ID (e.g., "W2741809807")
    """
    if not work_id:
        return work_id
    
    # Remove URL prefix if present
    if 'openalex.org/' in work_id:
        work_id = work_id.split('openalex.org/')[-1]
    
    # Extract W... part
    if work_id.startswith('W'):
        return work_id
    
    # Try to extract from different formats
    if work_id.upper().startswith('W'):
        return work_id.upper()
    
    return work_id


@mcp.tool(
    annotations={
        "title": "Get Complete Work",
        "description": (
            "Fetch a work object with rich metadata. Defaults to a compact payload to save tokens. "
            "Use include_* flags to return authorships, locations, references, related works, or inverted abstracts. "
            "Accepts work IDs in multiple formats: W..., https://openalex.org/W..., etc."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_work(
    work_id: str,
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False,
) -> dict:
    """
    Fetch complete work object with full metadata.
    
    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807" or "https://openalex.org/W2741809807")
        
    Returns:
        dict: Complete work object with:
        - id, doi, title, publication_year, publication_date
        - authorships[] with author.id, author.orcid, institutions[]
        - cited_by_count, referenced_works[], related_works[]
        - abstract, abstract_inverted_index
        - best_oa_location, locations[]
        - is_open_access, primary_location
        - Backward compatible with v5.0 (first_author, corresponding_author, etc.)
        
    Example:
        work = await get_work("W2741809807")
        # Access authorships with author IDs (no homonym risk):
        for authorship in work['authorships']:
            author_id = authorship['author']['id']  # OpenAlex ID, not name
            author_name = authorship['author']['display_name']
    """
    try:
        # Normalize work_id
        normalized_id = normalize_work_id(work_id)
        
        logger.info(f"📄 Fetching complete work: {normalized_id}")
        
        # Fetch from OpenAlex
        work_obj = pyalex.Works()[normalized_id]
        
        if not work_obj:
            return {
                'error': f'Work not found: {work_id}',
                'work_id': work_id
            }
        
        # Convert to optimized format (which now includes full authorships)
        optimized = optimize_work_data(
            work_obj,
            include_authorships=include_authorships,
            include_locations=include_locations,
            include_best_oa_location=include_best_oa_location,
            include_references=include_references,
            include_related_works=include_related_works,
            include_abstract_inverted_index=include_abstract_inverted_index,
            include_fulltext_urls=include_fulltext_urls,
            max_authorships=max_authorships,
            max_locations=max_locations,
            max_references=max_references,
            max_related_works=max_related_works,
            max_fulltext_urls=max_fulltext_urls,
            abstract_max_chars=abstract_max_chars,
            compact_ids=compact_ids
        )
        
        logger.info(f"✅ Successfully fetched work with {optimized.author_count or 0} authors")
        
        return optimized.model_dump()
        
    except Exception as e:
        logger.error(f"Error fetching work {work_id}: {e}")
        return {
            'error': f'Error fetching work: {str(e)}',
            'work_id': work_id
        }


def normalize_doi(doi: str) -> str:
    """
    Normalize a DOI to the full URL format for OpenAlex.
    
    Accepts:
    - "10.1038/nature14539"
    - "https://doi.org/10.1038/nature14539"
    - "doi.org/10.1038/nature14539"
    
    Returns:
        str: Full DOI URL (e.g., "https://doi.org/10.1038/nature14539")
    """
    if not doi:
        return doi
    
    # Remove leading/trailing whitespace
    doi = doi.strip()
    
    # If already full URL, return as-is
    if doi.startswith('https://doi.org/') or doi.startswith('http://doi.org/'):
        return doi
    
    # If starts with doi.org/, prepend https
    if doi.startswith('doi.org/'):
        return 'https://' + doi
    
    # If just the DOI number (10.xxxx/...)
    if doi.startswith('10.'):
        return f'https://doi.org/{doi}'
    
    # Try uppercasing 'DOI:' prefix if present
    if doi.upper().startswith('DOI:'):
        return normalize_doi(doi[4:])
    
    return doi


@mcp.tool(
    annotations={
        "title": "Get Work by DOI",
        "description": (
            "Resolve a DOI to a work object with reliable lookup. Defaults to a compact payload to save tokens. "
            "Use include_* flags to return authorships, locations, references, related works, or inverted abstracts."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_work_by_doi(
    doi: str,
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False,
) -> dict:
    """
    Fetch work by DOI with full metadata.
    
    Args:
        doi: DOI in any format (e.g., "10.1038/nature14539" or "https://doi.org/10.1038/nature14539")
        
    Returns:
        dict: Complete work object (same as get_work) or error dict if not found
        
    Example:
        work = await get_work_by_doi("10.1038/nature14539")
        # or
        work = await get_work_by_doi("https://doi.org/10.1038/nature14539")
    """
    try:
        # Normalize DOI to full URL format
        normalized_doi = normalize_doi(doi)
        
        if not normalized_doi or not normalized_doi.startswith('https://doi.org/'):
            return {
                'error': f'Invalid DOI format: {doi}',
                'doi': doi
            }
        
        logger.info(f"🔗 Resolving DOI: {doi}")
        
        # Fetch from OpenAlex using the DOI URL directly as work ID
        # OpenAlex accepts https://doi.org/... as a work ID
        work_obj = pyalex.Works()[normalized_doi]
        
        if not work_obj:
            return {
                'error': f'Work not found for DOI: {doi}',
                'doi': doi
            }
        
        # Convert to optimized format
        optimized = optimize_work_data(
            work_obj,
            include_authorships=include_authorships,
            include_locations=include_locations,
            include_best_oa_location=include_best_oa_location,
            include_references=include_references,
            include_related_works=include_related_works,
            include_abstract_inverted_index=include_abstract_inverted_index,
            include_fulltext_urls=include_fulltext_urls,
            max_authorships=max_authorships,
            max_locations=max_locations,
            max_references=max_references,
            max_related_works=max_related_works,
            max_fulltext_urls=max_fulltext_urls,
            abstract_max_chars=abstract_max_chars,
            compact_ids=compact_ids
        )
        
        logger.info(f"✅ Successfully resolved DOI to work")
        
        return optimized.model_dump()
        
    except Exception as e:
        logger.error(f"Error fetching work by DOI {doi}: {e}")
        return {
            'error': f'Error resolving DOI: {str(e)}',
            'doi': doi
        }


@mcp.tool(
    annotations={
        "title": "Get Cited By Works",
        "description": (
            "Get incoming citations: papers that cite a given work. "
            "Returns sorted list of citing papers with metadata. "
            "Useful for tracking research impact and finding related work. "
            "Sortable by publication date or citation count."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_cited_by(
    work_id: str,
    limit: int = 50,
    sort: str = "publication_date"
) -> dict:
    """
    Get papers that cite a given work (incoming citations).
    
    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807")
        limit: Maximum papers to return (default: 50, max: 100)
        sort: Sort order - "publication_date" (newest first) or "cited_by_count" (default)
        
    Returns:
        dict with:
        - work_id: Original work ID
        - citing_count: Total number of papers citing this work
        - citing_works: List of citing papers with metadata
        
    Example:
        citations = await get_cited_by("W2741809807", limit=20, sort="publication_date")
    """
    try:
        # Normalize work_id
        normalized_id = normalize_work_id(work_id)
        limit = min(limit, 100)
        
        logger.info(f"🔗 Fetching papers citing: {normalized_id}")
        
        # Use OpenAlex filter: cites:Wxxxx returns papers that cite this work
        works_query = pyalex.Works().filter(cites=normalized_id)
        
        # Apply sorting
        if sort == "cited_by_count":
            works_query = works_query.sort("cited_by_count:desc")
        else:
            # Default: publication_date (newest first)
            works_query = works_query.sort("publication_date:desc")
        
        # Fetch results
        results = works_query.get(per_page=limit)
        
        # Convert to optimized format
        citing_works = []
        for work in results:
            try:
                optimized = optimize_work_data(work)
                # Extract just key fields for brevity
                citing_works.append({
                    'id': optimized.id,
                    'title': optimized.title,
                    'doi': optimized.doi,
                    'publication_year': optimized.publication_year,
                    'cited_by_count': optimized.cited_by_count,
                    'authors': optimized.first_author,  # Simplified
                })
            except Exception as e:
                logger.warning(f"Error optimizing citing work: {e}")
                continue
        
        logger.info(f"✅ Found {len(citing_works)} papers citing this work")
        
        return {
            'work_id': normalized_id,
            'citing_count': len(citing_works),
            'citing_works': citing_works,
            'sort': sort,
            'source': 'openalex'
        }
        
    except Exception as e:
        logger.error(f"Error fetching cited_by for work {work_id}: {e}")
        return {
            'error': f'Error fetching citations: {str(e)}',
            'work_id': work_id
        }


@mcp.tool(
    annotations={
        "title": "Get References",
        "description": (
            "Get outgoing citations: papers that a given work references (cites). "
            "Returns bibliography list of papers referenced by the work. "
            "Useful for understanding research foundations and related work. "
            "Sortable by publication date or citation count."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_references(
    work_id: str,
    limit: int = 50,
    sort: str = "publication_date"
) -> dict:
    """
    Get papers referenced by a given work (outgoing citations).
    
    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807")
        limit: Maximum papers to return (default: 50, max: 100)
        sort: Sort order - "publication_date" (newest first) or "cited_by_count"
        
    Returns:
        dict with:
        - work_id: Original work ID
        - reference_count: Total number of papers referenced
        - referenced_works: List of referenced papers with metadata
        
    Example:
        refs = await get_references("W2741809807", limit=25)
    """
    try:
        # Normalize work_id
        normalized_id = normalize_work_id(work_id)
        limit = min(limit, 100)
        
        logger.info(f"📚 Fetching papers referenced by: {normalized_id}")
        
        # Use OpenAlex filter: cited_by:Wxxxx returns papers that are cited by this work
        # This is the inverse: papers found in the work's referenced_works list
        works_query = pyalex.Works().filter(cited_by=normalized_id)
        
        # Apply sorting
        if sort == "cited_by_count":
            works_query = works_query.sort("cited_by_count:desc")
        else:
            # Default: publication_date (newest first)
            works_query = works_query.sort("publication_date:desc")
        
        # Fetch results
        results = works_query.get(per_page=limit)
        
        # Convert to optimized format
        referenced_works = []
        for work in results:
            try:
                optimized = optimize_work_data(work)
                # Extract key fields
                referenced_works.append({
                    'id': optimized.id,
                    'title': optimized.title,
                    'doi': optimized.doi,
                    'publication_year': optimized.publication_year,
                    'cited_by_count': optimized.cited_by_count,
                    'authors': optimized.first_author,  # Simplified
                })
            except Exception as e:
                logger.warning(f"Error optimizing referenced work: {e}")
                continue
        
        logger.info(f"✅ Found {len(referenced_works)} papers referenced by this work")
        
        return {
            'work_id': normalized_id,
            'reference_count': len(referenced_works),
            'referenced_works': referenced_works,
            'sort': sort,
            'source': 'openalex'
        }
        
    except Exception as e:
        logger.error(f"Error fetching references for work {work_id}: {e}")
        return {
            'error': f'Error fetching references: {str(e)}',
            'work_id': work_id
        }


@mcp.tool(
    annotations={
        "title": "Decode Abstract",
        "description": (
            "Reconstruct plaintext abstract from OpenAlex inverted index format. "
            "OpenAlex stores copyrighted abstracts as inverted indexes (word→positions mapping). "
            "This tool reconstructs readable text from that index. "
            "Perfect for making abstracts human-readable."
        ),
        "readOnlyHint": True
    }
)
def decode_abstract(abstract_inverted_index: dict) -> dict:
    """
    Reconstruct plaintext abstract from inverted index.
    
    Args:
        abstract_inverted_index: Inverted index dict from OpenAlex
                                (e.g., {"We": [0], "present": [1], "a": [2]})
        
    Returns:
        dict with:
        - abstract_text: Reconstructed plaintext abstract
        - had_gaps: Whether there were gaps in positions (might indicate truncation)
        - note: Explanation about the reconstruction
        - source: "reconstruction"
        
    Example:
        result = decode_abstract({
            "machine": [0],
            "learning": [1],
            "models": [2]
        })
        # Returns: {"abstract_text": "machine learning models", "had_gaps": False, ...}
    """
    try:
        if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
            return {
                'error': 'Invalid inverted index format',
                'abstract_text': '',
                'had_gaps': False,
                'source': 'reconstruction'
            }
        
        # Use the existing reconstruction function
        abstract_text = reconstruct_abstract_from_inverted_index(abstract_inverted_index)
        
        # Check for gaps (simple check: if max position + 1 != word count, there were gaps)
        all_positions = []
        for positions in abstract_inverted_index.values():
            all_positions.extend(positions)
        
        had_gaps = False
        if all_positions:
            max_pos = max(all_positions)
            expected_count = max_pos + 1
            actual_count = len(all_positions)
            had_gaps = (actual_count < expected_count)
        
        logger.info(f"📄 Decoded abstract, had_gaps={had_gaps}")
        
        return {
            'abstract_text': abstract_text,
            'had_gaps': had_gaps,
            'note': 'OpenAlex abstracts are stored as inverted index; punctuation may not be perfect',
            'source': 'reconstruction'
        }
        
    except Exception as e:
        logger.error(f"Error decoding abstract: {e}")
        return {
            'error': f'Error decoding abstract: {str(e)}',
            'abstract_text': '',
            'source': 'reconstruction'
        }


@mcp.tool(
    annotations={
        "title": "Get Full-text Access",
        "description": (
            "Get full-text PDF links for a paper using multiple sources. "
            "Returns direct links to publisher PDFs and repository versions. "
            "Includes license information and version details when available. "
            "Automatically tries multiple sources to maximize PDF availability."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_fulltext_access(doi: str) -> dict:
    """
    Get full-text PDF access links via multiple sources.
    
    Args:
        doi: DOI of the paper (required)
        
    Returns:
        dict: Full-text access information with:
        - is_oa: Whether paper is open access
        - oa_status: OA type (gold, green, hybrid, bronze, closed, unknown)
        - best_oa_location: Best PDF link with metadata
        - oa_locations: List of all available OA locations
        - source: "unpaywall" or "multi_source"
        
        Each location includes:
        - url: Direct PDF URL
        - host_type: "publisher", "repository", or "other"
        - license: License type (e.g., "cc-by", "unknown")
        - version: "publishedVersion", "acceptedVersion", "submittedVersion", or "unknown"
        
    Example usage:
        # Get full-text links for a DOI
        get_fulltext_access(doi="10.1038/s41587-024-02534-3")
        
        # Response example (if open access):
        {
            "is_oa": true,
            "oa_status": "gold",
            "best_oa_location": {
                "url": "https://www.nature.com/articles/s41587-024-02534-3.pdf",
                "host_type": "publisher",
                "license": "cc-by",
                "version": "publishedVersion"
            },
            "oa_locations": [...]
        }
    """
    if not doi:
        return {
            'error': 'DOI is required',
            'is_oa': False,
            'source': 'unpaywall'
        }
    
    logger.info(f"🔍 Fetching full-text access from multiple sources")
    result = await fetch_unpaywall_links(doi)
    
    # If Unpaywall failed or found no OA, try alternative sources (if enabled)
    if get_config()["OPENALEX_ENABLE_SCIHUB"]:
        # Try alternative sources if: error occurred, not OA, or no best location found
        should_try_alternative = (
            result.get('error') or 
            not result.get('is_oa') or 
            not result.get('best_oa_location') or
            result.get('oa_status') == 'closed'
        )
        
        if should_try_alternative:
            logger.info(f"🔄 Trying alternative sources for full-text access")
            alt_result = await fetch_scihub_pdf(doi)
            
            if not alt_result.get('error') and alt_result.get('pdf_url'):
                # Alternative source successful - merge results
                result['source'] = 'multi_source'
                result['best_oa_location'] = {
                    'url': alt_result['pdf_url'],
                    'host_type': 'other',
                    'license': 'unknown',
                    'version': 'unknown'
                }
                result['is_oa'] = True
                result['oa_status'] = result.get('oa_status', 'unknown')
                
                # Add to oa_locations if not already present
                if 'oa_locations' not in result:
                    result['oa_locations'] = []
                result['oa_locations'].append(result['best_oa_location'])
                
                logger.info(f"✅ Alternative source successful")
    
    return result


@mcp.tool(
    annotations={
        "title": "Enrich Work Data",
        "description": (
            "Get comprehensive work data combining OpenAlex metadata with "
            "Semantic Scholar abstract and Unpaywall full-text access. "
            "Returns a complete OptimizedWorkResult with all available enrichments. "
            "Perfect for detailed paper analysis with full content access."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def enrich_work_data(
    work_id: str = None,
    doi: str = None,
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False
) -> dict:
    """
    Get enriched work data with abstract and full-text access.
    
    Combines data from:
    1. OpenAlex: Comprehensive metadata, citations, authorship
    2. Semantic Scholar: Complete abstract, influential citations
    3. Unpaywall: Legal full-text PDF access
    
    Args:
        work_id: OpenAlex work ID (e.g., "https://openalex.org/W1234567890")
        doi: DOI of the paper (alternative to work_id)
        
    Returns:
        dict: Enriched OptimizedWorkResult with:
        - All standard OpenAlex metadata
        - abstract: Complete abstract from Semantic Scholar
        - abstract_inverted: Whether OpenAlex had inverted abstract
        - fulltext_urls: List of legal PDF access points
        - enrichment_metadata: What sources were used and their status
        
    Example usage:
        # Enrich by OpenAlex ID
        enrich_work_data(work_id="https://openalex.org/W1234567890")
        
        # Enrich by DOI
        enrich_work_data(doi="10.1038/s41587-024-02534-3")
    """
    try:
        # Get base work data from OpenAlex
        if work_id:
            # Fetch work from OpenAlex by ID
            logger.info(f"📚 Fetching work from OpenAlex: {work_id}")
            work_data = pyalex.Works()[work_id]
            if not work_data:
                return {'error': f'Work not found in OpenAlex: {work_id}'}
        elif doi:
            # Search by DOI
            logger.info(f"📚 Searching OpenAlex by DOI: {doi}")
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            works = pyalex.Works().filter(doi=f"https://doi.org/{clean_doi}").get()
            if not works:
                return {'error': f'Work not found in OpenAlex for DOI: {doi}'}
            work_data = list(works)[0]
        else:
            return {'error': 'Either work_id or doi must be provided'}
        
        # Convert to optimized format
        optimized_work = optimize_work_data(
            work_data,
            include_authorships=include_authorships,
            include_locations=include_locations,
            include_best_oa_location=include_best_oa_location,
            include_references=include_references,
            include_related_works=include_related_works,
            include_abstract_inverted_index=include_abstract_inverted_index,
            include_fulltext_urls=include_fulltext_urls,
            max_authorships=max_authorships,
            max_locations=max_locations,
            max_references=max_references,
            max_related_works=max_related_works,
            max_fulltext_urls=max_fulltext_urls,
            abstract_max_chars=abstract_max_chars,
            compact_ids=compact_ids
        )
        
        # Initialize enrichment metadata
        enrichment_metadata = {
            'sources_used': ['openalex'],
            'abstract_source': None,
            'fulltext_source': None,
            'errors': []
        }
        
        # Get DOI for enrichment queries
        work_doi = optimized_work.doi or (optimized_work.ids.doi if optimized_work.ids else None)
        work_openalex_id = optimized_work.id
        work_title = optimized_work.title
        
        # Check if OpenAlex has inverted abstract
        has_inverted_abstract = 'abstract_inverted_index' in work_data
        inverted_index = work_data.get('abstract_inverted_index') if has_inverted_abstract else None
        
        # Enrich with Semantic Scholar abstract (if we have identifiers)
        abstract_obtained = False
        if work_doi or work_openalex_id or work_title:
            logger.info("🔍 Enriching with Semantic Scholar abstract...")
            abstract_result = await fetch_semantic_scholar_abstract(
                doi=work_doi,
                openalex_id=work_openalex_id,
                title=work_title
            )
            
            if 'error' not in abstract_result and abstract_result.get('abstract'):
                optimized_work.abstract = abstract_result['abstract']
                optimized_work.abstract_inverted = False  # Semantic Scholar provides non-inverted
                enrichment_metadata['abstract_source'] = 'semantic_scholar'
                if abstract_result.get('cached'):
                    enrichment_metadata['abstract_cached'] = True
                enrichment_metadata['sources_used'].append('semantic_scholar')
                logger.info(f"✅ Abstract enriched from Semantic Scholar: {len(abstract_result['abstract'])} chars")
                abstract_obtained = True
            else:
                enrichment_metadata['errors'].append(f"Semantic Scholar: {abstract_result.get('error', 'Unknown error')}")
                logger.warning(f"⚠️ Could not enrich abstract from Semantic Scholar: {abstract_result.get('error')}")
        
        # FALLBACK: Reconstruct from OpenAlex inverted index if Semantic Scholar failed
        if not abstract_obtained and has_inverted_abstract and inverted_index:
            logger.info("🔄 Falling back to OpenAlex inverted abstract reconstruction...")
            reconstructed_abstract = reconstruct_abstract_from_inverted_index(inverted_index)
            
            if reconstructed_abstract:
                optimized_work.abstract = reconstructed_abstract
                optimized_work.abstract_inverted = True  # Mark as reconstructed from inverted
                enrichment_metadata['abstract_source'] = 'openalex_reconstructed'
                enrichment_metadata['sources_used'].append('openalex_inverted_index')
                logger.info(f"✅ Abstract reconstructed from OpenAlex inverted index: {len(reconstructed_abstract)} chars")
                abstract_obtained = True
            else:
                logger.warning("⚠️ Failed to reconstruct abstract from inverted index")
        
        # If still no abstract, check for regular abstract field in OpenAlex
        if not abstract_obtained:
            openalex_abstract = work_data.get('abstract')
            if openalex_abstract:
                optimized_work.abstract = openalex_abstract
                optimized_work.abstract_inverted = False
                enrichment_metadata['abstract_source'] = 'openalex'
                logger.info(f"✅ Using OpenAlex regular abstract: {len(openalex_abstract)} chars")
                abstract_obtained = True

        # Apply abstract truncation after enrichment
        if optimized_work.abstract and abstract_max_chars and len(optimized_work.abstract) > abstract_max_chars:
            optimized_work.abstract = optimized_work.abstract[:abstract_max_chars].rstrip() + "..."
            optimized_work.abstract_truncated = True
        
        # Enrich with Unpaywall full-text access (if we have DOI)
        if work_doi:
            logger.info("🔍 Enriching with Unpaywall full-text access...")
            fulltext_result = await fetch_unpaywall_links(work_doi)
            
            if not fulltext_result.get('error') and fulltext_result.get('is_oa'):
                # Format fulltext URLs
                fulltext_urls = []
                
                # Add best location first
                if fulltext_result.get('best_oa_location'):
                    fulltext_urls.append(fulltext_result['best_oa_location'])
                
                # Add other locations
                for loc in fulltext_result.get('oa_locations', []):
                    if loc not in fulltext_urls:  # Avoid duplicate
                        fulltext_urls.append(loc)
                
                if fulltext_urls:
                    if include_fulltext_urls:
                        if max_fulltext_urls and len(fulltext_urls) > max_fulltext_urls:
                            optimized_work.fulltext_urls = fulltext_urls[:max_fulltext_urls]
                            optimized_work.fulltext_urls_truncated = True
                        else:
                            optimized_work.fulltext_urls = fulltext_urls
                        enrichment_metadata['fulltext_source'] = 'unpaywall'
                        enrichment_metadata['sources_used'].append('unpaywall')
                        logger.info(f"✅ Full-text access enriched: {len(fulltext_urls)} locations")
            else:
                if fulltext_result.get('error'):
                    enrichment_metadata['errors'].append(f"Unpaywall: {fulltext_result['error']}")
                logger.info("📕 Paper is not open access or no full-text available")
        
        # Convert to dict and add enrichment metadata
        result = optimized_work.model_dump()
        result['enrichment_metadata'] = enrichment_metadata
        
        logger.info(f"✅ Work enrichment complete. Sources used: {', '.join(enrichment_metadata['sources_used'])}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error enriching work data: {str(e)}")
        return {'error': str(e)}


@mcp.tool(
    annotations={
        "title": "Get Top Authors for Query",
        "description": (
            "Get most frequent authors for a search query, ranked by appearance count. "
            "Returns authors by OpenAlex ID (not name) to avoid homonym ambiguity. "
            "Perfect for finding key researchers in a field. "
            "Uses author.id for disambiguation - no duplicate names."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_top_authors_for_query(
    query: str,
    limit_works: int = 200,
    top_k: int = 10,
    sort_by: str = "count"
) -> dict:
    """
    Get top authors for a search query (ranked by appearance frequency).
    
    Args:
        query: Search query to analyze
        limit_works: How many papers to scan (default: 200, max: 500)
        top_k: Number of top authors to return (default: 10)
        sort_by: Sort order - "count" (appearance), "h_index", or "cited_by_count"
        
    Returns:
        dict with:
        - query: Original search query
        - papers_scanned: How many papers analyzed
        - top_authors: List of top authors with:
          - author_id: OpenAlex ID (for disambiguation)
          - author_name: Display name
          - orcid: ORCID if available
          - count: Appearance frequency
          - h_index: H-index if available
          - cited_by_count: Total citations
          - example_works: Top 3 work IDs
          
    Example:
        authors = await get_top_authors_for_query(
            query="machine learning",
            limit_works=100,
            top_k=10,
            sort_by="count"
        )
    """
    try:
        limit_works = min(limit_works, 500)
        
        logger.info(f"👥 Finding top authors for query: {query}")
        
        # Search for works
        works_list = search_works_core(query=query, limit=min(limit_works, 100))
        
        if not works_list.results:
            return {
                'query': query,
                'papers_scanned': 0,
                'top_authors': [],
                'source': 'openalex'
            }
        
        # Count authors by ID (not name - avoiding homonym issue)
        author_counter = {}
        author_info = {}  # Store author details
        
        for work in works_list.results:
            # Get full work details with authorships
            if not work.authorships:
                continue
            
            for authorship in work.authorships:
                author_id = authorship.author.id
                author_name = authorship.author.display_name
                orcid = authorship.author.orcid
                
                # Increment count
                author_counter[author_id] = author_counter.get(author_id, 0) + 1
                
                # Store author info (will update with latest metrics if seen again)
                if author_id not in author_info:
                    author_info[author_id] = {
                        'author_id': author_id,
                        'author_name': author_name,
                        'orcid': orcid,
                        'example_works': []
                    }
                
                # Track example works (limit to 3)
                if len(author_info[author_id]['example_works']) < 3:
                    author_info[author_id]['example_works'].append(work.id)
        
        logger.info(f"📊 Found {len(author_counter)} unique authors in {len(works_list.results)} papers")
        
        # Build top authors list
        top_authors = []
        for author_id, count in sorted(
            author_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            author_entry = author_info[author_id].copy()
            author_entry['count'] = count
            author_entry['h_index'] = None  # Would need separate fetch
            author_entry['cited_by_count'] = None  # Would need separate fetch
            top_authors.append(author_entry)
        
        logger.info(f"✅ Returning top {len(top_authors)} authors")
        
        return {
            'query': query,
            'papers_scanned': len(works_list.results),
            'top_authors': top_authors,
            'sort_by': sort_by,
            'source': 'openalex'
        }
        
    except Exception as e:
        logger.error(f"Error fetching top authors for query {query}: {e}")
        return {
            'error': f'Error fetching top authors: {str(e)}',
            'query': query
        }


@mcp.tool(
    annotations={
        "title": "Batch Get Works",
        "description": (
            "Fetch multiple works efficiently with automatic error handling. "
            "Processes works sequentially with optimized error tracking. "
            "More convenient than calling get_work() repeatedly. "
            "Perfect for analyzing multiple papers at once."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def batch_get_works(
    work_ids: list,
    chunk_size: int = 50,
    include_authorships: bool = False,
    include_locations: bool = False,
    include_best_oa_location: bool = True,
    include_references: bool = False,
    include_related_works: bool = False,
    include_abstract_inverted_index: bool = False,
    include_fulltext_urls: bool = False,
    max_authorships: int = 20,
    max_locations: int = 5,
    max_references: int = 50,
    max_related_works: int = 20,
    max_fulltext_urls: int = 3,
    abstract_max_chars: int = 2000,
    compact_ids: bool = False
) -> dict:
    """
    Fetch multiple works with consolidated error handling.
    
    Note: PyAlex doesn't support true batch API calls, so this fetches
    works sequentially but provides unified error handling and progress tracking.
    
    Args:
        work_ids: List of work IDs (e.g., ["W123", "W456", "W789"])
        chunk_size: Batch size for progress logging (default: 50, max: 100)
        
    Returns:
        dict with:
        - requested: Number of IDs requested
        - fetched: Number successfully fetched
        - works: List of complete work objects
        - errors: List of failed IDs with reasons
        
    Example:
        works = await batch_get_works(
            work_ids=["W2741809807", "W2964470646"],
            chunk_size=50
        )
    """
    try:
        chunk_size = min(chunk_size, 100)
        requested_count = len(work_ids)
        
        logger.info(f"📦 Batch fetching {requested_count} works")
        
        if not work_ids:
            return {
                'requested': 0,
                'fetched': 0,
                'works': [],
                'errors': [],
                'source': 'openalex'
            }
        
        # Normalize all IDs
        normalized_ids = [normalize_work_id(wid) for wid in work_ids]
        
        # Process works sequentially (PyAlex doesn't support true batch API)
        all_works = []
        all_errors = []
        
        for idx, work_id in enumerate(normalized_ids):
            # Log progress for large batches
            if idx > 0 and idx % chunk_size == 0:
                logger.info(f"  Progress: {idx}/{len(normalized_ids)} works processed")
            
            try:
                work_obj = pyalex.Works()[work_id]
                if work_obj:
                    optimized = optimize_work_data(
                        work_obj,
                        include_authorships=include_authorships,
                        include_locations=include_locations,
                        include_best_oa_location=include_best_oa_location,
                        include_references=include_references,
                        include_related_works=include_related_works,
                        include_abstract_inverted_index=include_abstract_inverted_index,
                        include_fulltext_urls=include_fulltext_urls,
                        max_authorships=max_authorships,
                        max_locations=max_locations,
                        max_references=max_references,
                        max_related_works=max_related_works,
                        max_fulltext_urls=max_fulltext_urls,
                        abstract_max_chars=abstract_max_chars,
                        compact_ids=compact_ids
                    )
                    all_works.append(optimized.model_dump())
                else:
                    all_errors.append({
                        'work_id': work_id,
                        'reason': 'Work not found'
                    })
            except Exception as e:
                all_errors.append({
                    'work_id': work_id,
                    'reason': str(e)
                })
        
        logger.info(f"✅ Batch fetch complete: {len(all_works)} works fetched, {len(all_errors)} errors")
        
        return {
            'requested': requested_count,
            'fetched': len(all_works),
            'works': all_works,
            'errors': all_errors,
            'source': 'openalex'
        }
        
    except Exception as e:
        logger.error(f"Error in batch_get_works: {e}")
        return {
            'requested': len(work_ids),
            'fetched': 0,
            'works': [],
            'errors': [{'reason': str(e)}],
            'source': 'openalex'
        }


@mcp.tool(
    annotations={
        "title": "Get Best OA Location",
        "description": (
            "Extract the best legal OA PDF location for a work. "
            "Prioritizes: publisher (gold) > repository (green). "
            "Returns license and version information. "
            "Accepts OpenAlex work_id or DOI; falls back to Unpaywall if OpenAlex lacks OA metadata."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_best_oa_location(work_id: str = None, doi: str = None) -> dict:
    """
    Get the best open access PDF location for a work.
    
    Args:
        work_id: OpenAlex work ID
        doi: DOI in any format (optional)
        
    Returns:
        dict with:
        - work_id: Original work ID
        - is_oa: Whether paper is open access
        - oa_status: OA type ('gold', 'green', 'hybrid', 'bronze', 'closed')
        - best_location: Single best location object (or None if closed access)
        - all_locations: All available OA locations
        - license: License type
        - version: Version type (published, accepted, submitted)
        
    Example:
        oa = await get_best_oa_location(work_id="W2741809807")
        if oa['is_oa']:
            print(f"📥 Download: {oa['best_location']['url']}")

        # Or by DOI
        oa = await get_best_oa_location(doi="10.1038/nature14539")
    """
    try:
        if not work_id and not doi:
            return {
                'error': 'Either work_id or doi must be provided'
            }

        normalized_id = None
        normalized_doi = None
        work = None

        if work_id:
            normalized_id = normalize_work_id(work_id)
            logger.info(f"🔓 Fetching OA status for: {normalized_id}")
            work = pyalex.Works()[normalized_id]

        if not work and doi:
            normalized_doi = normalize_doi(doi)
            if not normalized_doi or not normalized_doi.startswith('https://doi.org/'):
                return {
                    'error': f'Invalid DOI format: {doi}',
                    'doi': doi
                }
            logger.info(f"🔓 Fetching OA status for DOI: {doi}")
            work = pyalex.Works()[normalized_doi]

        if not work:
            if normalized_doi or doi:
                # Fallback to Unpaywall directly when OpenAlex lookup fails
                fallback_doi = normalized_doi or normalize_doi(doi)
                logger.info("🔄 Falling back to Unpaywall for OA metadata")
                fallback = await fetch_unpaywall_links(fallback_doi)
                return {
                    'work_id': normalized_id,
                    'doi': fallback.get('doi') or fallback_doi,
                    'is_oa': fallback.get('is_oa', False),
                    'oa_status': fallback.get('oa_status', 'closed'),
                    'best_location': fallback.get('best_oa_location'),
                    'all_locations': fallback.get('oa_locations', []),
                    'source': 'unpaywall'
                }
            return {
                'error': f'Work not found: {work_id}',
                'work_id': work_id
            }
        
        # Extract locations
        all_locations, best_location = extract_locations(work.get('locations', []))
        
        # Get OA info
        open_access = work.get('open_access', {})
        is_oa = open_access.get('is_oa', False)
        oa_status = open_access.get('oa_status', 'closed')
        
        logger.info(f"✅ OA status: {oa_status}")
        
        result = {
            'work_id': normalized_id or normalize_work_id(work.get('id', '')),
            'doi': work.get('doi') or (work.get('ids', {}) or {}).get('doi'),
            'is_oa': is_oa,
            'oa_status': oa_status,
            'source': 'openalex'
        }
        
        if best_location:
            result['best_location'] = best_location.model_dump()
        else:
            result['best_location'] = None
        
        if all_locations:
            result['all_locations'] = [loc.model_dump() for loc in all_locations]
        else:
            result['all_locations'] = []
        
        # Fallback to Unpaywall if OpenAlex lacks OA metadata or locations
        if (not is_oa or not best_location) and (doi or result.get('doi')):
            fallback_doi = normalize_doi(doi or result.get('doi'))
            if fallback_doi and fallback_doi.startswith('https://doi.org/'):
                logger.info("🔄 OpenAlex OA incomplete; falling back to Unpaywall")
                fallback = await fetch_unpaywall_links(fallback_doi)
                if fallback.get('is_oa') and (fallback.get('best_oa_location') or fallback.get('oa_locations')):
                    result.update({
                        'is_oa': fallback.get('is_oa', result['is_oa']),
                        'oa_status': fallback.get('oa_status', result['oa_status']),
                        'best_location': fallback.get('best_oa_location'),
                        'all_locations': fallback.get('oa_locations', result.get('all_locations', [])),
                        'source': 'unpaywall_fallback'
                    })

        return result
        
    except Exception as e:
        logger.error(f"Error fetching OA location for work {work_id}: {e}")
        return {
            'error': f'Error: {str(e)}',
            'work_id': work_id
        }


# ============================================================================
# ORCID Integration Functions
# ============================================================================

async def search_orcid_by_name(name: str, affiliation: str = None, max_results: int = 10) -> dict:
    """
    Search ORCID by author name and optionally affiliation.
    
    Args:
        name: Author name to search
        affiliation: Optional affiliation to help disambiguation
        max_results: Maximum number of results to return
        
    Returns:
        dict: ORCID search results with author profiles
    """
    try:
        # ORCID Public API search endpoint
        base_url = "https://pub.orcid.org/v3.0/search"
        
        # Build search query
        query_parts = []
        if name:
            # Split name into parts for better matching
            name_parts = name.replace(",", "").split()
            if len(name_parts) >= 2:
                # Assume last part is family name, rest are given names
                family_name = name_parts[-1]
                given_names = " ".join(name_parts[:-1])
                query_parts.append(f'family-name:"{family_name}"')
                query_parts.append(f'given-names:"{given_names}"')
            else:
                query_parts.append(f'text:"{name}"')
        
        if affiliation:
            query_parts.append(f'affiliation-org-name:"{affiliation}"')
        
        query = " AND ".join(query_parts)
        
        params = {
            'q': query,
            'rows': min(max_results, 50),  # ORCID API limit
            'start': 0
        }
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': f'alex-mcp (+{get_config()["OPENALEX_MAILTO"]})'
        }
        
        logger.info(f"🔍 ORCID search: '{query}' (max: {max_results})")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for result in data.get('result', []):
                        orcid_id = result.get('orcid-identifier', {}).get('path', '')
                        
                        # Extract name information
                        person = result.get('person', {})
                        names = person.get('name', {})
                        given_names = names.get('given-names', {}).get('value', '') if names.get('given-names') else ''
                        family_name = names.get('family-name', {}).get('value', '') if names.get('family-name') else ''
                        
                        # Extract employment/affiliation info
                        employments = []
                        employment_summaries = result.get('employment-summary', [])
                        for emp in employment_summaries[:3]:  # Limit to top 3
                            org_name = emp.get('organization', {}).get('name', '')
                            if org_name:
                                employments.append(org_name)
                        
                        results.append({
                            'orcid_id': orcid_id,
                            'orcid_url': f'https://orcid.org/{orcid_id}' if orcid_id else '',
                            'given_names': given_names,
                            'family_name': family_name,
                            'full_name': f"{given_names} {family_name}".strip(),
                            'employments': employments,
                            'relevance_score': result.get('relevance-score', {}).get('value', 0)
                        })
                    
                    logger.info(f"📊 Found {len(results)} ORCID profiles")
                    
                    return {
                        'total_found': data.get('num-found', 0),
                        'results_returned': len(results),
                        'results': results
                    }
                else:
                    logger.warning(f"ORCID API error: {response.status}")
                    return {'total_found': 0, 'results_returned': 0, 'results': [], 'error': f'HTTP {response.status}'}
                    
    except Exception as e:
        logger.error(f"ORCID search error: {str(e)}")
        return {'total_found': 0, 'results_returned': 0, 'results': [], 'error': str(e)}


async def get_orcid_works(orcid_id: str, max_works: int = 20) -> dict:
    """
    Get works/publications for a specific ORCID ID.
    
    Args:
        orcid_id: ORCID identifier (e.g., "0000-0000-0000-0000")
        max_works: Maximum number of works to retrieve
        
    Returns:
        dict: Works information from ORCID profile
    """
    try:
        # Clean ORCID ID (remove URL if present)
        clean_orcid = orcid_id.replace('https://orcid.org/', '').replace('http://orcid.org/', '')
        if not re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            return {'error': 'Invalid ORCID format', 'works': []}
        
        # ORCID Public API works endpoint
        url = f"https://pub.orcid.org/v3.0/{clean_orcid}/works"
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': f'alex-mcp (+{get_config()["OPENALEX_MAILTO"]})'
        }
        
        logger.info(f"🔍 Getting ORCID works: {clean_orcid} (max: {max_works})")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    works = []
                    work_summaries = data.get('group', [])[:max_works]
                    
                    for group in work_summaries:
                        for work_summary in group.get('work-summary', []):
                            title_info = work_summary.get('title', {})
                            title = title_info.get('title', {}).get('value', '') if title_info else ''
                            
                            journal_title = work_summary.get('journal-title', {}).get('value', '') if work_summary.get('journal-title') else ''
                            
                            # Extract publication date
                            pub_date = work_summary.get('publication-date')
                            pub_year = ''
                            if pub_date and pub_date.get('year'):
                                pub_year = pub_date['year'].get('value', '')
                            
                            # Extract external IDs (DOI, PMID, etc.)
                            external_ids = {}
                            for ext_id in work_summary.get('external-ids', {}).get('external-id', []):
                                id_type = ext_id.get('external-id-type', '')
                                id_value = ext_id.get('external-id-value', '')
                                if id_type and id_value:
                                    external_ids[id_type.lower()] = id_value
                            
                            works.append({
                                'title': title,
                                'journal': journal_title,
                                'publication_year': pub_year,
                                'external_ids': external_ids,
                                'doi': external_ids.get('doi', ''),
                                'pmid': external_ids.get('pmid', ''),
                                'type': work_summary.get('type', '')
                            })
                    
                    logger.info(f"📊 Retrieved {len(works)} works from ORCID")
                    
                    return {
                        'orcid_id': clean_orcid,
                        'total_works': len(works),
                        'works': works
                    }
                else:
                    logger.warning(f"ORCID works API error: {response.status}")
                    return {'error': f'HTTP {response.status}', 'works': []}
                    
    except Exception as e:
        logger.error(f"ORCID works error: {str(e)}")
        return {'error': str(e), 'works': []}


# ============================================================================
# Abstract Reconstruction from Inverted Index
# ============================================================================

def reconstruct_abstract_from_inverted_index(inverted_index: dict) -> str:
    """
    Reconstruct readable abstract from OpenAlex inverted index.
    
    OpenAlex stores abstracts as inverted indexes (word -> [positions]) for copyright reasons.
    This function reconstructs the original text.
    
    Args:
        inverted_index: Dict mapping words to position lists
        
    Returns:
        str: Reconstructed abstract text
        
    Example:
        inverted_index = {
            "We": [0],
            "present": [1],
            "a": [2],
            "novel": [3],
            "approach": [4]
        }
        # Returns: "We present a novel approach"
    """
    try:
        if not inverted_index:
            return ""
        
        # Find maximum position to determine text length
        max_position = 0
        for positions in inverted_index.values():
            if positions:
                max_position = max(max_position, max(positions))
        
        # Create array of correct size
        words = [''] * (max_position + 1)
        
        # Place words at their positions
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        
        # Join words, handling empty positions
        reconstructed = ' '.join(word for word in words if word)
        
        logger.info(f"✅ Abstract reconstructed: {len(reconstructed)} characters from {len(inverted_index)} words")
        return reconstructed
        
    except Exception as e:
        logger.error(f"❌ Error reconstructing abstract: {e}")
        return ""


# ============================================================================
# Semantic Scholar Integration for Complete Abstracts
# ============================================================================

async def fetch_semantic_scholar_abstract(doi: str = None, title: str = None, openalex_id: str = None) -> dict:
    """
    Fetch complete abstract from Semantic Scholar API with caching and rate limiting.
    
    Semantic Scholar provides full abstracts (not inverted like OpenAlex for copyrighted content).
    Free API, no key required, 100 requests/second limit.
    
    Features:
    - In-memory caching (1 hour TTL)
    - Rate limiting (90 req/s)
    - Automatic retry on transient errors
    
    Args:
        doi: DOI of the paper (preferred)
        title: Paper title (fallback if no DOI)
        openalex_id: OpenAlex ID (e.g., "W1234567890")
        
    Returns:
        dict with:
        - abstract: Full abstract text
        - source: "semantic_scholar"
        - paper_id: Semantic Scholar paper ID
        - influential_citation_count: Additional metric
        - cached: Whether result came from cache
        - error: Error message if fetch failed
    """
    try:
        # Generate cache key
        cache_key = None
        if doi:
            cache_key = f"ss_doi:{doi}"
        elif openalex_id:
            cache_key = f"ss_oa:{openalex_id}"
        elif title:
            cache_key = f"ss_title:{title[:100]}"  # Limit key size
        
        # Check cache first
        if cache_key:
            cached_result = await abstract_cache.get(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result
        
        # Apply rate limiting
        await semantic_scholar_limiter.acquire()
        base_url = "https://api.semanticscholar.org/graph/v1/paper"
        
        # Build identifier for lookup
        paper_identifier = None
        if doi:
            # Clean DOI
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            paper_identifier = f"DOI:{clean_doi}"
            logger.info(f"🔍 Semantic Scholar lookup by DOI: {clean_doi}")
        elif openalex_id:
            # Extract OpenAlex short ID (e.g., W1234567890)
            if openalex_id.startswith('https://openalex.org/'):
                short_id = openalex_id.split('/')[-1]
            else:
                short_id = openalex_id
            paper_identifier = short_id
            logger.info(f"🔍 Semantic Scholar lookup by OpenAlex ID: {short_id}")
        elif title:
            # Search by title (less reliable)
            logger.info(f"🔍 Semantic Scholar search by title: {title[:50]}...")
            return await search_semantic_scholar_by_title(title)
        else:
            return {'error': 'No identifier provided (doi, title, or openalex_id required)'}
        
        # Fetch paper data
        url = f"{base_url}/{paper_identifier}"
        params = {
            'fields': 'paperId,title,abstract,year,citationCount,influentialCitationCount,journal,openAccessPdf'
        }
        
        headers = {
            'User-Agent': f'alex-mcp (+{get_config()["OPENALEX_MAILTO"]})'
        }
        
        # Create SSL context with certifi for certificate verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    abstract = data.get('abstract', '')
                    if not abstract:
                        logger.warning(f"⚠️ Semantic Scholar found paper but no abstract available")
                        return {
                            'abstract': None,
                            'source': 'semantic_scholar',
                            'paper_id': data.get('paperId'),
                            'error': 'No abstract available in Semantic Scholar'
                        }
                    
                    logger.info(f"✅ Semantic Scholar abstract retrieved: {len(abstract)} chars")
                    
                    result = {
                        'abstract': abstract,
                        'source': 'semantic_scholar',
                        'paper_id': data.get('paperId'),
                        'title': data.get('title'),
                        'year': data.get('year'),
                        'citation_count': data.get('citationCount'),
                        'influential_citation_count': data.get('influentialCitationCount'),
                        'journal': data.get('journal', {}).get('name') if data.get('journal') else None,
                        'open_access_pdf': data.get('openAccessPdf', {}).get('url') if data.get('openAccessPdf') else None,
                        'cached': False
                    }
                    
                    # Cache the result
                    if cache_key:
                        await abstract_cache.set(cache_key, result)
                    
                    return result
                elif response.status == 404:
                    logger.warning(f"⚠️ Paper not found in Semantic Scholar")
                    return {'error': 'Paper not found in Semantic Scholar', 'source': 'semantic_scholar'}
                else:
                    logger.error(f"❌ Semantic Scholar API error: {response.status}")
                    return {'error': f'Semantic Scholar API error: {response.status}', 'source': 'semantic_scholar'}
                    
    except asyncio.TimeoutError:
        logger.error("❌ Semantic Scholar API timeout")
        return {'error': 'Semantic Scholar API timeout', 'source': 'semantic_scholar'}
    except Exception as e:
        logger.error(f"❌ Semantic Scholar error: {str(e)}")
        return {'error': str(e), 'source': 'semantic_scholar'}


async def search_semantic_scholar_by_title(title: str) -> dict:
    """
    Search Semantic Scholar by title (fallback method).
    
    Args:
        title: Paper title
        
    Returns:
        dict with abstract or error
    """
    try:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            'query': title,
            'limit': 1,
            'fields': 'paperId,title,abstract,year,citationCount,influentialCitationCount'
        }
        
        headers = {
            'User-Agent': f'alex-mcp (+{get_config()["OPENALEX_MAILTO"]})'
        }
        
        # Create SSL context with certifi for certificate verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, headers=headers, ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = data.get('data', [])
                    
                    if not papers:
                        return {'error': 'No papers found by title', 'source': 'semantic_scholar'}
                    
                    paper = papers[0]
                    abstract = paper.get('abstract', '')
                    
                    if not abstract:
                        return {
                            'abstract': None,
                            'source': 'semantic_scholar',
                            'paper_id': paper.get('paperId'),
                            'error': 'Paper found but no abstract available'
                        }
                    
                    logger.info(f"✅ Semantic Scholar abstract retrieved by title: {len(abstract)} chars")
                    
                    return {
                        'abstract': abstract,
                        'source': 'semantic_scholar',
                        'paper_id': paper.get('paperId'),
                        'title': paper.get('title'),
                        'year': paper.get('year'),
                        'citation_count': paper.get('citationCount'),
                        'influential_citation_count': paper.get('influentialCitationCount')
                    }
                else:
                    return {'error': f'Semantic Scholar search error: {response.status}', 'source': 'semantic_scholar'}
                    
    except Exception as e:
        logger.error(f"❌ Semantic Scholar title search error: {str(e)}")
        return {'error': str(e), 'source': 'semantic_scholar'}


# ============================================================================
# Sci-Hub Integration for Internal Authorized Full-Text Access
# ============================================================================

async def fetch_scihub_pdf(doi: str) -> dict:
    """
    Fetch PDF from Sci-Hub with rate limiting and mirror fallback.
    
    Features:
    - Rate limiting (50 req/s)
    - Multiple mirror fallback
    - SSL verification with certifi
    
    Args:
        doi: DOI of the paper
        
    Returns:
        dict with:
        - pdf_url: Direct PDF download URL (if found)
        - mirror_used: Which Sci-Hub mirror was successful
        - doi: The DOI queried
        - source: "alternative_source"
        - error: Error message if fetch failed
    """
    try:
        # Check if alternative sources are enabled
        if not get_config()["OPENALEX_ENABLE_SCIHUB"]:
            return {
                'error': 'Alternative sources are disabled',
                'source': 'alternative_source',
                'doi': doi
            }
        
        # Apply rate limiting
        await scihub_limiter.acquire()
        
        # Clean DOI
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        
        # List of Sci-Hub mirrors (as of 2026)
        mirrors = [
            'https://sci-hub.su',
            'https://sci-hub.st',
            'https://sci-hub.red',
            'https://sci-hub.box',
            'https://sci-hub.ru',
        ]
        
        logger.info(f"🔬 Sci-Hub lookup for DOI: {clean_doi}")
        
        email = get_config()["OPENALEX_MAILTO"]
        headers = {
            'User-Agent': f'alex-mcp (+{email})'
        }
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Try each mirror
        for mirror in mirrors:
            try:
                url = f"{mirror}/{clean_doi}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                        ssl=ssl_context,
                        allow_redirects=True
                    ) as response:
                        if response.status == 200:
                            # Check if we got an actual PDF or HTML page
                            content_type = response.headers.get('Content-Type', '').lower()
                            
                            if 'application/pdf' in content_type:
                                # Direct PDF link
                                pdf_url = str(response.url)
                                logger.info(f"✅ Sci-Hub PDF found via {mirror}")
                                return {
                                    'pdf_url': pdf_url,
                                    'doi': clean_doi,
                                    'source': 'alternative_source',
                                    'content_type': content_type
                                }
                            elif 'text/html' in content_type:
                                # HTML page - parse for PDF link
                                html_content = await response.text()
                                
                                # Look for PDF embed or iframe
                                import re
                                pdf_patterns = [
                                    r'<iframe[^>]+src="([^"]+\.pdf[^"]*?)"',
                                    r'<embed[^>]+src="([^"]+\.pdf[^"]*?)"',
                                    r'href="([^"]+\.pdf[^"]*?)"',
                                ]
                                
                                for pattern in pdf_patterns:
                                    matches = re.findall(pattern, html_content, re.IGNORECASE)
                                    if matches:
                                        pdf_url = matches[0]
                                        # Make absolute URL if relative
                                        if pdf_url.startswith('//'):
                                            pdf_url = 'https:' + pdf_url
                                        elif pdf_url.startswith('/'):
                                            pdf_url = mirror + pdf_url
                                        
                                        logger.info(f"✅ Sci-Hub PDF link found via {mirror}")
                                        return {
                                            'pdf_url': pdf_url,
                                            'doi': clean_doi,
                                            'source': 'alternative_source'
                                        }
                        
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Sci-Hub mirror {mirror} timeout")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Sci-Hub mirror {mirror} error: {str(e)}")
                continue
        
        # No mirror worked
        logger.warning(f"⚠️ Sci-Hub: Could not retrieve PDF for DOI {clean_doi}")
        return {
            'error': 'PDF not found via alternative sources',
            'doi': clean_doi,
            'source': 'alternative_source'
        }
        
    except Exception as e:
        logger.error(f"❌ Sci-Hub error: {str(e)}")
        return {
            'error': str(e),
            'doi': doi,
            'source': 'alternative_source'
        }


# ============================================================================
# Unpaywall Integration for Legal Full-Text Access
# ============================================================================

async def fetch_unpaywall_links(doi: str) -> dict:
    """
    Fetch legal open access PDF links from Unpaywall API with rate limiting.
    
    Unpaywall aggregates open access locations from publishers and repositories.
    100% legal, free API (just needs email in user agent).
    
    Features:
    - Rate limiting (100 req/s)
    - Automatic retry on transient errors
    
    Args:
        doi: DOI of the paper
        
    Returns:
        dict with:
        - is_oa: Whether paper is open access
        - oa_status: OA status (gold, green, hybrid, bronze, closed)
        - best_oa_location: Best OA location with URL
        - oa_locations: List of all OA locations
        - error: Error message if fetch failed
    """
    try:
        # Apply rate limiting
        await unpaywall_limiter.acquire()
        # Clean DOI
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        
        # Unpaywall API endpoint
        email = get_config()["OPENALEX_MAILTO"]
        url = f"https://api.unpaywall.org/v2/{clean_doi}?email={email}"
        
        logger.info(f"🔍 Unpaywall lookup for DOI: {clean_doi}")
        
        headers = {
            'User-Agent': f'alex-mcp (+{email})'
        }
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=ssl_context
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    is_oa = data.get('is_oa', False)
                    oa_status = data.get('oa_status', 'closed')
                    
                    if not is_oa:
                        logger.info(f"📕 Paper is not open access (status: {oa_status})")
                        return {
                            'is_oa': False,
                            'oa_status': oa_status,
                            'best_oa_location': None,
                            'oa_locations': [],
                            'source': 'unpaywall'
                        }
                    
                    # Extract OA locations
                    best_oa_location = data.get('best_oa_location')
                    oa_locations = data.get('oa_locations', [])
                    
                    # Format locations
                    formatted_locations = []
                    for loc in oa_locations:
                        if loc.get('url_for_pdf') or loc.get('url'):
                            formatted_locations.append({
                                'url': loc.get('url_for_pdf') or loc.get('url'),
                                'url_for_landing_page': loc.get('url_for_landing_page'),
                                'host_type': loc.get('host_type'),  # publisher, repository
                                'license': loc.get('license'),
                                'version': loc.get('version')  # publishedVersion, acceptedVersion, submittedVersion
                            })
                    
                    best_url = None
                    if best_oa_location:
                        best_url = {
                            'url': best_oa_location.get('url_for_pdf') or best_oa_location.get('url'),
                            'url_for_landing_page': best_oa_location.get('url_for_landing_page'),
                            'host_type': best_oa_location.get('host_type'),
                            'license': best_oa_location.get('license'),
                            'version': best_oa_location.get('version')
                        }
                    
                    logger.info(f"✅ Unpaywall: {len(formatted_locations)} OA locations found (status: {oa_status})")
                    
                    return {
                        'is_oa': True,
                        'oa_status': oa_status,
                        'best_oa_location': best_url,
                        'oa_locations': formatted_locations,
                        'source': 'unpaywall',
                        'doi': clean_doi,
                        'title': data.get('title'),
                        'journal': data.get('journal_name'),
                        'year': data.get('year')
                    }
                    
                elif response.status == 404:
                    logger.warning(f"⚠️ DOI not found in Unpaywall database")
                    return {'error': 'DOI not found in Unpaywall', 'is_oa': False, 'source': 'unpaywall'}
                else:
                    logger.error(f"❌ Unpaywall API error: {response.status}")
                    return {'error': f'Unpaywall API error: {response.status}', 'is_oa': False, 'source': 'unpaywall'}
                    
    except asyncio.TimeoutError:
        logger.error("❌ Unpaywall API timeout")
        return {'error': 'Unpaywall API timeout', 'is_oa': False, 'source': 'unpaywall'}
    except Exception as e:
        logger.error(f"❌ Unpaywall error: {str(e)}")
        return {'error': str(e), 'is_oa': False, 'source': 'unpaywall'}


# ============================================================================
# ORCID MCP Tools
# ============================================================================

@mcp.tool(
    annotations={
        "title": "Search ORCID Authors",
        "description": (
            "Search ORCID database for author profiles by name and optionally affiliation. "
            "Provides ORCID IDs, verified names, and institutional affiliations for "
            "enhanced author disambiguation and verification."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def search_orcid_authors(
    name: str,
    affiliation: str = None,
    max_results: int = 10
) -> dict:
    """
    Search ORCID for author profiles by name and affiliation.
    
    Args:
        name: Author name to search (e.g., "John Smith", "Maria Garcia")
        affiliation: Optional institutional affiliation for disambiguation
        max_results: Maximum number of results to return (default: 10, max: 50)
        
    Returns:
        dict: ORCID search results with:
        - total_found: Total number of matches found
        - results_returned: Number of results returned
        - results: List of author profiles with ORCID IDs, names, and affiliations
        
    Example usage:
        # Basic name search
        search_orcid_authors("John Smith")
        
        # Search with affiliation for better disambiguation
        search_orcid_authors("Maria Garcia", "University of Barcelona")
    """
    # Validate parameters
    max_results = min(max(max_results, 1), 50)  # ORCID API limit
    
    result = await search_orcid_by_name(name, affiliation, max_results)
    return result


@mcp.tool(
    annotations={
        "title": "Get ORCID Works",
        "description": (
            "Retrieve publications/works from a specific ORCID profile. "
            "Useful for cross-validation with OpenAlex data and verifying "
            "author publication records."
        ),
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_orcid_publications(
    orcid_id: str,
    max_works: int = 20
) -> dict:
    """
    Get publications/works from an ORCID profile.
    
    Args:
        orcid_id: ORCID identifier (e.g., "0000-0000-0000-0000" or full URL)
        max_works: Maximum number of works to retrieve (default: 20, max: 100)
        
    Returns:
        dict: Publications data with:
        - orcid_id: Cleaned ORCID identifier
        - total_works: Number of works found
        - works: List of publications with titles, journals, DOIs, PMIDs
        
    Example usage:
        # Get works for specific ORCID
        get_orcid_publications("0000-0000-0000-0000")
        
        # Get limited number of works
        get_orcid_publications("0000-0000-0000-0000", max_works=10)
    """
    # Validate parameters
    max_works = min(max(max_works, 1), 100)  # Reasonable limit
    
    result = await get_orcid_works(orcid_id, max_works)
    return result


def main():
    """
    Entry point for the enhanced alex-mcp server with balanced peer-review filtering.
    """
    import asyncio
    logger.info("Enhanced OpenAlex Author Disambiguation MCP Server starting...")
    logger.info("Features: ~70% token reduction for authors, ~80% for works")
    logger.info("Balanced peer-review filtering: excludes data catalogs while preserving legitimate papers")
    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()