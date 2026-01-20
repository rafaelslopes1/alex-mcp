#!/usr/bin/env python3
"""
Optimized data models for the OpenAlex MCP server.

Streamlined versions focusing on essential information for author disambiguation
and work retrieval while minimizing token usage. Enhanced to preserve comprehensive
ID information (DOI, PMID, PMCID, OpenAlex, MAG).
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class WorkIDs(BaseModel):
    """
    Comprehensive work identifiers from OpenAlex.
    
    Preserves all available identifiers for cross-database linkage.
    """
    openalex: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    mag: Optional[str] = None


class OptimizedAuthorResult(BaseModel):
    """
    Streamlined author representation focusing on disambiguation essentials.
    
    Reduces token usage by ~70% compared to full OpenAlex author object.
    """
    id: str
    display_name: str
    orcid: Optional[str] = None
    display_name_alternatives: Optional[List[str]] = None
    
    # Simplified affiliations - just institution names as strings
    current_affiliations: Optional[List[str]] = None
    past_affiliations: Optional[List[str]] = None
    
    # Key metrics for research impact
    cited_by_count: int = 0
    works_count: int = 0
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    
    # Research fields (simplified)
    research_fields: Optional[List[str]] = None
    
    # Basic metadata
    last_known_institutions: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    
    # For API access
    works_api_url: Optional[str] = None


class OptimizedWorkResult(BaseModel):
    """
    Streamlined work representation focusing on essential publication info.
    
    Reduces token usage by ~80% compared to full OpenAlex work object while
    preserving comprehensive identifier information.
    
    Enhanced with abstract and full-text access fields for enriched content.
    """
    id: str
    title: Optional[str] = None
    doi: Optional[str] = None  # Kept for backward compatibility
    publication_year: Optional[int] = None
    type: Optional[str] = None  # journal-article, book-chapter, etc.
    
    # COMPREHENSIVE ID INFORMATION - This was missing!
    ids: Optional[WorkIDs] = None
    
    # Citation metrics
    cited_by_count: Optional[int] = 0
    
    # Publication venue (simplified)
    journal_name: Optional[str] = None
    journal_issn: Optional[str] = None
    publisher: Optional[str] = None
    
    # Open access info (simplified)
    is_open_access: Optional[bool] = None
    
    # Author info (minimal)
    author_count: Optional[int] = None
    first_author: Optional[str] = None
    corresponding_author: Optional[str] = None
    
    # Research categorization (simplified)
    primary_field: Optional[str] = None
    concepts: Optional[List[str]] = None
    
    # ENHANCED CONTENT - Abstract and Full-text Access
    abstract: Optional[str] = None  # Complete abstract text (from Semantic Scholar or OpenAlex)
    abstract_inverted: Optional[bool] = None  # True if OpenAlex abstract is inverted/copyrighted
    fulltext_urls: Optional[List[Dict[str, str]]] = None  # List of {url, source, license} dicts from Unpaywall


class OptimizedSearchResponse(BaseModel):
    """
    Streamlined search response.
    """
    query: str
    total_count: int
    results: List[OptimizedAuthorResult]
    search_time: Optional[datetime] = Field(default_factory=datetime.now)


class OptimizedWorksSearchResponse(BaseModel):
    """
    Streamlined works search response for author works.
    """
    author_id: str
    author_name: Optional[str] = None
    total_count: int
    results: List[OptimizedWorkResult]
    search_time: Optional[datetime] = Field(default_factory=datetime.now)
    filters: Optional[Dict[str, Any]] = None


class OptimizedGeneralWorksSearchResponse(BaseModel):
    """
    Streamlined works search response for general work searches.
    """
    query: str
    total_count: int
    results: List[OptimizedWorkResult]
    search_time: Optional[datetime] = Field(default_factory=datetime.now)
    filters: Optional[Dict[str, Any]] = None


class AutocompleteAuthorCandidate(BaseModel):
    """
    A single author candidate from autocomplete API.
    
    Optimized for fast disambiguation with essential context.
    """
    openalex_id: str
    display_name: str
    institution_hint: Optional[str] = None  # Current/last known institution
    works_count: int = 0
    cited_by_count: int = 0
    entity_type: str = "author"
    external_id: Optional[str] = None  # ORCID or other external ID


class AutocompleteAuthorsResponse(BaseModel):
    """
    Response model for author autocomplete with multiple candidates.
    
    Enables intelligent disambiguation by providing multiple options
    with institutional context and research metrics.
    """
    query: str
    context: Optional[str] = None
    total_candidates: int
    candidates: List[AutocompleteAuthorCandidate]
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


def extract_institution_names(affiliations: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """
    Extract and categorize institution names from OpenAlex affiliation objects.
    
    Returns:
        tuple: (current_affiliations, past_affiliations)
    """
    current = []
    past = []
    
    if not affiliations:
        return current, past
    
    for affiliation in affiliations:
        institution = affiliation.get('institution', {})
        if not institution:
            continue
            
        institution_name = institution.get('display_name')
        if not institution_name:
            continue
        
        # Determine if current or past based on years
        years = affiliation.get('years', [])
        if years:
            current_year = datetime.now().year
            # Consider current if active in last 3 years
            if max(years) >= current_year - 3:
                current.append(institution_name)
            else:
                past.append(institution_name)
        else:
            # Default to current if no year info
            current.append(institution_name)
    
    return current, past


def extract_research_fields(concepts_or_topics: List[Dict[str, Any]]) -> List[str]:
    """
    Extract research field names from concepts or topics.
    
    Args:
        concepts_or_topics: List of concept/topic objects from OpenAlex
        
    Returns:
        List of field names, limited to top 5 most relevant
    """
    fields = []
    
    if not concepts_or_topics:
        return fields
    
    # Sort by score/level and take top fields
    sorted_items = sorted(
        concepts_or_topics, 
        key=lambda x: x.get('score', 0) or x.get('count', 0), 
        reverse=True
    )
    
    for item in sorted_items[:5]:  # Limit to top 5
        name = item.get('display_name')
        if name:
            fields.append(name)
    
    return fields


def extract_journal_info(locations: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract journal information from OpenAlex locations.
    
    Returns:
        tuple: (journal_name, journal_issn, publisher)
    """
    if not locations:
        return None, None, None
    
    # Look for primary location (usually first) or journal location
    for location in locations:
        source = location.get('source', {})
        if source and source.get('type') == 'journal':
            journal_name = source.get('display_name')
            issn = None
            if source.get('issn'):
                issn = source['issn'][0] if isinstance(source['issn'], list) else source['issn']
            
            publisher = source.get('host_organization_name')
            return journal_name, issn, publisher
    
    # Fallback to first location
    if locations:
        source = locations[0].get('source', {})
        if source:
            return source.get('display_name'), None, source.get('host_organization_name')
    
    return None, None, None


def extract_authorship_info(authorships: List[Dict[str, Any]]) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Extract simplified authorship information.
    
    Returns:
        tuple: (author_count, first_author, corresponding_author)
    """
    if not authorships:
        return None, None, None
    
    author_count = len(authorships)
    first_author = None
    corresponding_author = None
    
    # Find first author (author_position == 'first')
    for authorship in authorships:
        if authorship.get('author_position') == 'first':
            author = authorship.get('author', {})
            first_author = author.get('display_name')
            break
    
    # Find corresponding author
    for authorship in authorships:
        if authorship.get('is_corresponding'):
            author = authorship.get('author', {})
            corresponding_author = author.get('display_name')
            break
    
    return author_count, first_author, corresponding_author


def extract_comprehensive_ids(work_data: Dict[str, Any]) -> WorkIDs:
    """
    Extract comprehensive identifier information from OpenAlex work data.
    
    This was the missing piece! OpenAlex provides comprehensive IDs in the 'ids' object.
    
    Args:
        work_data: Full OpenAlex work object
        
    Returns:
        WorkIDs object with all available identifiers
    """
    ids_data = work_data.get('ids', {})
    
    # Extract all available IDs
    openalex_id = ids_data.get('openalex') or work_data.get('id')
    doi = ids_data.get('doi') or work_data.get('doi')  # Fallback to standalone doi
    pmid = ids_data.get('pmid')
    pmcid = ids_data.get('pmcid')
    mag = ids_data.get('mag')
    
    return WorkIDs(
        openalex=openalex_id,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        mag=mag
    )


def optimize_author_data(author_data: Dict[str, Any]) -> OptimizedAuthorResult:
    """
    Convert full OpenAlex author object to optimized version.
    
    Args:
        author_data: Full OpenAlex author object
        
    Returns:
        OptimizedAuthorResult with essential information only
    """
    # Extract basic info
    author_id = author_data.get('id', '')
    display_name = author_data.get('display_name', '')
    orcid = author_data.get('orcid')
    alternatives = author_data.get('display_name_alternatives', [])
    
    # Process affiliations
    affiliations = author_data.get('affiliations', [])
    current_affiliations, past_affiliations = extract_institution_names(affiliations)
    
    # Extract metrics
    cited_by_count = author_data.get('cited_by_count', 0)
    works_count = author_data.get('works_count', 0)
    
    # Extract summary stats
    summary_stats = author_data.get('summary_stats', {})
    h_index = summary_stats.get('h_index')
    i10_index = summary_stats.get('i10_index')
    
    # Extract research fields from concepts or topics
    research_fields = []
    concepts = author_data.get('x_concepts', []) or author_data.get('topics', [])
    research_fields = extract_research_fields(concepts)
    
    # Extract geographic info
    countries = []
    if affiliations:
        for affiliation in affiliations:
            institution = affiliation.get('institution', {})
            country = institution.get('country_code')
            if country and country not in countries:
                countries.append(country)
    
    # API URL
    works_api_url = author_data.get('works_api_url')
    
    return OptimizedAuthorResult(
        id=author_id,
        display_name=display_name,
        orcid=orcid,
        display_name_alternatives=alternatives[:3] if alternatives else None,  # Limit alternatives
        current_affiliations=current_affiliations[:3] if current_affiliations else None,  # Limit to 3 most recent
        past_affiliations=past_affiliations[:3] if past_affiliations else None,  # Limit to 3 most recent
        cited_by_count=cited_by_count,
        works_count=works_count,
        h_index=h_index,
        i10_index=i10_index,
        research_fields=research_fields[:5] if research_fields else None,  # Top 5 fields
        last_known_institutions=current_affiliations[:2] if current_affiliations else past_affiliations[:2],
        countries=countries[:3] if countries else None,  # Limit countries
        works_api_url=works_api_url
    )


def optimize_work_data(work_data: Dict[str, Any]) -> OptimizedWorkResult:
    """
    Convert full OpenAlex work object to optimized version.
    
    NOW INCLUDES COMPREHENSIVE ID EXTRACTION!
    
    Args:
        work_data: Full OpenAlex work object
        
    Returns:
        OptimizedWorkResult with essential information AND comprehensive IDs
    """
    # Basic work info
    work_id = work_data.get('id', '')
    title = work_data.get('title')
    doi = work_data.get('doi')  # Kept for backward compatibility
    publication_year = work_data.get('publication_year')
    work_type = work_data.get('type')
    
    # EXTRACT COMPREHENSIVE IDS - This is the fix!
    comprehensive_ids = extract_comprehensive_ids(work_data)
    
    # Citation metrics
    cited_by_count = work_data.get('cited_by_count', 0)
    
    # Journal information
    locations = work_data.get('locations', [])
    journal_name, journal_issn, publisher = extract_journal_info(locations)
    
    # Open access info
    open_access = work_data.get('open_access', {})
    is_open_access = open_access.get('is_oa') if open_access else None
    
    # Authorship info
    authorships = work_data.get('authorships', [])
    author_count, first_author, corresponding_author = extract_authorship_info(authorships)
    
    # Research categorization
    primary_topic = work_data.get('primary_topic', {})
    primary_field = primary_topic.get('display_name') if primary_topic else None
    
    # Simplified concepts (top 3)
    concepts = work_data.get('concepts', [])
    concept_names = []
    if concepts:
        sorted_concepts = sorted(concepts, key=lambda x: x.get('score', 0), reverse=True)
        concept_names = [c.get('display_name') for c in sorted_concepts[:3] if c.get('display_name')]
    
    return OptimizedWorkResult(
        id=work_id,
        title=title,
        doi=doi,  
        publication_year=publication_year,
        type=work_type,
        ids=comprehensive_ids,  
        cited_by_count=cited_by_count,
        journal_name=journal_name,
        journal_issn=journal_issn,
        publisher=publisher,
        is_open_access=is_open_access,
        author_count=author_count,
        first_author=first_author,
        corresponding_author=corresponding_author,
        primary_field=primary_field,
        concepts=concept_names if concept_names else None
    )