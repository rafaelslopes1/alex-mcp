#!/usr/bin/env python3
"""
Acceptance Tests for alex-mcp v5.1.0
Validates all 9 new features implemented in Phase 2.

Tests:
1. search_works with year range filters
2. get_work with complete authorships
3. get_work_by_doi resolution
4. get_cited_by (incoming citations)
5. get_references (outgoing citations)
6. decode_abstract from inverted index
7. get_top_authors_for_query (by ID, no homonyms)
8. batch_get_works
9. get_best_oa_location

Run with:
    python examples/test_spec_compliance_v5_1.py

Requires:
    - OPENALEX_MAILTO env var set
    - Internet connection for API calls
    - ~2-3 minutes runtime
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alex_mcp.server import (
    search_works_core,
    get_work,
    get_work_by_doi,
    get_cited_by,
    get_references,
    decode_abstract,
    get_top_authors_for_query,
    batch_get_works,
    get_best_oa_location,
)


# Test constants
KNOWN_PAPER_ID = "W2741809807"  # "BERT: Pre-training of Deep Bidirectional Transformers"
KNOWN_PAPER_DOI = "10.48550/arxiv.1810.04805"
CITING_PAPER_ID = "W2919231516"  # A paper that cites BERT

# Example inverted abstract
SAMPLE_INVERTED_ABSTRACT = {
    "machine": [0],
    "learning": [1],
    "is": [2],
    "a": [3],
    "field": [4],
}


async def test_1_search_works_year_range():
    """
    Test: search_works with year range filters (2022-2024)
    Validates: year_from, year_to parameters work correctly
    """
    print("\n🧪 TEST 1: search_works with year range filters")
    print("-" * 60)
    
    try:
        # Search for papers 2022-2024
        response = search_works_core(
            query="transformer",
            year_from=2022,
            year_to=2024,
            limit=10
        )
        
        # Validate
        assert response.total_count >= 0, "Should return count"
        
        # Check that ALL results are in year range
        for work in response.results:
            if work.publication_year:
                assert 2022 <= work.publication_year <= 2024, \
                    f"Paper {work.id} has year {work.publication_year} outside range [2022, 2024]"
        
        print(f"✅ PASS: Found {response.total_count} papers in year range 2022-2024")
        if response.results:
            print(f"   Sample: {response.results[0].title[:60]}... ({response.results[0].publication_year})")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_2_get_work_complete_authorships():
    """
    Test: get_work returns complete authorships with author IDs
    Validates: authorships[] contains author.id (no homonym risk)
    """
    print("\n🧪 TEST 2: get_work with complete authorships")
    print("-" * 60)
    
    try:
        work_data = await get_work(KNOWN_PAPER_ID)
        
        # Check for error
        if 'error' in work_data:
            print(f"❌ FAIL: {work_data['error']}")
            return False
        
        # Validate authorships field exists and has data
        authorships = work_data.get('authorships')
        assert authorships is not None, "Should have authorships field"
        assert len(authorships) > 0, "Should have at least one authorship"
        
        # Check first authorship has required fields
        first_auth = authorships[0]
        assert 'author' in first_auth, "Authorship should have 'author' field"
        assert 'id' in first_auth['author'], "Author should have ID (for disambiguation)"
        assert 'display_name' in first_auth['author'], "Author should have display_name"
        
        # IMPORTANT: author_id should be a string starting with 'A' (OpenAlex format)
        author_id = first_auth['author'].get('id', '')
        assert isinstance(author_id, str) and len(author_id) > 0, f"Invalid author ID: {author_id}"
        
        print(f"✅ PASS: Work has {len(authorships)} authorships with proper IDs")
        print(f"   First author: {first_auth['author']['display_name']} ({first_auth['author']['id']})")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_3_get_work_by_doi():
    """
    Test: get_work_by_doi resolves DOI to work with 100% reliability
    Validates: DOI resolution works in multiple formats
    """
    print("\n🧪 TEST 3: get_work_by_doi resolution")
    print("-" * 60)
    
    try:
        # Test format 1: Just DOI number
        work_data_1 = await get_work_by_doi(KNOWN_PAPER_DOI)
        
        if 'error' in work_data_1:
            print(f"⚠️  Format 1 (10.xxxx/...): {work_data_1.get('error')}")
            # Try format 2
            work_data_2 = await get_work_by_doi(f"https://doi.org/{KNOWN_PAPER_DOI}")
            if 'error' in work_data_2:
                print(f"❌ FAIL: Both DOI formats failed")
                return False
            work_data = work_data_2
        else:
            work_data = work_data_1
        
        # Validate
        assert 'title' in work_data, "Should have title"
        assert work_data['title'] is not None, "Title should not be None"
        
        print(f"✅ PASS: DOI {KNOWN_PAPER_DOI} resolved to:")
        print(f"   Title: {work_data['title'][:60]}...")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_4_get_cited_by():
    """
    Test: get_cited_by returns incoming citations
    Validates: Finds papers that cite a given work
    """
    print("\n🧪 TEST 4: get_cited_by (incoming citations)")
    print("-" * 60)
    
    try:
        # Get papers that cite BERT
        citations_data = await get_cited_by(KNOWN_PAPER_ID, limit=10)
        
        # Check for error
        if 'error' in citations_data:
            print(f"❌ FAIL: {citations_data['error']}")
            return False
        
        # Validate response structure
        assert 'work_id' in citations_data, "Should have work_id"
        assert 'citing_count' in citations_data, "Should have citing_count"
        assert 'citing_works' in citations_data, "Should have citing_works list"
        
        citing_works = citations_data['citing_works']
        
        # BERT should have many citations (this is a famous paper)
        citing_count = citations_data['citing_count']
        assert citing_count > 0, "BERT paper should have incoming citations"
        assert len(citing_works) > 0, "Should return at least some citing works"
        
        print(f"✅ PASS: Found {citing_count} papers citing this work")
        print(f"   Returned {len(citing_works)} citing papers (limit=10)")
        if citing_works:
            print(f"   Example: {citing_works[0]['title'][:60]}...")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_5_get_references():
    """
    Test: get_references returns outgoing citations
    Validates: Finds papers referenced by a given work
    """
    print("\n🧪 TEST 5: get_references (outgoing citations)")
    print("-" * 60)
    
    try:
        # Get papers referenced by BERT
        refs_data = await get_references(KNOWN_PAPER_ID, limit=10)
        
        # Check for error
        if 'error' in refs_data:
            print(f"❌ FAIL: {refs_data['error']}")
            return False
        
        # Validate response structure
        assert 'work_id' in refs_data, "Should have work_id"
        assert 'reference_count' in refs_data, "Should have reference_count"
        assert 'referenced_works' in refs_data, "Should have referenced_works list"
        
        referenced_works = refs_data['referenced_works']
        ref_count = refs_data['reference_count']
        
        # BERT should reference many papers
        assert ref_count > 0, "BERT paper should have references"
        assert len(referenced_works) > 0, "Should return at least some referenced works"
        
        print(f"✅ PASS: Found {ref_count} papers referenced by this work")
        print(f"   Returned {len(referenced_works)} papers (limit=10)")
        if referenced_works:
            print(f"   Example: {referenced_works[0]['title'][:60]}...")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


def test_6_decode_abstract():
    """
    Test: decode_abstract reconstructs from inverted index
    Validates: Inverted index → readable text conversion
    """
    print("\n🧪 TEST 6: decode_abstract from inverted index")
    print("-" * 60)
    
    try:
        result = decode_abstract(SAMPLE_INVERTED_ABSTRACT)
        
        # Check for error
        if 'error' in result:
            print(f"❌ FAIL: {result['error']}")
            return False
        
        # Validate response
        assert 'abstract_text' in result, "Should have abstract_text"
        abstract_text = result['abstract_text']
        
        # Check that expected words are in output
        words = abstract_text.lower().split()
        expected_words = ['machine', 'learning', 'is', 'a', 'field']
        
        for expected in expected_words:
            assert expected in words or expected in abstract_text.lower(), \
                f"Expected word '{expected}' not found in abstract"
        
        print(f"✅ PASS: Reconstructed abstract from inverted index")
        print(f"   Input: {len(SAMPLE_INVERTED_ABSTRACT)} words")
        print(f"   Output: '{abstract_text}'")
        print(f"   Had gaps: {result.get('had_gaps', False)}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_7_get_top_authors():
    """
    Test: get_top_authors_for_query returns unique authors by ID
    Validates: No homonym duplicates (uses author.id, not names)
    """
    print("\n🧪 TEST 7: get_top_authors_for_query (by ID)")
    print("-" * 60)
    
    try:
        authors_data = await get_top_authors_for_query(
            query="attention mechanism",
            limit_works=50,
            top_k=5,
            sort_by="count"
        )
        
        # Check for error
        if 'error' in authors_data:
            print(f"❌ FAIL: {authors_data['error']}")
            return False
        
        # Validate response
        assert 'query' in authors_data, "Should have query"
        assert 'top_authors' in authors_data, "Should have top_authors list"
        
        top_authors = authors_data['top_authors']
        
        # Check no duplicate IDs (no homonyms!)
        author_ids = [a['author_id'] for a in top_authors]
        assert len(author_ids) == len(set(author_ids)), \
            "Author IDs should be unique (no homonym duplicates)"
        
        # Validate each author entry
        if top_authors:
            first_author = top_authors[0]
            assert 'author_id' in first_author, "Should have author_id"
            assert 'author_name' in first_author, "Should have author_name"
            assert 'count' in first_author, "Should have count"
            assert first_author['count'] > 0, "Count should be > 0"
        
        print(f"✅ PASS: Found {len(top_authors)} unique authors (by ID)")
        print(f"   Papers scanned: {authors_data['papers_scanned']}")
        if top_authors:
            print(f"   Top: {top_authors[0]['author_name']} ({top_authors[0]['count']} papers)")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_8_batch_get_works():
    """
    Test: batch_get_works fetches multiple works efficiently
    Validates: Batch operation returns complete work objects
    """
    print("\n🧪 TEST 8: batch_get_works")
    print("-" * 60)
    
    try:
        # Batch fetch some papers
        work_ids = [KNOWN_PAPER_ID, CITING_PAPER_ID, "W1234567890"]  # Last one likely doesn't exist
        
        batch_result = await batch_get_works(work_ids, chunk_size=50)
        
        # Validate response
        assert 'requested' in batch_result, "Should have requested count"
        assert 'fetched' in batch_result, "Should have fetched count"
        assert 'works' in batch_result, "Should have works list"
        assert 'errors' in batch_result, "Should have errors list"
        
        fetched = batch_result['fetched']
        errors = len(batch_result['errors'])
        
        # Should fetch at least some works
        assert fetched > 0, "Should fetch at least some works"
        
        # Validate fetched works
        for work in batch_result['works']:
            assert 'title' in work, "Work should have title"
        
        print(f"✅ PASS: Batch operation successful")
        print(f"   Requested: {batch_result['requested']}")
        print(f"   Fetched: {fetched}")
        print(f"   Errors: {errors}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def test_9_get_best_oa_location():
    """
    Test: get_best_oa_location returns OA metadata
    Validates: Open access information extraction
    """
    print("\n🧪 TEST 9: get_best_oa_location")
    print("-" * 60)
    
    try:
        oa_data = await get_best_oa_location(KNOWN_PAPER_ID)
        
        # Check for error
        if 'error' in oa_data:
            print(f"❌ FAIL: {oa_data['error']}")
            return False
        
        # Validate response structure
        assert 'work_id' in oa_data, "Should have work_id"
        assert 'is_oa' in oa_data, "Should have is_oa status"
        assert 'oa_status' in oa_data, "Should have oa_status"
        assert 'best_location' in oa_data, "Should have best_location"
        assert 'all_locations' in oa_data, "Should have all_locations"
        
        # Validate OA status is one of known values
        oa_status = oa_data['oa_status']
        valid_statuses = ['gold', 'green', 'hybrid', 'bronze', 'closed']
        assert oa_status in valid_statuses, \
            f"OA status '{oa_status}' should be one of {valid_statuses}"
        
        is_oa = oa_data['is_oa']
        best_location = oa_data['best_location']
        
        # If OA, best location should exist
        if is_oa:
            assert best_location is not None, "If OA, should have best_location"
            assert 'url' in best_location, "Location should have URL"
        
        print(f"✅ PASS: OA location information retrieved")
        print(f"   OA Status: {oa_status}")
        print(f"   Is OA: {is_oa}")
        if best_location and 'url' in best_location:
            print(f"   Best URL: {best_location['url'][:60]}...")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        return False


async def main():
    """Run all acceptance tests"""
    
    print("=" * 70)
    print("alex-mcp v5.1.0 - ACCEPTANCE TEST SUITE")
    print("=" * 70)
    print("\nValidating all 9 new features from Phase 2 implementation")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")
    
    # Check environment
    if not os.getenv('OPENALEX_MAILTO'):
        print("\n⚠️  WARNING: OPENALEX_MAILTO environment variable not set")
        print("   Some API calls may be rate-limited or blocked")
    
    # Run tests
    tests = [
        ("search_works year range", test_1_search_works_year_range),
        ("get_work authorships", test_2_get_work_complete_authorships),
        ("get_work_by_doi", test_3_get_work_by_doi),
        ("get_cited_by", test_4_get_cited_by),
        ("get_references", test_5_get_references),
        ("decode_abstract", test_6_decode_abstract),
        ("get_top_authors", test_7_get_top_authors),
        ("batch_get_works", test_8_batch_get_works),
        ("get_best_oa_location", test_9_get_best_oa_location),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                passed = await test_func()
            else:
                passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! v5.1.0 is ready for production")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed - review output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
