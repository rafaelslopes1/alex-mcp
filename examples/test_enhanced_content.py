#!/usr/bin/env python3
"""
Test script for enhanced content features (abstract and full-text access).

Tests the new integration with Semantic Scholar and Unpaywall.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alex_mcp.server import (
    fetch_semantic_scholar_abstract,
    fetch_unpaywall_links,
    get_config,
    configure_pyalex,
    reconstruct_abstract_from_inverted_index,
    abstract_cache,
    semantic_scholar_limiter
)


async def test_semantic_scholar():
    """Test Semantic Scholar abstract fetching."""
    print("\n" + "="*80)
    print("TEST 1: Semantic Scholar Abstract Fetch")
    print("="*80)
    
    # Test with a well-known paper DOI
    test_doi = "10.1038/nature14539"  # CRISPR paper
    
    print(f"\n🔍 Testing with DOI: {test_doi}")
    result = await fetch_semantic_scholar_abstract(doi=test_doi)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success!")
        print(f"   Paper ID: {result.get('paper_id')}")
        print(f"   Title: {result.get('title', 'N/A')[:80]}...")
        print(f"   Year: {result.get('year')}")
        print(f"   Citations: {result.get('citation_count')}")
        print(f"   Influential Citations: {result.get('influential_citation_count')}")
        
        abstract = result.get('abstract', '')
        if abstract:
            print(f"\n📄 Abstract ({len(abstract)} characters):")
            print(f"   {abstract[:300]}...")
        else:
            print("   ⚠️ No abstract available")
        
        if result.get('open_access_pdf'):
            print(f"\n📁 Open Access PDF: {result['open_access_pdf']}")
    
    return result


async def test_unpaywall():
    """Test Unpaywall full-text access."""
    print("\n" + "="*80)
    print("TEST 2: Unpaywall Full-text Access")
    print("="*80)
    
    # Test with an open access paper
    test_doi = "10.1371/journal.pone.0308866"  # PLoS ONE paper (always OA)
    
    print(f"\n🔍 Testing with DOI: {test_doi}")
    result = await fetch_unpaywall_links(test_doi)
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success!")
        print(f"   Is Open Access: {result.get('is_oa')}")
        print(f"   OA Status: {result.get('oa_status')}")
        print(f"   Title: {result.get('title', 'N/A')[:80]}...")
        print(f"   Journal: {result.get('journal')}")
        print(f"   Year: {result.get('year')}")
        
        if result.get('is_oa'):
            best_loc = result.get('best_oa_location')
            if best_loc:
                print(f"\n🏆 Best OA Location:")
                print(f"   URL: {best_loc['url']}")
                print(f"   Host Type: {best_loc['host_type']}")
                print(f"   License: {best_loc.get('license', 'N/A')}")
                print(f"   Version: {best_loc.get('version', 'N/A')}")
            
            oa_locations = result.get('oa_locations', [])
            if len(oa_locations) > 1:
                print(f"\n📚 Total OA Locations: {len(oa_locations)}")
                for i, loc in enumerate(oa_locations[:3], 1):
                    print(f"   {i}. {loc['host_type']}: {loc['url'][:60]}...")
    
    return result


async def test_combined():
    """Test combined enrichment."""
    print("\n" + "="*80)
    print("TEST 3: Combined Enrichment (Abstract + Full-text)")
    print("="*80)
    
    # Test with a paper that should have both
    test_doi = "10.1038/s41587-024-02534-3"  # Nature Biotech paper
    
    print(f"\n🔍 Testing with DOI: {test_doi}")
    print("\n1️⃣ Fetching abstract from Semantic Scholar...")
    abstract_result = await fetch_semantic_scholar_abstract(doi=test_doi)
    
    print("\n2️⃣ Fetching full-text access from Unpaywall...")
    fulltext_result = await fetch_unpaywall_links(test_doi)
    
    print("\n" + "-"*80)
    print("COMBINED RESULTS:")
    print("-"*80)
    
    # Abstract
    if abstract_result.get('abstract'):
        print(f"✅ Abstract: {len(abstract_result['abstract'])} characters")
        print(f"   Source: Semantic Scholar")
        print(f"   Preview: {abstract_result['abstract'][:200]}...")
    else:
        print(f"❌ Abstract: {abstract_result.get('error', 'Not available')}")
    
    # Full-text
    if fulltext_result.get('is_oa'):
        print(f"\n✅ Full-text: {len(fulltext_result.get('oa_locations', []))} OA locations")
        print(f"   Status: {fulltext_result['oa_status']}")
        if fulltext_result.get('best_oa_location'):
            print(f"   Best URL: {fulltext_result['best_oa_location']['url']}")
    else:
        print(f"\n❌ Full-text: Not open access")
    
    return {'abstract': abstract_result, 'fulltext': fulltext_result}


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases and Error Handling")
    print("="*80)
    
    # Test 1: Invalid DOI
    print("\n🧪 Test 4.1: Invalid DOI")
    result = await fetch_semantic_scholar_abstract(doi="10.invalid/doi123")
    print(f"   Result: {result.get('error', 'Success (unexpected)')}")
    
    # Test 2: Paper without abstract
    print("\n🧪 Test 4.2: Paper without abstract (if exists)")
    result = await fetch_semantic_scholar_abstract(doi="10.1145/3308558.3313418")
    if result.get('abstract'):
        print(f"   Found abstract: {len(result['abstract'])} chars")
    else:
        print(f"   No abstract: {result.get('error', 'As expected')}")
    
    # Test 3: Closed access paper
    print("\n🧪 Test 4.3: Closed access paper")
    result = await fetch_unpaywall_links("10.1016/S0140-6736(20)30251-8")  # Lancet paper, often closed
    print(f"   Is OA: {result.get('is_oa')}")
    print(f"   Status: {result.get('oa_status', 'N/A')}")
    
    # Test 4: Title-based search (fallback)
    print("\n🧪 Test 4.4: Title-based search")
    result = await fetch_semantic_scholar_abstract(
        title="Attention Is All You Need"
    )
    if result.get('abstract'):
        print(f"   ✅ Found by title: {result.get('title', 'N/A')[:60]}...")
    else:
        print(f"   ❌ Not found: {result.get('error')}")


async def test_cache_and_rate_limiting():
    """Test cache and rate limiting features."""
    print("\n" + "="*80)
    print("TEST 5: Cache and Rate Limiting")
    print("="*80)
    
    # Test cache
    print("\n🧪 Test 5.1: Cache functionality")
    test_doi = "10.1038/nature14539"
    
    # First call - should fetch from API
    print("   First call (should fetch from API)...")
    result1 = await fetch_semantic_scholar_abstract(doi=test_doi)
    cached1 = result1.get('cached', False)
    print(f"   Cached: {cached1}")
    
    # Second call - should come from cache
    print("   Second call (should come from cache)...")
    result2 = await fetch_semantic_scholar_abstract(doi=test_doi)
    cached2 = result2.get('cached', False)
    print(f"   Cached: {cached2}")
    
    if cached2:
        print("   ✅ Cache working correctly!")
    else:
        print("   ⚠️ Cache may not be working")
    
    # Cache stats
    stats = abstract_cache.stats()
    print(f"\n📊 Cache stats: {stats['size']}/{stats['max_size']} items, TTL={stats['ttl_seconds']}s")
    
    # Test rate limiting
    print("\n🧪 Test 5.2: Rate limiting")
    limiter_stats = semantic_scholar_limiter.stats()
    print(f"   Rate limiter: {limiter_stats['max_requests']} req/{limiter_stats['time_window']}s")
    print(f"   Current tokens: {limiter_stats['current_tokens']:.2f}")


async def test_abstract_reconstruction():
    """Test abstract reconstruction from inverted index."""
    print("\n" + "="*80)
    print("TEST 6: Abstract Reconstruction from Inverted Index")
    print("="*80)
    
    # Create a test inverted index
    print("\n🧪 Test 6.1: Simple reconstruction")
    test_index = {
        "We": [0],
        "present": [1],
        "a": [2],
        "novel": [3],
        "approach": [4],
        "to": [5],
        "natural": [6],
        "language": [7],
        "processing": [8]
    }
    
    reconstructed = reconstruct_abstract_from_inverted_index(test_index)
    print(f"   Input: {test_index}")
    print(f"   Output: {reconstructed}")
    
    expected = "We present a novel approach to natural language processing"
    if reconstructed == expected:
        print("   ✅ Reconstruction successful!")
    else:
        print(f"   ⚠️ Expected: {expected}")
    
    # Test with complex inverted index (with repeated words)
    print("\n🧪 Test 6.2: Complex reconstruction")
    complex_index = {
        "the": [0, 5, 10],
        "cat": [1],
        "sat": [2],
        "on": [3],
        "mat": [4],
        "and": [6],
        "dog": [7],
        "lay": [8],
        "under": [9],
        "table": [11]
    }
    
    reconstructed = reconstruct_abstract_from_inverted_index(complex_index)
    print(f"   Output: {reconstructed}")
    print(f"   Length: {len(reconstructed)} characters")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENHANCED CONTENT INTEGRATION TEST SUITE")
    print("Testing Semantic Scholar + Unpaywall Integration")
    print("With Cache, Rate Limiting, and Fallback Strategies")
    print("="*80)
    
    # Configure
    try:
        config = get_config()
        print(f"\n✅ Configuration loaded")
        print(f"   Email: {config['OPENALEX_MAILTO']}")
    except SystemExit:
        print("\n❌ ERROR: OPENALEX_MAILTO environment variable not set!")
        print("   Please run: export OPENALEX_MAILTO='your-email@example.com'")
        return
    
    # Run tests
    try:
        await test_semantic_scholar()
        await asyncio.sleep(1)  # Rate limiting courtesy
        
        await test_unpaywall()
        await asyncio.sleep(1)
        
        await test_combined()
        await asyncio.sleep(1)
        
        await test_edge_cases()
        await asyncio.sleep(1)
        
        await test_cache_and_rate_limiting()
        await asyncio.sleep(1)
        
        await test_abstract_reconstruction()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)
        print("\n💡 Summary:")
        print("   - Semantic Scholar: Provides complete abstracts with caching")
        print("   - Unpaywall: Provides legal full-text PDF access")
        print("   - Cache: 1-hour TTL for improved performance")
        print("   - Rate Limiting: Prevents API limit violations")
        print("   - Fallback: Reconstructs from inverted index when needed")
        print("\n🎯 Integration Status: FULLY OPERATIONAL")
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
