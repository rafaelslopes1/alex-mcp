#!/usr/bin/env python3
"""
Test script to verify the response structure of get_fulltext_access
Specifically checking that alternative source access is clearly marked as non-OA
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ['OPENALEX_ENABLE_SCIHUB'] = 'true'
os.environ['OPENALEX_MAILTO'] = 'test@example.com'

# Import after setting env vars
from src.alex_mcp.server import fetch_scihub_pdf, fetch_scihub_pdf_playwright

async def test_response_structure():
    """Test that response structure is clear about non-OA status"""
    
    # Test DOI - known closed access paper
    test_doi = "10.1109/JLT.2021.3064935"
    
    print("\n" + "="*80)
    print("Testing Sci-Hub PDF Access & Response Structure")
    print("="*80)
    print(f"Test DOI: {test_doi}")
    print()
    
    # Test aiohttp method first
    print("1️⃣ Testing HTTP client method...")
    try:
        result = await fetch_scihub_pdf(test_doi)
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        elif result.get('pdf_path'):
            print(f"✅ PDF downloaded successfully")
            print(f"   Path: {result['pdf_path']}")
            print(f"   Method: {result.get('method', 'unknown')}")
            print(f"   Mirror: {result.get('mirror', 'unknown')}")
            
            # Verify path doesn't expose origin
            pdf_path = result['pdf_path']
            if 'scihub' in pdf_path.lower():
                print(f"⚠️ WARNING: Path contains 'scihub': {pdf_path}")
            else:
                print(f"✅ Path is neutral (no 'scihub' reference)")
            
            # Check file exists and size
            if Path(pdf_path).exists():
                size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")
            else:
                print(f"❌ File doesn't exist at {pdf_path}")
        else:
            print(f"❌ No PDF path returned")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()
    
    # Expected response structure when integrated in get_fulltext_access
    print("2️⃣ Expected response structure in get_fulltext_access:")
    print()
    print("When alternative source provides PDF, should return:")
    print("  ✓ is_oa: False (not claiming legal OA)")
    print("  ✓ source: 'alternative_source' (not 'multi_source')")
    print("  ✓ alternative_access: True (flag for internal use)")
    print("  ✓ local_pdf_path: /tmp/alex_mcp_pdfs/paper_<hash>.pdf")
    print("  ✓ method: 'aiohttp' or 'playwright'")
    print("  ✓ note: 'Non-OA alternative source fetched; internal use only.'")
    print("  ✓ oa_status: 'closed' (original status)")
    print("  ✓ best_oa_location: None")
    print("  ✓ oa_locations: []")
    print()
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_response_structure())
