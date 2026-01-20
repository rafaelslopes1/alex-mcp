<div align="center">
  <img src="img/oam_logo_rectangular.png" alt="OpenAlex MCP Server" width="600"/>
  
  # OpenAlex Author Disambiguation MCP Server

  [![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-Compatible-blue)](https://modelcontextprotocol.io/)
  [![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
  [![OpenAlex](https://img.shields.io/badge/OpenAlex-API-orange)](https://openalex.org)
  [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
  [![Optimized](https://img.shields.io/badge/AI%20Agent-Optimized-brightgreen)](https://github.com/drAbreu/alex-mcp)
</div>

A **streamlined** Model Context Protocol (MCP) server for author disambiguation and academic research using the OpenAlex.org API. Specifically designed for AI agents with optimized data structures and enhanced functionality.

---

## 🎯 Key Features

### 🔍 **Core Capabilities**
- **Advanced Author Disambiguation**: Handles complex career transitions and name variations
- **Institution Resolution**: Current and past affiliations with transition tracking
- **Academic Work Retrieval**: Journal articles, letters, and research papers
- **Citation Analysis**: H-index, citation counts, and impact metrics
- **ORCID Integration**: Highest accuracy matching with ORCID identifiers

### 🚀 **AI Agent Optimized**
- **Streamlined Data**: Focused on essential information for disambiguation
- **Fast Processing**: Optimized data structures for rapid analysis
- **Smart Filtering**: Enhanced filtering options for targeted queries
- **Clean Output**: Structured responses optimized for AI reasoning

### 🤖 **Agent Integration**
- **Multiple Candidates**: Ranked results for automated decision-making
- **Structured Responses**: Clean, parseable output optimized for LLMs
- **Error Handling**: Graceful degradation with informative messages
- **Enhanced Filtering**: Journal-only, citation thresholds, and temporal filters

### 🏛️ **Professional Grade**
- **MCP Best Practices**: Built with FastMCP following official guidelines
- **Tool Annotations**: Proper MCP tool annotations for optimal client integration
- **Resource Management**: Efficient HTTP client management and cleanup
- **Rate Limiting**: Respectful API usage with proper delays

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- MCP-compatible client (e.g., Claude Desktop)
- Email address (for OpenAlex API courtesy)

### Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/drAbreu/alex-mcp.git
   cd alex-mcp
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Configure environment:**
   ```bash
   export OPENALEX_MAILTO=your-email@domain.com
   ```

5. **Run the server:**
   ```bash
   ./run_alex_mcp.sh
   # Or, if installed as a CLI tool:
   alex-mcp
   ```

---

## ⚙️ MCP Configuration

### Claude Desktop Configuration

Add to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "alex-mcp": {
      "command": "/path/to/alex-mcp/run_alex_mcp.sh",
      "env": {
        "OPENALEX_MAILTO": "your-email@domain.com"
      }
    }
  }
}
```

Replace `/path/to/alex-mcp` with the actual path to the repository on your system.

---

## 🤖 Using with AI Agents

### OpenAI Agents Integration

You can load this MCP server in your OpenAI agent workflow using the [`agents.mcp.MCPServerStdio`](https://github.com/openai/openai-agents) interface:

```python
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="OpenAlex MCP For Author disambiguation and works",
    cache_tools_list=True,
    params={
        "command": "uvx",
        "args": [
            "--from", "git+https://github.com/drAbreu/alex-mcp.git@4.1.0",
            "alex-mcp"
        ],
        "env": {
            "OPENALEX_MAILTO": "your-email@domain.com"
        }
    },
    client_session_timeout_seconds=10
) as alex_mcp:
    await alex_mcp.connect()
    tools = await alex_mcp.list_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
```

### Academic Research Agent Integration

This MCP server is specifically optimized for academic research workflows:

```python
# Optimized for academic research workflows
from alex_agent import run_author_research

# Enhanced functionality with streamlined data
result = await run_author_research(
    "Find J. Abreu at EMBO with recent publications"
)

# Clean, structured output for AI processing
print(f"Success: {result['workflow_metadata']['success']}")
print(f"Quality: {result['research_result']['metadata']['result_analysis']['quality_score']}/100")
```

### Direct Launch with uvx

```bash
# Standard launch
uvx --from git+https://github.com/drAbreu/alex-mcp.git@4.1.0 alex-mcp

# With environment variables
OPENALEX_MAILTO=your-email@domain.com uvx --from git+https://github.com/drAbreu/alex-mcp.git@4.1.0 alex-mcp
```

---

## 🛠️ Available Tools

### 1. **autocomplete_authors** ⭐ NEW
Get multiple author candidates using OpenAlex autocomplete API for intelligent disambiguation.

**Parameters:**
- `name` (required): Author name to search (e.g., "James Briscoe", "M. Ralser")
- `context` (optional): Context for disambiguation (e.g., "Francis Crick Institute developmental biology")
- `limit` (optional): Maximum candidates (1-10, default: 5)

**Key Features:**
- ⚡ **Fast**: ~200ms response time
- 🎯 **Smart**: Multiple candidates with institutional hints
- 🧠 **AI-Ready**: Perfect for context-based selection
- 📊 **Rich**: Works count, citations, institution info

**Streamlined Output:**
```json
{
  "query": "James Briscoe",
  "context": "Francis Crick Institute",
  "total_candidates": 3,
  "candidates": [
    {
      "openalex_id": "https://openalex.org/A5019391436",
      "display_name": "James Briscoe",
      "institution_hint": "The Francis Crick Institute, UK",
      "works_count": 415,
      "cited_by_count": 24623,
      "external_id": "https://orcid.org/0000-0002-1020-5240"
    }
  ]
}
```

**Usage Pattern:**
```python
# Get multiple candidates for disambiguation
candidates = await autocomplete_authors(
    "James Briscoe", 
    context="Francis Crick Institute developmental biology"
)

# AI selects best match based on institutional context
# Much more accurate than single search result!
```

### 2. **get_work_abstract** 🆕 Enhanced Content
Fetch complete, readable abstract from Semantic Scholar (non-inverted).

**Parameters:**
- `doi` (optional): DOI of the paper (preferred)
- `title` (optional): Paper title (fallback)
- `openalex_id` (optional): OpenAlex work ID

**Key Features:**
- 📄 **Complete Abstracts**: Full text, not alphabetically sorted
- 🎯 **AI-Powered Metrics**: Influential citation counts
- 📁 **PDF Access**: Open access PDF links when available
- 🆓 **Free API**: No authentication required

**Output:**
```json
{
  "abstract": "We present a novel approach to...",
  "source": "semantic_scholar",
  "paper_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
  "citation_count": 1234,
  "influential_citation_count": 89,
  "open_access_pdf": "https://arxiv.org/pdf/1706.03762.pdf"
}
```

**Usage:**
```python
# Get abstract by DOI
abstract_data = await get_work_abstract(
    doi="10.1038/s41587-024-02534-3"
)

# Fallback to title search
abstract_data = await get_work_abstract(
    title="Attention Is All You Need"
)
```

### 3. **get_fulltext_access** 🆕 Enhanced Content
Get legal open access PDF links via Unpaywall.

**Parameters:**
- `doi` (required): DOI of the paper

**Key Features:**
- 📕 **100% Legal**: Only legitimate open access content
- 🏆 **Multiple Sources**: Publisher + repository versions
- ⚖️ **License Info**: Clear licensing information
- 📦 **Version Control**: Published/accepted/submitted versions

**Output:**
```json
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
```

**Usage:**
```python
# Get full-text access
fulltext = await get_fulltext_access(
    doi="10.1038/s41587-024-02534-3"
)

if fulltext['is_oa']:
    pdf_url = fulltext['best_oa_location']['url']
    print(f"Download: {pdf_url}")
```

### 4. **enrich_work_data** 🆕 Enhanced Content
Get comprehensive work data with abstract and full-text in one call.

**Parameters:**
- `work_id` (optional): OpenAlex work ID
- `doi` (optional): DOI of the paper

**Key Features:**
- 🎁 **All-in-One**: OpenAlex + Semantic Scholar + Unpaywall
- 📊 **Complete**: Metadata + abstract + PDF links
- 🎯 **Optimized**: Single call for full enrichment
- 📝 **Metadata**: Shows which sources succeeded

**Output:**
```json
{
  "id": "https://openalex.org/W1234567890",
  "title": "...",
  "abstract": "Complete abstract text...",
  "abstract_inverted": false,
  "fulltext_urls": [{
    "url": "https://...",
    "host_type": "publisher",
    "license": "cc-by"
  }],
  "enrichment_metadata": {
    "sources_used": ["openalex", "semantic_scholar", "unpaywall"],
    "abstract_source": "semantic_scholar",
    "fulltext_source": "unpaywall"
  }
}
```

**Usage:**
```python
# Get everything in one call
enriched = await enrich_work_data(
    doi="10.1038/s41587-024-02534-3"
)

print(f"Abstract: {enriched['abstract'][:200]}...")
print(f"PDF: {enriched['fulltext_urls'][0]['url']}")
```

### 5. **search_authors**
Search for authors with streamlined output for AI agents.

**Parameters:**
- `name` (required): Author name to search
- `institution` (optional): Institution name filter
- `topic` (optional): Research topic filter
- `country_code` (optional): Country code filter (e.g., "US", "DE")
- `limit` (optional): Maximum results (1-25, default: 20)

**Features**: Clean structure optimized for AI reasoning and disambiguation

---

### 6. **retrieve_author_works**
Retrieve works for a given author with enhanced filtering capabilities.

**Parameters:**
- `author_id` (required): OpenAlex author ID
- `limit` (optional): Maximum results (1-50, default: 20)
- `order_by` (optional): "date" or "citations" (default: "date")
- `publication_year` (optional): Filter by specific year
- `type` (optional): Work type filter (e.g., "journal-article")
- `authorships_institutions_id` (optional): Filter by institution
- `is_retracted` (optional): Filter retracted works
- `open_access_is_oa` (optional): Filter by open access status

**Enhanced Output:**
```json
{
  "author_id": "https://openalex.org/A123456789",
  "total_count": 25,
  "results": [
    {
      "id": "https://openalex.org/W123456789",
      "title": "A platform for the biomedical application of large language models",
      "doi": "10.1038/s41587-024-02534-3",
      "publication_year": 2025,
      "type": "journal-article",
      "cited_by_count": 42,
      "authorships": [
        {
          "author": {
            "display_name": "Jorge Abreu-Vicente"
          },
          "institutions": [
            {
              "display_name": "European Molecular Biology Organization"
            }
          ]
        }
      ],
      "locations": [
        {
          "source": {
            "display_name": "Nature Biotechnology",
            "type": "journal"
          }
        }
      ],
      "open_access": {
        "is_oa": true
      },
      "primary_topic": {
        "display_name": "Biomedical Engineering"
      }
    }
  ]
}
```

**Features**: Comprehensive work data with flexible filtering for targeted queries

---

## 📊 Data Optimization

### Focused Information Architecture
This MCP server provides focused, structured data specifically designed for AI agent consumption:

### Author Data Features
- **Identity Resolution**: Names, ORCID, alternatives for disambiguation
- **Affiliation Tracking**: Current and historical institutional connections
- **Impact Metrics**: Citation counts, h-index, and scholarly impact
- **Research Context**: Fields, concepts, and domain expertise
- **Career Analysis**: Temporal affiliation changes and transitions

### Work Data Features
- **Publication Metadata**: Title, DOI, venue, and publication details
- **Impact Assessment**: Citation counts and scholarly influence
- **Access Information**: Open access status and availability
- **Authorship Details**: Complete author lists and institutional affiliations
- **Research Classification**: Topics, concepts, and domain categorization

### Enhanced Filtering

```python
# Target high-impact journal articles
works = await retrieve_author_works(
    author_id="https://openalex.org/A123456789",
    type="journal-article",      # Focus on journal publications
    open_access_is_oa=True,      # Open access only
    order_by="citations",        # Most cited first
    limit=15
)

# Career transition analysis
authors = await search_authors(
    name="J. Abreu",
    institution="EMBO",          # Current institution
    topic="Machine Learning",    # Research focus
    limit=10
)
```

---

## 🧪 Example Usage

### Author Disambiguation

```python
from alex_mcp.server import search_authors_core

# Comprehensive author search
results = search_authors_core(
    name="J Abreu Vicente",
    institution="EMBO",
    topic="Machine Learning",
    limit=20
)

print(f"Found {results.total_count} candidates")
for author in results.results:
    print(f"- {author.display_name}")
    if author.affiliations:
        current_inst = author.affiliations[0].institution.display_name
        print(f"  Institution: {current_inst}")
    print(f"  Metrics: {author.cited_by_count} citations, h-index {author.summary_stats.h_index}")
    if author.x_concepts:
        fields = [c.display_name for c in author.x_concepts[:3]]
        print(f"  Research: {', '.join(fields)}")
```

### Academic Work Analysis

```python
from alex_mcp.server import retrieve_author_works_core

# Comprehensive work retrieval
works = retrieve_author_works_core(
    author_id="https://openalex.org/A5058921480",
    type="journal-article",      # Academic focus
    order_by="citations",        # Impact-based ordering
    limit=20
)

print(f"Found {works.total_count} publications")
for work in works.results:
    print(f"- {work.title}")
    if work.locations:
        journal = work.locations[0].source.display_name
        print(f"  Published in: {journal} ({work.publication_year})")
    print(f"  Impact: {work.cited_by_count} citations")
    if work.open_access and work.open_access.is_oa:
        print("  ✓ Open Access")
```

### Institution and Field Analysis

```python
# Analyze career transitions
def analyze_career_path(author_result):
    affiliations = author_result.affiliations
    if len(affiliations) > 1:
        print("Career path:")
        for aff in sorted(affiliations, key=lambda x: min(x.years)):
            years = f"{min(aff.years)}-{max(aff.years)}"
            print(f"  {years}: {aff.institution.display_name}")
    
    # Research evolution
    if author_result.x_concepts:
        print("Research areas:")
        for concept in author_result.x_concepts[:5]:
            print(f"  {concept.display_name} (score: {concept.score:.2f})")

# Usage
results = search_authors_core("Jorge Abreu Vicente")
if results.results:
    analyze_career_path(results.results[0])
```

---

## 🔧 Configuration Options

### Environment Variables

```bash
# Required
export OPENALEX_MAILTO=your-email@domain.com

# Optional settings
export OPENALEX_MAX_AUTHORS=100             # Maximum authors per query
export OPENALEX_USER_AGENT=research-agent-v1.0
export ALEX_MCP_VERSION=4.1.0

# Rate limiting (respectful usage)
export OPENALEX_RATE_PER_SEC=10
export OPENALEX_RATE_PER_DAY=100000
```

### Performance Tuning

```python
# For comprehensive research applications
config = {
    "max_authors_per_query": 25,     # Detailed author analysis
    "max_works_per_author": 50,      # Complete publication history
    "enable_all_filters": True,      # Full filtering capabilities
    "detailed_affiliations": True,   # Complete institutional data
    "research_concepts": True        # Detailed concept analysis
}
```

---

## 🧑‍💻 Development & Testing

### Project Structure
```
alex-mcp/
├── src/alex_mcp/
│   ├── server.py              # Main MCP server
│   ├── data_objects.py        # Data models and structures
│   └── utils.py               # Utility functions
├── examples/
│   ├── basic_usage.py         # Simple examples
│   ├── advanced_queries.py    # Complex query examples
│   └── integration_demo.py    # AI agent integration
├── tests/
│   ├── test_server.py         # Server functionality tests
│   └── test_integration.py    # Integration tests
└── docs/
    └── api_reference.md       # Detailed API documentation
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run functionality tests
pytest tests/test_server.py -v

# Test with real queries
python examples/basic_usage.py

# Test AI agent integration
python examples/integration_demo.py
```

### Development Examples

```bash
# Test author disambiguation
python examples/basic_usage.py --query "J. Abreu" --institution "EMBO"

# Test work retrieval
python examples/advanced_queries.py --author-id "A123456789" --type "journal-article"

# Test integration patterns
python examples/integration_demo.py --workflow "career-analysis"
```

---

## 📈 Integration Examples

### Academic Research Workflows

Perfect integration with AI-powered research analysis:

```python
# Enhanced academic research agent
from alex_agent import AcademicResearchAgent

agent = AcademicResearchAgent(
    mcp_servers=[alex_mcp],  # Streamlined data processing
    model="gpt-4.1-2025-04-14"
)

# Complex research queries with structured data
result = await agent.research_author(
    "Find J. Abreu at EMBO with machine learning publications"
)

# Rich, structured output for AI reasoning
print(f"Quality Score: {result.quality_score}/100")
print(f"Author disambiguation: {result.confidence}")
print(f"Research fields: {result.research_domains}")
```

### Multi-Agent Systems

```python
# Collaborative research analysis
async def research_collaboration_network(seed_author):
    # Find primary author
    authors = await alex_mcp.search_authors(seed_author)
    primary = authors['results'][0]
    
    # Get their works
    works = await alex_mcp.retrieve_author_works(
        primary['id'], 
        type="journal-article"
    )
    
    # Analyze co-authors and build network
    collaborators = set()
    for work in works['results']:
        for authorship in work.get('authorships', []):
            collaborators.add(authorship['author']['display_name'])
    
    return {
        'primary_author': primary,
        'publication_count': len(works['results']),
        'collaborator_network': list(collaborators),
        'research_impact': sum(w['cited_by_count'] for w in works['results'])
    }
```

---

## 🤝 Contributing

We welcome contributions to improve functionality and add new features:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/enhanced-filtering`
3. **Add tests**: Ensure your changes maintain data quality and structure
4. **Submit a pull request**: Include examples and documentation

### Development Priorities

- [ ] Enhanced filtering capabilities
- [ ] Additional data enrichment
- [ ] Performance optimizations
- [ ] Integration examples
- [ ] Documentation improvements

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🌐 Links

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/ContextualAI/fastmcp)
- [OpenAI Agents](https://github.com/openai/openai-agents)
- [Academic Research Examples](examples/)
