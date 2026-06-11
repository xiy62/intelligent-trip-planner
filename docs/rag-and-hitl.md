# RAG And Human-in-the-Loop Ingestion

## Why RAG Exists

AMap provides real POI candidates and the weather service provides authoritative forecasts, but neither source explains how to combine destination knowledge into a coherent itinerary. The RAG layer supplies route grouping, recommended duration, seasonal context, transport advice, and planning tips.

RAG is reference context, not the source of truth for POI existence or weather.

## Runtime Retrieval

`TravelRAGService` supports two modes:

- `local_lightweight`: small generic planning rules used as a baseline.
- `chroma_retrieval`: destination-focused retrieval from approved knowledge documents.

Chroma uses OpenAI `text-embedding-3-small` embeddings. The runtime builds a query from city, trip duration, transportation, accommodation, preferences, free-text intent, and top attraction candidates. It retrieves up to four chunks and writes them into `rag_chunks`.

The planner receives serialized RAG context alongside POI candidates and weather. The evaluator later checks whether generated recommendations map back to retrieved evidence.

## Knowledge Document Schema

Approved documents contain:

- Stable `doc_id`, country, city, district, title, and language.
- Themes, POI names, best-use cases, and recommended duration.
- Seasonality, transport advice, and planning tips.
- Source type, source URL, last verification date, and source-supported content.

The repository includes approved seed knowledge for Beijing, Shanghai, Hangzhou, and Guangzhou.

## Human Review Workflow

```text
source manifest
  -> fetch and rule-based extraction
  -> reviewable draft JSON
  -> human source verification and edits
  -> approved promotion
  -> Chroma index rebuild
```

Generate drafts:

```bash
cd backend
venv/bin/python scripts/ingest_rag_sources.py \
  --manifest data/sources/china_travel_sources.sample.json
```

Only drafts explicitly marked `review_status: "approved"` pass schema validation and promotion:

```bash
venv/bin/python scripts/promote_rag_drafts.py
venv/bin/python scripts/build_rag_index.py --rebuild
```

Raw pages and review drafts are intentionally excluded from the public repository. Runtime retrieval only reads approved knowledge.

## Why Human Review

Official travel sites vary in layout and often mix factual content with marketing language. Fully automated extraction can over-infer route advice, preserve stale claims, or lose the relationship between a recommendation and its source.

The reviewer verifies source URLs, source support, POI/theme metadata, transport advice, and any closure, reservation, ticketing, or opening-hour claims before promotion.

## Retrieval Evaluation

The fixed benchmark labels each request with expected document IDs:

- `retrieval_hit_rate`: fraction of labeled requests retrieving at least one expected document.
- `retrieval_recall_at_4`: average fraction of expected documents found in the first four chunks.

Current fixed 12-request internal benchmark:

- Retrieval hit rate: `100%`
- Retrieval recall@4: `91.67%`

These metrics evaluate retrieval quality. They do not prove that every generated sentence is correct; output grounding and evidence attribution are evaluated separately.

## Limitations

- The current corpus is a curated seed corpus, not a complete automatically scraped official knowledge base.
- Retrieval labels are manually defined at document level.
- Chroma is local and must be rebuilt after approved knowledge changes.
- Expanding to US cities requires a new map provider and a new reviewed corpus, but not a graph-state redesign.
