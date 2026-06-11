# Evaluation, Observability, And Benchmarks

## Evaluation As Control Plane

Evaluation is a dedicated LangGraph node. It does more than report a score: hard failures select the next graph edge, while soft diagnostics explain itinerary quality without automatically destabilizing generation.

## Hard Validation

The deterministic evaluator checks:

- Schema correctness: model output parses into `TripPlan`.
- Date coverage: daily plans and authoritative weather cover requested dates.
- Budget consistency: totals align with itemized attractions, hotels, meals, and transportation.
- Current-request alignment: explicit transportation and accommodation choices override historical memory.
- Retrieval grounding: attractions and hotels map to retrieved candidates or supported evidence.

Hard failures can rerun the planner, rerun a targeted retrieval node, or route to fallback after retry exhaustion.

## Quality Diagnostics

Soft diagnostics are explainable engineering proxies:

- Pacing score estimates whether daily attraction duration and buffers are overloaded or underfilled.
- Route coherence score uses coordinate distance to detect implausible city-scale jumps.
- Preference match score checks overlap with explicit preferences and free-text intent.
- Attribution coverage checks whether generated attractions and hotels have evidence links.

Soft warnings are recorded but do not trigger retries by default.

## Evidence Attribution

The evaluator builds deterministic evidence links:

- Attraction recommendations map to attraction candidates.
- Hotel recommendations map to hotel candidates.
- Destination-specific themes and advice map to retrieved RAG chunks where possible.

Each link records the entity, evidence type, evidence ID, source metadata, confidence, and match reason. This makes unsupported recommendations inspectable rather than treating all failures as generic hallucinations.

## Observability

The local observability service stores run summaries and ordered node events in SQLite. It records:

- Request and final-plan summaries.
- Hard failures, scores, warnings, and unsupported entities.
- Node latency and attempts.
- Retry and routing decisions.
- Retrieved RAG source summaries and evidence links.

The `/observability` frontend route and `/api/observability` endpoints provide local debugging views. Persistence is best-effort and never blocks a successful planning response.

## Benchmark Methodology

The benchmark harness runs the same fixed requests with:

- Generic local planning knowledge.
- Chroma destination-focused retrieval.

It records request-level reports and aggregated metrics. Current public summary is available at `backend/benchmarks/results/public-summary.json`.

Current Chroma results on the fixed 12-request internal dataset:

| Metric | Result |
| --- | ---: |
| Retrieval recall@4 | 91.67% |
| Retrieval hit rate | 100% |
| Hard validation pass rate | 100% |
| Evidence attribution coverage | 100% |
| Initially failed runs | 3 |
| Recovery rate among initially failed runs | 100% |
| Fallback rate | 0% |

## Interpreting The Results

- Recall@4 measures whether retrieval found expected documents.
- Grounding and attribution measure whether output recommendations map to evidence.
- Recovery rate measures whether initially failed generations passed after controlled retries.
- Hard validation pass rate measures final evaluator acceptance.

The benchmark is small and internal. It supports regression testing and engineering decisions, but it is not evidence of production-scale quality.

## Reproduce

```bash
cd backend
venv/bin/python scripts/build_rag_index.py --rebuild
venv/bin/python scripts/benchmark_trip_planners.py \
  --dataset benchmarks/trip_requests.rag_benchmark.json \
  --output benchmarks/results/trip_planner_rag_benchmark.json \
  --persist-observability
```

## Future Improvements

- Add larger adversarial and multilingual benchmark sets.
- Add route-time evaluation using a map route API.
- Calibrate deterministic thresholds against human ratings.
- Add durable checkpointing and shared observability storage.
- Add optional model-assisted quality evaluation while keeping deterministic checks as hard gates.
