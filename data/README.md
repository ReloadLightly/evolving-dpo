# data/

## What is committed here, and what is not

Committed: counts, cards, and identifiers.
Never committed: review text.

That split is a commitment, not an oversight (HANDOFF §8). OpenReview comments
are CC BY 4.0 and reviewers wrote them for authors and area chairs, not as
preference-optimization data. We release pair **identifiers and construction
code** so anyone can rebuild the dataset from the source of record, rather than
redistributing the text in bulk. `.gitignore` enforces it: `data/raw/`,
`data/pairs.jsonl` and `data/*.jsonl` are excluded.

| Path | Committed | What it is |
|---|---|---|
| `iclr_split_stats.json` | yes | Task 1 yield measurement, from papercopilot/paperlists |
| `raw/cache/iclr{year}.json.gz` | no | verbatim OpenReview responses, replies included |
| `raw/iclr{year}.jsonl` | no | normalized per-paper records (Task 2 output) |
| `raw/fetch_summary.json` | no | per-year counts, measured vs the Task 1 reference |
| `raw/field_census.json` | no | which content fields each year actually uses |
| `pairs.jsonl` | no | the preference pairs (Task 3 output) |
| `DATASET_CARD.md` | yes, once Task 3 runs | counts by year and area, licence, exclusions |

## Rebuilding from scratch

```bash
pip install openreview-py
python scripts/fetch_openreview.py --check              # is the API reachable?
python scripts/fetch_openreview.py --years 2023 --limit 25   # smoke test
python scripts/fetch_openreview.py --years 2020-2025    # the real pull
```

The download is the expensive part and is cached per year, so re-parsing after
an extractor change never re-downloads:

```bash
python scripts/fetch_openreview.py --years 2020-2025 --reparse
```

## The record schema of `raw/iclr{year}.jsonl`

One JSON object per submission:

```json
{"paper_id": "...", "forum": "...", "number": 123, "year": 2024, "api_version": 2,
 "title": "...", "abstract": "...", "keywords": ["..."], "primary_area": "...",
 "venue": "ICLR 2024 poster", "venueid": "ICLR.cc/2024/Conference",
 "decision": "Accept (poster)", "decision_source": "Decision|Meta_Review|venue",
 "accepted": true, "withdrawn": false, "desk_rejected": false,
 "n_reviews": 4, "n_ratings": 4, "rating_spread": 5.0, "n_authors": 6,
 "reviews": [{"review_id": "...", "rating": 8.0, "rating_field": "rating",
              "rating_raw": 8, "confidence": 4.0,
              "aux_scores": {"soundness": 3.0},
              "text": "## Summary\n\n...", "text_fields": {"summary": "..."},
              "n_chars": 812, "tcdate": 0, "tmdate": 0}],
 "meta_review": "..."}
```

Two fields are worth knowing about before Task 3 uses them:

- **`text_fields`** keeps the review's sections separately (`summary`,
  `strengths`, `weaknesses`, …) alongside the concatenated `text`. Task 3 can
  therefore build prompts from, say, weaknesses only, without re-downloading.
- **`decision_source`** records where the verdict came from. ICLR 2020 has no
  `Decision` note — its verdict lives in the meta-review's `recommendation` —
  and for API v2 years the submission's `venue` string ("ICLR 2024 poster" vs
  "Submitted to ICLR 2024") stays public even when the Decision note does not.
  A year whose decisions all come from `venue` is fine; a year with no decision
  source at all is a bug, and `fetch_summary.json` will show it as a low
  "papers with a decision" percentage.

## Known per-year differences the extractor handles

| Years | API | Score field | Prose fields |
|---|---|---|---|
| 2020, 2021 | v1 | `rating` | one `review` field |
| 2022, 2023 | v1 | `recommendation` | `summary_of_the_paper`, `main_review`, `summary_of_the_review` |
| 2024, 2025 | v2 | `rating` | `summary`, `strengths`, `weaknesses`, `questions` |

Nothing here is hardcoded per year — the fields are detected per review and
what was found is recorded. Run `--census` on a year to see its actual schema
before trusting the output.

`primary_area` exists only from 2024, which constrains the H3 area split to the
later years unless areas are derived from keywords for earlier ones.
