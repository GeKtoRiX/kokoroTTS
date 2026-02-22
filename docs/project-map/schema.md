# Data Schema

## SQLite (`MORPH_DB_PATH`)

### `morph_lexemes`

| Column | Definition |
| --- | --- |
| `dedup_key` | `TEXT NOT NULL PRIMARY KEY` |
| `lemma` | `TEXT NOT NULL` |
| `upos` | `TEXT NOT NULL` |
| `feats_json` | `TEXT NOT NULL` |
| `created_at` | `TEXT NOT NULL DEFAULT (datetime('now'))` |

### `morph_token_occurrences`

| Column | Definition |
| --- | --- |
| `id` | `INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT` |
| `source` | `TEXT NOT NULL` |
| `part_index` | `INTEGER NOT NULL` |
| `segment_index` | `INTEGER NOT NULL` |
| `token_index` | `INTEGER NOT NULL` |
| `voice` | `TEXT NOT NULL` |
| `token_text` | `TEXT NOT NULL` |
| `lemma` | `TEXT NOT NULL` |
| `upos` | `TEXT NOT NULL` |
| `feats_json` | `TEXT NOT NULL` |
| `start_offset` | `INTEGER NOT NULL` |
| `end_offset` | `INTEGER NOT NULL` |
| `dedup_key` | `TEXT NOT NULL` |
| `text_sha1` | `TEXT NOT NULL` |
| `created_at` | `TEXT NOT NULL DEFAULT (datetime('now'))` |

Constraints:
- `UNIQUE ("source", "text_sha1", "part_index", "segment_index", "token_index", "dedup_key", "start_offset", "end_offset")`

### `morph_expressions`

| Column | Definition |
| --- | --- |
| `id` | `INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT` |
| `source` | `TEXT NOT NULL` |
| `part_index` | `INTEGER NOT NULL` |
| `segment_index` | `INTEGER NOT NULL` |
| `expression_index` | `INTEGER NOT NULL` |
| `voice` | `TEXT NOT NULL` |
| `expression_text` | `TEXT NOT NULL` |
| `expression_lemma` | `TEXT NOT NULL` |
| `expression_type` | `TEXT NOT NULL` |
| `start_offset` | `INTEGER NOT NULL` |
| `end_offset` | `INTEGER NOT NULL` |
| `expression_key` | `TEXT NOT NULL` |
| `match_source` | `TEXT NOT NULL` |
| `wordnet_hit` | `INTEGER NOT NULL DEFAULT 0` |
| `text_sha1` | `TEXT NOT NULL` |
| `created_at` | `TEXT NOT NULL DEFAULT (datetime('now'))` |

Constraints:
- `UNIQUE ("source", "text_sha1", "part_index", "segment_index", "expression_index", "expression_key", "start_offset", "end_offset")`

### Indexes

- `CREATE INDEX IF NOT EXISTS "morph_token_occurrences_dedup_key_idx" ON "morph_token_occurrences" ("dedup_key")`
- `CREATE INDEX IF NOT EXISTS "morph_expressions_key_idx" ON "morph_expressions" ("expression_key")`
- `CREATE INDEX IF NOT EXISTS "morph_expressions_type_idx" ON "morph_expressions" ("expression_type")`

## Pronunciation Rules JSON (`PRONUNCIATION_RULES_PATH`)

- Root object: language code -> dictionary
- Language keys: normalized single-letter Kokoro language code (`a,b,e,f,h,i,j,p,z`)
- Entry value: word -> phoneme string
- Invalid language buckets are skipped at load-time
