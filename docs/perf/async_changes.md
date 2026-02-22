# Async Changes

Files: `kokoro_tts/application/state.py`

## What changed

1. Added async backlog queue for morphology ingest payloads:
   - `_pending_morph_payloads`
   - `_morph_payload_backlog_limit`
2. Added non-blocking draining path:
   - `_drain_pending_morph_payloads()`
   - callbacks now schedule backlog work when futures complete.
3. Improved wait/shutdown behavior:
   - `wait_for_pending_morphology()` now loops until both running futures and backlog are empty.
   - `shutdown(wait=True)` flushes deferred backlog payloads before executor shutdown.
4. Reduced lock-held work in enqueue path:
   - payload is prepared before lock.
   - submission logic centralized in `_submit_morph_payload()`.

## Why

- Previous behavior fell back to synchronous writes when queue was full, increasing tail latency under load.
- New backlog queue preserves correctness while avoiding immediate caller-path blocking in common surge cases.

## Validation

- Existing async tests remain green:
  - `tests/application/test_state.py::test_async_morphology_ingest_enqueue_and_flush`
  - `tests/application/test_state.py::test_shutdown_clears_async_executor_and_pending_futures`
- No functional regressions detected in full suite and smoke checks.

