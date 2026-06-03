# Prefix and Session Cache Product Surface

Ferrum exposes cache controls as explicit serving options. Defaults remain
conservative: prefix cache is off unless enabled, and session cache is off unless
a caller opts in.

## CLI Flags

```bash
ferrum serve qwen3:0.6b \
  --enable-prefix-cache \
  --session-cache memory \
  --session-cache-max-entries 128 \
  --session-cache-max-tokens 4096
```

Flags map to runtime config entries and appear in `--effective-config-json`:

| Flag | Runtime key |
|---|---|
| `--enable-prefix-cache` | `FERRUM_PREFIX_CACHE=1` |
| `--disable-prefix-cache` | `FERRUM_PREFIX_CACHE=0` |
| `--session-cache off` | `FERRUM_SESSION_CACHE=off` |
| `--session-cache memory` | `FERRUM_SESSION_CACHE=memory` |
| `--session-cache-max-entries N` | `FERRUM_SESSION_CACHE_MAX_ENTRIES=N` |
| `--session-cache-max-tokens N` | `FERRUM_SESSION_CACHE_MAX_TOKENS=N` |

The vLLM-compatible `--enable-prefix-caching` and
`--no-enable-prefix-caching` flags remain supported aliases for migration.

## Session API

Ferrum supports an explicit opt-in session id:

```text
X-Ferrum-Session: <opaque-session-id>
```

The body metadata alternative is also accepted:

```json
{"metadata":{"ferrum_session_id":"session-a"}}
```

Rules:

- Session cache is disabled unless `--session-cache memory` is set.
- No session id means stateless OpenAI-compatible behavior.
- Session ids are opaque strings.
- Session cache is bounded by max entries and approximate max retained tokens.
- Evictions and hit/miss counters are exposed through `/metrics` and `/health`.

## Observability

`/health` includes:

```json
{
  "cache": {
    "prefix_cache": {"enabled": true, "entries": 12, "hits": 34, "misses": 8},
    "session_cache": {"mode": "memory", "entries": 3}
  }
}
```

`/metrics` always includes the G3 cache metrics:

```text
ferrum_prefix_cache_hits_total
ferrum_prefix_cache_misses_total
ferrum_prefix_cache_evictions_total
ferrum_prefix_cache_saved_prefill_tokens_total
ferrum_prefix_cache_entries
ferrum_prefix_cache_bytes
ferrum_session_cache_hits_total
ferrum_session_cache_misses_total
ferrum_session_cache_evictions_total
ferrum_session_cache_entries
ferrum_session_cache_tokens
```

## Correctness Policy

Prefix/session cache must not corrupt deterministic greedy output, strict JSON
schema responses, tool calls, or session isolation. G3 gates verify:

- identical prompts with prefix cache enabled produce byte-identical greedy output
- shared-prefix prompts with different suffixes do not cross-talk
- strict JSON schema remains valid with prefix cache enabled
- required tool calls remain OpenAI-shaped with prefix cache enabled
- same-session history can be reused when the caller opts in
- different sessions do not leak secrets
- disabled prefix cache leaves hit counters at zero
