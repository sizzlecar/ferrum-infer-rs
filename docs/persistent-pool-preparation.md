# Persistent pool preparation

Compiled-plan executors can allocate their configured persistent state envelope
before becoming ready. The default is `demand-driven`; it adds no startup pool
allocation. Select the option explicitly for either product entrypoint:

```text
ferrum run MODEL --persistent-pool-preparation configured-envelope ...
ferrum serve MODEL --persistent-pool-preparation configured-envelope ...
```

The equivalent configuration is:

```toml
[runtime]
persistent_pool_preparation = "configured-envelope"
```

CLI values override the configuration file. An environment variable cannot
select this behavior. Executors without compiled-plan allocation authority
reject `configured-envelope` before loading weights.

The compiler evaluates the complete selected context length, active sequence
limit, and scheduled-token wave. The check includes static weights, separately
quantized per-sequence state, pool minimums, transient workspace and declared
reusable workspace. The full peak must fit the original usable memory budget.
Unsupported storage geometry fails explicitly. The initial supported domain is
Sequence-only persistent pools with paged token-derived demand or uniform fixed
contiguous cells; page-count demands require separate evidence.

After static initialization finishes, the executor allocates only the computed
persistent targets through its original maintenance controller. This preserves
the existing global capacity reservation, allocator validation and failure
cleanup. It does not allocate every pool's maximum or preallocate transient
workspace. The model health field `startup_persistent_preparation` records the
mode, resolved shape, checked peak, selected targets and actual growth receipt.

This is initial residency, not a hard floor. Normal maintenance may reclaim it;
admission still uses original owners, and scheduler capacity observations still
inspect actual backing. The receipt does not establish future availability,
request release timing or SLO compliance. Performance comparisons must declare
this option and use the same allocation mode in every arm on a given backend.
