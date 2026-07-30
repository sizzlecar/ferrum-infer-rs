# CUDA Native Operator Sources

Ferrum product builds consume versioned native operator artifacts. They do not
compile Marlin, vLLM Marlin, vLLM MoE Marlin, or vLLM paged-attention sources
from the Rust workspace.

The locked source definitions are under `source-definitions/`. Their exact
source members are distributed as the GitHub release asset described by:

```text
source-bundles/ferrum-native-cuda-v1.json
```

The bundle manifest pins the archive SHA256, every member SHA256, upstream
repositories, revisions, and licenses. Materialize it outside the Git worktree:

```bash
python3 scripts/release/native_operator_source_bundle.py materialize \
  --manifest native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json \
  --out /tmp/ferrum-native-cuda-v1
```

Then use `ferrum-native-ops-builder` with that external source root. The builder
is the only supported source compilation path. It creates source locks,
reproducible build receipts, packages, and the artifact-set lock consumed by
`FERRUM_NATIVE_OPERATOR_SET_LOCK`.

The product build fails closed when an active CUDA native unit is not covered by
the supplied artifact-set lock. It never falls back to compiling the source
bundle.
