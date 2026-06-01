# Native FA2 q2 Grouping Experiment - Negative Full-Model Result

Purpose: test whether grouping two adjacent query rows in the native source
FA2 kernel can recover the remaining c32 gap after the restored source path.

Candidate:

- commit: `3a5ab00`
- change: add a large-prefill q2 grouped native FA2 kernel branch
- status: reverted by `2197077`

Microbench artifact:

- `/workspace/m3-native-fa2-q2-microbench-20260601.log`

Microbench comparison, same restored pod:

| shape (`num_seqs q_len kv_len`) | baseline `451a97f` avg_us | q2 `3a5ab00` avg_us | result |
|---|---:|---:|---:|
| `4 288 289` | `644.553` | `479.611` | `+34.4%` faster |
| `3 285 291` | `493.914` | `361.968` | `+36.5%` faster |
| `32 4 256` | `99.169` | `98.446` | `+0.7%` faster |
| `32 1 256` | `36.567` | `36.356` | `+0.6%` faster |

Full-model c32 N=3 artifact:

- `/workspace/m3-fa2-q2-c32-n3-20260601`

Artifact validation:

```bash
cd /workspace/ferrum-fa2-native-restore-git-ac3dfab
python3 scripts/m3_validate_runner_artifact.py \
  --require-bench /workspace/m3-fa2-q2-c32-n3-20260601
```

Validator result:

- `ok=true`
- `summary_rows=2`

Full-model result:

| case | throughput tok/s | notes |
|---|---:|---|
| q2 source FA2 | `1462.15 ± 119.46` | Paris, multi-turn, three-turn recall passed |
| FA-layout baseline | `1345.28 ± 111.26` | Paris, multi-turn, three-turn recall passed |

Conclusion:

- The q2 candidate is a microbench-positive but full-model-negative path.
- Compared with the current source-FA2 all-cell c32 row (`1488.08 tok/s`), q2
  regressed by about `-1.7%`.
- Do not reintroduce q2 grouping as a primary path without a new profile that
  explains why the full-model scheduler/fill mix differs from the isolated
  large-prefill microbench.
