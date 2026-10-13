# Receipt: local main consolidation

**Date:** 2026-10-13  
**Agent:** Codex  
**Scope:** consolidate owned Qwen/Strata MLX work onto local `main` branches without pushing.

## MLX

- Local `main` merged `research/strata-mxfp4-optane`, bringing the campaign
  architecture and evidence into the branch.
- Local `main` already contained the selective upstream integration through
  `388f227f4` with `494c8a6c9` excluded.
- Added the three Strata/PLE storage benchmarks, the storage-tier plan, the
  upstream/storage receipt, and the preserved `_io_ple` prototype under
  `evidence/prototypes/`.
- The early `python/mlx/_io_ple.py` remains archived rather than live because
  the production implementation is in mlx-lm.
- Source head before this receipt commit: `019cd487c`.

Validation:

```text
ruff check benchmarks/python/ple_row_width_bench.py benchmarks/python/storage_table_bench.py benchmarks/python/strata_mxfp4_ple_bench.py
ruff format --check benchmarks/python/ple_row_width_bench.py benchmarks/python/storage_table_bench.py benchmarks/python/strata_mxfp4_ple_bench.py
python -m py_compile <the same three benchmark files>
<each benchmark> --help
git diff --check
```

All exited 0.

## mlx-lm

- Local `main` fast-forwarded from `4918996` to `18bb2e7`.
- This brings the qwen4-exp port, dynamic native MXFP4 modes, exact greedy fast
  path, idle warmup, packed-T1 GDN, external PLE runtime, cache lifecycle
  repairs, performance harnesses, and fused GDN decode convolution/state kernel
  onto local `main`.
- `dynamic-mxfp4` and `main` now point to `18bb2e7`.

Validation on the newly selected `main`:

```text
PYTHONPATH="$PWD" python -m pytest \
  tests/test_qwen4_exp_ple.py tests/test_dynamic_quant.py \
  tests/test_greedy_fast.py tests/test_ple_runtime.py tests/test_server.py -q
```

Result: 59 passed, 9 skipped.

## Non-claims

- No remote branch was changed and nothing was pushed.
- No new model-quality or performance claim is made by the branch movement.
- No Optane result.
