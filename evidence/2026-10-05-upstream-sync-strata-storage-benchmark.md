# Receipt: upstream synchronization and Strata-oriented storage benchmark

**Date:** 2026-10-05
**Agent:** Codex parent (GLM) plus three read-only CCR workers
**Repository state:** local `main`, not pushed; `origin/main` untouched

## Scope

This work synchronized the local MLX fork with upstream, audited the merge and storage-related integration seams, updated the separate Engram/Optane research corpus with pinned Strata evidence, and added the first local storage-table benchmark. It did not submit anything upstream. After the benchmark was ready, it created only a new benchmark directory on the mounted 750 GB Optane volume and ran a bounded synthetic test there.

## Git synchronization

- Starting local head: `53aa4566732c6e86a09fe3101e44f1a354ab1f84`
- Starting upstream head: `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`
- Fetched target: `264c14fe650e1dd9bb0c3902dc06b08033d8f89c`
- Merge result: `af219595b289cea24eb01dd7dd915bcc785fe1b7`
- Only pre-merge local source change was the `.gitignore` local-agent/index block.
- The expected `.gitignore` conflict was resolved by retaining upstream `.idea` and `wip` plus local `.sindexer/`, `.worktrees/`, and `.venv/`.
- Relative to upstream after the merge, the only committed difference is `.gitignore`.

## Worker audit

Three read-only CCR workers completed on requested route `zai,glm-5.3-flashx`:

1. Upstream history classification, `/Users/kearm/mlx/build/reports/upstream-audit/result.txt`
2. MLX storage/zero-copy integration audit, `/Users/kearm/mlx/build/reports/mlx-integration-audit/result.txt`
3. Optane/Thunderbolt measurement design, `/Users/kearm/mlx/build/reports/optane-benchmark-audit/result.txt`

The parent inspected the resulting implementation choices and did not accept worker claims without source checks. In particular, the parent validated page-aligned `PROT_READ` adoption and `mx.take` on Metal before implementing the benchmark.

## Research update

The Engram/Optane corpus now contains a pinned Strata study at:

`/Users/kearm/Documents/papers/2601.07372-engram/strata-study/README.md`

Pinned Strata revision: `6f32ec070f23ced9f50e704d854d775da52591ab`. The source archive SHA-256 is `c043e9f4c55fef75bfffeed2c034b3c91eb02e5e5fca44739839baca58f961ce`. All preserved source files passed `shasum -a 256 -c`. The corrected primary storage model is 320,001,536 IQ4_NL rows of 90 bytes: 28.800 GB / 26.822 GiB, with grouped 4 KiB direct reads and a bounded row cache. BF16 and FP8 remain precision controls. The earlier 64 KiB/token figure remains only the all-miss, one-independent-page-per-row worst case.

## Added benchmark

`benchmarks/python/storage_table_bench.py` implements:

- a Strata-shaped `issue`/`collect` ticket split so row requests can overlap later work;
- grouped `os.pread` requests for variable-width rows up to one page;
- page grouping, page deduplication, offset sorting, and bounded queue depth;
- an optional LRU row cache with hit/page/byte accounting;
- macOS `F_NOCACHE` as an unverified approximation of Linux `O_DIRECT`;
- an owned MLX staging transfer after row collection;
- a page-aligned, chunk-routed Metal mmap/`take` control path;
- synthetic row verification and JSON non-claims.

## Verification

Editable rebuild:

```text
uv pip install --python .venv/bin/python 'setuptools>=80' cmake ninja typing_extensions
uv pip install --python .venv/bin/python -e . --no-build-isolation
```

Exit 0. Installed version: `0.32.4.dev20261005+af219595b`.

Targeted tests:

```text
PYTHONPATH=/Users/kearm/mlx/python/tests \
  .venv/bin/python -m unittest discover -s python/tests -p 'test_load.py'
PYTHONPATH=/Users/kearm/mlx/python/tests \
  .venv/bin/python -m unittest discover -s python/tests -p 'test_zero_copy.py'
PYTHONPATH=/Users/kearm/mlx/python/tests \
  .venv/bin/python -m unittest discover -s python/tests -p 'test_memory.py'
```

Exit 0: 18, 9, and 4 tests respectively.

Full Python suite:

```text
PYTHONPATH=/Users/kearm/mlx/python/tests .venv/bin/python python/tests/run.py -v
```

Exit 0: **943 tests, 160.666 s, OK, 55 skipped.** One stream-context destructor warning was printed by `test_stream_context_exited_on_another_thread`, but that test passed.

Locked M4 smoke:

```text
uv run --project /Users/kearm/.local/share/alphaheng/environments/m4-mlx-smoke --frozen \
  python /Users/kearm/.local/share/alphaheng/environments/m4-mlx-smoke/smoke_mlx.py
```

Exit 0 on `Device(gpu, 0)`, Apple M4 Max, checksum `8386560.0`, 125.176 ms.

Benchmark lint and formatting:

```text
ruff check --fix benchmarks/python/storage_table_bench.py
ruff format benchmarks/python/storage_table_bench.py
ruff check benchmarks/python/storage_table_bench.py
```

Exit 0.

Live benchmark modes:

- 90-byte grouped `pread`, cache disabled: exit 0; latest receipt `build/reports/storage-table-bench/ple-90-final.json`, with separate issue, wait, I/O, and transfer accounting.
- 90-byte grouped `pread`, 1,024-row cache: exit 0; cache reached 1,020 rows and measured runs issued zero page reads.
- page-aligned 16,384-byte Metal mmap/take in four chunks: exit 0; latest receipt `build/reports/storage-table-bench/metal-16384-final.json`.
- local 90-byte fresh-index control: exit 0; receipt `build/reports/storage-table-bench/ple-90-fresh-local.json`.

## Thunderbolt Optane run

Read-only inventory first found:

- NVMe model `ET000750KWJTF`, BSD `disk12`, 750.16 GB, serial `PHKS826300DP750BGN`
- PCIe link 8.0 GT/s ×4, SMART verified
- ExFAT volume `disk12s2` mounted at `/Volumes/OPTANE`, 749.94 GB

A synthetic 4,050,000,000-byte, 45,000,000-row, 90-byte table was written under `/Volumes/OPTANE/mlx-storage-table-bench/`. The no-row-cache direct-read run used 256 tokens × 16 rows, 5 warmups, and 20 measured runs:

- page reads: 81,140
- storage bytes: 1,336,279,040
- storage-byte rate: 1,372,655,204.55 B/s
- useful-row rate: 7,573,502.23 B/s
- aggregate synthetic token rate: 5,259.38 tokens/s
- measured I/O wait portion: 0.692095418 s of 0.973499416 s total

Receipt: `build/reports/storage-table-bench/ple-90-optane.json`. A 1,048,576-row cache arm then served all 20,480 measured rows from cache with zero page reads (`ple-90-optane-cache.json`).

The volume disappeared during subsequent drive reconfiguration. System logs show `disk12s2` unmounting because media became absent; no ejection or mount command was issued by this work. A new external Intel `SSDPED1D480GAH`, 480.10 GB, serial `PHMB8360010R480JGN`, then appeared as `disk12`. After user-side partition setup, its ExFAT `P905` volume mounted at `/Volumes/P905`.

The stronger P905 run used the new `--fresh-indices` mode, so every warmup and measured run drew new random row IDs. A 4.05 GB file measured 1,330,680,230.15 storage B/s and 5,087.51 synthetic tokens/s (`ple-90-p905-fresh.json`).

The full-size synthetic Strata artifact was then written as 320,001,536 rows × 90 B = 28,800,138,240 B at `/Volumes/P905/mlx-storage-table-bench/ple-90-full.raw`. With five warmups, 20 measured fresh-index runs, 256 tokens × 16 rows per run, no row cache, and queue depth 256:

- page reads: 81,811
- storage bytes: 1,348,173,824
- storage-byte rate: 1,271,049,604.83 B/s
- useful-row rate: 6,951,028.39 B/s
- aggregate synthetic token rate: 4,827.10 tokens/s
- measured time: 1.060677584 s
- I/O collect wait: 0.766019669 s

Receipt: `build/reports/storage-table-bench/ple-90-p905-full-fresh.json`. The 1M-row cache arm on P905 served all 20,480 measured rows from cache with zero page reads (`ple-90-p905-cache.json`).

The benchmark files under `build/` are machine-state controls, not durability evidence; this receipt is the durable record.

## Non-claims

- The successful 750 GB run is not a sustained cold-cache benchmark; it used repeated indices, `F_NOCACHE`, and a 4.05 GB file, and macOS cache behavior was not independently traced.
- The 480 GB P905 runs used fresh indices and `F_NOCACHE`, but macOS cache behavior was still not independently traced and no five-minute thermal/sustained run was performed.
- The P905 artifact contains deterministic synthetic bytes, not model weights or a real Qwen token trace.
- No Qwen/Strata model shard was downloaded and no model output was validated.
- `F_NOCACHE` behavior was not independently proven equivalent to Linux `O_DIRECT`.
- Synthetic benchmark timings do not establish cold-storage throughput or end-to-end generation speed.
- No CUDA host was used; CUDA-only upstream changes remain unexercised.
- No C++ test binary was available in this checkout, so the full Python suite and locked Metal smoke are the merge acceptance evidence.
- No commit, push, pull request, issue comment, or upstream response was created.
