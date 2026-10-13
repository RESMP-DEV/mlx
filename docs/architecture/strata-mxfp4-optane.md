# Strata-inspired Qwen3.8-Flash-Next on MLX with MXFP4 and Optane

Date: 2026-10-05

## Purpose

Build a memory-safe MLX implementation of Qwen3.8-Flash-Next that uses native MXFP4 kernels for the resident model, a Thunderbolt Optane volume for the n-gram/PLE table, and the B550 CUDA machine for GPTQ-style calibration and packing. The design follows Strata's placement ideas, but remains MLX-native rather than porting Strata's CUDA code.

This is the implementation architecture. It does not claim that a full calibrated Flash-Next artifact exists.

## Verified starting points

- Local MLX core: `af219595b289cea24eb01dd7dd915bcc785fe1b7`, synchronized with upstream `264c14fe6`.
- Local MLX-LM qwen4-exp port: `/Users/kearm/mlx-lm`, branch `qwen4-exp`, head `e396b18b87e6274822c05bd6af14e223e9fbfce6`.
- Tiny-reference parity harness: `/Users/kearm/mlx-lm/research/qwen38-flash/verify/README.md`.
- Strata study: `/Users/kearm/Documents/papers/2601.07372-engram/strata-study/README.md`, pinned revision `6f32ec070f23ced9f50e704d854d775da52591ab`.
- Existing direct-storage benchmark: `benchmarks/python/storage_table_bench.py`.
- Diagnostic real-checkpoint control: `mlx-community/Qwen3.8-Flash-Next-4bit`, revision `07b5dc6c54600a359b87f1e53e7adf6351c72a2c`, 103.86 GiB of safetensors. It is retained only to exercise the real qwen4-exp key/layout/load path. Its default and per-module modes are affine, not MXFP4; it is not the target or publication artifact.
- Official FlashNext BF16 source is not present in the Mac HF cache. Its audited metadata size is 359,999,963,128 bytes across 1,658 tensors.
- P905: Intel `SSDPED1D480GAH`, 480.10 GB, ExFAT `/Volumes/P905`, PCIe 8.0 GT/s x4.
- Full synthetic Strata-sized table: 320,001,536 rows, 28,800,138,240 bytes.
- Fresh-index, zero-row-cache 90-byte P905 control: 1,271,049,604.83 storage B/s and 4,827.10 synthetic tokens/s at batch 256. `F_NOCACHE` was requested, but macOS cache residency was not independently traced.
- B550 alias `B550` / `compute`: two RTX 3090 Ti GPUs, 91 GiB RAM, and about 2.9 TB free disk.

## Non-goals for the first implementation

1. Do not port Strata's CUDA graphs, CPU expert kernels, or Windows/Linux direct-I/O layer.
2. Do not place the full PLE table in RAM or in one Metal buffer.
3. Do not use the stock MLX-LM loader on the BF16 Flash-Next checkpoint.
4. Do not implement MTP before text-only correctness and PLE offload are stable.
5. Do not launch B550 calibration while its swap is completely committed and another workload owns a GPU unless the user explicitly co-schedules them.

## Priority order

1. Tiny-reference correctness.
2. Bounded memory and no jetsam kills.
3. Standard MLX MXFP4 representation and kernels.
4. PLE storage latency and queue behavior.
5. Model-level throughput.
6. Fused Metal kernels and MTP speculation.

## Artifact architecture

```text
<artifact>/
  config.json
  tokenizer.json
  model-00000-of-NNNN.safetensors
  ...
  ple_manifest.json
  ple/codes-00000.bin
  ple/scales-00000.bin
  ...
  QUANT_RECEIPT.json
```

Normal safetensors shards contain resident model state only. PLE tensors are absent from the normal model index and are referenced by `ple_manifest.json`, preventing generic loaders from making the table resident.

### Resident model format

- Backbone linears, routed experts, GDN/QSA projections, and shared experts: native MLX MXFP4, group size 32, bits 4.
- Router gates: BF16.
- Sensitive norms and small scalars: BF16.
- Token embedding and output head: BF16 initially.

The B550 stage must emit standard MLX MXFP4 codes and E8M0 scales. It must not invent a PLE-only affine format when the target is native MXFP4 execution.

### External PLE format

One row has 160 scalar values. Native MXFP4 encodes it as 80 bytes of packed E2M1 codes plus five E8M0 scale bytes:

- codes plane: `uint32[rows, 20]`;
- scales plane: `uint8[rows, 5]`;
- total 85 bytes per logical row.

Two planes map directly to `mx.dequantize(..., mode="mxfp4")`. For 320,001,536 rows, codes use 25,600,122,880 bytes, scales use 1,600,007,680 bytes, and the total is 27,200,130,560 bytes.

The first two-plane implementation is validated by `benchmarks/python/strata_mxfp4_ple_bench.py`. On the current 750 GB `/Volumes/OPTANE`, a full 320,001,536-row fresh-index run read 2,661,133,824 bytes in 2.427013167 seconds, including 0.344787254 seconds of native dequantization: 2,109.59 synthetic tokens/s with peak Metal memory of 4,292,608 bytes. This is slower than the 90-byte control because independent code and scale planes can request two pages for one row; it is still far above the initial 50-100 tokens/s model target, but it is not a model-throughput claim.

`ple_manifest.json` records schema version, row count, logical head sizes and offsets, shard sizes, plane paths, SHA-256 hashes, dtype/layout, and cache defaults. A loader rejects any size or hash mismatch.

## Runtime architecture

The qwen4-exp implementation is the model base. Relevant seams are:

- `NGramEmbedding.__call__`, lines 671-687: EOS-aware hashing and global row IDs.
- `_ShardedEmbedding.__call__`, lines 708-720: shard routing and resident lookup.
- `PLELayer.__call__`, lines 766-780: PLE projection, gating, convolution, and residual integration.
- `Model.sanitize`, lines 1115-1196: key normalization and expert stacking.
- `Model.quant_predicate`, lines 1198-1206: current quantization exclusions.

Replace `_ShardedEmbedding` with an `ExternalPleTable`; do not rewrite `PLELayer` first.

### Issue/collect reader

The external table exposes:

```python
ticket = table.issue(global_row_ids)
values = table.collect(ticket)
```

`issue` validates row bounds, converts global IDs to shard-local IDs, checks an LRU row cache, deduplicates code and scale pages, sorts by offset, and admits no more than `max_inflight` outstanding page jobs across both planes. `collect` waits for that ticket, constructs `uint32[N, 20]` codes and `uint8[N, 5]` scales, calls native `mx.dequantize`, and restores original row order.

Worker count alone is not a queue bound. Large prompt requests are split into collect windows before issue; a single ticket may not create an unbounded future list. The benchmark caps selected rows per ticket, while the production reader uses bounded admission generations.

Row IDs are known from token IDs and preceding context. The model issues PLE reads before layer 0 and collects before the PLE layer. Decode issues once per verify window; prefill issues once per chunk.

The table artifact is immutable. The default row cache is a bounded, thread-safe LRU of 1,048,576 rows, about 85 MiB of payload, and reports capacity, hits, misses, and avoided bytes. Strata's reported 82% hit rate is context only, not a local claim.

The reader is bounded independently from the model: staging is bounded by queue depth, the cache by configured rows, and dequantized output by the current batch/window. It never maps or allocates the whole table.

## OOM prevention

### Known failure

The previous attempt did not fail because MXFP4 required the full checkpoint. The stock loader created lazy arrays for the approximately 104 GiB BF16 checkpoint and evaluated them only at the end; macOS jetsam killed the process while swap headroom was thin.

The working pattern is preserved in `/Users/kearm/mlx-lm/research/qwen38-flash/verify/smoke_4bit.py`.

### Hard loader invariants

1. Parse safetensors headers before allocating weights.
2. Build the quantized module tree from header names, shapes, and dtypes.
3. Load one shard at a time.
4. Evaluate every array in that shard immediately.
5. Record loaded names.
6. Release the shard dictionary before the next shard.
7. Clear or strictly bound the MLX cache.
8. Reject missing tensors before generation.
9. Report active and peak memory after every shard.
10. Reject unexpected checkpoint keys after normalization.
11. Validate manifest sizes and hashes before allocation and again after load.

The architecture forbids accumulating `weights.update(mx.load(shard))` across all shards followed by one final `mx.eval(model.parameters())`.

The M4 Max has 128 GiB RAM. The first measured gate keeps active model memory below 72 GiB and records process RSS, MLX active/cache/peak memory, and OS pressure after every shard. This is intentionally conservative; the wired limit is increased only from measured headroom after streaming remains stable.

### B550 global-memory protection

The prior B550 OOM was global compiler fan-out: about 320 `cicc` processes exhausted 91 GiB RAM plus swap. It was not CUDA quantizer VRAM exhaustion.

Every Stage A job uses:

```bash
systemd-run --user --scope \
  -p MemoryHigh=20G -p MemoryMax=28G \
  -p MemorySwapMax=0 -p TasksMax=256 \
  env MAX_JOBS=2 ...
```

Additional gates are no compiler fan-out, one campaign process, per-GPU budgets derived from live free VRAM, outputs on the data volume, at least 45 GiB available host memory, and no launch while all swap is committed unless co-scheduling is explicit.

At documentation time, B550 has one active GPU evaluation using 6.8 GiB on GPU 0, GPU 1 is idle, 80.5 GiB host memory is available, and all 8 GiB swap is committed. Stage A remains unlaunched until swap pressure is understood or explicitly accepted.

## Quantization pipeline

Use `/Users/kearm/research/ml/RESMP-DEV/mlx-gptq-mxfp`, committed head `7131c407af8b8991fb5a8148a49cf2f323c7afd3`. It has unrelated dirty work; preserve it.

Proven properties include standard MLX affine/MXFP4/MXFP8 grids, native group size 32, sequential GPTQ within scale groups, E8M0 scale search, routed-expert capture, resumable layer artifacts, and bit-exact packing against live `mlx.core`.

### Stage A on B550

Stage A calibrates resident backbone linears and experts only. It uses a meta skeleton and streams one layer at a time; it never fully loads the BF16 checkpoint. The first artifact omits MTP. Current qwen4-exp sanitization drops `mtp.*`; M6 introduces a separately indexed draft artifact rather than leaving ambiguous uncalibrated tensors in the main index.

Required Flash-Next adaptations:

1. Validate qwen4-exp traversal by `CaptureManager`.
2. Add multimodal/text-wrapper namespace mappings.
3. Exclude PLE table tensors from ordinary linear calibration.
4. Capture GDN/QSA/PLE post-processing projections as linear modules.
5. Preserve routed activations for all experts and record any fallback.
6. Emit a manifest mapping every consumed checkpoint tensor to an artifact.

A representative command shape, not a launch authorization, is:

```bash
python -m mlx_gptq.calibrate \
  --model /path/Qwen3.8-Flash-Next-BF16 \
  --output /data/calib/qwen38-flash-mxfp4 \
  --calibration-tokens /path/qwen38-flash.npy \
  --mode mxfp4 --mxfp-algorithm safe-scale-search \
  --nsamples 256 --seqlen 2048 --batch-size 1 \
  --layers-attr model.layers \
  --checkpoint-model-prefix model.language_model \
  --artifact-layers-prefix language_model.model.layers \
  --devices cuda:0,cuda:1 --vram-gb 14,14
```

Batch size begins at one because this exact heterogeneous architecture has not yet passed Stage A smoke.

### PLE quantization

The first PLE policy is streaming row-local MXFP4 on B550, not the backbone's dense sequential GPTQ solve:

1. Stream each raw PLE shard.
2. Quantize each row independently with E8M0 scale search.
3. Write codes and scales directly to the two-plane artifact.
4. Hash each completed plane and resume at shard boundaries.
5. Retain a small BF16 row sample for parity.

This dynamic row-local packing is no longer the default final-storage decision. Calibration must read exact selected BF16 source rows, and the source BF16 shards must be retained. A full BF16 table is 102,400,491,520 bytes; native MXFP4 planes are 27,200,130,560 bytes. Both the 320-byte BF16 row and 85-byte MXFP4 row fit inside one 16 KiB page, so random row access may be page-traffic-bound rather than logical-byte-bound.

The serving default is now native BF16-on-disk until a same-index Optane benchmark on a real trace proves that MXFP4 PLE planes improve end-to-end token latency without failing the PLE quality gate. Any later MXFP4 PLE conversion remains reversible from the retained BF16 shards.

The operational media plan is specified separately in
`docs/architecture/local-ple-storage-tiers.md`: serve first from exact BF16 PLE
source shards on the internal SSD, then move those immutable shards to Optane
and compare the same real trace. That document owns the storage-tier contracts
and promotion gates.

### Stage B on Mac

Stage B is a streaming rewrite:

1. Read headers.
2. Build the final quantized module tree.
3. Allocate final tensors only.
4. Inject each B550 artifact exactly once.
5. Write one output shard at a time.
6. Verify no artifact is unconsumed.
7. Run tiny-reference parity before and after external PLE replacement.

It must not first create a complete RTN intermediate artifact.

## Implementation milestones

### M0: isolate the existing port

Pin or import qwen4-exp head `e396b18`, rerun the tiny-reference logits test, and keep resident `_ShardedEmbedding` as control.

Acceptance: all existing tiny variants remain within their recorded tolerances.

Live M0 recheck on 2026-10-05 passed all six variants (`tiny-qwen4exp`, GDN, attention, PLE, no-PLE, and spelling alias) and all three prompts per variant. Maximum absolute logit difference was `1.192e-07`, below the `2e-4` gate. Receipt: `build/reports/strata-mxfp4-workers/qwen4-exp-tiny-parity.log`.

### M1: external PLE reader

Implement the two-plane reader, synthetic row parity, and a tiny-model deterministic table. Replace `_ShardedEmbedding` behind a runtime flag.

Acceptance: exact row order, unchanged EOS-reset behavior, equality with a resident MXFP4 control built from the exact served `uint32[N, 20]` codes and `uint8[N, 5]` scales, and less than 2 GiB additional peak memory including ticket objects and host RSS. The BF16 tiny-reference tolerance remains the M0 gate, not the quantized-row M1 gate.

### M2: streaming artifact loader

Move `smoke_4bit.py` from research code into a reusable loader with per-shard receipts and resume support.

Acceptance: a real 4-bit artifact loads without jetsam and reports active/peak memory after each shard.

### M3: MXFP4 artifact generation

Add qwen4-exp support to the B550 campaign, run a tiny Stage A smoke under the cgroup, pack one PLE shard, and stream-pack a synthetic full-model artifact.

Acceptance: bit-exact MXFP4 planes, complete artifact consumption, no unconsumed calibration tensor, and no host/GPU OOM.

### M4: real model smoke

Generate the same fixed prompt twice on the full artifact. Require finite logits, no NaN or infinity, deterministic logits within the declared tolerance, bounded resident/peak/RSS, no whole-table allocation, and captured PLE wait/cache metrics. Comparison with the existing affine 4-bit checkpoint is contextual because the quantization formats differ.

Acceptance remains output-relative until a BF16 or FP8 reference path is available on CUDA.

### M5: performance and fusion

Only after M0-M4, fuse selected-row dequantization with `key_proj`/`value_proj`, batch PLE reads across draft tokens, evaluate a custom E2M1/E8M0 gather-dequant Metal kernel, and consider byte-sized expert caching.

### M6: MTP

Extract `mtp.*` into a separately loadable draft model first. Preserve rollback for QSA indexer state, GDN state, and PLE short-conv/n-gram history before changing the generate loop.

## Receipts

Each phase writes `build/reports/<phase>/receipt.json` with Git head and dirty flag, artifact hashes, commands and exit codes, active/peak/cache memory, B550 cgroup and VRAM telemetry, PLE page/cache/latency statistics, and explicit non-claims.

Initial worker timeout receipts are retained in `build/reports/strata-mxfp4-workers/`. They document incomplete fan-out and are not acceptance evidence.

## Current non-claims

- No full Flash-Next MXFP4 artifact exists.
- No qwen4-exp Stage A calibration smoke has run.
- B550 currently has one GPU evaluation and completely committed swap; launching now would violate the conservative memory gate.
- P905 performance used synthetic random IDs, not a real Qwen hash trace.
- No model-quality comparison has run.
- Vision, MTP, fused PLE Metal kernels, and adaptive expert caching remain unimplemented.

## Recheck addendum, later on 2026-10-05

### Why prior GPTQ-MXFP4 did not produce FlashNext

The MXFP4 GPTQ packing itself is proven on other architectures, including the durable LFM2.5 MXFP4 artifact at `/Users/kearm/models/LFM2.5-2.6B-2604ec2-MLX-GPTQ-MXFP4`. It was not merely reusable unchanged for FlashNext:

1. The quantizer checkout is pinned at `7131c407af8b8991fb5a8148a49cf2f323c7afd3`, but its active virtual environment has Transformers 5.14.1. `qwen4_exp` support was verified in Transformers 5.17.0; the older environment rejects the model type.
2. `materialize_module` fills every parameter in a decoder layer. Real layer 1 includes the complete 102.4 GB BF16 PLE table, so the existing layer-streaming Stage A still OOMs on that layer. PLE must be split into its own row-plane packer before backbone calibration.
3. A normal non-PLE FlashNext layer already has approximately 5.03 GB of BF16 expert weights. The current 8 GB Strix card can host a tiny smoke and possibly a carefully chunked single-layer experiment, but it is not the host for the unmodified full-source campaign. B550's 24 GB cards are the first real-layer target after PLE is excluded.
4. The official 360 GB BF16 source is absent locally and no qwen4-exp Stage A artifacts exist.
5. The local community 4-bit artifact is a diagnostic affine control, not our desired GPTQ-MXFP4 artifact and not a durable model target.

The target artifact is owned by this campaign: B550 GPTQ-calibrated native MXFP4 for resident backbone tensors, our row-local MXFP4 PLE planes, our manifest and hashes, and our streaming Mac packer. The community checkpoint may be used only for loader/layout comparison until that artifact exists.

### Additional real-model correctness gap

The available real snapshot enables `mrope_interleaved: true` and `mrope_section: [11, 11, 10]`. The qwen4-exp MLX-LM port currently reads only `rope_theta` and `partial_rotary_factor` and applies conventional partial rotation. The tiny reference generator does not set the MRoPE fields, so its six passing variants do not cover this snapshot path.

Interleaved MRoPE must be implemented or explicitly disabled for a text-only artifact with a parity receipt before a durable real model is claimed.

### Live compute state

- Strix (`ssh strix`, 192.168.1.182): reachable, RTX 5070 Laptop, 8,151 MiB total, 7,707 MiB free, 91 GiB RAM, 3.4 TB disk. It has no quantizer checkout, model cache, or Python ML environment yet.
- B550: both RTX 3090 Ti GPUs are effectively idle, with 76 GiB memory available, but all 8 GiB swap remains committed. It remains the better real-layer host after the PLE split and cgroup gate.

### Focused worker outcome

The broad quantizer worker timed out again and is not evidence. The focused qwen4-exp state worker completed and found the MRoPE gap plus the absent real-checkpoint load receipt. The focused Strix planning worker completed with a bounded tiny-model smoke design. Future delegation must be one narrow question per worker; no worker should own the complete architecture.

## Dynamic MXFP4 addendum, later on 2026-10-05

Dynamic quantization now supports native `mxfp4`, `mxfp8`, and `nvfp4` modes in
addition to affine mode. The `--mxfp4` preset uses MXFP4/group 32 as baseline and
MXFP8/group 32 for sensitive layers, emits per-layer mode overrides, and skips
leaves whose alignment cannot satisfy the selected native grids. This is the MLX
orchestration change required before GPTQ calibration; it does not replace the
campaign-owned GPTQ solve.

The tiny qwen4-exp path also required two model fixes. Gradients are stopped on
QSA block and MoE expert indices so calibrated sensitivities do not request VJPs
through discrete gathers. Exact int64 n-gram ID construction moved host-side so
Strix's Linux CPU MLX backend can execute the path; the embedding tensors and all
neural work remain in MLX.

The change is present locally at
`/Users/kearm/mlx-lm/.worktrees/dynamic-mxfp4` and on Strix at
`/home/kearm/mlx-dynamic-mxfp4`. Final patch SHA-256:
`80a9b1c83fb3cda96ccdc69f6b18bce2ca3c61f2fd0ae78636648ae9d3a760b3`.

Mac and Strix each passed five dynamic tests and the full tiny-qwen4-exp
dynamic CLI. Both saved models contain 59 MXFP4 modules and one MXFP8 override;
both reload with finite float32 logits. The Mac run peaked at 0.627 GiB MLX
memory, and the modified model passed the existing dense, sparse, and EOS-reset
BF16 tiny-reference parity with maximum absolute difference `1.043e-07`. The
Strix run completed under a 3 GiB cgroup at 0.815 GiB MLX peak and
1,724,260 KiB RSS with zero swaps. The consolidated receipt is
`build/reports/dynamic-mxfp4/receipt.json`.

This does not advance the real-model non-claims: the run used the tiny resident
PLE control, not the external Optane planes; it was RTN sensitivity selection,
not GPTQ against the official BF16 source; and interleaved MRoPE remains
unimplemented. The full qwen4-exp model harness also retains a separate Strix
Linux CPU float16 `GatherMM` limitation even though its float32 CLI and reload
pass.

## Tiny GPTQ-MXFP4 artifact addendum, later on 2026-10-05

The isolated GPTQ branch now executes the complete qwen4-exp Stage A/B path on
the tiny control. Stage A uses the Transformers fused PLE representation for
model loading but excludes the entire PLE subtree, routers, gates, and
shared-expert gates from calibration. Stage B uses the matching sharded-PLE MLX
checkpoint for conversion and injects every Stage A artifact into a standard
MLX MXFP4/group-32 model.

The resulting campaign-owned tiny checkpoint is durable on both machines:

- Mac: `/Users/kearm/.cache/mlx-lm/dynamic-mxfp4/runs/gptq-stage-b-tiny-20261005T4/mlx_model`
- Strix: `/home/kearm/mlx-dynamic-mxfp4/artifacts/tiny-qwen4exp-gptq-mxfp4`
- Weight SHA-256: `751758990f4af3813e34006b28c9cb3a53f69c5e4f6d041aefffc620775c36cd`

Mac Stage A completed four layers and 144 tensor records with zero PLE tensors.
Stage B consumed all records, injected 60 packed tensors, retained one expected
out-of-layer RTN tensor, and produced a model that reloads and generates. Strix's
RTX 5070 completed the same Stage A under a 3 GiB cgroup with 290 MiB sampled
GPU peak and zero swaps. The source fix moves concrete buffers to the forward
device because Qwen RoPE buffers otherwise remain on CPU during CUDA execution.

This promotes M3 from unstarted to tiny-control exercised. It does not promote
the real-model milestones: PLE is still resident, the fixture is tiny, the
calibration set is only 3 by 64 tokens, official BF16 weights are absent,
interleaved MRoPE remains unresolved, and B550 has not run.

## BF16 source transfer addendum, later on 2026-10-05

The official BF16 source is being pulled to both execution machines from
immutable revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`:

- Mac: `/Users/kearm/models/Qwen3.8-Flash-Next-BF16`
- Strix: `/home/kearm/models/Qwen3.8-Flash-Next-BF16`

The expected tree is 144 files and 360,023,351,514 bytes. Both downloads use
detached tmux session `qwen-bf16`, explicit `hfd.sh` wrappers over hfd 0.1.1
plus aria2c, 12 jobs, 12 connections per file, resume, and configured full hash
verification. Strix was switched from stalled Xet staging to this route after
its worker stopped making file progress. The in-progress receipt is
`build/reports/bf16-transfer/receipt.json`.

This changes the prior source-availability boundary: the BF16 transfer has begun
but is not complete, and full hash verification remains pending. No real-model
calibration claim is made until the verified tree is available.

## Real Stage A readiness addendum, later on 2026-10-05

Stage A is now ready for a verified real source tree. `--lazy-ple` replaces the
fused `320,001,536 x 160` PLE parameter with selected-row reads from official
shard tensors, so layer 1 no longer needs resident PLE memory. The real index
maps PLE shards across files 5 through 37. A tiny sharded-source run produced
bit-identical layer artifacts to the fused control with zero PLE artifacts.

The implementation passed 34 tests and bounded CUDA smoke runs on the RTX 5070
and both B550 RTX 3090 Ti GPUs. B550 now has the cloned Python 3.12,
Transformers 5.17, Torch 2.14.1 CUDA environment. A `qwen-to-b550` tmux watcher
waits for full Strix verification and then copies the tree directly over LAN.

The staged calibration input is 32 by 2,048 tokens from the pinned model-agnostic
PTQ corpus. Its current tokenizer copy comes from the diagnostic conversion;
publication tokens must be regenerated after the official tokenizer is verified.
Consolidated receipt:
`build/reports/gptq-real-readiness/receipt.json`.

The launch chain is automatic after source verification: Strix copies the
verified tree directly to B550, then starts detached session `qwen-stage-a`.
That run uses both B550 GPUs with 18 GiB budgets, a 68 GiB host-memory cgroup,
native MXFP4, and lazy PLE source reads.

This does not make the final artifact ready: streaming full-model Stage B,
external Optane MXFP4 PLE planes, interleaved MRoPE parity, and real-model
quality acceptance remain open.

## Real source and Stage A live addendum, 2026-10-07

The full pinned BF16 source is now fully hash-verified on both Mac and B550:
144/144 files and 360,023,351,514 bytes on each. Strix was unreachable during
the check and has no current completion claim.

Real Stage A is running on B550 with both RTX 3090 Ti GPUs, 32 by 2,048
calibration tokens, native MXFP4, and exact selected-row BF16 `--lazy-ple`.
Layers 0, 1, and 2 completed; layer 3 was running at the receipt. Layer 1
completed in 275.2 seconds and proves the real PLE path without materializing
the fused 95.37 GiB table.

The live receipt is
`build/reports/gptq-real-stage-a-live/receipt.json`. Stage A is not complete,
and final Stage B, MRoPE parity, and quality gates remain open.

## Real campaign-owned serving addendum, 2026-10-07

Real Stage A completed all 48 layers on B550: 74,400 GPTQ tensor records,
native MXFP4/group 32, exact selected-row BF16 PLE, exit 0, 74.7 minutes, and
64,194,932 KiB sampled peak RSS with zero swaps. Stage B completed 34 shards
and 2,160 tensors. The resident backbone is `171,213,918,824` model bytes;
PLE, embedding, and LM head remain exact BF16.

The documented text-only MRoPE blocker is closed. With the real configuration,
HF expands identical text positions across all MRoPE channels. HF interleaved
MRoPE and the current partial-RoPE path differ only by BF16 rounding: maximum
cosine delta about `0.00195` and maximum sine delta about `0.00189`. This does
not claim vision support.

The external exact-BF16 PLE backend is implemented in the `dynamic-mxfp4`
MLX-LM worktree. It parses safetensors byte offsets, reads selected rows only,
deduplicates requests, coalesces contiguous rows, uses a bounded thread pool
and global-row LRU cache, and records gather/cache/read metrics. The runtime
builder emits a 13-file loader artifact, prunes mixed shard 13, preserves
tokenizer metadata, and omits full-artifact hash manifests from the runtime.

The complete packed artifact transferred to the Mac and matched all 34 shard
SHA-256 hashes in 63.28 seconds. The local runtime contains 2,032 backbone
keys and `68,813,407,880` model bytes. Its 84 rewritten mixed-shard tensors
match source byte ranges exactly.

Two identical real fixed-prompt Metal runs completed 32/32 greedy tokens each
with byte-identical text. Resident parameters were `68,813,152,280` bytes; peak
process RSS was `46.80 GB` and `47.25 GB`. Token rates were `1.130` and
`1.433` tokens/s. Across the two runs PLE gathered 672 requested rows, hit 144
cache rows, read 528 unique rows, read `168,960` bytes per run, and spent only
`29.03 ms` and `27.29 ms` cumulative gather time. This proves correctness and
bounded residency, but PLE storage media is not the current bottleneck and this
is not a tuned throughput result.

The consolidated implementation and smoke receipt is
`build/reports/local-ple-storage/external-ple-implementation-20261007.json`.
Remaining open work is kernel/decode throughput optimization, real quality
evaluation, and same-trace Optane media comparison.

## Warmed decode optimization addendum, 2026-10-07

A real Xcode Instruments Metal System Trace completed with exit 0 and exported
its table of contents. The 199 MiB trace had Shader Timeline disabled, so it
does not identify kernels by name. Independent timing through 257 tokens showed
a flat steady-state p50 of `27.248` to `27.990` ms in every 32-token bucket,
which ruled out sequence-length growth and established that trace overhead had
inflated apparent 73-87 ms frames.

Warmup now exercises both the prompt-sized and single-token forwards through a
real prompt cache, resets cache state, and clears PLE rows before measurement.
Focused lint, nine unit tests, and the real smoke exited 0. First-token latency
fell from `866.218` ms to `169.544` ms with the same greedy text. Decode p50 is
`28.062` ms and p95 is `28.598` ms.

An isolated trial quantized the BF16 LM head to naive native MXFP4/group 32.
It reduced resident parameters by `933,683,200` bytes and decode p50 to
`25.828` ms, but short-context logits had only `40.0%` top-1 agreement with
BF16 and a `31.281` maximum absolute delta. The trial is rejected as a default;
promotion requires a B550-calibrated output-head solve and the same quality
gate.

## Performance hill-climb addendum, later on 2026-10-07

The real request warmup now performs a one-token `generate_step`, then resets
the prompt cache and PLE row cache. This exercises the exact asynchronous
generation path. A fresh logical request now has `120.220` ms first-token
latency, `27.171` ms decode p50, `27.893` ms p95, and `0.965` s generation for
32 tokens. The multi-second cost is one-time process warmup, not steady server
request latency. Repeated requests in one process measured about `120` ms.

Per-layer decode profiling found 47 ordinary layers at roughly `0.70` ms, the
PLE layer at `2.683` ms, and mixer plus BF16 LM head at `2.902` ms. Exact
SwitchGLU gate/up MXFP4 fusion is retained: it is bit-identical and reduced
single-stream decode p50 from `28.062` to `27.769` ms before warmup repair.
Optional decode-shape `mx.compile`, Metal-side n-gram hashing, a 1M-row PLE
cache, deeper PLE admission, and a 100 GiB Metal wired limit were measured and
did not produce material end-to-end wins.

Batched native decode scales aggregate throughput through expert-row reuse.
The durable benchmark measured `33.923` tokens/s at batch 1, `117.519` at
batch 8, and `210.792` at batch 32. Per-sequence throughput falls to `6.587`
tokens/s at batch 32, so batch 8 is the current latency/throughput knee. At
batch 32, 6,873 of 20,992 requested PLE rows were cache hits.

B550 replayed the completed 48-layer Stage A artifacts to emit
`32 x 2048 x 10240` final residual activations in 1,669.006 seconds. The exact
top-level hyper-connection mixer then streamed those to `65,536 x 2560` head
activations in 6.088 seconds. A calibrated MXFP4 head solved in 51.789 seconds
with 0.120809 median relative weight error; MXFP8 solved in 13.246 seconds with
0.027915 median error. MXFP4 reduced decode p50 to about 28.01 ms and MXFP8 to
26.347 ms, but both changed the exact continuation near a step-10 near tie and
are rejected for the exact-output artifact. The BF16 head remains the default.

Optane was checked live and Thunderbolt reported no external device connected;
no Optane mount or benchmark is claimed.

## Real PLE trace and cache tuning addendum, 2026-10-07

The batch benchmark can now capture exact PLE row IDs without changing model
execution. A B=64, 32-token trace contains 41,984 requested rows, 14,960 unique
rows, and 32 tickets. Replaying it across cache capacities shows that 16,384
rows is the smallest capacity at the direct-storage plateau; all capacities
return the same value hash. The campaign runtimes on Mac and B550 now use this
setting. Repeated exact-output model runs favor 16K over 4K for both
single-stream p50 and B=64 aggregate throughput.

## Auto hash-backend addendum, later on 2026-10-07

After global PLE deduplication, Metal-side n-gram hashing improved repeated
single-stream decode p50 from 27.233 to 26.594 ms while keeping the same text.
It remains slower for large batches, so `hash_backend=auto` is now the campaign
default: batch size 1 uses Metal hashing and larger batches retain host hashing.
The runtime builder emits this default. The canonical 64-token run used the
auto policy, preserved exact output, and reached 26.476 ms decode p50.

## Shared-expert fusion addendum, later on 2026-10-07

The shared expert's packed MXFP4 gate and up projections now fuse into one
quantized matmul at load time. The transformation is bit-identical at layer and
model scope. All 48 shared experts fuse. Three exact 64-token runs reached a
26.104 ms decode-p50 median and 26.542 ms p95, with unchanged output from the
prior auto-hash control.

## Attention key/value fusion addendum, later on 2026-10-07

Full-attention K and V packed MXFP4 projections now fuse into one quantized
matmul in all 12 full-attention layers. K and V remain bit-identical. The
isolated layer saves 0.0086 ms, and three exact 64-token runs improved decode
p50 median from 26.104 to 25.951 ms with unchanged output.

## GDN projection-fusion addendum, later on 2026-10-07

GDN QKV+Z and B+A packed MXFP4 projections now fuse in all 36 linear-attention
layers. Outputs are bit-identical, and exact 64-token decode p50 improved from
25.951 to 25.900 ms. A later Q+K/V full-attention fusion was exact and faster in
isolation but did not improve real p50, so K/V-only fusion remains. The current
post-revert exact control is about 25.9 ms p50.

## Greedy fast-path addendum, later on 2026-10-07

`generate_step` now supports `greedy_fast=True`. It performs argmax on raw logits
and skips log-sum-exp normalization while preserving the exact greedy token
stream. The normalized-log-probability API remains the default; temperature-zero
CLI generation uses the fast path automatically. Three exact real 64-token runs
improved decode p50 median from 25.900 to 25.814 ms. A vectorized Metal hash and
mixed BF16/MXFP8 head were exact but slower end to end and remain rejected.

## Compiled decode addendum, later on 2026-10-07

With all projection fusions active, compiling each SparseMoeBlock now improves
exact batch-1 decode. The compiled wrapper is guarded to batch size one and
sequence length one, leaving prefill and batched execution eager. Mac and B550
campaign configs set `compile_moe=true`. Three exact 64-token runs improved p50
median from 25.896 to 25.336 ms and p95 to 25.912 ms.

## Compiled residual addendum, later on 2026-10-07

The 96 per-layer `GatedResidual` mixers and the final hyper-connection mixer now
compile for batch-1, sequence-length-1 decode. Larger batches and prefill remain
eager. The Mac campaign config sets `compile_residual=true`; B550's packed source
has the same implementation available for its next config refresh. Three
interleaved exact runs improved p50 in every pair, with mean paired savings of
0.381 ms p50 and 0.542 ms p95 over the `compile_moe` control. A durable-config
run reached 24.878 ms p50, 25.337 ms p95, 114.917 ms first token, and unchanged
64-token text. No batched throughput claim is made because the compile guard
leaves batch execution eager. Optane remains disconnected, so all PLE storage
results still use the internal SSD.

A later GDN projection/output compilation probe was exact but rejected and
reverted. Its median p50 improved by only 0.142 ms and median p95 regressed by
0.105 ms, with inconsistent paired wins. The recurrence and cache path were
never compiled.

A PLE post-projection compile probe was also exact but rejected and reverted:
paired p50 gains stayed below 0.13 ms, p95 regressed in two pairs, and one
candidate warmup reached 10.502 seconds. The external BF16 lookup and short
convolution cache path remained eager.

Full-attention projection/output compilation likewise improved every p50 pair
but was rejected and reverted after one p95 regression and an 11.769 second
warmup outlier. QSA, RoPE, cache updates, and SDPA were not compiled.

The final accepted source passed the 20-test focused matrix and reproduced the
same continuation at 24.971 ms p50 after all reverts. B550 holds the final source
bundle and a runtime config with only `compile_moe` and `compile_residual`
enabled. Thunderbolt still reports no external Optane device connected.

## Pre-fused MoE artifact addendum, 2026-10-07

Profiling showed that runtime gate/up fusion, not Python decode dispatch or PLE
shutdown, dominated cold model preparation. Removing the redundant per-layer MLX
cache clear did not help. The durable fix is a `prefused_moe` artifact contract:
`SwitchGLU` omits `up_proj`, sanitization maps explicit `gate_up_proj` tensors,
and the loader skips the large concatenation.

The tiny control has bit-identical logits. The real 13-shard Mac runtime was
rebuilt with all 48 routed-expert gate/up pairs pre-fused; every rewritten shard
hash verifies. Three interleaved real runs preserved the exact continuation and
reduced median cold-process load, parameter evaluation, and forward warmup from
23.907 to 12.195 seconds. Parameter evaluation rises because freshly mapped
prefused tensors reach Metal there, but load and forward warmup fall by more.
No decode p50, p95, or batch throughput improvement is claimed.

Final verification passed the 21-test focused matrix and reproduced the exact
64-token continuation from the pre-fused artifact. B550 holds the matching
source bundle while retaining its split-layout runtime, which remains valid and
does not set `prefused_moe`.

The pre-fused artifact uses its own `prefused-runtime.sha256.json` manifest; the
inherited split-runtime sidecar was removed so no stale hashes remain beside the
rewritten shards.

Paired B8/B32/B64 batch runs found no material throughput or latency difference
between split and pre-fused layouts. An interleaved packed-GDN lane layout was
also exact and faster in isolation, but full-model p50 did not improve and
warmup regressed consistently, so it was rejected and reverted.

## Compiled block-injection addendum, later on 2026-10-07

The 96 per-token block-injection combines now share one compiled function,
used only for batch-1 sequence-1 decode; a single module-level capture serves
every call site instead of one capture per layer. Three interleaved real
64-token runs improved decode p50 median from 24.960 to 24.849 ms and p95
median from 25.480 to 25.260 ms, with unchanged output and lower warmup.
`text_config.compile_injection` is promoted in both Mac artifacts.

Storage probes on the internal SSD rejected page-coalesced `pread`, `mmap` as
the default despite an isolated replay win, and macOS `F_NOCACHE`. The captured
single-stream trace also shows that in-model PLE gather timing includes Metal
hash synchronization; direct hot replay of its 1,184 rows costs about 17 ms.
PLE storage remains exact and bounded but is not the dominant decode cost.

Two further exact probes were rejected and reverted: concatenated GDN Q/K RMS
normalization regressed paired decode latency, and projection-only attention
compilation lost the last two p50 pairs plus the last three p95 pairs when the
initial three-pair experiment was extended to five pairs.

A token-only compiled LM-head argmax path also regressed median p50 and was
reverted. Decomposing attention compilation in the other direction succeeded:
`compile_attention_output` improves seven-pair median p50 and p95 while keeping
exact output, and is promoted with the other decode compile flags.

The packed GDN QKV+Z and B+A projections can now cross-fuse into one MXFP4
gather. `fuse_gdn_all` is bit-identical, improves five-pair median p50 and p95,
and is promoted alongside the compile flags.

The GDN gated norm and output projection also share one compiled decode graph
across all linear-attention layers. It improves every paired p50 and six of
seven paired p95 values; median cold-process preparation changes by only
0.047 seconds.

Fusing the BF16 shared-expert gate into the router row was exact on the real
artifact but regressed all five paired p50 measurements, so it remains separate.

Compiling the combined GDN projection and split graph was likewise rejected:
after background interference cleared, every clean pair regressed p50 and p95.

A shared attention-output graph also failed to improve median p50 and won only
two of five p95 pairs, so per-layer attention-output captures remain.

Current B8/B32/B64 runs remain within noise of the earlier pre-fused-layout
campaign. Fusing the QSA indexer projection into attention K/V was likewise
rejected: its tiny decode gain did not justify a consistent warmup regression.

The durable artifact now also pre-fuses each GDN QKV/Z/B/A projection. Tiny
logits are exact, all rewritten real shards verify, and five interleaved real
runs improve both median p50 and p95 with unchanged output. The canonical
decode artifact uses `prefused_gdn=true` and `fuse_gdn_all=false`.

Bounded-gap PLE row coalescing was also tested and rejected: the real trace's
same-shard rows are separated by at least 7.1 MiB, so thresholds through
16 KiB cannot reduce the 1,184 reads.

Appending the shared expert as an extra routed switch expert is exact but
copies nearly the full expert tensors at load and regresses both warmup and
decode, so the shared expert remains separate.

The latest synchronized profile shows PLE's apparent in-model cost is dominated
by Metal synchronization: direct row replay issues in about 0.054 ms and
collects in about 0.114 ms. A shared residual graph also regressed every paired
p50 measurement, so per-layer residual captures remain.

Using only the indexer k rows below the sparse-attention budget is likewise
slower end to end than the optimized fused Q/K projection, so the full
projection remains active at every context length.

Fusing each full-attention Q/gate projection with its packed K/V projection at
load is bit-identical, but five interleaved real pairs left median p50
unchanged within 0.005 ms and regressed median p95 by 0.093 ms. The temporary
source flag and symlinked candidate artifact were removed. Receipt:
`build/reports/local-ple-storage/attention-input-fusion-rejected-20261007.json`.

Preallocating the QSA indexer keys in 256-token chunks is exact, but the
five-pair gate regressed median p50 by 0.012 ms while improving median p95 by
only 0.018 ms. The temporary cache flag and candidate config were removed.
Receipt:
`build/reports/local-ple-storage/chunked-indexer-cache-rejected-20261007.json`.

Compiling the complete prefused GDN pre-recursion lane, including convolution
and Q/K normalization, is bit-identical and improves median p95 by 0.139 ms.
It nevertheless regresses median p50 by 0.014 ms and leaves the recurrence and
output graph unchanged, so it was rejected and reverted. Receipt:
`build/reports/local-ple-storage/gdn-prefusion-compile-rejected-20261007.json`.

Enabling `MLX_METAL_FAST_SYNC` on the accepted artifact regressed decode p50
by 0.272 ms and p95 by 0.463 ms in a single paired pilot, so it was rejected
without the five-pair gate. The default Metal synchronization path remains.
Receipt: `build/reports/local-ple-storage/metal-fast-sync-rejected-20261007.json`.

Overriding SDPA reduction blocks did not improve the canonical short-context
decode. `MLX_SDPA_BLOCKS=32` regressed both pilot metrics; 128 blocks produced
inconsistent p95-only gains, and a clean second five-pair block regressed median
p50 by 0.009 ms. The hardware default of 64 remains. Receipt:
`build/reports/local-ple-storage/sdpa-blocks-rejected-20261007.json`.

Command-buffer operator budgets of 32, 64, and 100 all regressed both decode
p50 and p95 relative to the M4 Max default of 50 operations. The default
scheduler limits remain. Receipt:
`build/reports/local-ple-storage/metal-op-buffer-rejected-20261007.json`.

Command-buffer memory budgets of 25, 100, and 300 MB regressed both decode
metrics. A 200 MB pilot improved p50 by 0.146 ms but moved p95 by only 0.001 ms,
so it did not justify a five-pair run or replacing the 50 MB default. Receipt:
`build/reports/local-ple-storage/metal-mb-buffer-rejected-20261007.json`.

The 200 MB command-buffer memory budget is accepted only as a pre-launch
process environment setting: `MLX_MAX_MB_PER_BUFFER=200`. Two five-pair
external-launch blocks improved combined median p50 by 0.303 ms and median p95
by 0.221 ms, and the durable run reached 24.159 ms p50 and 24.775 ms p95 with
unchanged output. Mutating the variable inside Python after startup did not
reproduce the gain and is rejected; launch commands must set it before Python
and MLX initialize. Receipt:
`build/reports/local-ple-storage/metal-mb-buffer-launch-env-accepted-20261007.json`.

With the accepted 200 MB memory budget, operator budgets of 64 and 100 both
regressed p50 and p95. The canonical launch state remains 200 MB plus the
default 50-operation cap. Receipt:
`build/reports/local-ple-storage/metal-mb200-op-buffer-rejected-20261007.json`.

Bracketing the accepted command-buffer budget found 150 MB improves only p50
while 250 MB regresses both metrics, so the measured 200 MB launch setting is
retained. Receipt:
`build/reports/local-ple-storage/metal-mb-bracket-rejected-20261007.json`.

Narrowing the evaluation-tape BFS width to 10 improved median p50 by 0.097 ms
but regressed median p95 by 0.154 ms with large tails; width 40 drove p95 to
30.956 ms. The default width of 20 remains. Receipt:
`build/reports/local-ple-storage/mlx-bfs-width-rejected-20261007.json`.

The earlier fast-sync rejection used the misspelled variable
`MLX_METAL_FAST_SYNC`. A corrected probe using `MLX_METAL_FAST_SYNCH=1` with
the accepted 200 MB budget regressed p50 by 0.241 ms and p95 by 0.411 ms, so
fast synchronization remains disabled. Receipt:
`build/reports/local-ple-storage/metal-fast-synch-rejected-20261007.json`.

Capping the Metal buffer cache at 16 GiB produced a transient wrapper-path win,
but both durable model-config paths failed the p50 gate: constructor-time
integration regressed median p50 by 0.030 ms and post-load integration by
0.021 ms. Zero cache retention was much slower. The cap was rejected and
reverted; the platform allocator limit stays. Receipt:
`build/reports/local-ple-storage/metal-cache-16g-rejected-20261007.json`.

Raising the PLE row cache to 32,768 was rejected. The canonical trace retains
only 760 rows, so hit/miss behavior is unchanged; five pairs regressed median
p50 by 0.040 ms and median p95 by 0.039 ms. Retain 16,384 rows. Receipt:
`build/reports/local-ple-storage/ple-cache-32768-rejected-20261007.json`.

The durable PLE cache is now context-adaptive. Prompts of 2,048 tokens or
fewer retain the accepted 16,384-row cache; `generate_step` grows it to 32,768
rows before prefill when the full prompt exceeds that threshold, and prompt
reset restores the short-context capacity. On a real 3,200-token prompt, five
pairs improved median p50 by 0.096 ms and reduced disk reads by 864 rows in
every pair with exact output. Five short-context pairs also improved both
medians. Receipt:
`build/reports/local-ple-storage/adaptive-ple-cache-accepted-20261007.json`.

B550 now holds both the focused adaptive-cache source bundle and a complete
source-only validation bundle. The complete bundle hash verified remotely, and
CPU-only focused tests passed with 16 runs and 8 platform-specific skips. Its
split-layout runtime is intentionally unchanged, and no calibration or remote
inference is claimed. Receipt:
`build/reports/local-ple-storage/b550-adaptive-ple-cache-source-sync-20261007.json`.

The real 3,200-token trace justified rechecking PLE admission depth, but depth
16 and a prefill/decode phase adaptation were both rejected. Fixed depth 16
traded a 271.203 ms median first-token regression for small decode gains, while
phase adaptation still regressed median decode p50 by 0.112 ms and median
first-token latency by 456.652 ms. Depth 64 also regressed. The canonical
admission depth remains 32. Receipt:
`build/reports/local-ple-storage/long-context-inflight-rejected-20261007.json`.

Increasing `generate_step` prefill chunks to 1,024 or 2,048 tokens changed the
real greedy continuation and raised Metal peak memory. A separate QSA
complement-mask probe had exact block selection in a focused test but regressed
real p50 by 0.112 ms and p95 by 0.339 ms. Both were rejected and reverted; the
512-token prefill default and direct QSA top-k construction remain. Receipts:
`build/reports/local-ple-storage/long-prefill-chunks-rejected-20261007.json` and
`build/reports/local-ple-storage/qsa-complement-mask-rejected-20261007.json`.

The long-context SDPA block sweep also rejected 32, 128, and 256 reduction
blocks. Every alternative preserved text but regressed decode p50 and first-token
latency, so the hardware default of 64 remains. Receipt:
`build/reports/local-ple-storage/long-sdpa-blocks-rejected-20261007.json`.

A new synchronized long-context stage profiler shows QSA at 0.427 ms median
per full-attention call and attention overall at 0.797 ms. An exact incremental
QSA block-pool cache improved a clean second block's decode p50/p95 but repeatedly
regressed first-token latency; a decode-only build also failed. It was rejected
and reverted. Receipts:
`build/reports/local-ple-storage/long-context-stage-profile-20261007.json` and
`build/reports/local-ple-storage/incremental-qsa-blocks-rejected-20261007.json`.

A preallocated QSA block-pool cache was stepwise exact after repairing a
valid-count/capacity copy bug. Its repaired real pilot improved decode p50, but
five pairs regressed median p95 by 0.132 ms and median first-token latency by
233.382 ms. It was rejected and reverted; direct QSA pooling remains. Receipt:
`build/reports/local-ple-storage/preallocated-qsa-blocks-rejected-20261007.json`.

An unpadded single-token QSA fast path removed redundant visibility work and
was stepwise exact. Its real pilot improved p50 by only 0.082 ms while regressing
p95 by 0.028 ms and first-token latency by 53.404 ms, so it was rejected and
reverted. Receipt:
`build/reports/local-ple-storage/fast-qsa-decode-rejected-20261007.json`.

A tail-only QSA own-mask construction was stepwise exact but regressed real
decode p50, p95, and first-token latency in the pilot. It was rejected and
reverted. Receipt: `build/reports/local-ple-storage/fast-qsa-tail-rejected-20261007.json`.

A bit-exact batched-matmul QSA score contraction was faster in isolation but
failed the real-model gate. The clean second five-pair block regressed median
p95 by 0.488 ms and first-token latency by 531.847 ms. It was rejected and
reverted; einsum remains. Receipt:
`build/reports/local-ple-storage/matmul-qsa-scores-rejected-20261007.json`.

Forcing host PLE hashing on the real 3,200-token prompt preserved text but
regressed decode p50 by 1.012 ms, p95 by 1.350 ms, and first-token latency by
191.638 ms. Reported PLE elapsed time fell only because hashing moved into the
issue phase. `hash_backend=auto` remains correct for long single-stream prompts.
Receipt: `build/reports/local-ple-storage/long-host-hash-rejected-20261007.json`.

Padding QSA pooled blocks to a 1,024-block capacity and compiling the score
graph was bit-exact and faster in isolation. The real five-pair gate regressed
median p50 by 0.043 ms, p95 by 0.004 ms, and first-token latency by 97.603 ms.
It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/aligned-qsa-scores-rejected-20261007.json`.

At a real 8,192-token prompt the promoted 32,768-row long-context cache stays
best. Raising it to 65,536 rows cut 4,521 disk rows but regressed decode p50 by
0.194 ms and first token by 985.286 ms; 131,072 rows regressed p50 by 0.726 ms
and first token by 5,862.577 ms. The adaptive policy is confirmed at four times
the context depth where it was promoted. Receipt:
`build/reports/local-ple-storage/context-8192-cache-capacity-rejected-20261007.json`.

Bracketing the 8,192-token cache confirmed the 32,768-row policy. Capacities of
16,384, 24,576, and 40,960 rows all regressed decode p50 and p95; the two
non-power-of-two capacities also produced severe first-token outliers. Receipt:
`build/reports/local-ple-storage/context-8192-cache-bracket-rejected-20261007.json`.

The 8,192-token working set also justified rechecking PLE admission depth.
Depths 16 and 64 preserved the exact continuation and made the same 104,982
disk reads, but both regressed decode p50, decode p95, and first-token latency.
Depth 16 lost 1,044.842 ms of first-token latency; depth 64 lost 1,491.957 ms.
The canonical `max_inflight=32` remains. Receipt:
`build/reports/local-ple-storage/context-8192-inflight-rejected-20261007.json`.

A synchronized 8,192-token stage profile reconfirmed that PLE is not the leading
serialized stage: PLE median is 2.159 ms, while QSA, GDN, MoE, residual, and
injection medians remain substantial. The profile guides candidate selection;
its serialized totals are not steady-state decode latency. Receipt:
`build/reports/local-ple-storage/context-8192-stage-profile-20261008.json`.

An exact capacity-doubling raw QSA indexer-key cache was then tested because
the existing cache concatenates the full history on each decode step. Although
focused growth, trim, and replacement tests were bit-exact, the real candidate
regressed decode p50 by 0.487 ms, p95 by 0.360 ms, first token by 957.721 ms,
and peak RSS by about 433 MiB. It was rejected and the source restored
byte-for-byte to the B550-validated bundle. Receipt:
`build/reports/local-ple-storage/capacity-indexer-cache-rejected-20261008.json`.

An exact immutable QSA block-RoPE prefix cache was also tested. It matched the
direct masks after table growth and decode, but regressed p50 by 0.246 ms, p95
by 0.488 ms, first token by 914.016 ms, and RSS by about 1.29 GiB. It was
rejected and reverted for the same reason. Receipt:
`build/reports/local-ple-storage/cached-qsa-rope-rejected-20261008.json`.

A bounded exact full-context PLE result cache was then corrected onto the
synchronous path and reached 31 hits in 72 eligible decode contexts. Apparent
PLE elapsed time fell by 286.264 ms, but exact end-to-end p50 regressed by
0.389 ms and p95 by 0.561 ms. The PLE interval is synchronization-heavy; its
reduction shifts or increases other graph boundaries. The result cache was
rejected and reverted. Receipt:
`build/reports/local-ple-storage/ple-result-cache-rejected-20261008.json`.

An exact three-token decode specialization of the existing Metal n-gram hash was
also tested after an isolated 0.888 ms median gain. Its exhaustive focused test
passed, but the real model regressed p50 by 0.050 ms, p95 by 0.146 ms, and first
token by 1.895 ms. It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/specialized-decode-hash-rejected-20261008.json`.

Lazy PLE collection was exact in focused row comparison and slightly improved
first token, but decode p50 regressed by 0.049 ms and p95 by 0.146 ms. The
forced row evaluation remains the durable default. Receipt:
`build/reports/local-ple-storage/lazy-ple-collect-rejected-20261008.json`.

The least intrusive subset of the audited injection/residual fusion was then
tested: all 48 attention-output injections were fused into their same-layer MoE
residual graphs. The focused graph was bit-exact, but the real candidate
regressed decode p50 by 0.104 ms and p95 by 0.313 ms without a first-token gain.
It was rejected and reverted; separate compiled injection and residual graphs
remain canonical. Receipt:
`build/reports/local-ple-storage/fused-injection-residual-rejected-20261008.json`.

Upstream core MLX was then rechecked. `ArgPartition` uses the Metal sort path,
so upstream commit `0b37b0c7` was cherry-picked into an isolated build. Five
short-context pairs improved median p50 by 0.142 ms and median p95 by 0.210 ms
with exact output and five upstream sort tests passing. Five 8,192-token pairs
were variance-dominated: median p50 regressed by 0.194 ms while median p95
improved by 0.373 ms. The isolated build is retained for follow-up, but the
canonical MLX environment is unchanged. Receipt:
`build/reports/local-ple-storage/upstream-sort-contiguous-kernel-conditional-20261008.json`.

A follow-up 3,200-token gate resolved that ambiguity against promotion. All five
pairs were exact, but median p50 regressed by 0.057 ms, median p95 by 0.201 ms,
and median first token by 81.508 ms. The isolated build remains diagnostic only;
it must not replace the canonical multi-context core environment.

The final hyper-connection mixer was also compiled with the exact BF16 LM head.
The focused real-shape graph was bit-exact after repairing an initial
double-mixer defect, but the real candidate regressed decode p50 by 0.299 ms
while leaving p95 and first token effectively unchanged. It was rejected and
reverted. Receipt:
`build/reports/local-ple-storage/compiled-lm-mixer-rejected-20261008.json`.

A decode-only GDN short-convolution/cache Metal kernel was then tested. It was
bit-exact for both the four-tap output and shifted cache at 10,240 channels.
Five real pairs improved median p95 by 0.280 ms, but median p50 regressed by
0.052 ms and only one pair improved p50. It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/gdn-short-conv-kernel-rejected-20261008.json`.

The GDN kernel was also combined with the isolated upstream-sort build to test
whether their complementary short-context gains composed. Output stayed exact,
but p50 regressed by 0.032 ms and p95 by 0.028 ms. The combination was rejected
on pilot evidence. Receipt:
`build/reports/local-ple-storage/combined-sort-gdn-rejected-20261008.json`.

The depthwise Conv1D launch geometry was separately isolated in a disposable
core build. A 512-thread width was chosen from an exact geometry sweep, but the
real model left p50 effectively unchanged and regressed p95 by 0.181 ms. The
default 1,024-thread launch remains, and the disposable worktree was removed.
Receipt: `build/reports/local-ple-storage/depthwise-launch-512-rejected-20261008.json`.

A real 8,192-token PLE row trace was then captured without forward warmup. It
records 131,088 requested rows across 18 gathers and enables direct reader
replay without repeatedly executing the full model. Receipt:
`build/reports/local-ple-storage/context-8192-row-trace-capture-20261008.json`;
trace `/Users/kearm/models/ple-runtime-build-logs/context-8192-ple-trace-20261008.npz`.

An exact fast PLE issue path batched cache locking and vectorized shard
grouping. It matched all semantic stats and value hashes in direct replay and
reduced replay wall time by 27.7–42.6 ms, but the real 8,192-token model
regressed p50 by 0.392 ms, p95 by 0.322 ms, and first token by 1,055.635 ms.
It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/fast-ple-issue-rejected-20261008.json`.

A new Metal capture segmented steady decode into 101 repeated GPU commands per
token. It identified the BF16 LM head at about 2.678 ms and the packed GDN
recurrence at about 255 µs across 36 GDN layers. An exact T=1 specialization
of the packed GDN kernel reduced isolated recurrence time from 0.130458 ms to
0.098792 ms per layer. Five short pairs improved median p50 by 0.079 ms and
p95 by 0.298 ms; eight 3,200-token pairs improved median p50 by 0.084 ms and
p95 by 0.300 ms, all with exact output. Global use at 8,192 tokens failed the
p95 gate, so the canonical artifact enables T=1 only through 4,096 tokens and
falls back to the generic packed kernel above that boundary. Receipt:
`build/reports/local-ple-storage/gdn-packed-t1-threshold-accepted-20261008.json`.

The full source bundle containing the thresholded T=1 kernel was synchronized to
B550 and verified by SHA-256. Bounded CPU validation ran 17 tests with nine
platform-specific skips. B550 still had all 8 GiB of swap committed, so no GPU,
calibration, or inference work was launched there. Receipt:
`build/reports/local-ple-storage/b550-gdn-t1-source-sync-20261008.json`.

A tokenizer-exact 4,096-token boundary gate then ran four fresh interleaved
pairs. Output remained exact, but no pair improved p95; median p50 regressed by
0.036 ms and median p95 by 0.254 ms. A same-config A/A follow-up encountered
multi-second external spikes and was inconclusive, so the conservative direct
gate controls the decision. The canonical threshold was restored to the directly
measured 3,200-token boundary, and a restored 4,096-token fallback smoke matched
all eight gate outputs exactly. Receipts:
`build/reports/local-ple-storage/gdn-packed-t1-ctx4096-rejected-20261008.json`
and
`build/reports/local-ple-storage/gdn-packed-t1-restored-ctx4096-20261008.json`.

A fresh Metal System Trace captured the accepted T=1 runtime. Seven steady
decode segments each contained 101 GPU commands; their summed GPU time ranged
from 20.212 to 20.826 ms, and the exact BF16 head remained the dominant command
at 2.770 ms median. Both Metal System Trace and Game Performance exports left
shader-profiler intervals empty, so command-to-shader attribution remains
unavailable. Receipt:
`build/reports/local-ple-storage/gdn-t1-current-capture-20261008.json`.

Three follow-up GDN probes were rejected. Alternative launch geometries were
bitwise exact but slower than `(32,2,1)`. A float4 state variant was both slower
and not bitwise exact. A fused Q/K RMS-normalization kernel became bitwise exact
after correcting its epsilon, but five real pairs produced only +0.012 ms median
p50 and regressed median p95 by 0.058 ms. It was reverted. Receipts:
`build/reports/local-ple-storage/gdn-t1-geometry-sweep-20261008.json`,
`build/reports/local-ple-storage/gdn-t1-float4-sweep-20261008.json`, and
`build/reports/local-ple-storage/gdn-fused-qk-rejected-20261008.json`.

The 4,096-token PLE row trace was then replayed across cache capacities. A
65,536-row cache reduced direct replay decode work from 7.562 ms to 6.423 ms,
but a same-process eight-pair model gate regressed median p50 by 0.147 ms and
p95 by 0.531 ms. Output stayed exact. The canonical 32,768-row long-context
cache remains. Receipt:
`build/reports/local-ple-storage/ple-cache65536-ctx4096-rejected-20261008.json`.

Upstream MLX was re-audited through `6553c47ba`. The only relevant new candidate
was `a59cc2319`, an optional Metal residency refresh interval. In an isolated
base-plus-commit build, 1-second and 100-ms refresh intervals preserved exact
output but still left an approximately 687 ms first post-idle token; warmed
decode remained about 24 ms. The patch was rejected for canonical use. Receipts:
`build/reports/local-ple-storage/upstream-residency-audit-20261008.json` and
`build/reports/local-ple-storage/upstream-residency-refresh-rejected-20261008.json`.

A persistent prompt cache, one-second and 100-ms tiny Metal heartbeats, and the
upstream residency refresh all failed to remove the first post-idle spike. The
native sample profile showed the cost inside the first full model pass, not PLE
disk access or prompt-cache allocation. A bounded one-token background warm-up
after idle reduced the measured user-facing first token from 863.790 ms to
110.689 ms while preserving exact output and ~24.8 ms decode. The policy is now
wired into `mlx_lm.server` behind disabled-by-default
`--idle-warmup-seconds` and `--idle-warmup-prompt` flags. A live real-model HTTP
probe returned the same text in 0.228 seconds after proactive warm-up. Receipts:
`build/reports/local-ple-storage/idle-full-warmup-accepted-20261008.json`.
`build/reports/local-ple-storage/idle-warmup-server-implementation-20261008.json`.

The full source bundle containing the server warm-up was synchronized to B550
and verified by SHA-256. CPU validation passed 17 campaign tests with nine
GPU-only skips, and the focused server helper test passed with a no-op import
stub for the unavailable optional `requests` dependency. B550 swap remained
fully committed, so no GPU, inference, or calibration work was launched. Receipt:
`build/reports/local-ple-storage/b550-idle-warmup-source-sync-20261008.json`.

The idle-onset sweep then fixed the threshold question. After gaps of 1, 5, 15,
30, 60, and 90 seconds the first full model pass cost 1,065.920, 1,091.203,
1,009.949, 954.710, 923.969, and 919.610 ms, so the cold pass is already
expensive after one second and does not grow with idle length. A one-second
threshold is therefore the earliest point that captures the spike. Interleaved
traffic with proactive warm-ups showed no steady-decode cost: 24.378 ms warmed
versus 24.392 ms plain, with the keeper token costing 118-144 ms between
requests and exact output throughout. Receipt:
`build/reports/local-ple-storage/idle-warmup-threshold-and-no-degradation-20261008.json`.

A 4,096-token gate then bounded the warm-up to short-prompt serving. After a
60-second idle the long-context first token measured 6,647.014 ms without a
keeper and 6,448.527 ms with one, only 198.487 ms better, because external PLE
prefill dominates at that length. The flag is therefore a short-prompt serving
improvement and is not claimed to help long-context first tokens. Receipt:
`build/reports/local-ple-storage/idle-warmup-longctx-scope-limited-20261008.json`.

The T=1 threshold was then re-gated with the same-process method, which removes
the process-order bias of the earlier cross-process runs. At 4,096 tokens the
median p50 improved 0.086 ms and median p95 moved 0.002 ms; at 8,192 tokens the
median p50 improved 0.274 ms and median p95 moved -0.046 ms. Both contexts kept
exact output, but the isolated kernel saving of about 1.14 ms per decode step
never appeared at model scale, so the canonical threshold stays 3,200. This
receipt supersedes the variance-limited cross-process rejection. Receipts:
`build/reports/local-ple-storage/gdn-t1-threshold-same-process-20261008.json` and
`build/reports/local-ple-storage/gdn-t1-threshold-same-process-rejected-20261008.json`.

A chunked exact-argmax variant of the BF16 LM head was also probed on canonical
hidden and vocabulary shapes. It was slower than materializing full logits
(5.3499 ms versus 5.2883 ms at 65,536-token chunks) and did not reproduce the
reference index, so it was rejected without a model gate. Receipt:
`build/reports/local-ple-storage/lm-head-chunked-argmax-rejected-20261008.json`.

The PLE interval was then corrected. Measuring the real table directly on the
canonical 4,096-token trace gives 0.057 ms to issue and 0.123 ms to collect, or
0.183 ms per decode token, against the 4.278 ms the model reported. The
in-model interval wraps a lazy `mx.eval`, so it mostly measures the
synchronization of the surrounding graph boundary rather than read latency. This
explains why every PLE storage knob has produced near-null or negative results,
and it means future PLE work must gate on the issue and collect split. The
external exact-BF16 PLE path is not a decode bottleneck at these contexts.
Receipts: `build/reports/local-ple-storage/ple-interval-composition-20261008.json`
and `build/reports/local-ple-storage/ple-interval-correction-20261008.json`.

Decode attribution then reached the largest component. Short-circuiting every
MoE block removed 8.705 ms of a 24.582 ms step, and splitting that block showed
the shared expert is free at batch one while the routed experts carry 6.224 ms
per token. Applying router weights before the down projection is exact but
slower: median p50 worsened 0.075 ms short and 0.601 ms at 4,096 tokens, so the
weighting order is not the cost and a gather-level change is required. The
LM-head ablation arm was invalid because zero logits change downstream
behaviour, so it is excluded rather than reported. Receipts:
`build/reports/local-ple-storage/decode-wide-ablation-20261008.json`,
`build/reports/local-ple-storage/moe-split-probe-20261008.json`, and
`build/reports/local-ple-storage/moe-attribution-and-weight-before-down-rejected-20261008.json`.

Two routed-expert alternatives were then closed on the largest measured
component. Replacing `gather_qmm` with a gathered-weight batched matmul is
bit-identical but slower on both projections (0.260 versus 0.183 ms gate/up,
0.378 versus 0.242 ms down). An xctrace capture of the real projections shows
they run `mxfp4_gather_qmv` and its fast variant, because M=10 falls below
`get_qmv_batch_limit` for K=2560/N=1280 and K=640/N=2560, so the non-NAX
`gather_qmm` tile geometry is never used on this path. Call-level and
tile-level alternatives are therefore closed; further gains would require a
kernel change. Receipts:
`build/reports/local-ple-storage/routed-gather-qmm-confirmed-20261008.json` and
`build/reports/local-ple-storage/gather-qmv-routing-confirmed-20261008.json`.

The routed-expert cost was then shown to be a bandwidth problem with a known
gap. Each decode token streams 1.10 GiB of MXFP4 expert weights across 48 MoE
layers, an implied 189.5 GB/s, while measured device streaming reaches
434.9 GB/s, so the gather achieves about 44 percent of the achievable rate.
A wider `gather_qmv` tile was attempted in an isolated build and was 15 percent
faster in isolation (0.0156 versus 0.0184 ms gate/up, 0.0151 versus 0.0186 ms
down) but numerically unsound: the down projection diverged to about 2.1e38
because the fast kernel's output stride assumes the original tile width. The
change was reverted and the canonical MLX core is unchanged. Receipts:
`build/reports/local-ple-storage/gather-qmv-occupancy-20261008.json`,
`build/reports/local-ple-storage/moe-bandwidth-gap-20261008.json`, and
`build/reports/local-ple-storage/gather-qmv-wider-tile-rejected-20261008.json`.

The wide-tile idea was then retried coherently for MXFP4 only. The kernel,
launch width, JIT dependencies, and prebuilt `_r_8` symbols agree; affine
gather returns to its canonical unsuffixed r=4 symbols. Focused quantized
tests passed 34 cases plus 4,250 subtests, and full normalized float32 output
hashes matched for both real routed projections. Nevertheless, five
alternating cross-process model pairs per context preserved text but regressed:
short median p50 by 0.239 ms and 4,096-token median p50 by 0.519 ms, with no
context wins. The candidate stays in
`/Users/kearm/mlx/.worktrees/gather-qmv-tile` only as a diagnostic branch and
canonical MLX is unchanged. Receipt:
`build/reports/local-ple-storage/gather-qmv-wide-tile-interleaved-20261008.json`.

The opposite geometry was also measured. An exact MXFP4 r=2 fast gather, with
`bn=4` and matching prebuilt symbols, passed the focused quantized suite and
matched both full projection hashes. Three alternating real-model pairs per
context kept exact text, but short median p50 moved by only -0.014 ms and the
4,096-token gain was 0.066 ms with mixed wins. This is below the campaign
noise gate, so r=2 remains a diagnostic worktree and canonical r=4 is
unchanged. Receipt:
`build/reports/local-ple-storage/gather-qmv-narrow-tile-pilot-rejected-20261008.json`.

The FP4 arithmetic path was then bounded. Directly constructing IEEE-754
float bits initially corrupted the subnormal 0.5 code; after repairing that
special case, all 50 quantized tests plus 5,064 subtests passed and both real
projection hashes matched. The first real short-context pair nevertheless
regressed p50 from 24.427 to 35.864 ms, so the pilot stopped early. The
synthesized-half conversion remains canonical. Receipt:
`build/reports/local-ple-storage/fp4-direct-float-rejected-20261008.json`.

The accepted mlx-lm work was then cleaned and committed locally on the private
fork branch `dynamic-mxfp4`. The six commits cover native dynamic MXFP4/MXFP8
quantization, exact greedy fast generation, idle server warm-up, packed-T1 GDN,
the exact BF16 external-PLE Qwen4 runtime, and performance harnesses. Cleanup
made cached PLE rows own their buffers, validated the read backend before
executor creation, fixed batched prompt-cache reset, made the optional fixture
path portable, and added resident-versus-external greedy-ID equality. The clean
real smoke produced the exact continuation at 112.520 ms first token, 24.710 ms
decode p50, and 25.083 ms p95. Head `e628e0db395afa62273290ee9e6fe49b87d74654`;
nothing was pushed upstream. Receipt:
`build/reports/local-ple-storage/private-fork-cleanup-commit-20261008.json`.

## TPS targets and decode roofline, 2026-10-08

The performance goal is now explicit. Current warmed short decode is 24.710 ms
or 40.469 TPS. The measured read-only Metal bandwidth is 449.7 GB/s best and
429.1 GB/s median; the multi-pass `argmax` result is excluded as a bandwidth
ceiling. One optimistic short decode step must move about 4.854 GB: all common
parameters, ten selected experts' MXFP4 codes and scales, one embedding row,
16 external BF16 PLE rows, and recurrent-state read/write traffic. This gives
a 10.793 ms memory floor, or 92.653 TPS, so current decode operates at 43.7%
of the optimistic measured-memory roofline.

The target ladder is T1 42 short and 40 at 4,096 tokens; T2 50 short and 42 at
4,096; and T3 60 short and 45 at 4,096. Aggregate serving targets are T1
130/250/275 TPS at batch 8/32/64, T2 150/260/300, and T3 175/275/325. The
conservative measured-memory aggregate bounds are about 236/283/349 TPS.
Batch 32 is already at 76.2% of that bound, while batch 8 is at 49.3%, making
small-batch scheduling and expert reuse the strongest serving targets. The
routed expert stream itself has a 6.224 ms measured cost against a 2.787 ms
code-and-scale floor, but the larger absolute gap lies in the non-expert path.
Receipt and calculator:
`build/reports/local-ple-storage/decode-tps-goals-roofline-20261008.json` and
`build/reports/local-ple-storage/decode_tps_roofline.py`.

### Primary single-request ladder

The active priority is now single-request use, not aggregate batch throughput.
Three same-process rounds used nested real prompts at 10, 512, 2,048, 4,096,
8,192, and 10,841 tokens, generated 64 tokens per run, and preserved identical
generated IDs at every context. Median decode TPS was 39.270, 38.713, 35.772,
34.544, 34.186, and 34.043. Median TTFT was 114.930 ms, 1.009625 s, 3.994460
s, 8.094999 s, 18.543681 s, and 30.431235 s. Long single requests are therefore
TTFT-limited; decode remains comparatively flat.

The primary T1 ladder is 42/42/38/37/36/36 decode TPS, TTFT at most
100/850/3,400/7,000/15,000/26,000 ms, and end-to-end 64-token TPS of at least
40/27/13/7/4/2. T2 targets 50/48/44/42/40/40 decode TPS; T3 targets
60/58/55/50/48/48. The receipt records the corresponding TTFT and end-to-end
goals. Batch optimization is deferred until the single-request T1 ladder is
met.

The measured 8,192-token stage profile attributes 40.4 percent of synchronized
stage time to MoE, 25.9 percent to residual work, 11.0 percent to GDN, and only
1.7 percent to PLE. Serialized stages account for 11.4 percent of measured
TTFT, so the remaining prefill work is outside those profiled stage boundaries;
PLE storage is not the long-context TTFT bottleneck. Receipts:
`build/reports/local-ple-storage/single-stream-context-ladder-20261008.json`,
`build/reports/local-ple-storage/prefill-attribution-20261008.json`, and
`build/reports/local-ple-storage/decode-tps-goals-roofline-20261008.json`.

## Perplexity and hybrid-port audit, 2026-10-08

Optimization candidates now also pass a real-model perplexity gate. The corpus
is WikiText-2 raw test at immutable revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, tokenized once by the served
tokenizer. The baseline scores all next-token positions in sixteen 1,024-token
windows and four 4,096-token windows: PPL 5.7379 with mean loss 1.747095 at
short context, and PPL 4.7214 with mean loss 1.552111 beyond the 2,048-token
QSA budget. A candidate fails if either regime increases relative PPL by more
than 0.5 percent or mean loss by more than 0.0005. A deliberate 5 percent LM
head corruption raised PPL by 23.3–32.5 percent and failed, while the
unmodified runtime reproduced the baseline exactly. Receipts:
`build/reports/local-ple-storage/ppl-gate-accepted-20261008.json`,
`build/reports/local-ple-storage/ppl-gate-baseline-verify-20261008.json`, and
`build/reports/local-ple-storage/ppl-gate-selftest-corrupt-20261008.json`.

The Qwen hybrid port audit against Transformers 5.17.0 found the text-only
serving pieces complete: full attention, QSA block selection and fallback,
Gated DeltaNet, partial RoPE, external PLE, hyper-residual paths, and hybrid
cache operations. The audit also found and repaired two lifecycle defects:
unequal-batch attention/indexer extension could drop rows, and private Qwen
cache classes could not be resolved after save/reload. Commit `588298b` fixes
both; 49 focused tests passed with the real tiny fixture. Interleaved MRoPE is
equivalent for text-only input because the reference expands identical T/H/W
positions, but distinct-axis multimodal MRoPE and the vision wrapper remain
unported. Receipts:
`build/reports/local-ple-storage/qwen-hybrid-port-audit-20261008.json` and
`build/reports/local-ple-storage/qwen-hybrid-port-audit.txt`.

## Gate correction and long-prefill attribution, 2026-10-09

The October 8 perplexity tolerances were too strict for the gate's role. PPL is
now a loose coherence check: either context regime may increase relative PPL by
at most five percent or mean loss by at most 0.01, while exact greedy IDs and a
separate distribution gate own strict matching. This section supersedes the
earlier 0.5 percent PPL / 0.0005 loss wording. The recorded baseline remains
short PPL 5.7379 / mean loss 1.747095 over 16,368 tokens and long PPL 4.7214 /
mean loss 1.552111 over 16,380 tokens. A deliberate 1.05 LM-head scale raised
short PPL by 23.2768 percent, long PPL by 32.5412 percent, and mean loss by
0.209257 / 0.281717, so the looser gate still rejects the corruption probe.
Receipts:
`build/reports/local-ple-storage/ppl-gate-baseline-coherence-20261009.json`,
`build/reports/local-ple-storage/ppl-gate-coherence-corrupt-20261009.json`,
and `build/reports/local-ple-storage/ppl_gate.py`.

Strict numerical matching now has an executable two-phase KL gate. Reference
and candidate models are never resident together. The reference phase writes
one window of full-vocabulary float32 logits to transient local scratch; the
candidate phase reloads only that window, computes stable full-vocabulary log
probabilities in float64 by small position chunks, reports both
`KL(reference || candidate)` and `KL(candidate || reference)`, rejects on the
maximum direction, and deletes the reference after each comparison. It also
reports top-1 agreement and the first divergence position. The default strict
threshold is `1e-5` nats per scored token. A canonical one-window self-test
scored 1,023 positions at zero KL in both directions, 100 percent top-1
agreement, zero maximum log-probability delta, and no first divergence. It took
22.318 seconds for the reference phase and 25.093 seconds for the candidate
phase, proving the sequential-memory design on the real model. The gate is
`build/reports/local-ple-storage/kl_gate.py`; the one-window receipt is
`build/reports/local-ple-storage/kl-gate-self-20261009.json`. Multimodal input
remains outside this gate and outside the current optimization scope.

The corrected serialized 8,192-token prefill profile supersedes the earlier
stage attribution. Its wall time was 38.063814 seconds, with MoE at 24.445433
seconds / 64.22 percent, GDN at 4.328385 seconds / 11.37 percent, residual at
3.982837 seconds / 10.46 percent, full attention at 2.311834 seconds / 6.07
percent, and PLE at 0.995587 seconds / 2.62 percent. Inside MoE, the pre-fused
gate/up projection cost 14.656165 seconds and down cost 8.128703 seconds across
768 calls. Direct PLE storage spent 853.071 ms total, issued in 66.774 ms, and
read 33,661,120 bytes for 105,191 disk rows after deduplication. Native MoE
compute, not PLE storage, is the long-context target. Receipts:
`build/reports/local-ple-storage/prefill-budget-8192-20261008.json` and
`build/reports/local-ple-storage/prefill-stage-profile-8192-20261008.json`.

Five same-process alternating pairs rejected final-position-only prefill LM
head evaluation. Outputs were exact, but median TTFT worsened by 0.897 ms at
512 tokens and 426.243 ms at 8,192 tokens. Receipt:
`build/reports/local-ple-storage/prefill-head-ab-gate-20261008.json`.
Three exact prefill micro-optimizations were also rejected because their
isolated gains were too small to justify graph complexity: sorted pre-scatter
weighted combine saved 0.063 ms per layer, compiled SparseMoEBlock saved 0.148
ms per layer, and cacheless compiled GDN saved 0.177 ms per layer. A 512-token
prefill step remains canonical: at 8,192 tokens, three-round medians were
16.804771, 14.227218, and 14.668550 seconds for steps 256, 512, and 1024,
respectively. The warmed 512 result approaches the 15 second T1 target but is a
pilot, not a code promotion. Receipts:
`build/reports/local-ple-storage/sorted-weighted-combine-probe-20261009.json`,
`build/reports/local-ple-storage/compiled-prefill-moe-probe-20261009.json`,
`build/reports/local-ple-storage/compiled-prefill-gdn-probe-20261009.json`,
and `build/reports/local-ple-storage/prefill-step-pilot-8192-20261009.json`.

The native sorted MXFP4 `gather_qmm_rhs` tile family was closed for this
iteration. Four exact variants, BM=64; BN=128; BK=128; and BM=16/BN=128/WM=1/
WN=4, differed from the canonical 32/64/64/2/2 geometry by only noise-level
one-layer medians and none passed an interleaved real-model promotion gate. The
canonical geometry stays promoted. The diagnostic `gather-qmm-rhs-bm` worktree
was restored byte-for-byte and removed non-forced. Receipt:
`build/reports/local-ple-storage/gather-qmm-rhs-tiles-consolidated-rejected-20261009.json`.

## Executable strict KL gate, 2026-10-09

The gate described in the earlier contract is now an executable tool at
`build/reports/local-ple-storage/kl_gate.py`, so exact numerical matching no
longer depends on a specification nobody runs. It follows the contract: full
vocabulary, both directions, reject on the maximum, report top-1 agreement and
the first divergence position, and never retain all logits.

Two constraints shaped the implementation. The artifact is about 68 GB
resident, so reference and candidate cannot both be loaded. The orchestrator
therefore runs them as two sequential processes and keeps only one transient
float32 logits file per window, written through a memory-mapped array, compared
in position chunks of 16, and deleted immediately after that window's
comparison. The reference process costs about 117 seconds and the candidate
process about 207 seconds for the full twenty-window corpus, which is cheaper
than a second resident model would be.

Measured results, both on the canonical artifact:

- Full canonical self-comparison over 32,748 scored positions: both directions
  exactly 0.0 nats per token, top-1 agreement 1.0, maximum absolute log-prob
  difference 0.0, first divergence none. The build is bit-stable against itself,
  so the gate has a true zero reference. Receipt:
  `build/reports/local-ple-storage/kl-gate-full-baseline-20261009.json`.
- Negative control, scaling only the candidate LM head by 1.01 on one window:
  maximum direction KL 6.2189e-3 nats per token, top-1 agreement 0.9951, first
  divergence at position 0, verdict fail. That is about 620 times the 1e-5
  strict default, so the gate rejects a one percent head change while the loose
  perplexity gate, at 5 percent tolerance, would not necessarily notice. Receipt:
  `build/reports/local-ple-storage/kl-gate-corrupt-selftest-20261009.json`.

A first attempt at the negative control returned a false pass with both
directions at exactly 0.0. The cause was a harness defect, not a gate defect:
the corruption flag was passed through the shared argument list to both phases,
so the reference was perturbed by exactly the same factor and the two
distributions cancelled. The orchestrator now forwards the perturbation only to
the candidate phase. This is recorded because a self-test that cannot fail is
worse than no self-test, and because the same pattern would silently disable
any future one-sided perturbation.

Gate layering is now settled and explicitly three-tiered. Exact generated-ID
equality is the cheapest check and covers the common case of a numerically
neutral change. The KL gate is the strict check for any candidate that changes
numerics: kernel geometry, fusion, cache layout, or precision. Perplexity is
only the loose coherence gate at 5 percent relative PPL and 0.01 mean loss,
because a coherence audit should not demand near-exact matching; if strict
matching is wanted, KL and exact IDs are the instruments for it. Multimodal
inputs remain out of scope for this campaign, so distinct-axis MRoPE and the
vision wrapper stay unported by decision, not by oversight.

## Where decode time actually goes, 2026-10-09

Two long-open questions were closed today with measurements rather than
speculation.

The 8,192-token prefill budget is now corrected and recorded. A serialized
8,192-token prefill costs 38.063814 seconds, of which MoE is 64.22 percent
(14.656165 s gate/up and 8.128703 s down over 768 calls), GDN 11.37 percent,
residual 10.46 percent, full attention 6.07 percent, and external PLE only
2.62 percent. Direct PLE measurement on the same run is 853.071 ms, of which
66.774 ms is issue and 105,191 of 131,072 requested rows came from disk. Long
single-request TTFT is therefore compute-bound in native MoE, and no storage
tier change can move it. Receipt:
`build/reports/local-ple-storage/prefill-budget-8192-20261008.json`.

Per-kernel GPU attribution for steady decode is not obtainable on this machine,
and it is worth saying so plainly rather than estimating. The Metal System
Trace template ships with Shader Timeline disabled, so exported command
intervals carry only Metal encoder identifiers. MLX's own capture path does
work, but only with `MTL_CAPTURE_ENABLED=1` set in the process environment, and
its GPUTraceDocument destination serialized every Metal buffer into 67 GiB with
no kernel table; that capture and both xctrace bundles were deleted. The one
real result is that the captured run reproduced the control's generated IDs
exactly while inflating per-token latency about 379 times, so Instruments can
check numerics here but cannot time kernels. Kernel identity must come from
MLX's own shader list, which is what the routed-gather and gather-qmv-routing
receipts already used. Receipt:
`build/reports/local-ple-storage/decode-kernel-attribution-20261009.json`.

What the existing evidence does support is the decode budget: 26.196 ms p50, of
which the GDN recurrence upper bound is 1.601 ms, the Q/K normalization upper
bound 0.565 ms, and about 19.75 ms is unattributed to any single stage. The
routed experts are the largest known component at 6.224 ms per token against a
2.787 ms code-and-scale memory floor, so roughly half of that cost is gather
access pattern rather than unavoidable traffic. Receipts:
`build/reports/local-ple-storage/decode-walltime-budget-20261008.json` and
`build/reports/local-ple-storage/moe-bandwidth-gap-20261008.json`.

The four MXFP4 `gather_qmm_rhs` tile variants were consolidated into one
rejection receipt, and the diagnostic `gather-qmm-rhs-bm` worktree was restored
byte-for-byte and removed without force. The canonical 32/64/64/2/2 geometry
stands. Upstream's grouped-MM branch was re-read as design input: its float NAX
tile is BN=128 with BK=128 and WN=4, whose roughly 34.8 KB threadgroup tile
exceeds what the quantized loader can hold, which is consistent with BK=128
having measured slower here. Receipts:
`build/reports/local-ple-storage/gather-qmm-rhs-tiles-consolidated-rejected-20261009.json`
and `build/reports/local-ple-storage/kl_distribution_gate_contract.json`.

## Shader Timeline correction and real kernel attribution, 2026-10-09

The earlier “per-kernel GPU attribution is not obtainable” conclusion was wrong.
It was based on the stock `Metal System Trace` template and the public
`xctrace --show-recording-options` surface, which hides the Metal Application
instrument’s Shader Timeline switch. The stock template’s archived
NSKeyedArchiver state contains `shaderprofiler=false`.

`make_shader_timeline_template.py` clones that template, changes only the
`shaderprofiler` default to true, and writes a task-owned 32 KiB template. No
developer credential and no system-file modification are required. A small MLX
matmul probe then reports `Shader Timeline: Enabled` and exports both
`metal-shader-profiler-shader-list` (kernel name and PC range) and
`gpu-shader-profiler-interval` (timestamp, duration, and PC). Receipt:
`build/reports/local-ple-storage/shader-timeline-attainable-20261009.json`.

A real 48-token Qwen capture produced a 189,251,584-byte trace, 242 Python-owned
shaders, and 202,404 PC intervals. `shader_timeline_summary.py` resolved
Instruments XML references and matched 155,497 intervals to kernel names. The
steady-state parser then selected the final seven repeated 101-command decode
segments. Their GPU command spans were 20.275-20.854 ms; 13,509 PC intervals in
those windows matched a kernel and 77 did not. Matched attribution mass is
18.811075 ms per step, close enough to the command span to be useful for
ranking targets, while still not being exclusive wall time.

The top steady-decode kernels are:

| Kernel family | Attribution ms/step |
| --- | ---: |
| Dense/common `mxfp4_qmv_fast` pr0 | 5.269496 |
| Exact BF16 LM-head GEMV | 2.782589 |
| `mxfp4_gather_qmv_fast` | 2.334535 |
| `mxfp4_qmv_fast` pr1 | 1.625976 |
| `mxfp4_gather_qmv` | 1.449446 |
| `mxfp4_qmv` pr0 | 1.210649 |
| BF16 copy | 0.968845 |
| block sort | 0.466376 |
| packed T1 GDN recurrence | 0.250768 |

Across all observed MXFP4 variants, dense/common QMV carries 8.106121 ms/step
and gather-QMV variants carry 3.783982 ms/step. This revises the optimization
priority: the dense/common MXFP4 QMV family is now the largest named target,
larger than either individual gather variant. The exact BF16 LM-head GEMV is
independently confirmed at 2.783 ms/step, while the packed T1 GDN recurrence is
only 0.251 ms of attributed work. Receipt:
`build/reports/local-ple-storage/decode-shader-timeline-steady-20261009.json`.

The GPUTraceDocument attempt remains rejected: `MTL_CAPTURE_ENABLED=1` makes
`mx.metal.start_capture` work, but it serializes every Metal buffer and wrote
67 GiB. That path should not be used for this model. The enabled xctrace
template is the working route.

## Fused GDN decode convolution and state update, 2026-10-09

The Shader Timeline attribution led to an accepted decode optimization. Each of
the 36 GDN layers was concatenating the old three-token convolution state with
the current token, running the depthwise convolution, slicing the concatenated
result, and calling `mx.contiguous` to update the cache. The copy alone carried
0.968845 ms of attributed work per decode step.

Three experiments established the boundary. Replacing the contiguous copy with
another direct concatenate remained exact but regressed p50 by 0.100 ms because
it still copied. Keeping a non-contiguous cache view also remained exact but
regressed p50 by 0.108 ms because the next step paid strided access. The
working design is a fused `gdn_conv_decode_update_v2` Metal kernel that reads
the old state and current projection, computes the T=1 depthwise convolution
and production BF16 SILU, emits the current convolution output, and writes the
rolled state in one launch. Prefill, batching, left-padding lengths, and
non-Metal paths retain the original implementation.

The kernel is bitwise on the production `[1, 4, 10240]` shape for both output
and state. A same-resident-model full-vocabulary decode distribution gate
compared 24 generated positions: logits were exactly equal, both KL directions
were zero, top-1 agreement was 100%, and no first divergence existed. Five
alternating short-context pairs remained ID-exact and improved median p50 from
25.579 to 25.277 ms and median p95 from 26.199 to 25.793 ms. Three alternating
4,096-token pairs also remained ID-exact and improved median p50 from 28.515 to
28.421 ms and median p95 from 29.579 to 28.968 ms. Shader Timeline on the fused
path reduced the old copy attribution to 0.001006 ms/step while the new fused
kernel costs 0.050103 ms/step.

The change is committed locally on private fork branch `dynamic-mxfp4` at
`18bb2e77609a3573fe462e9fd3bdcd0dad331f9a`; it was not pushed or upstreamed.
Focused validation passed 16 tests with eight context-dependent skips, and the
dynamic-branch real smoke produced the exact control continuation at 25.374 ms
decode p50. Promotion receipt:
`build/reports/local-ple-storage/gdn-conv-fused-promoted-20261009.json`.

The same hill-climb closed four dense MXFP4 QMV read/tile variants. A global
four-pack-per-thread build changed a small QSA projection and collapsed real
output to repeated exclamation marks. Shape-gated four-pack, `r=2`, and
one-pack variants were exact on isolated probes but regressed the dominant GDN
projection; all were rejected. QuantizedLinear call census and C++ stride logs
show that the real dense QMV inputs are already row-contiguous, so there was no
cheap copy-elimination seam in QMV itself. Receipt:
`build/reports/local-ple-storage/decode-hillclimb-rejections-20261009.json`.

## Upstream synchronization through 388f227f4, 2026-10-13

Upstream advanced from `77bf1fa01` to `388f227f4`. A full merge was built and
audited first. It included useful correctness work across foreign-buffer
ownership, compile caches, negative strides, large reductions, convolution
bounds, batched `qvm_split_k`, sort layout selection, scalar conversion, FFT,
optimizer overflow, and IO, plus new thin-N/thin-M NAX GEMM dispatch. The full
merge passed 83 focused quantized/GEMM tests with 6,238 subtests, but it changed
the fixed 4,096-token Qwen continuation at a near tie.

Commit `494c8a6c9`, which changes quantized split-K partials from the input type
to float32, was identified as the divergence source. Reverting only that commit
restored byte-identical 4,096-token output. That change may improve generic
numerical accuracy, but it is incompatible with this campaign's exact-output
contract until a separate quality budget explicitly accepts its distributional
change. It remains excluded locally.

The selective integration passed 82 focused quantized/GEMM tests with 6,226
subtests and a broader 270-test regression set with 6,739 subtests. A strict
one-window full-vocabulary comparison against the pre-merge core produced
exactly equal logits, zero KL in both directions, and 100 percent top-1
agreement. Short-context output stayed exact and latency remained within noise.
At 4,096 tokens, three exact-output pairs improved median decode p50 from
27.299 to 27.090 ms and p95 from 27.706 to 27.555 ms. An instrumented 8,192-token
prefill profile remained compute-dominated and showed no material TTFT change.

The validated result is local MLX `main` merge `24f048a33`, containing upstream
through `388f227f4` with `494c8a6c9` reverted. The promoted main Python/Metal
build passed an M4 Max GPU smoke and reproduced the canonical Qwen continuation
at 24.683 ms decode p50. Nothing was pushed or upstreamed. Receipt:
`build/reports/local-ple-storage/upstream-388f-selective-promoted-20261013.json`.
