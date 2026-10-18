# DFlash2 target distillation architecture correction

Date: 2026-10-18

## Scope

This records the corrected Qwen3.8-27B-to-DFlash2 training boundary and the
reusable adapter plan for Qwen3.8 Flash Next. It corrects the scope of the
Qwen3.5-0.8B DistillKit runs: those runs validated plumbing and a Qwen3.5
architecture guard, but were not DFlash2-head distillation.

## Concrete target and student

- Frozen target/verifier: `Qwen/Qwen3.8-27B`, immutable revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- BF16 target manifest: 18 model shards, approximately 54.5 GiB.
- DFlash2 head: local `z-lab/Qwen3.8-27B-DFlash2` lineage.
- Remote B550 head:
  `/home/kearm/dflash2-distill/model/model.safetensors`.
- DFlash2 model SHA-256:
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- DFlash2 config SHA-256:
  `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.
- DFlash2 shape: 1,924,404,480 BF16 parameters, five decoder layers,
  target layers 5/19/33/47/61, 25,600-wide raw target features, 5,120-wide
  draft hidden states, selector rank 256, runtime selector top-k 16, and
  vocabulary 248,320.

The target is the authority and remains frozen. The DFlash2 head is the only
student. Its output remains target-verified and speculative.

## Existing capture and training state

The existing full replay corpus on B550 contains 1,981 target-verified cycles
from 24 prompts, 256 generated tokens per prompt, and target top-64 teacher
signals:

- corpus: `/home/kearm/dflash2-distill/capture`;
- size: approximately 1.2 GB;
- capture elapsed time: 358.437 seconds;
- prior target path: the calibrated local Qwen3.8-27B MLX MXFP4 runtime;
- prior draft path: local BF16 DFlash2 with affine W4 runtime.

The existing trainer is selector-only: it freezes the transformer, target
projection, and both selector codebooks, and trains only the 1,310,720-parameter
selector hidden projection. Those candidates passed exact output parity but
failed the paired live-acceptance gate and were rejected. No full DFlash2
transformer or LoRA distillation has completed yet.

## Efficient training modes

The reusable architecture now records these modes:

- selector projection only: existing diagnostic mode;
- selector LoRA: future low-rank selector experiment;
- draft LoRA: low-rank adapters on the target projection and five decoder
  layers, the preferred next full-head mode;
- selective full-head tuning: unfreeze selected projections when adapter
  capacity is insufficient;
- QLoRA/NF4: memory-limited differentiable-base path, requiring a measured
  speed comparison;
- MXFP4 GPTQ: post-training deployment compression, not training.

Measured head shape makes draft LoRA inexpensive relative to the full student:
rank-8 adapters over the five decoder layers are about 4.44 million parameters,
rank-16 about 8.89 million, rank-32 about 17.78 million, and rank-64 about
35.55 million. Rank-32 is approximately 0.92 percent of all DFlash2 parameters.
A rank-16 adapter on the 25,600-by-5,120 target projection adds 491,520
parameters.

BF16 LoRA is the first implementation target because the frozen base remains in
a native fast dtype. QLoRA is a fallback for memory pressure; on Ampere it must
prove equal or better wall-clock speed. The existing MXFP4 GPTQ pack is not a
differentiable QLoRA implementation.

## Dataset reuse and future Flash Next port

The Fable and Infatoshi trace corpora are prompt sources for target-verified
cycle capture. They are not used to fine-tune the 27B target itself. Prompt
splits stay session-grouped; training, held-out, and full corpus captures carry
target/draft revisions, tokenizer identity, target layer IDs, top-k, block
size, generation policy, and quantization provenance.

A Qwen3.8 Flash Next target can reuse the capture and adapter-training contract
after a target adapter proves hidden-state extraction, embedding/logits
interfaces, cache rollback, verification semantics, vocabulary/tokenizer
identity, and exact native-forward parity. Different hidden width or depth
requires new feature dimensions and a new target projection.

Promotion remains unchanged in every mode: exact target-controlled output and
improved paired live acceptance. Training loss, offline agreement, compression
ratio, memory reduction, or throughput alone is not acceptance.

## Artifacts and implementation

- DFlash MLX architecture document:
  `dflash-mlx/docs/dflash2-distillation.md`.
- Branch: `work/local-mxfp4-dflash2`.
- Commit: `36859b2` (`docs: record DFlash2 distillation boundaries`).
- Remote: `RESMP-DEV/dflash-mlx`.
- B550 pinned target download: detached tmux session `qwen38-27b-hfd`, output
  `/home/kearm/training/models/Qwen3.8-27B-1d4bf0f`, full HFD verification
  requested after transfer.

At the time of this receipt the pinned target transfer was still active. This
receipt does not claim that the new BF16 target transfer, LoRA implementation,
or full-head distillation has completed.
