# DFlash2 full-logit remote distillation pilot

Date: 2026-10-19

## Scope

This records the first real full-vocabulary DFlash2 teacher-logit replay on a
remote on-demand CUDA card, the exact promoted-checkpoint reconstruction, the
objective-control measurements, and two trainer defects found by live execution.
It does not promote a model.

## Live inputs

- Local full-logit capture remains active in tmux session
  `dflash-full-logits-896`. At 17:04 Pacific it had 203 completed prompts,
  13,850 indexed cycles, and 66 GB on disk. The final receipt is not written yet.
- The immutable 8-prompt pilot capture contains 432 cycles and is 2.3 GB. Its
  receipt records `full_logits: true`; it is real target output, not a proxy.
- Prime Intellect pod `9e96876cc684431ca7731f09d55fe479` is on-demand
  (not spot), with one RTX 6000 Ada, 49,140 MiB GPU memory, 12 vCPUs, and 70 GB
  host RAM. The earlier unentered pod was terminated after the account's missing
  SSH key was identified and fixed.

## Exact artifact reconstruction

The remote draft was rebuilt from the public DFlash2 base and the pinned G1, G2,
and G3 adapter lineage rather than copying 3.6 GB over the slow Mac-to-pod path.

- Public base model tensor file SHA-256:
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- G1 adapters SHA-256:
  `e2c62c7d8b7a4a816f074d18b2f436bbd04a8aa56e222283d952f1b6331299a5`.
- G2 adapters SHA-256:
  `531a0bc502bb392449e68b0c9764905ce04e45413e93e9ed2df548135207f03d`.
- G3 adapters SHA-256:
  `13b3114067342d7953b6bc0fee85bf9bb11b53f54461caaa78b33bdc2459cc65`.
- The reconstructed remote draft matched all 81 tensor hashes in the local
  promoted G3 manifest. The local promoted model file SHA-256 remains
  `c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`.
- The remote Qwen output-head tensor data SHA-256 is
  `d922b751f014ee1139488cb94c0e164a1eb3da5c14048070ed1b784e00f92723`,
  matching the local exported tensor data. Different safetensors container
  headers make whole-file hashes differ; tensor data is the comparison used.

## Defects fixed by live execution

The first remote run failed before training because `--selection-metric loss`
was accepted although full-logit objectives did not publish a `loss` metric. A
second failure showed that validation totals were not initialized for the first
metric. Both are fixed and covered by tests.

A more important replay defect was then exposed by distribution-only training:
full captures contain `selector_hidden`, and the old forward path always used
that shortcut. Distribution and hybrid LoRA therefore had no gradient path
through the draft transformer. Distribution and hybrid modes now replay the full
draft from captured context; selector mode deliberately retains the shortcut.

Code is on `RESMP-DEV/mlx-gptq-mxfp` branch `work/dflash2-calibration`:

- `0d34e19` `Fix full-logit selection metric`
- `938cb99` `Initialize validation metric totals`
- `12254e5` `Record CUDA training memory`
- `5a3d204` `Replay full draft for distribution loss`

Verification: `PYTHONPATH=. uv run --frozen --all-groups pytest -q` passed with
43 tests passed and 4 skipped.

## Matched 128-cycle objective controls

All controls use the same stratified 128-cycle selection, seed 43, rank-8 LoRA,
learning rate `1e-6`, predicted predecessors, and one epoch (99 training steps).
Selector uses the captured hidden-state shortcut. Distribution and hybrid replay
the full draft.

| Objective | Training seconds | Seconds/cycle | Peak CUDA reserved |
| --- | ---: | ---: | ---: |
| Selector-only | 5.307 | 0.0536 | 4.629 GB |
| Distribution CE+TV | 16.796 | 0.1697 | 9.428 GB |
| Hybrid selector+CE+TV | 15.427 | 0.1558 | 9.604 GB |

Initial and best validation values were identical in these deliberately tiny
runs. The initial full-replay distribution metrics were CE `6.82766` and TV
`0.93319`; the initial hybrid total loss was `8.35579`. These are diagnostic
only.

The existing Triton dynamic-convolution path also ran successfully on Ada, but
the 128-cycle hybrid time was 17.535 seconds (`0.1771 s/cycle`) versus 15.427
seconds eager. A separate block-size/warp sweep is in progress; no Ada Triton
claim is made yet.

## Scaling decision

The full-logit path peaks below 10 GB on a 49 GB card and the trainer is
batch-size one. A second GPU is therefore not required for memory and is not
justified for throughput until full-logit training survives paired live
acceptance gates and shows a material advantage over the selector incumbent.
The current pod remains one GPU.

## Transfer and next gates

A compressed incremental transfer of the first 12,000 immutable local cycles to
the pod is running in tmux session `dflash-full-logits-transfer-0-12000`. The
capture index and final receipt will be transferred only after local capture
completes.

Required before any promotion:

1. complete and verify the 896-prompt capture;
2. train matched objectives on the full session-grouped corpus;
3. perform exact target-controlled parity;
4. run paired live acceptance against the promoted G3 incumbent on disjoint
   held-out prompts;
5. only then reconsider a second GPU for throughput.

## Non-claims

- No objective is claimed superior to the incumbent.
- Offline loss, prefix accuracy, memory use, and successful remote execution are
  not acceptance.
- The tiny pilot learning rate and one-epoch run were plumbing and performance
  probes, not a training result.
- The 896-prompt capture was still running when this receipt was written.
- No candidate artifact was uploaded to Hugging Face.
