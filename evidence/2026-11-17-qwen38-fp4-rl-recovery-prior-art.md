# Qwen3.8 FP4 RL-recovery prior art

Date: 2026-11-17 America/Los_Angeles

## Scope

Research prior art for taking a Qwen3.8-class post-trained/RL-style checkpoint to NVFP4 or MXFP4 while preserving or improving BF16 reasoning and agentic behavior. It covers QAT, quantization-aware distillation (QAD), quantization-aware RL (QARL), LoRA/QLoRA, thinking-trace artifacts, dense versus MoE behavior, compute/token budgets, and n-gram/PLE memory. Primary evidence and parsed texts are cached under ignored `build/reports/qwen38-fp4-prior-art/`.

This is a source and computation review. It does not run a new model or claim a measured result on local hardware.

## Linked checkpoint ground truth

`Qwen/Qwen3.8-27B` at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` is a dense Qwen3.5-family vision-language model, not MoE. Its 18 BF16 safetensors total 55,563,006,776 bytes (51.747 GiB), consistent with roughly 27.8B BF16-stored parameters. The language stack has 64 layers, hidden size 5,120, vocabulary 248,320, hybrid linear/full attention, and one MTP hidden layer.

The card calls the checkpoint "Pre-training & Post-training" and describes thinking controls, but does not disclose an RL algorithm, RL data, or post-training compute. November 5 and 7 repository changes were presentation/template changes; the weight upload dates remain August 2026. "This checkpoint is RL-trained" is plausible from Qwen3.8-family lineage, but is not source-verified for this dense 27B artifact.

## Closest prior art

1. **TRACE, arXiv:2610.07767v2.** Rollout-guided FP4 QAT for MoE RL. On Qwen3.8-Flash-Next (125B-A6B), joint NVFP4 W/A/KV rollout reached Terminal-Bench 70.6 versus BF16 rollout 68.8 after 400 RL steps. It also matched BF16 on DeepSWE and GDPval, reached up to 5.4x rollout speedup, and kept the large PLE/n-gram table BF16/offloaded in the throughput study. Limit: MoE and training-with-FP4-rollout, not post-hoc healing of the dense 27B.

2. **`nvidia/Qwen3.8-27B-NVFP4`.** Exact dense-27B PTQ baseline. Mixed NVFP4 MLP/head plus FP8 attention, Local-Hessian calibration on 2,048 samples: GPQA 88.92 to 88.01, Terminal-Bench 75.56 to 74.02, AA-LCR 72.63 to 73.38, MMMU-Pro 75.14 to 74.86, SciCode 47.93 to 48.41, IFBench 80.07 to 78.93. Limit: mixed precision, no recovery training, and no confidence interval in the card.

3. **`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`.** Exact dense-27B true W4A4 MXFP4 artifact. MXFP4 weights and dynamic MXFP4 activations, 128 pileval samples, vision tower BF16. Thinking-mode GSM8K is 94.996 versus BF16 93.33 (101.8% recovery); non-thinking is 89.92 versus 90.67. Limit: one benchmark family, generic calibration, no Terminal-Bench/tool-use result.

4. **`nvidia/Qwen3.8-Flash-Next-NVFP4-QAD`.** Same-family QAD artifact, exported after 100 updates, batch 512, 51,200 sequences, max length 16,384, FSDP2 world size 64. Terminal-Bench is exactly 85.96 for BF16 and QAD; GPQA drops 0.45, IFBench gains 0.33, SciCode gains 0.89, AA-LCR drops 0.68, and MMMU-Pro drops 0.41. PLE n-gram is frozen FP8/BF16-scale, GDN/QSA are MXFP8, experts are NVFP4, and KV remains BF16. Limit: MoE/mixed precision; the card explicitly says task-accuracy gains are not statistically established and Terminal-Bench retry/timeout accounting is incomplete.

5. **NVIDIA QAD report, arXiv:2601.20088v3.** Strongest recovery-training match for RL-heavy checkpoints. On Nemotron 3 Nano 30B-A3B and AceReason 7B, QAD recovers much more RL capability than PTQ or ordinary QAT; QAT can be worse than PTQ. KL distillation from the BF16 teacher preserves the policy distribution better than task loss. Limit: reasoning benchmarks rather than tool-use agents, with >100x variation in convergence tokens.

6. **ReQAT, arXiv:2606.15682.** For W4A4/W4A4KV4 reasoning models, trace-aligned QAT plus selective entropy minimization and Q-FIT exceeded matched-budget BF16 fine-tuning on AIME (65.94% versus 65.46% for R1-Qwen-14B). It reports 3.1x B200 and up to 3.9x DGX Spark serving throughput. Limit: dense distilled math reasoners, not a 27B agentic VLM.

7. **QATFactory, arXiv:2609.39223v2.** QAD/QARL framework for 8B-230B dense and MoE models, NVFP4/MXFP4/Q4_K, full and LoRA training. NVFP4 QARL was 1.23x faster and +2.7 points versus BF16 RL then RTN on Qwen3-8B math. Limit: the QARL table is math reasoning, not agentic tool use.

8. **QUADS and FourSight, arXiv:2607.15810 and 2611.04887.** Native NVFP4 W4A4 RL stabilization by aligning training and rollout paths. QUADS matched BF16 average pass@1 within 0.29 points on a 25B-A2.8B MoE. FourSight used sequence filtering plus 64 BF16 channels and reached 1.76-2.60x faster step-to-accuracy on Qwen3-8B math. Limit: both alter the RL path and neither is a post-hoc dense-27B healing recipe.

9. **Reasoning-QAT, arXiv:2601.14888.** PTQ initialization, KD recovery, then cold-start RL works on RL-trained Qwen3 reasoning models. KD is preferred over SFT; direct low-bit RL from a badly degraded policy can fail. Limit: small, weight-only 2/3-bit experiments.

10. **QeRL, arXiv:2510.11696.** NVFP4 weight-only rollout plus LoRA can train a 32B model on one H100-80GB and beat BF16 LoRA/QLoRA on math reward. Controlled quantization noise can improve exploration. Limit: W4A16 rather than native W4A4, and math rather than agentic tasks.

## Thinking-trace artifacts

The artifact mode is real and measured:

- Meta's arXiv:2606.00206 finds aggressive PTQ increases CoT length. In up to 52% of failures, the quantized model reaches a correct intermediate answer but fails to stop. High teacher-student KL occurs at high-entropy positions and disproportionately samples "wait", "but", and "alternatively". A training-free marker logit penalty reduces CoT length by 12-23% while preserving or improving accuracy.
- arXiv:2606.25519 introduces CoT Token Inflation Ratio. INT4/INT3 can preserve final-answer accuracy while generating more reasoning tokens. On BFCLv4 multi-turn, Qwen3-4B INT4 changed tool calls only from 7.56 to 7.85 while CoT tokens rose from 6,556 to 7,425; Qwen3-30B INT3 raised CoT tokens 23.5% with tool calls nearly unchanged.
- OnQ, arXiv:2610.29378, shows why fixed-response QAT misses this failure: the quantized student drifts onto its own prefixes. On-policy distillation with candidate-wise dynamic KL improves Qwen3-8B W2 average from 50.10 after calibration to 70.68, reduces truncation, and reduces repetitive errors.

Parity gates must therefore include generated tokens, truncation, repeated spans, tool-call count, and end-to-end wall time, not only pass rate.

## Agentic degradation and evaluation hygiene

- ACBench, arXiv:2505.19433, finds 4-bit PTQ generally causes 1-3% workflow/tool-use drops but 10-15% degradation in real-world application tasks.
- arXiv:2608.06564 models tool-call decision flips as shrunken/shifted decision margins, `m' ~ c*m + b`. Whether to call a tool is more fragile than which tool to call.
- arXiv:2610.07781 shows 8-bit versus 4-bit tool-failure recovery can reverse sign with prompt, task screening, or strict/tolerant scoring. Matched tasks and full-pipeline deployment scoring are mandatory.
- Coding-agent and multi-turn studies reported by workers likewise show path, timeout, and loop-count effects that aggregate accuracy hides.

## Compute and scaling

No clean cross-family law predicts agentic BF16 parity from parameter count alone. The strongest anchors are:

- NVIDIA QAD convergence budgets: 0.3B tokens for 49B Llama Nemotron Super, 6B for 9B Nemotron Nano V2, 0.5B for 12B VLM, 2.5B for 30B-A3B MoE, and 0.8B for 7B AceReason. This spans 0.006 to 0.667 tokens per total parameter (0.833 per active parameter for the 30B-A3B case), a >100x spread.
- ReQAT R1-Qwen-14B budgets are 140M, 210M, 280M, and 350M tokens, or 0.010 to 0.025 tokens per parameter. The 350M budget exceeded matched BF16 fine-tuning on AIME in that setting.
- OnQ runs 300 steps for Qwen3-4B/8B with 128 sampled responses per step. At the 4K-16K response caps, the rollout-token envelope is roughly 0.15-0.63B tokens; actual lengths are variable and lower than the cap.
- QATFactory LoRA-QAD reduces Qwen3.5-9B training memory 2.91x (167 GiB to 57 GiB), but rank 4/16/64/256 does not close the quality gap to full QAD (best LoRA average 68.0 versus 68.9 full QAD).
- The Qwen3.8-Flash-Next QAD artifact reports sequence/update topology but not non-padding tokens or GPU-hours.

For a dense 27.8B model, the defensible first planning envelope is roughly 0.3-2.5B recovery tokens: ReQAT's direct ratio suggests about 0.7B and QAD's 30B MoE anchor suggests about 2.3B. This is an interpolation range, not a law. Dense QAD with teacher forward and student forward/backward is approximately 8ND FLOP before architecture-specific costs; 0.3-2.5B tokens is about 67-556 EFLOP. GPU-hours cannot be inferred without measured achieved throughput.

Loss-only scaling work exists but does not answer agent parity directly. QAT scaling law arXiv:2505.14302 fits 268 runs up to 973M parameters with `delta(N,D,G)=k*D^gammaD*log2(G)^gammaG/N^gammaN`; error falls with model size and finer groups but rises with training tokens. Compute-optimal QAT arXiv:2509.22935 fits 86M-2.2B models and predicts the optimal QAT fraction from tokens per parameter-byte, but addresses FP-then-QAT pretraining rather than recovery from an RL-post-trained agent. Precision and low-bit scaling work (arXiv:2411.04330 and 2411.17691) finds more pretrained tokens can make PTQ worse; compression scaling arXiv:2502.16440 uses effective parameter count.

## N-gram and PLE boundary

There are two distinct n-gram mechanisms:

1. **Speculative n-gram lookup affects speed, not the target distribution.** Leviathan et al.'s speculative-decoding formula gives expected accepted tokens `(1-alpha^(gamma+1))/(1-alpha)` and divides walltime gain by `gamma*c+1`, where alpha is acceptance and c is drafter cost. Exact rejection preserves the target distribution. Quantization changes alpha and overhead, so acceptance and end-to-end tokens/second must be measured; this is not an accuracy ceiling.

2. **Learned PLE/n-gram tables are model capacity.** The linked dense 27B config has no PLE/n-gram table. Qwen3.8-Flash-Next does. NVIDIA's QAD artifact keeps its PLE table frozen FP8 E4M3 with BF16 scales; TRACE's throughput study leaves BF16 PLE/n-gram embeddings offloaded to CPU. Low-bit n-gram work (arXiv:2610.31646 and 2610.21810) shows 3/4-bit tables can retain most table benefit and short QAT recovers lower-bit loss, but those are table/perplexity/small-model results, not Qwen3.8 agentic parity.

## Recommended synthesis

The best dense Qwen3.8-27B path is not ordinary task-loss QAT:

1. Start from format-aware PTQ (Local-Hessian, MR-GPTQ, or AWQ) plus a small BF16 island set chosen by measured decision-margin and activation-channel sensitivity.
2. Recover with BF16-teacher QAD on the student's own agentic trajectories and tool-call decisions, not only fixed SFT responses. Add candidate-wise dynamic KL plus termination and anti-repetition signals.
3. If continuing RL, use rollout-aligned QARL/TRACE. Align quantized rollout and trainer execution; independent per-path accuracy can still leave policy mismatch.
4. Treat LoRA as a memory control, not an assumed quality equivalent. Full-parameter QAD currently retains a measurable edge.
5. Gate on Terminal-Bench/SWE/tool-use plus matched-task recovery, CoT inflation, tool-call count, timeouts, truncation, and end-to-end latency.

## Verification and worker consolidation

- Primary sources were fetched and inspected directly with `exq search`, `curl`, the Hugging Face model API, and arXiv HTML. The Qwen API/config, exact 27B artifact cards, QAD paper, TRACE, ReQAT, QATFactory, QUADS, FourSight, thinking-trace papers, scaling papers, and n-gram papers were read locally.
- Worker A `01a1eceb-51f5-73d0-b2b9-4cc68c5db59f` and Worker B `01a1eceb-7da9-74c2-bb77-60d90f0f68f9` completed with network access and no source changes. Worker C `01a1eceb-97e7-78a0-b6fa-1a26973eafab` completed only a partial report and had four failed search commands; its useful leads were independently fetched and are not relied on from the worker alone.
- No model training, inference, quantization, or benchmark was run for this receipt.

## Non-claims

- This does not establish that the linked dense Qwen3.8-27B checkpoint was RL trained.
- It does not show a clean compute-per-parameter law for agentic BF16 parity.
- It does not validate NVIDIA PTQ, AMD W4A4, TRACE, QAD, or ReQAT numbers on local hardware.
- It does not claim uniform W4A4/KV4 parity for Qwen3.8-27B. NVIDIA's exact 27B artifact is mixed NVFP4/FP8; the same-family QAD artifact is MoE and keeps KV BF16.
- It does not prove n-gram table compression preserves Qwen3.8-Flash-Next Terminal-Bench behavior.
