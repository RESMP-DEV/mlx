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

## 2026-11-18 refinement: the coding-agent dominance gap

The narrower product thesis is sharper than generic QAD recovery: build a deployment-native NVFP4/MXFP4 coding-agent student that dominates the frozen BF16 deployment policy on a defined coding-agent distribution across task success, output tokens, input/prefill work, tool calls, retries, timeout rate, and wall-clock cost. Targeted review found adjacent work on every component, but not that combined claim.

### Exact dense-27B full-W4A4 artifacts

`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` is the closest exact-model artifact:

- Dense Qwen/Qwen3.8-27B base.
- 496/496 transformer linears in NVFP4 W4A4; 19.7 GB artifact.
- One epoch of loss-aware NVFP4 QAD against the frozen BF16 teacher, global batch 32, learning rate 1e-6, 2,446 steps.
- GPQA-D 90.91 versus BF16 91.41; AIME'26 100.0 versus 100.0.
- MRCR 79.1 versus 83.4, HMMT'26 73.4 versus 75.8, SuperGPQA 62.4 versus 63.0, LiveCodeBench 79.2 versus 81.0, and RULER 32K/64K/128K within 0.4 points of BF16.
- Serves with MTP, tool-call parsing, and reasoning parsing, but reports no Terminal-Bench/SWE/DeepSWE trajectory-length result.

`minima-ai/mnma_qwen3.8_27b_nvfp4` is an exact dense-27B PTQ-only comparator:

- All 496 backbone linear layers, including all 48 Gated DeltaNet layers and their gate projections, are NVFP4 W4A4.
- Calibrated static FP8 KV-cache scales are included.
- Text-only model extraction; no vision tower.
- Calibration is 128 samples by 32K tokens; no QAT or QAD.
- BF16 versus Minima: MMLU-Pro 80.4/79.7, GSM8K 95.5/95.5, AIME'25 86.7/86.7, GPQA-D 86.5/85.1, LiveCodeBench 79.0/78.5; RULER is perfect for both at the tested lengths.
- Weights: 50.13/17.53 GiB.
- Decode at concurrency 32: 621 versus 1,154 tokens/s. 32K-token prefill TTFT: 6.90 versus 4.03 seconds.
- Mean AIME generation length is essentially unchanged at 14,531 versus 14,532 tokens, and Minima hits the 32K cap slightly less often.

Static dense-27B full-W4A4 quality, memory, prefill, decode, and math reasoning length are already close enough to make the missing coding-agent objective credible.

`orcarouter/OrcaSAQ-2-27B` is the closest low-bit commercial coding-agent artifact, though it is sensitivity-aware mixed precision at an average 3.21 bits per weight, not a published NVFP4/MXFP4 recipe:

- 12.3 GB text-only Qwen3.8-27B derivative with 262K context, MTP, thinking, and tool calling.
- Static fidelity: WikiText-2 perplexity 5.6468 BF16 versus 5.6482, top-1 agreement 93.2%, mean KLD 0.031.
- Public reference scores: 70.0% SWE-bench Verified and 58.4% Terminal-Bench 2.1.
- vLLM/MTP single-stream throughput is reported at 90.1 tokens/s under a 15.7-GiB memory cap.
- The card explicitly says agent scores use different scaffolds and are not apples-to-apples model-only comparisons; it does not report paired BF16 token/tool/wall-time deltas.

### Agent-specific closest work

The strongest score-only partial exception is NVIDIA's Nemotron 3.5 Lightning
30B-A3B NVFP4-QAD recipe. The final conservative deployment is predominantly
W4A16 rather than uniform W4A4, but it already exceeds the BF16 teacher on
several agentic scores:

- Terminal-Bench v2.1: BF16 24.44, PTQ 22.05, QAD 25.84.
- SWE-bench Multilingual: 37.13/37.00/38.07.
- SWE-bench Verified: 52.20/52.20/52.60.
- PinchBench: 83.86/84.78/85.15.
- BrowseComp: 39.50/39.50/40.00.

The same table has QAD below BF16 on tau-squared Telecom, tau-cubed Banking,
SciCode, GPQA, GDPval, IFBench, and AA-LCR; median score recovery is 98.97%.
Most importantly for the proposed product thesis, the post does not report
generated tokens, tool calls, retries, timeout rates, or task-level wall time.
Thus it shows that an NVFP4 student can beat a BF16 teacher on selected
agentic scores, while leaving the multi-axis dominance claim unmeasured.

SteerDistill (arXiv:2611.13410) is the closest published method. A full-precision teacher steers the quantized student at the first behavioral divergence, execution feedback verifies the correction, and the corrected behavior is distilled for unassisted inference.

Qwen3-8B W4A16 results:

- BFCL V4: 46.84 versus BF16 46.89.
- tau-squared Bench: 31.79 versus 32.64.
- WebShop: 68.41 versus 68.58.
- ALFWorld: 79.75 versus 79.92.
- Search QA: 39.12 versus 39.65.
- Five-benchmark average: 53.18 versus 53.54.
- tau-squared output tokens per successful task: 4.0K versus BF16 3.9K and quantized baselines 4.1-5.2K.
- Successful-trajectory input/steps/tool calls: 192K/11.6/4.24 versus BF16 185K/11.4/4.2.
- It beats QeRL by 1.84 points with 67% fewer training GPU hours.

Limitations: W4A16 rather than native W4A4, no Terminal-Bench/SWE/DeepSWE coding suite, and no claim of beating BF16 on score and token cost simultaneously.

The coding-agent diagnostic arXiv:2611.02368 shows why aggregate success is insufficient. On OJBench, final accuracy showed no statistically significant 4-bit loss, but AWQ changed Qwen-7B from 17 one-shot solves and 10 loop solves at BF16 to zero one-shot and 25 loop solves. Product evaluation must report solve strategy, executor calls, failures, and tails.

The task-level energy study arXiv:2610.32417 makes the same deployment point: 4-bit usually saves energy per token, but extra task-level output can raise energy per completed task by up to 3.5x. This is the exact failure mode that a token- and tool-aware student must train away.

NVIDIA's Qwen3.6-35B-A3B W4A4 QAD recipe gives the strongest warning:

- 500 QAD iterations at 32K sequence length on 128 GB200 GPUs took 5.7 hours, about 735 GPU-hours.
- W4A4 reached BF16 statistical parity on IFBench and recovered about 40% of the MMMU-Pro gap.
- SciCode score rose slightly above BF16, but mean completion length rose 91%, and 3.6% of sub-steps hit the 131K cap versus 0.7% for BF16.
- QAD also increased guessing rather than abstention on a knowledge probe.

Score recovery alone is therefore not market dominance. Length, termination, calibration, and tool behavior need explicit objectives.

### Independent Qwen3.8-27B serving evidence

A community shootout in Qwen/Qwen3.8-27B discussion #192 compared BF16, FP8, QUASAR full-W4A4, mixed NVFP4 recipes, and INT4 on one DGX Spark with vLLM, 262K context, FP8 KV, and MTP:

- QUASAR full-W4A4 and every other listed checkpoint scored 100% on the 20 agentic scenarios; the small suite was saturated.
- QUASAR: MBPP 68.9 versus BF16 70.8, HumanEval 93.9 versus 93.3, GSM8K 96.8 versus 97.0, tool success 100 versus 100.
- QUASAR decode throughput was 25.4 tokens/s versus BF16 14.6; NVIDIA mixed NVFP4/FP8 reached 38.6 tokens/s.
- The evaluator reported similar real-world coding success across variants, but also reasoning loops across all NVFP4 checkpoints under some settings.
- No task-level token/tool/retry accounting was published.

This community evidence is deployment signal, not a controlled research result. It shows that full-W4A4 quality is already plausible and that stability/termination, rather than static score, is a primary remaining risk.

### Why the exact experiment has not been claimed

1. Academic works optimize broad average score, not a deployment product's Pareto-dominance contract.
2. Agent environments are expensive, slow, noisy, and often saturated or contaminated.
3. Token inflation was only recently named and measured as a first-class cost.
4. Many recovery papers use W4A16, which does not unlock FP4 tensor-core GEMMs.
5. Full W4A4/KV4 pipelines require aligned rollout/training numerics, sensitive-path selection, and real serving kernels.
6. Beating a BF16 teacher requires paired tasks, repeated seeds, and confidence intervals on multiple axes.
7. Commercial teams with production coding-agent telemetry have stronger incentives to ship the artifact than publish the recipe.

### Audacious but defensible experiment

The target should be called an FP4-specialized agent, not pure quantization recovery: the student receives additional task-specific training and verifier feedback after the BF16 checkpoint. This is valid against a frozen BF16 deployment policy, but it is not a claim that 4-bit arithmetic is intrinsically superior to BF16.

A narrow claim is plausible because coding has external verifiers: tests, compilers, shell exit status, repository state, and tool schemas. A student can internalize searched or verified BF16 trajectories and then outperform one-shot BF16 deployment on success and total cost.

Adjacent theory and agent-specialization evidence support this without proving it:

- Stanton et al. (arXiv:2106.05945) note that a self-distilled student can exceed its teacher, but only because it fails to match the teacher perfectly; imperfect distillation can act as useful regularization.
- Self-distillation theory (arXiv:2002.05715 and 2006.05065) explains improved held-out performance as stronger/instance-specific regularization, not new information.
- Agent-workflow compilation (arXiv:2605.22502) shows that full-parameter specialization of 3B/8B models can retain 87-98% of frontier in-context quality while reducing token volume 2-7x and conversation cost 128-462x on procedural agent tasks. It is not quantized, but it demonstrates that narrow agent behavior and token volume can be internalized far beyond generic benchmark fine-tuning.
- No-free-lunch results and in-domain fine-tuning distortion results bound the scope: a coding-agent-specialized student may dominate a frozen teacher on the selected deployment distribution while regressing elsewhere. The claim must name the task distribution and report general-capability deltas; it cannot be a universal BF16-dominance claim.

PEFT should therefore be a screening lane, not the presumed final lane. The
strongest triangulation is now:

- QATFactory: LoRA-QAD ranks 4/16/64/256 all trail full-parameter NVFP4 QAD;
  rank 16 cuts memory 2.91x but the best LoRA average is 68.0 versus 68.9.
- Procedural agent fine-tuning (arXiv:2607.21612): LoRA ranks 16-128 score
  task success 2.10-2.54 versus 4.11 for full fine-tuning. SVD puts the mean
  effective rank of the required full update at 761-1,026; rank 128 captures
  only 43-51% of squared Frobenius norm.

No source directly compares LoRA/DoRA/QLoRA with full-parameter QAD on the same
coding-agent benchmark and trajectory metrics. Existing separate evidence
points the same way: useful for cheap search, weak as a final high-rank
procedural-policy substitute.

Objective: maximize task success subject to paired constraints against BF16 on generated tokens, input/prefill work, tool calls, interaction steps, retries, timeouts, truncations, wall time, and energy. Tail p95/p99 constraints matter as much as means.

Training design:

1. Start from QUASAR full-W4A4 for exact-model quality or Minima for full-GDN W4A4 plus FP8 KV.
2. Generate a `teacher++` corpus with BF16 best-of-N, high reasoning effort, verifier feedback, and shortest-successful-trajectory selection.
3. Use SteerDistill-style state restoration at the first behavioral divergence and local verification.
4. Add OnQ-style dynamic KL on student prefixes.
5. Add constrained agent reward or a Lagrangian penalty on output tokens, tool calls, retries, and wall time.
6. Add termination and loop suppression at high-entropy branching positions.
7. Preserve tool-call decision margins and low-entropy symbolic decisions.
8. Keep BF16/FP8 islands only where measured channel or decision-margin error pays for their cost.
9. Compare LoRA/DoRA plus trainable scales for iteration cost, but promote through full-parameter QAD if PEFT leaves a quality or trajectory-tail gap.

Evaluation must use paired, unsaturated, temporally split tasks and report success confidence intervals; output/input/cached tokens; tool calls and arguments; retries and malformed calls; compiler/runtime failures; timeout/truncation/loop rates; p50/p95/p99 trajectory cost; wall time and energy per completed task; MTP acceptance and cache-path divergence; and a private holdout.

No found source reports an NVFP4/MXFP4 W4A4 student satisfying all those constraints against its BF16 teacher. That is the unclaimed result.

## Verification and worker consolidation

- Primary sources were fetched and inspected directly with `exq search`, `curl`, the Hugging Face model API, and arXiv HTML. The Qwen API/config, exact 27B artifact cards, QAD paper, TRACE, ReQAT, QATFactory, QUADS, FourSight, SteerDistill, QUASAR, Minima, OrcaSAQ2, coding-agent papers, thinking-trace papers, scaling papers, self-distillation papers, and n-gram papers were read locally.
- Worker A `01a1eceb-51f5-73d0-b2b9-4cc68c5db59f` and Worker B `01a1eceb-7da9-74c2-bb77-60d90f0f68f9` completed with network access and no source changes. Worker C `01a1eceb-97e7-78a0-b6fa-1a26973eafab` completed only a partial report and had four failed search commands; its useful leads were independently fetched and are not relied on from the worker alone.
- The 2026-11-18 refinement used three additional workers: exact coding-agent claims `01a1ed0a-4fb6-7ea2-b4d9-930ab075f652`, PEFT/agent comparison `01a1ed0a-79da-7e72-978d-01d42b957816`, and teacher-beating theory `01a1ed0a-a017-7221-93cf-8703eef7b9a8`. All completed. Their load-bearing claims were independently fetched, including SteerDistill, the exact Qwen3.8-27B QUASAR/Minima/OrcaSAQ2 cards, NVIDIA Nemotron 3.5 Lightning QAD, procedural LoRA rank failure, and self-distillation theory.
- No model training, inference, quantization, or benchmark was run for this receipt.

## Non-claims

- This does not establish that the linked dense Qwen3.8-27B checkpoint was RL trained.
- It does not show a clean compute-per-parameter law for agentic BF16 parity.
- It does not validate NVIDIA PTQ, AMD W4A4, TRACE, QAD, or ReQAT numbers on local hardware.
- It does not claim uniform W4A4/KV4 parity for Qwen3.8-27B. NVIDIA's exact 27B artifact is mixed NVFP4/FP8; the same-family QAD artifact is MoE and keeps KV BF16.
- It does not prove n-gram table compression preserves Qwen3.8-Flash-Next Terminal-Bench behavior.
