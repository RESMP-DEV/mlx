# Upstream synchronization policy

This is the standing update procedure for the private MLX and mlx-lm `main`
branches. The machine-readable contract is
`docs/architecture/upstream-sync-policy.json`; `scripts/upstream_sync.py`
reports drift, checks repository health, and can create an integration
worktree.

## Workflow

1. Run `python scripts/upstream_sync.py status`. Update MLX first, then mlx-lm.
2. Run `python scripts/upstream_sync.py begin --repo mlx`. Build and test that
   integration worktree. A core change needs an old-core/new-core Qwen A/B gate.
3. Merge the validated MLX integration branch into private `main`.
4. Run `python scripts/upstream_sync.py begin --repo mlx-lm`. Resolve the two
   expected shared-file conflicts as feature unions. `rerere` is enabled and
   will remember the shapes.
5. Run focused mlx-lm tests and a real Qwen smoke against the new MLX core.
6. Record the upstream heads, decisions, tests, and non-claims in an evidence
   receipt. Only then merge private `main`.

Do not rebase private `main`. Merge commits preserve the audit trail and make
upstream numerical changes visible. Never push to `upstream`, open an upstream PR,
prepare an upstream patch, or respond to upstream review. Upstream contribution
is permanently out of scope: the maintainers do not want our contributions and
have not asked for them. Publish our work on the RESMP-DEV remotes instead.

## Boundary decisions

MLX stays close to upstream. Campaign implementation lives in mlx-lm; MLX keeps
architecture, evidence, benchmarks, and narrowly justified core changes. The
one active core exception is the local revert of upstream `494c8a6c9`, because
float32 quantized split-K partials changed the fixed 4,096-token Qwen output.
Revisit that exclusion only with a full-vocabulary KL budget, perplexity
coherence, exact-ID policy, and long-context latency gates.

mlx-lm owns the Qwen runtime and private optimizations. Its recurring merge
conflicts are currently limited to `server.py` and `utils.py`; preserve both
sides rather than choosing one. Future private work should add dedicated
modules instead of expanding edits in shared loader/server files.

## Minimum gates

For MLX, run the focused quantized/GEMM, convolution, SDPA, and reduction
suites. For a Qwen-sensitive core diff, build old and new cores separately and
run at least five alternating short-context pairs and three 4,096-token pairs.
Exact generated text is mandatory. If numerics can differ, also run the strict
full-vocabulary KL gate and loose perplexity gate.

For mlx-lm, run the Qwen/PLE, dynamic quantization, greedy fast path, PLE
runtime, and server suites. Then run a real-model smoke against the candidate
MLX core. Same-process A/B is preferred for Python-only runtime changes.

`python scripts/upstream_sync.py doctor` checks clean repositories, `rerere`,
and `zdiff3`; it does not prove that the branches are current or that tests
pass.
