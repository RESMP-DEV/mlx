# Machine-wide no-upstream policy audit

Date: 2026-10-17

## Scope

Audit the machine-wide agent policy for upstream contribution behavior and make
the standing no-upstream boundary enforceable rather than advisory. Covers
`~/AGENTS.md` and its generated Codex/ZCode copies plus local Git remotes. It
does not modify any upstream repository.

## Rule provenance

The no-upstreaming paragraph existed as a ZCode-only rule stated by the user on
2026-09-25, recorded in ZCode memory
`no-actions-no-upstream-policy.md`, with the rationale "pretty difficult to get
them to accept upstreaming." The 2026-09-28 global instruction rebuild promoted
it into the machine-wide core (`global-agents-md-topology.md`), so `~/AGENTS.md`,
`~/.codex/AGENTS.md`, and `~/.zcode/AGENTS.md` all carried it while Claude
inherited it through the `@~/AGENTS.md` import.

## Audit findings

1. The rule ended with "Pushing anything to an upstream is a user-explicit
   instruction only." That exception is what allowed upstream work to be
   considered at all, and the user has now reversed it. The exception is removed.
2. Only `mlx` and `mlx-lm` had push-blocked upstream remotes. Eleven other local
   checkouts still had writable upstream push URLs, including `Strata`, `3FS`,
   `phonon`, `ktransformers`, and `cli`.
3. `RESMP-DEV/mlx` and `RESMP-DEV/mlx-lm` are GitHub fork-links (`isFork:
   true`), which the core rule forbids. Thirty-five RESMP-DEV repositories are
   fork-links. Converting them to plain mirrors is repository-structure work and
   was not performed here.

## Changes

- `~/AGENTS.md` core: the no-upstreaming paragraph is now a permanent policy. It
  forbids upstream PRs, issues, review comments, patches, contribution drafts,
  upstream pushes, and fork-track/transfer, and it forbids upstream
  security/bug/feature reports. Fetching upstream, reading upstream
  code/docs/issues, and recording provenance remain allowed. The previous
  explicit-instruction exception is explicitly reversed.
- Generated Codex and ZCode blocks regenerated with
  `agent-instructions-sync --sync`; parity verified.
- Every local repository with an `upstream` remote now has
  `remote.upstream.pushurl` set to the sentinel
  `/Users/kearm/.config/no-upstream-push`. Fetch URLs are unchanged.

## Verification

Thirteen local repositories have an `upstream` remote and all thirteen now have
the sentinel push URL. A dry-run push to `upstream` fails in every case, and
`git ls-remote upstream refs/heads/main` still returns the upstream head. The
global instruction parity check passes.

An earlier attempt used `/Users/kearm/.git/no-upstream-push`. Creating that
directory made the home directory look like a Git repository. It was removed
after verification that the directory was empty, and the sentinel was moved to
`~/.config/no-upstream-push`.

## Non-claims

This audit does not convert existing fork-link repositories to plain mirrors, does
not withdraw anything already published on RESMP-DEV, and does not prove that no
future task could find an upstream contribution worth discussing. It records the
current standing boundary.
