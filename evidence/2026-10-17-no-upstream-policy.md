# No-upstream standing policy

Date: 2026-10-17

## Scope

The user set a permanent boundary: MLX work is never contributed upstream to
`ml-explore/mlx`. The maintainers do not want our contributions and have not
asked for them. All our work is published on RESMP-DEV instead.

## Changes

- `AGENTS.md`: replaced inherited upstream-PR ban language with a standing
  no-upstream policy. RESMP-DEV pushes and pull requests remain allowed.
- `docs/architecture/upstream-sync.md`: extended the workflow boundary to
  prohibit upstream pushes, PRs, patches, and review responses.
- `docs/architecture/upstream-sync-policy.json`: recorded the standing
  no-upstream contribution policy and fetch-only upstream remote mode.

## Remote enforcement

Both upstream push URLs now point to the deliberately invalid sentinel
`/Users/kearm/.git/no-upstream-push`. Fetch URLs are unchanged. A dry-run push
to `upstream` fails immediately with exit 128 and never reaches the network:

```text
fatal: '/Users/kearm/.git/no-upstream-push' does not appear to be a git repository
exit=128
```

Fetch remains working; `git ls-remote upstream refs/heads/main` still returns
the current upstream head.

## Non-claims

This receipt does not imply an upstream reviewer opinion beyond the recorded
boundary, does not modify any upstream repository, and does not withdraw prior
publication from the public RESMP-DEV remote.
