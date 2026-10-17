#!/usr/bin/env python3
"""Report and prepare the two-repo MLX upstream sync."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_POLICY = (
    Path(__file__).parents[1] / "docs/architecture/upstream-sync-policy.json"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_policy(path: Path) -> dict:
    return json.loads(path.read_text())


def repo_status(policy: dict, name: str) -> dict:
    spec = policy["repositories"][name]
    repo = Path(spec["path"])
    branch = spec["branch"]
    remote = spec["remote"]
    dirty = bool(git(repo, "status", "--porcelain"))
    local_head = git(repo, "rev-parse", branch)
    remote_head = git(repo, "rev-parse", f"{remote}/main")
    upstream_commits = git(
        repo,
        "log",
        "--format=%H",
        f"{branch}..{remote}/main",
    ).splitlines()
    private_commits = git(
        repo,
        "log",
        "--format=%H",
        f"{remote}/main..{branch}",
    ).splitlines()
    changed = git(repo, "diff", "--name-only", branch, f"{remote}/main").splitlines()
    sensitive = sorted(set(changed).intersection(spec["sensitive_paths"]))
    return {
        "repository": name,
        "path": str(repo),
        "branch": branch,
        "dirty": dirty,
        "local_head": local_head,
        "upstream_head": remote_head,
        "upstream_commits": len(upstream_commits),
        "private_commits": len(private_commits),
        "sensitive_changed_paths": sensitive,
        "rerere_enabled": git(repo, "config", "--get", "rerere.enabled") == "true",
        "conflict_style": git(repo, "config", "--get", "merge.conflictstyle") or None,
    }


def exclusions(policy: dict) -> list[dict]:
    rows = []
    mlx = Path(policy["repositories"]["mlx"]["path"])
    for item in policy["numerical_exclusions"]:
        short = item["commit"][:12]
        revert = git(
            mlx,
            "log",
            "--format=%H",
            "--grep",
            short,
            "main",
        )
        rows.append(
            {
                **item,
                "revert_commit": revert or None,
            }
        )
    return rows


def begin(policy: dict, name: str) -> dict:
    spec = policy["repositories"][name]
    repo = Path(spec["path"])
    status = repo_status(policy, name)
    if status["dirty"]:
        raise RuntimeError(f"{name} has uncommitted changes; refuse to start merge")
    if status["upstream_commits"] == 0:
        return {"repository": name, "action": "none", "reason": "already current"}

    git(repo, "fetch", spec["remote"], "main")
    branch = f"research/upstream-sync-{datetime.now():%Y%m%d}"
    worktree = repo / spec["integration_worktree"]
    if worktree.exists():
        raise RuntimeError(f"integration worktree already exists: {worktree}")
    git(repo, "worktree", "add", str(worktree), "-b", branch, "main")
    merge = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "merge",
            "--no-edit",
            f"{spec['remote']}/main",
        ],
        check=False,
        text=True,
    )
    return {
        "repository": name,
        "action": "integration-worktree",
        "worktree": str(worktree),
        "branch": branch,
        "merge_returncode": merge.returncode,
        "next": "resolve conflicts if any, then run policy tests and Qwen gates",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--repo",
        choices=("mlx", "mlx-lm", "all"),
        default="all",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("doctor")
    subparsers.add_parser("begin")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    names = policy["update_order"] if args.repo == "all" else [args.repo]
    result = {"recorded_at": datetime.now().isoformat()}
    if args.command == "status":
        result["repositories"] = [repo_status(policy, name) for name in names]
        result["numerical_exclusions"] = exclusions(policy)
    elif args.command == "doctor":
        rows = [repo_status(policy, name) for name in names]
        result["repositories"] = rows
        result["passed"] = all(
            not row["dirty"]
            and row["rerere_enabled"]
            and row["conflict_style"] == "zdiff3"
            for row in rows
        )
    else:
        result["repositories"] = [begin(policy, name) for name in names]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
