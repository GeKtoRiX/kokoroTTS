#!/usr/bin/env python3
"""Restore a Git working tree directly from .git/objects without git executable.

This is intentionally minimal and supports loose objects and packed refs are ignored.
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import zlib
from pathlib import Path


def read_obj(objects_dir: Path, sha: str) -> tuple[str, bytes]:
    obj_path = objects_dir / sha[:2] / sha[2:]
    if not obj_path.is_file():
        raise FileNotFoundError(f"Missing git object: {sha}")
    raw = zlib.decompress(obj_path.read_bytes())
    nul = raw.index(b"\x00")
    header = raw[:nul].decode("ascii", errors="replace")
    typ, _size = header.split(" ", 1)
    return typ, raw[nul + 1 :]


def parse_tree(data: bytes) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    i = 0
    while i < len(data):
        sp = data.index(b" ", i)
        mode = data[i:sp].decode("ascii", errors="replace")
        nul = data.index(b"\x00", sp)
        name = data[sp + 1 : nul].decode("utf-8", errors="replace")
        sha = data[nul + 1 : nul + 21].hex()
        entries.append((mode, name, sha))
        i = nul + 21
    return entries


def get_head_commit(repo_root: Path) -> str:
    git_dir = repo_root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        return (git_dir / ref).read_text(encoding="utf-8").strip()
    return head


def get_commit_tree(objects_dir: Path, commit_sha: str) -> str:
    typ, body = read_obj(objects_dir, commit_sha)
    if typ != "commit":
        raise RuntimeError(f"HEAD object is not a commit: {commit_sha} ({typ})")
    for line in body.decode("utf-8", errors="replace").splitlines():
        if line.startswith("tree "):
            return line.split()[1]
    raise RuntimeError(f"Commit {commit_sha} has no tree")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_blob(target: Path, data: bytes, mode: str, force: bool) -> str:
    if target.exists() and not force:
        return "skipped"
    ensure_parent(target)
    target.write_bytes(data)
    if mode == "100755":
        current = target.stat().st_mode
        target.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return "written"


def walk_tree(
    objects_dir: Path,
    tree_sha: str,
    out_root: Path,
    rel_prefix: Path,
    force: bool,
    stats: dict[str, int],
) -> None:
    typ, body = read_obj(objects_dir, tree_sha)
    if typ != "tree":
        raise RuntimeError(f"Expected tree {tree_sha}, got {typ}")

    for mode, name, sha in parse_tree(body):
        rel_path = rel_prefix / name
        if rel_path.parts and rel_path.parts[0] == ".git":
            stats["ignored"] += 1
            continue

        if mode == "40000":
            walk_tree(objects_dir, sha, out_root, rel_path, force, stats)
            continue

        if mode not in {"100644", "100755", "120000"}:
            stats["ignored"] += 1
            continue

        typ2, blob = read_obj(objects_dir, sha)
        if typ2 != "blob":
            raise RuntimeError(f"Expected blob {sha}, got {typ2}")

        target = out_root / rel_path
        result = write_blob(target, blob, mode, force)
        stats[result] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    git_dir = repo_root / ".git"
    objects_dir = git_dir / "objects"

    if not git_dir.is_dir():
        print(f".git not found under: {repo_root}", file=sys.stderr)
        return 2

    commit = get_head_commit(repo_root)
    tree = get_commit_tree(objects_dir, commit)

    stats: dict[str, int] = {"written": 0, "skipped": 0, "ignored": 0}
    walk_tree(objects_dir, tree, repo_root, Path("."), args.force, stats)

    print(f"HEAD commit: {commit}")
    print(f"Tree: {tree}")
    print(f"Written: {stats['written']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Ignored: {stats['ignored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
