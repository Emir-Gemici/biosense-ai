"""Check the Git index and optionally all reachable history before publication.

This checks file types, paths, sizes, UTF-8 text and common credential patterns.
It cannot determine whether arbitrary prose/code contains confidential information.
Review the contents of every newly allowed file before changing ALLOWED_FILES.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FILES = frozenset({
    ".gitattributes", ".gitignore", ".githooks/pre-commit", ".githooks/pre-push",
    ".github/workflows/ci.yml", ".streamlit/config.toml",
    "README.md", "CONTRIBUTING.md", "NOTICE.md", "requirements.txt", "train.py",
    "apps/evaluate.py", "pipelines/train_f2.py", "pipelines/train_f1.py",
    "docs/DATA_POLICY.md", "docs/USAGE.md", "docs/DEVELOPMENT_HISTORY.md",
    "docs/source-history.json", "docs/VALIDATION.md",
    "scripts/check_public_repo.py", "scripts/make_demo_data.py",
    "scripts/smoke_demo.py", "tests/test_repository.py",
})
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_" + r"[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA" + r"[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{32,}"),
)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def check_blob(path, content, mode="100644"):
    errors = []
    if path not in ALLOWED_FILES:
        errors.append(f"{path}: not on the explicit public file list")
    if mode not in ("100644", "100755"):
        errors.append(f"{path}: links, submodules and non-regular files are prohibited")
    if len(content) > 750_000:
        errors.append(f"{path}: exceeds the 750 KB source-file limit")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{path}: non-UTF-8 or binary content")
        return errors
    if "\0" in text:
        errors.append(f"{path}: NUL byte / binary content")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        errors.append(f"{path}: possible credential (value deliberately not printed)")
    return errors


def check_index(root=ROOT):
    errors = []
    count = 0
    for record in git("ls-files", "--stage", "-z", root=root).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            errors.append(f"{path}: unresolved merge conflict")
        errors.extend(check_blob(path, git("cat-file", "blob", oid, root=root), mode))
        count += 1
    return count, errors


def check_history(root=ROOT):
    errors = []
    seen = set()
    commits = git("rev-list", "--all", root=root).decode().splitlines()
    for commit in commits:
        for record in git("ls-tree", "-rz", commit, root=root).split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            path = raw_path.decode("utf-8")
            key = (mode, oid, path)
            if key in seen:
                continue
            seen.add(key)
            if kind != "blob":
                errors.append(f"{commit[:8]} {path}: non-blob entry")
                continue
            errors.extend(f"{commit[:8]} {e}" for e in
                          check_blob(path, git("cat-file", "blob", oid, root=root), mode))
    return len(commits), len(seen), errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also inspect every reachable commit")
    args = parser.parse_args(argv)
    count, errors = check_index()
    print(f"Checked {count} indexed files.")
    if args.history:
        commits, blobs, historical_errors = check_history()
        errors.extend(historical_errors)
        print(f"Checked {commits} commits and {blobs} distinct historical file entries.")
    if errors:
        for error in errors:
            print(f"BLOCKED: {error}", file=sys.stderr)
        return 1
    print("PASS: only approved source/documentation files; no detected credential patterns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
