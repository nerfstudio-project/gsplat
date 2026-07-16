# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lint/format-code.sh CLI.

These exercise the harness, not a real formatter, so every test is CPU-only
(git + bash + python). A fake clang-format on PATH mirrors the real tool's
contract (file list on stdin, --dry-run --Werror check semantics, upward
.clang-format lookup): it normalizes runs of spaces to one and treats a file
containing FAILME as unformattable. A fake python intercepts `-m black` and
records its argv. Every test builds its own throwaway git repository under
tmp_path and never touches the real checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FORMAT_CODE = REPO_ROOT / "lint/format-code.sh"
METADATA_HELPER = REPO_ROOT / "gsplat/build_support/pyproject_metadata.py"


def _isolated_env(**extra):
    """os.environ with every GIT_* variable dropped and the global/system config
    disabled: a global core.hooksPath would redirect hook installs OUTSIDE
    tmp_path, commit.gpgsign or init.templateDir would break commits, and
    GIT_DIR / GIT_CONFIG_COUNT injection would pierce a config-only override."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env.update(extra)
    return env


# A fake clang-format the harness drives. `--version` names the pinned version;
# `-i` normalizes in place -- runs of spaces to one, and runs of blank lines to
# one, so formatting can change the line count like the real tool's reflowing;
# `--dry-run` reports without rewriting and fails only with `--Werror`; a file
# containing FAILME is treated as unformattable; a file whose upward
# .clang-format lookup does not resolve to the test-owned config is skipped,
# mirroring the real tool's silent use of whatever config it resolves.
FAKE_CLANG_FORMAT = r"""#!/usr/bin/env bash
if [[ "$1" == --version ]]; then echo "clang-format version 99.9.9"; exit 0; fi
mapfile -t files
mode=format
werror=0
[[ "$*" == *--dry-run* ]] && mode=check
[[ "$*" == *--Werror* ]] && werror=1
normalize() {
    sed 's/  */ /g' "$1" | awk 'BEGIN{b=0}{if($0==""){b++; if(b>1) next}else b=0; print}'
}
has_config() {
    local d
    d="$(cd "$(dirname "$1")" && pwd)"
    while :; do
        if [[ -e "${d}/.clang-format" ]]; then
            grep -q "BasedOnStyle: test" "${d}/.clang-format"
            return
        fi
        [[ "${d}" == / ]] && return 1
        d="$(dirname "${d}")"
    done
}
rc=0
for f in "${files[@]}"; do
    [[ -z "$f" ]] && continue
    if grep -q FAILME "$f"; then rc=1; continue; fi
    has_config "$f" || continue
    if [[ "$mode" == check ]]; then
        if ! normalize "$f" | cmp -s - "$f"; then
            echo "$f: not formatted" >&2
            (( werror )) && rc=1
        fi
    else
        normalize "$f" > "$f.fmt" && mv "$f.fmt" "$f"
    fi
done
exit "$rc"
"""

# A fake python that intercepts `-m black` and records its argv (one entry per
# line, appended to black-argv.log in the current directory), so a test can
# assert which flags the harness composed; a file containing BLACKFAIL makes it
# exit non-zero. Anything else is unexpected.
FAKE_PYTHON = r"""#!/usr/bin/env bash
if [[ "$1" == -m && "$2" == black ]]; then
    shift 2
    printf '%s\n' "$@" >> black-argv.log
    for a in "$@"; do
        [[ -f "$a" ]] && grep -q BLACKFAIL "$a" && exit 1
    done
    exit 0
fi
echo "unexpected python invocation: $*" >&2
exit 97
"""

# A minimal pyproject whose dev extra pins the formatters to the fakes'
# versions, so the harness's pin checks pass without depending on the repo's
# real config.
PYPROJECT = """\
[project]
name = "t"
version = "0"

[project.optional-dependencies]
dev = ["black==88.8.8", "clang-format==99.9.9"]
"""

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="requires the git and bash executables",
)


def _git(args, cwd):
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=_isolated_env(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo: the real format-code.sh + metadata helper, a test-owned
    pyproject and .clang-format, and the fake formatters on PATH."""
    root = tmp_path / "repo"
    (root / "lint").mkdir(parents=True)
    (root / "gsplat/build_support").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "lint/format-code.sh").write_text(FORMAT_CODE.read_text())
    (root / "lint/format-code.sh").chmod(0o755)
    (root / "gsplat/build_support/pyproject_metadata.py").write_text(
        METADATA_HELPER.read_text()
    )
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / ".clang-format").write_text("BasedOnStyle: test\n")
    for name, body in (("clang-format", FAKE_CLANG_FORMAT), ("python", FAKE_PYTHON)):
        (root / "bin" / name).write_text(body)
        (root / "bin" / name).chmod(0o755)
    _git(["init", "-q"], root)
    _git(["config", "user.email", "t@t"], root)
    _git(["config", "user.name", "t"], root)
    # The .py metadata helper stays untracked: no selection mode then ever picks
    # a Python file unless a test stages one deliberately.
    _git(["add", "--", "lint/format-code.sh", "pyproject.toml", "bin"], root)
    _git(["commit", "-qm", "base"], root)
    return root


def _run(repo, *args, env_extra=None):
    env = _isolated_env(PATH=f"{repo / 'bin'}:{os.environ['PATH']}")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(repo / "lint/format-code.sh"), *args],
        cwd=str(repo),
        env=env,
        text=True,
        capture_output=True,
    )


def _commit_file(repo, name, content):
    (repo / name).write_text(content)
    _git(["add", name], repo)
    _git(["commit", "-qm", name], repo)


# --- selection modes -------------------------------------------------------


def test_conflicting_mode_flags_rejected(repo):
    result = _run(repo, "--full", "--changed", "HEAD")
    assert result.returncode != 0
    assert "combined" in result.stderr.lower()


def test_full_mode_formats_tracked_files(repo):
    _commit_file(repo, "w.cu", "int  w;\n")  # committed unformatted
    assert _run(repo, "--full").returncode == 0
    assert (repo / "w.cu").read_text() == "int w;\n"  # rewritten in place


def test_full_check_reports_unformatted(repo):
    _commit_file(repo, "w.cu", "int  w;\n")
    result = _run(repo, "--check", "--full")
    assert result.returncode != 0
    assert "w.cu" in result.stdout + result.stderr


def test_black_check_receives_check_flags(repo):
    # The fake python logs `-m black` argv: the check action must compose
    # --check (else a broken check would silently reformat and exit 0) and the
    # pinned --required-version.
    _commit_file(repo, "p.py", "code\n")
    assert _run(repo, "--check", "--full").returncode == 0
    log = (repo / "black-argv.log").read_text()
    assert "--check" in log and "--diff" in log
    assert "--required-version\n88.8.8" in log
    assert "p.py" in log


def test_black_format_omits_check_flags(repo):
    _commit_file(repo, "p.py", "code\n")
    assert _run(repo, "--full").returncode == 0
    log = (repo / "black-argv.log").read_text()
    assert "--check" not in log
    assert "p.py" in log


def test_black_failure_propagates(repo):
    # A black failure must fail the run -- otherwise a check would silently pass
    # over unformatted Python.
    _commit_file(repo, "bad.py", "BLACKFAIL\n")
    assert _run(repo, "--check", "--full").returncode != 0


def test_changed_selects_files_since_ref(repo):
    # --changed <ref> selects only files changed since <ref>. old.cu is committed
    # unformatted before the ref, so over-selection would flag it.
    _commit_file(repo, "old.cu", "int  o;\n")
    ref = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _commit_file(repo, "new.cu", "int  n;\n")  # committed after ref, unformatted
    result = _run(repo, "--check", "--changed", ref)
    assert result.returncode != 0  # new.cu is unformatted
    out = result.stdout + result.stderr
    assert "new.cu" in out
    assert "old.cu" not in out
