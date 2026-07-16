# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the pre-commit hook and its bootstrap installer.

They need nothing beyond git, bash and Python -- no black, clang-format, torch
or CUDA. Every test builds its own throwaway git repository under ``tmp_path``
and copies the scripts into it, so the tests never assume they run inside a git
repository and never touch the real checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BOOTSTRAP = REPO_ROOT / "bootstrap.sh"
HOOK = REPO_ROOT / "hooks/pre-commit"
FORMAT_CODE = REPO_ROOT / "lint/format-code.sh"
METADATA_HELPER = REPO_ROOT / "gsplat/build_support/pyproject_metadata.py"
BASH = shutil.which("bash") or "bash"


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


# A fake clang-format the harness drives end-to-end. `--version` names the
# pinned version; `-i` normalizes runs of spaces to one in place; `--dry-run`
# reports without rewriting and fails only with `--Werror`, like the real tool;
# a file containing FAILME is treated as unformattable; a file whose upward
# .clang-format lookup does not resolve to the test-owned config is skipped,
# mirroring the real tool's silent use of whatever config it resolves.
FAKE_CLANG_FORMAT = r"""#!/usr/bin/env bash
if [[ "$1" == --version ]]; then echo "clang-format version 99.9.9"; exit 0; fi
mapfile -t files
mode=format
werror=0
[[ "$*" == *--dry-run* ]] && mode=check
[[ "$*" == *--Werror* ]] && werror=1
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
        if ! sed 's/  */ /g' "$f" | cmp -s - "$f"; then
            echo "$f: not formatted" >&2
            (( werror )) && rc=1
        fi
    else
        sed 's/  */ /g' "$f" > "$f.fmt" && mv "$f.fmt" "$f"
    fi
done
exit "$rc"
"""

# A minimal pyproject pinning clang-format to the fake's version.
PYPROJECT = """\
[project]
name = "t"
version = "0"

[project.optional-dependencies]
dev = ["clang-format==99.9.9"]
"""

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="requires the git and bash executables",
)


def _run(cmd, cwd):
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_isolated_env(),
        text=True,
        capture_output=True,
    )


def _git(args, cwd):
    result = _run(["git", *args], cwd)
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def repo(tmp_path):
    """An isolated git repo containing copies of the scripts under test."""
    root = tmp_path / "repo"
    (root / "lint").mkdir(parents=True)
    (root / "hooks").mkdir()
    for src, dst in (
        (BOOTSTRAP, "bootstrap.sh"),
        (HOOK, "hooks/pre-commit"),
    ):
        target = root / dst
        target.write_text(src.read_text())
        target.chmod(0o755)
    _git(["init", "-q"], root)
    _git(["config", "user.email", "t@t"], root)
    _git(["config", "user.name", "t"], root)
    _git(["add", "-A"], root)
    _git(["commit", "-qm", "base"], root)
    return root


@pytest.fixture
def real_repo(repo):
    """The bootstrap/hook repo plus the real formatter, a test-owned pyproject,
    and the fake clang-format on PATH, so a commit runs the real harness."""
    (repo / "lint/format-code.sh").write_text(FORMAT_CODE.read_text())
    (repo / "lint/format-code.sh").chmod(0o755)
    (repo / "gsplat/build_support").mkdir(parents=True)
    (repo / "gsplat/build_support/pyproject_metadata.py").write_text(
        METADATA_HELPER.read_text()
    )
    (repo / "pyproject.toml").write_text(PYPROJECT)
    (repo / ".clang-format").write_text("BasedOnStyle: test\n")
    (repo / "bin").mkdir()
    (repo / "bin/clang-format").write_text(FAKE_CLANG_FORMAT)
    (repo / "bin/clang-format").chmod(0o755)
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "add formatter"], repo)
    return repo


def _installed_hook(repo):
    """Resolve the shared hooks/pre-commit path for ``repo`` (absolute)."""
    rel = _git(["rev-parse", "--git-path", "hooks/pre-commit"], repo).stdout.strip()
    path = Path(rel)
    return path if path.is_absolute() else (repo / path)


def _bootstrap(repo, cwd=None):
    """Run the repo's bootstrap.sh (optionally from an unrelated directory)."""
    return _run(["bash", str(repo / "bootstrap.sh")], cwd if cwd else repo)


def _write_formatter(repo, advertised, *, record=None, exit_code=0):
    """Write lint/format-code.sh as a stub advertising ``advertised`` flags.

    On ``--help`` it prints an options list with one entry per advertised flag --
    the same shape the hook detects support from; on any other invocation it
    appends its arguments to ``record`` (if given) and exits with ``exit_code``.
    This lets a test see which flags the hook chose without running a real
    formatter.
    """
    help_lines = ["Options:"] + [f"  {flag}    stub option" for flag in advertised]
    echo_block = "\n".join(f'    echo "{line}"' for line in help_lines)
    record_line = f'printf "%s\\n" "$*" >> "{record}"\n' if record else ""
    body = (
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--help" || "$1" == "-h" ]]; then\n'
        f"{echo_block}\n"
        "    exit 0\n"
        "fi\n"
        f"{record_line}"
        f"exit {exit_code}\n"
    )
    path = repo / "lint" / "format-code.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _commit(repo, name):
    """Stage a fresh file and commit; return the CompletedProcess (hook runs)."""
    (repo / name).write_text("x\n")
    _git(["add", name], repo)
    return _run(["git", "commit", "-m", name], repo)


# --- bootstrap.sh install / backup behaviour ------------------------------


def test_installs_regular_executable_file_with_identifier(repo):
    _bootstrap(repo)
    hook = _installed_hook(repo)
    assert hook.is_file() and not hook.is_symlink()
    assert hook.stat().st_mode & 0o111
    assert "GSPLAT PRE-COMMIT HOOK" in hook.read_text()


def test_reinstall_over_own_hook_makes_no_backup(repo):
    _bootstrap(repo)
    _bootstrap(repo)
    hook = _installed_hook(repo)
    assert not Path(str(hook) + ".backup").exists()


def test_backs_up_a_foreign_hook(repo):
    hook = _installed_hook(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho foreign\n")
    hook.chmod(0o755)
    result = _bootstrap(repo)
    assert result.returncode == 0, result.stderr
    assert "foreign" in Path(str(hook) + ".backup").read_text()
    assert "GSPLAT PRE-COMMIT HOOK" in hook.read_text()


def test_numbered_backup_never_clobbers_earlier_one(repo):
    hook = _installed_hook(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    Path(str(hook) + ".backup").write_text("earlier backup\n")
    hook.write_text("#!/bin/sh\necho foreign_two\n")
    hook.chmod(0o755)
    result = _bootstrap(repo)
    assert result.returncode == 0, result.stderr
    assert "earlier backup" in Path(str(hook) + ".backup.~1~").read_text()
    assert "foreign_two" in Path(str(hook) + ".backup").read_text()


def test_old_symlink_install_does_not_clobber_formatter(repo):
    formatter = repo / "lint" / "format-code.sh"
    formatter.write_text("#!/usr/bin/env bash\n# real formatter\n")
    formatter.chmod(0o755)
    hook = _installed_hook(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.symlink_to(formatter)
    _bootstrap(repo)
    assert "# real formatter" in formatter.read_text()  # not written through the link
    assert not hook.is_symlink()
    assert "GSPLAT PRE-COMMIT HOOK" in hook.read_text()


def test_installs_when_run_from_unrelated_directory(repo, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _bootstrap(repo, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert _installed_hook(repo).is_file()
    assert not (elsewhere / ".git").exists()


def test_bootstrap_skips_without_git(repo, tmp_path):
    # git absent from PATH: bootstrap must warn and exit 0, not crash.
    empty = tmp_path / "empty_path"
    empty.mkdir()
    result = subprocess.run(
        [BASH, str(repo / "bootstrap.sh")],
        cwd=str(repo),
        env={"PATH": str(empty)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "git" in result.stderr.lower()


# --- hook formatter dispatch (via stub format-code.sh) --------------------


def test_hook_uses_precommit_when_advertised(repo, tmp_path):
    record = tmp_path / "argv.log"
    _write_formatter(
        repo, ["--check", "--full", "--staged", "--precommit"], record=record
    )
    _bootstrap(repo)
    result = _commit(repo, "a")
    assert result.returncode == 0, result.stderr
    assert record.read_text().strip() == "--precommit"


def test_hook_ignores_flag_only_mentioned_in_prose(repo, tmp_path):
    # --precommit appears only inside another option's description, not as its
    # own entry, so the hook must not treat it as supported -- it falls back.
    record = tmp_path / "argv.log"
    formatter = repo / "lint" / "format-code.sh"
    formatter.parent.mkdir(parents=True, exist_ok=True)
    formatter.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--help" || "$1" == "-h" ]]; then\n'
        '    echo "Options:"\n'
        '    echo "  --check    stub option"\n'
        '    echo "  --full     stub option"\n'
        '    echo "  --staged   like --precommit, used by the hook"\n'
        "    exit 0\n"
        "fi\n"
        f'printf "%s\\n" "$*" >> "{record}"\n'
        "exit 0\n"
    )
    formatter.chmod(0o755)
    _bootstrap(repo)
    assert _commit(repo, "a").returncode == 0
    assert record.read_text().strip() == "--check --full"


def test_hook_falls_back_to_check_full(repo, tmp_path):
    record = tmp_path / "argv.log"
    _write_formatter(repo, ["--check", "--full", "--changed"], record=record)
    _bootstrap(repo)
    result = _commit(repo, "a")
    assert result.returncode == 0, result.stderr
    assert record.read_text().strip() == "--check --full"
    assert "WARNING" not in result.stderr  # fallback is silent


def test_hook_warns_and_skips_invalid_formatter(repo, tmp_path):
    record = tmp_path / "argv.log"
    _write_formatter(repo, ["--frobnicate"], record=record)
    _bootstrap(repo)
    result = _commit(repo, "a")
    assert result.returncode == 0, result.stderr  # commit still proceeds
    assert "appears invalid" in result.stderr
    assert not record.exists()  # formatter was never invoked


def test_hook_blocks_commit_when_formatter_fails(repo):
    _write_formatter(repo, ["--check", "--full", "--precommit"], exit_code=1)
    _bootstrap(repo)
    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    result = _commit(repo, "a")
    assert result.returncode != 0  # non-zero formatter blocks the commit
    assert _git(["rev-parse", "HEAD"], repo).stdout.strip() == head  # nothing recorded


def test_hook_warns_when_formatter_missing(repo):
    # No runnable formatter in the worktree -> hook warns but the commit proceeds.
    _bootstrap(repo)
    result = _commit(repo, "a")
    assert result.returncode == 0, result.stderr
    assert "not formatted" in result.stderr


def test_hook_resolves_formatter_per_worktree(repo, tmp_path):
    # Shared hook, but each worktree must run its own formatter.
    main_log = tmp_path / "main.log"
    _write_formatter(repo, ["--precommit"], record=main_log)
    _bootstrap(repo)

    linked = tmp_path / "linked"
    _git(["worktree", "add", "-q", str(linked), "-b", "feature"], repo)
    linked_log = tmp_path / "linked.log"
    _write_formatter(linked, ["--precommit"], record=linked_log)

    assert _commit(linked, "w").returncode == 0
    assert linked_log.exists() and not main_log.exists()


# --- end-to-end: real git commit through the installed hook ---------------


def test_hook_refuses_empty_commit_after_formatting(real_repo):
    # A staged change that is purely a formatting regression normalizes back to
    # HEAD when the hook formats it -- the commit would be empty. The hook must
    # refuse it rather than record an empty commit.
    (real_repo / "mod.cu").write_text("int a;\n")  # already formatted
    _git(["add", "mod.cu"], real_repo)
    _git(["commit", "-qm", "add mod"], real_repo)
    _bootstrap(real_repo)
    head = _git(["rev-parse", "HEAD"], real_repo).stdout.strip()

    (real_repo / "mod.cu").write_text("int  a;\n")  # formats straight back to HEAD
    _git(["add", "mod.cu"], real_repo)
    env = _isolated_env(PATH=f"{real_repo / 'bin'}:{os.environ['PATH']}")
    result = subprocess.run(
        ["git", "commit", "-m", "fmt only"],
        cwd=str(real_repo),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0  # commit refused, not recorded
    assert _git(["rev-parse", "HEAD"], real_repo).stdout.strip() == head  # HEAD unmoved
    assert (real_repo / "mod.cu").read_text() == "int a;\n"  # left formatted


def test_hook_commits_partial_stage_formatted(real_repo):
    # A partially-staged file commits through the hook with its staged content
    # formatted, while the unstaged edit stays in the working tree.
    base = "".join(f"int {c};\n" for c in "abcdefg")
    (real_repo / "p.cu").write_text(base)
    _git(["add", "p.cu"], real_repo)
    _git(["commit", "-qm", "add p"], real_repo)
    _bootstrap(real_repo)

    (real_repo / "p.cu").write_text(base.replace("int a;", "int  A;"))  # stage line 1
    _git(["add", "p.cu"], real_repo)
    staged = (real_repo / "p.cu").read_text()
    (real_repo / "p.cu").write_text(staged.replace("int g;", "int  G;"))  # unstaged
    env = _isolated_env(PATH=f"{real_repo / 'bin'}:{os.environ['PATH']}")
    result = subprocess.run(
        ["git", "commit", "-m", "partial"],
        cwd=str(real_repo),
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    committed = _git(["show", "HEAD:p.cu"], real_repo).stdout
    assert committed == base.replace("int a;", "int A;")  # formatted
    # Worktree: formatting synced, unstaged edit intact.
    assert (real_repo / "p.cu").read_text() == base.replace("int a;", "int A;").replace(
        "int g;", "int  G;"
    )
