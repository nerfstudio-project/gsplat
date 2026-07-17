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


# --- CLI contract (no formatter needed) -----------------------------------


def test_help_advertises_precommit(repo):
    # The installed hook detects --precommit as an indented options-list entry
    # (leading space, then the flag); a bare mention would not satisfy it, so
    # --help must list it that way.
    import re

    result = _run(repo, "--help")
    assert result.returncode == 0
    assert re.search(r"(?m)^[ \t]+--precommit(\s|$)", result.stdout)


def test_precommit_implies_staged_mode(repo):
    # --precommit expands to --staged; combining it with --full must be rejected
    # as a mode conflict -- proving the expansion without running a formatter.
    result = _run(repo, "--precommit", "--full")
    assert result.returncode != 0
    assert "combined" in result.stderr.lower()


def test_changed_rejects_unknown_ref(repo):
    # --changed with a ref that does not resolve must fail loudly, not exit 0.
    result = _run(repo, "--check", "--changed", "no-such-ref-xyz")
    assert result.returncode != 0
    assert "no-such-ref-xyz" in result.stderr


# --- --staged perform mode ------------------------------------------------


def test_fully_staged_formats_and_restages(repo):
    _commit_file(repo, "x.cu", "int x;\n")
    (repo / "x.cu").write_text("int  y;\n")  # a real change, unformatted
    _git(["add", "x.cu"], repo)
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":x.cu"], repo).stdout == "int y;\n"  # formatted + re-staged
    assert (repo / "x.cu").read_text() == "int y;\n"  # mirrored into the worktree


def test_partial_staged_formats_staged_only(repo):
    # A partially-staged file has its staged content formatted and re-staged;
    # an unstaged edit far enough from the formatting (outside its patch
    # context) stays in the working tree intact, with the delta applied
    # around it.
    base = "int a;\nint b;\nint c;\nint d;\nint e;\n"
    _commit_file(repo, "two.cu", base)
    (repo / "two.cu").write_text("int  A;" + base[6:])  # stage line 1, unformatted
    _git(["add", "two.cu"], repo)
    staged = (repo / "two.cu").read_text()
    (repo / "two.cu").write_text(
        staged.replace("int e;", "int  E;")
    )  # line 5, unstaged
    assert _run(repo, "--staged").returncode == 0
    formatted = base.replace("int a;", "int A;")
    # The staged content is formatted and re-staged...
    assert _git(["show", ":two.cu"], repo).stdout == formatted
    # ...and the working tree got only the formatting delta: line 1 formatted,
    # the unstaged line-5 edit byte-identical.
    assert (repo / "two.cu").read_text() == formatted.replace("int e;", "int  E;")


def test_partial_overlap_leaves_worktree_alone(repo):
    # When an unstaged edit overlaps the formatting delta, the delta cannot be
    # applied to the working tree: the commit still gets the formatted content,
    # and the working-tree copy is left byte-identical.
    _commit_file(repo, "v.cu", "int v;\n")
    (repo / "v.cu").write_text("int  v1;\n")  # stage a change to the line
    _git(["add", "v.cu"], repo)
    (repo / "v.cu").write_text("int  v2;\n")  # re-edit the same line, unstaged
    result = _run(repo, "--staged")
    assert result.returncode == 0
    assert _git(["show", ":v.cu"], repo).stdout == "int v1;\n"  # index formatted
    assert (repo / "v.cu").read_text() == "int  v2;\n"  # worktree untouched
    assert "overlap" in result.stderr  # the desync is called out


def test_staged_deleted_file_formats_index_only(repo):
    # A file staged then deleted from the working tree still commits formatted;
    # there is no working-tree copy to sync.
    _commit_file(repo, "gone.cu", "int g;\n")
    (repo / "gone.cu").write_text("int  g2;\n")
    _git(["add", "gone.cu"], repo)
    (repo / "gone.cu").unlink()
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":gone.cu"], repo).stdout == "int g2;\n"  # formatted
    assert not (repo / "gone.cu").exists()  # still deleted


def test_sibling_bookkeeping_dirs_do_not_collide(repo):
    # The scratch tree keeps staged copies and its own orig/ and work/ backups
    # in disjoint subtrees, so repo files living under directories literally
    # named orig/ or work/ cannot collide with a sibling file's copies.
    (repo / "orig").mkdir()
    (repo / "work").mkdir()
    _commit_file(repo, "x.cu", "int top;\n")
    _commit_file(repo, "orig/x.cu", "int nested;\n")
    _commit_file(repo, "work/x.cu", "int w;\n")
    (repo / "x.cu").write_text("int  top2;\n")
    (repo / "orig/x.cu").write_text("int  nested2;\n")
    (repo / "work/x.cu").write_text("int  w2;\n")
    _git(["add", "x.cu", "orig/x.cu", "work/x.cu"], repo)
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":x.cu"], repo).stdout == "int top2;\n"
    assert _git(["show", ":orig/x.cu"], repo).stdout == "int nested2;\n"
    assert _git(["show", ":work/x.cu"], repo).stdout == "int w2;\n"


def test_crlf_staged_bytes_survive_scratch_hashing(repo):
    # The scratch tree lives under TMPDIR, which can sit inside an unrelated
    # repo whose .gitattributes converts line endings; re-staging must hash the
    # exact blob bytes, not a filtered version.
    attr = repo / "attrtmp"
    attr.mkdir()
    _git(["init", "-q"], attr)
    (attr / ".gitattributes").write_text("* text=auto\n")
    (repo / "c.cu").write_bytes(b"int  crlf;\r\n")
    _git(["-c", "core.safecrlf=false", "add", "c.cu"], repo)
    assert _run(repo, "--staged", env_extra={"TMPDIR": str(attr)}).returncode == 0
    raw = subprocess.run(
        ["git", "show", ":c.cu"],
        cwd=str(repo),
        env=_isolated_env(),
        capture_output=True,
    ).stdout
    assert raw == b"int crlf;\r\n"  # formatted, line ending untouched


def test_abort_mid_run_restores_index_and_worktree(repo):
    # If the run dies between mutations (here: hash-object fails on the second
    # file), the EXIT trap must restore the index and every touched worktree
    # file to the pre-run state.
    base = "int a;\nint b;\nint c;\nint d;\nint e;\n"
    _commit_file(repo, "one.cu", base)
    _commit_file(repo, "two.cu", "int x;\n")
    (repo / "one.cu").write_text(base.replace("int a;", "int  A;"))
    (repo / "two.cu").write_text("int  X;\n")
    _git(["add", "one.cu", "two.cu"], repo)
    staged_one = (repo / "one.cu").read_text()
    (repo / "one.cu").write_text(staged_one.replace("int e;", "int  E;"))  # unstaged
    worktree_one = (repo / "one.cu").read_text()
    gitwrap = repo / "gitwrap"
    gitwrap.mkdir()
    (gitwrap / "git").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == hash-object && "$*" == *two.cu* ]]; then exit 42; fi\n'
        f'exec {shutil.which("git")} "$@"\n'
    )
    (gitwrap / "git").chmod(0o755)
    result = _run(
        repo,
        "--staged",
        env_extra={"PATH": f"{gitwrap}:{repo / 'bin'}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0
    assert "interrupted" in result.stderr
    # one.cu was mutated before the abort; the trap must have undone it.
    assert _git(["show", ":one.cu"], repo).stdout == staged_one
    assert (repo / "one.cu").read_text() == worktree_one
    assert _git(["show", ":two.cu"], repo).stdout == "int  X;\n"


def test_partial_near_miss_takes_note_path(repo):
    # An unstaged edit inside the delta's context lines (but not on the
    # formatted lines) still makes the patch unappliable: worktree left alone,
    # note printed, commit content formatted.
    base = "int a;\nint b;\nint c;\nint d;\nint e;\n"
    _commit_file(repo, "near.cu", base)
    (repo / "near.cu").write_text(base.replace("int a;", "int  A;"))  # stage line 1
    _git(["add", "near.cu"], repo)
    staged = (repo / "near.cu").read_text()
    worktree = staged.replace("int c;", "int  C;")  # line 3: within context
    (repo / "near.cu").write_text(worktree)
    result = _run(repo, "--staged")
    assert result.returncode == 0
    assert _git(["show", ":near.cu"], repo).stdout == base.replace("int a;", "int A;")
    assert (repo / "near.cu").read_text() == worktree  # byte-identical
    assert "left as is" in result.stderr


def test_formatter_line_count_change_syncs_worktree(repo):
    # The formatter itself can change the line count (blank-line squeezing);
    # the delta then shifts every line below it and must still apply around a
    # far-away unstaged edit.
    base = "int a;\n\nint b;\nint c;\nint d;\nint e;\nint f;\n"
    _commit_file(repo, "flow.cu", base)
    # A real change (a -> aa) bloated with blank lines the formatter removes.
    bloated = base.replace("int a;\n\n", "int  aa;\n\n\n\n")
    (repo / "flow.cu").write_text(bloated)
    _git(["add", "flow.cu"], repo)
    (repo / "flow.cu").write_text(bloated.replace("int f;", "int  F;"))  # unstaged
    assert _run(repo, "--staged").returncode == 0
    formatted = base.replace("int a;", "int aa;")  # squeezed + formatted
    assert _git(["show", ":flow.cu"], repo).stdout == formatted
    assert (repo / "flow.cu").read_text() == formatted.replace("int f;", "int  F;")


def test_partial_edit_above_formatting_applies_at_offset(repo):
    # An unstaged insertion above the formatting shifts every line below it;
    # the delta still lands on the right lines, found at its shifted offset.
    base = "".join(f"int {c};\n" for c in "abcdefgh")
    _commit_file(repo, "off.cu", base)
    (repo / "off.cu").write_text(base.replace("int h;", "int  H;"))  # stage line 8
    _git(["add", "off.cu"], repo)
    staged = (repo / "off.cu").read_text()
    (repo / "off.cu").write_text("int  X;\n" + staged)  # unstaged insert at top
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":off.cu"], repo).stdout == base.replace("int h;", "int H;")
    assert (repo / "off.cu").read_text() == "int  X;\n" + base.replace(
        "int h;", "int H;"
    )


def test_partial_overlap_skips_the_whole_file_delta(repo):
    # The delta is applied per file, all-or-nothing: when an unstaged edit
    # overlaps one of two formatting hunks, the other hunk is not applied
    # either -- the working tree is either fully synced or left alone.
    base = "".join(f"int {c};\n" for c in "abcdefghijkl")
    _commit_file(repo, "atomic.cu", base)
    two = base.replace("int a;", "int  A;").replace("int l;", "int  L;")
    (repo / "atomic.cu").write_text(two)  # stage two far-apart unformatted hunks
    _git(["add", "atomic.cu"], repo)
    (repo / "atomic.cu").write_text(
        two.replace("int  L;", "int  L2;")
    )  # overlap hunk 2
    result = _run(repo, "--staged")
    assert result.returncode == 0
    # Index: both hunks formatted.
    assert _git(["show", ":atomic.cu"], repo).stdout == base.replace(
        "int a;", "int A;"
    ).replace("int l;", "int L;")
    # Worktree: byte-identical -- hunk 1 was NOT applied on its own.
    assert (repo / "atomic.cu").read_text() == two.replace("int  L;", "int  L2;")
    assert "overlap" in result.stderr


def test_multiple_files_mixed_sync_outcomes(repo):
    # One run, three files: fully staged (mirrored), partial with the delta
    # applying (synced), partial with an overlap (left alone). Each file gets
    # its own outcome; all three commit formatted.
    base = "".join(f"int {c};\n" for c in "abcde")
    _commit_file(repo, "full.cu", "int f;\n")
    _commit_file(repo, "sync.cu", base)
    _commit_file(repo, "skip.cu", "int s;\n")
    (repo / "full.cu").write_text("int  F2;\n")  # fully staged
    (repo / "sync.cu").write_text(base.replace("int a;", "int  A;"))
    (repo / "skip.cu").write_text("int  s1;\n")
    _git(["add", "full.cu", "sync.cu", "skip.cu"], repo)
    synced = (repo / "sync.cu").read_text()
    (repo / "sync.cu").write_text(synced.replace("int e;", "int  E;"))  # far edit
    (repo / "skip.cu").write_text("int  s2;\n")  # overlapping edit
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":full.cu"], repo).stdout == "int F2;\n"
    assert (repo / "full.cu").read_text() == "int F2;\n"
    assert _git(["show", ":sync.cu"], repo).stdout == base.replace("int a;", "int A;")
    assert (repo / "sync.cu").read_text() == base.replace("int a;", "int A;").replace(
        "int e;", "int  E;"
    )
    assert _git(["show", ":skip.cu"], repo).stdout == "int s1;\n"
    assert (repo / "skip.cu").read_text() == "int  s2;\n"  # untouched


def test_partial_formatting_only_staged_change_is_refused(repo):
    # A formatting-only staged change on a partially-staged file normalizes to
    # HEAD: the empty commit is refused and the unstaged edit stays intact.
    base = "".join(f"int {c};\n" for c in "abcde")
    _commit_file(repo, "po.cu", base)
    (repo / "po.cu").write_text(base.replace("int a;", "int  a;"))  # fmt-only, staged
    _git(["add", "po.cu"], repo)
    staged = (repo / "po.cu").read_text()
    (repo / "po.cu").write_text(staged.replace("int e;", "int  E;"))  # far, unstaged
    result = _run(repo, "--staged")
    assert result.returncode != 0
    assert "nothing to commit" in result.stderr
    assert _git(["show", ":po.cu"], repo).stdout == base  # normalized to HEAD
    # The worktree kept the unstaged edit (formatting delta applied around it).
    assert (repo / "po.cu").read_text() == base.replace("int e;", "int  E;")


def test_partial_staged_formatted_content_passes(repo):
    # Already-formatted staged bytes on a partially-staged file are fine: the
    # commit proceeds and the unstaged edits stay untouched.
    _commit_file(repo, "p2.cu", "int a;\n")
    (repo / "p2.cu").write_text("int b;\n")  # staged: formatted
    _git(["add", "p2.cu"], repo)
    (repo / "p2.cu").write_text("int b;\nint  extra;\n")  # unstaged edit on top
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":p2.cu"], repo).stdout == "int b;\n"
    assert (repo / "p2.cu").read_text() == "int b;\nint  extra;\n"


def test_partial_staged_formatter_failure_preserves_unstaged(repo):
    # The staged content is formatted on a scratch copy first, so if the staged
    # bytes fail to format the working tree is never touched and the unstaged
    # edits survive.
    _commit_file(repo, "m.cu", "int m;\n")
    (repo / "m.cu").write_text("FAILME\n")  # the fake refuses to format this
    _git(["add", "m.cu"], repo)
    (repo / "m.cu").write_text("FAILME\nint precious;\n")  # unstaged edit on top
    result = _run(repo, "--staged")
    assert result.returncode != 0  # the format failure blocks the commit
    assert _git(["show", ":m.cu"], repo).stdout == "FAILME\n"  # index untouched
    # Worktree byte-identical: the unstaged edit is preserved on the staged bytes.
    assert (repo / "m.cu").read_text() == "FAILME\nint precious;\n"


def test_staged_formatting_only_change_is_refused(repo):
    _commit_file(repo, "only.cu", "int e;\n")  # HEAD is already formatted
    (repo / "only.cu").write_text("int  e;\n")  # a formatting-only regression
    _git(["add", "only.cu"], repo)
    result = _run(repo, "--staged")
    # Formatting normalizes the change away; committing it would be an empty
    # commit, so --staged refuses instead of recording one.
    assert result.returncode != 0
    assert "nothing to commit" in result.stderr
    assert "--allow-empty --no-verify" in result.stderr  # the escape hatch
    # Left formatted (== HEAD), not rolled back to the unformatted staged bytes.
    assert _git(["show", ":only.cu"], repo).stdout == "int e;\n"
    assert (repo / "only.cu").read_text() == "int e;\n"


def test_already_formatted_file_is_not_rewritten(repo):
    # A staged change that is already formatted must not have its working-tree
    # file rewritten, which would spuriously invalidate a build.
    import time

    _commit_file(repo, "c.cu", "int c;\n")
    (repo / "c.cu").write_text("int d;\n")  # a real change, already formatted
    _git(["add", "c.cu"], repo)
    before = (repo / "c.cu").stat().st_mtime_ns
    time.sleep(0.01)
    assert _run(repo, "--staged").returncode == 0
    assert (repo / "c.cu").stat().st_mtime_ns == before  # untouched


def test_same_name_files_in_different_dirs(repo):
    # Each file is formatted at its repo-relative path in the scratch tree, so
    # two files sharing a basename in different directories do not collide.
    (repo / "a").mkdir()
    (repo / "b").mkdir()
    _commit_file(repo, "a/f.cu", "int a;\n")
    _commit_file(repo, "b/f.cu", "int b;\n")
    (repo / "a/f.cu").write_text("int  A;\n")
    (repo / "b/f.cu").write_text("int  B;\n")
    _git(["add", "a/f.cu", "b/f.cu"], repo)
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":a/f.cu"], repo).stdout == "int A;\n"
    assert _git(["show", ":b/f.cu"], repo).stdout == "int B;\n"


def test_mixed_real_and_formatting_only_change_commits(repo):
    # The empty-commit guard fires only when the whole index equals HEAD, so a
    # real change staged alongside a formatting-only one still commits.
    _commit_file(repo, "real.cu", "int r;\n")
    _commit_file(repo, "fmt.cu", "int f;\n")
    (repo / "real.cu").write_text("int s;\n")  # real change, already formatted
    (repo / "fmt.cu").write_text("int  f;\n")  # formatting-only regression
    _git(["add", "real.cu", "fmt.cu"], repo)
    assert _run(repo, "--staged").returncode == 0  # not refused
    assert _git(["show", ":real.cu"], repo).stdout == "int s;\n"
    assert _git(["show", ":fmt.cu"], repo).stdout == "int f;\n"


def test_check_staged_judges_staged_bytes_only(repo):
    # --staged --check reads each staged blob into a scratch tree and checks
    # that. The staged bytes differ from both HEAD and the working tree here, so
    # a check that judged either of those instead would give the opposite
    # verdict.
    _commit_file(repo, "p.cu", "int p;\n")
    (repo / "p.cu").write_text("int q;\n")  # staged: formatted, differs from HEAD
    _git(["add", "p.cu"], repo)
    (repo / "p.cu").write_text("int  q;\nint  r;\n")  # worktree: unformatted
    assert _run(repo, "--staged", "--check").returncode == 0
    # And unformatted staged bytes fail even when the working tree is formatted.
    (repo / "p.cu").write_text("int  s;\n")
    _git(["add", "p.cu"], repo)
    (repo / "p.cu").write_text("int s;\n")  # worktree: formatted
    assert _run(repo, "--staged", "--check").returncode != 0


def test_staged_ignores_unstaged_only_bystander(repo):
    # --staged selects index-vs-HEAD, so a file with only unstaged edits (its
    # staged bytes equal HEAD) is not this commit's content and must be ignored
    # -- even when those staged-equal bytes are unformatted.
    _commit_file(repo, "by.cu", "int  b;\n")  # committed unformatted
    (repo / "by.cu").write_text("int  b;\nint  c;\n")  # unstaged edit, never staged
    _commit_file(repo, "ok.cu", "int o;\n")
    (repo / "ok.cu").write_text("int k;\n")  # the actual staged change
    _git(["add", "ok.cu"], repo)
    assert _run(repo, "--staged").returncode == 0
    assert _git(["show", ":by.cu"], repo).stdout == "int  b;\n"  # untouched
    assert (repo / "by.cu").read_text() == "int  b;\nint  c;\n"


def test_staged_symlink_is_left_alone(repo):
    # A symlink's staged blob is its target string; both staged modes must skip
    # it rather than format the string or write through the link. The target
    # name carries a double space, so a missing skip would "format" the blob.
    (repo / "link.cu").symlink_to("tgt  t.cu")
    _git(["add", "link.cu"], repo)
    assert _run(repo, "--staged").returncode == 0
    assert _run(repo, "--staged", "--check").returncode == 0
    entry = _git(["ls-files", "--stage", "--", "link.cu"], repo).stdout
    assert entry.startswith("120000")  # still a symlink in the index
    assert _git(["show", ":link.cu"], repo).stdout == "tgt  t.cu"  # blob untouched
    assert (repo / "link.cu").is_symlink()
    assert not (repo / "tgt  t.cu").exists()  # nothing written through the link


def test_precommit_formats_and_restages(repo):
    # --precommit must perform (format + re-stage), not merely check: a revert to
    # --staged --check would leave the staged blob unformatted and exit non-zero.
    _commit_file(repo, "pc.cu", "int p;\n")
    (repo / "pc.cu").write_text("int  q;\n")
    _git(["add", "pc.cu"], repo)
    assert _run(repo, "--precommit").returncode == 0
    assert _git(["show", ":pc.cu"], repo).stdout == "int q;\n"


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


def test_local_mode_checks_only_changed_files(repo):
    # With no mode flag the default is local: only files that differ from HEAD
    # are considered. clean.cu is committed unformatted, so a whole-tree default
    # would flag it -- local must not even look at it.
    _commit_file(repo, "clean.cu", "int  c;\n")
    _commit_file(repo, "dirty.cu", "int d;\n")
    (repo / "dirty.cu").write_text("int  d;\n")  # local change, unformatted
    result = _run(repo, "--check")  # no mode flag -> local
    assert result.returncode != 0  # dirty.cu is unformatted
    out = result.stdout + result.stderr
    assert "dirty.cu" in out
    assert "clean.cu" not in out  # an unchanged file is never inspected


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
