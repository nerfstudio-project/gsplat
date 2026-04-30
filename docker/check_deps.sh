#!/bin/bash
set -euo pipefail

usage()
{
    cat << EOF
${0##*/} — Validate docker pip dependency lists against source files.

Exits 0 if all checks pass, 1 on any mismatch.

Usage: ${0##*/} [-f <file>[:<section>]]... -- <pkg1> <pkg2> ...

Source files (-f):
  -f requirements.txt           requirements-format file
  -f setup.py:install           setup.py install_requires
  -f setup.py:dev               setup.py extras_require["dev"]

Packages in the docker image (after --):
  The exact dependency strings from the docker image to check against
  the source files.

Source files are processed in order.  Each docker image package must
satisfy the specs of every source file that mentions it.  As soon as
a package fails a check in any file, it is marked as failing and
skipped in subsequent files.

Checks:
  - Every source dep has a matching docker image package (by normalized name)
  - Every docker image package has a matching source dep (by normalized name)
  - Pinned versions in the docker image satisfy source version specs
  - Git URLs reference the same repo and commit
EOF
}

die()
{
    echo "ERROR: $1"
    usage
    exit 2
} >&2

# ── Name normalization (PEP 503) ────────────────────────────────────────
# Lowercase, then replace every run of [-_.] with a single dash.

normalize_name()
{
    local name=${1,,}
    name="${name//[-_.]/-}"
    while [[ $name == *--* ]]; do
        name="${name//--/-}"
    done
    echo "$name"
}

# ── Source file parsers ──────────────────────────────────────────────────

# Extract dependency strings from a setup.py section.
# Imports the setup.py as a module (its setup() call is gated on
# __name__ == "__main__") and reads INSTALL_REQUIRES / get_extras_require()
# directly, so the parser sees whatever the build sees — no string-shape
# assumptions.
extract_setup_section()
{
    local file=$1 section=$2

    python3 - "$file" "$section" <<'PY'
import importlib.util
import sys

setup_path, section = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("setup", setup_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if section == "install":
    items = module.INSTALL_REQUIRES
else:
    items = module.get_extras_require()[section]

print("\n".join(items))
PY
}

# Extract deps from requirements.txt (skip comments and blank lines).
# Inline comments are stripped by matching " #" (space before hash), which
# preserves URL fragments like git+https://...#egg=foo (no space before #).
extract_requirements()
{
    sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d; s/ #.*$//' "$1"
}

# ── Dependency string field extraction ───────────────────────────────────
#
# Formats handled:
#   ninja                                        bare name
#   rich>=12                                     name + constraint
#   black[jupyter]==22.3.0                       name + extras + constraint
#   typing_extensions; python_version<'3.8'      name + marker
#   PLAS @ git+https://...                       name + URL
#   git+https://github.com/user/repo@ref         bare git URL

# Package name from a dep string.
dep_name()
{
    local dep=$1

    # Bare git URL → repo name from the last path segment (any host)
    if [[ $dep == git+* ]]; then
        sed -nE 's|.*/([^/@.]*)(\.git)?(@.*)?$|\1|p' <<< "$dep"
        return
    fi

    dep=${dep%%;*}             # strip markers          typing_extensions; ... → typing_extensions
    dep=${dep%% @*}            # strip URL               PLAS @ git+... → PLAS
    dep=${dep%%\[*}            # strip extras            black[jupyter]==... → black
    dep=${dep%%[><=!~]*}       # strip version ops       rich>=12 → rich
    echo "${dep// /}"
}

# Version constraint (e.g. ">=12", "==1.0.0", ">=1,<2") from a dep string.
dep_version_constraint()
{
    local dep=$1
    dep=${dep%%;*}             # strip markers
    dep=${dep%% @*}            # strip URL
    local prefix=${dep%%[><=!~]*}
    echo "${dep#"$prefix"}"
}

# Pinned version from a constraint — non-empty only for "==X.Y.Z" pins.
pinned_version()
{
    local constraint=$1
    if [[ $constraint == ==* ]]; then
        echo "${constraint#==}"
    fi
}

# Git URL from a dep string (empty if none).
dep_git_url()
{
    local dep=$1
    if [[ $dep == git+* ]]; then
        echo "$dep"
    elif [[ $dep == *" @ "* ]]; then
        echo "${dep#* @ }"
    fi
}

# Base repository URL (lowercased, .git stripped, @ref stripped).
git_url_base()
{
    local url=${1%@*}          # strip @ref
    url=${url%.git}
    echo "${url,,}"
}

# Ref (commit hash or tag) from a git URL (empty if none).
git_url_ref()
{
    local url=$1
    if [[ $url == *@* ]]; then
        echo "${url##*@}"
    fi
}

# Check that two git URLs reference the same repo and commit.
# Args: name docker_pkg_url src_url
# Prints an error message and returns 1 on mismatch, 0 on match.
check_git_urls()
{
    local name=$1 docker_pkg_url=$2 src_url=$3

    local docker_pkg_base src_base
    docker_pkg_base=$(git_url_base "$docker_pkg_url")
    src_base=$(git_url_base "$src_url")

    if [[ $docker_pkg_base != "$src_base" ]]; then
        printf "Git repo mismatch for '%s':\n    docker: %s\n    source:     %s\n" \
            "$name" "$docker_pkg_url" "$src_url"
        return 1
    fi

    local docker_pkg_ref src_ref
    docker_pkg_ref=$(git_url_ref "$docker_pkg_url")
    src_ref=$(git_url_ref "$src_url")

    if [[ -n $docker_pkg_ref && -n $src_ref && $docker_pkg_ref != "$src_ref" ]]; then
        printf "Git ref mismatch for '%s':\n    docker: %s\n    source:     %s\n" \
            "$name" "$docker_pkg_ref" "$src_ref"
        return 1
    fi

    return 0
}

# ── Version constraint check (the only part that needs Python) ───────────
# Reads tab-separated "file_spec\tname\tpinned\tverspec" lines on stdin.
# Prints one error message per failed check on stdout.
# Exits 0 when all pass, 1 when any fail.

check_versions()
{
    python3 -c '
import sys
try:
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
except ImportError:
    from pip._vendor.packaging.version import Version
    from pip._vendor.packaging.specifiers import SpecifierSet

failed = False
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    file_spec, name, pinned, verspec = line.split("\t", 3)
    try:
        if Version(pinned) not in SpecifierSet(verspec):
            print(f"[{file_spec}] Docker image has {name}=={pinned} but source requires '\''{verspec}'\''")
            failed = True
    except Exception as e:
        print(f"[{file_spec}] Warning: could not parse version check for {name}: {e}", file=sys.stderr)
sys.exit(1 if failed else 0)
'
}

# ── CLI ──────────────────────────────────────────────────────────────────

file_specs=()
while (( $# > 0 )); do
    case "$1" in
        -f)
            if (( $# < 2 )); then
                die "-f requires an argument"
            fi
            file_specs+=("$2")
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "Unknown flag: $1"
            ;;
        *)
            break
            ;;
    esac
done

if (( ${#file_specs[@]} == 0 )); then
    die "at least one -f <file> is required"
fi

# ── Collect packages ────────────────────────────────────────────────────

declare -A docker_pkg_raw docker_pkg_verspec docker_pkg_url

for dep in "$@"; do
    if [[ -z $dep ]]; then
        continue
    fi
    name=$(normalize_name "$(dep_name "$dep")")
    if [[ -z $name ]]; then
        continue
    fi
    docker_pkg_raw[$name]=$dep
    docker_pkg_verspec[$name]=$(dep_version_constraint "$dep")
    docker_pkg_url[$name]=$(dep_git_url "$dep")
done

# ── Check packages against each source file in order ───────────────────
#
# For every source dep: the matching package must exist and satisfy
# that file's specs.  A package that fails any file is marked
# as failed and skipped in later files.

errors=()
declare -A failed_docker_pkgs     # packages already marked as failing
declare -A all_spec_names   # union of all source dep names (for coverage)
version_checks=()         # batched version checks (tab-separated lines)

# Validate file specs up front (must run in the main shell so die works)
for file_spec in "${file_specs[@]}"; do
    if [[ $file_spec == *:* ]]; then
        case ${file_spec##*:} in
            install|dev)
                ;;
            *)
                die "unknown setup.py section '${file_spec##*:}' in '$file_spec'"
                ;;
        esac
    fi
done

for file_spec in "${file_specs[@]}"; do
    if [[ $file_spec == *:* ]]; then
        readarray -t src_deps < <(extract_setup_section "${file_spec%:*}" "${file_spec##*:}")
    else
        readarray -t src_deps < <(extract_requirements "$file_spec")
    fi

    for src_dep in "${src_deps[@]}"; do
        if [[ -z $src_dep ]]; then
            continue
        fi
        name=$(normalize_name "$(dep_name "$src_dep")")
        if [[ -z $name ]]; then
            continue
        fi

        all_spec_names[$name]=1

        # Already failed from an earlier file — don't pile on
        if [[ -n ${failed_docker_pkgs[$name]+x} ]]; then
            continue
        fi

        # 1. Every source dep must have a matching package
        if [[ -z ${docker_pkg_raw[$name]+x} ]]; then
            errors+=("[$file_spec] '$src_dep' ($name) is in source but not in docker image")
            failed_docker_pkgs[$name]=1
            continue
        fi

        # 2. Git URLs must reference the same repo and commit
        spec_url=$(dep_git_url "$src_dep")
        docker_pkg_url_val=${docker_pkg_url[$name]}
        if [[ -n $spec_url && -n $docker_pkg_url_val ]]; then
            git_err=$(check_git_urls "$name" "$docker_pkg_url_val" "$spec_url") || {
                errors+=("[$file_spec] $git_err")
                failed_docker_pkgs[$name]=1
                continue
            }
        fi

        # 3. Pinned version must satisfy this file's version spec
        spec_verspec=$(dep_version_constraint "$src_dep")
        docker_pkg_pin=$(pinned_version "${docker_pkg_verspec[$name]}")
        if [[ -n $docker_pkg_pin && -n $spec_verspec ]]; then
            version_checks+=("${file_spec}	${name}	${docker_pkg_pin}	${spec_verspec}")
        fi
    done
done

# Run batched version checks
if (( ${#version_checks[@]} > 0 )); then
    ver_errors=$(printf '%s\n' "${version_checks[@]}" | check_versions) || true
    if [[ -n $ver_errors ]]; then
        while IFS= read -r line; do
            errors+=("$line")
        done <<< "$ver_errors"
    fi
fi

# 5. Every package must appear in some source file
while IFS= read -r name; do
    if [[ -z ${all_spec_names[$name]+x} ]]; then
        errors+=("'${docker_pkg_raw[$name]}' ($name) is in the docker image but not in any source file")
    fi
done < <(printf '%s\n' "${!docker_pkg_raw[@]}" | sort)

# ── Report ──────────────────────────────────────────────────────────────

if (( ${#errors[@]} > 0 )); then
    for e in "${errors[@]}"; do
        echo "$e" >&2
    done
    echo >&2
    echo "check_deps: FAILED (see errors above)." >&2
    echo "The docker image package list isn't in sync with the packages in the source files." >&2
    exit 1
fi
