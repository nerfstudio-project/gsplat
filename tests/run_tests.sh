#!/bin/bash

set -euo pipefail

SDIR=$(dirname "$(readlink -f "$0")")
REPOROOT=$(realpath -e "$SDIR/..")

source "$REPOROOT/docker/utils.sh"

LOCAL_CACHE_VOLUME=gsplat-cache
LOCAL_HOME_VOLUME=gsplat-home-$(id -un)

usage()
{
    echo "Build gsplat and run its tests inside its docker container"
    echo
    echo "Usage: ${0##*/} [global flags] [feature flags] [ENVVAR=value ...] [extra shell/CTest args]"
    echo "global flags:"
    echo "   --configure  Stop after configuring the CMake build tree"
    echo "   --build    Configure and build; do not run tests"
    echo "   --test     Configure, build, and run CTest (default)"
    echo "   --shell    Enter into the shell inside the container."
    echo "   --sanitize Build dev-debug and run CTest under CUDA compute-sanitizer"
    echo "   --reset-cache  Delete the internal cache directory"
    echo "   --reset-home   Delete the internal home directory"
    echo "   --verbose  Show the Docker invocation and verbose build/test output"
    echo "   --gpus=spec Use the given GPUs inside the container."
    echo "              The syntax is:"
    echo "                 --gpus=<count>              - use this many GPUs"
    echo "                 --gpus=device=<id1,id2,...> - use the GPUs given by their device index"
    echo "                 --gpus=all                  - use all GPUs (default)"
    echo "   --listen=[HOST:]PORT | -p [HOST:]PORT"
    echo "              Expose a container port on the host."
    echo "              Can be given  multiple times."
    echo "              PORT maps container:PORT to host:PORT;"
    echo "              HOST:PORT maps container:PORT to host:HOST."
    echo "   --ssh[=PORT]"
    echo "              Start an SSH server in the container and forward PORT (default: 2222)"
    echo "              to port 22 inside the container. Requires ~/.ssh/authorized_keys."
    echo "              Runs sshd in the foreground; press Ctrl-C to stop."
    echo "   --help|-h  Show this help message"
    echo "   --debug    Shorthand for --preset=dev-debug"
    echo "   --preset=NAME Use the given CMake configure preset instead of"
    echo "              dev-release (or dev-debug for --sanitize)."
    echo "ENVVAR=value:"
    echo "   Environment variables can be passed to the container."
    echo "feature flags"
    echo "   --2dgs     Build 2dgs"
    echo "   --3dgs     Build 3dgs"
    echo "   --3dgut    Build 3dgut"
    echo "   --all      Build all families"
    echo "Feature flags apply at configure time and define the complete"
    echo "selection: passing any of them reconfigures the preset's build tree"
    echo "for exactly those families. Without feature flags the tree keeps its"
    echo "current selection; a fresh tree builds all families."
    echo
    echo "Runner flags can be given via the envvar GSPLAT_TEST_PARAMS."
    echo "CTest and shell-command arguments must be passed on the command line."
    echo "Command-line runner flags are processed after the environment defaults."
    echo
    echo "Container build trees live inside the repository at"
    echo "build/docker/<preset>, so the host sees the same tree the"
    echo "container writes (artifacts, compile_commands.json, CTest logs)."
    echo "Those directories belong to the container toolchain; do not configure"
    echo "them with a host cmake."
    echo
    echo "Refer to README.md for usage examples."
}

runshell=false

step=""
preset_override=""
do_3dgut=false
do_3dgs=false
do_2dgs=false
do_all_features=false
do_reset_home=false
do_reset_cache=false
do_sanitize=false
do_debug=false
do_verbose=false
gpus=all

envvars=()
run_args=()
port_specs=()
do_ssh=false
ssh_port=2222

# Prepend default runner flags from the environment. CTest or shell-command
# arguments belong on the command line, where the parser can leave them intact.
if [[ -v GSPLAT_TEST_PARAMS ]]; then
    set -- ${GSPLAT_TEST_PARAMS} "$@"
fi

while (( $# >= 1 )); do
    case $1 in
    --shell)
        runshell=true
        ;;
    --configure)
        step=configure
        ;;
    --build)
        step=build
        ;;
    --test)
        step=test
        ;;
    --reset-cache)
        do_reset_cache=true
        ;;
    --reset-home)
        do_reset_home=true
        ;;
    --sanitize)
        do_sanitize=true
        ;;
    --3dgut)
        do_3dgut=true
        ;;
    --3dgs)
        do_3dgs=true
        ;;
    --2dgs)
        do_2dgs=true
        ;;
    --all)
        do_all_features=true
        ;;
    --debug)
        do_debug=true
        ;;
    --preset=*)
        preset_override=${1#--preset=}
        if [[ -z $preset_override ]]; then
            echo "Error: --preset requires a preset name." >&2
            exit 1
        fi
        ;;
    --verbose)
        do_verbose=true
        ;;
    --gpus=*)
        gpus=${1#--gpus=}
        ;;
    --listen=*)
        val="${1#*=}"
        if [[ -z "$val" ]]; then
            echo "Error: --listen requires a port specification." >&2
            exit 1
        fi
        if [[ "$val" != *:* ]]; then
            val="$val:$val"
        fi
        port_specs+=("$val")
        ;;
    -p)
        if (( $# < 2 )); then
            echo "Error: -p requires a port specification." >&2
            exit 1
        fi
        shift
        val="$1"
        if [[ "$val" != *:* ]]; then
            val="$val:$val"
        fi
        port_specs+=("$val")
        ;;
    --ssh)
        do_ssh=true
        ssh_port=2222
        ;;
    --ssh=*)
        do_ssh=true
        ssh_port="${1#--ssh=}"
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    --)
        # All remaining parameters will be given to the container
        shift
        break
        ;;
    *)
        # Is it a envvar spec?
        if [[ $1 =~ ^[a-zA-Z_]+[a-zA-Z0-9_]*= ]]; then
            envvars+=("$1")
        else
            # If not, end processing, current parameter will be given
            # to the container
            break
        fi
        ;;
    esac
    shift
done

if $runshell && $do_ssh; then
    echo "Error: --shell and --ssh are mutually exclusive." >&2
    exit 1
fi
if [[ -n $step ]] && { $runshell || $do_ssh; }; then
    echo "Error: --configure/--build/--test cannot be combined with --shell or --ssh." >&2
    exit 1
fi
if $do_sanitize && [[ -n $step && $step != test ]]; then
    echo "Error: --sanitize only applies to the test step." >&2
    exit 1
fi
if $do_sanitize && { $runshell || $do_ssh; }; then
    echo "Error: --sanitize cannot be combined with --shell or --ssh." >&2
    exit 1
fi
if $do_ssh && [[ $# -gt 0 ]]; then
    echo "Error: --ssh cannot be combined with an explicit command." >&2
    exit 1
fi

if $runshell; then
    step=shell
elif $do_ssh; then
    step=ssh
elif [[ -z $step ]]; then
    step=test
fi

check_if_installed docker nvidia-container-runtime

# Load config variables
load_config "$REPOROOT/config.yaml"

# Per-image-version venv volume. Docker auto-populates a fresh volume from
# the image's /var/cache/venv-${IMAGE_TAG} on first mount, so runtime package
# state survives --rm and image bumps coexist without forcing a cache reset.
LOCAL_VENV_VOLUME="gsplat-venv-$IMAGE_TAG"

if $do_reset_home || $do_reset_cache; then
    if $do_reset_cache; then
        echo -n "Removing container cache volume... " >&2
        docker volume rm "$LOCAL_CACHE_VOLUME" > /dev/null 2>&1 || true
        echo "OK"
        echo -n "Removing venv volume $LOCAL_VENV_VOLUME... " >&2
        docker volume rm "$LOCAL_VENV_VOLUME" > /dev/null 2>&1 || true
        echo "OK"
    fi
    if $do_reset_home; then
        echo -n "Removing container home volume... " >&2
        docker volume rm "$LOCAL_HOME_VOLUME" > /dev/null 2>&1 || true
        echo "OK"
    fi
    exit
fi


HOST_USER="$(id -un)"
HOST_GROUP="$(id -gn)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_HOME="$HOME"
RUN_AS="$HOST_USER"

if $do_ssh; then
    host_ip="$(ip route get 1.1.1.1 | sed -n 's/.* src \([^ ]*\).*/\1/p')"
    ssh_remote="$HOST_USER@${host_ip:-localhost}"
    container_name="gsplat-ssh-${HOST_USER}-$ssh_port"
    run_args+=(--name "$container_name")

    existing_id=$(docker ps --format '{{.ID}} {{.Ports}}' \
                  | awk -v port=":$ssh_port->" '$0 ~ port {print $1}')
    if [[ -n "$existing_id" ]]; then
        existing_image=$(docker inspect "$existing_id" --format '{{.Image}}')
        existing_name=$(docker inspect "$existing_id" --format '{{.Name}}' | sed 's|^/||')
        new_image=$(docker image inspect "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" \
                    --format '{{.Id}}' 2>/dev/null || true)
        if [[ "$existing_image" == "$new_image" ]]; then
            echo "SSH container ('$existing_name') already running with the current image." >&2
            echo "Connect with: ssh -p $ssh_port $ssh_remote" >&2
            exit 0
        else
            echo "Error: a container ('$existing_name') is already using port $ssh_port with a stale image." >&2
            echo "Stop it first with: docker stop $existing_name" >&2
            exit 1
        fi
    fi

    run_args+=(-ti)
    run_args+=(--init)
    run_args+=(--publish "$ssh_port:22")
    RUN_AS=root
    if [[ -f "$HOME/.ssh/authorized_keys" ]]; then
        run_args+=(-v "$HOME/.ssh/authorized_keys:$HOST_HOME/.ssh/authorized_keys:ro")
    else
        echo "Warning: $HOME/.ssh/authorized_keys not found; SSH key authentication unavailable." >&2
    fi

    echo "Starting SSH container. Connect with:" >&2
    echo "  ssh -p $ssh_port $ssh_remote" >&2
fi

run_args+=(
    "--gpus=$gpus"
    --rm
    -v "$REPOROOT:$REPOROOT"
    -w "$REPOROOT"

    -e HOST_USER="$HOST_USER"
    -e HOST_GROUP="$HOST_GROUP"
    -e HOST_UID="$HOST_UID"
    -e HOST_GID="$HOST_GID"
    -e HOST_HOME="$HOST_HOME"
    -e RUN_AS="$RUN_AS"
    -e TERM="$TERM"

    -v "$LOCAL_HOME_VOLUME:$HOST_HOME"
    -v "$LOCAL_CACHE_VOLUME:/var/cache"
    -v "$LOCAL_VENV_VOLUME:/var/cache/venv-$IMAGE_TAG"

    --hostname "$(hostname)-gsdev"
    --ipc=host
)

# A linked Git worktree stores its administrative directory outside the
# worktree root. Source-level tests call Git through the worktree's `.git`
# pointer, so expose that external common directory at the same absolute path.
# It is metadata-only for this workflow and remains read-only in the container.
git_common_dir=$(git -C "$REPOROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [[ -n $git_common_dir && $git_common_dir != "$REPOROOT/.git" ]]; then
    run_args+=(-v "$git_common_dir:$git_common_dir:ro")
fi

for port_spec in "${port_specs[@]}"; do
    run_args+=(--publish "$port_spec")
done

# Add user envvars as -e KEY=VALUE pairs without breaking on spaces
for kv in "${envvars[@]}"; do
    run_args+=(-e "$kv")
done


if $do_all_features && { $do_2dgs || $do_3dgs || $do_3dgut; }; then
    echo "Error: --all cannot be combined with individual feature flags." >&2
    exit 1
fi

# Feature flags define the complete selection for this invocation. The
# container reconfigures the preset's tree only when a selection is given;
# otherwise the tree keeps whatever selection its cache already holds.
features_given=false
if $do_2dgs; then
    features_given=true
    run_args+=(-e GSPLAT_TEST_BUILD_2DGS=1)
fi
if $do_3dgs; then
    features_given=true
    run_args+=(-e GSPLAT_TEST_BUILD_3DGS=1)
fi
if $do_3dgut; then
    features_given=true
    run_args+=(-e GSPLAT_TEST_BUILD_3DGUT=1)
fi
if $do_all_features; then
    features_given=true
fi
if $features_given; then
    run_args+=(-e GSPLAT_TEST_FEATURES_GIVEN=1)
fi

build_preset=dev-release
if $do_debug || $do_sanitize; then
    build_preset=dev-debug
fi
if [[ -n $preset_override ]]; then
    if $do_debug; then
        echo "Error: --preset and --debug are contradictory; pick one." >&2
        exit 1
    fi
    build_preset=$preset_override
fi

# Keep the per-preset CMake trees inside the repository, which is mounted at
# the same path as on the host: the host then sees the exact tree the
# container writes (artifacts, compile_commands.json, CTest logs).
container_build_dir="$REPOROOT/build/docker/${build_preset}"

run_args+=(
    -e GSPLAT_CMAKE_PRESET="$build_preset"
    -e GSPLAT_BUILD_DIR="$container_build_dir"
    -e GSPLAT_IMAGE_TAG="$IMAGE_TAG"
)

# Share the host's ccache with the container toolchain, when the host has one
# with an existing cache directory:
# - ccache creates the directory lazily; a configured but absent one keeps the
#   fallback
# - the container resolves CCACHE_DIR to /var/cache/<user>/ccache, so a bind
#   mount at that path takes precedence over the cache volume's subtree
# - the host's cache-resident ccache.conf (size limits) rides along
host_ccache_dir=""
if command -v ccache > /dev/null 2>&1; then
    host_ccache_dir=$(ccache --get-config cache_dir 2>/dev/null || true)
fi
if [[ -n $host_ccache_dir && -d $host_ccache_dir ]]; then
    run_args+=(-v "$host_ccache_dir:/var/cache/$HOST_USER/ccache")
    # The host's size limit usually lives in the user configuration, which the
    # container's home volume shadows. Carry it through the environment so the
    # container never evicts the shared cache down to the image's default.
    host_ccache_max=$(ccache --get-config max_size 2>/dev/null || true)
    if [[ -n $host_ccache_max ]]; then
        run_args+=(-e CCACHE_MAXSIZE="$host_ccache_max")
    fi
fi

if $do_verbose; then
    run_args+=(-e GSPLAT_TEST_VERBOSE=1)
fi

run_args+=(-e GSPLAT_TEST_STEP="$step")
if [[ $step == shell && $# == 0 ]]; then
    run_args+=(-ti)
fi

if $do_sanitize; then
    run_args+=(-e GSPLAT_TEST_SANITIZE=1)
    # compute-sanitizer can require ptrace capabilities unavailable in the
    # default container security profile.
    run_args+=(--privileged)
fi

container_script='
set -euo pipefail

# A given selection is complete: GSPLAT_KERNEL_FAMILIES names exactly the
# families to build (empty selects all), so explicit choices from a previous
# invocation cannot leak into this one. Without a selection the cached value
# stays untouched.
cmake_args=()
if [[ ${GSPLAT_TEST_FEATURES_GIVEN:-0} == 1 ]]; then
    selection=""
    for family in 2DGS 3DGS 3DGUT; do
        var="GSPLAT_TEST_BUILD_${family}"
        if [[ ${!var:-0} == 1 ]]; then
            selection="${selection:+${selection},}${family}"
        fi
    done
    cmake_args+=("-DGSPLAT_KERNEL_FAMILIES=${selection}")
fi

build_args=()
if [[ ${GSPLAT_TEST_VERBOSE:-0} == 1 ]]; then
    set -x
    build_args+=(--verbose)
fi

# SSH keeps PID 1 as root so sshd can accept logins. Run each build command in
# a login shell belonging to the mapped host user, so profile-based cache paths
# are initialized with the correct HOME and the persistent tree remains owned
# by that user. Other modes already run this script as the host user.
run_builder()
{
    if [[ $GSPLAT_TEST_STEP != ssh ]]; then
        "$@"
    elif [[ ${HOST_UID:-0} == 0 ]]; then
        /bin/bash -lc '\''exec "$@"'\'' bash "$@"
    else
        gosu "$HOST_USER" \
            env "HOME=$HOST_HOME" \
            /bin/bash -lc '\''exec "$@"'\'' bash "$@"
    fi
}

echo "Container build tree: $GSPLAT_BUILD_DIR" >&2

# The docker build trees are container-owned: never adopt a CMake cache this
# workflow did not write, and reconfigure from scratch when the image
# toolchain changed under an existing tree.
marker="$GSPLAT_BUILD_DIR/.gsplat-docker-image-tag"
fresh_args=()
if [[ -f "$GSPLAT_BUILD_DIR/CMakeCache.txt" && ! -f "$marker" ]]; then
    echo "ERROR: $GSPLAT_BUILD_DIR was not configured by this container workflow" >&2
    echo "       (missing ${marker##*/}). Remove the directory to let the container own it." >&2
    exit 1
fi
if [[ -f "$marker" ]] && [[ $(< "$marker") != "$GSPLAT_IMAGE_TAG" ]]; then
    echo "Docker image tag changed ($(< "$marker") -> $GSPLAT_IMAGE_TAG); reconfiguring the build tree." >&2
    fresh_args+=(--fresh)
fi

run_builder cmake \
    --preset "$GSPLAT_CMAKE_PRESET" \
    -S "$PWD" \
    -B "$GSPLAT_BUILD_DIR" \
    "${fresh_args[@]}" \
    "${cmake_args[@]}"
run_builder bash -c '\''printf "%s\n" "$GSPLAT_IMAGE_TAG" > "$GSPLAT_BUILD_DIR/.gsplat-docker-image-tag"'\''

if [[ $GSPLAT_TEST_STEP == configure ]]; then
    exit 0
fi

run_builder cmake \
    --build "$GSPLAT_BUILD_DIR" \
    "${build_args[@]}"

case $GSPLAT_TEST_STEP in
build)
    exit 0
    ;;
test)
    ctest_args=(
        --test-dir "$GSPLAT_BUILD_DIR"
        --output-on-failure
    )
    if [[ ${GSPLAT_TEST_VERBOSE:-0} == 1 ]]; then
        ctest_args+=(--verbose)
    fi
    if [[ ${GSPLAT_TEST_SANITIZE:-0} == 1 ]]; then
        exec /usr/local/cuda/bin/compute-sanitizer \
            --target-processes all \
            --error-exitcode 1 \
            ctest "${ctest_args[@]}" "$@"
    fi
    exec ctest "${ctest_args[@]}" "$@"
    ;;
shell)
    export PYTHONPATH="$GSPLAT_BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}"
    if (( $# > 0 )); then
        exec "$@"
    fi
    exec /bin/bash -l
    ;;
ssh)
    # PAM reads this file for SSH login sessions. Replace dynamic entries from
    # any earlier container before publishing this build tree.
    sed -i \
        -e "/^GSPLAT_BUILD_DIR=/d" \
        -e "/^PYTHONPATH=/d" \
        /etc/environment
    printf '\''GSPLAT_BUILD_DIR="%s"\nPYTHONPATH="%s"\n'\'' \
        "$GSPLAT_BUILD_DIR" \
        "$GSPLAT_BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        >> /etc/environment
    exec /usr/sbin/sshd -D -e
    ;;
*)
    echo "Unknown GSPLAT_TEST_STEP: $GSPLAT_TEST_STEP" >&2
    exit 2
    ;;
esac
'

if $do_ssh; then
    # Never evaluate the mapped user\'s login profile in the root sshd process.
    shell_args=(/bin/bash -c "$container_script" bash)
else
    shell_args=(/bin/bash -lc "$container_script" bash)
fi

if $do_verbose; then
    # Show the whole docker run invocation
    set -x
fi

docker run "${run_args[@]}" "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" "${shell_args[@]}" "$@"
