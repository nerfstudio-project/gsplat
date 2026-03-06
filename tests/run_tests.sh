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
    echo "Usage: ${0##*/} [global flags] [feature flags] [ENVVAR=value ...] [extra shell/pytest args]"
    echo "global flags:"
    echo "   --shell    Enter into the shell inside the container."
    echo "   --sanitize Run tests under CUDA compute-sanitizer"
    echo "   --reset-cache  Delete the internal cache directory"
    echo "   --reset-home   Delete the internal home directory"
    echo "   --verbose  Show intermediate information"
    echo "   --gpus=spec Use the given GPUs inside the container."
    echo "              The syntax is:"
    echo "                 --gpus=<count>              - use this many GPUs"
    echo "                 --gpus=device=<id1,id2,...> - use the GPUs given by their device index"
    echo "                 --gpus=all                  - use all GPUs (default)"
    echo "   --help|-h  Show this help message"
    echo "   --debug    Show the docker run invocation"
    echo "ENVVAR=value:"
    echo "   Environment variables can be passed to the container."
    echo "feature flags"
    echo "   --2dgs     Build 2dgs"
    echo "   --3dgs     Build 3dgs"
    echo "   --3dgut    Build 3dgut "
    echo "The multiple feature flags can be given,"
    echo "only the selected features will be built."
    echo "If none are given, all features will be built."
    echo
    echo "Parameters can be given via the envvar GSPLAT_TEST_PARAMS."
    echo "The parameters given in the command line have precedence."
    echo
    echo "Refer to README.md for usage examples."
}

runshell=false

do_3dgut=false
do_3dgs=false
do_2dgs=false
do_reset_home=false
do_reset_cache=false
do_sanitize=false
do_debug=false
do_verbose=false
gpus=all

envvars=()

# Prepend the parameters given by the environment variable, if any.
if [[ -v GSPLAT_TEST_PARAMS ]]; then
    set -- ${GSPLAT_TEST_PARAMS} "$@"
fi

while (( $# >= 1 )); do
    case $1 in
    --shell)
        runshell=true
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
    --debug)
        do_debug=true
        ;;
    --verbose)
        do_verbose=true
        ;;
    --gpus=*)
        gpus=${1#--gpus=}
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

check_if_installed docker nvidia-container-runtime

# Load config variables
load_config "$REPOROOT/config.yaml"

if $do_reset_home || $do_reset_cache; then
    if $do_reset_cache; then
        echo -n "Removing container cache volume... " >&2
        docker volume rm "$LOCAL_CACHE_VOLUME" > /dev/null 2>&1 || true
        echo "OK"
    fi
    if $do_reset_home; then
        echo -n "Removing container home volume... " >&2
        docker volume rm "$LOCAL_HOME_VOLUME" > /dev/null 2>&1 || true
        echo "OK"
    fi
    exit
fi

HOST_USER=$(id -un)
HOST_GROUP=$(id -gn)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_HOME="$HOME"

run_args=(
    "--gpus=$gpus"
    --rm
    -v "$REPOROOT:$REPOROOT"
    -w "$REPOROOT"

    -e HOST_USER="$HOST_USER"
    -e HOST_GROUP="$HOST_GROUP"
    -e HOST_UID="$HOST_UID"
    -e HOST_GID="$HOST_GID"
    -e HOST_HOME="$HOST_HOME"
    -e TERM="$TERM"

    -e PYTHONPATH="$REPOROOT"

    -v "$LOCAL_HOME_VOLUME:$HOST_HOME"
    -v "$LOCAL_CACHE_VOLUME:/var/cache"

    --hostname "$(hostname)-gsdev"
)

# Add user envvars as -e KEY=VALUE pairs without breaking on spaces
for kv in "${envvars[@]}"; do
    run_args+=(-e "$kv")
done


if $do_3dgut; then
    run_args+=(-e BUILD_3DGUT=1)
fi
if $do_2dgs; then
    run_args+=(-e BUILD_2DGS=1)
fi
if $do_3dgs; then
    run_args+=(-e BUILD_3DGS=1)
fi

if $do_debug; then
    run_args+=(-e DEBUG=1)
fi

shell_args=()

if $runshell; then
    # No arguments given?
    if [[ $# == 0 ]]; then
        # Drop us into an interactive shell in the container
        run_args+=(-ti)
    fi
else
    if $do_sanitize; then
        # CUDA compute-sanitizer needs the full path of the program to be analyzed
        sanitizer_cmd='/usr/local/cuda/bin/compute-sanitizer "$(command -v pytest)"'
        if $do_verbose; then
            sanitizer_cmd+=' -sv'
        fi
        shell_args+=(/bin/bash -c "$sanitizer_cmd")
        run_args+=(-e DEBUG=1) # it's helpful for triggering asserts and full symbol info
        run_args+=(--privileged) # compute-sanitizer sometimes segfaults if not running on privileged container
    else
        # We want to run pytest, possibly with users' parameters
        shell_args+=(pytest)
        if $do_verbose; then
            shell_args+=(-sv) # show C++ build as it happens
        fi
    fi
fi

if $do_verbose; then
    # Show the whole docker run invocation
    set -x
fi

docker run "${run_args[@]}" "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" "${shell_args[@]}" "$@"

