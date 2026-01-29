#!/bin/bash

set -euo pipefail

SDIR=$(dirname "$(readlink -f "$0")")
REPOROOT=$SDIR/..

source "$REPOROOT/docker/utils.sh"

LOCAL_CACHE_NAME=gsplat_cache

usage()
{
    echo "Build gsplat and run its tests inside its docker container"
    echo
    echo "Usage: ${0##*/} [global flags] [feature flags] [ENVVAR=value ...] [extra shell/pytest args]"
    echo "global flags:"
    echo "   --shell    Enter into the shell inside the container."
    echo "   --sanitize Run tests under CUDA compute-sanitizer"
    echo "   --reset    Delete the internal build cache"
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
    echo "Examples:"
    echo "- Build everything and run all tests with all features:"
    echo "    ${0##*/}"
    echo "- Build only 3dgut and run only the basic tests:"
    echo "    ${0##*/} --3dgut tests/test_basic.py"
    echo "- Build 3dgut and 3dgs only and run the rasterization tests:"
    echo "    ${0##*/} --3dgut --3dgs tests/test_rasterization.py"
    echo "- Set parameters via environment variable:"
    echo "    export GSPLAT_TEST_PARAMS='--gpus=device=1 --3dgut'"
    echo "    ${0##*/}"
}

runshell=false

do_3dgut=false
do_3dgs=false
do_2dgs=false
do_reset=false
do_sanitize=false
do_debug=false
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
    --reset)
        do_reset=true
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

check_if_installed yq docker nvidia-container-runtime

# Load config variables
load_config "$REPOROOT/config.yaml"

if $do_reset; then
    echo -n "Removing gsplat's local cache volume..." >&2
    docker volume rm "$LOCAL_CACHE_NAME" > /dev/null 2>&1 || true
    echo " OK" >&2
fi

run_args=(
    "--gpus=$gpus"
    --rm
    -v "$REPOROOT:/root/gsplat"
    -v "$LOCAL_CACHE_NAME:/var/cache"
    --entrypoint /bin/bash # To avoid the CUDA banner when the container starts.
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

# We need a login shell in order to load ~/.profile, it loads up the python venv.
shell_args=(--login)

if $runshell; then
    # No arguments given?
    if [[ $# == 0 ]]; then
        # Drop us into an interactive shell in the container
        shell_args+=(-i)
        run_args+=(-ti)
    else
        # Execute user's commands inside the container
        shell_args+=(-c "$*")
    fi
else
    if $do_sanitize; then
        # CUDA compute-sanitizer needs the full path of the program to be analyzed
        shell_args+=(-c "/usr/local/cuda/bin/compute-sanitizer \$(command -v pytest) $*")
        run_args+=(-e DEBUG=1) # it's helpful for triggering asserts and full symbol info
        run_args+=(--privileged) # compute-sanitizer sometimes segfaults if not running on privileged container
    else
        # We want to run pytest, possibly with users' parameters
        shell_args+=(-c "pytest $*")
    fi
fi

if $do_debug; then
    # Show the whole docker run invocation
    set -x
fi
docker run "${run_args[@]}" "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" "${shell_args[@]}"

