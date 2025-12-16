#!/bin/bash

set -euo pipefail

SDIR=$(dirname "$(readlink -f "$0")")
REPOROOT=$SDIR/..

source "$SDIR/utils.sh"

LOCAL_CACHE_NAME=gsplat_cache

usage()
{
    echo "Build gsplat and run its tests inside its docker container"
    echo
    echo "Usage: ${0##*/} [global flags] [feature flags] [extra docker/pytest args]"
    echo "global flags:"
    echo "   --shell    Enter into the shell inside the container."
    echo "              You then can run 'pytest' to run the tests"
    echo "   --reset    Delete the internal build cache"
    echo "   --help|-h  Show this help message"
    echo "feature flags"
    echo "   --2dgs     Build 2dgs"
    echo "   --3dgs     Build 3dgs"
    echo "   --3dgut    Build 3dgut "
    echo "The multiple feature flags can be given,"
    echo "only the selected features will be built."
    echo "If none are given, all features will be built."
    echo
    echo "Examples:"
    echo "- Build everything and run all tests with all features:"
    echo "    ${0##*/}"
    echo "- Build only 3dgut and run only the basic tests:"
    echo "    ${0##*/} --3dgut tests/test_basic.py"
    echo "- Build 3dgut and 3dgs only and run the rasterization tests:"
    echo "    ${0##*/} --3dgut --3dgs tests/test_rasterization.py"
}

runshell=false

do_3dgut=false
do_3dgs=false
do_2dgs=false
do_reset=false

while (( $# >= 1 )); do
    case $1 in
    --shell)
        runshell=true
        ;;
    --reset)
        do_reset=true
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
    --help|-h)
        usage
        exit 0
        ;;
    *)
        break
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
    --gpus=all
    --rm
    -ti
    -v $REPOROOT:/root/gsplat
    -v "$LOCAL_CACHE_NAME:/var/cache"
)

if $do_3dgut; then
    run_args+=(-e BUILD_3DGUT=1)
fi
if $do_2dgs; then
    run_args+=(-e BUILD_2DGS=1)
fi
if $do_3dgs; then
    run_args+=(-e BUILD_3DGS=1)
fi

if $runshell; then
    cmd='/bin/bash --login -i'
else
    cmd='/bin/bash --login -c pytest'
fi

docker run "${run_args[@]}" "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" $cmd "$@"

