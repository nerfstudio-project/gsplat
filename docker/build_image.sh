#!/bin/bash

set -euo pipefail

SDIR=$(dirname "$(readlink -f "$0")")
REPOROOT="$SDIR/.."

source "$SDIR/utils.sh"

usage()
{
    echo "Build gsplat's development/ci docker image gsplat and optionally pushes it to the registry."
    echo
    echo "Usage: ${0##*/} [global flags] [extra docker build flags]"
    echo "global flags:"
    echo "   --push     Push the built image to the docker registry,"
    echo "              fail if it already exists."
    echo "   --force    When used with --push, force overwriting the existing"
    echo "              image in the docker registry."
    echo "   --help|-h  Show this help message"
}

force_overwrite=false
push_image=false

while (( $# >= 1 )); do
    case $1 in
    --force)
        force_overwrite=true;
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    --push)
        push_image=true
        ;;
    *)
        # Forward unknown flags to docker
        break
        ;;
    esac
    shift
done

if $force_overwrite && ! $push_image; then
    die "--force only makes sense if accompanied by --push"
fi

check_if_installed "docker buildx"

load_config "$REPOROOT/config.yaml"

IMAGE_URL="$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

# Pull the remote image if it exists,
# this way we could reuse the cached layers if they
# didn't change.
docker pull --quiet "$IMAGE_URL" 2>/dev/null || true

# Avoid overwriting the remote image if it exists and we're not forcing push.
if $push_image && ! $force_overwrite && docker manifest inspect "$IMAGE_URL" >/dev/null 2>&1; then
    die "Image already exists in the docker registry, pass --force to overwrite it."
fi

build_args=(
  --build-context "gsplatrepo=$REPOROOT"
  --target main
  -t "$IMAGE_URL"
)

if $push_image; then
    build_args+=(--push)
fi


docker buildx build "${build_args[@]}" "$@" "$SDIR"
