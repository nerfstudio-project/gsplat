#!/bin/bash

SDIR=$(dirname "$(readlink -f "$0")")
REPOROOT=$SDIR/..

die()
{
    echo "${0##*/}: $@"
    exit 1
} >&2

check_if_installed()
{
    for needed in "$@"; do
        if ! $needed --help > /dev/null 2>&1; then
            die "'$needed' must be installed before building the image"
        fi
    done
}

load_config()
{
    local cfgname=$1
    shift

    source <(yq -r '.variables | to_entries[] | "\(.key)=\"\(.value)\""' "$cfgname")
}
