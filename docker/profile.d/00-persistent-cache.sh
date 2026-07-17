#!/bin/sh

# Consolidate the cache under /var/cache/<user> (via the symlink below), which
# run_tests.sh bind-mounts to persistent host storage.
export XDG_CACHE_HOME="$HOME/.cache"

# Setup user's cache
mkdir -p "/var/cache/$(id -un)"
if [ "$(readlink "$XDG_CACHE_HOME" 2>/dev/null)" != "/var/cache/$(id -un)" ]; then
    if [ -e "$XDG_CACHE_HOME" ] || [ -L "$XDG_CACHE_HOME" ]; then
        mv --backup=numbered "$XDG_CACHE_HOME" "$XDG_CACHE_HOME.backup"
    fi
    ln -sf "/var/cache/$(id -un)" "$XDG_CACHE_HOME"
fi

# Setup ccache's cache
export CCACHE_DIR="$XDG_CACHE_HOME/ccache"
mkdir -p "$CCACHE_DIR"

# Setup CUDA JIT cache
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$CUDA_CACHE_PATH"

