#!/bin/bash

# ccache-4.12.2 does'nt recognize some of nvcc's full parameter names
# We replace them by their short names so that the invocation can be cached.
set -- $(printf "%s\n" "$@" | sed 's/--generate-dependencies-with-compile/-MD/g;s/--generate-nonsystem-dependencies-with-compile/-MMD/g;s/--dependency-output/-MF/g;')

exec ccache compiler_type=nvcc /usr/local/cuda/bin/nvcc-real "$@"

