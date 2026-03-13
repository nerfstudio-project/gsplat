#!/bin/bash
set -e

HOST_UID="${HOST_UID:-0}"
RUN_AS="${RUN_AS:-${HOST_USER:-root}}"

# We expect to start as root
if (( $(id -u) != 0 )); then
    exec "$@"
fi

# If host user isn't root, mirror the host user/group inside the container
if [ "$HOST_UID" != 0 ]; then
    if [ -z "$HOST_USER" ]; then
        echo "ERROR: HOST_USER must be specified when HOST_UID is specified" >&2
        exit 1
    fi
    if [ "$HOST_UID" -lt 1000 ]; then
        echo "WARNING: HOST_UID ($HOST_UID) is below 1000; this may collide with a system account inside the container." >&2
    fi
    HOST_GID="${HOST_GID:-$HOST_UID}"
    HOST_GROUP="${HOST_GROUP:-$HOST_USER}"
    HOST_HOME="${HOST_HOME:-/home/$HOST_USER}"
    if [ "$HOST_GID" -lt 1000 ]; then
        echo "WARNING: HOST_GID ($HOST_GID) is below 1000; this may collide with a system group inside the container." >&2
    fi

    # Set up group (skip if exact match already exists, e.g. container restart)
    existing_group="$(getent group "$HOST_GID" | cut -d: -f1 || true)"
    useradd_group="$HOST_GROUP"
    if [ -z "$existing_group" ]; then
        # GID is free — check that the name is also free
        if getent group "$HOST_GROUP" > /dev/null 2>&1; then
            echo "ERROR: group '$HOST_GROUP' already exists with a different GID" >&2
            exit 1
        fi
        groupadd -g "$HOST_GID" "$HOST_GROUP"
    elif [ "$existing_group" != "$HOST_GROUP" ]; then
        echo "WARNING: HOST_GID ($HOST_GID) is already taken by group '$existing_group' inside the container" >&2
        echo "         The group name of gsplat code inside and outside the container won't match, but this is harmless." >&2
        useradd_group="$HOST_GID"
    fi

    # Set up user (skip if exact match already exists, e.g. container restart)
    existing_user="$(getent passwd "$HOST_UID" | cut -d: -f1 || true)"
    if [ -z "$existing_user" ]; then
        # UID is free — check that the name is also free
        if getent passwd "$HOST_USER" > /dev/null 2>&1; then
            echo "ERROR: user '$HOST_USER' already exists with a different UID" >&2
            exit 1
        fi
        useradd -u "$HOST_UID" -g "$useradd_group" -d "$HOST_HOME" -s /bin/bash "$HOST_USER"

        # Create home directory if it doesn't exist yet
        if [ ! -d "$HOST_HOME" ]; then
            mkdir -p "$HOST_HOME"
            chown "$HOST_USER:" "$HOST_HOME"
            gosu "$HOST_USER" cp -ra /etc/skel/. "$HOST_HOME"/
        else
            chown "$HOST_USER:" "$HOST_HOME"
        fi
    elif [ "$existing_user" != "$HOST_USER" ]; then
        echo "ERROR: UID $HOST_UID is already taken by user '$existing_user'" >&2
        exit 1
    fi

    # Sudo configuration — gated on container isolation level
    is_privileged() {
        seccomp=$(sed -n 's/^Seccomp:[[:space:]]*//p' /proc/1/status 2>/dev/null)
        [ "$seccomp" = "0" ]
    }

    has_docker_socket() {
        [ -S /var/run/docker.sock ]
    }

    SUDOERS_FILE="/etc/sudoers.d/$HOST_USER"

    if is_privileged || has_docker_socket; then
        echo "$HOST_USER ALL=(ALL) NOPASSWD:/usr/bin/apt,/usr/bin/apt-get,/usr/bin/pip*,/usr/bin/conda" \
          > "$SUDOERS_FILE"
    else
        echo "$HOST_USER ALL=(ALL) NOPASSWD:ALL" > "$SUDOERS_FILE"
    fi
    chmod 0440 "$SUDOERS_FILE"

    export HOME="$HOST_HOME"
fi

# Check if gsplat source deps match what's installed in the image.
# This warns (but does not block) when the source has been updated
# but the Docker image hasn't been rebuilt.
if [[ ! -f /opt/dep-check/all_packages.txt ]]; then
    echo "ERROR: /opt/dep-check/all_packages.txt not found in the docker image." >&2
    echo "The docker image was not built correctly." >&2
    exit 1
fi

if [[ ! -f "$PWD/setup.py" ]] || ! grep -q 'gsplat' "$PWD/setup.py" 2>/dev/null; then
    echo "WARNING: gsplat setup.py not found in $PWD — skipping dependency check." >&2
else
    mapfile -t pkgs < /opt/dep-check/all_packages.txt
    args=(-f "$PWD/setup.py:install" -f "$PWD/setup.py:dev")
    if [[ -f "$PWD/examples/requirements.txt" ]]; then
        args+=(-f "$PWD/examples/requirements.txt")
    else
        echo "WARNING: $PWD/examples/requirements.txt not found — skipping examples dependency check." >&2
    fi
    check_gsplat_deps.sh "${args[@]}" -- "${pkgs[@]}" || true
fi

# Run the command as RUN_AS user
exec gosu "$RUN_AS" "$@"
