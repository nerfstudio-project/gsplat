#!/bin/sh
set -e

HOST_UID="${HOST_UID:-0}"
RUN_AS="${RUN_AS:-${HOST_USER:-root}}"

# We expect to start as root
if [ "$(id -u)" -ne 0 ]; then
    exec "$@"
fi

# If host user isn't root, mirror the host user/group inside the container
if [ "$HOST_UID" != 0 ]; then
    if [ -z "$HOST_USER" ]; then
        echo "ERROR: HOST_USER must be specified when HOST_UID is specified" >&2
        exit 1
    fi
    if [ "$HOST_UID" -lt 1000 ]; then
        echo "ERROR: HOST_UID ($HOST_UID) must be >= 1000" >&2
        exit 1
    fi
    HOST_GID="${HOST_GID:-$HOST_UID}"
    HOST_GROUP="${HOST_GROUP:-$HOST_USER}"
    HOST_HOME="${HOST_HOME:-/home/$HOST_USER}"
    if [ "$HOST_GID" -lt 1000 ]; then
        echo "ERROR: HOST_GID ($HOST_GID) must be >= 1000" >&2
        exit 1
    fi

    # Set up group (skip if exact match already exists, e.g. container restart)
    existing_group="$(getent group "$HOST_GID" | cut -d: -f1 || true)"
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
    fi

    # Set up user (skip if exact match already exists, e.g. container restart)
    existing_user="$(getent passwd "$HOST_UID" | cut -d: -f1 || true)"
    if [ -z "$existing_user" ]; then
        # UID is free — check that the name is also free
        if getent passwd "$HOST_USER" > /dev/null 2>&1; then
            echo "ERROR: user '$HOST_USER' already exists with a different UID" >&2
            exit 1
        fi
        useradd -u "$HOST_UID" -g "$HOST_GROUP" -d "$HOST_HOME" -s /bin/bash "$HOST_USER"

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

# Run the command as RUN_AS user
exec gosu "$RUN_AS" "$@"
