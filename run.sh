#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_python="$HOME/miniconda3/bin/python"

if [[ ! -x "$base_python" ]]; then
    echo "Miniconda base Python not found: $base_python" >&2
    echo "Update base_python in run.sh and COMMAND in controllers/epuck/runtime.ini." >&2
    exit 1
fi

if ! command -v webots >/dev/null 2>&1; then
    echo "Webots was not found in PATH. Install Webots R2025a first." >&2
    exit 1
fi

if ! "$base_python" -c 'import numpy; import cv2'; then
    echo "Install the dependencies in Miniconda base first:" >&2
    echo "conda activate base" >&2
    echo "python -m pip install -r \"$project_dir/requirements.txt\"" >&2
    exit 1
fi

exec webots --mode=realtime "$@" "$project_dir/worlds/world.wbt"
