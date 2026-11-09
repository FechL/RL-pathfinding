#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_python="$HOME/miniconda3/bin/python"

mode="${1:-sim}"
case "$mode" in
    train|graphs)
        shift
        training_python="$project_dir/.venv/bin/python"
        if [[ ! -x "$training_python" ]]; then
            echo "Lingkungan training belum tersedia. Ikuti instalasi di README.md." >&2
            exit 1
        fi
        if [[ "$mode" == "train" ]]; then
            if [[ "${1:-}" == "--algorithm" ]]; then
                case "${2:-}" in
                    naf)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_naf.py" --render --render-speed 1 "$@"
                        ;;
                    rl)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_continuous.py" --rl-mode --mission-only --learning-rate 1e-7 --render "$@"
                        ;;
                    sac) shift 2 ;;
                    *)
                        echo "Pilih algoritma training: naf, sac, atau rl." >&2
                        exit 2
                        ;;
                esac
            fi
            exec "$training_python" "$project_dir/tools/train_continuous.py" --warm-start --render "$@"
        fi
        exec "$training_python" "$project_dir/tools/plot_training.py" "$@"
        ;;
    help|--help|-h)
        echo "bash run.sh                  Jalankan robot di Webots"
        echo "bash run.sh train            Lihat robot, monitor SAC, dan grafik training live"
        echo "bash run.sh train --render-speed 1   Tampilan kecepatan normal"
        echo "bash run.sh train --no-render       Training tanpa jendela"
        echo "bash run.sh train --steps 100000"
        echo "bash run.sh train --algorithm naf   Training Continuous Deep Q-Learning"
        echo "bash run.sh sim --algorithm naf     Jalankan hasil NAF di Webots"
        echo "bash run.sh train --algorithm rl    Fine-tune SAC + monitor reward, Q critic, dan loss live"
        echo "bash run.sh train --algorithm rl --timeout 180 --render-speed 1"
        echo "bash run.sh train --algorithm rl --episodes 1000 --no-render"
        echo "bash run.sh graphs           Buat grafik dari latihan tersimpan"
        echo "bash run.sh graphs --report-dir /path/ke/run"
        exit 0
        ;;
    sim)
        if [[ $# -gt 0 ]]; then shift; fi
        if [[ "${1:-}" == "--algorithm" ]]; then
            case "${2:-}" in
                naf|sac|rl|rl_unguided|q_learning|astar) export MICROMOUSE_ALGORITHM="$2"; shift 2 ;;
                *)
                    echo "Pilih algoritma simulasi: naf, sac, rl, rl_unguided, q_learning, atau astar." >&2
                    exit 2
                    ;;
            esac
        fi
        ;;
esac

if [[ ! -x "$base_python" ]]; then
    echo "Miniconda base Python not found: $base_python" >&2
    echo "Update base_python in run.sh and COMMAND in controllers/micromouse/runtime.ini." >&2
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
