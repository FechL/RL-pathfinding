#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_python="$HOME/miniconda3/bin/python"

mode="${1:-sim}"
webots_mode="realtime"
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
                    q_learning|tabular)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_q_learning.py" "$@"
                        ;;
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
                        echo "Pilih algoritma training: q_learning, naf, sac, atau rl." >&2
                        exit 2
                        ;;
                esac
            fi
            exec "$training_python" "$project_dir/tools/train_q_learning.py" "$@"
        fi
        if [[ "${1:-}" == "--algorithm" ]]; then
            case "${2:-}" in
                q_learning|tabular) shift 2 ;;
                naf|sac|rl) shift 2; exec "$training_python" "$project_dir/tools/plot_training.py" "$@" ;;
                *) echo "Pilih grafik: q_learning, naf, sac, atau rl." >&2; exit 2 ;;
            esac
        fi
        exec "$training_python" "$project_dir/tools/plot_q_learning.py" "$@"
        ;;
    help|--help|-h)
        echo "bash run.sh                                  Jalankan Q-table di Webots"
        echo "bash run.sh train                            Training Q-learning + monitor live"
        echo "bash run.sh train --episodes 6000            Jumlah episode training"
        echo "bash run.sh train --render-speed 1           Refresh monitor tiap langkah"
        echo "bash run.sh train --render-speed 64          Default: refresh jarang, training lebih cepat"
        echo "bash run.sh train --no-render                Training headless paling cepat"
        echo "bash run.sh train --algorithm q_learning     Pilih tabular Q-learning eksplisit"
        echo "bash run.sh graphs                           Grafik reward, step, success, state-space"
        echo "bash run.sh graphs --report-dir reports/tabular_q_learning/runs/<run>"
        echo "bash run.sh sim --fast                       Jalankan Webots secepat mungkin"
        echo "bash run.sh sim --algorithm naf              Pilih controller NAF lama"
        exit 0
        ;;
    sim)
        if [[ $# -gt 0 ]]; then shift; fi
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --fast) webots_mode="fast"; shift ;;
                --algorithm)
                    case "${2:-}" in
                        naf|sac|rl|rl_unguided|q_learning|astar) export MICROMOUSE_ALGORITHM="$2"; shift 2 ;;
                        *) echo "Pilih algoritma simulasi: q_learning, naf, sac, rl, atau astar." >&2; exit 2 ;;
                    esac
                    ;;
                *) break ;;
            esac
        done
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

exec webots "--mode=$webots_mode" "$@" "$project_dir/worlds/world.wbt"
