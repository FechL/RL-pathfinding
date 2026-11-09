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
                    rl|hidden_goal)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_continuous.py" --hidden-goal --render "$@"
                        ;;
                    guided_rl)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_continuous.py" --rl-mode --mission-only --learning-rate 1e-7 --render "$@"
                        ;;
                    sac)
                        shift 2
                        exec "$training_python" "$project_dir/tools/train_continuous.py" --warm-start --render "$@"
                        ;;
                    *)
                        echo "Pilih algoritma training: hidden_goal (atau rl), sac, guided_rl, atau naf." >&2
                        exit 2
                        ;;
                esac
            fi
            exec "$training_python" "$project_dir/tools/train_continuous.py" --hidden-goal --render "$@"
        fi
        exec "$training_python" "$project_dir/tools/plot_training.py" "$@"
        ;;
    help|--help|-h)
        echo "bash run.sh                  Jalankan robot di Webots"
        echo "bash run.sh train            Lihat robot, monitor SAC, dan grafik training live"
        echo "bash run.sh train --render-speed 1   Tampilan kecepatan normal"
        echo "bash run.sh train --no-render       Training tanpa jendela"
        echo "bash run.sh train --steps 100000"
        echo "bash run.sh train --algorithm rl    Training SAC continuous sensor goal"
        echo "bash run.sh train --algorithm naf   Training continuous NAF baseline"
        echo "bash run.sh sim --algorithm naf     Jalankan hasil NAF di Webots"
        echo "bash run.sh sim --fast              Percepat simulasi Webots"
        echo "bash run.sh sim --policy path.json  Jalankan actor tertentu"
        echo "bash run.sh train --algorithm rl --timeout 180 --render-speed 16"
        echo "bash run.sh graphs           Buat grafik dari latihan tersimpan"
        echo "bash run.sh graphs --report-dir /path/ke/run"
        exit 0
        ;;
    sim)
        shift
        webots_mode=realtime
        webots_args=()
        export MICROMOUSE_ALGORITHM=hidden_goal
        while (($#)); do
            case "$1" in
                --fast) webots_mode=fast; shift ;;
                --realtime) webots_mode=realtime; shift ;;
                --algorithm)
                    case "${2:-}" in
                        hidden_goal|rl|naf|sac|rl_unguided|q_learning|astar)
                            if [[ "$2" == "rl" ]]; then
                                export MICROMOUSE_ALGORITHM=hidden_goal
                            else
                                export MICROMOUSE_ALGORITHM="$2"
                            fi
                            shift 2
                            ;;
                        *) echo "Pilih algoritma simulasi: hidden_goal (atau rl), naf, sac, rl_unguided, q_learning, atau astar." >&2; exit 2 ;;
                    esac
                    ;;
                --policy)
                    if [[ -z "${2:-}" ]]; then echo "--policy memerlukan path actor JSON." >&2; exit 2; fi
                    if [[ "$2" == /* ]]; then
                        export MICROMOUSE_POLICY="$2"
                    else
                        export MICROMOUSE_POLICY="$PWD/$2"
                    fi
                    shift 2
                    ;;
                *) webots_args+=("$1"); shift ;;
            esac
        done
        set -- "${webots_args[@]}"
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

exec webots --mode="$webots_mode" "$@" "$project_dir/worlds/world.wbt"
