"""Export the trained DQN checkpoint for the NumPy-only Webots runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'controllers/micromouse'))
from lidar_dqn import schema
from lidar_dqn_agent import load_inference, VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model,saved=load_inference(args.checkpoint)
    layers=[dict(weight=m.weight.tolist(),bias=m.bias.tolist())
            for m in model.layers if hasattr(m,'weight')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        algorithm='masked_double_dqn',version=VERSION,schema=schema(),layers=layers,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        maze_signature=saved['progress']['maze_signature']),allow_nan=False)+'\n')


if __name__=='__main__':
    main()
