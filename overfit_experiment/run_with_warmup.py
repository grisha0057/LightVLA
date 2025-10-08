#!/usr/bin/env python3
"""Wrapper to apply coverage warmup before launching finetune."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage_warmup_steps", type=int, default=200)
    parser.add_argument("--coverage_target", type=float, default=0.98)
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = Path(os.environ["VLA_PATH"]) / "config.json"
    with config_path.open() as f:
        cfg = json.load(f)
    cfg["coverage_warmup_steps"] = args.coverage_warmup_steps
    cfg["coverage_target_after_warmup"] = args.coverage_target
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    cmd = ["python", "vla-scripts/finetune.py", *args.remaining]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
