#!/usr/bin/env python3
"""Run m6A-MAC from the repository root without an editable installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'predict'))
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'predict'):
        parser.parse_args()
        return
    command, remaining = sys.argv[1], sys.argv[2:]
    if command == 'train':
        from m6a.training.train import main as train
        train(remaining)
    else:
        from m6a.models import M6AMAC
        from m6a.models.sequence_model import run_cli
        sys.argv = [sys.argv[0], *remaining]
        run_cli(M6AMAC)


if __name__ == '__main__':
    main()
