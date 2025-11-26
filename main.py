"""Pronunciation Assessment 통합 진입점."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Pronunciation Assessment Pipeline"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "eval", "demo"],
        help="실행 모드: preprocess / train / eval / demo"
    )

    args, remaining_args = parser.parse_known_args()

    if args.mode == "preprocess":
        from preprocess import main as preprocess_main
        sys.argv = [sys.argv[0]] + remaining_args
        preprocess_main()

    elif args.mode == "train":
        from LopeScript.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining_args
        train_main()

    elif args.mode == "eval":
        from LopeScript.evaluation import main as eval_main
        sys.argv = [sys.argv[0]] + remaining_args
        eval_main()

    elif args.mode == "demo":
        from LopeScript.Demo.demo import main as demo_main
        sys.argv = [sys.argv[0]] + remaining_args
        demo_main()


if __name__ == "__main__":
    main()
