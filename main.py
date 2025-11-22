# LopeScript/main.py

# LopeScript/main.py

import argparse
import sys

# ---- import core modules ----
from LopeScript.config import get_config
from LopeScript.train import main as train_main
from LopeScript.evaluation import main as eval_main
from LopeScript.Demo import main as demo_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lope Pronunciation Assessment Main Script"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "demo"],
        help="Select mode: train / eval / demo"
    )

    # demo 모드에서 사용
    parser.add_argument(
        "--audio",
        type=str,
        help="demo 모드에서 평가할 wav 파일 경로"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="demo 모드에서 사용할 학습된 모델 체크포인트 경로"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------------
    #  Train Mode
    # ---------------------------
    if args.mode == "train":
        print("\n[MODE] Train")
        train_main()
        return

    # ---------------------------
    #  Eval Mode
    # ---------------------------
    elif args.mode == "eval":
        print("\n[MODE] Evaluation")
        eval_main()
        return

    # ---------------------------
    #  Demo Mode
    # ---------------------------
    elif args.mode == "demo":
        print("\n[MODE] Demo (Single Sentence Pronunciation Assessment)")
        if args.audio is None or args.checkpoint is None:
            print("Error: --audio 와 --checkpoint 를 반드시 지정해야 합니다.")
            sys.exit(1)
        demo_main()
        return


if __name__ == "__main__":
    main()
