"""L2-ARCTIC 데이터셋 전처리 및 분할."""

import argparse
import logging
from pathlib import Path

from LopeScript.DataProc.preprocessing import DatasetProcessor
from LopeScript.DataProc.data_split import split_dataset


def main():
    parser = argparse.ArgumentParser(
        description="L2-ARCTIC preprocessing and splitting"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/l2arctic",
        help="L2-ARCTIC dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preprocessed_perceived.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--split_output_dir",
        type=str,
        default="data",
        help="Directory for train/dev/test split files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    if not data_root.exists() or not data_root.is_dir():
        logging.error(f"Data root not found: {data_root}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = DatasetProcessor(str(data_root), str(output_path))
    result = processor.process_all_files()

    if not result:
        logging.warning("Result is empty")
        return

    split_dataset(
        input_path=str(output_path),
        output_dir=args.split_output_dir
    )


if __name__ == "__main__":
    main()
