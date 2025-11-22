"""
Simple preprocessing entry point for L2-ARCTIC (perceived-only).

- 우리가 만든 DatasetProcessor를 이용해서
  wav, duration, spk_id, perceived_aligned, perceived_train_target, wrd
  만 포함된 JSON을 생성한다.
- 에러 라벨 생성, CV split, disjoint split 등은 전부 제외했다.
"""

import argparse
import logging
from pathlib import Path

from LopeScript.DataProc.preprocessing import DatasetProcessor


def main():
    parser = argparse.ArgumentParser(
        description="L2-ARCTIC perceived-only preprocessing"
    )

    # 굳이 서브커맨드 나누지 말고, 단일 엔트리만 사용
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/l2arctic",
        help="L2-ARCTIC dataset root directory "
             "(speaker 폴더들이 있는 최상위 경로)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preprocessed_perceived.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more detailed logging",
    )

    args = parser.parse_args()

    # ===== logging 설정 =====
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    # ===== 기본 체크 =====
    if not data_root.exists() or not data_root.is_dir():
        logging.error(f"Data root not found or not a directory: {data_root}")
        return

    # 상위 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("====================================================")
    logging.info("  L2-ARCTIC perceived-only preprocessing start")
    logging.info("----------------------------------------------------")
    logging.info(f"  data_root : {data_root}")
    logging.info(f"  output    : {output_path}")
    logging.info("====================================================")

    processor = DatasetProcessor(str(data_root), str(output_path))
    result = processor.process_all_files()

    logging.info("====================================================")
    logging.info("  Preprocessing finished")
    logging.info(f"  Total   files: {processor.total}")
    logging.info(f"  Success files: {processor.success}")
    logging.info(f"  Annotation   : {processor.annotation_used}")
    logging.info(f"  TextGrid     : {processor.textgrid_used}")
    logging.info("====================================================")

    # 필요하다면 result를 여기서 후처리하거나, 간단 검증을 넣어도 됨.
    if not result:
        logging.warning("Result dictionary is empty. "
                        "Check if TextGrid/wav files are present.")


if __name__ == "__main__":
    main()
