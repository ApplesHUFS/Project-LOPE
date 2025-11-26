"""데이터셋을 train/dev/test로 분할하는 유틸리티."""

import json
import os
from typing import Dict, Any

TEST_SPK = ['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK']
DEV_SPK = ['MBMPS']


def load_json(path: str) -> Dict:
    """JSON 파일 로드.

    Args:
        path: JSON 파일 경로

    Returns:
        로드된 딕셔너리
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str):
    """딕셔너리를 JSON 파일로 저장.

    Args:
        data: 저장할 딕셔너리
        path: 출력 파일 경로
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_dataset(
    input_path: str = "data/preprocessed_perceived.json",
    output_dir: str = "data"
) -> Dict[str, int]:
    """preprocessed JSON을 train/dev/test로 분할.

    Args:
        input_path: 입력 전처리 JSON 파일 경로
        output_dir: 출력 디렉토리

    Returns:
        각 split별 샘플 수를 담은 딕셔너리
    """
    dataset = load_json(input_path)

    train_data = {}
    dev_data = {}
    test_data = {}
    skipped = 0

    for key, sample in dataset.items():
        if not isinstance(sample, dict):
            skipped += 1
            continue

        speaker = sample.get("spk_id", None)

        if speaker in TEST_SPK:
            test_data[key] = sample
        elif speaker in DEV_SPK:
            dev_data[key] = sample
        elif speaker is None:
            skipped += 1
        else:
            train_data[key] = sample

    os.makedirs(output_dir, exist_ok=True)

    save_json(train_data, os.path.join(output_dir, "train.json"))
    save_json(dev_data, os.path.join(output_dir, "dev.json"))
    save_json(test_data, os.path.join(output_dir, "test.json"))

    stats = {
        "train": len(train_data),
        "dev": len(dev_data),
        "test": len(test_data),
        "skipped": skipped
    }

    print("\n=== Dataset Split Complete ===")
    print(f"Train: {stats['train']} samples")
    print(f"Dev:   {stats['dev']} samples")
    print(f"Test:  {stats['test']} samples")
    print(f"Skipped: {stats['skipped']}\n")

    return stats


if __name__ == "__main__":
    split_dataset()
