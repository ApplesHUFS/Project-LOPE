import os
import json
import re
import copy
from pathlib import Path
from typing import Dict, List, Optional
from textgrid import TextGrid, IntervalTier

# ========== silence helpers ==========
def is_silence(phoneme: str) -> bool:
    return phoneme.lower() in {"sil", "sp", "spn", "pau", ""}

def collapse_silences(seq: List[str]) -> List[str]:
    """앞뒤 sil 제거 + 내부 연속 sil은 하나로 압축."""
    if not seq:
        return seq
    # trim ends
    i, j = 0, len(seq) - 1
    while i <= j and is_silence(seq[i]): i += 1
    while j >= i and is_silence(seq[j]): j -= 1
    if i > j:
        return []
    core = seq[i:j+1]
    out, prev_sil = [], False
    for p in core:
        s = is_silence(p)
        if s:
            if not prev_sil:
                out.append("sil")
            prev_sil = True
        else:
            out.append(p)
            prev_sil = False
    return out

# ========== tag normalization ==========
def normalize_phoneme_for_perceived(annotation: str,
                                    remove_annotation: bool = True,
                                    keep_artificial_sil: bool = False) -> Optional[str]:
    """
    TextGrid의 phones tier에 들어있는 표기(예: 'p,b,s')를 인식자(perceived) 기준으로 단일 폰으로 정규화.
    - parts: [canonical, perceived, error_type?] 라고 가정
      error_type: a(add), d(del), s(subst), c(correct) 등
    - perceived 스트림을 만들 때, 'd'(삭제)는 실제 발화에 해당 폰이 없음 → 제거
      나머지는 perceived 표기를 택함
    - 인공침묵을 제거하고 싶다면 keep_artificial_sil=False로
    """
    text = annotation.lower()
    parse_tag = re.sub(r"[^a-z,]", "", text)

    if is_silence(parse_tag):
        return "sil"
    if not parse_tag:
        return None

    parts = parse_tag.split(",")
    if len(parts) == 1:
        # 단일 표기
        return parts[0]

    if not remove_annotation:
        return parse_tag

    per_p = parts[1] if len(parts) > 1 else parts[0]
    err   = parts[2] if len(parts) > 2 else None

    # 인공침묵을 제거하려는 모드가 아니라면 태그 그대로 사용
    if keep_artificial_sil:
        return per_p

    # perceived 스트림 규칙: deletion('d')은 제거
    if err == 'd':
        return None

    return per_p

def normalize_tier_perceived(tier: IntervalTier,
                             keep_artificial_sil: bool = False) -> IntervalTier:
    tier = copy.deepcopy(tier)
    out = IntervalTier()
    for itv in tier.intervals:
        p = normalize_phoneme_for_perceived(itv.mark,
                                            remove_annotation=True,
                                            keep_artificial_sil=keep_artificial_sil)
        if p is None:
            continue
        itv.mark = p
        out.addInterval(itv)
    return out

def tier_to_phoneme_list(tier: IntervalTier) -> List[str]:
    return [itv.mark for itv in tier]

# ========== safe tier lookup ==========
def get_phone_tier_safely(tg: TextGrid) -> IntervalTier:
    candidates = ["phones", "phone", "phoneme", "Phones", "Phone", "Phoneme"]
    for name in candidates:
        try:
            return tg.getFirst(name)
        except Exception:
            pass
    # fallback: IntervalTier 중 첫 번째
    for t in tg.tiers:
        if isinstance(t, IntervalTier) and getattr(t, "intervals", None):
            return t
    raise ValueError("No suitable phone/phoneme tier found in TextGrid.")

# ========== main processor ==========
class DatasetProcessor:
    """
    L2-ARCTIC 전처리 (perceived 전용)
    - 출력 필드: wav, duration, spk_id, perceived_aligned, perceived_train_target, wrd(있으면)
    """

    def __init__(self, data_root: str, output_path: str):
        self.data_root = Path(data_root)
        self.output_path = Path(output_path)

        self.total = 0
        self.success = 0
        self.annotation_used = 0
        self.textgrid_used = 0

    def extract_perceived_from_textgrid(self,
                                        tg: TextGrid,
                                        keep_artificial_sil: bool = False,
                                        compress_and_trim: bool = True) -> str:
        """
        TextGrid로부터 perceived 폰열을 추출.
        - keep_artificial_sil=True  → sil도 그대로 유지(연속 미압축)
        - compress_and_trim=True    → 앞뒤 sil 제거 + 내부 sil 압축
        """
        phone_tier = get_phone_tier_safely(tg)
        per_tier = normalize_tier_perceived(phone_tier, keep_artificial_sil=keep_artificial_sil)
        phones = tier_to_phoneme_list(per_tier)
        if compress_and_trim:
            phones = collapse_silences(phones)
        return " ".join(phones)

    def process_single_file(self, speaker_id: str, filename: str) -> Optional[Dict]:
        annotation_path = self.data_root / speaker_id / 'annotation' / f'{filename}.TextGrid'
        textgrid_path   = self.data_root / speaker_id / 'textgrid'   / f'{filename}.TextGrid'
        wav_path        = self.data_root / speaker_id / 'wav'        / f'{filename}.wav'
        transcript_path = self.data_root / speaker_id / 'transcript' / f'{filename}.txt'

        if annotation_path.exists():
            tg_path = annotation_path
            has_annotation = True
            self.annotation_used += 1
        elif textgrid_path.exists():
            tg_path = textgrid_path
            has_annotation = False
            self.textgrid_used += 1
        else:
            return None

        if not wav_path.exists():
            return None

        transcript = ""
        if transcript_path.exists():
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            except Exception:
                transcript = ""

        try:
            tg = TextGrid()
            tg.read(str(tg_path))
            duration = float(tg.maxTime)

            # perceived_aligned: 인공침묵 유지, 연속 sil 미압축
            perceived_aligned = self.extract_perceived_from_textgrid(
                tg, keep_artificial_sil=True, compress_and_trim=False
            )

            # perceived_train_target: 인공침묵 제거, 앞뒤 trim + 내부 압축
            perceived_train_target = self.extract_perceived_from_textgrid(
                tg, keep_artificial_sil=False, compress_and_trim=True
            )

            wav_absolute = str(wav_path.resolve())
            return {
                "wav": wav_absolute,
                "duration": duration,
                "spk_id": speaker_id,
                "perceived_aligned": perceived_aligned,
                "perceived_train_target": perceived_train_target,
                "wrd": transcript
            }
        except Exception as e:
            print(f"Error processing {speaker_id}/{filename}: {str(e)}")
            return None

    def process_all_files(self) -> Dict[str, Dict]:
        print(f"Processing L2-ARCTIC dataset from {self.data_root}")
        result: Dict[str, Dict] = {}

        speaker_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            print(f"Processing speaker: {speaker_id}")

            wav_dir = speaker_dir / 'wav'
            if not wav_dir.exists():
                continue

            wav_files = sorted(list(wav_dir.glob('*.wav')))
            for wav_file in wav_files:
                filename = wav_file.stem
                self.total += 1

                data = self.process_single_file(speaker_id, filename)
                if data:
                    result[data['wav']] = data
                    self.success += 1

                if self.total % 100 == 0:
                    print(f"  Processed {self.total} files, {self.success} successful")

        print(f"Saving results to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print(f"Total files processed: {self.total}")
        print(f"Successful: {self.success}")
        print(f"Annotation files used: {self.annotation_used}")
        print(f"TextGrid files used: {self.textgrid_used}")
        print("="*80)
        return result
    print("✅ Preprocessing script executed successfully!")


