"""CMU Pronouncing Dictionary 기반 phoneme sequence 생성 유틸리티."""

import re
from typing import List, Optional


class CMUDict:
    """CMU Pronouncing Dictionary 기반 텍스트-음소 변환 클래스."""

    def __init__(self):
        """CMU dictionary 초기화."""
        self.cmu_dict = self._load_cmu_dict()

    def _load_cmu_dict(self):
        """CMU dictionary를 메모리에 로드."""
        try:
            from nltk.corpus import cmudict
            return cmudict.dict()
        except (ImportError, LookupError):
            import nltk
            nltk.download('cmudict', quiet=True)
            from nltk.corpus import cmudict
            return cmudict.dict()

    def _clean_phoneme(self, phoneme: str) -> str:
        """음소에서 스트레스 마커(숫자) 제거.

        Args:
            phoneme: 스트레스 마커를 포함한 음소 (예: 'AE1')

        Returns:
            스트레스 마커가 제거된 음소 (예: 'ae')
        """
        return re.sub(r'\d+', '', phoneme).lower()

    def word_to_phonemes(self, word: str) -> Optional[List[str]]:
        """단어를 phoneme sequence로 변환.

        Args:
            word: 영어 단어

        Returns:
            phoneme 리스트 또는 None (사전에 없는 경우)
        """
        word_lower = word.lower()

        if word_lower in self.cmu_dict:
            phonemes_raw = self.cmu_dict[word_lower][0]
            return [self._clean_phoneme(p) for p in phonemes_raw]

        return None

    def text_to_phonemes(self, text: str, add_sil: bool = False) -> List[str]:
        """텍스트를 phoneme sequence로 변환.

        Args:
            text: 영어 문장
            add_sil: 단어 사이에 'sil' 추가 여부

        Returns:
            phoneme 리스트
        """
        words = re.findall(r"[a-zA-Z']+", text)

        phoneme_sequence = []

        for word in words:
            phonemes = self.word_to_phonemes(word)

            if phonemes:
                phoneme_sequence.extend(phonemes)
                if add_sil:
                    phoneme_sequence.append('sil')

        if add_sil and phoneme_sequence and phoneme_sequence[-1] == 'sil':
            phoneme_sequence.pop()

        return phoneme_sequence

    def text_to_phoneme_ids(
        self,
        text: str,
        phoneme_to_id: dict,
        add_sil: bool = False
    ) -> List[int]:
        """텍스트를 phoneme ID sequence로 변환.

        Args:
            text: 영어 문장
            phoneme_to_id: phoneme -> ID 매핑 딕셔너리
            add_sil: 단어 사이에 'sil' 추가 여부

        Returns:
            phoneme ID 리스트
        """
        phonemes = self.text_to_phonemes(text, add_sil=add_sil)
        return [phoneme_to_id.get(p, 0) for p in phonemes if p in phoneme_to_id]


def get_canonical_phonemes(text: str, add_sil: bool = False) -> List[str]:
    """텍스트에서 canonical phoneme sequence 생성 (편의 함수).

    Args:
        text: 영어 문장
        add_sil: 단어 사이에 'sil' 추가 여부

    Returns:
        phoneme 리스트
    """
    cmu = CMUDict()
    return cmu.text_to_phonemes(text, add_sil=add_sil)


def get_canonical_phoneme_ids(
    text: str,
    phoneme_to_id: dict,
    add_sil: bool = False
) -> List[int]:
    """텍스트에서 canonical phoneme ID sequence 생성 (편의 함수).

    Args:
        text: 영어 문장
        phoneme_to_id: phoneme -> ID 매핑 딕셔너리
        add_sil: 단어 사이에 'sil' 추가 여부

    Returns:
        phoneme ID 리스트
    """
    cmu = CMUDict()
    return cmu.text_to_phoneme_ids(text, phoneme_to_id, add_sil=add_sil)
