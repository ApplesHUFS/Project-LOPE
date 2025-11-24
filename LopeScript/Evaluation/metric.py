"""
Metric utilities for pronunciation assessment.

This module implements basic sequence-level metrics such as
Levenshtein distance and phoneme error rate (PER).
"""

from typing import List, Sequence


def levenshtein_distance(
    ref: Sequence[int],
    hyp: Sequence[int]
) -> int:
    """Computes Levenshtein edit distance between two sequences.

    The distance is the minimum number of substitutions, deletions,
    and insertions needed to transform ref into hyp.

    Args:
        ref: Reference sequence (e.g., target phoneme indices).
        hyp: Hypothesis sequence (e.g., predicted phoneme indices).

    Returns:
        Integer edit distance.
    """
    m = len(ref)
    n = len(hyp)

    # DP table of size (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row/column
    for i in range(m + 1):
        dp[i][0] = i  # i deletions
    for j in range(n + 1):
        dp[0][j] = j  # j insertions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


def utterance_per(
    ref: Sequence[int],
    hyp: Sequence[int]
) -> float:
    """Computes phoneme error rate (PER) for a single utterance.

    PER = edit_distance(ref, hyp) / len(ref)

    Args:
        ref: Reference phoneme index sequence.
        hyp: Hypothesis phoneme index sequence.

    Returns:
        PER value for this utterance (0.0 ~ 1.0). Returns 0.0 if ref is empty.
    """
    if len(ref) == 0:
        return 0.0

    dist = levenshtein_distance(ref, hyp)
    return dist / float(len(ref))


def corpus_per(
    refs: List[Sequence[int]],
    hyps: List[Sequence[int]]
) -> float:
    """Computes corpus-level phoneme error rate (PER).

    All reference/hypothesis pairs are concatenated conceptually and
    a global PER is computed as:

        PER = (sum of edit distances) / (sum of reference lengths)

    Args:
        refs: List of reference sequences.
        hyps: List of hypothesis sequences.

    Returns:
        Corpus-level PER (0.0 ~ 1.0). Returns 0.0 if total ref length is 0.
    """
    assert len(refs) == len(hyps), \
        "refs and hyps must have the same number of utterances."

    total_dist = 0
    total_ref_len = 0

    for ref, hyp in zip(refs, hyps):
        total_dist += levenshtein_distance(ref, hyp)
        total_ref_len += len(ref)

    if total_ref_len == 0:
        return 0.0

    return total_dist / float(total_ref_len)
