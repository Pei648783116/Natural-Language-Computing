'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of

distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x
from typing import List, Sequence, Iterable


def grouper(seq:Sequence[str], n:int) -> List:
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''

    ngrams = []
    for i, token in enumerate(seq):
        if i >= len(seq) - (n-1):
            break
        else:
            ngrams.append(" ".join(str(seq[i:i+n])))

    return ngrams


def n_gram_precision(reference:Sequence[str], candidate:Sequence[str], n:int) -> float:
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    if len(candidate) == 0:
        p_n = 0
        return p_n
    else:
        candidate_gram = grouper(candidate, n)
        ref_gram = grouper(reference, n)
        c = 0
        for i, candidates in enumerate(candidate_gram):
            if candidates in ref_gram:
                c += 1
        num_ngrams = len(candidate_gram)
        p_n = c / num_ngrams if num_ngrams else 0.0

        # num_words = (len(candidate) - (n - 1))
        # if num_words > 0:
        #     p_n = c / num_words
        # else:
        #     p_n = 0
        # # assert False, "Fill me"
        return p_n



def brevity_penalty(reference:Sequence[str], candidate:Sequence[str]) -> float:
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    length = len(reference)
    # nearest_length = min((abs(l - len(candidate)), l) for l in lengths)[1]
    if len(candidate) == 0:
        BP = 0
        return BP
    else:
        brev = length / len(candidate)
        if brev < 1:
            BP = 1
        else:
            BP = exp(1.0 - brev)
        # assert False, "Fill me"
        return BP


def BLEU_score(reference:Sequence[str], candidate:Sequence[str], n) -> float:
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    brev_penalty = brevity_penalty(reference, candidate)
    total_p = 1

    for i in range(1, n+1):
        n_gram_p = n_gram_precision(reference, candidate, i)
        total_p = total_p * n_gram_p
    if n != 0:
        bleu = brev_penalty * (total_p ** (1/n))
    else:
        bleu = brev_penalty * total_p
    return bleu
