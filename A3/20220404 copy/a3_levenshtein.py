import os
import numpy as np
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    R = len(r)
    H = len(h)
    r.insert(0, "<s>")
    h.insert(0, "<s>")
    r.append("</s>")
    h.append("</s>")
    M = np.zeros((R+2, H+2))

    S = np.zeros((R+2, H+2))
    I = np.zeros((R+2, H+2))
    D = np.zeros((R+2, H+2))
    for i in range(1, R+2):
        M[i, 0] = i
        D[i, 0] = i
        for j in range(1, H+2):
            M[0, j] = j
            I[0, j] = j
            if r[i] == h[j]:            # Match
                M[i, j] = M[i-1, j-1]
                op = -1
            else:
                ops = [M[i-1, j-1] + 1, # Substitution
                       M[i, j-1] + 1,   # Insertion
                       M[i-1, j] + 1]   # Deletion

                M[i, j] = min(ops)
                op = np.argmin(ops)

            if op == -1:
                S[i, j] = S[i-1, j-1]
                I[i, j] = I[i-1, j-1]
                D[i, j] = D[i-1, j-1]
            if op == 0:
                S[i, j] = S[i-1, j-1] + 1
                I[i, j] = I[i-1, j-1]
                D[i, j] = D[i-1, j-1]
            if op == 1:
                S[i, j] = S[i, j-1]
                I[i, j] = I[i, j-1] + 1
                D[i, j] = D[i, j-1]
            if op == 2:
                S[i, j] = S[i-1, j]
                I[i, j] = I[i-1, j]
                D[i, j] = D[i-1, j] + 1
    '''
    if not M[R, H] == S[R, H] + I[R, H] + D[R, H]:
        bound = -1
        print(r[:bound])
        print(h[:bound])
        print(M[:bound, :bound])
        print(S[:bound, :bound])
        print(I[:bound, :bound])
        print(D[:bound, :bound])
    '''
    assert M[R, H] == S[R, H] + I[R, H] + D[R, H]
    wer = 100.0 * M[R, H] / R if R != 0 else Inf
    return [wer, int(S[R, H]), int(I[R, H]), int(D[R, H])]

def Preprocess(line):
    # In: line: str
    # Out: seq: List[str]
    
    # Remove punctuation, tags
    filtered = line
    filtered = re.sub(r'\<[A-Z]*\>', r'', filtered)
    filtered = re.sub(r'(\[[a-z]*\])', r'', filtered)
    filtered = re.sub(r'[A-Z]*\/[A-Z]*\:[A-Z]*', r'', filtered)
    filtered = re.sub(r'[\.\/\-\,\'\?\!]', r'', filtered)
    filtered = re.sub(r'[0-9]', r'', filtered)
    filtered = re.sub(r'\s+', r' ', filtered)
    filtered = re.sub(r'^\s', r'', filtered)
    filtered = re.sub(r'\s$', r'', filtered)
    # Lowercase
    lower = filtered.lower()
    # Split into tokens
    words = lower.split(' ')

    return words

'''
# Get mean/stdev
if __name__ == "__main__":
    f = open('asrDiscussion.txt', 'r')
    line = f.readline()
    g = []
    k = []
    while line:
        spl = line.split(' ')
        sys = spl[1]
        wer = float(spl[3])
        if sys == "Kaldi":
            k.append(wer)
        if sys == "Google":
            g.append(wer)
        line = f.readline()

    print("Google: mean=" + str(np.mean(g)) + " stdev=" + str(np.std(g)))
    print("Kaldi: mean=" + str(np.mean(k)) + " stdev=" + str(np.std(k)))
'''

if __name__ == "__main__":
    # print( 'TODO' ) 
    data_path = '/u/cs401/A3/data'
    for subdir, dirs, files in os.walk(data_path):
        for speaker in dirs:
            rfile = data_path + '/' + speaker + '/transcripts.txt'
            for system in ['Kaldi', 'Google']:
                hfile = data_path + '/' + speaker + '/transcripts.' + system + '.txt'
                ref = open(rfile, 'r')
                hyp = open(hfile, 'r')

                i = 0
                eof = False
                while not eof:
                    rline = ref.readline()
                    hline = hyp.readline()
                    if not rline or not hline:
                        eof = True
                    else:
                        r = Preprocess(rline)
                        h = Preprocess(hline)
                        wer, S, I, D = Levenshtein(r, h)
                        print(speaker + ' ' + system + ' ' + str(i) + ' ' +
                              str(wer) + ' S:' + str(S) + ', I:' + str(I) + ', D:' + str(D))
                        i += 1
                
