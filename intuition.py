from collections import Counter
import math

A = 'aaabbb'
vocab_a = {
    'a': 'a',
    'b': 'b',
}

B = 'XY' # X means aaa, Y means bbb
vocab_b ={
    'a': 'a',
    'b': 'b',
    'X': 'aaa',
    'Y': 'bbb',
}

def entropy(s):
    total = 0
    l = len(s)
    cnt = Counter(s) 
    for k in cnt:
        p = cnt[k] / l
        total += -p * math.log2(p)
    return total

def avg_entropy(s, vocab):
    original = entropy(s)
    avg_l = sum([len(v) for v in vocab.values()]) / len(vocab)
    
    return original / avg_l

print(entropy(A)) # 1.0
print(entropy(B)) # 1.0

print(avg_entropy(A, vocab_a)) # 1.0
print(avg_entropy(B, vocab_b)) # 0.5