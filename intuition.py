from collections import Counter
import math


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


################
# Explain Why naive entropy is not enough

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


print(entropy(A)) # 1.0
print(entropy(B)) # 1.0

print(avg_entropy(A, vocab_a)) # 1.0 
print(avg_entropy(B, vocab_b)) # 0.5

######################
A = 'aadabbacc'
vocab_a = {
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
}

B = 'Xdabbacc' # X means aa
vocab_b ={
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'X': 'aa',

}

C = 'aaaabcdbc' # rearrange A
vocab_c ={
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
}

D = 'XXYdY' # X means aa, Y means bb
vocab_d ={
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'X': 'aa',
    'Y': 'bb',
}



print(avg_entropy(A, vocab_a)) # baseline 1.837
print(avg_entropy(B, vocab_b)) # + MC 1.875
print(avg_entropy(C, vocab_c)) # + RAC 1.837  
print(avg_entropy(D, vocab_d)) # + RMC 1.087 