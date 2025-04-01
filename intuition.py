from collections import Counter
import math


def entropy(s):
    naive_entropy = 0
    cnt = Counter(s) 
    for c,freq in cnt.items():
        p = freq / len(s)
        naive_entropy += -p * math.log2(p)
    return naive_entropy

def avg_entropy(s, vocab):
    naive_entropy = 0
    avg_l = 0
    cnt = Counter(s)
    for c,freq in cnt.items():
        p = freq / len(s)
        naive_entropy += -p * math.log2(p)
        avg_l += p * len(vocab[c])
    return naive_entropy / avg_l


######################
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
print(avg_entropy(B, vocab_b)) # 0.33

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
    'd': 'd',
    'X': 'aa',
    'Y': 'bb',
}



print(avg_entropy(A, vocab_a)) # baseline 1.837
print(avg_entropy(B, vocab_b)) # + MC 2.0
print(avg_entropy(C, vocab_c)) # + RAC 1.837  
print(avg_entropy(D, vocab_d)) # + RMC 0.846