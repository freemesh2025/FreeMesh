import time 
import os
import sys
sys.path.append(os.getcwd()[:-3])
import sentencepiece as spm
from data.data_utils import decimal_to_chinese




serialization_type_list = ['raw_mc','amt_mc','amt_mc','amt_rmc','edr_mc','edr_rmc']  

defined_symbols_mpt = {
    'raw_mc': [],
    'raw_rmc': [],
    'amt_mc': [decimal_to_chinese(128)],
    'amt_rmc': [decimal_to_chinese(128)],
    'edr_mc': [decimal_to_chinese(2)],  
    'edr_rmc': [decimal_to_chinese(2)], 

}
vocab_size_mpt = {
    'raw_mc': [256,512,1024,2048,4096,8192],
    'raw_rmc': [256,512,1024,2048,4096,8192],
    'amt_mc': [256,512,1024,2048,4096,8192],
    'amt_rmc': [256,512,1024,2048,4096,8192],
    'edr_mc': [256,512,1024,2048,4096,8192], 
    'edr_rmc': [256,512,1024,2048,4096,8192], 
} 

name_mpt = {
    'raw_mc': 'RAW',
    'raw_rmc': 'RAW_RAC',
    'amt_mc': 'AMT',
    'amt_rmc': 'AMT_RAC',
    'edr_mc': 'EDR',
    'edr_rmc': 'EDR_RAC',
}

for serialization_type in serialization_type_list:
    vocab_size_list = vocab_size_mpt[serialization_type]
    for vocab_size in vocab_size_list:
        start = time.time()
        model_path = os.path.join('/apdcephfs_cq5/share_300600172/jackjliu/code/meshgpt/bpe/bpe_model',f'tokenizer_bpe_{serialization_type}_{vocab_size}_split.model')
        spm.SentencePieceTrainer.train(input=f'corpus/{name_mpt[serialization_type]}_train.txt', model_type='bpe', model_prefix=f'bpe_model/tokenizer_bpe_{serialization_type}_{vocab_size}_split', vocab_size=vocab_size, 
                                            character_coverage=1.0, user_defined_symbols=defined_symbols_mpt[serialization_type], max_sentence_length=65536, unk_id=0, bos_id=-1, eos_id=-1)
        print(f'{vocab_size} {time.time()-start} ') # ~2500