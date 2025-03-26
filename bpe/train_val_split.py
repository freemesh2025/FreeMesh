import os
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append(os.getcwd()[:-3])  
from data.data_utils import decimal_to_chinese


vocab = {
    'RAW':128,
    'RAW_RAC':128,
    'AMT':129,
    'AMT_RAC':129,
    'EDR':131,
    'EDR_RAC':131,
}

serialization_types = ['RAW','AMT','EDR','RAW_RAC','AMT_RAC','EDR_RAC']


for serialization_type in serialization_types:
    vocab_size = vocab[serialization_type]

    default_str = ''.join([decimal_to_chinese(x) for x in range(vocab_size+1)]) # guarantee that all characters are in the vocab

    def read_file(file):
        with open(f'corpus/{serialization_type}/{file}', 'r') as f:
            data = f.readlines()
            if len(data)==0: return None
            return data[0].strip()

    def write_data(data, file):
        with open(f'corpus/{file}', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

    # Create a multiprocessing Pool
    with Pool(processes=16) as pool:
        files = os.listdir(f'corpus/{serialization_type}')
        all_data = list(tqdm(pool.imap(read_file, files), total=len(files)))

    # Remove None values
    all_data = [data for data in all_data if data is not None]

    train_data = [default_str] + all_data[:int(len(all_data)*0.8)]
    val_data = [default_str] + all_data[int(len(all_data)*0.8):]

    write_data(train_data, f'{serialization_type}_train.txt')
    write_data(val_data, f'{serialization_type}_val.txt')