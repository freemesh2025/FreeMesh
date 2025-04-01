import numpy as np
import trimesh
import networkx as nx
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.getcwd()[:-3])  
from data.data_utils import  decimal_to_chinese, load_process_mesh, to_mesh
from data.serialization import serialize_meshxl, serialize_meshany, serialize_edgerunner
import matplotlib.pyplot as plt
import sentencepiece as spm



model_files = {
    'RAW_RMC': 'bpe/bpe_model/tokenizer_bpe_raw_rmc_8192_split.model',
    'AMT_RMC':'bpe/bpe_model/tokenizer_bpe_amt_rmc_8192_split.model',
    'EDR_RMC':'bpe/bpe_model/tokenizer_bpe_edr_rmc_8192_split.model',
}
 

def meshxl_tokenization(mesh, rearrange=False, sp=None):
    final_sequence = serialize_meshxl(mesh, rearrange=rearrange)
    if sp is not None:
        final_sequence = sp.encode(''.join([decimal_to_chinese(int(x)) for x in final_sequence]))
    return len(final_sequence)

def meshany_tokenization(mesh, rearrange=False, sp=None):
    final_sequence = serialize_meshany(mesh, rearrange=rearrange)
    if sp is not None:
        final_sequence = sp.encode(''.join([decimal_to_chinese(int(x)) for x in final_sequence]))
    return len(final_sequence)

def edgerunner_tokenization(mesh, rearrange=False, sp=None):
    final_sequence = serialize_edgerunner(mesh, rearrange=rearrange)
    if sp is not None:
        final_sequence = sp.encode(''.join([decimal_to_chinese(int(x)) for x in final_sequence]))
    return len(final_sequence)


def process_mesh(mesh_path):
    try:
        mesh = load_process_mesh(mesh_path, serialize_type=serialization_type.split('_')[0])
        verts, faces = mesh['vertices'], mesh['faces']
        faces = np.array(faces)
        face_num = faces.shape[0]
        mesh = to_mesh(vertices=verts, faces=faces, transpose=True)   
        if serialization_type in model_files:
            sp = spm.SentencePieceProcessor(model_file=model_files[serialization_type]) 
        if serialization_type == 'RAW_RMC':
            seq_len = meshxl_tokenization(mesh, rearrange=True, sp=sp)
        elif serialization_type == 'AMT_RMC':
            seq_len = meshany_tokenization(mesh, rearrange=True, sp=sp)   
        elif serialization_type == 'EDR_RMC':
            seq_len = edgerunner_tokenization(mesh, rearrange=True, sp=sp)
        elif serialization_type== 'RAW_RAC':
            seq_len = meshxl_tokenization(mesh, rearrange=True)
        elif serialization_type == 'AMT_RAC':
            seq_len = meshany_tokenization(mesh, rearrange=True)
        elif serialization_type == 'EDR_RAC':
            seq_len = edgerunner_tokenization(mesh, rearrange=True)

        return (face_num, seq_len)
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None
 
     

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def make_args_parser():
    parser = argparse.ArgumentParser("BPE", add_help=False)
    parser.add_argument('--serialization_types', nargs='+', default=['RAW_RAC','AMT_RAC','EDR_RAC','RAW_RMC','AMT_RMC','EDR_RMC'], type=str)
    args = parser.parse_args()
    return args


def visualize_distribution(distribution, serialization_type, name):
    if len(distribution) == 0:
        print(f"No data to visualize for {serialization_type} {name}")
        return
    plt.hist(distribution, bins=100, alpha=0.7, color='blue')
    plt.axvline(np.mean(distribution), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'{serialization_type} {name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    os.makedirs('distribution', exist_ok=True)
    plt.savefig(f'distribution/{serialization_type}_{name}.png')
    plt.close()

def main():
    data_list = read_file_list('mesh_list.txt')  # Your filter mesh dataset
    if len(data_list) == 0:
        print("Data list is empty. Exiting.")
        return
    data_list = np.random.choice(data_list, 10000, replace=False)
    args = make_args_parser()
    global serialization_type
    for serialization_type in args.serialization_types[3:]:
        try:
            # process_mesh(data_list[0])
            with Pool(processes=32) as pool:
                results = list(tqdm(pool.imap(process_mesh, data_list), total=len(data_list)))
            
           
            face_distribution = [result[0] for result in results if result is not None and result[1] < 9000]
            seq_len_distribution = [result[1] for result in results if result[1] < 9000]
            
            if len(face_distribution) == 0 or len(seq_len_distribution) == 0:
                print(f"No valid data for {serialization_type}. Skipping.")
                continue
            
            print(f"Serialization type: {serialization_type}")
            print(f"Average face number: {np.mean(face_distribution):.2f}")
            print(f"Average sequence length: {np.mean(seq_len_distribution):.2f}")
            
            visualize_distribution(face_distribution, serialization_type, 'face_distribution')
            visualize_distribution(seq_len_distribution, serialization_type, 'seq_len_distribution')
        
        except Exception as e:
            print(f"Error in main() for {serialization_type}: {e}")

if __name__ == "__main__":
    main()