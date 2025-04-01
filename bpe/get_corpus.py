
import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append(os.getcwd()[:-3]) # add the root path
from data.data_utils import  decimal_to_chinese, load_process_mesh, to_mesh
from data.serialization import serialize_meshxl, serialize_meshany, serialize_edgerunner

 

def meshxl_tokenization(mesh, name, corpus_dir, rearrange=False):
    faces = mesh.faces.copy()
    naive_v_length = faces.shape[0] * 9
    
    final_sequence = serialize_meshxl(mesh, rearrange=rearrange)
    final_sequence = ''.join(decimal_to_chinese(int(x)) for x in final_sequence)
    cur_ratio = len(final_sequence) / naive_v_length
    
    with open(f'{corpus_dir}/{name}.txt', 'w') as f:
        f.write(final_sequence)
    
    return cur_ratio

def meshany_tokenization(mesh, name, corpus_dir, rearrange=False):
    faces = mesh.faces.copy()
    naive_v_length = faces.shape[0] * 9
    
    final_sequence = serialize_meshany(mesh, rearrange=rearrange)
    final_sequence = ''.join(decimal_to_chinese(int(x)) for x in final_sequence)
    cur_ratio = len(final_sequence) / naive_v_length
    
    with open(f'{corpus_dir}/{name}.txt', 'w') as f:
        f.write(final_sequence)
    
    return cur_ratio

def edgerunner_tokenization(mesh, name, corpus_dir, rearrange=False):
    faces = mesh.faces.copy()
    naive_v_length = faces.shape[0] * 9
    
    final_sequence = serialize_edgerunner(mesh, rearrange=rearrange)
    final_sequence = ''.join(decimal_to_chinese(int(x)) for x in final_sequence)
    cur_ratio = len(final_sequence) / naive_v_length
    
    with open(f'{corpus_dir}/{name}.txt', 'w') as f:
        f.write(final_sequence)
    
    return cur_ratio

def process_mesh(mesh_path):
    try:
        mesh = load_process_mesh(mesh_path, serialize_type=serialization_type.split('_')[0])
        verts, faces = mesh['vertices'], mesh['faces']
        faces = np.array(faces)
        mesh = to_mesh(vertices=verts, faces=faces, transpose=True)    
        name = os.path.basename(mesh_path).split('.')[0]
        if serialization_type == 'RAW':
            cur_ratio = meshxl_tokenization(mesh, name, corpus_dir, rearrange=False)
        elif serialization_type == 'AMT':
            cur_ratio = meshany_tokenization(mesh, name, corpus_dir, rearrange=False)   
        elif serialization_type == 'EDR':
            cur_ratio = edgerunner_tokenization(mesh, name, corpus_dir, rearrange=False)
        elif serialization_type== 'RAW_RAC':
            cur_ratio = meshxl_tokenization(mesh, name, corpus_dir, rearrange=True)
        elif serialization_type == 'AMT_RAC':
            cur_ratio = meshany_tokenization(mesh, name, corpus_dir, rearrange=True)
        elif serialization_type == 'EDR_RAC':
            cur_ratio = edgerunner_tokenization(mesh, name, corpus_dir, rearrange=True)
        return cur_ratio
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None
 
     

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def make_args_parser():
    parser = argparse.ArgumentParser("BPE", add_help=False)
    parser.add_argument('--serialization_types', nargs='+', default=['RAW','AMT','EDR','RAW_RAC','AMT_RAC','EDR_RAC'], type=str)
    args = parser.parse_args()
    return args



def main():
    data_list = read_file_list('mesh_list.txt') # your filtered mesh dataset
    data_list = sorted(data_list)[:100]
    args = make_args_parser()
    global corpus_dir 
    global serialization_type
    for serialization_type in args.serialization_types:
        serialization_type = serialization_type
        corpus_dir = f'corpus/{serialization_type}'   
        os.makedirs(corpus_dir, exist_ok=True)
        # process_mesh(data_list[0])
        try:
            with Pool(processes=32) as pool:
                results = list(tqdm(pool.imap(process_mesh, data_list), total=len(data_list)))
            
            ratio_list = [result for result in results if result is not None]
            
            print(f"Serialization type: {serialization_type}")
            print(f"Mean ratio: {np.mean(ratio_list)}, Variance ratio: {np.var(ratio_list)}")
        except Exception as e:
            print(f"Error in main() for {serialization_type}: {e}")


if __name__ == "__main__":
    main()