import trimesh
import numpy as np
import os
import sys
sys.path.append(os.getcwd()[:-4]) # Add root path
from data.data_utils import load_process_mesh, to_mesh, decimal_to_chinese, chinese_to_decimal, undiscretize # undiscretize
import networkx as nx
import torch
import sentencepiece as spm
from data.edgerunner import Engine


def rac_encode(nums):
    """RAC encode the number sequence"""
    X = nums[0::3]
    Y = nums[1::3]
    Z = nums[2::3]
    return X + Y + Z

def rac_encode2(nums): 
    """RAC encode with improved handling for larger sequences"""
    if len(nums) < 9: 
        return rac_encode(nums)
    remainder = len(nums) % 9
    head_start = 0
    head_end = 9
    head = rac_encode(nums[head_start:head_end])
    neck_start = head_end
    neck_end = len(nums) - remainder
    neck_len = (neck_end - neck_start) // 9
    neck = []
    for i in range(neck_len):
        cur_seq = nums[neck_start+i*9:neck_start+(i+1)*9]
        neck.extend(rac_encode(cur_seq))
    if remainder > 0:
        tail = rac_encode(nums[neck_end:]) 
    else:
        tail = []
    return head + neck + tail

def rac_decode(nums):
    """RAC decode the number sequence"""
    k = len(nums) // 3
    X = nums[:k]
    Y = nums[k:2*k]
    Z = nums[2*k:]
    return [val for triplet in zip(X, Y, Z) for val in triplet]

def rac_decode2(nums): 
    """RAC decode with error handling for invalid sequences"""
    if len(nums) < 9: 
        return rac_decode(nums)
    remainder = len(nums) % 9
    if remainder < 3:
        nums = nums[:-remainder]
        remainder = 0
    elif remainder < 6:
        nums = nums[:- (remainder - 3)]
        remainder = 3
    else:
        nums = nums[:- (remainder - 6)]
        remainder = 6
    
    head_start = 0
    head_end = 9
    head = rac_decode(nums[head_start:head_end])
    neck_start = head_end
    neck_end = len(nums) - remainder
    neck_len = (neck_end - neck_start) // 9
    neck = []
    for i in range(neck_len):
        cur_seq = nums[neck_start+i*9:neck_start+(i+1)*9]
        neck.extend(rac_decode(cur_seq))
    tail = rac_decode(nums[neck_end:]) if remainder > 0 else []
    return head + neck + tail


def serialize_meshxl(mesh: trimesh.Trimesh, quantization_bits=7, rearrange=False):
    """
    Serialize mesh using MeshXL format
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        quantization_bits (int): Number of quantization bits
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        np.ndarray: Serialized sequence
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    assert (vertices <= 0.5).all() and (vertices >= -0.5).all()  # [-0.5, 0.5]
    
    dis_vertices = np.asarray((vertices + 0.5) * (1 << quantization_bits))
    final_sequence = []
    for face in faces:
        for token_id in face:
            final_sequence.extend(dis_vertices[token_id].tolist())
    if rearrange:
        final_sequence = rac_encode2(final_sequence)
    
    return np.array(final_sequence).astype(np.int64)

def deserialize_meshxl(sequence, quantization_bits=7, coor_continuous_range=(-0.5, 0.5), rearrange=False):
    """
    Deserialize MeshXL sequence
    
    Args:
        sequence (np.ndarray): Serialized sequence
        quantization_bits (int): Number of quantization bits
        coor_continuous_range (tuple): Continuous coordinate range
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        np.ndarray: Deserialized vertices
    """
    vertices = []
    n_discrete_size = 1 << quantization_bits
    
    if len(sequence) % 9 != 0: 
        sequence = sequence[:-(len(sequence) % 9)]
    
    if rearrange:
        sequence = rac_decode2(sequence)
        
    for i in range(0, len(sequence), 9):
        for j in range(3):
            for k in range(3):
                id = sequence[i + 3*j + k] 
                vertices.append(undiscretize(id, coor_continuous_range[0], coor_continuous_range[1], n_discrete_size))
                
    if len(vertices) % 9 != 0:
        vertices = vertices[:-(len(vertices) % 9)]
    return np.array(vertices).reshape(-1, 3)

def serialize_meshany(mesh: trimesh.Trimesh, quantization_bits=7, rearrange=False):
    """
    Serialize mesh using MeshAny format
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        quantization_bits (int): Number of quantization bits
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        np.ndarray: Serialized sequence
    """
    graph = mesh.vertex_adjacency_graph
    unvisited_faces = mesh.faces.copy()
    vertices = mesh.vertices.copy()
    assert (vertices <= 0.5).all() and (vertices >= -0.5).all()  # [-0.5, 0.5]
    
    resolution = 1 << quantization_bits
    dis_vertices = np.asarray((vertices + 0.5) * resolution).astype(int)
    sequence = []
    
    while unvisited_faces.shape[0] > 0:
        if not sequence or sequence[-1] == -1:
            cur_face = unvisited_faces[0]
            unvisited_faces = unvisited_faces[1:]
            sequence.extend(cur_face.tolist())
        else:
            last_vertices = sequence[-2:]
            commons = sorted(list(nx.common_neighbors(graph, last_vertices[0], last_vertices[1])))
            next_token = None
            for common in commons:
                common_face = sorted(last_vertices + [common])
                equals = np.where((unvisited_faces == common_face).all(axis=1))[0]
                if len(equals) == 1:
                    next_token = common
                    next_face_index = equals[0]
                    break
            if next_token is not None:
                unvisited_faces = np.delete(unvisited_faces, next_face_index, axis=0)
                sequence.append(int(next_token))
            else:
                sequence.append(-1)

    final_sequence = []
    for token_id in sequence:
        if token_id == -1:
            final_sequence.append(resolution)
        else:
            final_sequence.extend(dis_vertices[token_id].tolist())
            
    if rearrange:
        final_sequence = []
        sub_sequence = []
        for token_id in sequence:
            if token_id == -1:
                final_sequence.extend(rac_encode2(sub_sequence))
                final_sequence.append(resolution)
                sub_sequence = []
            else:
                sub_sequence.extend(dis_vertices[token_id].tolist())
        if sub_sequence:
            final_sequence.extend(rac_encode2(sub_sequence))
    
    return np.array(final_sequence).astype(np.int64)


def deserialize_meshany(sequence, n_max_triangles=10000*3, quantization_bits=7, coor_continuous_range=(-0.5, 0.5), rearrange=False):
    """
    Deserialize MeshAny sequence
    
    Args:
        sequence (np.ndarray): Serialized sequence
        n_max_triangles (int): Maximum number of triangles
        quantization_bits (int): Number of quantization bits
        coor_continuous_range (tuple): Continuous coordinate range
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        np.ndarray: Deserialized vertices
    """
    continuous_coors = torch.zeros(n_max_triangles, 3)
    continuous_coors[...] = float('nan')
    continuous_coors[:3, :] = torch.tensor([[-0.1, 0.0, 0.1], [-0.1, 0.1, 0.2], [-0.3, 0.3, 0.2]])
    coor_loop_check = 0
    vertice_count = 0
    n_discrete_size = 1 << quantization_bits
    
    if rearrange:
        new_sequence = []
        sub_sequence = []
        for id in sequence:
            if id == n_discrete_size:
                new_sequence.extend(rac_decode2(sub_sequence))
                new_sequence.append(n_discrete_size)
                sub_sequence = []
            else:
                sub_sequence.append(id)
        if sub_sequence:
            new_sequence.extend(rac_decode2(sub_sequence))
        sequence = new_sequence
    
    for id in sequence:
        if id == n_discrete_size:
            if coor_loop_check < 9:
                break
            if coor_loop_check % 3 != 0:
                break
            coor_loop_check = 0
        else:
            if coor_loop_check % 3 == 0 and coor_loop_check >= 9:
                continuous_coors[vertice_count] = continuous_coors[vertice_count-2]
                continuous_coors[vertice_count+1] = continuous_coors[vertice_count-1]
                vertice_count += 2
            continuous_coors[vertice_count, coor_loop_check % 3] = undiscretize(id, coor_continuous_range[0], coor_continuous_range[1], n_discrete_size)
            if coor_loop_check % 3 == 2: 
                vertice_count += 1
            coor_loop_check += 1
            
    valid_mask = torch.all(~torch.isnan(continuous_coors), dim=1)
    vertices = continuous_coors[valid_mask] 
    if len(vertices) % 9 != 0: 
        vertices = vertices[:-(len(vertices) % 9)]
    return np.array(vertices)

def serialize_edgerunner(mesh: trimesh.Trimesh, quantization_bits=7, rearrange=False):
    """
    Serialize mesh using EdgeRunner format
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        quantization_bits (int): Number of quantization bits
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        np.ndarray: Serialized sequence
    """
    engine = Engine(discrete_bins=1 << quantization_bits)
    vertices, faces = mesh.vertices, mesh.faces
    
    final_sequence, _, _ = engine.encode(vertices, faces)

    if rearrange:
        sub_sequences = []
        sub_sequence = []
        sub_special_sequence = []
        for num in final_sequence:
            if num in {0, 1}:
                sub_special_sequence.append(num)
            elif num == 2:
                if sub_sequence:
                    sub_sequence = [2] + sub_special_sequence + rac_encode2(sub_sequence)
                    sub_sequences.extend(sub_sequence)
                    sub_special_sequence = []
                    sub_sequence = []
            else:
                sub_sequence.append(num)
        if sub_sequence:
            sub_sequence = [2] + sub_special_sequence + rac_encode2(sub_sequence)
            sub_sequences.extend(sub_sequence)
        final_sequence = sub_sequences
    return np.array(final_sequence).astype(np.int64)


def deserialize_edgerunner(sequence, quantization_bits=7, coor_continuous_range=(-0.5, 0.5), rearrange=False):
    """
    Deserialize EdgeRunner sequence
    
    Args:
        sequence (np.ndarray): Serialized sequence
        quantization_bits (int): Number of quantization bits
        coor_continuous_range (tuple): Continuous coordinate range
        rearrange (bool): Whether to apply RAC rearrangement
        
    Returns:
        tuple: (vertices, decoded_faces)
    """
    engine = Engine(discrete_bins=1 << quantization_bits)
    if rearrange:
        new_sequence = []
        sub_sequence = []
        sub_special_sequence = []
        for num in sequence:
            if num in {0, 1}:
                sub_special_sequence.append(num)
            elif num == 2:
                if sub_sequence:
                    sub_sequence = rac_decode2(sub_sequence)
                    cur_sequence = [2] + sub_sequence[:9]
                    sub_sequence = sub_sequence[9:]
                    for i in range(0, len(sub_sequence), 3):
                        cur_sequence.append(sub_special_sequence[i//3])
                        cur_sequence.extend(sub_sequence[i:i+3])
                    new_sequence.extend(cur_sequence)
                    sub_special_sequence = []
                    sub_sequence = []
            else:
                sub_sequence.append(num)
        if sub_sequence:
            sub_sequence = rac_decode2(sub_sequence)
            cur_sequence = [2] + sub_sequence[:9]
            sub_sequence = sub_sequence[9:]
            for i in range(0, len(sub_sequence), 3):
                cur_sequence.append(sub_special_sequence[i//3])
                cur_sequence.extend(sub_sequence[i:i+3])
            new_sequence.extend(cur_sequence)
        sequence = new_sequence
    
    vertices, decoded_faces, decoded_face_type = engine.decode(sequence)
    return vertices, decoded_faces


def bpe_encode_decode(sp, codes):
    """
    Perform BPE encoding and decoding
    
    Args:
        sp (spm.SentencePieceProcessor): BPE model
        codes (np.ndarray): Input codes
        
    Returns:
        tuple: (encoded_codes, decoded_codes)
    """
    codes = sp.encode(''.join([decimal_to_chinese(int(x)) for x in codes]))
    decodes = sp.decode(codes)
    return codes, np.array([chinese_to_decimal(x) for x in decodes])

def validate(mesh_path='phone.obj', serialize_type='RAW', quantization_bits=7, rearrange=False):
    """
    Validate serialization/deserialization pipeline
    
    Args:
        mesh_path (str): Path to input mesh
        serialize_type (str): Serialization type (RAW/AMT/EDR)
        quantization_bits (int): Number of quantization bits
        rearrange (bool): Whether to apply RAC rearrangement
    """
    mesh = load_process_mesh(mesh_path, quantization_bits=quantization_bits, serialize_type=serialize_type)
    verts, faces, verts_no_quant, faces_no_quant = mesh['vertices'], mesh['faces'], mesh['vertices_no_quant'], mesh['faces_no_quant']
    verts = np.array(verts)
    faces = np.array(faces)
    verts_no_quant = np.array(verts_no_quant)
    faces_no_quant = np.array(faces_no_quant)
    
    mesh_no_quant = to_mesh(vertices=verts_no_quant, faces=faces_no_quant, transpose=True)
    mesh = to_mesh(vertices=verts, faces=faces, transpose=True)
    
    model_files = {
        'RAW_MC': 'bpe/bpe_model/tokenizer_bpe_raw_mc_8192_split.model',
        'RAW_RMC': 'bpe/bpe_model/tokenizer_bpe_raw_rmc_8192_split.model',
        'AMT_MC':'bpe/bpe_model/tokenizer_bpe_amt_mc_8192_split.model',
        'AMT_RMC':'bpe/bpe_model/tokenizer_bpe_amt_rmc_8192_split.model',
        'EDR_MC':'bpe/bpe_model/tokenizer_bpe_edr_mc_8192_split.model',
        'EDR_RMC':'bpe/bpe_model/tokenizer_bpe_edr_rmc_8192_split.model',
    }
    model_name = f"{serialize_type}_RMC" if rearrange else f"{serialize_type}_MC"
    sp = spm.SentencePieceProcessor(model_file=model_files[model_name])
    
    if serialize_type == 'RAW':
        serialized_mesh = serialize_meshxl(mesh, quantization_bits=quantization_bits, rearrange=rearrange)
        codes, serialized_mesh = bpe_encode_decode(sp, serialized_mesh)
        vertices = deserialize_meshxl(serialized_mesh, quantization_bits=quantization_bits, rearrange=rearrange)
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
    elif serialize_type == 'AMT':
        serialized_mesh = serialize_meshany(mesh, quantization_bits=quantization_bits, rearrange=rearrange)
        codes, serialized_mesh = bpe_encode_decode(sp, serialized_mesh)
        vertices = deserialize_meshany(serialized_mesh, quantization_bits=quantization_bits, rearrange=rearrange)
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
    elif serialize_type == 'EDR':
        serialized_mesh = serialize_edgerunner(mesh, quantization_bits=quantization_bits, rearrange=rearrange)
        codes, serialized_mesh = bpe_encode_decode(sp, serialized_mesh)
        vertices, faces = deserialize_edgerunner(serialized_mesh, quantization_bits=quantization_bits, rearrange=rearrange)
    
    decoded_mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
    num_faces = len(decoded_mesh.faces)
    face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
    face_colors = np.tile(face_color, (num_faces, 1))
    decoded_mesh.visual.face_colors = face_colors
    
    target_dir = f'examples/test/{model_name}'
    os.makedirs(target_dir, exist_ok=True)
    mesh_name = os.path.basename(mesh_path).split('.')[0]
    
    decoded_mesh.export(f'{target_dir}/{mesh_name}_{len(codes)/(num_faces * 9):.2f}.obj')

if __name__ == '__main__':
    mesh_path = 'examples/phone.obj'
    for serialize_type in ('RAW', 'AMT', 'EDR'): 
        validate(mesh_path=mesh_path, serialize_type=serialize_type, rearrange=False) 
        validate(mesh_path=mesh_path, serialize_type=serialize_type, rearrange=True)