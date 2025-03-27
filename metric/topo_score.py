import numpy as np
import math
import os
from collections import defaultdict

def read_obj(obj_path, print_shape=False):
    """
    Read OBJ file and parse into mesh data structure.
    
    Args:
        obj_path (str): Path to OBJ file
        print_shape (bool): Whether to print vertex/face counts
        
    Returns:
        dict: Mesh data containing vertices, faces, and optional UVs/normals
    """
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0

    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a) > 0]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a) > 0]
            max_face_length = max(max_face_length, len(face))
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1]) > 0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a) > 0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(items[0].split('/')[2]) > 0:
                face_normal = [int(a.split('/')[2]) for a in items if len(a) > 0]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            uv = [float(a) for a in line.strip().split(' ')[1:] if len(a) > 0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            vn = [float(a) for a in line.strip().split(' ')[1:] if len(a) > 0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)

    mesh = {
        'vertices': vertices[:, :3],
        'faces': faces
    }
    if vertices.shape[1] > 3:
        mesh['colors'] = vertices[:, 3:]

    if uvs:
        mesh['UVs'] = np.array(uvs).astype(np.float32)
    if vns:
        mesh['normals'] = np.array(vns).astype(np.float32)
    if faces_uv:
        mesh['faces_uv'] = np.array(faces_uv).astype(np.int32)
    if faces_normal:
        mesh['faces_normal'] = np.array(faces_normal).astype(np.int32)

    if print_shape:
        print(f'Vertices: {len(vertices)}, Faces: {len(faces)}')
    
    return mesh

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors in degrees (0-180)
    
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    
    Returns:
        float: Angle in degrees, NaN if zero vectors
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return float('nan')
    cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

def safe_mean(data, default=0.0):
    """
    Safely calculate mean avoiding division by zero
    
    Args:
        data (list): Input data list
        default (float): Default value if data is empty
    
    Returns:
        float: Mean value or default
    """
    return np.mean(data) if len(data) > 0 else default

def evaluate_topology(obj_path):
    """
    Evaluate quad mesh topological quality
    
    Args:
        obj_path (str): Path to input OBJ file
    
    Returns:
        float: Quality score (0-100)
    """
    quad_mesh = read_obj(obj_path)
    vertices = quad_mesh.get('vertices', [])
    faces = quad_mesh.get('faces', [])
    
    n_tri = 0
    n_quad = 0
    edge_map = defaultdict(list)
    quad_metrics = {
        'ratio_wh': [],
        'min_angle': [],
        'max_angle': [],
        'edge_ratios': []
    }
    neighbor_metrics = {'wh_consistency': []}
    quad_index_map = {}
    quad_count = 0

    # First pass: Collect basic metrics
    for face_idx, face in enumerate(faces):
        quad_index_map[face_idx] = quad_count
        quad_count += 1
        face = [a - 1 for a in face]
        n_edges = len(face)

        if n_edges == 3:
            n_tri += 1
        elif n_edges == 4:
            n_quad += 1
            edges = []
            for i in range(4):
                v1 = face[i]
                v2 = face[(i+1)%4]
                edge = tuple(sorted([v1, v2]))
                edges.append(edge)
                edge_map[edge].append(face_idx)

            edge_lengths = [
                np.linalg.norm(vertices[face[i]] - vertices[face[(i+1)%4]])
                for i in range(4)
            ]

            opposite_ratio = [
                max(edge_lengths[0], edge_lengths[2]) / min(edge_lengths[0], edge_lengths[2]),
                max(edge_lengths[1], edge_lengths[3]) / min(edge_lengths[1], edge_lengths[3])
            ]
            ratio_wh = max(opposite_ratio)
            quad_metrics['ratio_wh'].append(ratio_wh)

            angles = []
            for i in range(4):
                vec1 = vertices[face[(i+1)%4]] - vertices[face[i]]
                vec2 = vertices[face[i-1]] - vertices[face[i]]
                angle = calculate_angle(vec1, vec2)
                if not math.isnan(angle):
                    angles.append(angle)

            if angles:
                quad_metrics['min_angle'].append(min(angles))
                quad_metrics['max_angle'].append(max(angles))

            max_len = max(edge_lengths)
            if max_len > 0:
                edge_ratio = sum(l/max_len for l in edge_lengths) / 4
                quad_metrics['edge_ratios'].append(edge_ratio)

    # Second pass: Calculate neighbor consistency
    for face_idx, face in enumerate(faces):
        if len(face) != 4:
            continue
        
        current_quad_index = quad_index_map.get(face_idx, -1)
        if current_quad_index == -1:
            continue

        if current_quad_index >= len(quad_metrics['ratio_wh']):
            continue

        neighbors = set()
        for i in range(4):
            v1 = face[i] - 1
            v2 = face[(i+1)%4] - 1
            edge = tuple(sorted([v1, v2]))
            for neighbor in edge_map[edge]:
                if neighbor != face_idx:
                    neighbors.add(neighbor)

        if neighbors:
            neighbor_ratios = []
            for n_idx in neighbors:
                if len(faces[n_idx]) == 4 and n_idx < len(faces):
                    if n_idx < len(quad_metrics['ratio_wh']):
                        neighbor_ratios.append(quad_metrics['ratio_wh'][n_idx])

            if neighbor_ratios:
                current_ratio = quad_metrics['ratio_wh'][current_quad_index]
                avg_diff = np.mean([abs(current_ratio - r) for r in neighbor_ratios])
                neighbor_metrics['wh_consistency'].append(1 / (1 + avg_diff))

    # Calculate final score
    total_faces = n_tri + n_quad
    if total_faces == 0:
        return 0.0

    WEIGHTS = {
        'quad_ratio': 0.4,
        'shape_quality': 0.3,
        'angle_quality': 0.2,
        'neighbor_consistency': 0.1
    }

    quad_ratio = n_quad / total_faces
    shape_quality = 0.5 * (1 / safe_mean(quad_metrics['ratio_wh'], 1)) + \
                   0.5 * safe_mean(quad_metrics['edge_ratios'], 0)
    angle_deviation = (np.abs(safe_mean(quad_metrics['min_angle']) - 90) +
                       np.abs(safe_mean(quad_metrics['max_angle']) - 90))
    angle_quality = 1 - (angle_deviation / 180)
    neighbor_consistency = safe_mean(neighbor_metrics['wh_consistency'], 0)

    score = 100 * (
        WEIGHTS['quad_ratio'] * quad_ratio +
        WEIGHTS['shape_quality'] * shape_quality +
        WEIGHTS['angle_quality'] * angle_quality +
        WEIGHTS['neighbor_consistency'] * neighbor_consistency
    )

    return np.clip(score, 0, 100)