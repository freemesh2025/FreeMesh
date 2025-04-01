import os
import numpy as np
import trimesh
import pymeshlab
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from mesh_utils import read_obj, write_obj2
import shutil
import sys
import pymeshfix

# pip install pymeshlab==0.2.1

sys.setrecursionlimit(10000)

def fill_holes(obj_path, save_path):
    """
    Fill small holes in a mesh using pymeshfix.
    
    Args:
        obj_path: Path to input OBJ file
        save_path: Path to save repaired mesh
    """
    tin = pymeshfix.PyTMesh()
    tin.load_file(obj_path)
    tin.fill_small_boundaries(nbe=20, refine=True)
    tin.save_file(save_path)

def tri_to_quad(obj_path, save_path, level=1):
    """
    Convert triangular mesh to quad-dominant mesh using pymeshlab.
    
    Args:
        obj_path: Path to input OBJ file
        save_path: Path to save output mesh
        level: Aggressiveness level of quad conversion (1-3)
    """
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(obj_path)
        ms.apply_filter('repair_non_manifold_edges_by_removing_faces')
        ms.apply_filter('remove_duplicate_faces')
        ms.apply_filter('turn_into_quad_dominant_mesh', level=level)
        ms.save_current_mesh(save_path)
        repair_quad_mesh(obj_path, save_path, save_path, split_angle_threshold=45)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def repair_folded_faces(obj_path, save_path):
    """
    Repair folded faces by adjusting vertex positions based on normals.
    
    Args:
        obj_path: Path to input OBJ file
        save_path: Path to save repaired mesh
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    current_mesh = ms.current_mesh()
    
    # Select problematic faces
    ms.apply_filter('select_problematic_faces', usear=False, usenf=True, nfratio=179.9)
    all_vertices = current_mesh.vertex_matrix()
    all_faces = current_mesh.face_matrix()
    all_face_normals = current_mesh.face_normal_matrix()

    # Move selected vertices to new layer
    ms.apply_filter('move_selected_vertices_to_another_layer', deleteoriginal=False)
    
    # Adjust vertex positions based on normals
    vert_to_face = dict()
    for ind in range(len(all_faces)):
        for v in all_faces[ind]:
            if v not in vert_to_face:
                vert_to_face[v] = []
            vert_to_face[v].append(ind)

    for v in vert_to_face:
        if len(vert_to_face[v]) == 1:
            cur_face = vert_to_face[v][0]
            cur_face_normal = all_face_normals[cur_face]
            all_vertices[v] += cur_face_normal * 0.002

    # Save repaired mesh
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    mesh.export(save_path)

def repair_quad_mesh(tri_obj_path, quad_obj_path, save_path, split_angle_threshold=30):
    """
    Repair quadrilateral mesh by splitting quads with high angle deviation.
    
    Args:
        tri_obj_path: Path to original triangular mesh
        quad_obj_path: Path to quad-dominant mesh
        save_path: Output path for repaired mesh
        split_angle_threshold: Angle threshold for splitting quads (degrees)
    """
    tri_mesh = read_obj(tri_obj_path)
    quad_mesh = read_obj(quad_obj_path)
    ori_faces = {'-'.join([str(a) for a in sorted(face)]): face for face in tri_mesh['faces']}
    
    new_faces = []
    for face in quad_mesh['faces']:
        face = [a-1 for a in face]
        if len(face) == 3:
            new_faces.append([a+1 for a in face])
        elif len(face) == 4:
            # Calculate normals for potential splits
            face1_verts = quad_mesh['vertices'][face[:3]]
            face2_verts = quad_mesh['vertices'][[face[0], face[2], face[3]]]
            normal1 = np.cross(face1_verts[1]-face1_verts[0], face1_verts[2]-face1_verts[0])
            normal2 = np.cross(face2_verts[1]-face2_verts[0], face2_verts[2]-face2_verts[0])
            normal1 /= np.linalg.norm(normal1)
            normal2 /= np.linalg.norm(normal2)
            
            # Split quads with large angle deviation
            degree = np.degrees(np.arccos(np.dot(normal1, normal2)))
            if degree > split_angle_threshold:
                new_faces.extend([[face[0]+1, face[1]+1, face[2]+1],
                                [face[0]+1, face[2]+1, face[3]+1]])
            else:
                new_faces.append([face[0]+1, face[1]+1, face[2]+1, face[3]+1])

    # Save repaired mesh
    write_obj2(save_path, {'vertices': quad_mesh['vertices'], 'faces': new_faces})

def repair_mesh(obj_path, save_path):
    """
    Main mesh repair pipeline combining multiple repair steps.
    
    Args:
        obj_path: Path to input OBJ file
        save_path: Path to save repaired mesh
    """
    # Hole filling
    fill_holes(obj_path, save_path)
    cur_path = save_path
    
    # Remove small disconnected components
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cur_path)
    ms.apply_filter('select_small_disconnected_component', nbfaceratio=0.08, nonclosedonly=True)
    ms.apply_filter('delete_selected_faces_and_vertices')
    ms.save_current_mesh(save_path)
    cur_path = save_path
    
    # Repair folded faces
    repair_folded_faces(cur_path, save_path)


def mesh_post_process(obj_path, target_root=None, repair=True, to_quad=True):
    """
    Complete post-processing pipeline for 3D meshes.
    
    Args:
        obj_path: Path to input OBJ file
        target_root: Output directory
        repair: Enable mesh repair steps
        to_quad: Enable quad conversion
    """
    name = os.path.basename(obj_path)
    basename = os.path.splitext(name)[0]
    target_root = target_root or os.path.join(os.path.dirname(obj_path), basename)
    os.makedirs(target_root, exist_ok=True)

    # Processing pipeline
    cur_path = os.path.join(target_root, f'{basename}_ori.obj')
    shutil.copy(obj_path, cur_path)
    
    if repair:
        cur_path = os.path.join(target_root, f'{basename}_repair.obj')
        repair_mesh(obj_path, cur_path)
    
    if to_quad:
        cur_path = os.path.join(target_root, f'{basename}_quad.obj')
        tri_to_quad(cur_path, cur_path)
    
    return {'final_model_path': cur_path}

def process_mesh_batch(data_root, target_root=None):
    """
    Batch process all meshes in a directory.
    
    Args:
        data_root: Input directory containing OBJ files
        target_root: Output directory for processed files
    """
    target_root = target_root or f'{data_root}_postprocess'
    os.makedirs(target_root, exist_ok=True)
    
    names = [name for name in os.listdir(data_root) if name.endswith('.obj')]
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(
            lambda name: mesh_post_process(os.path.join(data_root, name), target_root),
            names
        ), total=len(names)))

if __name__ == '__main__':
    data_roots = [
        '/path/to/mesh/directory',
    ]
    for root in data_roots:
        process_mesh_batch(root)