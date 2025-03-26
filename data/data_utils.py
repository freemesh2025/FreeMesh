"""Mesh data utilities."""
import numpy as np
import trimesh
import mesh2sdf.core
import skimage.measure


def chinese_to_decimal(chinese_char):
    return ord(chinese_char) - 20000

def decimal_to_chinese(number):
    return chr(number + 20000)


def process_mesh(vertices, faces, quantization_bits=7, serialize_type='AMT'):
    """Process mesh vertices and faces."""

    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    vertices = vertices.clip(-0.5, 0.5)
    
    vocab_size = 1 << quantization_bits
    # Transpose so that z-axis is vertical.
    vertices = vertices[:, [2, 0, 1]]
    
    vertices_no_quant = vertices.copy()
    faces_no_quant = faces.copy()

    vertices = (vertices + 0.5) * vocab_size  # [0, num_tokens]
    vertices -= 0.5  # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
    vertices_quantized_ = np.clip(vertices.round(), 0, vocab_size - 1).astype(int)  # [0, num_tokens -1]

    cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces)

    cur_mesh.merge_vertices()
    cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
    cur_mesh.update_faces(cur_mesh.unique_faces())
    cur_mesh.remove_unreferenced_vertices()
    
    vertices = cur_mesh.vertices
    sort_inds = np.lexsort(cur_mesh.vertices.T)
    vertices = vertices[sort_inds]
    faces = [np.argsort(sort_inds)[f] for f in cur_mesh.faces]

    if 'AMT' in serialize_type:
        faces = [sorted(sub_arr) for sub_arr in faces] # it will corrupt the face normal
        
    def sort_faces(face):
        return sorted(face)

    faces = sorted(faces, key=sort_faces)

    vertices = vertices / vocab_size - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing

    # vertices = vertices[:, [1, 2, 0]]
    
    return {
        "vertices": vertices,
        "faces": faces,
        "vertices_no_quant": vertices_no_quant,
        "faces_no_quant": faces_no_quant
    }

def load_process_mesh(mesh_obj_path, quantization_bits=7, mc=False, serialize_type='AMT'):
    """Load obj file and process."""

    mesh = trimesh.load(mesh_obj_path, force='mesh', process=False)
    if mc:
        mesh = export_to_watertight(mesh, octree_depth=quantization_bits)
    return process_mesh(mesh.vertices, mesh.faces, quantization_bits,serialize_type=serialize_type)


def normalize_vertices(vertices, scale=0.95):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale

def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
        Convert the non-watertight mesh to watertight.

        Args:
            input_path (str): normlized path
            octree_depth (int):

        Returns:
            mesh(trimesh.Trimesh): watertight mesh

        """
    size = 2 ** octree_depth
    level = 2 / size
    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(normalized_mesh.vertices)

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    vertices = vertices / size * 2 - 1 # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    # vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh

def undiscretize(
    t,
    low=-0.5,#-0.5
    high=0.5,# 0.5
    num_discrete=128,
):
    # t = t.float() #[0, num_discrete-1]
    if isinstance(t, np.ndarray):
        t = t.astype(np.float32)
    t /= num_discrete  # 0<=t<1
    t = t * (high - low) + low # -0.5 <= t < 0.5
    return t

def to_mesh(vertices, faces, transpose=True, post_process=False):
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
        
    if faces.min() == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if post_process:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    return mesh
