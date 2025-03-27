import trimesh
import point_cloud_utils as pcu

def apply_normalize(mesh):
    """
    Normalize the given mesh by translating it to the origin and scaling it.

    Args:
        mesh (trimesh.Trimesh): The input mesh to be normalized.

    Returns:
        trimesh.Trimesh: The normalized mesh.
    """
    # Compute the bounding box of the mesh
    bbox = mesh.bounds
    # Calculate the center of the bounding box
    center = (bbox[1] + bbox[0]) / 2
    # Determine the maximum side length of the bounding box
    scale = (bbox[1] - bbox[0]).max()

    # Translate the mesh to the origin
    mesh.apply_translation(-center)
    # Scale the mesh to fit within a certain range
    mesh.apply_scale(1 / scale * 2 * 0.95)
    return mesh

def sample_pc(mesh, pc_num):
    """
    Sample a point cloud from the given mesh after normalization.

    Args:
        mesh (trimesh.Trimesh): The input mesh to sample from.
        pc_num (int): The number of points to sample.

    Returns:
        numpy.ndarray: The sampled point cloud.
    """
    # Normalize the mesh
    mesh = apply_normalize(mesh)
    # Sample points from the mesh
    points, _ = mesh.sample(pc_num, return_index=True)
    return points

def cal_pc_similarity(new_obj_path, ori_obj, pc_num=16384):
    """
    Calculate the Hausdorff and Chamfer distances between point clouds sampled from two meshes.

    Args:
        new_obj_path (str): The path to the new mesh file.
        ori_obj (str): The path to the original mesh file.
        pc_num (int, optional): The number of points to sample from each mesh. Defaults to 16384.

    Returns:
        tuple: A tuple containing the Hausdorff distance and the Chamfer distance.
    """
    # Load the new mesh from the file
    sample_mesh = trimesh.load(new_obj_path, force='mesh', process=False)
    # Load the original mesh from the file
    ori_obj = trimesh.load(ori_obj, force='mesh', process=False)

    # Sample point clouds from the meshes
    sample, ref = sample_pc(sample_mesh, pc_num), sample_pc(ori_obj, pc_num)
    # Calculate the Hausdorff distance between the point clouds
    hausdorff_dist = pcu.hausdorff_distance(sample, ref)
    # Calculate the Chamfer distance between the point clouds
    chamfer_dist = pcu.chamfer_distance(sample, ref)
    return hausdorff_dist, chamfer_dist