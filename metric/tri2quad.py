import pymeshlab
# pip install pymeshlab==0.2.1

def tri_to_quad(obj_path, save_path, level=1):
    """
    Convert triangle mesh to quad-dominant mesh
    
    Args:
        obj_path (str): Path to input OBJ file
        save_path (str): Path to save output mesh
        level (int): Quad conversion level (1-5)
    
    Returns:
        None: Output mesh saved to disk
    """
    ms = pymeshlab.MeshSet()
    
    try:
        # Load original mesh
        ms.load_new_mesh(obj_path)

        # Preprocessing
        ms.apply_filter('repair_non_manifold_edges_by_removing_faces')
        ms.apply_filter('remove_duplicate_faces')
        
        # Quad conversion
        ms.apply_filter('turn_into_quad_dominant_mesh', level=level)
        
        # Postprocessing and save
        ms.save_current_mesh(save_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise