import numpy as np
import trimesh

def cal_boundary_edge(mesh):
    # Most of the time, the boundary edge rate is proportional to the face hole rate.
    edges = mesh.edges
    # Find boundary edges (i.e., edges used by only one face).
    edges_sorted = np.sort(edges, axis=1)
    edge_count = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    boundary_edges = edges[edge_count]
    return boundary_edges.shape[0], edges.shape[0] 