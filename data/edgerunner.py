import numpy as np
import math
from collections import defaultdict
import sys
sys.setrecursionlimit(1000000)


class Vertex:
    def __init__(self, x=0, y=0, z=0, i=-1, m=0, discrete_bins=None):
        # assert -0.5 <= x < 0.5 and -0.5 <= y < 0.5 and -0.5 <= z < 0.5
        if discrete_bins is not None:
            self.x = int((x+0.5)*discrete_bins)
            self.y = int((y+0.5)*discrete_bins)
            self.z = int((z+0.5)*discrete_bins)
        else:
            self.x = x
            self.y = y
            self.z = z
        self.i = i
        self.m = m

    def undiscrete(self, discrete_bins):
        return [
            self.x / discrete_bins + 0.5,
            self.y / discrete_bins + 0.5,
            self.z / discrete_bins + 0.5
        ]

    def __add__(self, v):
        return Vertex(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vertex(self.x - v.x, self.y - v.y, self.z - v.z)

    def __eq__(self, v):
        return self.x == v.x and self.y == v.y and self.z == v.z

    def __lt__(self, v):
        return self.y < v.y or (self.y == v.y and self.z < v.z) or (self.y == v.y and self.z == v.z and self.x < v.x)


class Vector3f:
    def __init__(self, x=0, y=0, z=0, v1=None, v2=None):
        if v1 is not None and v2 is not None:
            self.x = v2.x - v1.x
            self.y = v2.y - v1.y
            self.z = v2.z - v1.z
        elif v1 is not None:
            self.x = v1.x
            self.y = v1.y
            self.z = v1.z
        else:
            self.x = x
            self.y = y
            self.z = z

    def __add__(self, v):
        return Vector3f(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vector3f(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, s):
        return Vector3f(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s):
        return Vector3f(self.x / s, self.y / s, self.z / s)

    def __eq__(self, v):
        return self.x == v.x and self.y == v.y and self.z == v.z

    def __lt__(self, v):
        return self.y < v.y or (self.y == v.y and self.z < v.z) or (self.y == v.y and self.z == v.z and self.x < v.x)

    def cross(self, v):
        return Vector3f(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        n = self.norm()
        return Vector3f(self.x / n, self.y / n, self.z / n)


class HalfEdge:
    def __init__(self, v=None, s=None, e=None, t=None, n=None, p=None, o=None, i=-1):
        self.v = v
        self.s = s
        self.e = e
        self.t = t
        self.n = n
        self.p = p
        self.o = o
        self.i = i

    def __lt__(self, e):
        if self.o is None:
            return True
        elif e.o is None:
            return False
        else:
            return Vector3f(self.v, self.o.v).norm() < Vector3f(e.v, e.o.v).norm()


class Facet:
    def __init__(self, vertices=None, half_edges=None, i=-1, ic=-1, m=0, center=None):
        if vertices is None:
            vertices = [None, None, None]
        if half_edges is None:
            half_edges = [None, None, None]
        if center is None:
            center = Vector3f()
        self.vertices = vertices
        self.half_edges = half_edges
        self.i = i
        self.ic = ic
        self.m = m
        self.center = center

    def flip(self):
        for i in range(3):
            self.half_edges[i].s, self.half_edges[i].e = self.half_edges[i].e, self.half_edges[i].s
            self.half_edges[i].n, self.half_edges[i].p = self.half_edges[i].p, self.half_edges[i].n

    def __lt__(self, f):
        if self.ic != f.ic:
            return self.ic < f.ic
        else:
            return self.center < f.center


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


class Mesh:
    def __init__(self, vertices, triangles, discrete_bins=256, verbose=False):
        self.discrete_bins = discrete_bins
        self.verbose = verbose

        # Discretize vertices
        self.verts = [Vertex(v[0], v[1], v[2], i, discrete_bins=discrete_bins) for i, v in enumerate(vertices)]
        self.num_vertices = len(self.verts)

        # Build faces and half-edges
        self.faces = []
        edge2halfedge = {}
        for i, triangle in enumerate(triangles):
            f = Facet()
            f.i = i
            half_edges = []
            for j in range(3):
                e = HalfEdge()
                e.v = self.verts[triangle[j]]
                e.s = self.verts[triangle[(j + 1) % 3]]
                e.e = self.verts[triangle[(j + 2) % 3]]
                e.t = f
                e.i = j
                half_edges.append(e)
                key = edge_key(triangle[(j + 1) % 3], triangle[(j + 2) % 3])
                if key in edge2halfedge:
                    e.o = edge2halfedge[key]
                    edge2halfedge[key].o = e
                    del edge2halfedge[key]
                else:
                    edge2halfedge[key] = e
            f.half_edges = half_edges
            f.vertices = [self.verts[v] for v in triangle]
            for j in range(3):
                f.half_edges[j].n = f.half_edges[(j + 1) % 3]
                f.half_edges[j].p = f.half_edges[(j + 2) % 3]
            f.center = Vector3f(
                sum(v.x for v in f.vertices) / 3,
                sum(v.y for v in f.vertices) / 3,
                sum(v.z for v in f.vertices) / 3
            )
            self.faces.append(f)

        self.num_faces = len(self.faces)
        self.num_edges = len(edge2halfedge)

        # Mark boundary vertices
        for f in self.faces:
            for e in f.half_edges:
                if e.o is None:
                    e.s.m = 1
                    e.e.m = 1

        # Sort faces
        self.faces.sort()

        # Find connected components
        self.num_components = 0
        for f in self.faces:
            if f.ic == -1:
                self.num_components += 1
                queue = [f]
                while queue:
                    current = queue.pop(0)
                    if current.ic != -1:
                        continue
                    current.ic = self.num_components
                    for e in current.half_edges:
                        if e.o and e.o.t.ic == -1:
                            queue.append(e.o.t)


class Engine_LR_ABSCO:
    class OP:
        OP_L = 0  # left face visited, move to right
        OP_R = 1  # right face visited, move to left
        OP_BOM = 2  # begin of a submesh
        OP_NUM = 3  # total number of OPs

    def __init__(self, discrete_bins=256, verbose=False):
        self.discrete_bins = discrete_bins
        self.verbose = verbose
        self.mesh = None
        self.tokens = []
        self.face_order = []
        self.face_type = []
        self.num_submesh = 0
        self.num_faces = 0

    def offset_coord(self, x):
        return x + self.OP.OP_NUM

    def restore_coord(self, x):
        return x - self.OP.OP_NUM

    def compress_face(self, c, init=False):
        c.t.m = 1
        self.face_order.append(c.t.i)
        self.num_faces += 1

        if not init:
            if not (c.s.i == c.o.e.i and c.e.i == c.o.s.i):
                c.t.flip()
            self.tokens.extend([self.offset_coord(c.v.x), self.offset_coord(c.v.y), self.offset_coord(c.v.z)])

        tip_visited = c.v.m
        left_visited = c.p.o is None or c.p.o.t.m
        right_visited = c.n.o is None or c.n.o.t.m

        if not tip_visited:
            c.v.m = 1
            self.tokens.append(self.OP.OP_L)
            self.face_type.append(self.OP.OP_L)
            self.compress_face(c.n.o)
        elif left_visited and right_visited:
            self.face_type.append(self.OP.OP_BOM)
            return
        elif left_visited:
            self.tokens.append(self.OP.OP_L)
            self.face_type.append(self.OP.OP_L)
            self.compress_face(c.n.o)
        elif right_visited:
            self.tokens.append(self.OP.OP_R)
            self.face_type.append(self.OP.OP_R)
            self.compress_face(c.p.o)
        else:
            len_left = 0
            len_right = 0
            cur = c.n.o
            while True:
                len_left += 1
                cur = cur.n
                while cur.o is not None and not cur.o.t.m:
                    cur = cur.o.n
                if cur == c.n.o:
                    break
            cur = c.p.o
            while True:
                len_right += 1
                cur = cur.p
                while cur.o is not None and not cur.o.t.m:
                    cur = cur.o.p
                if cur == c.p.o:
                    break

            if len_left < len_right:
                self.tokens.append(self.OP.OP_L)
                self.face_type.append(self.OP.OP_L)
                self.compress_face(c.n.o)
                self.compress_submesh(c.p.o)
            else:
                self.tokens.append(self.OP.OP_R)
                self.face_type.append(self.OP.OP_R)
                self.compress_face(c.p.o)
                self.compress_submesh(c.n.o)

    def compress_submesh(self, c):
        if c.t.m:
            return

        self.tokens.append(self.OP.OP_BOM)
        self.num_faces = 0

        self.tokens.extend([self.offset_coord(c.v.x), self.offset_coord(c.v.y), self.offset_coord(c.v.z),
                            self.offset_coord(c.s.x), self.offset_coord(c.s.y), self.offset_coord(c.s.z),
                            self.offset_coord(c.e.x), self.offset_coord(c.e.y), self.offset_coord(c.e.z)])

        c.s.m = 1
        c.e.m = 1

        self.compress_face(c, True)

    def encode(self, vertices, triangles):
        self.mesh = Mesh(vertices, triangles, self.discrete_bins, self.verbose)
        self.tokens = []
        self.face_order = []
        self.face_type = []
        self.num_submesh = 0

        for f in self.mesh.faces:
            if not f.m:
                self.compress_submesh(f.half_edges[0])

        return self.tokens, self.face_order, self.face_type

    def decode(self, tokens):
        vertices = []
        faces = []
        self.face_type = []

        v0, v1, v2, v = None, None, None, None
        num_vertices = 0
        num_faces = 0
        num_submesh = 0

        i = 0
        while i < len(tokens):
            if tokens[i] == self.OP.OP_BOM:
                if i + 9 >= len(tokens):
                    if self.verbose:
                        print(f"[DECODE] ERROR: incomplete face at {i}")
                    break
                    # Submesh start
                if self.verbose:
                    print(f"[DECODE] Submesh start: {num_submesh}")
                    num_submesh += 1
                # Read 3 consecutive vertices
                v0 = Vertex(self.restore_coord(tokens[i+1]), 
                            self.restore_coord(tokens[i+2]), 
                            self.restore_coord(tokens[i+3]), 
                            num_vertices)
                v1 = Vertex(self.restore_coord(tokens[i+4]), 
                            self.restore_coord(tokens[i+5]), 
                            self.restore_coord(tokens[i+6]), 
                            num_vertices + 1)
                v2 = Vertex(self.restore_coord(tokens[i+7]), 
                            self.restore_coord(tokens[i+8]), 
                            self.restore_coord(tokens[i+9]), 
                            num_vertices + 2)
                # Add vertices
                vertices.extend([v0.undiscrete(self.discrete_bins),
                                v1.undiscrete(self.discrete_bins),
                                v2.undiscrete(self.discrete_bins)])
                # Add the first triangle
                faces.append([v0.i, v1.i, v2.i])
                if i != 0:
                    self.face_type.append(self.OP.OP_BOM)
                if self.verbose:
                    print(f"[DECODE] Add Init face: {num_faces} = [{v0.i}, {v1.i}, {v2.i}]")
                    num_faces += 1
                # Move index
                i += 10  # Skip OP_BOM and the next 9 tokens
                num_vertices += 3  # Increment after adding vertices
            else:
                # tokens[i] should be an OP
                if tokens[i] >= self.OP.OP_NUM:
                    if self.verbose:
                        print(f"[DECODE] ERROR: position should be OP at {i}")
                    break
                # Read the new vertex
                if i + 3 >= len(tokens):
                    if self.verbose:
                        print(f"[DECODE] ERROR: incomplete vertex at {i}")
                    break
                v = Vertex(self.restore_coord(tokens[i+1]), 
                        self.restore_coord(tokens[i+2]), 
                        self.restore_coord(tokens[i+3]))
                if tokens[i] == self.OP.OP_L:
                    # Move to right
                    v.i = num_vertices
                    num_vertices += 1
                    vertices.append(v.undiscrete(self.discrete_bins))
                    faces.append([v.i, v0.i, v2.i])
                    if self.verbose:
                        print(f"[DECODE] Add R face: {num_faces} = [{v.i}, {v0.i}, {v2.i}]")
                        num_faces += 1
                    # Update v0, v1, v2
                    v1 = v0
                    v0 = v
                elif tokens[i] == self.OP.OP_R:
                    # Move to left
                    v.i = num_vertices
                    num_vertices += 1
                    vertices.append(v.undiscrete(self.discrete_bins))
                    faces.append([v.i, v1.i, v0.i])
                    if self.verbose:
                        print(f"[DECODE] Add L face: {num_faces} = [{v.i}, {v1.i}, {v0.i}]")
                        num_faces += 1
                    # Update v0, v1, v2
                    v2 = v0
                    v0 = v
                self.face_type.append(tokens[i])
                i += 4  # Skip OP and the next 3 tokens

        self.face_type.append(self.OP.OP_BOM)  # Last face
        return np.array(vertices), np.array(faces), np.array(self.face_type)
class Engine:
    def __init__(self, discrete_bins, verbose=False, backend='LR_ABSCO'):
        self.discrete_bins = discrete_bins
        self.verbose = verbose
        self.impl = Engine_LR_ABSCO(discrete_bins, verbose)

    def encode(self, vertices, triangles):
        tokens, face_order, face_type = self.impl.encode(vertices, triangles)
        return np.array(tokens), np.array(face_order), np.array(face_type)

    def decode(self, tokens):
        vertices, faces, face_type = self.impl.decode(tokens)
        return np.array(vertices), np.array(faces), np.array(face_type)                    