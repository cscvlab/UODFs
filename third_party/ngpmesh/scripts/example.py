import numpy as np
import torch
import trimesh
import pyngpmesh

mesh = trimesh.load_mesh('/media/cscvlab/d1/project/lyj/UODF_new/thingi32_normalization/44234.obj', '.obj')
print(mesh)
print(mesh.triangles.shape)

# calculate sdf
X = np.random.rand(10, 3)
rtu = pyngpmesh.NGPMesh(mesh.triangles)
d = rtu.signed_distance(X)
d = np.asarray(d).reshape(10, 1)
print(d)

# tracing
o = np.asarray([
    [0, 0, 4],
    [1, 1, 0],
    [0, 1, 0]
]).astype(np.float32)
d = np.asarray([
    [0, 0, -1],
    [-1, -1, 0],
    [0, -1, 0]
]).astype(np.float32)

hit = rtu.trace(o, d)
hit = np.asarray(hit).reshape(3, 3)
print(hit)

# unsigned distance
ud = rtu.unsigned_distance(X)
ud = np.asarray(ud).reshape(10, 1)
print('unsigned distance')
print(ud)

# nearest point
p = rtu.nearest_point(X)
p = np.asarray(p).reshape(10, 3)
print(p)
dist = np.square(X - p)
dist = np.sum(dist, axis=1)
dist = np.sqrt(dist)
# dist = np.reshape(10, 1)
print('nearest point')
print(dist)