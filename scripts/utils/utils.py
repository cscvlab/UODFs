import os
import torch
import h5py
import numpy as np
import trimesh
import pymeshlab
from scipy.spatial import cKDTree as KDTree
import pickle


def init_path(args):
    path = os.path.abspath(os.path.dirname(__file__))
    path = path[:path.rindex('scripts')]

    if (args.meshPath == None):
        meshPath = path + "datasets/thingi32_normalization/"
        if (not os.path.exists(meshPath)):
            os.mkdir(meshPath)
        args.meshPath = meshPath

    if (args.dataPath == None):
        dataPath = path + "data/"
        if (not os.path.exists(dataPath)):
            os.mkdir(dataPath)
        args.dataPath = dataPath
    if (args.ExpPath == None):
        ExpPath = path + "experiments/"
        if (not os.path.exists(ExpPath)):
            os.mkdir(ExpPath)
        args.ExpPath = ExpPath

    if (args.testPath == None):
        testPath = path + "test/"
        if (not os.path.exists(testPath)):
            os.mkdir(testPath)
        args.testPath = testPath

    if (args.dictPath == None):
        dictPath = path + "test/dict/"
        if (not os.path.exists(dictPath)):
            os.mkdir(dictPath)
        args.dictPath = dictPath
def save_h5(path ,data):
    with h5py.File(path, 'w') as f:
        f["data_old"] = data

def load_h5(filename):
    f = h5py.File(filename, 'r')
    data = np.array(f.get('data_old'))
    return data

def findRow(mat,row):
    return np.where((mat == row).any(1))[0]


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("\033[0;31;40m[Create Directory]\033[0m{}".format(path))
    return path

def save_dict_to_txt(save_dict,save_path):
    with open(save_path, 'w') as f:
        for key, value in save_dict.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')
        f.close()

def degenerate_mesh(pcd_file,possion_file,non_water_file,degreThreshold):
    pts = np.loadtxt(pcd_file)
    model_mesh = trimesh.load(possion_file)

    pcd_kd_tree = KDTree(pts)
    V, F = model_mesh.vertices, model_mesh.faces
    V, F = np.array(V), np.array(F)

    distances, vertex_ids = pcd_kd_tree.query(V)
    distances = np.square(distances)

    V_idx = np.arange(len(V))
    idx = np.where(distances > degreThreshold)[0]
    mask = (distances <= degreThreshold).squeeze()
    V = V[mask]
    V_idx = V_idx[mask]

    F_deg_idx = []
    for id in idx:
        F_deg_idx.extend(findRow(F,id))

    F = np.delete(F,F_deg_idx,axis = 0)

    for i in range(len(V)):
        F[F == V_idx[i]] = i

    print(V.shape)
    print(np.max(F),np.min(F))
    mesh = trimesh.Trimesh(V,F)
    mesh.remove_degenerate_faces()
    mesh.export(non_water_file)


def compute_trimesh_chamfer_mesh(
    gtfile, genfile, num_mesh_samples=100000
):
    """
	This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

	gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see compute_metrics.ply
				for more documentation)

	gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction method
				(see compute_metrics.py for more)

	"""

    try:
        gt_mesh = trimesh.load(gtfile,force='mesh')
        gen_mesh = trimesh.load(genfile,force='mesh')

        gen_points_sampled = trimesh.sample.sample_surface(
            gen_mesh, num_mesh_samples
        )[0]
        gen_points_sampled = gen_points_sampled


        gt_points_np = trimesh.sample.sample_surface(
            gt_mesh, num_mesh_samples
        )[0]

        gen_points_kd_tree = KDTree(gen_points_sampled)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
        gt_to_temp = np.square(one_distances)
        gt_to_gen_chamfer = np.mean(gt_to_temp)

        gt_points_kd_tree = KDTree(gt_points_np)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
        gen_to_gt_temp = np.square(two_distances)
        gen_to_gt_chamfer = np.mean(gen_to_gt_temp)

        if type == "single":
            return gen_to_gt_chamfer

        return gt_to_gen_chamfer + gen_to_gt_chamfer
    except:
        print('error: '+os.path.split(genfile)[1])
        return -1


def compute_trimesh_chamfer_pcd(
        gt_pts, gen_pts,type
):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see compute_metrics.ply
                for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction method
                (see compute_metrics.py for more)

    """

    gen_points_sampled = gen_pts
    gt_points_np = gt_pts

    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_temp = np.square(one_distances)
    gt_to_gen_chamfer = np.mean(gt_to_temp)

    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_temp = np.square(two_distances)
    gen_to_gt_chamfer = np.mean(gen_to_gt_temp)

    if type == "single":
        return gen_to_gt_chamfer

    # return gt_to_gen_chamfer
    return gt_to_gen_chamfer + gen_to_gt_chamfer

def load_obj_pcd_and_normal(filename):
    vertices = []
    vertex_norm = []
    for line in open(filename, "r",encoding="utf-8"):
        values = line.split()
        if (values == []):
            continue
        if (values == '#'):
            continue
        if (values[0] == 'v'):
            vertices.append([float(values[1]), float(values[2]), float(values[3])])
        if (values[0] == 'vn'):
            vertex_norm.append([float(values[1]), float(values[2]), float(values[3])])
    return np.array(vertices),np.array(vertex_norm)

def meshlab_possion(pcd_file,possion_file,mlx_file):
    ms = pymeshlab.MeshSet()
    ms.load_filter_script(mlx_file)
    ms.load_new_mesh(pcd_file)
    ms.apply_filter_script()
    ms.save_current_mesh(possion_file)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))

def save_checkpoint(epoch,train_accuracy,test_accuracy,model,optimizer,path,modelnet = 'checkpoint'):
    savepath = path + '/%s-%f-%04d.pth' %(modelnet,test_accuracy,epoch)
    state = {
        'epoch':epoch,
        'train_accuracy':train_accuracy,
        'test_accuracy':test_accuracy,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }
    torch.save(state,savepath)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setparam(args, param, paramstr):
    argsparam = getattr(args, paramstr, None)
    if param is not None or argsparam is None:
        return param
    else:
        return argsparam



