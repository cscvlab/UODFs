import scipy
import numpy as np
import trimesh


OCCNET_FSCORE_EPS = 1e-09
def dot_product(a, b):
    if len(a.shape) != 2:
        raise ValueError('Dot Product with input shape: %s' % repr(a.shape))
        raise ValueError('Dot Product with input shape: %s' % repr(b.shape))
    return np.sum(a * b, axis=1)

def normalize_npy(
    V ,F,scale):

    # Find the max distance to origin
    max_dist = np.sqrt(np.max(np.sum(V**2, axis=-1)))
    V_scale = scale / max_dist
    V *= V_scale
    return V, F

def sample_points_and_face_normals(mesh, sample_count):
    points, indices = mesh.sample(sample_count, return_index=True)
    points = points.astype(np.float32)
    normals = mesh.face_normals[indices]
    return points, normals

def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0
def f_score(a_to_b, b_to_a, thresh):
    precision = percent_below(a_to_b, thresh)
    recall = percent_below(b_to_a, thresh)

    return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)
def pointcloud_neighbor_distances_indices(source_points, target_points):
    target_kdtree = scipy.spatial.cKDTree(target_points)
    distances, indices = target_kdtree.query(source_points)
    return distances, indices

def mesh_metrics(pred_mesh,gt_mesh):
    """Computes the chamfer distance and normal consistency metrics."""
    element = dict()

    sample_count = 100000
    points_pred, normals_pred = sample_points_and_face_normals(pred_mesh, sample_count)
    points_gt, normals_gt = sample_points_and_face_normals(
      gt_mesh, sample_count)
    

    pred_to_gt_dist, pred_to_gt_indices = pointcloud_neighbor_distances_indices(
      points_pred, points_gt)
    gt_to_pred_dist, gt_to_pred_indices = pointcloud_neighbor_distances_indices(
      points_gt, points_pred)

    pred_to_gt_normals = normals_gt[pred_to_gt_indices]
    gt_to_pred_normals = normals_pred[gt_to_pred_indices]

    # We take abs because the OccNet code takes abs
    pred_to_gt_normal_consistency = np.abs(
      dot_product(normals_pred, pred_to_gt_normals))
    gt_to_pred_normal_consistency = np.abs(
      dot_product(normals_gt, gt_to_pred_normals))

    # The 100 factor is because papers multiply by 100 for display purposes.
    chamfer = 1000.0 * (np.mean(pred_to_gt_dist**2) + np.mean(gt_to_pred_dist**2))

    nc = 0.5 * np.mean(pred_to_gt_normal_consistency) + 0.5 * np.mean(
      gt_to_pred_normal_consistency)

    tau = 0.005
    f_score_tau = f_score(pred_to_gt_dist, gt_to_pred_dist, tau)
    f_score_2tau = f_score(pred_to_gt_dist, gt_to_pred_dist, 2.0 * tau)

    element['chamfer'] = chamfer
    element['normal_consistency'] = nc
    element['f_score_tau'] = f_score_tau
    element['f_score_2tau'] = f_score_2tau

    return element


def scale_mesh(mesh,save_path,scale):
    V,F = mesh.vertices,mesh.faces
    V,F = normalize_npy(V,F,scale)

    new_mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    export = trimesh.exchange.obj.export_obj(new_mesh, include_normals=False)
    with open(save_path, "w") as f:
        f.write(export)

