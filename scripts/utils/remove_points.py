import numpy as np
from scipy.spatial import cKDTree as KDTree


def remove_points(pts2,pts1,pts0,res):
    k = 3
    two_thresold = 2 / (res-1) * 1.5
    one_thresold = 2 / (res-1) * 2

    xy = np.concatenate((pts1,pts0))
    xz = np.concatenate((pts2,pts0))
    yz = np.concatenate((pts2,pts1))

    xy_kd_tree = KDTree(xy)
    xz_kd_tree = KDTree(xz)
    yz_kd_tree = KDTree(yz)

    z_kd_tree = KDTree(pts2)
    y_kd_tree = KDTree(pts1)
    x_kd_tree = KDTree(pts0)


    dist_z_two,z_ids = xy_kd_tree.query(pts2)
    dist_y_two,y_ids = xz_kd_tree.query(pts1)
    dist_x_two,x_ids = yz_kd_tree.query(pts0)

    dist_z_one,z_ids = z_kd_tree.query(pts2,k = k)
    dist_y_one,y_ids = y_kd_tree.query(pts1,k = k)
    dist_x_one,x_ids = x_kd_tree.query(pts0,k = k)


    mask2_two = dist_z_two > two_thresold
    mask1_two = dist_y_two > two_thresold
    mask0_two = dist_x_two > two_thresold

    mask2_one = dist_z_one[:,k-1] > one_thresold
    mask1_one = dist_y_one[:,k-1] > one_thresold
    mask0_one = dist_x_one[:,k-1] > one_thresold

    mask2 = ~(mask2_two & mask2_one)
    mask1 = ~(mask1_two & mask1_one)
    mask0 = ~(mask0_two & mask0_one)

    pts = np.concatenate((pts2[mask2],pts1[mask1],pts0[mask0]))
    return pts

