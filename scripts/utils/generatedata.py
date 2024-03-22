import torch
import pyngpmesh
import trimesh
import numpy as np
from utils.utils import save_h5

class GenerateData():
    def __init__(self,args,filename):
        self.args = args
        self.filename = filename
        self.mesh = trimesh.load_mesh(self.args.meshPath + self.filename)
        self.renderer = pyngpmesh.NGPMesh(self.mesh.triangles)
        print("Init Over....")

    def distance(self,x1,x2):
        d = np.sqrt((x1[:, 0] - x2[:, 0]) * (x1[:, 0] - x2[:, 0])
                    + (x1[:, 1] - x2[:, 1]) * (x1[:, 1] - x2[:, 1]) +
                    (x1[:, 2] - x2[:, 2]) * (x1[:, 2] - x2[:, 2]))
        return d.reshape(-1, 1)
    def ray_intersect(self, pts, ray_d):
        x = np.array(self.renderer.trace(pts, ray_d))
        t = self.distance(pts, x)
        return x, t
    def generateData(self,res,dir,sample_num):
        x = np.linspace(-1, 1, res)
        if dir == 2:
            pts = np.array([(i,j,-1) for i in x for j in x])
        elif dir == 1:
            pts = np.array([(i,-1,j) for i in x for j in x])
        elif dir == 0:
            pts = np.array([(-1,i,j) for i in x for j in x])

        ray_pos = np.zeros_like(pts)
        ray_neg = np.zeros_like(pts)
        ray_pos[:, dir] = 1
        ray_neg[:, dir] = -1

        hit_pos, d_pos = self.ray_intersect(pts, ray_pos)
        hit_neg, d_neg = self.ray_intersect(pts, ray_neg)

        t = np.zeros_like(d_pos)
        mask = (d_pos < d_neg).squeeze()
        t[mask] = d_pos[mask]
        t[~mask] = d_neg[~mask]
        hit_mask = (abs(t) < 3).squeeze()

        pts = pts[hit_mask]

        # compute gt
        if pts.shape[0] == 0:
            print("zeros")
            return
        ray_pos = np.zeros((pts.shape[0], sample_num, 3))
        ray_pos[:, :, dir] = 1
        ray_neg = np.zeros((pts.shape[0], sample_num, 3))
        ray_neg[:, :, dir] = -1

        t = np.linspace(0, 2, num=sample_num).reshape(1, -1)
        t = np.repeat(t, pts.shape[0], axis=0)

        pts = pts[:, None, :] + t[:, :, None] * ray_pos


        '''compute gt'''
        pts = pts.reshape(-1,3)
        ray_pos = ray_pos.reshape(-1,3)
        ray_neg = ray_neg.reshape(-1,3)

        hit_pos, d_pos = self.ray_intersect(pts, ray_pos)
        hit_neg, d_neg = self.ray_intersect(pts, ray_neg)

        t = np.zeros_like(d_pos)
        mask = (d_pos < d_neg).squeeze()
        t[mask] = d_pos[mask]
        t[~mask] = d_neg[~mask]
        hit_mask = (abs(t) < 3).squeeze()

        print("pts shape : ",pts.shape)
        print("hit pts shape : ",pts[hit_mask].shape)

        # save
        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/pts_"+ str(dir) + ".h5", pts)
        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/gt_" + str(dir) + ".h5", pts)
        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/hit_mask_"+ str(dir) + ".h5", hit_mask)


    def AxisSize_generator(self,res,dir):
        '''init'''
        x = np.linspace(-1, 1, res)
        pts = np.array([(i, j, k) for i in x for j in x for k in x])

        '''pts'''
        ray_pos = np.zeros_like(pts)
        ray_pos[:,dir] = 1
        ray_neg = np.zeros_like(pts)
        ray_neg[:,dir] = -1

        print(pts.shape)
        mesh = trimesh.load_mesh(self.args.meshPath + self.filename)
        renderer = pyngpmesh.NGPMesh(self.mesh.triangles)

        hit_pos = np.array(renderer.trace(pts, ray_pos))
        d_pos = self.distance(pts, hit_pos)
        hit_neg = np.array(renderer.trace(pts, ray_neg))
        d_neg = self.distance(pts, hit_neg)

        t = np.zeros_like(d_pos)
        hit_pts = np.zeros_like(pts)
        mask =  (d_pos < d_neg).squeeze()

        t[mask] = d_pos[mask]
        t[~mask] = d_neg[~mask]

        hit_pts[mask] = hit_pos[mask]
        hit_pts[~mask] = hit_neg[~mask]
        hit_mask = abs(t) < 3

        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/{}/AxisSize_{}_gt_{}.h5".format(res,res,dir), t)
        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/{}/AxisSize_{}_hit_mask_{}.h5".format(res,res,dir), hit_mask)
        save_h5(self.args.dataPath + str(self.filename.split(".")[0]) +
                "/{}/AxisSize_{}_hit_pts_{}.h5".format(res,res,dir), hit_pts)



