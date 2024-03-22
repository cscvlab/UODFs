import numpy as np
import torch
import math
import pyngpmesh
import trimesh


class DirectionalRaySample():
    def __init__(self,ray_init,dir,filename):
        self.mesh = trimesh.load_mesh(filename)
        self.renderer = pyngpmesh.NGPMesh(self.mesh.triangles)
        self.ray_init = ray_init
        self.dir = dir
    def ray_intersect(self, pts, ray_d):
        x = np.array(self.renderer.trace(pts, ray_d))
        t = self.distance(pts, x)
        return x, t
    def distance(self,x1,x2):
        d = np.sqrt((x1[:, 0] - x2[:, 0]) * (x1[:, 0] - x2[:, 0])
                    + (x1[:, 1] - x2[:, 1]) * (x1[:, 1] - x2[:, 1]) +
                    (x1[:, 2] - x2[:, 2]) * (x1[:, 2] - x2[:, 2]))
        return d.reshape(-1, 1)

    def epoch_sample(self,sample_num,seed):
        '''compute pts and gt'''
        np.random.seed(seed)
        variance = np.random.rand(self.ray_init.shape[0],) * (2/(sample_num-1))
        pts = self.ray_init.copy()
        pts[:,self.dir] += variance


        ray_pos = np.zeros_like(pts)
        ray_pos[:,self.dir] = 1
        ray_neg = np.zeros_like(pts)
        ray_neg[:,self.dir] = -1

        hit_pos, d_pos = self.ray_intersect(pts, ray_pos)
        hit_neg, d_neg = self.ray_intersect(pts, ray_neg)

        gt = np.zeros_like(d_pos)
        mask = (d_pos < d_neg).squeeze()
        gt[mask] = d_pos[mask]
        gt[~mask] = d_neg[~mask]

        return pts,ray_pos,gt

class MaskRaySample():
    def __init__(self,res,dir,filename):
        self.res = res
        self.dir = dir
        self.mesh = trimesh.load_mesh(filename)
        self.renderer = pyngpmesh.NGPMesh(self.mesh.triangles)

    def ray_intersect(self, pts, ray_d):
        x = np.array(self.renderer.trace(pts, ray_d))
        t = self.distance(pts, x)
        return x, t

    def distance(self,x1,x2):
        d = np.sqrt((x1[:, 0] - x2[:, 0]) * (x1[:, 0] - x2[:, 0])
                    + (x1[:, 1] - x2[:, 1]) * (x1[:, 1] - x2[:, 1]) +
                    (x1[:, 2] - x2[:, 2]) * (x1[:, 2] - x2[:, 2]))
        return d.reshape(-1, 1)

    def sampler(self):
        x = np.linspace(-1, 1, self.res)
        if self.dir == 2:
            pts = np.array([(i, j, -1) for i in x for j in x])
        elif self.dir == 1:
            pts = np.array([(i, -1, j) for i in x for j in x])
        elif self.dir == 0:
            pts = np.array([(-1, i, j) for i in x for j in x])

        print("Mask pts shape: ",pts.shape)
        '''compute hit_mask'''
        ray_pos = np.zeros_like(pts)
        ray_pos[:, self.dir] = 1
        ray_neg = np.zeros_like(pts)
        ray_neg[:, self.dir] = -1

        hit_pos, d_pos = self.ray_intersect(pts, ray_pos)
        hit_neg, d_neg = self.ray_intersect(pts, ray_neg)

        t = np.zeros_like(d_pos)
        mask = (d_pos < d_neg).squeeze()
        t[mask] = d_pos[mask]
        t[~mask] = d_neg[~mask]
        hit_mask = (abs(t) < 3).squeeze()

        hit_mask = hit_mask.astype(np.int32)

        pts = torch.tensor(pts,dtype = torch.float)
        hit_mask = torch.tensor(hit_mask,dtype = torch.float)
        return pts, hit_mask



class BaseDataset:
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = 0

        self.X = torch.zeros(0, 3)
        self.Y = torch.zeros(0, 1)

    def _shuffle(self):
        if self.X.shape[0] <= 0:
            return
        if self.shuffle:
            p = np.random.permutation(self.X.shape[0])
            p = torch.LongTensor(p)
            self.X = self.X[p]
            self.Y = self.Y[p]

    def __len__(self):
        return int(math.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        i = index * self.batch_size
        X = self.X[i: i + self.batch_size] if i + self.batch_size < len(self.X) else self.X[i:]
        Y = self.Y[i: i + self.batch_size] if i + self.batch_size < len(self.Y) else self.Y[i:]

        return X, Y

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.X.shape[0]:
            i = self.count
            X = self.X[i: i + self.batch_size] if i + self.batch_size < len(self.X) else self.X[i:]
            Y = self.Y[i: i + self.batch_size] if i + self.batch_size < len(self.Y) else self.Y[i:]
            self.count += self.batch_size
            return X, Y
        raise StopIteration()

class UodfDataSet(BaseDataset):
    def __init__(self,sampler,seed,batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)
        self.sampler = sampler
        self.seed = seed
    def _sample_base(self,sample_num):
        pts,ray_d,gt = self.sampler.epoch_sample(sample_num,self.seed)
        sign = np.zeros((pts.shape[0], 1))
        self.seed +=1

        self.X = np.concatenate((pts,ray_d,sign),axis = 1)
        self.X = torch.Tensor(self.X)
        self.Y = gt
        self.Y = torch.Tensor(self.Y)

        # self.pts_perturb = pts

        self._shuffle()

class UodfMaskDataSet(BaseDataset):
    def __init__(self,X,Y,batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)
        self.X = X
        self.Y = Y
        self._shuffle()

class UodfTestDataSet(BaseDataset):
    def __init__(self,X,Y,batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)
        self.X = X
        self.Y = Y

        self.X = torch.Tensor(self.X)
        self.Y = torch.Tensor(self.Y)




if __name__ == "__main__":
    rays_init = np.random.random((1000000,3))
    dir_dim = 2
    filename = "/media/cscvlab/d1/project/lyj/UODF/thingi32_normalization/armadillo.obj"
    sampler = DirectionalRaySample(rays_init,dir_dim,filename)

    dataSet = UodfDataSet(sampler,seed = 0)
    dataSet._sample_base(128)

    for data in dataSet:
        X, y = data
        print(X.shape)






