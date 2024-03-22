import os
import numpy as np
from utils.utils import load_h5,save_h5
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import copy


class PcdPostProcess():
    def __init__(self,args):
        self.args = args
    def init(self,dir,filename,res):
        dataPath = self.args.dataPath

        x = np.linspace(-1, 1, res)
        pts = np.array([(i,j,k) for i in x for j in x for k in x])

        ray_d = np.zeros_like(pts)
        ray_d[:, dir] = 1

        pred = load_h5(dataPath + filename + "/{}/pred_{}_{}.h5".format(res,res,dir))
        pred_sign = load_h5(dataPath + filename + "/{}/pred_sign_{}_{}.h5".format(res,res,dir))
        pred_hit_mask = load_h5(dataPath + filename.split(".")[0] + "/{}/pred_hit_mask_{}_{}.h5".format(res,res,dir))
        pred_hit_mask = pred_hit_mask.squeeze()

        if dir == 1:
            pts = pts.reshape(res,res,res,3).transpose(0,2,1,3)
            ray_d = ray_d.reshape(res,res,res,3).transpose(0,2,1,3)
            pred = pred.reshape(-1,res,res).transpose(0,2,1)
            pred_sign = pred_sign.reshape(-1,res,res).transpose(0,2,1)
            pred_hit_mask = pred_hit_mask.reshape(res,res,res).transpose(0,2,1)

        elif dir == 0:
            pts = pts.reshape(res,res,res,3).transpose(1,2,0,3)
            ray_d = ray_d.reshape(res,res,res,3).transpose(1,2,0,3)
            pred = pred.reshape(-1,res,res).transpose(1,2,0)
            pred_sign = pred_sign.reshape(-1,res,res).transpose(1,2,0)
            pred_hit_mask = pred_hit_mask.reshape(res,res,res).transpose(1,2,0)

        pts = pts[pred_hit_mask]
        ray_d = ray_d[pred_hit_mask]
        pred = pred[pred_hit_mask]
        pred_sign = pred_sign[pred_hit_mask]

        pts = pts.reshape(-1, res, 3)
        ray_d = ray_d.reshape(-1, res, 3)
        pred = pred.reshape(-1, res, 1)
        pred_sign = pred_sign.reshape(-1, res, 1)

        pred_hit_mask = pred_hit_mask.reshape(-1, res)
        pred_hit_mask = np.sum(pred_hit_mask,axis = 1)
        pred_hit_mask = (pred_hit_mask > 0).squeeze()

        rays_id = np.arange((res * res))
        rays_id = rays_id[pred_hit_mask]

        return pts,ray_d,pred,pred_sign,rays_id,pred_hit_mask

    def init_gt(self,dir,filename,res):
        dataPath = self.args.dataPath

        x = np.linspace(-1, 1, res)
        pts = np.array([(i,j,k) for i in x for j in x for k in x])

        gt_path = dataPath + filename + "/{}/AxisSize_{}_gt_{}.h5".format(res,res,dir)
        hit_pts_path = dataPath + filename + "/{}/AxisSize_{}_hit_pts_{}.h5".format(res,res,dir)
        hit_mask_path = dataPath + filename + "/{}/AxisSize_{}_hit_mask_{}.h5".format(res,res,dir)

        # if(os.path.exists(gt_path) and os.path.exists(hit_pts_path)):
        gt = load_h5(gt_path)
        hit_pts = load_h5(hit_pts_path)
        hit_mask = load_h5(hit_mask_path).squeeze()

        if dir == 1:
            pts = pts.reshape(res,res,res,3).transpose(0,2,1,3)
            gt = gt.reshape(res,res,res).transpose(0,2,1)
            hit_pts = hit_pts.reshape(res, res, res, 3).transpose(0, 2, 1, 3)
            hit_mask = hit_mask.reshape(res,res,res).transpose(0,2,1)
        elif dir == 0:
            pts = pts.reshape(res,res,res,3).transpose(1,2,0,3)
            gt = gt.reshape(res,res,res).transpose(1,2,0)
            hit_pts = hit_pts.reshape(res, res, res, 3).transpose(1, 2, 0, 3)
            hit_mask = hit_mask.reshape(res,res,res).transpose(1,2,0)


        pts = pts[hit_mask].reshape(-1, res, 3)
        gt = gt[hit_mask].reshape(-1, res, 1)
        hit_pts = hit_pts[hit_mask].reshape(-1, res, 3)

        hit_mask = hit_mask.reshape(-1, res)
        hit_mask = np.sum(hit_mask,axis = 1)
        hit_mask = (hit_mask > 0).squeeze()
        return pts,gt,hit_pts,hit_mask

    def _pcdpostprocess(self,res,pts,pred,pred_sign,ray_d,dir,threshold):
        num_pos = np.ones((pts.shape[0], res))
        num_neg = np.ones((pts.shape[0], res))
        hit_pts = [[] for i in range(pts.shape[0])]

        '''Positive Direction'''
        ori_sign = pred_sign[:, 0].reshape(-1, 1)
        ori_hit_p = pts[:, 0] + pred[:, 0] * pred_sign[:, 0] * ray_d[:, 0]
        for i in range(1, res):
            mask = (ori_sign == pred_sign[:, i].reshape(-1, 1)).squeeze()
            temp_hit_p = pts[:, i] + pred[:, i] * pred_sign[:, i] * ray_d[:, i]
            num_pos[mask, i] = num_pos[mask, i - 1] + 1

            ori_hit_p[mask] += temp_hit_p[mask]
            ori_hit_p[~mask] = ori_hit_p[~mask] / num_pos[~mask, i - 1].reshape(-1, 1)

            if ori_hit_p[~mask].shape[0]:
                add_mask = ~mask & (num_pos[:, i - 1] > threshold) & (pred_sign[:, i] == -1).squeeze() & (pred[:, i] <= 1).squeeze()
                add_id = np.where(add_mask == True)[0]
                for idx in add_id:
                    hit_pts[idx].extend(ori_hit_p[idx].reshape(-1, 3).tolist())
            ori_hit_p[~mask] = temp_hit_p[~mask]
            ori_sign = pred_sign[:, i]

        '''Negative Direction'''
        ori_sign = pred_sign[:, res - 1].reshape(-1, 1)
        ori_hit_p = pts[:, res - 1] + pred[:, res - 1] * pred_sign[:, res - 1] * ray_d[:, res - 1]

        for i in range(res - 2, -1, -1):
            mask = (ori_sign == pred_sign[:, i].reshape(-1, 1)).squeeze()
            temp_hit_p = pts[:, i] + pred[:, i] * pred_sign[:, i] * ray_d[:, i]
            num_neg[mask, i] = num_neg[mask, i + 1] + 1

            ori_hit_p[mask] += temp_hit_p[mask]
            ori_hit_p[~mask] = ori_hit_p[~mask] / num_neg[~mask, i + 1].reshape(-1, 1)

            if ori_hit_p[~mask].shape[0]:
                add_mask = ~mask & (num_neg[:, i + 1] > threshold) & (pred_sign[:, i] == 1).squeeze() & (
                            pred[:, i] <= 1).squeeze()
                add_id = np.where(add_mask == True)[0]
                for idx in add_id:
                    hit_pts[idx].extend(ori_hit_p[idx].reshape(-1, 3).tolist())
            ori_hit_p[~mask] = temp_hit_p[~mask]
            ori_sign = pred_sign[:, i]

        '''Unique'''
        uni_hit_pts = np.zeros((1, 3))
        uni_pred_hit_pts_z = [[] for i in range(pts.shape[0])]

        print("rays_shape:",len(hit_pts))

        for i in range(len(hit_pts)):
            row = np.array(hit_pts[i]).reshape(-1, 3)
            z = row[:, dir]
            uni_z, uni_pred_hit_pts_z = self.unique_z(z,uni_pred_hit_pts_z, i)

            if dir == 2:
                origin_pts = np.repeat(row[0, :2].reshape(1, -1), len(uni_z), axis=0)
                uni_hit_pts = np.concatenate((uni_hit_pts, np.concatenate((origin_pts, uni_z), axis=1)))
            elif dir == 1:
                origin_pts = np.repeat(
                    np.concatenate((row[0, 0].reshape(-1, 1), row[0, 2].reshape(-1, 1)), axis=1).reshape(1, -1),
                    len(uni_z), axis=0)
                origin_pts = np.concatenate((origin_pts[:, 0].reshape(-1, 1), uni_z, origin_pts[:, 1].reshape(-1, 1)),
                                            axis=1)
                uni_hit_pts = np.concatenate((uni_hit_pts, origin_pts))
            elif dir == 0:
                origin_pts = np.repeat(row[0, 1:].reshape(1, -1), len(uni_z), axis=0)
                uni_hit_pts = np.concatenate((uni_hit_pts, np.concatenate((uni_z, origin_pts), axis=1)))

        uni_hit_pts = uni_hit_pts[1:]
        print("pred shape:", uni_hit_pts.shape)
        return uni_hit_pts,uni_pred_hit_pts_z

    def _pcdpostprocess_gt(self,pts,gt_hit_pts,dir):
        inter_num =  np.zeros((len(gt_hit_pts),1))
        gt_hit_pts_uni_z = [[] for i in range(pts.shape[0])]
        gt_hit_pts_uni = np.zeros((1, 3))

        for i in range(len(gt_hit_pts)):
            row = gt_hit_pts[i]
            z = np.unique(np.around(row[:,dir], 5)).reshape(-1,1)
            inter_num[i] = len(z)
            gt_hit_pts_uni_z[i].extend(z)
            if dir == 2:
                origin_pts = np.repeat(pts[i,0,:2].reshape(1,-1),len(z),axis = 0)
                gt_hit_pts_uni = np.concatenate((gt_hit_pts_uni,np.concatenate((origin_pts,z),axis = 1)))
            elif dir == 1:
                origin_pts = np.repeat(np.concatenate((pts[i,0,0].reshape(-1,1),pts[i,0,2].reshape(-1,1)),axis = 1).reshape(1,-1), len(z), axis=0)
                origin_pts = np.concatenate((origin_pts[:,0].reshape(-1,1),z,origin_pts[:,1].reshape(-1,1)),axis = 1)
                gt_hit_pts_uni = np.concatenate((gt_hit_pts_uni,origin_pts))
            elif dir == 0:
                origin_pts = np.repeat(pts[i, 0,1:].reshape(1, -1), len(z), axis=0)
                gt_hit_pts_uni = np.concatenate((gt_hit_pts_uni, np.concatenate((z,origin_pts), axis=1)))

        gt_hit_pts_uni = gt_hit_pts_uni[1:]
        print("gt {} shape:".format(dir),gt_hit_pts_uni.shape)
        return gt_hit_pts_uni,gt_hit_pts_uni_z,inter_num

    def unique_z(self,z,uni_pred_hit_pts_z,i):
        uni_z = -np.ones((1,))
        for j in range(len(z)):
            # uni_mask = abs(uni_z - z[j]) < (2 / 128) * 0.1
            uni_mask = abs(uni_z - z[j]) < 1/128

            if not np.where(uni_mask == True)[0].shape[0]:
                uni_z = np.concatenate((uni_z, np.array(z[j]).reshape(1)))
        uni_z = np.array(uni_z[1:]).reshape(-1, 1)
        uni_pred_hit_pts_z[i].extend(uni_z.tolist())
        return uni_z,uni_pred_hit_pts_z

    def pcdpostprocess_main(self,res,f,dir,test_path):
        '''init'''
        pts,ray_d,pred,pred_sign,rays_id,pred_hit_mask = self.init(dir = dir,filename = f,res=res)

        '''init folds'''
        dataPath = self.args.dataPath + f + "/" + str(res) + "/"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        test_res_path = test_path + "/" + str(res)
        if not os.path.exists(test_res_path):
            os.makedirs(test_res_path)

        '''pred post_process'''
        threshold = self.args.postprocessingThreshold
        pred_hit_pts,pred_hit_pts_uni_z = self._pcdpostprocess(res,pts,pred,pred_sign,ray_d,dir,threshold)

        '''saving'''
        np.savetxt(test_res_path + "/hit_pts_processing_"+ str(dir) + ".xyz",pred_hit_pts)
        save_h5(dataPath+ "/hit_pts_pred_"+ str(dir) + ".h5",pred_hit_pts)

        return pred_hit_pts.shape

    def pcdpostprocess_main_gt(self,res,f,dir,test_path):
        pts,gt,hit_pts,hit_mask = self.init_gt(dir,f,res)
        gt_hit_pts_uni,gt_hit_pts_uni_z,gt_inter_num = self._pcdpostprocess_gt(pts,hit_pts,dir)

        dataPath = self.args.dataPath + f + "/" + str(res) + "/"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        test_res_path = test_path + "/" + str(res)
        if not os.path.exists(test_res_path):
            os.makedirs(test_res_path)

        np.savetxt(test_res_path + "/gt_hit_pts_processing_{}.xyz".format(dir),gt_hit_pts_uni)
        save_h5(dataPath+ "/gt_hit_pts_processing_{}.h5".format(dir),gt_hit_pts_uni)
