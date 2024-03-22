import os
import trimesh
import numpy as np
from pathlib import Path
from utils.options import parse_args
from utils.utils import init_path,load_h5,compute_trimesh_chamfer_pcd
from utils.metricsFromLDIF import mesh_metrics
from utils.generatedata import GenerateData
from utils.PcdPostProcess import PcdPostProcess
                                                           

def init_gt_data(args,filename,res):
    print("[INFO] Init {} GT data".format( str(filename.split(".")[0]) ))
    datapath = args.dataPath + str(filename.split(".")[0]) + "/"
    datapath_dir = Path(datapath)
    datapath_dir.mkdir(exist_ok=True)

    if not os.path.exists(datapath + "/{}/AxisSize_{}_gt_{}.h5".format(res,res,0)):
        datagenerator = GenerateData(args, filename)
        print("-" * 80)
        print("[INFO] Dir(0,0,1)")
        datagenerator.AxisSize_generator(dir = 2, res = res)
        print("*" * 80)
        print("[INFO] Dir(0,1,0)")
        datagenerator.AxisSize_generator(dir = 1, res = res)
        print("*" * 80)
        print("[INFO] Dir(1,0,0)")
        datagenerator.AxisSize_generator(dir = 0, res = res)


def pcdpostprocessing_gt(res,f,data_path,test_path):
    pp = PcdPostProcess(args=args)

    pp.pcdpostprocess_main_gt(res, f, 2, test_path)
    pp.pcdpostprocess_main_gt(res, f, 1, test_path)
    pp.pcdpostprocess_main_gt(res, f, 0, test_path)

    gt_pts2 = load_h5(data_path + "/{}/gt_hit_pts_processing_{}.h5".format(res,2))
    gt_pts1 = load_h5(data_path + "/{}/gt_hit_pts_processing_{}.h5".format(res,1))
    gt_pts0 = load_h5(data_path + "/{}/gt_hit_pts_processing_{}.h5".format(res,0))
    gt_pts = np.concatenate((gt_pts0, gt_pts1, gt_pts2))
    print("gt shape:",gt_pts.shape)

    mask = abs(gt_pts) > 1
    mask = np.sum(mask, axis=1)
    mask = mask < 1

    np.savetxt(test_path + "/{}/gt_hit_pts_processing.xyz".format(res), gt_pts[mask])


def metricsFromLDIF(res,test_path,filename,type = "WaterTight"):
    gt_file = args.meshPath + filename
    if type == "WaterTight":
        pred_file = test_path + "/{}/possion_watertight.obj".format(res)
    elif type == "NonWaterTight":
        pred_file = test_path + "/{}/non_watertight.obj".format(res)

    gt_mesh = trimesh.load(gt_file)
    pred_mesh = trimesh.load(pred_file)

    element = mesh_metrics(pred_mesh, gt_mesh)

    print("-" * 60)
    print("GT to Pred")
    for key in element:
        print(key, ":", element[key])
    return element['chamfer'],element['normal_consistency']

def eval_pcd(res,test_path):
    gt_pts = np.loadtxt(test_path + "/{}/gt_hit_pts_processing.xyz".format(res))
    gen_pts = np.loadtxt(test_path + "/{}/hit_pts_processing.xyz".format(res))

    print("gt shape :", gt_pts.shape)
    print("gen shape :", gen_pts.shape)
    pcd_cd = compute_trimesh_chamfer_pcd(gt_pts, gen_pts, type) * 1000
    print("Resolution: " + str(res) + "\n pcd_cd: ", pcd_cd)
    return pcd_cd


if __name__ == "__main__":
    args = parse_args()
    init_path(args)

    type = ["WaterTight","NonWaterTight"]
    filename = os.listdir(args.meshPath)
    filename.sort()

    res = args.pred_res

    for i in range(len(filename)):
        f = str(filename[i].split(".")[0])
  
        data_path = args.dataPath + f + "/"
        test_path = args.testPath + f + "/"

        # init gt data
        print("--------------Init GT Data Start-------------")
        init_gt_data(args,filename[i],res)
        # GT GEP
        print("--------------Init GT GEP Pts Start-------------")
        pcdpostprocessing_gt(res,f,data_path,test_path)
        # eval GEP
        print("--------------Eval GEP Start-------------")
        eval_pcd(res,test_path)
        # eval mesh pcd
        print("--------------Eval Mesh cd Start-------------")
        metricsFromLDIF(res,test_path,filename[i],type = type[0])












