import os
import sys
import numpy as np
from utils.utils import load_h5
from utils.PcdPostProcess import PcdPostProcess
from utils.remove_points import remove_points
from utils.utils import init_path,load_h5,meshlab_possion,degenerate_mesh
from utils.options import parse_args
from utils.timer import Timer


class PostProccess():
    def __init__(self,args):
        self.args = args
    def pcdpostprocessing(self,res,f,obj_dict,test_path):
        print("*" * 60)
        ppp = PcdPostProcess(args = self.args)
        print("[INFO] PostPocessing ...Dir(0,0,1)")
        num_pred_pts2 = ppp.pcdpostprocess_main(res = res, f=f,dir=2, test_path=test_path)
        print("[INFO] PostPocessing ...Dir(0,1,0)")
        num_pred_pts1 = ppp.pcdpostprocess_main(res = res, f=f,dir=1, test_path=test_path)
        print("[INFO] PostPocessing ...Dir(1,0,0)")
        num_pred_pts0 = ppp.pcdpostprocess_main(res = res, f=f,dir=0, test_path=test_path)

        dataPath = self.args.dataPath + f + "/" + str(res) + "/"
        test_res_path = test_path + str(res) + "/"

        gen_pts2 = load_h5(dataPath + "/hit_pts_pred_" +str(2) + ".h5")
        gen_pts1 = load_h5(dataPath + "/hit_pts_pred_" +str(1) + ".h5")
        gen_pts0 = load_h5(dataPath + "/hit_pts_pred_" +str(0) + ".h5")

        gen_pts = remove_points(gen_pts2,gen_pts1,gen_pts0,res)


        np.savetxt(test_res_path + "hit_pts_processing.xyz", gen_pts)

        obj_dict["num_point_of_pred2"] = num_pred_pts2
        obj_dict["num_point_of_pred1"] = num_pred_pts1
        obj_dict["num_point_of_pred0"] = num_pred_pts0

        obj_dict["num_point_of_pred_all"] = gen_pts2.shape[0] + gen_pts1.shape[0] + gen_pts0.shape[0]
        obj_dict["num_point_of_pred_all_remove_outer"] = gen_pts.shape

        print("origin_all_pts_num:",gen_pts2.shape[0] + gen_pts1.shape[0] + gen_pts0.shape[0])
        print("remove_outer_pts_num:",gen_pts.shape)
        print("*" * 60)

    def possion_reconstruction(self,res,test_path):
        test_res_path = test_path + str(res)

        pcd_file = test_res_path + "/hit_pts_processing.xyz"
        possion_file = test_res_path + "/possion_watertight.obj"
        mlx_file = os.getcwd() + "/scripts/src/Uodf_normals_possion.mlx"

        meshlab_possion(pcd_file,possion_file,mlx_file)

    def delete_mesh_from_possion(self,res,test_path,degreThreshold):
        test_res_path = test_path + str(res)

        pcd_file = test_res_path + "/hit_pts_processing.xyz"
        possion_file = test_res_path + "/possion_watertight.obj"
        non_water_file = test_res_path + "/non_watertight.obj"

        degenerate_mesh(pcd_file = pcd_file,possion_file = possion_file,non_water_file = non_water_file,degreThreshold = degreThreshold)

if __name__ == "__main__":
    args = parse_args()
    pp = PostProccess(args)
    init_path(args)

    filename = os.listdir(args.meshPath)
    filename.sort()

    res = args.pred_res
    args.use_pcd_post_processing = True     # set False after using test_single.py
    args.use_possion_rec = True
    args.use_delete_mesh = True

    num = 0

    for i in range(len(filename)):
        f = filename[i].split(".")[0]
        
        # if f in ["131971"]:

        obj_dict = dict()
        obj_dict["name"] = f

        test_path = args.testPath + f + "/"
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        print("%" * 60)
        print("Start Pocessing: " + str(i) + " " + f)

        print("--------------Point cloud PostPocessing Start-------------")
        if args.use_pcd_post_processing:
            pp.pcdpostprocessing(res,f,obj_dict,test_path = test_path)
        else :
            test_path = test_path + "single_"
        print()


        print("--------------Reconstruction WaterTight Model------------")
        if args.use_possion_rec:
            pp.possion_reconstruction(res,test_path)
        print()

        print("--------------Deg Mesh to Non-WaterTight Model ------------")
        degreThreshold = 2 / res * 0.007
        if args.use_delete_mesh:
            pp.delete_mesh_from_possion(res,test_path,degreThreshold)
        print() 
