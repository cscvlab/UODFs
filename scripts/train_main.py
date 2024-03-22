import os
import time
import torch
from pathlib import Path
from utils.options import parse_args
from utils.generatedata import GenerateData
from utils.utils import init_path
from train_uodf import run
from train_mask import run_mask


def init_data(args,filename):
    print("[INFO] Init {} data".format( str(filename.split(".")[0]) ))
    datapath = args.dataPath + str(filename.split(".")[0]) + "/"
    datapath_dir = Path(datapath)
    datapath_dir.mkdir(exist_ok=True)
    datagenerator = GenerateData(args, filename)

    print("-" * 80)
    print("[INFO] Dir(0,0,1)")
    datagenerator.generateData(res = args.train_res,dir = 2,sample_num = args.train_sample_num)
    print("*" * 80)
    print("[INFO] Dir(0,1,0)")
    datagenerator.generateData(res=args.train_res, dir = 1, sample_num=args.train_sample_num)
    print("*" * 80)
    print("[INFO] Dir(1,0,0)")
    datagenerator.generateData(res=args.train_res, dir = 0, sample_num=args.train_sample_num)

def train(args,filename,fun_run):
    f = str(filename.split(".")[0])
    args.file_path = filename
    args.file_name = f

    print(filename)
    print("*" * 60)
    start0 = time.time()

    print("[INFO] Training Dir (0,0,1)")
    args.dir = 2
    fun_run(args)

    print("[INFO] Training Dir (0,1,0)")
    args.dir = 1
    fun_run(args)

    print("[INFO] Training Dir (1,0,0)")
    args.dir = 0
    fun_run(args)

    print("*" * 60)

    end0 = time.time()
    seconds = end0 - start0

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("all time: %d:%02d:%02d" % (h, m, s))
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # init config
    args = parse_args()
    init_path(args)

    filename = os.listdir(args.meshPath)
    filename.sort()

    for i in range(len(filename)):
        f = str(filename[i].split(".")[0])

        # init data
        print("-" * 100)
        init_data(args,filename[i])

        # train UODF
        print("-" * 100)
        train(args,filename[i],run)

        # train Mask
        print("-" * 100)
        train(args, filename[i], run_mask)
        break










