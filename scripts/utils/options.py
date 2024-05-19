import argparse


def parse_args():
    parse =  argparse.ArgumentParser()
    parse.add_argument('--batch_size',type = int,default= 1024,help = 'input batch size')
    parse.add_argument('--epoch',type = int,default= 100 ,help = 'number of epoch in training')
    parse.add_argument('--train_res',type = int,default= 256 + 1 ,help = 'number of training resolution of model')
    parse.add_argument('--train_sample_num',type = int,default= 256 + 1,help = 'training sample number of model')
    parse.add_argument('--test_res',type = int,default= 32 ,help = 'number of testing resolution of model')
    parse.add_argument('--file_path',type = str,default= None ,help = 'mesh(obj) path of meshPath')
    parse.add_argument('--file_name',type = str,default= None ,help = 'mesh(obj) name of meshPath')
    parse.add_argument('--dir',type = int,default= None ,help = 'the direction of rays')
    parse.add_argument('--gpu',type = str,default='0',help = 'specify gpu device')
    parse.add_argument('--pretrain',type = str,default=None,help = 'whether use pertrain model')
    parse.add_argument('--mask_pretrain',type = str,default=None,help = 'whether use mask pertrain model')
    parse.add_argument('--train_metric',type = str,default=True,help = 'whether evaluate on traning dataset')
    parse.add_argument('--model_name',type = str,default= 'UODF',help = 'model name')
    parse.add_argument('--learning_rate',type = float,default= 0.001,help = 'learning rate in training')
    parse.add_argument('--optimizer',type = str,default='adam',help = 'optimizer for training')
    parse.add_argument('--decay_rate',type = float,default=1e-4,help = 'decay raye of learning rate')
    parse.add_argument('--use_embedder',type = bool,default= True,help = 'whether use embedder of point')
    parse.add_argument('--multries_xyz',default= 10,help = 'log2 of ,ax freq for positional encoding(3D location)')

    '''path setting'''
    parse.add_argument('--meshPath', help='path to input mesh folder of meshes', type=str,
                        default= None)
    parse.add_argument('--dataPath', help='path to data folder of gt', type=str,
                        default= None)
    parse.add_argument('--ExpPath', help='path to exp folder of train', type=str,
                        default= None)


    parse.add_argument('--testPath', help='path to test folder of pred', type=str,
                        default= None)
    parse.add_argument('--dictPath', help='path to test folder of pred', type=str,
                        default= None)


    parse.add_argument('--experiment_path', help='path to exp folder of train', type=str,
                        default=None)


    '''training setting'''
    parse.add_argument('--sample_num',type = int,default=  + 1 ,help = 'sample num of hit rays')

    '''pred setting'''
    parse.add_argument('--pred_res',type = int,default= 256 + 1,
                       help = 'pred resolution of models')
    parse.add_argument('--postprocessingThreshold',type = int,default= 1 ,
                       help = 'number of points to aggregation')
    parse.add_argument('--use_pcd_post_processing',type = bool,default= True ,
                       help = 'whether use post processing of point')
    parse.add_argument('--use_eval_pcd',type = bool,default= False ,
                       help = 'whether eval point cloud cd distance ')
    parse.add_argument('--use_possion_rec', type=bool, default=False,
                       help='whether use possion reconstrucion of point cloud ')
    parse.add_argument('--use_delete_mesh', type=bool, default=False,
                       help='whether use delete mesh of watertight model ')
    parse.add_argument('--use_metrics_from_LDIF', type=bool, default=False,
                       help='whether use metrics from LDIF ')


    return parse.parse_args()
