import os
import torch
import numpy as np
import time
from model.UodfModel import UodfModel,UodfMaskModel
from model.UodfDataSet import UodfTestDataSet
from model.Trainer import BaseTrainer
from utils.options import parse_args
from utils.utils import save_h5,load_h5
from utils.utils import init_path
from utils.timer import Timer
from model.Embedder import PositionalEmbedding,BaseEmbedding
from utils.remove_points import remove_points


time_list = []

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)
class UodfTestTrainer(BaseTrainer):
    def __init__(self,
                 dir,
                 model:UodfModel,
                 mask_model:UodfMaskModel,
                 test_set:UodfTestDataSet,
                 args,
                 res,
                 **kwargs
                 ):
        super().__init__(model,test_set,args,**kwargs)
        self.mask_net = mask_model
        self.dir = dir
        self.res = res

        self.loss1_rate = 3e3
        self.loss2_rate = 5e1
        self.loss3_rate = 1e3
        
        if args.use_embedder:
            self.embedder = PositionalEmbedding(x_dim=2,level=10)
            self.PE_embedder = PositionalEmbedding(x_dim=2,level=10)
        else:
            self.PE_embedder = None
            self.embedder = BaseEmbedding(x_dim=2)


    def validate_time(self):
        self.net = self.net.eval()
        self.mask_net = self.mask_net.eval()

        total_pred = []
        total_y_hat = []
        total_sign = []

        i = 0
        '''compute all batch'''
        for data in self.train_set:
            i+=1
            X, Y = data
            X, Y = X.cuda(), Y.cuda()

            pts, ray_d, sign, gt = X[:, :3], X[:, 3:6], X[:, 6:7], Y[:,0:1]

            with torch.no_grad():
                y_hat = self.mask_net(pts,self.dir)

            with TemporaryGrad():
                pts = pts.requires_grad_(True)

            bc_pred_hit_mask = torch.where(y_hat >= 0.5, torch.ones_like(y_hat),torch.zeros_like(y_hat)).bool().squeeze()

            pts = pts[bc_pred_hit_mask]
            sign = sign[bc_pred_hit_mask]

            pred, input_pts = self.net(pts, self.dir)
            grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                       grad_outputs=torch.ones_like(pred), create_graph=True,
                                       retain_graph=True)[0]

            grad = grad[:, self.dir].view(-1, 1)
            sign = torch.where(grad < 0, torch.ones_like(sign), -torch.ones_like(sign))


            total_pred.append(pred.detach())
            total_y_hat.append(y_hat.detach())
            total_sign.append(sign.detach())

        # all result
        total_pred = torch.cat(total_pred, dim=0)
        total_y_hat = torch.cat(total_y_hat, dim=0)
        total_sign = torch.cat(total_sign, dim=0)

        pred_hit_mask = torch.where(total_y_hat >= 0.5, torch.ones_like(total_y_hat),
                                    torch.zeros_like(total_y_hat)).cpu().numpy().astype(bool)

        self._save_data(total_pred,total_sign,pred_hit_mask)


    def validate(self):
        self.net = self.net.eval()
        self.mask_net = self.mask_net.eval()

        total_pred = []
        total_sign = []
        
        test_time_sum = 0
        


        '''compute all batch'''
        for data in self.train_set:
            X, Y = data
            X, Y = X.cuda(), Y.cuda()

            pts, ray_d, sign, gt = X[:, :3], X[:, 3:6], X[:, 6:7], Y[:, 0:1]
            
            if self.dir == 2:
                pts_2D = pts[:,:2]
            elif self.dir == 1:
                pts_2D = torch.cat((pts[:,0].view(-1,1),pts[:,2].view(-1,1)),dim = -1)
            elif self.dir == 0:
                pts_2D = pts[:,1:]
            
            # pts_embedder = self.embedder(pts_2D).cuda()
            # pts_latent2D_PE_all = self.PE_embedder(pts_2D).cuda()  #[257*257*2,42]
            

            mask_net_start_event = torch.cuda.Event(enable_timing=True)
            mask_net_end_event = torch.cuda.Event(enable_timing=True)
            mask_net_start_event.record()
            
            with torch.no_grad():
                y_hat = self.mask_net(pts, self.dir)
                # y_hat = self.mask_net(pts_embedder, self.dir) 
                
            mask_net_end_event.record()    
            mask_net_end_event.synchronize()   
                
            pred_hit_mask = torch.where(y_hat >= 0.5, torch.ones_like(sign),
                                    torch.zeros_like(sign)).cpu().numpy().astype(bool).squeeze()
                
                
                
            with TemporaryGrad():
                pts = pts.requires_grad_(True)
                
            pts_masked = pts[pred_hit_mask]
            ray_d = ray_d[pred_hit_mask]
            sign = sign[pred_hit_mask]
            
            # indices = torch.round((pts_masked + 1) / 2 * (256)).long()
            # if self.dir == 2:
            #     pts_2D_masked = pts_masked[:,:2]
            #     indices = (indices[:,0]*257*2 + indices[:,1]*2 + (pts_masked[:,2]+1)/2).long()
            # elif self.dir == 1:
            #     pts_2D_masked = torch.cat((pts_masked[:,0].view(-1,1),pts_masked[:,2].view(-1,1)),dim = -1)
            #     indices = (indices[:,0]*257*2 + indices[:,2]*2 + (pts_masked[:,1]+1)/2).long()
            # elif self.dir == 0:
            #     pts_2D_masked = pts_masked[:,1:]
            #     indices = (indices[:,1]*2 + indices[:,2]*257*2 + (pts_masked[:,0]+1)/2).long()
            # pts_latent2D_PE = self.PE_embedder(pts_2D_masked).cuda()    #[n,42],n=2*k
            # pts_latent2D_tri = self.triplane_embedder(pts_masked).cuda()
            # pts_latent2D_PE = pts_latent2D_PE_all[indices]
            
            
            uodf_net_start_event = torch.cuda.Event(enable_timing=True)
            uodf_net_end_event = torch.cuda.Event(enable_timing=True)
            uodf_net_start_event.record()

            # pred, input_pts = self.net(pts_masked, pts_latent2D_PE, self.dir)
            pred, input_pts = self.net(pts_masked, self.dir)
            
            uodf_net_end_event.record() 
            uodf_net_end_event.synchronize()  
            
            grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                       grad_outputs=torch.ones_like(pred), create_graph=True,
                                       retain_graph=True)[0]

            grad = grad[:, self.dir].view(-1, 1)
            sign = torch.where(grad < 0, torch.ones_like(sign), -torch.ones_like(sign))

            total_pred = pred
            total_sign = sign

            hit_pts = pts_masked + total_pred * total_sign * ray_d

            pts_num=len(hit_pts)
            # threshold
            hit_threshold = 2/(res-1) 
            circu_num = 0
            hit_pts_odd = hit_pts[::2]
            hit_pts_even = hit_pts[1::2]
            mask_odd = torch.ones_like(hit_pts_odd[:,0], dtype=torch.float32).bool()
            mask_even = torch.ones_like(hit_pts_even[:,0], dtype=torch.float32).bool()
            
            point_start_event = torch.cuda.Event(enable_timing=True)
            point_end_event = torch.cuda.Event(enable_timing=True)
            point_start_event.record()

            while(pts_num > 0):
                circu_num += 1
                   
                mid_hit_pts = (hit_pts_odd + hit_pts_even)/2

                ray_d = torch.zeros_like(mid_hit_pts, dtype=torch.float32)
                ray_d[:, args.dir] = 1
                
                indices = torch.round((mid_hit_pts + 1) / 2 * (256)).long()
                
                # if self.dir == 2:
                #     mid_hit_pts_2D = mid_hit_pts[:,:2]
                #     pts_coords = indices[:, 0]*257*2 + indices[:, 1]*2
                # elif self.dir == 1:
                #     mid_hit_pts_2D = torch.cat((mid_hit_pts[:,0].view(-1,1),mid_hit_pts[:,2].view(-1,1)),dim = -1)
                #     pts_coords = indices[:, 0]*257*2 + indices[:, 2]*2
                # elif self.dir == 0:
                #     mid_hit_pts_2D = mid_hit_pts[:,1:]
                #     pts_coords = indices[:, 1]*2 + indices[:, 2]*257*2
                # mid_hit_pts_PE = self.PE_embedder(mid_hit_pts_2D).cuda()
                # mid_hit_pts_PE = pts_latent2D_PE_all[pts_coords]
                
                # pred, input_pts = self.net(mid_hit_pts, mid_hit_pts_PE,self.dir)
                pred, input_pts = self.net(mid_hit_pts,self.dir)
                
                
                grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                       grad_outputs=torch.ones_like(pred), create_graph=True,
                                       retain_graph=True)[0]

                grad = grad[:, self.dir].view(-1, 1)
                sign = torch.where(grad < 0, torch.ones_like(grad), -torch.ones_like(grad))

                
                hit_pts_new = mid_hit_pts + pred * sign * ray_d
                hit_pts_new_fake = mid_hit_pts + (pred-0.2/(res-1)) * sign * ray_d * (-1)

                
                sign_mask = ((pred * sign ) < 0).reshape(-1) 
                
                save_mask_0 = (((mask_odd.int() - ((hit_pts_new[:,self.dir] - hit_pts_odd[:,self.dir]) > hit_threshold).int()) < 1
                              )&((mask_even.int() - ((hit_pts_even[:,self.dir] - hit_pts_new[:,self.dir]) > hit_threshold).int()) < 1))
                save_mask = save_mask_0
                mask_2fake = (~mask_odd)&(~mask_even)
                dis_fake_pts = hit_pts_even[:,self.dir] - hit_pts_odd[:,self.dir]
                not_save_mask = ((dis_fake_pts <= hit_threshold)&mask_2fake).reshape(-1)
                save_mask = ~not_save_mask&save_mask_0

                
                hit_new_num = torch.sum(save_mask)
                # print("cir num",circu_num,"pts num", pts_num)
                pts_num = 2 * hit_new_num

                hit_pts_save = hit_pts_new[save_mask]
                hit_pts = torch.cat((hit_pts,hit_pts_save),dim=0)

                hit_pts_odd_true = hit_pts_odd[sign_mask&save_mask]
                hit_pts_new_true_left = hit_pts_new[sign_mask&save_mask]
                mask_odd_true = mask_odd[sign_mask&save_mask]
                mask_new_true_left = torch.ones_like(mask_odd_true)
                
                hit_pts_even_true = hit_pts_even[~sign_mask&save_mask]
                hit_pts_new_true_right = hit_pts_new[~sign_mask&save_mask]
                mask_even_true = mask_even[~sign_mask&save_mask]
                mask_new_true_right = torch.ones_like(mask_even_true)
                
                hit_pts_odd_fake = hit_pts_odd[~sign_mask&save_mask]
                hit_pts_new_fake_left = hit_pts_new_fake[~sign_mask&save_mask]
                mask_odd_fake = mask_odd[~sign_mask&save_mask]
                mask_new_fake_left = torch.zeros_like(mask_odd_fake)
                
                hit_pts_even_fake = hit_pts_even[sign_mask&save_mask]
                hit_pts_new_fake_right = hit_pts_new_fake[sign_mask&save_mask]
                mask_even_fake = mask_even[sign_mask&save_mask]
                mask_new_fake_right = torch.zeros_like(mask_even_fake)
                
                hit_pts_odd = torch.cat((hit_pts_odd_true,hit_pts_new_true_right,hit_pts_odd_fake,hit_pts_new_fake_right),dim=0)
                hit_pts_even = torch.cat((hit_pts_new_true_left,hit_pts_even_true,hit_pts_new_fake_left,hit_pts_even_fake),dim=0)
                mask_odd = torch.cat((mask_odd_true,mask_new_true_right,mask_odd_fake,mask_new_fake_right),dim=0).bool()
                mask_even = torch.cat((mask_new_true_left,mask_even_true,mask_new_fake_left,mask_even_fake),dim=0).bool()
                
            
            point_end_event.record() 
            point_end_event.synchronize()  
            test_mask_time = mask_net_start_event.elapsed_time(mask_net_end_event) / 1000 
            test_uodf_time = uodf_net_start_event.elapsed_time(uodf_net_end_event) / 1000
            test_point_time = point_start_event.elapsed_time(point_end_event) / 1000
            test_time_sum = test_mask_time + test_uodf_time + test_point_time
            # print("mask ",test_mask_time)
            # print("uodf ",test_uodf_time)
            # print("point ",test_point_time)
            # print("time:",test_time_sum)
            time_list.append(test_time_sum)
            
            # print(circu_num,pts_num)
            
                           
        # print("1")
        '''saving'''
        test_path = args.testPath + f + "/"
        test_res_path = test_path + "/single_" + str(res)
        if not os.path.exists(test_res_path):
            os.makedirs(test_res_path)

        np.savetxt(test_res_path + "/hit_pts_processing_"+ str(self.dir) + ".xyz",hit_pts.detach().cpu().numpy())
        save_h5(test_res_path+ "/hit_pts_pred_"+ str(self.dir) + ".h5",hit_pts.detach().cpu().numpy())
        
        dataPath = args.dataPath + args.file_name + "/" + str(res) + "/"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)


    def _save_data(self,pred,sign,pred_hit_mask):
        pred = pred.cpu().numpy()
        sign = sign.cpu().numpy()


        dataPath = args.dataPath + args.file_name + "/" + str(res) + "/"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        # OLD save in dataPath
        save_h5(dataPath + "/pred_single_{}_{}.h5".format(self.res,self.dir),pred)
        save_h5(dataPath + "/pred_single_sign_{}_{}.h5".format(self.res,self.dir),sign)
        save_h5(dataPath + "/pred_single_hit_mask_{}_{}.h5".format(self.res,self.dir),pred_hit_mask)
        # #NEW
        # test_res_path = args.testPath + args.file_name + "/" + str(res) + "/"
        # if not os.path.exists(test_res_path):
        #     os.makedirs(test_res_path)
        # save_h5(test_res_path + "/pred_single_{}_{}.h5".format(self.res,self.dir),pred)
        # save_h5(test_res_path + "/pred_single_sign_{}_{}.h5".format(self.res,self.dir),sign)
        # save_h5(test_res_path + "/pred_single_hit_mask_{}_{}.h5".format(self.res,self.dir),pred_hit_mask)

    def _compute_loss(self, pred, gt, grad, pred_t_hat):
        loss1 = self._compute_loss1(pred, gt)
        loss2 = self._compute_loss2(grad)

        loss3 = self._compute_loss3(pred_t_hat)

        loss = loss1 + loss2 + loss3
        return {"loss1": loss1.detach().item()/self.loss1_rate,
                "loss2": loss2.detach().item()/self.loss2_rate,
                "loss3": loss3.detach().item()/self.loss3_rate,
                "loss": loss}

    def _compute_loss1(self, pred, gt):
        loss = torch.nn.functional.l1_loss(pred * self.loss1_rate, gt * self.loss1_rate)
        # loss = torch.nn.L1Loss(pred * self.loss1_rate, gt * self.loss1_rate)
        return loss

    def _compute_loss2(self, grad):
        '''loss2'''
        grad_constraint = torch.abs(grad.reshape(-1,1).norm(dim=-1) - 1)
        loss = grad_constraint.mean() * self.loss2_rate
        return loss

    def _compute_loss3(self, pred_t_hat):
        loss = torch.abs(pred_t_hat * self.loss3_rate).mean()
        return loss

    def _distance(self,x1,x2):
        d = np.sqrt((x1[:, 0] - x2[:, 0]) * (x1[:, 0] - x2[:, 0])
                    + (x1[:, 1] - x2[:, 1]) * (x1[:, 1] - x2[:, 1]) +
                    (x1[:, 2] - x2[:, 2]) * (x1[:, 2] - x2[:, 2]))
        return d.reshape(-1, 1)

def run_test(args,res):
    print("[INFO] Testing " + str(args.dir) + "....")

    if args.dir == 2:
        x = np.linspace(-1, 1, res)
        y = np.array([-1,1])
        pts = np.array([(i,j,k) for i in x for j in x for k in y])
        pts = torch.tensor(pts, dtype=torch.float32)
        ray_d = torch.zeros_like(pts, dtype=torch.float32)
        ray_d[:, args.dir] = 1
        sign = torch.zeros((pts.shape[0], 1))
    elif args.dir == 1:
        x = np.linspace(-1, 1, res)
        y = np.array([-1,1])
        pts = np.array([(i,j,k) for i in x for k in x for j in y])
        pts = torch.tensor(pts, dtype=torch.float32)
        ray_d = torch.zeros_like(pts, dtype=torch.float32)
        ray_d[:, args.dir] = 1
        sign = torch.zeros((pts.shape[0], 1))
    elif args.dir == 0:
        x = np.linspace(-1, 1, res)
        y = np.array([-1,1])
        pts = np.array([(i,j,k) for k in x for j in x for i in y])
        pts = torch.tensor(pts, dtype=torch.float32)
        ray_d = torch.zeros_like(pts, dtype=torch.float32)
        ray_d[:, args.dir] = 1
        sign = torch.zeros((pts.shape[0], 1))

    X = torch.cat((pts,ray_d,sign),dim = 1)
    Y = torch.zeros(X.shape[0],1)

    # testDataset = UodfTestDataSet(X,Y,batch_size=65536)
    testDataset = UodfTestDataSet(X,Y,batch_size=res*res*2)

    model = UodfModel(args=args).cuda()
    mask_model = UodfMaskModel(args=args).cuda()

    '''loading checkpoints'''
    if args.pretrain is not None:
        print('Use pertrain model....')
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No existing model,starting training from scratch...")

    if args.mask_pretrain is not None:
        print('Use pertrain mask model....')
        mask_checkpoints = torch.load(args.mask_pretrain)
        mask_model.load_state_dict(mask_checkpoints['model_state_dict'])


    TestTrainer = UodfTestTrainer(args.dir,model,mask_model,testDataset,args,res)
    # TestTrainer.validate_time()
    TestTrainer.validate()
    # print("loss1:{}\nloss2:{}\nloss3:{}\n".format(loss_dict["loss1"],loss_dict["loss2"],loss_dict["loss3"]))

def test(args,res):
    for dir in range(2,-1,-1):
        test_timer = Timer()
        args.dir = dir
        test_path = args.ExpPath + f + "/{}_{}".format(f, dir) + "/checkpoints/"
        if os.path.exists(test_path):
            file = os.listdir(test_path)
            file.sort()
            args.pretrain = test_path + file[0]
        print("[INFO] Uodf Model Path: ",args.pretrain)

        mask_pretrain_path = args.ExpPath + f + "/{}_mask_{}".format(f, dir) + "/checkpoints/"
        if os.path.exists(mask_pretrain_path):
            mask_file = os.listdir(mask_pretrain_path)
            mask_file.sort()
        args.mask_pretrain = mask_pretrain_path + mask_file[0]
        print("[INFO] Uodf Mask Model Path: ",args.mask_pretrain)

        if dir == 2:
            print("[INFO] Testing Dir (0,0,1)")
            time_start = time.perf_counter()

            run_test(args, res)
            
            time_end = time.perf_counter()
            print("time:",time_end-time_start)
            print("[INFO] Test Time")
        if dir == 1:
            print("[INFO] Testing Dir (0,1,0)")
            time_start = time.perf_counter()

            run_test(args, res)
            
            time_end = time.perf_counter()
            print("time:",time_end-time_start)
            print("[INFO] Test Time")
        if dir == 0:
            print("[INFO] Testing Dir (1,0,0)")

            time_start = time.perf_counter()

            run_test(args, res)
            
            time_end = time.perf_counter()
            print("time:",time_end-time_start)
            print("[INFO] Test Time")


if __name__ == "__main__":
    args = parse_args()
    init_path(args)

    if(not os.path.exists(args.meshPath)):
        print("[Error] No File")

    filename = os.listdir(args.meshPath)
    filename.sort()
    res = args.pred_res
    num_time = 200
    warm_time = 50

    for i in range(len(filename)):
        timer = Timer()
        f = str(filename[i].split(".")[0])
        time_list = []

        for j in range(num_time + warm_time):  
            args.file_path = filename[i]
            args.file_name = f

            path_name = os.listdir(args.ExpPath + f)
            path_name.sort()

            # time_start_0 = time.perf_counter()
        

            print(str(i) + ":" + f)
            print(str(j))
            print("*" * 60)
            test(args,res)
            print("*" * 60)
            
        
        print("*" * 60)
        test_res_path = args.testPath + args.file_name + "/single_" + str(res) + "/"
        gen_pts2 = load_h5(test_res_path + "/hit_pts_pred_" +str(2) + ".h5")
        gen_pts1 = load_h5(test_res_path + "/hit_pts_pred_" +str(1) + ".h5")
        gen_pts0 = load_h5(test_res_path + "/hit_pts_pred_" +str(0) + ".h5")

        gen_pts = remove_points(gen_pts2,gen_pts1,gen_pts0,res)
        np.savetxt(test_res_path + "hit_pts_processing.xyz", gen_pts)
        print("origin_all_pts_num:",gen_pts2.shape[0] + gen_pts1.shape[0] + gen_pts0.shape[0])
        print("remove_outer_pts_num:",gen_pts.shape)
        
        #time save
        time_list_new = time_list[warm_time*3:]
        print(len(time_list_new))
        
        time_list_sum = [sum(time_list_new[i:i+3]) for i in range(0, len(time_list_new), 3)]
        sorted_time = sorted(time_list_sum)
        n = len(time_list_sum)
        print(n)
        if n%2 == 0:
            median = (sorted_time[n//2-1]+sorted_time[n//2])/2
        else:
            median = sorted_time[n//2]
        
        time_sum = sum(time_list_sum)
        time_avg = time_sum/num_time
        print("sum_of_{}_time".format(num_time),time_sum)
        print("avg_of_time",time_avg)
        print("median_of_time",median)
        
        
        time_list_sum.append(0)
        time_list_sum.append(time_avg)
        time_list_sum.append(median)
        np.savetxt(test_res_path + "time_list_sum.txt", time_list_sum)
        print("*" * 60)
        
        torch.cuda.empty_cache()