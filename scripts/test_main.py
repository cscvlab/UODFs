import os
import torch
import numpy as np
import time
from model.UodfModel import UodfModel,UodfMaskModel
from model.UodfDataSet import UodfTestDataSet
from model.Trainer import BaseTrainer
from utils.options import parse_args
from utils.utils import save_h5
from utils.utils import init_path
from utils.timer import Timer

res_list = [32,64,128,256]

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
        total_gt = []
        total_grad = []
        total_pred_t_hat = []
        total_y_hat = []
        total_sign = []

        '''compute all batch'''
        for data in self.train_set:
            X, Y = data
            X, Y = X.cuda(), Y.cuda()

            pts, ray_d, sign, gt = X[:, :3], X[:, 3:6], X[:, 6:7], Y[:, 0:1]

            with torch.no_grad():
                y_hat = self.mask_net(pts, self.dir)

            with TemporaryGrad():
                pts = pts.requires_grad_(True)

            pred, input_pts = self.net(pts, self.dir)
            grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                       grad_outputs=torch.ones_like(pred), create_graph=True,
                                       retain_graph=True)[0]

            grad = grad[:, self.dir].view(-1, 1)
            sign = torch.where(grad < 0, torch.ones_like(sign), -torch.ones_like(sign))

            pred_hit_pts = pts + sign * torch.abs(pred) * ray_d
            pred_t_hat, _ = self.net(pred_hit_pts, self.dir)

            total_pred.append(pred.detach())
            total_gt.append(gt.detach())
            total_grad.append(grad.detach())
            total_pred_t_hat.append(pred_t_hat.detach())
            total_y_hat.append(y_hat.detach())
            total_sign.append(sign.detach())

        # all result
        total_pred = torch.cat(total_pred, dim=0)
        total_gt = torch.cat(total_gt, dim=0)
        total_grad = torch.cat(total_grad, dim=0)
        total_pred_t_hat = torch.cat(total_pred_t_hat, dim=0)
        total_y_hat = torch.cat(total_y_hat, dim=0)
        total_sign = torch.cat(total_sign, dim=0)

        pred_hit_mask = torch.where(total_y_hat >= 0.5, torch.ones_like(total_pred),
                                    torch.zeros_like(total_pred)).cpu().numpy().astype(bool)

        self._save_data(total_pred, total_sign, pred_hit_mask)

        # metrics
        loss_dict = self._compute_loss(total_pred[pred_hit_mask],
                                       total_gt[pred_hit_mask],
                                       total_grad[pred_hit_mask],
                                       total_pred_t_hat[pred_hit_mask])
        acc = loss_dict["loss1"] / self.loss1_rate \
              + loss_dict["loss2"] / self.loss2_rate +\
              loss_dict["loss3"] / self.loss3_rate

        return loss_dict,acc

    def _save_data(self,pred,sign,pred_hit_mask):
        pred = pred.cpu().numpy()
        sign = sign.cpu().numpy()


        dataPath = args.dataPath + args.file_name + "/" + str(res) + "/"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        save_h5(dataPath + "/pred_{}_{}.h5".format(self.res,self.dir),pred)
        save_h5(dataPath + "/pred_sign_{}_{}.h5".format(self.res,self.dir),sign)
        save_h5(dataPath + "/pred_hit_mask_{}_{}.h5".format(self.res,self.dir),pred_hit_mask)

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


def run_test(args,res):
    print("[INFO] Testing " + str(args.dir) + "....")
    x = np.linspace(-1, 1, res)
    pts = np.array([(i, j, k) for i in x for j in x for k in x])
    pts = torch.tensor(pts, dtype=torch.float32)
    ray_d = torch.zeros_like(pts, dtype=torch.float32)
    ray_d[:, args.dir] = 1
    sign = torch.zeros((pts.shape[0], 1))

    X = torch.cat((pts,ray_d,sign),dim = 1)
    Y = torch.zeros(X.shape[0],1)

    testDataset = UodfTestDataSet(X,Y,batch_size=65536)

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
    loss_dict,acc = TestTrainer.validate()
    print("loss1:{}\nloss2:{}\nloss3:{}\n".format(loss_dict["loss1"],loss_dict["loss2"],loss_dict["loss3"]))

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
            test_timer.start()

            run_test(args, res)

            test_timer.stop()
            test_timer.print()
            print("[INFO] Test Time")
        if dir == 1:
            print("[INFO] Testing Dir (0,1,0)")
            test_timer.start()

            run_test(args, res)

            test_timer.stop()
            test_timer.print()
            print("[INFO] Test Time")
        if dir == 0:
            print("[INFO] Testing Dir (1,0,0)")
            test_timer.start()

            run_test(args, res)

            test_timer.stop()
            test_timer.print()
            print("[INFO] Test Time")




if __name__ == "__main__":
    args = parse_args()
    init_path(args)

    if(not os.path.exists(args.meshPath)):
        print("[Error] No File")

    filename = os.listdir(args.meshPath)
    filename.sort()
    res = 256

    for i in range(len(filename)):
        timer = Timer()
        f = str(filename[i].split(".")[0])

        args.file_path = filename[i]
        args.file_name = f

        path_name = os.listdir(args.ExpPath + f)
        path_name.sort()

        timer.start()

        print(str(i) + ":" + f)
        print("*" * 60)
        test(args,res)
        print("*" * 60)

        timer.stop()
        timer.print()

        torch.cuda.empty_cache()

