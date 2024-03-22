import os
import datetime
import time
import torch
import logging
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from model.UodfModel import UodfModel
from model.UodfDataSet import DirectionalRaySample,UodfDataSet
from model.Trainer import BaseTrainer
from utils.utils import load_h5,create_directory,save_checkpoint

warnings.filterwarnings('ignore')

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)

class UodfTrainer(BaseTrainer):
    def __init__(self,
                 dir,
                 model:UodfModel,
                 train_set:UodfDataSet,
                 args,
                 **kwargs
                 ):
        super().__init__(model,train_set,args,**kwargs)
        self.dir = dir
        self.loss1_rate = 3e3
        self.loss2_rate = 5e1
        self.loss3_rate = 1e3
        self.i_resample = 30
        self.args = args

        self.global_epoch = 0
        self.best_tst_accuracy = 1000
        self.meter = defaultdict(list)
        self.checkpoints_dir = kwargs.get('checkpoints_dir', 'None')
        self.experiment_dir = kwargs.get('experiment_dir', 'None')

    def _training_step(self, data):
        X,Y = data
        X, Y = X.cuda(), Y.cuda()

        pts,ray_d,sign,gt = X[:,:3],X[:,3:6],X[:,6:7],Y

        self.optimizer.zero_grad()
        with TemporaryGrad():
            pts = pts.requires_grad_(True)


        pred,input_pts = self.net(pts,self.dir)

        grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                       grad_outputs=torch.ones_like(pred), create_graph=True,
                                       retain_graph=True)[0]

        grad = grad[:, self.dir].view(-1,1)
        sign = torch.where(grad < 0 ,torch.ones_like(sign),-torch.ones_like(sign))

        pred_hit_pts = pts + sign * torch.abs(pred) * ray_d
        pred_t_hat, _ = self.net(pred_hit_pts, self.dir)

        loss_dict = self._compute_loss(pred,gt,grad,pred_t_hat)

        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.detach().item(),
                "loss1":loss_dict["loss1"],
                "loss2":loss_dict["loss2"],
                "loss3":loss_dict["loss3"]}


    def _training(self):
        self.net = self.net.train()
        loss_total = 0
        for data in self.train_set:
            loss_dict = self._training_step(data)
            loss_total += loss_dict["loss1"] / self.loss1_rate + loss_dict["loss2"] / self.loss2_rate + loss_dict["loss3"] / self.loss3_rate

        loss_total = loss_total / len(self.train_set)
        return loss_dict,loss_total
    
    def _compute_loss1(self,pred,gt):
        loss = torch.nn.functional.l1_loss(pred * self.loss1_rate, gt * self.loss1_rate)
        return loss
    
    def _compute_loss2(self,grad):
        grad_constraint = torch.abs(grad.norm(dim=-1) - 1)
        loss = grad_constraint.mean() * self.loss2_rate
        return loss
    
    def _compute_loss3(self,pred_t_hat):
        loss = torch.abs(pred_t_hat * self.loss3_rate).mean()
        return loss

    def _compute_loss(self,pred,gt,grad,pred_t_hat):
        loss1 = self._compute_loss1(pred,gt)
        loss2 = self._compute_loss2(grad)
        loss3 = self._compute_loss3(pred_t_hat)

        loss = loss1 + loss2 + loss3

        return {"loss1":loss1.detach().item(),
                "loss2":loss2.detach().item(),
                "loss3":loss3.detach().item(),
                "loss":loss}

    def _validate(self, i=-1):
        self.net = self.net.eval()

        total_pred = []
        total_gt = []
        total_grad = []
        total_pred_t_hat = []

        '''compute all batch'''
        for data in self.train_set:
            X, Y = data
            X, Y = X.cuda(), Y.cuda()

            pts, ray_d, sign, gt = X[:, :3], X[:, 3:6], X[:,6:7], Y

            with TemporaryGrad():
                pts = pts.requires_grad_(True)


            pred,input_pts = self.net(pts,self.dir)

            grad = torch.autograd.grad(outputs=pred, inputs=input_pts,
                                           grad_outputs=torch.ones_like(pred), create_graph=True,
                                           retain_graph=True)[0]

            grad = grad[:, self.dir].view(-1,1)
            sign = torch.where(grad < 0 ,torch.ones_like(sign),-torch.ones_like(sign))

            pred_hit_pts = pts + sign * torch.abs(pred) * ray_d
            pred_t_hat, _ = self.net(pred_hit_pts, self.dir)

            total_pred.append(pred.detach())
            total_gt.append(gt)
            total_grad.append(grad.detach())
            total_pred_t_hat.append(pred_t_hat.detach())

        # all result
        total_pred = torch.cat(total_pred,dim = 0)
        total_gt = torch.cat(total_gt,dim = 0)
        total_grad = torch.cat(total_grad,dim = 0)
        total_pred_t_hat = torch.cat(total_pred_t_hat,dim = 0)

        # metrics
        loss_dict = self._compute_loss(total_pred, total_gt, total_grad, total_pred_t_hat)
        acc = loss_dict["loss1"] / self.loss1_rate + loss_dict["loss2"] / self.loss2_rate + loss_dict["loss3"] / self.loss3_rate
        return acc


    def train(self,total_epoch,**kwargs):
        logger =  kwargs.get('logger', 'None')
        train_metric =  kwargs.get('train_metric', 'True')


        self.net = self.net.cuda()
        task = tqdm(range(total_epoch), colour='green')

        for epoch_i in task:
            start0 = time.time()

            if epoch_i % self.i_resample == 0:
                self.train_set._sample_base(sample_num=self.args.sample_num)


            loss_dict,loss_total= self._training()
            acc = self._validate(epoch_i)

            self._update_scheduler(epoch_i,acc)

            log_str = "epoch:{} loss1: {:6f} loss2: {:6f} loss3: {:6f} loss: {:6f} ".format(
                epoch_i,
                loss_dict["loss1"] / self.loss1_rate,
                loss_dict["loss2"] / self.loss2_rate,
                loss_dict["loss3"] / self.loss3_rate,
                acc)

            task.set_description(log_str)


            if train_metric:
                logger.info(log_str)

            self.meter['train_acc'].append(acc)
            if (acc <= self.best_tst_accuracy) and epoch_i > 5:
                self.best_tst_accuracy = acc
                logger.info('save model...')
                save_checkpoint(
                    self.global_epoch + 1,
                    acc if train_metric else 0.0,
                    acc,
                    self.net,
                    self.optimizer,
                    str(self.checkpoints_dir),
                    "UODF",
                )
                print('Saving model...')
            self.global_epoch += 1

            end0 = time.time()
            seconds = end0 -start0

            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("epoch time: %d:%02d:%02d"% (h, m, s))

        print('Best Accuray: %f' % self.best_tst_accuracy)

        plt.plot(range(total_epoch), self.meter['train_acc'], label='Loss')
        plt.grid(True)
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(self.experiment_dir + 'accuracy.png')
        plt.clf()

        logger.info('End of training...')
        logging.shutdown()

def init_dir(args):
    file_name = args.file_name
    dir = args.dir

    experiment_path = create_directory(os.path.join(args.ExpPath, file_name))
    experiment_dir = create_directory(os.path.join(experiment_path,file_name+ "_"+ str(dir) + "/"))
    checkpoints_dir = create_directory(os.path.join(experiment_dir,'checkpoints'))
    log_dir = create_directory(os.path.join(experiment_dir,'logs'))

    return checkpoints_dir,experiment_path,experiment_dir


def init_log(args,experiment_path):
    '''log'''
    logger = logging.getLogger('UODF')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(experiment_path + 'logs/train_%s_' % str(args.model_name)
                                       + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('-------------------------------------------traning-------------------------------------')
    logger.info('Parameter...')
    logger.info(args)
    '''Data loading'''
    logger.info('Load dataset...')
    return logger

def run(args):
    checkpoints_dir,experiment_path,experiment_dir = init_dir(args)
    logger = init_log(args,experiment_dir)

    ray_init_path = os.path.join(args.dataPath, args.file_name,"pts_{}.h5".format(args.dir))

    if not os.path.exists(ray_init_path):
        print("No Points")
        return

    ray_init = load_h5(ray_init_path)
    print("loading...")
    logger.info("The number of  data is : %d", ray_init.shape[0])

    print('-------------UODF Training Start-------------')
    print(f"The number of traning data is :{ray_init.shape[0]}")

    # create model
    model = UodfModel(args=args).cuda()
    # create point sampler
    sampler = DirectionalRaySample(ray_init,args.dir,args.meshPath + args.file_path)
    # create dataset
    dataSet = UodfDataSet(sampler,seed = 0,batch_size=args.batch_size)
    dataSet._sample_base(sample_num=args.train_sample_num)
    # create trainer
    Trainer = UodfTrainer(args.dir,model,dataSet,args,
                          checkpoints_dir=checkpoints_dir,
                          experiment_dir=experiment_dir)
    # train
    Trainer.train(total_epoch=args.epoch,logger=logger,train_metric = args.train_metric)