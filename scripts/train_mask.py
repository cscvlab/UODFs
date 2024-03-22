import os
import torch
import datetime
import logging
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from model.UodfModel import UodfMaskModel
from model.Trainer import BaseTrainer
from model.UodfDataSet import UodfMaskDataSet,MaskRaySample
from utils.utils import create_directory,save_checkpoint


class UodfMaskTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 dir,
                 model:UodfMaskModel,
                 train_set:UodfMaskDataSet,
                 **kwargs
                 ):
        super().__init__(model, train_set, args, **kwargs)
        self.dir = dir
        self.loss_rate = 1e3
        self.num = 0

        self.global_epoch = 0
        self.best_tst_accuracy = 1000
        self.meter = defaultdict(list)
        self.checkpoints_dir = kwargs.get('checkpoints_dir', 'None')
        self.experiment_dir = kwargs.get('experiment_dir', 'None')

    def _training_step(self, data):
        pts, y = data
        pts, y = pts.cuda(), y.cuda()

        self.optimizer.zero_grad()

        y_hat = self.net(pts, self.dir)

        # compute loss
        loss_dict = self._compute_loss(y_hat, y)
        loss = loss_dict['loss']

        loss.backward()

        self.optimizer.step()
        # compute num
        y_hat = torch.where(y_hat >= 0.5, torch.ones_like(y_hat), torch.zeros_like(y_hat))
        y_num = torch.where(y > 0)[0].shape[0]
        y_hat_num = torch.where(y_hat > 0)[0].shape[0]
        if (y_num != y_hat_num):
            self.num += abs(y_num - y_hat_num)

        return {'loss': loss.detach().item()}

    def _training(self):
        self.net = self.net.train()
        for data in self.train_set:
            loss_dict = self._training_step(data)

        return loss_dict

    def _compute_loss(self, y_hat, y):
        loss = F.binary_cross_entropy(y_hat.view(-1, 1), y.view(-1, 1)) * self.loss_rate
        return {"loss": loss}

    def _validate(self, i=-1):
        self.net = self.net.eval()
        total_y_hat = []
        total_y = []

        for data in self.train_set:
            pts, y = data
            pts, y = pts.cuda(), y.cuda()
            y_hat = self.net(pts, self.dir)

            total_y_hat.append(y_hat.detach())
            total_y.append(y.detach())

        total_y_hat = torch.cat(total_y_hat, dim=0)
        total_y = torch.cat(total_y, dim=0)
        loss_dict = self._compute_loss(total_y_hat, total_y)
        acc = loss_dict["loss"] / self.loss_rate
        return acc

    def train(self, total_epoch, **kwargs):
        logger = kwargs.get('logger', 'None')
        train_metric = kwargs.get('train_metric', 'True')

        self.net = self.net.cuda()
        task = tqdm(range(total_epoch), colour='green')

        for epoch_i in task:
            self.num = 0
            loss_dict = self._training()
            acc = self._validate(epoch_i)
            self._update_scheduler(epoch_i, acc)

            log_str = "epoch:{} loss: {:6f} ".format(
                epoch_i,
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
                # print('Saving model...')
            self.global_epoch += 1

            print("epoch:{} num:{}".format(epoch_i, self.num))
            self.num = 0

        print('Best Accuray: %f' % self.best_tst_accuracy)

        # plt.plot(range(total_epoch), self.meter['train_acc'], label='Loss')
        # plt.grid(True)
        # plt.title('Accuracy Curve')
        # plt.legend()
        # plt.savefig(self.experiment_dir + '/accuracy.png')
        # plt.clf()

        logger.info('End of training...')
        logging.shutdown()

def init_dir(args):
    file_name = args.file_name
    dir = args.dir

    experiment_path = create_directory(os.path.join(args.ExpPath, file_name))
    experiment_dir = create_directory(os.path.join(experiment_path, file_name + "_mask_" + str(dir) + "/"))
    checkpoints_dir = create_directory(os.path.join(experiment_dir,'checkpoints'))
    log_dir = create_directory(os.path.join(experiment_dir,'logs'))

    return checkpoints_dir, experiment_path, experiment_dir

def init_log(args,experiment_path):
    '''log'''
    logger = logging.getLogger('UODF_MASK')
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

def run_mask(args):
    checkpoints_dir,experiment_path,experiment_dir = init_dir(args)
    logger = init_log(args,experiment_dir)

    args.epoch = 100
    args.batch_size = 1024
    # create model
    model = UodfMaskModel(args = args).cuda()

    # create sampler
    sampler = MaskRaySample(args.train_res,args.dir,args.meshPath + args.file_path)
    pts,y = sampler.sampler()

    # create dataset

    MaskDataSet = UodfMaskDataSet(pts,y,batch_size = args.batch_size)


    # create trainer
    MaskTrainer = UodfMaskTrainer(args,args.dir,model,MaskDataSet,
                          checkpoints_dir=checkpoints_dir,
                          experiment_dir=experiment_dir)


    MaskTrainer.train(total_epoch=args.epoch,logger=logger,train_metric = args.train_metric)
