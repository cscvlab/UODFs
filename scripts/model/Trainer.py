import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self,
                 model,
                 train_set,
                 args,
                 **kwargs
                 ):

        # parameters
        self.optimizer_type = args.optimizer
        self.optimizer_lr_init = args.learning_rate
        self.optimizer_decay = args.decay_rate
        self.optimizer_momentum = kwargs.get('optimizer_momentum', 0)
        self.optimizer_betas = kwargs.get('optimizer_betas', (0.9, 0.999))
        self.optimizer_eps = kwargs.get('optimizer_eps', 1e-08)

        self.scheduler_type = kwargs.get('scheduler_type', 'plateau')
        self.scheduler_verbose = kwargs.get('scheduler_verbose', True)
        self.scheduler_expLR_gamma = kwargs.get('scheduler_expLR_gamma', 0.65)
        self.scheduler_plateau_factor = kwargs.get('scheduler_plateau_factor', 0.5)
        self.scheduler_plateau_mode = kwargs.get('scheduler_plateau_mode', 'max')

        # network
        self.net = model
        self.optimizer, self.scheduler = self._create_optimizer_scheduler()

        # Dataset
        self.train_set = train_set

        # Record
        self.history = pd.DataFrame()

        # assist configs
        self.sequence = kwargs.get('sequence', [])
        self.architecture_list = kwargs.get('architecture_list', [])

        self.tensorboard_output = kwargs.get('tensorboard_output', '')
        self.figure_output = kwargs.get('figure_output', '')
        self.recon_output = kwargs.get('recon_output', '')

        self.i_scheduler = kwargs.get('i_scheduler', 1)
        self.i_figure = kwargs.get('i_figure', 10)
        self.i_recon = kwargs.get('i_recon', 10)

        self.tensorboard_items = kwargs.get('tensorboard_items', [])
        self.figure_items = kwargs.get('figure_items', [])

        self.writer = None if self.tensorboard_output == '' else SummaryWriter(self.tensorboard_output)

    def _create_optimizer_scheduler(self):
        # set optimizer
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.optimizer_lr_init,
                momentum=self.optimizer_momentum,
                weight_decay=self.optimizer_decay)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.optimizer_lr_init,
                betas = self.optimizer_betas,
                eps = self.optimizer_eps,
                weight_decay=self.optimizer_decay)
        else:
            raise ValueError('[MLP Generator]Invalid optimizer')

        # Set scheduler

        if self.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='max',
                factor=self.scheduler_plateau_factor,
                min_lr=1e-7,
                verbose=self.scheduler_verbose)
        elif self.scheduler_type == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=self.scheduler_expLR_gamma,
                verbose=self.scheduler_verbose)
        else:
            raise ValueError('Invalid scheduler setting: {}'.format(self.scheduler))


        return optimizer, scheduler

    def _compute_loss(self, pred, y):
        """
        Should be implemented

        Args:
            pred (_type_): predict value
            y (_type_): ground truth value

        Returns:
            Tuple: {'loss': torch.Tensor, ...}

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('Compute Loss haven\'t been implemented. ')

    def _training_step(self, data):
        x, y = data
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()

        pred = self.net(x)
        loss = self._compute_loss(pred, y)['loss']

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _training(self):
        self.net = self.net.train()
        for data in self.train_set:
            loss = self._training_step(data)


    def _update_scheduler(self, i, output):
        if self.i_scheduler < 0:
            return
        if i % self.i_scheduler != 0:
            return
        if i == 0:
            return
        self.scheduler.step(output)

    def _validate(self, i=-1):
        pass

    def _print_to_tensorboard(self, i: int, metrics: dict, key=None):
        if not self.writer is None:
            if key == None:
                for key, value in metrics.items():
                    if key in self.tensorboard_items:
                        self.writer.add_scalar(key, value, i)
            else:
                self.writer.add_scalar(key, metrics[key], i)

    def _print_to_figure(self, i=-1):
        if i != -1 or i % self.i_figure != 0:
            return
        if self.figure_output == '':
            return

        for key in self.figure_items:
            plt.plot(range(len(self.history)), self.history[key])
            plt.title(key)
            plt.savefig(os.path.join(self.figure_output, '{}.jpg'.format(key)))
            plt.cla()

    def train(self, epoch, **kwargs):
        raise NotImplementedError('Method \"train\" is not implemented.')
