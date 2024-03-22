import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Embedder import PositionalEmbedding,BaseEmbedding

class UodfModel(nn.Module):
    def __init__(self,args):
        super(UodfModel,self).__init__()
        self.args = args
        self.dim = 256
        self.input_dim = 3

        if args.use_embedder:
            self.embedder = PositionalEmbedding(x_dim=2,level=10)
        else:
            self.embedder = BaseEmbedding(x_dim=2)

        self.input_2D = self.embedder.output_dim()
        self.fc1 = torch.nn.Linear(self.input_dim + self.input_2D,self.dim)
        self.fc2 = torch.nn.Linear(self.dim,self.dim)
        self.fc3 = torch.nn.Linear(self.dim,self.dim)
        self.fc4 = torch.nn.Linear(self.dim,self.dim)
        self.fc5 = torch.nn.Linear(self.dim,self.dim)
        self.fc6 = torch.nn.Linear(self.dim,self.dim)
        self.fc7 = torch.nn.Linear(self.dim,self.dim)
        self.fc8 = torch.nn.Linear(self.dim,self.dim)
        self.fc9 = torch.nn.Linear(self.dim,self.dim)
        self.fc10 = torch.nn.Linear(self.dim,1)


    def forward(self,x,dir):
        if dir == 2:
            pts_2D = x[:,:2]
        elif dir == 1:
            pts_2D = torch.cat((x[:,0].view(-1,1),x[:,2].view(-1,1)),dim = -1)
        elif dir == 0:
            pts_2D = x[:,1:]


        pts_latent2D = self.embedder(pts_2D)

        res = torch.cat((x,pts_latent2D),dim = 1)
        h = F.relu(self.fc1(res))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))
        h = F.relu(self.fc9(h))
        h = self.fc10(h)

        return h,res

class UodfMaskModel(nn.Module):
    def __init__(self,args):
        super(UodfMaskModel,self).__init__()
        self.args = args
        self.dim = 256

        if args.use_embedder:

            self.embedder = PositionalEmbedding(x_dim=2,level=10)
        else:
            self.embedder = BaseEmbedding(x_dim=2)
        self.input_dim = self.embedder.output_dim()

        self.fc1 = torch.nn.Linear(self.input_dim,self.dim)
        self.fc2 = torch.nn.Linear(self.dim,self.dim)
        self.fc3 = torch.nn.Linear(self.dim,1)

    def forward(self,x,dir):
        if dir == 2:
            x = x[:,:2]
        elif dir == 1:
            x = torch.cat((x[:,0].view(-1,1),x[:,2].view(-1,1)),dim = -1)
        elif dir == 0:
            x = x[:,1:]

        x = self.embedder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

