import torch
import torch.nn as nn

# Positional encoding
class Embedder:

    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda  x:x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2'] # L-1
        N_freqs = self.kwargs['num_freqs'] # L

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0.,max_freq ,N_freqs)
        else:
            freq_bands = torch.linspace(2.**0 , 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda  x,p_fn = p_fn,freq = freq :
                                 p_fn(x *freq))
                out_dim +=d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)

def get_embedder(multires):

    embed_kwargs = {
        'include_input':True,
        'input_dims':3,
        'max_freq_log2': multires-1,
        'num_freqs':multires,
        'log_sampling':True,
        'periodic_fns': [torch.sin,torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo = embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim

def batchify(fn, chunk):
    '''Consturcts a version of 'fn' that applies to smaller batches'''
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0,inputs.shape[0],chunk)],0)
    return ret


class BaseEmbedding(nn.Module):
    def __init__(self, x_dim: int = 3) -> None:
        super(BaseEmbedding, self).__init__()
        self.x_dim = x_dim

    def forward(self, x):
        return x

    def output_dim(self):
        return self.x_dim

    def __str__(self):
        return 'BaseEmbedding: x_dim={}'.format(self.x_dim)


class PositionalEmbedding(BaseEmbedding):
    def __init__(self,
                 x_dim: int = 3,
                 level: int = 6,
                 include_input: bool = True) -> None:
        super().__init__(x_dim)
        self.level = level
        self.include_input = include_input

        # [1, 2, 4, 8, 16]
        mscales = torch.Tensor([2 ** i for i in range(self.level)])
        self.mscales = mscales.repeat(self.x_dim, 1)

    def forward(self, x: torch.Tensor):
        res = [x] if self.include_input else []

        for l in range(self.level):
            feat = x * (2 ** l)
            res += [torch.sin(feat), torch.cos(feat)]

        res = torch.cat(res, dim=1)
        return res

    def output_dim(self):
        output_shape = self.x_dim * 2 * self.level
        if self.include_input:
            output_shape += self.x_dim
        return output_shape

    def __str__(self):
        return 'Positional Embedding: x_dim={}, level={}, include_input={}, output_dim={}'.format(self.x_dim,
                                                                                                  self.level,
                                                                                                  self.include_input,
                                                                                                  self.output_dim())
