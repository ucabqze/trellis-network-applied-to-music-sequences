from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F

__author__ = 'shaojieb'

##############################################################################################################
#
# Temporal DropConnect in a feed-forward setting
#
##############################################################################################################

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, temporal=False):
        """
        Weight DropConnect, adapted from a recurrent setting by Merity et al. 2017
        :param module: The module whose weights are to be applied dropout on
        :param weights: A 2D list identifying the weights to be regularized. Each element of weights should be a
                        list containing the "path" to the weight kernel. For instance, if we want to regularize
                        module.layer2.weight3, then this should be ["layer2", "weight3"].
        :param dropout: The dropout rate (0 means no dropout)
        :param temporal: Whether we apply DropConnect only to the temporal parts of the weight (empirically we found
                         this not very important)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.temporal = temporal
        self._setup()

    def _setup(self):
        for path in self.weights:
            full_name_w = '.'.join(path)

            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            w = getattr(module, name_w)
            del module._parameters[name_w]
            module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            raw_w = getattr(module, name_w + '_raw')

            if len(raw_w.size()) > 2 and raw_w.size(2) > 1 and self.temporal:
                # Drop the temporal parts of the weight; if 1x1 convolution then drop the whole kernel
                w = torch.cat([F.dropout(raw_w[:, :, :-1], p=self.dropout, training=self.training),
                               raw_w[:, :, -1:]], dim=2)
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)

            setattr(module, name_w, w)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module.forward(*args, **kwargs)


##############################################################################################################
#
# Embedding dropout
#
##############################################################################################################

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """
    Apply embedding encoder (whose weight we apply a dropout)
    :param embed: The embedding layer
    :param words: The input sequence
    :param dropout: The embedding weight dropout rate
    :param scale: Scaling factor for the dropped embedding weight
    :return: The embedding output
    """
    ### ntoken = 10000
    ### ninp = 400
    ### embed = nn.Embedding(ntoken, ninp)
    ### encoder.weight.data
    
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1  ### padding the last words?

    # Handle PyTorch issue
    if '0.3' not in torch.__version__:
        X = F.embedding(
            words, masked_embed_weight,
            padding_idx,
            embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
    else:
        X = embed._backend.Embedding.apply(words, masked_embed_weight,
                                           padding_idx, embed.max_norm, embed.norm_type,
                                           embed.scale_grad_by_freq, embed.sparse
                                           )
    return X



##############################################################################################################
#
# Variational dropout (for input/output layers, and for hidden layers)
#
##############################################################################################################
    

### L: number of layer
### C: Channels x (layer width, i.e. number if channels in each hidden layer)
### N: timestep???
### M: M-truncated 


class VariationalDropout(nn.Module):
    def __init__(self):
        """
        Feed-forward version of variational dropout that applies the same mask at every time step
        """
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=0.5, dim=3):
        if not self.training or not dropout:
            return x
        if dim == 4:
            # Dimension (M, N, L, C), where C stands for channels
            m = x.data.new(x.size(0), x.size(1), 1, x.size(3)).bernoulli_(1 - dropout)
        else:
            # Dimension (N, L, C)
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
         
        ### why with torch.no_grad() ???    
        with torch.no_grad():
            mask = m / (1 - dropout)
            mask = mask.expand_as(x)
            
        ### x = torch.randn(3, requires_grad=True)
        ### print(x.requires_grad)
        ### print((x ** 2).requires_grad)
        
        ### with torch.no_grad():
        ###     print((x ** 2).requires_grad)
        
        ### True
        ### True
        ### False
            
        return mask * x


class VariationalHidDropout(nn.Module):
    def __init__(self, dropout=0.0):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer of TrellisNet
        :param dropout: The dropout rate (0 means no dropout is applied)
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def reset_mask(self, x):
        dropout = self.dropout

        # Dimension (N, C, L)
        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - dropout)
        with torch.no_grad():
            mask = m / (1 - dropout)
            self.mask = mask   
        return mask
    
    

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        
        assert self.mask is not None, "You need to reset mask before using VariationalHidDropout"
        
        ### if not expression:
        ###     raise AssertionError
            
        ### assert expression
        
        mask = self.mask.expand_as(x)  # Make sure the dimension matches
        return mask * x



##############################################################################################################
#
# Weight normalization. Modified from the original PyTorch's implementation of weight normalization.
#
##############################################################################################################

def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(object):
    def __init__(self, names, dim):
        """
        Weight normalization module
        :param names: The list of weight names to apply weightnorm on
        :param dim: The dimension of the weights to be normalized
        """
        self.names = names
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim)) ###  g/||v|| * v

    @staticmethod
    def apply(module, names, dim):
        fn = WeightNorm(names, dim)

        for name in names:
            weight = getattr(module, name)

            # remove w from parameter list
            del module._parameters[name]

            # add g and v as new parameters and express w as g/||v|| * v
            module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
            module.register_parameter(name + '_v', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.names:
            weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        # Typically, every time the module is called we need to recompute the weight. However,
        # in the case of TrellisNet, the same weight is shared across layers, and we can save
        # a lot of intermediate memory by just recomputing once (at the beginning of first call).
        pass


def weight_norm(module, names, dim=0):
    fn = WeightNorm.apply(module, names, dim)
    return module, fn






### Note of Qiqi ###

# 1. What does 'super' do in Python?: https://stackoverflow.com/questions/222877/what-does-super-do-in-python
#    
# 2. _single_leading_underscore: weak "internal use" indicator. E.g. from M import * does not import objects whose names start with an underscore.
#   single_trailing_underscore_: used by convention to avoid conflicts with Python keyword
#   _double_leading_underscore: when naming a class attribute, invokes name mangling (inside class FooBar, __boo becomes _FooBar__boo; see below).
#   __double_leading_and_trailing_underscore__: "magic" objects or attributes that live in user-controlled namespaces. 