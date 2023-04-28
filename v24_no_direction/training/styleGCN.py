import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from training.networks import *
from einops import repeat, rearrange

def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)
    
class GraphConvolution_style(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_channels, out_channels,w_dim,
        activation='lrelu', 
        resample_filter=[1,3,3,1],
        magnitude_ema_beta = -1, ):

        super(GraphConvolution_style, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.padding = 0
        self.activation = activation
        memory_format = torch.contiguous_format

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
               torch.randn([out_channels, in_channels, 1, 1]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        # self.reset_parameters()
        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))
        self.conv_clamp = None

    def forward(self, x, adj,w,gain=1,up=1,fused_modconv=None):
        assert x.ndim==3
        styles = self.affine(w) 
        act = self.activation
        flip_weight = True #
        # input feature shape b,p,3
        b,p,c = x.shape
        # x = x.view(b*p,c,1,1)
        x = x.permute(0,2,1)
        x =x.view(b,c,p,1)
        # import pdb
        # pdb.set_trace()
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training
        if x.size(0) > styles.size(0):
            styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
        
        # N,C 
        # support = torch.mm(input, self.weight)
        # G A T
        # output = torch.spmm(adj, support)



        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=up,
                padding=self.padding, resample_filter=self.resample_filter, 
                flip_weight=flip_weight, fused_modconv=fused_modconv)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = x.permute(0,2,1,3)
        x = x.view(b,p,self.out_channels)
        # x = batch_mm(adj, x)
        # print(x.shape)
        x = torch.stack([torch.sparse.mm(adj,x[i]) for i in range(b)])
        # print(x.shape)
        x = x.permute(0,2,1)
        x =x.view(b,self.out_channels,p,1)

        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=act, gain=act_gain, clamp=act_clamp)
        x = x.permute(0,2,1,3)
        x = x.view(b,p,self.out_channels)
        
        return x


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class StyleGCN(nn.Module):
    def __init__(self, n_in,n_hid,n_out,w_dim):
        super(StyleGCN, self).__init__()

        self.gc1 = GraphConvolution_style(n_in, 32,w_dim)

        self.gc2 = GraphConvolution_style(32, 32,w_dim)

        self.gc3 = GraphConvolution_style(32, 64,w_dim)

        self.gc4 = GraphConvolution_style(64, 64,w_dim)

        self.gc5 = GraphConvolution_style(64, 128,w_dim)
        self.gc6 = GraphConvolution_style(128, 128,w_dim)

        self.gc7 = GraphConvolution_style(128, 256,w_dim)

        self.gc8 = GraphConvolution_style(256, 512,w_dim)
        self.gc9 = GraphConvolution_style(512, n_out,w_dim)
        self.gc10 = GraphConvolution_style(n_out, n_out,w_dim)
        # self.gc11 = GraphConvolution_style(n_hid, n_hid,w_dim)
        # self.gc12 = GraphConvolution_style(n_hid, n_hid,w_dim)
        # self.gc13 = GraphConvolution_style(n_hid, n_out,w_dim)
        # self.dropout = dropout

    def forward(self, x_in, adj,ws):
        x_in = self.gc1(x_in, adj,ws)
        # x in 32
        x = self.gc2(x_in, adj,ws)
        x = x_in+x
        x_in = x 
        x = self.gc3(x, adj,ws)
        
        x_in = x

        x = self.gc4(x, adj,ws)

        x = x_in + x

        # x = x_in+x
        # x_in = x
        x = self.gc5(x, adj,ws)
        x_in = x

        x = self.gc6(x, adj,ws)
        # x = x_in+x
        x = x_in + x
        # x_in = x
        x = self.gc7(x, adj,ws)
        # x = x_in+x
        # x_in = x
        x = self.gc8(x, adj,ws)
        # x = x_in+x
        # x_in = x
        x = self.gc9(x, adj,ws)
        # x = x_in+x
        # x_in = x

        x = self.gc10(x, adj,ws)
        
        
        return x
