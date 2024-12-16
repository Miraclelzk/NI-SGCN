import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, surrogate
from spikingjelly.activation_based import layer as spiking_layer

from NIIFNode import NIIFNode

class ResnetBlockConv1d_snn(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, c_dim, size_in, mu, sigma, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = spiking_layer.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = spiking_layer.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = spiking_layer.Conv1d(size_in, size_h, 1)
        self.fc_1 = spiking_layer.Conv1d(size_h, size_out, 1)
        self.fc_c = spiking_layer.Conv1d(c_dim, size_out, 1)
        # self.actvn = nn.ReLU()
        # self.actvn = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.actvn = NIIFNode(surrogate_function=surrogate.ATan(), mu=mu, sigma=sigma, detach_reset=True)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = spiking_layer.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)

        return out


class ScoreNet_snn(nn.Module):

    def __init__(self, T, z_dim, dim, out_dim, hidden_size, num_blocks, mu, sigma):
        """
        Args:
            z_dim:   Dimension of context vectors. 
            dim:     Point dimension.
            out_dim: Gradient dim.
            hidden_size:   Hidden states dim.
        """
        super().__init__()
        self.T = T
        self.z_dim = z_dim 
        self.dim = dim 
        self.out_dim = out_dim 
        self.hidden_size = hidden_size #128
        self.num_blocks = num_blocks #4

        dim1 = 16

        self.conv_pp = spiking_layer.Conv1d(dim, dim1, 1)
        self.actvn_out_pp = NIIFNode(surrogate_function=surrogate.ATan(), mu=mu, sigma=sigma, detach_reset=True)

        # Input = Conditional = zdim (code) + dim (xyz)
        c_dim = z_dim + dim1
        self.conv_p = spiking_layer.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockConv1d_snn(c_dim, hidden_size, mu, sigma) for _ in range(num_blocks)
        ])
        self.bn_out = spiking_layer.BatchNorm1d(hidden_size)
        self.conv_out = spiking_layer.Conv1d(hidden_size, out_dim, 1)
        # self.actvn_out = nn.ReLU()
        # self.actvn_out = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.actvn_out = NIIFNode(surrogate_function=surrogate.ATan(), mu=mu, sigma=sigma, detach_reset=True)


    def forward(self, x, c):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim) Shape latent code
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        if self.T != 1:
            p = x.transpose(2, 3)  # (T, bs, K, 3) -> (T, bs, 3, K)
            p = self.actvn_out_pp(self.conv_pp(p))

            _, batch_size, D, num_points = p.size()

            # (T, bs, F) -> (T, bs, F, K)
            c_expand = c.unsqueeze(3).expand(-1, -1, -1, num_points)
            c_xyz = torch.cat([p, c_expand], dim=2)
        else:
            p = x.transpose(1, 2)  # (bs, K, 3) -> (bs, 3, K)
            p = self.actvn_out_pp(self.conv_pp(p))
            batch_size, D, num_points = p.size()

            # (bs, F) -> (bs, F, K)
            c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
            c_xyz = torch.cat([p, c_expand], dim=1) # (bs, F+3, K)

        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz)
        
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(-2, -1)

        return out

