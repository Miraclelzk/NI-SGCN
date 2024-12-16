import torch
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
from .utils_Noisy import *


def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k+offset)
    return knn_idx[:, :, offset:]


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper
        
    def extra_repr(self):
        return super().extra_repr() + f', oper={self.oper}'

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class DenseEdgeConv(Module):

    def __init__(self, in_channels, num_fc_layers, growth_rate, knn=16, aggr='max', activation='relu', relative_feat_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers #L
        self.growth_rate = growth_rate  # c
        self.relative_feat_only = relative_feat_only

        # Densely Connected Layers
        if relative_feat_only:
            self.layer_first = FCLayer(in_channels, growth_rate, bias=True, activation=activation)
        else:
            self.layer_first = FCLayer(3*in_channels, growth_rate, bias=True, activation=activation)
        self.layer_last = FCLayer(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers.append(FCLayer(in_channels + i * growth_rate, growth_rate, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = knn_group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        if self.relative_feat_only:
            edge_feat = knn_feat - x_tiled
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y
    

class DenseEdgeConv_snn_NoisySAN(Module):

    def __init__(self, T, in_channels, num_fc_layers, growth_rate, knn=16, aggr='max', activation='NCLIF', relative_feat_only=False, mu=0.0, sigma=0.2):
        super().__init__()
        self.T = T
        self.in_channels = in_channels
        self.knn = knn
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers #L
        self.growth_rate = growth_rate  # c
        self.relative_feat_only = relative_feat_only
        self.mu=mu
        self.sigma=sigma

        # Densely Connected Layers
        if relative_feat_only:
            self.layer_first = FCLayer_snn(in_channels, growth_rate, mu, sigma, bias=True, activation=activation)
        else:
            self.layer_first = FCLayer_snn(3*in_channels, growth_rate, mu, sigma, bias=True, activation=activation)
        self.layer_last = FCLayer_snn(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, mu, sigma, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers.append(FCLayer_snn(in_channels + i * growth_rate, growth_rate, mu, sigma, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (TB, N, d)
        :param  knn_idx:    (TB, N, K)
        :return (TB, N, K, 2*d)
        """
        knn_feat = knn_group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        if self.relative_feat_only:
            edge_feat = knn_feat - x_tiled
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (T, B, N, d)
        :return (T, B, N, d+L*c)
        """
        # snn
        if self.T != 1:
            x = x.flatten(0, 1)
            pos = pos.flatten(0, 1)

        knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)

        # snn (TB, N, K, 2*d)->(T, B, N, K, 2*d)
        if self.T != 1:
            TB, NN, KK, DD = edge_feat.shape
            edge_feat = edge_feat.reshape(self.T, -1, NN, KK, DD)
            TB, NN, DD = x.shape
            x = x.reshape(self.T, -1, NN, DD)

            y = torch.cat([
            self.layer_first(edge_feat),              # (T, B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, 1, self.knn, 1) # (T, B, N, K, d)
            ], dim=-1)  # (T, B, N, K, d+c)
        else:
            y = torch.cat([
                self.layer_first(edge_feat),              # (B, N, K, c)
                x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
            ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (T, B, N, K, c)
                y,                  # (T, B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (T, B, N, K, c)
            y                   # (T, B, N, K, d+(L-1)*c)
        ], dim=-1)  # (T, B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y

class FeatureExtraction_snn_Noisy(Module):

    def __init__(self, 
        T=4,
        in_channels=3, 
        dynamic_graph=True, 
        conv_channels=24, 
        num_convs=4, 
        conv_num_fc_layers=3, 
        conv_growth_rate=12, 
        conv_knn=16, 
        conv_aggr='mean', 
        activation='NIIF',
        mu=0.0, 
        sigma=0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_graph = dynamic_graph
        self.num_convs = num_convs

        self.T = T
        self.mu=mu
        self.sigma=sigma

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        for i in range(num_convs):
            if i == 0:
                trans = FCLayer_snn(in_channels, conv_channels, mu, sigma, bias=True, activation=None)
                conv = DenseEdgeConv_snn_NoisySAN(
                    self.T,
                    conv_channels, 
                    num_fc_layers=conv_num_fc_layers, 
                    growth_rate=conv_growth_rate, 
                    knn=conv_knn, 
                    aggr=conv_aggr, 
                    activation=activation,
                    relative_feat_only=True,
                    mu=mu,
                    sigma=sigma
                )
            else:
                trans = FCLayer_snn(in_channels, conv_channels, mu, sigma, bias=True, activation=activation)
                conv = DenseEdgeConv_snn_NoisySAN(
                    self.T,
                    conv_channels, 
                    num_fc_layers=conv_num_fc_layers, 
                    growth_rate=conv_growth_rate, 
                    knn=conv_knn, 
                    aggr=conv_aggr, 
                    activation=activation,
                    relative_feat_only=False,
                    mu=mu,
                    sigma=sigma
                )
            self.transforms.append(trans)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):


        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, x)
        return x

    def static_graph_forward(self, pos):
        x = pos
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, pos)
        return x 

    def forward(self, x):
        if self.dynamic_graph:
            return self.dynamic_graph_forward(x)
        else:
            return self.static_graph_forward(x)
