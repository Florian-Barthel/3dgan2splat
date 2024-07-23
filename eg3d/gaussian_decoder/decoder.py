import torch
from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, out_features, apply_scaling_softmax=True):
        super().__init__()
        self.hidden_dim = 64
        self.apply_scaling_softmax = apply_scaling_softmax

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, out_features, lr_multiplier=1)
        )

    def forward(self, sampled_features, xyz):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)
        x = torch.concat([x, xyz], dim=-1)
        x = self.net(x)
        x = x.view(N, M, -1)
        _opacity = x[..., 0:1] + 10
        _features_dc = torch.sigmoid(x[..., 1:4]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF # torch.tanh(x[..., 0:3]) / 2  #
        _rotation = x[..., 4:8]
        _scaling = torch.relu(x[..., 8:11])*-1 - 5
        # if self.apply_scaling_softmax:
        #     _scaling = torch.softmax(_scaling, dim=-1) * torch.sum(_scaling, dim=-1, keepdim=True)
        return {'_features_dc': _features_dc, "_rotation": _rotation, "_scaling": _scaling, "_opacity": _opacity}


class OSGDecoderSingle(torch.nn.Module):
    def __init__(self, n_features, out_features, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(hidden_dim, hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(hidden_dim, out_features)
        )

    def forward(self, sampled_features, xyz=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        N, M, C = x.shape
        x = x.view(N * M, C)
        if xyz is not None:
            x = torch.concat([x, xyz], dim=-1)
        x = self.net(x)
        x = x.view(N, M, -1)
        return x


class OpacityDecoder(torch.nn.Module):
    def __init__(self, n_features, apply_scaling_softmax=True):
        super().__init__()
        self.hidden_dim = 16
        self.apply_scaling_softmax = apply_scaling_softmax

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=1),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=1)
        )

    def forward(self, sampled_features, xyz):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)
        x = torch.concat([x, xyz], dim=-1)
        x = self.net(x)
        x = x.view(N, M, -1)
        _opacity = x[..., 0:1] * 10
        return {'_opacity': _opacity}
