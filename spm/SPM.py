import torch
import torch.nn as nn



import numpy as np


img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, num_inputs, i=0, include_input=True):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': num_inputs,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False, use_layer_norm=True,
                 spatial_softmax=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.spatial_softmax = spatial_softmax

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        fc_input_dim = num_filters * out_dim * out_dim  # TODO: this is huge (39,200)!! Do we want this?
        if spatial_softmax:
            embedding_multires = 10
            positional_encoding, embedding_dim = get_embedder(embedding_multires, num_filters, i=0, include_input=False)
            self.positional_encoding = positional_encoding
            fc_input_dim += embedding_dim * 2
        self.fc = nn.Linear(fc_input_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits
        self.use_layer_norm = use_layer_norm

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        features = conv.reshape(conv.size(0), -1)
        if self.spatial_softmax:
            b, c, h, w  = conv.shape
            conv_flat = conv.flatten(2)
            weights = torch.softmax(conv_flat, dim=-1)
            # Gives positions as [-1,-1,-1, 0,0,0, 1,1,1]
            device = conv.device
            # y_pos = torch.linspace(-1, 1, steps=h).repeat_interleave(w).reshape(1, 1, h * w).to(device)
            # # Gives positions as [-1,0,1, -1,0,1, -1,0,1]
            # x_pos = torch.linspace(-1, 1, steps=w).repeat(h).reshape(1, 1, h * w).to(device)

            # NEW, now with positional encoding
            y_pos = torch.arange(0, h).repeat_interleave(w).reshape(1, 1, h * w).to(device)
            x_pos = torch.arange(0, w).repeat(h).reshape(1, 1, h * w).to(device)

            avg_x_pos = torch.sum(weights * x_pos, dim=-1)
            avg_y_pos = torch.sum(weights * y_pos, dim=-1)
            avg_x_pos = self.positional_encoding(avg_x_pos)
            avg_y_pos = self.positional_encoding(avg_y_pos)

            features = torch.cat([features, avg_x_pos, avg_y_pos], dim=-1)  # TODO: features_dim >> spatial_softmax_dim.  Consider reducing features_dim, repeating spatial_softmax, or concatenating it in later.

        return features

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        if self.use_layer_norm:
            h_norm = self.ln(h_fc)
        else:
            h_norm = h_fc
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, use_layer_norm=True,
    spatial_softmax=False,
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits, use_layer_norm, spatial_softmax,
    )


class SimParamModel(nn.Module):
    def __init__(self, shape, action_space, ):
        self.encoder = make_encoder('pixel', shape, 32, )
        self.positional_encoder, embedding_dim = get_embedder(10, shape, i=0)
