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


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)



class SimParamModel(nn.Module):
    def __init__(self, shape, action_space, layers, units, device, obs_shape, encoder_type,
                 encoder_feature_dim, encoder_num_layers, encoder_num_filters, sim_param_lr=1e-3,
                 sim_param_beta=0.9,
                 dist='binary', act=nn.ELU, batch_size=32, traj_length=200, num_frames=10,
                 embedding_multires=10, use_img=False, state_dim=0, separate_trunks=False, param_names=[],
                 train_range_scale=1, prop_train_range_scale=False, clip_positive=False, dropout=0.5,
                 initial_range=None, single_window=False, share_encoder=False, normalize_features=True,
                 use_layer_norm=False, use_weight_init=False, frame_skip=1, spatial_softmax=False):
        super(SimParamModel, self).__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self.device = device
        self.encoder_type = encoder_type
        self.batch = batch_size
        self.traj_length = traj_length
        self.encoder_feature_dim = encoder_feature_dim
        self.num_frames = num_frames
        self.use_img = use_img
        self.separate_trunks = separate_trunks
        positional_encoding, embedding_dim = get_embedder(embedding_multires, shape, i=0)
        self.positional_encoding = positional_encoding
        pred_sim_param_dim = 0 if dist == 'normal' else embedding_dim
        self.param_names = param_names
        self.train_range_scale = train_range_scale
        self.prop_train_range_scale = prop_train_range_scale
        self.clip_positive = clip_positive
        self.initial_range = initial_range
        self.single_window = single_window
        self.share_encoder = share_encoder
        self.normalize_features = normalize_features
        self.use_layer_norm = use_layer_norm
        self.use_weight_init = use_weight_init
        self.feature_norm = torch.FloatTensor([-1])[0]
        self.frame_skip = frame_skip
        action_space_dim = np.prod(action_space.shape)

        if self.use_img:
            if self.share_encoder:
                encoder_dims = encoder_feature_dim * num_frames
            else:
                encoder_dims = encoder_feature_dim
            trunk_input_dim = encoder_dims + pred_sim_param_dim + (state_dim  * num_frames) + action_space_dim * num_frames * frame_skip
        else:
            trunk_input_dim = pred_sim_param_dim + (state_dim * num_frames) + action_space_dim * num_frames * frame_skip

        # If each sim param has its own trunk, create a separate trunk for each
        num_sim_params = shape
        if separate_trunks:
            trunk_list = []
            for _ in range(num_sim_params):
                trunk = []
                trunk.append(nn.Linear(trunk_input_dim, self._units))
                trunk.append(self._act())
                for index in range(self._layers - 1):
                    trunk.append(nn.Linear(self._units, self._units))
                    trunk.append(self._act())
                    trunk.append(nn.Dropout(p=dropout))
                trunk.append(nn.Linear(self._units, 1))
                trunk_list.append(nn.Sequential(*trunk).to(self.device))
            self.trunk = torch.nn.ModuleList(trunk_list)
        else:
            trunk = []
            trunk.append(nn.Linear(trunk_input_dim, self._units))
            trunk.append(self._act())
            for index in range(self._layers - 1):
                trunk.append(nn.Linear(self._units, self._units))
                trunk.append(self._act())
                trunk.append(nn.Dropout(p=dropout))
            trunk.append(nn.Linear(self._units, num_sim_params))
            self.trunk = nn.Sequential(*trunk).to(self.device)

        if self.use_img:
            # change obs_shape to account for the trajectory length, since images are stacked channel-wise
            c, h, w = obs_shape
            if self.share_encoder:
                obs_shape = (3, h, w)
                self.encoder = make_encoder(
                    encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
                    encoder_num_filters, output_logits=True, use_layer_norm=self.use_layer_norm,
                    spatial_softmax=spatial_softmax,
                ).to(self.device)
            else:
                obs_shape = (3 * self.num_frames, h, w)
                self.encoder = make_encoder(
                    encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
                    encoder_num_filters, output_logits=True, use_layer_norm=self.use_layer_norm,
                    spatial_softmax=spatial_softmax,
                ).to(self.device)

        if self.use_weight_init:
            self.apply(weight_init)

        parameters = list(self.trunk.parameters())
        if self.use_img:
            parameters += list(self.encoder.parameters())
        self.sim_param_optimizer = torch.optim.Adam(
            parameters, lr=sim_param_lr, betas=(sim_param_beta, 0.999)
        )

    def get_features(self, obs_traj):
        # Share encoder case
        if obs_traj[0][0].shape[0] == 9:
            input = []
            for i in range(len(obs_traj[0])):
                input.append(
                    torch.FloatTensor([traj[i].detach().cpu().numpy() for traj in obs_traj]).to(self.device))
        else:
            input = torch.stack([torch.cat([o for o in traj], dim=0) for traj in obs_traj], dim=0)

        if self.share_encoder:
            features = [self.encoder(img, detach=True) for img in input]
            features = torch.cat(features, dim=1)
        else:
            # Don't update the conv layers if we're sharing, otherwise to
            features = self.encoder(input, detach=self.share_encoder)
        self.feature_norm = torch.norm(features).detach()
        if self.normalize_features:
            features = features / torch.norm(features).detach()
        return features

    def forward(self, full_traj):
        """ obs traj list of lists, pred labels is array [B, num_sim_params] """
        # Turn np arrays into tensors   
        

        if type(full_traj[0][0][0]) is np.ndarray:
            full_traj = [[ (torch.FloatTensor(o).to(self.device),  torch.FloatTensor(s).to(self.device), torch.FloatTensor(a).to(self.device)) for o, s, a in traj]  for traj in full_traj]

        full_state_traj = []
        full_action_traj = []
        full_obs_traj = []

        for traj in full_traj:
            # TODO: previously, we were always taking the first window.  Now, we always take a random one.
            #   We could consider choosing multiple, or choosing a separate segmentation for each batch element.
            if self.single_window or len(traj) == self.num_frames * self.frame_skip:
                index = 0
            else:
                if len(traj) <= self.num_frames * self.frame_skip:
                    continue
                index = np.random.choice(len(traj) - self.num_frames * self.frame_skip + 1)

            if len(traj) != self.num_frames * self.frame_skip:
                traj = traj[index: index + self.num_frames * self.frame_skip]
            obs_traj, state_traj, action_traj = zip(*traj)

            obs_traj = obs_traj[::self.frame_skip]
            state_traj = state_traj[::self.frame_skip]
            state_traj = torch.Tensor(state_traj).to(self.device)
            action_traj = torch.Tensor(action_traj).to(self.device)

            full_state_traj.append(torch.cat(state_traj))
            full_action_traj.append(torch.cat(action_traj))
            # If we're using images, only use the first of the stacked frames
            if self.use_img and not self.share_encoder:
                full_obs_traj.append([o[:3] for o in obs_traj])
            else:
                full_obs_traj.append(obs_traj)


        if self.use_img:
            feat = self.get_features(full_obs_traj)
            fake_pred = torch.cat([feat.squeeze(), full_action_traj[0], full_state_traj[0]], dim=-1)
        else:
            fake_pred = torch.cat([full_action_traj[0], full_state_traj[0]], dim=-1)

        if self.separate_trunks:
            x = torch.cat([trunk(fake_pred) for trunk in self.trunk], dim=-1)
        else:
            x = self.trunk(fake_pred)
        return torch.exp(x)

    def forward_classifier(self, full_traj, pred_labels, step=5):
        """ obs traj list of lists, pred labels is array [B, num_sim_params] """
        # Turn np arrays into tensors
        if type(full_traj[0][0][0]) is np.ndarray:
            full_traj = [[(torch.FloatTensor(o).to(self.device),
                           torch.FloatTensor(s).to(self.device),
                           torch.FloatTensor(a).to(self.device))
                          for o, s, a in traj]
                         for traj in full_traj]

        full_obs_traj = []
        full_state_traj = []
        full_action_traj = []
        for traj in full_traj:
            # TODO: previously, we were always taking the first window.  Now, we always take a random one.
            #   We could consider choosing multiple, or choosing a separate segmentation for each batch element.
            if self.single_window or len(traj) == self.num_frames * self.frame_skip:
                index = 0
            else:
                if len(traj) <= self.num_frames * self.frame_skip:
                    continue
                index = np.random.choice(len(traj) - self.num_frames * self.frame_skip + 1)

            if len(traj) != self.num_frames * self.frame_skip:
                traj = traj[index: index + self.num_frames * self.frame_skip]
            obs_traj, state_traj, action_traj = zip(*traj)

            obs_traj = obs_traj[::self.frame_skip]
            state_traj = state_traj[::self.frame_skip]

            state_traj = tuple(torch.Tensor(state).to(self.device) for state in state_traj)
            action_traj = tuple(torch.Tensor(action).to(self.device) for action in action_traj)

            full_state_traj.append(torch.cat(state_traj))
            full_action_traj.append(torch.cat(action_traj))
            # If we're using images, only use the first of the stacked frames
            if self.use_img and not self.share_encoder:
                full_obs_traj.append([o[:3] for o in obs_traj])
            else:
                full_obs_traj.append(obs_traj)

        # normalize [-1, 1]

        if isinstance(pred_labels, np.ndarray) or isinstance(pred_labels, list):
            pred_labels = torch.FloatTensor(pred_labels).to(self.device)


        encoded_pred_labels = self.positional_encoding(pred_labels)  # 513, 2709 <- 513, 129
        full_action_traj = torch.stack(full_action_traj)
        full_state_traj = torch.stack(full_state_traj)
        B_label = len(pred_labels)
        B_traj = len(full_traj)
        assert B_label == 1 or B_traj == 1

        if self.use_img:
            feat = self.get_features(full_obs_traj)
            h1 = encoded_pred_labels.repeat(B_traj, 1)
            h2 = feat.repeat(B_label, 1)
            h3 = full_action_traj.repeat(B_label, 1)
            h4 = full_state_traj.repeat(B_label, 1)
            fake_pred = torch.cat([h1, h2, h3, h4], dim=-1)
        else:
            fake_pred = torch.cat([encoded_pred_labels.repeat(B_traj, 1),
                                   full_action_traj.repeat(B_label, 1),
                                   full_state_traj.repeat(B_label, 1)], dim=-1)

        if self.separate_trunks:
            x = torch.cat([trunk(fake_pred) for trunk in self.trunk], dim=-1)
        else:
            x = self.trunk(fake_pred)
        pred_class = torch.distributions.bernoulli.Bernoulli(logits=x)
        pred_class = pred_class.mean
        return pred_class

    def train_classifier(self, obs_traj, sim_params, distribution_mean):
        if self.initial_range is not None:
            dist_range = self.initial_range
        elif self.prop_train_range_scale:
            dist_range = self.train_range_scale * torch.FloatTensor(distribution_mean)
        else:
            dist_range = self.train_range_scale
        sim_params = torch.FloatTensor(sim_params)  # 1 - dimensional
        eps = 1e-3
        if self.clip_positive:
            low_val = torch.clamp(sim_params - dist_range, eps, float('inf'))
        else:
            low_val = sim_params - dist_range

        num_low = np.random.randint(0, self.batch * 4)
        low = torch.FloatTensor(
            np.random.uniform(size=(num_low, len(sim_params)), low=low_val,
                              high=sim_params)).to(self.device)

        high = torch.FloatTensor(
            np.random.uniform(size=(self.batch * 4 - num_low, len(sim_params)),
                              low=sim_params,
                              high=sim_params + dist_range)).to(self.device)
        dist_mean = torch.FloatTensor(distribution_mean).unsqueeze(0).to(self.device)
        fake_pred = torch.cat([low, high, dist_mean], dim=0)
        labels = (fake_pred > sim_params.unsqueeze(0).to(self.device)).long()

        # Shuffle all params but the last, which is distribution mean
        shuffled_indices = torch.stack([torch.randperm(len(fake_pred) - 1) for _ in range(len(sim_params))], dim=1).to(
            self.device)
        dist_mean_indices = torch.zeros(1, len(distribution_mean)).to(self.device).long() + len(shuffled_indices)
        shuffled_indices = torch.cat([shuffled_indices, dist_mean_indices])

        labels = torch.gather(labels, 0, shuffled_indices)
        fake_pred = torch.gather(fake_pred, 0, shuffled_indices)

        pred_class = self.forward_classifier([obs_traj], fake_pred)
        pred_class_flat = pred_class.flatten().unsqueeze(0).float()
        labels_flat = labels.flatten().unsqueeze(0).float()
        loss = nn.BCELoss()(pred_class_flat, labels_flat)

        full_loss = nn.BCELoss(reduction='none')(pred_class.float(), labels.float()).detach().cpu().numpy()
        accuracy = np.mean((torch.round(pred_class) == labels).float().detach().cpu().numpy())
        error = (pred_class - labels).float().detach().cpu().numpy()

        #return loss, (full_loss, accuracy, error, self.feature_norm.detach().cpu().numpy())
        return loss, accuracy


    def update(self, obs_list, sim_params, dist_mean):
        if self._dist == 'normal':
            loss = torch.nn.MSELoss()(self.forward([obs_list]), torch.FloatTensor(sim_params).to(self.device))
            self.sim_param_optimizer.zero_grad()
            loss.backward()
            self.sim_param_optimizer.step()
            return loss.cpu().item()
        else:
            loss, accuracy = self.train_classifier(obs_list, sim_params, dist_mean)
            self.sim_param_optimizer.zero_grad()
            loss.backward()
            self.sim_param_optimizer.step()
            return loss.cpu().item(), accuracy
            

    def save(self, model_dir, step):
        torch.save(
            self.state_dict(), '%s/sim_param_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.sim_param_optimizer.state_dict(), '%s/sim_param_optimizer_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.load_state_dict(
            torch.load('%s/sim_param_%s.pt' % (model_dir, step))
        )
        self.sim_param_optimizer.load_state_dict(
            torch.load('%s/sim_param_optimizer_%s.pt' % (model_dir, step))
        )



def predict_sim_params(sim_param_model, traj, current_sim_params, step=10, confidence_level=.3):
    segment_length = sim_param_model.num_frames
    windows = []
    index = 0
    while index <= len(traj) - segment_length * 1:
        windows.append(traj[index: index + segment_length * 1])
        index += step

    with torch.no_grad():
        preds = sim_param_model.forward_classifier(windows, current_sim_params).cpu().numpy()
        mask = (preds > confidence_level) & (preds < 1 - confidence_level)
        preds = np.round(preds)
        preds[mask] = 0.5
    confident_preds = np.mean(preds, axis=0)

    # Round to the nearest integer so each prediction is voting up or down
    # Alternatively, we could just take a mean of their probabilities
    # The only difference is whether we want to give each confident segment equal weight or not
    # And whether we want to be confident (e.g. if all windows predict .6, do we predict .6 or 1?
    return confident_preds










def update_sim_params(sim_param_model, sim_env, obs, step, L, real_dr_list, real_dr_params, updates):
    with torch.no_grad():
        current_sim_params = torch.FloatTensor(sim_env.distribution_mean).unsqueeze(0)
        pred_sim_params = []
        if len(obs) > 1:
            for ob in obs:
                pred_sim_params.append(predict_sim_params(sim_param_model, ob, current_sim_params))
        else:
            pred_sim_params.append(predict_sim_params(sim_param_model, obs[0], current_sim_params))
        pred_sim_params = np.mean(pred_sim_params, axis=0)

    updates = []
    for i, param in enumerate(real_dr_list):
        prev_mean = sim_env.dr[param]

        try:
            pred_mean = pred_sim_params[i]
        except:
            pred_mean = pred_sim_params
        alpha = 0.1

        new_update = - alpha * (np.mean(pred_mean) - 0.5)
        new_mean = prev_mean + new_update
        updates.append(new_update)

        new_mean = max(new_mean, 1e-3)
        sim_env.dr[param] = new_mean      
        print("NEW MEAN", param, new_mean, step, pred_mean, "!" * 30)
       
        real_dr_param = real_dr_params[param]
        if not np.mean(real_dr_param) == 0:
            sim_param_error = (new_mean - real_dr_param) / real_dr_param
        else:
            sim_param_error = new_mean - real_dr_param
        L.log(f'eval/agent-sim_param/{param}/sim_param_error', sim_param_error, step)

        # Update range scale, if necessary
        # proportion_through_training = float(step / args.num_train_steps)
        # start = args.anneal_range_scale_start
        # end = args.anneal_range_scale_end
        # new_range_scale = start * (1 - proportion_through_training) + end * proportion_through_training
        # print("Updating range scale to", new_range_scale)
        # sim_env.set_range_scale(new_range_scale)
        # sim_param_model.train_range_scale = new_range_scale

   
   
def evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim, num_episodes, L, step, args, use_policy,
             update_distribution, training_phase):
    
    def run_eval_loop(sample_stochastically=False):

    
        prefix = 'stochastic_' if sample_stochastically else ''
        obs_batch = []
        real_sim_params = real_env.reset()['sim_params']

        for i in range(num_episodes):


            obs_dict = real_env.reset()
            
            obs_traj = []


            while not done and len(obs_traj) < args.time_limit:

                # Collect trajectories from real_env
                
                obs_img = obs_dict['image']
                # center crop image

                action = agent.select_action(obs_img)
                obs_traj.append((obs_img, obs_state, action))
                obs_dict, reward, done, _ = real_env.step(action)
                video_real.record(real_env)
                episode_reward += reward

            video_real.save(f'real_{training_phase}_{step}.mp4')
            if log_reward:
                L.log('eval/' + prefix + f'episode_reward', episode_reward, step)
            if 'success' in obs_dict.keys():
                if log_reward:
                    L.log('eval/' + prefix + 'episode_success', obs_dict['success'], step)
                all_ep_success.append(obs_dict['success'])
            all_ep_rewards.append(episode_reward)
            obs_batch.append(obs_traj)


        if (not args.outer_loop_version == 0) and update_distribution:
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            evaluate_sim_params(sim_param_model, args, obs_batch, step, L, "test", real_sim_params,
                                current_sim_params)
            update_sim_params(sim_param_model, sim_env, args, obs_batch, step, L)

        filename = args.work_dir + '/eval_scores.npy'
        key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs

        np.save(filename, log_data)

        obs_dict = sim_env.reset()
        done = False
        obs_traj_sim = []
        video_sim.init(enabled=True)
        video_sim.record(sim_env)
        while not done and len(obs_traj_sim) < args.time_limit:
            obs_img = obs_dict['image']
            # center crop image
            if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                obs_img = utils.center_crop_image(obs_img, args.image_size)
            obs_state = obs_dict['state']
            with utils.eval_mode(agent):
                if not use_policy:
                    action = sim_env.action_space.sample()
                elif sample_stochastically:
                    action = agent.sample_action(obs_img)
                else:
                    action = agent.select_action(obs_img)
            obs_traj_sim.append((obs_img, obs_state, action))
            obs_dict, reward, done, _ = sim_env.step(action)

            video_sim.record(sim_env)
            sim_params = obs_dict['sim_params']
        if update_distribution and sim_param_model is not None:
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            evaluate_sim_params(sim_param_model, args, [obs_traj_sim], step, L, "val", sim_params, current_sim_params)

        video_sim.save(f'sim_{training_phase}_%d.mp4' % step)

    run_eval_loop(sample_stochastically=True)
    L.dump(step)


