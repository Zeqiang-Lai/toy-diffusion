import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import math
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm



eps = 1e-8


def exists(x):
    return x is not None


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def multinomial_kl(log_prob1, log_prob2):   # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl

# scheduler

def cos_alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999, exp=3):
    att = np.arange(0, time_step)
    att = (np.cos((att + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    att = att * (att_1 - att_T) + att_T
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step)
    ctt = (np.cos((ctt + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    ctt = ctt * (ctt_1 - ctt_T) + ctt_T
    ctt = np.concatenate(([0], ctt))

    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


    
class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_classes,
        image_size=32,
        timesteps=100,
        alpha_schedule='alpha1',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1, 1],
        sample_time_method='uniform',
        loss_type='vb_stochastic',
        parametrization='x0',
    ):
        super().__init__()

        self.model = model
        self.amp = False

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.shape = image_size
        self.num_timesteps = timesteps
        self.parametrization = parametrization
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        self.sample_time_method = sample_time_method

        if alpha_schedule == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes - 1)
        elif alpha_schedule == 'cos':
            at, bt, ct, att, btt, ctt = cos_alpha_schedule(self.num_timesteps, N=self.num_classes - 1)
        else:
            print("alpha_init_type is Wrong !! ")

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        # assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        # assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat([
            log_add_exp(log_x_t[:, :-1, ...] + log_at, log_bt),
            log_add_exp(log_x_t[:, -1:, ...] + log_1_min_ct, log_ct)
        ], dim=1)
        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~

        log_probs = torch.cat([
            log_add_exp(log_x_start[:, :-1, ...] + log_cumprod_at, log_cumprod_bt),
            log_add_exp(log_x_start[:, -1:, ...] + log_1_min_cumprod_ct, log_cumprod_ct)
        ], dim=1)
        return log_probs

    def predict_start(self, log_x_t, cond_emb, t):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)

        out = self.model(x_t, cond_emb, t)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == x_t.size()[1:], '{}, {}'.format(out.shape, x_t.shape)

        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        # if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
        self.zero_vector = torch.zeros(batch_size, 1, *log_x_t.shape[2:]).type_as(log_x_t) - 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        num_dims = len(log_x_t.shape) - 2

        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, *[1] * num_dims).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, *log_x_t.shape[2:])

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, ...]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, *[-1] * num_dims)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, ...], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, *[-1] * num_dims)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, :-1, ...] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t):
        # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_start = self.predict_start(log_x, cond_emb, t)
            log_model_pred = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t)
            log_x_start = log_model_pred # dummy log_x_start
        else:
            raise ValueError
        return log_model_pred, log_x_start

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        log_model_pred, log_x_start = self.p_pred(log_x, cond_emb, t)
        out = self.log_sample_categorical(log_model_pred)
        return out, log_x_start


    @torch.no_grad()
    def p_sample_with_x0(self, x0, log_x, t):
        out = index_to_log_onehot(x0, self.num_classes - 1)
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x.size()[0]
        self.zero_vector = torch.zeros(batch_size, 1, *log_x.shape[2:]).type_as(log_x) - 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_x_start = torch.clamp(log_pred, -70, 0)

        log_model_pred = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x, t=t)
        out = self.log_sample_categorical(log_model_pred)
        return out, log_x_start

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            # Overwrite decoder term with L1. 
            # Why overwrite: L0 compute nll, L1-LT compute kl, so it is not comparable. 
            # Therefore, we use L1 to replace L0 to obtain meaningful prob.
            Lt_sqrt[0] = Lt_sqrt[1] 
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device

        x_start = x
        t, pt = self.sample_time(b, device, self.sample_time_method)

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t)  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

        if self.loss_type == 'crossentropy':
            with torch.no_grad():
                acc = torchmetrics.functional.accuracy(log_x0_recon[:, :-1, ...],x_start).cpu()
            loss = F.nll_loss(log_x0_recon, x_start, ignore_index=0, reduction='none')
            loss = sum_except_batch(loss)
            return log_model_prob, loss, acc, t

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = multinomial_kl(log_x_start[:, :-1, ...], log_x0_recon[:, :-1, ...])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        # for log purpose
        with torch.no_grad():
            acc = torchmetrics.functional.accuracy(log_x0_recon[:, :-1, ...],x_start).cpu()
        
        return log_model_prob, vb_loss, acc, t

    @property
    def device(self):
        return next(self.model.parameters()).device

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.model.named_parameters()}  # if p.requires_grad}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
        self,
        input,
        return_loss=False,
        return_logits=True,
        return_att_weight=False,
        is_train=True,
        **kwargs
    ):
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = input['content_token'].shape[0]
        device = input['content_token'].device

        # 1) get embeddding for condition and content     prepare input
        sample_image = input['content_token'].type_as(input['content_token'])
        # cont_emb = self.content_emb(sample_image)

        if self.condition_emb is not None:
            with autocast(enabled=self.amp):
                if self.condition_emb.trainable:
                    cond_emb = self.condition_emb(input['condition_token'])
                else:
                    with torch.no_grad():
                        cond_emb = self.condition_emb(input['condition_token'])  # B x Ld x D   #256*1024
        else:  # share condition embeding with content
            if input.get('condition_embed_token') == None:
                cond_emb = None
            else:
                cond_emb = input['condition_embed_token']

        # now we get cond_emb and sample_image
        if is_train == True:
            log_model_prob, loss, acc, t = self._train_loss(sample_image, cond_emb)
            loss = loss / np.prod(sample_image.shape)

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss.sum()
            out['losses'] = loss
            out['t'] = t
            out['acc'] = acc
        self.amp = False
        return out

    @torch.no_grad()
    def sample(
        self,
        condition_token,
        content_token=None,
        sample_shape=None,
        content_token_init=None,
        filter_ratio=0.5,
        use_init=False,
        collect_timesteps=[0],
        skip_step=None,
    ):
        assert sample_shape is None or content_token is None, \
            'You cannot set "sample_shape" and "content_token" at the same time'
        
        batch_size = condition_token.shape[0]

        if exists(content_token):
            self.shape = content_token.shape[1:]

        if exists(sample_shape):
            self.shape = sample_shape
        
        if exists(self.condition_emb): 
            cond_emb = self.condition_emb(condition_token) 
        else: 
            cond_emb = None

        start_step = int(self.num_timesteps * filter_ratio)

        if exists(skip_step):
            results = self.sample_fast(
                batch_size,
                cond_emb=cond_emb,
                skip_step=skip_step,
                collect_timesteps=collect_timesteps
            )
        elif start_step == 0: 
            results = self.sample_scratch(
                batch_size, 
                cond_emb=cond_emb,
                x0=content_token_init,
                use_init=use_init, 
                collect_timesteps=collect_timesteps                              
            )
        else:
           results = self.sample_interm(
                content_token, 
                cond_emb=cond_emb,
                x0=content_token_init,
                start_step=start_step, 
                use_init=use_init, 
                collect_timesteps=collect_timesteps                              
            )

        output = {}
        output['tokens'] = {
            'x': [log_onehot_to_index(x) for x in results['log_z']],
            'x0': [log_onehot_to_index(x) for x in results['log_z0']],
        }
        output['logits'] = {
            'x': [torch.exp(x) for x in results['log_z']],
            'x0': [torch.exp(x) for x in results['log_z0']],
        }
        return output

    @torch.no_grad()
    def sample_fast(self, batch_size, cond_emb, skip_step=1, collect_timesteps=[0]):
        assert start_step == 0
        zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.shape), device=self.device)
        one_logits = torch.ones((batch_size, 1, self.shape), device=self.device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps


        step_list = [index for index in range(start_step - 1, -1, -1 - skip_step)]
        if step_list[-1] != 0:
            step_list.append(0)
        
        results = {}
        results['log_z'] = []
        results['log_z0'] = []
        for cur_step in step_list:
            t = torch.full((batch_size,), cur_step, device=self.device, dtype=torch.long)
            log_x_start = self.predict_start(log_z, cond_emb, t)
            if cur_step > skip_step:
                model_log_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t - skip_step)
            else:
                model_log_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t)

            log_z = self.log_sample_categorical(model_log_prob)
            
            if cur_step in collect_timesteps:
                results['log_z'].append(log_z)
                results['log_z0'].append(log_x_start)
                
        return results
    
    def sample_scratch(self, batch_size, cond_emb, x0=None, use_init=False, collect_timesteps=[0]):
        zero_logits = torch.zeros((batch_size, self.num_classes - 1, *self.shape), device=self.device)
        one_logits = torch.ones((batch_size, 1, *self.shape), device=self.device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        
        results = self.p_sample_loop(
            log_z, cond_emb, 
            x0=x0,
            start_step=start_step, 
            use_init=use_init, 
            collect_timesteps=collect_timesteps
        )
        
        return results
        
    
    def sample_interm(self, content, start_step, cond_emb, x0=None, use_init=False, collect_timesteps=[0]):
        batch_size = content.shape[0]
        t = torch.full((batch_size,), start_step - 1, device=self.device, dtype=torch.long)
        log_x_start = index_to_log_onehot(content, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        log_z = log_xt

        results = self.p_sample_loop(
            log_z, cond_emb, 
            x0=x0,
            start_step=start_step, 
            use_init=use_init,
            collect_timesteps=collect_timesteps
        )

        return results

    @torch.no_grad()
    def p_sample_loop(self, log_z, cond_emb, start_step, x0=None, use_init=False, collect_timesteps=[0]):
        batch_size = log_z.shape[0]
        
        results = {}
        results['log_z'] = []
        results['log_z0'] = []
            
        for cur_step in range(start_step - 1, -1, -1):
            t = torch.full((batch_size,), cur_step, device=self.device, dtype=torch.long)
            
            if cur_step == start_step - 1 and use_init:
                log_z, log_z0 = self.p_sample_with_x0(x0, log_z, t)
            else:
                log_z, log_z0 = self.p_sample(log_z, cond_emb, t)
                
            if cur_step in collect_timesteps:
                results['log_z'].append(log_z)
                results['log_z0'].append(log_z0)
            
        return results
        