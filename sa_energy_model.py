import math

import torch.nn.functional as F

import torch
from torch import nn as nn
from torch.nn import Parameter

from proj_utils import copy_model_weights, layers_nnz, fill_model_weights

# using hardware parameters from Eyeriss

default_s1 = int(100 * 1024 / 2)  # input cache
default_s2 = 1 * int(8 * 1024 / 2)  # kernel cache
default_m = 12
default_n = 14

# unit energy constants
default_e_mac = 1.0 + 1.0 + 1.0  # including both read and write RF
default_e_mem = 200.0
default_e_cache = 6.0
default_e_rf = 1.0


class Layer_energy(object):
    def __init__(self, **kwargs):
        super(Layer_energy, self).__init__()
        self.h = kwargs['h'] if 'h' in kwargs else None
        self.w = kwargs['w'] if 'w' in kwargs else None
        self.c = kwargs['c'] if 'c' in kwargs else None
        self.d = kwargs['d'] if 'd' in kwargs else None
        self.xi = kwargs['xi'] if 'xi' in kwargs else None
        self.g = kwargs['g'] if 'g' in kwargs else None
        self.p = kwargs['p'] if 'p' in kwargs else None
        self.m = kwargs['m'] if 'm' in kwargs else None
        self.n = kwargs['n'] if 'n' in kwargs else None
        self.s1 = kwargs['s1'] if 's1' in kwargs else None
        self.s2 = kwargs['s2'] if 's2' in kwargs else None
        self.r = kwargs['r'] if 'r' in kwargs else None
        self.is_conv = True if self.r is not None else False

        if self.h is not None:
            self.h_ = max(0.0, math.floor((self.h + 2.0 * self.p - self.r) / float(self.xi)) + 1)
        if self.w is not None:
            self.w_ = max(0.0, math.floor((self.w + 2.0 * self.p - self.r) / float(self.xi)) + 1)

        self.cached_Xenergy = None

    def get_alpha(self, e_mem, e_cache, e_rf):
        if self.is_conv:
            return e_mem + \
                   (math.ceil((float(self.d) / self.g) / self.n) * (self.r ** 2) / float(self.xi ** 2)) * e_cache + \
                   ((float(self.d) / self.g) * (self.r ** 2) / (self.xi ** 2)) * e_rf
        else:
            if self.c <= default_s1:
                return e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(self.d) * e_rf
            else:
                return math.ceil(float(self.d) / self.n) * e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(
                    self.d) * e_rf

    def get_beta(self, e_mem, e_cache, e_rf, in_cache=None):
        if self.is_conv:
            n = 1 if in_cache else math.ceil(self.h_ * self.w_ / self.m)
            return n * e_mem + math.ceil(self.h_ * self.w_ / self.m) * e_cache + \
                   (self.h_ * self.w_) * e_rf
        else:
            return e_mem + e_cache + e_rf

    def get_gamma(self, e_mem, k=None):
        if self.is_conv:
            rows_per_batch = math.floor(self.s1 / float(k))
            assert rows_per_batch >= self.r
            # print(self.__dict__)
            # print('###########', rows_per_batch, self.s1, k)
            # print('conv input data energy (2):{:.2e}'.format(float(k) * (self.r - 1) * (math.ceil(float(self.h) / (rows_per_batch - self.r + 1)) - 1)))

            return (float(self.d) * self.h_ * self.w_) * e_mem + \
                   float(k) * (self.r - self.xi) * \
                   max(0.0, (math.ceil(float(self.h) / (rows_per_batch - self.r + self.xi)) - 1)) * e_mem
        else:
            return float(self.d) * e_mem

    def get_knapsack_weight_W(self, e_mac, e_mem, e_cache, e_rf, in_cache=None, crelax=False):
        if self.is_conv:
            if crelax:
                # use relaxed computation energy estimation (larger than the real computation energy)
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * self.h_ * self.w_
            else:
                # computation energy will be included in other place
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * 0.0
        else:
            return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac

    def get_knapsack_bound_W(self, e_mem, e_cache, e_rf, X_nnz, k):
        if self.is_conv:
            return self.get_gamma(e_mem, k) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz
        else:
            return self.get_gamma(e_mem) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz


def build_energy_info(model, m=default_m, n=default_n, s1=default_s1, s2=default_s2):
    res = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            if layer_name in res:
                res[layer_name]['h'] = p.size()[1]
                res[layer_name]['w'] = p.size()[2]
            else:
                res[layer_name] = {'h': p.size()[1], 'w': p.size()[2]}
        elif name.endswith('.hw'):
            layer_name = name[:-len('hw') - 1]
            if layer_name in res:
                res[layer_name]['h'] = float(p.data[0])
                res[layer_name]['w'] = float(p.data[1])
            else:
                res[layer_name] = {'h': float(p.data[0]), 'w': float(p.data[1])}
        elif name.endswith('.xi'):
            layer_name = name[:-len('xi') - 1]
            if layer_name in res:
                res[layer_name]['xi'] = float(p.data[0])
            else:
                res[layer_name] = {'xi': float(p.data[0])}
        elif name.endswith('.g'):
            layer_name = name[:-len('g') - 1]
            if layer_name in res:
                res[layer_name]['g'] = float(p.data[0])
            else:
                res[layer_name] = {'g': float(p.data[0])}
        elif name.endswith('.p'):
            layer_name = name[:-len('p') - 1]
            if layer_name in res:
                res[layer_name]['p'] = float(p.data[0])
            else:
                res[layer_name] = {'p': float(p.data[0])}
        elif name.endswith('weight'):
            if len(p.size()) == 2 or len(p.size()) == 4:
                layer_name = name[:-len('weight') - 1]
                if layer_name in res:
                    res[layer_name]['d'] = p.size()[0]
                    res[layer_name]['c'] = p.size()[1]
                else:
                    res[layer_name] = {'d': p.size()[0], 'c': p.size()[1]}
                if p.dim() > 2:
                    # (out_channels, in_channels, kernel_size[0], kernel_size[1])
                    assert p.dim() == 4
                    res[layer_name]['r'] = p.size()[2]
        else:
            continue

        res[layer_name]['m'] = m
        res[layer_name]['n'] = n
        res[layer_name]['s1'] = s1
        res[layer_name]['s2'] = s2

    for layer_name in res:
        res[layer_name] = Layer_energy(**(res[layer_name]))
        if res[layer_name].g is not None and res[layer_name].g > 1:
            res[layer_name].c *= res[layer_name].g
    return res


def reset_Xenergy_cache(energy_info):
    for layer_name in energy_info:
        energy_info[layer_name].cached_Xenergy = None
    return energy_info


def conv_cache_overlap(X_supp, padding, kernel_size, stride, k_X):
    rs = X_supp.transpose(0, 1).contiguous().view(X_supp.size(1), -1).sum(dim=1).cpu()
    rs = torch.cat([torch.zeros(padding, dtype=rs.dtype, device=rs.device),
                   rs,
                    torch.zeros(padding, dtype=rs.dtype, device=rs.device)])
    res = 0
    beg = 0
    end = None
    while beg + kernel_size - 1 < rs.size(0):
        if end is not None:
            if beg < end:
                res += rs[beg:end].sum().item()
        n_elements = 0
        for i in range(rs.size(0) - beg):
            if n_elements + rs[beg+i] <= k_X:
                n_elements += rs[beg+i]
                if beg + i == rs.size(0) - 1:
                    end = rs.size(0)
            else:
                end = beg + i
                break
        assert end - beg >= kernel_size, 'can only hold {} rows with {} elements < {} rows in {}, cache size={}'.format(end - beg, n_elements, kernel_size, X_supp.size(), k_X)
        # print('map size={}. begin={}, end={}'.format(X_supp.size(), beg, end))
        beg += (math.floor((end - beg - kernel_size) / stride) + 1) * stride
    return res


def energy_eval2(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False, crelax=False):
    X_nnz_dict = layers_nnz(model, normalized=False, param_name='input_mask')

    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            einfo = energy_info[layer_name]

            if einfo.is_conv:
                X_nnz = einfo.h * einfo.w * einfo.c
            else:
                X_nnz = einfo.c
            if layer_name in X_nnz_dict:
                # this layer has sparse input
                X_nnz = X_nnz_dict[layer_name]

            if layer_name in X_supp_dict:
                X_supp = X_supp_dict[layer_name].unsqueeze(0)
            else:
                if einfo.is_conv:
                    X_supp = torch.ones(1, int(einfo.c), int(einfo.h), int(einfo.w), dtype=p.dtype, device=p.device)
                else:
                    X_supp = None

            unfoldedX = None

            # input data access energy
            if einfo.is_conv:
                h_, w_ = max(0.0, math.floor((einfo.h + 2 * einfo.p - einfo.r) / einfo.xi) + 1), max(0.0, math.floor((einfo.w + 2 * einfo.p - einfo.r) / einfo.xi) + 1)
                if verbose:
                    print('Layer: {}, input shape: ({}, {}, {}), output shape: ({}, {}, {}), weight shape: {}'
                          .format(layer_name, einfo.c, einfo.h, einfo.w, einfo.d, h_, w_, p.shape))
                unfoldedX = F.unfold(X_supp, kernel_size=int(einfo.r), padding=int(einfo.p), stride=int(einfo.xi)).squeeze(0)
                assert unfoldedX.size(1) == h_ * w_, 'unfolded X size={}, but h_ * w_ = {}, W.size={}'.format(unfoldedX.size(), h_ * w_, p.size())
                unfoldedX_nnz = (unfoldedX != 0.0).float().sum().item()

                X_energy_cache = unfoldedX_nnz * math.ceil((float(einfo.d) / einfo.g) / einfo.n) * e_cache
                X_energy_rf = unfoldedX_nnz * math.ceil(float(einfo.d) / einfo.g) * e_rf

                X_energy_mem = X_nnz * e_mem + \
                               conv_cache_overlap(X_supp.squeeze(0), int(einfo.p), int(einfo.r), int(einfo.xi), default_s1) * e_mem + \
                               unfoldedX.size(1) * einfo.d * e_mem
                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache
            else:
                X_energy_cache = math.ceil(float(einfo.d) / einfo.n) * e_cache * X_nnz
                X_energy_rf = float(einfo.d) * e_rf * X_nnz
                X_energy_mem = e_mem * (math.ceil(float(einfo.d) / einfo.n) * max(0.0, X_nnz - default_s1)
                                        + min(X_nnz, default_s1)) + e_mem * float(einfo.d)

                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache

            einfo.cached_Xenergy = X_energy_this
            X_energy.append(X_energy_this)

            # kernel weights data access energy
            if einfo.is_conv:
                output_hw = unfoldedX.size(1)
                W_energy_cache = math.ceil(output_hw / einfo.m) * W_nnz_dict[layer_name] * e_cache
                W_energy_rf = output_hw * W_nnz_dict[layer_name] * e_rf
                W_energy_mem = (math.ceil(output_hw / einfo.m) * max(0.0, W_nnz_dict[layer_name] - default_s2)\
                               + min(default_s2, W_nnz_dict[layer_name])) * e_mem
                W_energy_this = W_energy_cache + W_energy_rf + W_energy_mem
            else:
                W_energy_this = einfo.get_beta(e_mem, e_cache, e_rf, in_cache=None) * W_nnz_dict[layer_name]
            W_energy.append(W_energy_this)

            # computation enregy
            if einfo.is_conv:
                if crelax:
                    N_mac = energy_info[layer_name].h_ * float(energy_info[layer_name].w_) * W_nnz_dict[layer_name]
                else:
                    N_mac = torch.sum(
                        F.conv2d(X_supp, (p.data != 0.0).float(), None, int(energy_info[layer_name].xi),
                                 int(energy_info[layer_name].p), 1, int(energy_info[layer_name].g))).item()
                C_energy_this = e_mac * N_mac
            else:
                C_energy_this = e_mac * (W_nnz_dict[layer_name])

            C_energy.append(C_energy_this)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_eval2_relax(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False):
    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            X_energy.append(energy_info[layer_name].cached_Xenergy)
            assert X_energy[-1] > 0
            if not energy_info[layer_name].is_conv:
                # in_cache is not needed in fc layers
                in_cache = None
                W_energy.append(
                    energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache) * W_nnz_dict[layer_name])
                C_energy.append(e_mac * (W_nnz_dict[layer_name]))
                if verbose:
                    knapsack_weight1 = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                     in_cache=None, crelax=True)
                    if hasattr(knapsack_weight1, 'mean'):
                        knapsack_weight1 = knapsack_weight1.mean()
                    print(layer_name + " weight: {:.4e}".format(knapsack_weight1))

            else:
                beta1 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                beta2 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False)

                W_nnz = W_nnz_dict[layer_name]
                W_energy_this = beta1 * min(energy_info[layer_name].s2, W_nnz) + beta2 * max(0, W_nnz - energy_info[
                    layer_name].s2)
                W_energy.append(W_energy_this)
                C_energy.append(e_mac * energy_info[layer_name].h_ * float(energy_info[layer_name].w_) * W_nnz)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_proj2(model, energy_info, budget, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, grad=False, in_place=True, preserve=0.0, param_name='weight'):
    knapsack_bound = budget
    param_flats = []
    knapsack_weight_all = []
    score_all = []
    param_shapes = []
    bound_bias = 0.0

    for name, p in model.named_parameters():
        if name.endswith(param_name):
            if p is None or (param_name == 'weight' and p.dim() == 1):
                # skip batch_norm layer
                param_shapes.append((name, None))
                continue
            else:
                param_shapes.append((name, p.data.shape))

            layer_name = name[:-len(param_name) - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            if grad:
                p_flat = p.grad.data.view(-1)
            else:
                p_flat = p.data.view(-1)
            score = p_flat ** 2

            if param_name == 'weight':
                knapsack_weight = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                in_cache=True, crelax=True)
                if hasattr(knapsack_weight, 'view'):
                    knapsack_weight = knapsack_weight.view(1, -1, 1, 1)
                knapsack_weight = torch.zeros_like(p.data).add_(knapsack_weight).view(-1)

                # preserve part of weights
                if preserve > 0.0:
                    if preserve > 1:
                        n_preserve = preserve
                    else:
                        n_preserve = round(p_flat.numel() * preserve)
                    _, preserve_idx = torch.topk(score, k=n_preserve, largest=True, sorted=False)
                    score[preserve_idx] = float('inf')

                if energy_info[layer_name].is_conv and p_flat.numel() > energy_info[layer_name].s2:
                    delta = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False) \
                            - energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                    assert delta >= 0
                    _, out_cache_idx = torch.topk(score, k=p_flat.numel() - energy_info[layer_name].s2, largest=False,
                                                  sorted=False)
                    knapsack_weight[out_cache_idx] += delta

                bound_const = energy_info[layer_name].cached_Xenergy

                assert bound_const > 0
                bound_bias += bound_const
                knapsack_bound -= bound_const

            else:
                raise ValueError('not supported parameter name')

            score_all.append(score)
            knapsack_weight_all.append(knapsack_weight)
            # print(layer_name, X_nnz, knapsack_weight)
            param_flats.append(p_flat)

    param_flats = torch.cat(param_flats, dim=0)
    knapsack_weight_all = torch.cat(knapsack_weight_all, dim=0)
    score_all = torch.cat(score_all, dim=0) / knapsack_weight_all

    _, sorted_idx = torch.sort(score_all, descending=True)
    cumsum = torch.cumsum(knapsack_weight_all[sorted_idx], dim=0)
    res_nnz = torch.nonzero(cumsum <= knapsack_bound).max()
    z_idx = sorted_idx[-(param_flats.numel() - res_nnz):]

    if in_place:
        param_flats[z_idx] = 0.0
        copy_model_weights(model, param_flats, param_shapes, param_name)
    return z_idx, param_shapes


class myConv2d(nn.Conv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.h_in = h_in
        self.w_in = w_in
        self.xi = Parameter(torch.LongTensor(1), requires_grad=False)
        self.xi.data[0] = stride
        self.g = Parameter(torch.LongTensor(1), requires_grad=False)
        self.g.data[0] = groups
        self.p = Parameter(torch.LongTensor(1), requires_grad=False)
        self.p.data[0] = padding

    def __repr__(self):
        s = ('{name}({h_in}, {w_in}, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class FixHWConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FixHWConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)

        self.hw = Parameter(torch.LongTensor(2), requires_grad=False)
        self.hw.data[0] = h_in
        self.hw.data[1] = w_in

    def forward(self, input):
        # Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        assert input.size(2) == self.hw.data[0] and input.size(3) == self.hw.data[1], 'input_size={}, but hw={}'.format(
            input.size(), self.hw.data)
        return super(FixHWConv2d, self).forward(input)


class SparseConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.input_mask = Parameter(torch.Tensor(in_channels, h_in, w_in))
        self.input_mask.data.fill_(1.0)

    def forward(self, input):
        # print("###{}, {}".format(input.size(), self.input_mask.size()))
        return super(SparseConv2d, self).forward(input * self.input_mask)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

