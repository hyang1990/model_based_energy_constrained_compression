import copy
import math
import torch
import torch.nn as nn


def copy_model_weights(model, W_flat, W_shapes, param_name='weight'):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel


def reset_model_param(model):
    for M in model.modules():
        if hasattr(M, 'reset_parameters'):
            M.reset_parameters()


def idxproj(model, z_idx, W_shapes, param_name='weight'):
    assert type(z_idx) is torch.LongTensor or type(z_idx) is torch.cuda.LongTensor
    offset = 0
    i = 0
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            name_, shape = W_shapes[i]
            i += 1
            assert name_ == name
            if shape is None:
                continue
            mask = z_idx >= offset
            mask[z_idx >= (offset + W.numel())] = 0
            z_idx_sel = z_idx[mask]
            if len(z_idx_sel.shape) != 0:
                W.data.view(-1)[z_idx_sel - offset] = 0.0
            offset += W.numel()


def getmask(model, param_name='weight'):
    mask_model = copy.deepcopy(model)
    for name, W in mask_model.named_parameters():
        if name.endswith(param_name):
            W.data.copy_(W.data != 0.0)

    return mask_model


def maskproj(model, mask_model, param_name='weight'):
    mask_model_param = mask_model.named_parameters()
    for name1, W in model.named_parameters():
        name2, W_mask = next(mask_model_param)
        assert name1 == name2
        if name1.endswith(param_name) and W.dim() > 1:
            W.data.mul_(W_mask.data)


def idx2mask(mask_model, z_idx, W_shapes, param_name='weight'):
    fill_model_weights(mask_model, 1.0, param_name=param_name)
    offset = 0
    i = 0
    for name, W in mask_model.named_parameters():
        if name.endswith(param_name):
            name_, shape = W_shapes[i]
            assert name_ == name
            mask = z_idx >= offset
            mask[z_idx >= (offset + W.numel())] = 0
            z_idx_sel = z_idx[mask]
            if len(z_idx_sel.shape) != 0:
                W.data.view(-1)[z_idx_sel - offset] = 0.0
            i += 1
            offset += W.numel()

    return mask_model


def model_mask(model, param_name='weight'):
    mask_model = copy.deepcopy(model)
    fill_model_weights(mask_model, 1.0, param_name=param_name)

    model2_param = model.named_parameters()
    for name1, p1 in mask_model.named_parameters():
        name2, p2 = next(model2_param)
        assert name1 == name2
        if name1.endswith(param_name) and p1.dim() > 1:
            p1.data.copy_((p2.data != 0.0).float())

    return mask_model


def filtered_parameters(model, param_name, inverse=False):
    for name, param in model.named_parameters():
        if inverse != (name.endswith(param_name)):
            yield param


def l0proj(model, k, normalized=True, param_name='weight'):
    # get all the weights
    W_shapes = []
    res = []
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))

    res = torch.cat(res, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = math.floor(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes


def threshold_proj(model, thresh, param_name='weight'):
    assert thresh > 0.0
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            if W.dim() > 1:
                W.data[W.data.abs() < thresh] = 0.0


def print_model_weights(model, param_name='weight'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            print(name, W.data)


def model_weights_diff(model1, model2, param_name='weight'):
    res = 0.0
    model2_param = model2.named_parameters()
    for name1, W1 in model1.named_parameters():
        name2, W2 = next(model2_param)
        assert name1 == name2
        if name1.endswith(param_name):
            res += (W1.data - W2.data).abs().sum()

    return res


def model_sparsity(model, normalized=True, param_name='weight'):
    nnz = 0
    numel = 0
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                nnz += W_nz.shape[0]
            numel += torch.numel(W.data)

    return float(nnz) / float(numel) if normalized else float(nnz)


def model_sparsity_lb(model, param_name='weight'):
    numel = 0
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            numel += torch.numel(W.data)

    return 1.0 / float(numel)


def layers_nnz(model, normalized=True, param_name='weight'):
    res = {}
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name)-1]
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
            else:
                res[layer_name] = 0

    return res


def layers_nnz_hw(model, param_name='weight'):
    """
    Get a dict which contains each layer's nnz on the last two dimensions i.e. height and weight
    :param model: The model contains the layers
    :param param_name: The layers' parameter name, i.e. weight
    :return: Dict containing layer names and the nnz tensor
    """
    res = {}
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            if len(W.size()) < 3:
                res[layer_name] = (W.data != 0.0).float().sum().item()
            else:
                h_times_w = W.size()[-1] * W.size()[-2]
                W_nz = (W.data.view(*(W.size()[:-2]), h_times_w) != 0.0).float()
                res[layer_name] = torch.sum(W_nz, dim=-1)

    return res


def layers_nz_mask(model, param_name='weight'):
    res = {}
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            res[layer_name] = (W.data != 0.0).float()

    return res


def layers_stat(model, param_names=('weight',), param_filter=lambda p: True):
    if isinstance(param_names, str):
        param_names = (param_names,)
    def match_endswith(name):
        for param_name in param_names:
            if name.endswith(param_name):
                return param_name
        return None
    res = "########### layer stat ###########\n"
    for name, W in model.named_parameters():
        param_name = match_endswith(name)
        if param_name is not None:
            if param_filter(W):
                layer_name = name[:-len(param_name) - 1]
                W_nz = torch.nonzero(W.data)
                nnz = W_nz.shape[0] / W.data.numel() if W_nz.dim() > 0 else 0.0
                W_data_abs = W.data.abs()
                res += "{:>20}".format(layer_name) + 'abs(W): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'\
                    .format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer stat ###########"
    return res


def layers_grad_stat(model, param_name='weight'):
    res = "########### layer grad stat ###########\n"
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            W_nz = torch.nonzero(W.grad.data)
            nnz = W_nz.shape[0] / W.grad.data.numel() if W_nz.dim() > 0 else 0.0
            W_data_abs = W.grad.data.abs()
            res += "{:>20}".format(layer_name) + 'abs(W.grad): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'.format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer grad stat ###########"
    return res


def fill_model_weights(model, val, param_name='weight'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W.data.fill_(val)

    return model


def clamp_model_weights(model, min=0.0, max=1.0, param_name='input_mask'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W.data.clamp_(min=min, max=max)

    return model


def round_model_weights(model, param_name='input_mask'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W.data.round_()

    return model


def model_support_set(model, param_name='weight'):
    res = copy.deepcopy(model)
    res_param = res.named_parameters()
    for name1, W1 in model.named_parameters():
        name2, W2 = next(res_param)
        assert name1 == name2
        if name1.endswith(param_name):
            W2.data[:] = (W1.data != 0.0)

    return res


def argmax(a):
    return max(range(len(a)), key=a.__getitem__)


def num_dict_info(d):
    res = "{"
    for k in d:
        res += "{}: {:.4e}, ".format(k, d[k])

    res += '}'
    return res


if __name__ == '__main__':
    layers = [nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3), nn.Linear(16, 10)]
    model = nn.Sequential(*layers)
    print_model_weights(model)
    model_ = copy.deepcopy(model)
    z_idx, W_shapes = l0proj(model_, 100)
    print_model_weights(model_)
    idxproj(model, z_idx, W_shapes)
    print_model_weights(model)

    print("diff={}".format(model_weights_diff(model, model_)))
