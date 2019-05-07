import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import datasets, transforms
import torch.nn.functional as F


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_mnist32(batch_size, val_batch_size, data_root='./mnist_dataset', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    kwargs.pop('input_size', None)
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=val_batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)

        train_loader4eval = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=val_batch_size, shuffle=False, **kwargs)
        ds.append(train_loader4eval)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_data_loaders(data_dir, dataset='imagenet', batch_size=32, val_batch_size=512, num_workers=0, nsubset=-1,
                     normalize=None):
    if dataset == 'imagenet':
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        if normalize is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if nsubset > 0:
            rand_idx = torch.randperm(len(train_dataset))[:nsubset]
            print('use a random subset of data:')
            print(rand_idx)
            train_sampler = SubsetRandomSampler(rand_idx)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        # use 10K training data to see the training performance
        train_loader4eval = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            sampler=SubsetRandomSampler(torch.randperm(len(train_dataset))[:10000]))

        return train_loader, val_loader, train_loader4eval
    elif dataset == 'mnist-32':
        return get_mnist32(batch_size=batch_size, val_batch_size=val_batch_size, num_workers=num_workers)
    else:
        raise NotImplementedError


def ncorrect(output, target, topk=(1,)):
    """Computes the numebr of correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        res.append(correct_k)
    return res


def eval_loss_acc1_acc5(model, data_loader, loss_func=None, cuda=True, class_offset=0):
    val_loss = 0.0
    val_acc1 = 0.0
    val_acc5 = 0.0
    num_data = 0
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            num_data += target.size(0)
            target.data += class_offset
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if loss_func is not None:
                val_loss += loss_func(model, data, target).item()
            # val_loss += F.cross_entropy(output, target).item()
            nc1, nc5 = ncorrect(output.data, target.data, topk=(1, 5))
            val_acc1 += nc1
            val_acc5 += nc5
            # print('acc:{}, {}'.format(nc1 / target.size(0), nc5 / target.size(0)))

    val_loss /= len(data_loader)
    val_acc1 /= num_data
    val_acc5 /= num_data

    return val_loss, val_acc1, val_acc5


def cross_entropy(input, target, label_smoothing=0.0, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets (long tensor)
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if label_smoothing <= 0.0:
        return F.cross_entropy(input, target)
    assert input.dim() == 2 and target.dim() == 1
    target_ = torch.unsqueeze(target, 1)
    one_hot = torch.zeros_like(input)
    one_hot.scatter_(1, target_, 1)
    one_hot = torch.clamp(one_hot, max=1.0-label_smoothing, min=label_smoothing/(one_hot.size(1) - 1.0))

    if size_average:
        return torch.mean(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))


def joint_loss(model, data, target, teacher_model, distill, label_smoothing=0.0):
    criterion = lambda pred, y: cross_entropy(pred, y, label_smoothing=label_smoothing)
    output = model(data)
    if distill <= 0.0:
        return criterion(output, target)
    else:
        with torch.no_grad():
            teacher_output = teacher_model(data).data
        distill_loss = torch.mean((output - teacher_output) ** 2)
        if distill >= 1.0:
            return distill_loss
        else:
            class_loss = criterion(output, target)
            # print("distill loss={:.4e}, class loss={:.4e}".format(distill_loss, class_loss))
            return distill * distill_loss + (1.0 - distill) * class_loss


def argmax(a):
    return max(range(len(a)), key=a.__getitem__)


def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))


def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))
