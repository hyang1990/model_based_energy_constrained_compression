import argparse

import datetime
import numpy as np
import os
import math
import time
import torch
import random
import sys
import copy
from energynet.model_based.models import get_net_model
from energynet.model_based.proj_utils import fill_model_weights, layers_stat, model_sparsity, l0proj, model_sparsity_lb
from energynet.model_based.sa_energy_model import build_energy_info, energy_eval2, energy_eval2_relax, \
    reset_Xenergy_cache
from energynet.model_free.utils import get_data_loaders, joint_loss, PlotData, eval_loss_acc1_acc5
from utee import misc

from torchvision import transforms
from utee.misc import model_snapshot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparsity Constrained Training (Magnitude based pruning variant)')
    parser.add_argument('--net', default='alexnet', help='network arch')

    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')
    parser.add_argument('--datadir', default='/home/hyang/ssd2/ILSVRC_CLS', help='dataset dir in this machine')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=512, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for train')

    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='primal learning rate')
    parser.add_argument('--adlr', action='store_true', help='adaptive lr based on sparsity (lr = lr/sparsity)')
    parser.add_argument('--l2wd', type=float, default=0.0, help='l2 weight decay')
    parser.add_argument('--momentum', type=float, default=0.0, help='primal momentum')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='learning rate (default: 1)')
    parser.add_argument('--lr_decay_s', type=int, default=10, help='learning rate decay start epoch (default: 10)')
    parser.add_argument('--lr_decay_i', type=int, default=10, help='learning rate decay epoch interval (default: 10)')
    parser.add_argument('--lr_decay_add', action='store_true', help='use additive lr decay (otherwise use multiplicative)')

    parser.add_argument('--proj_int', type=int, default=1, help='how many batches for each projection')
    parser.add_argument('--nodp', action='store_true', help='turn off dropout')

    parser.add_argument('--randinit', action='store_true', help='use random init')
    parser.add_argument('--pretrain', default=None, help='file to load pretrained model')
    parser.add_argument('--eval', action='store_true', help='eval mode')

    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
    parser.add_argument('--save_interval', type=int, default=-1, help='how many epochs to wait before save a model')
    parser.add_argument('--logdir', default=None, help='folder to save to the log')
    parser.add_argument('--distill', type=float, default=0.0, help='distill loss weight')
    parser.add_argument('--budget', type=float, default=0.0, help='energy budget')
    parser.add_argument('--exp_bdecay', action='store_true', help='budget decay exponential')
    parser.add_argument('--mgpu', action='store_true', help='enable using multiple gpus')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if args.logdir is None:
        args.logdir = 'log/' + sys.argv[0] + str(datetime.datetime.now().strftime("_%Y_%m_%d_AT_%H_%M_%S"))

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    # rm old contents in dir
    print('remove old contents in {}'.format(args.logdir))
    os.system('rm -rf ' + args.logdir)

    # create log file
    misc.logger.init(args.logdir, 'train_log')
    print = misc.logger.info

    # backup the src
    os.system('zip -q ' + os.path.join(args.logdir, 'src.zip') + ' {}/*.py'.format(
        os.path.dirname(os.path.realpath(__file__))))

    print('command:\npython {}'.format(' '.join(sys.argv)))
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # set up random seeds
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get training and validation data loaders
    normalize = None
    class_offset = 0
    if args.net == 'mobilenet-imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        class_offset = 1
    tr_loader, val_loader, train_loader4eval = get_data_loaders(data_dir=args.datadir,
                                                                dataset=args.dataset,
                                                                batch_size=args.batch_size,
                                                                val_batch_size=args.val_batch_size,
                                                                num_workers=args.num_workers,
                                                                normalize=normalize)
    # get network model
    model, teacher_model = get_net_model(net=args.net, pretrained_dataset=args.dataset, dropout=(not args.nodp),
                                         pretrained=not args.randinit)

    # pretrained model
    if args.pretrain is not None and os.path.isfile(args.pretrain):
        print('load pretrained model:{}'.format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    elif args.pretrain is not None:
        print('fail to load pretrained model: {}'.format(args.pretrain))

    if args.mgpu:
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1
        model = torch.nn.DataParallel(model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    # for energy estimate
    print('================model energy summary================')
    energy_info = build_energy_info(model)
    energy_estimator = lambda m: sum(energy_eval2(m, energy_info, verbose=False).values())
    energy_estimator_relaxed = lambda m: sum(energy_eval2_relax(m, energy_info, verbose=False).values())

    reset_Xenergy_cache(energy_info)
    cur_energy = sum(energy_eval2(model, energy_info, verbose=True).values())
    cur_energy_relaxed = energy_estimator_relaxed(model)

    dense_model = fill_model_weights(copy.deepcopy(model), 1.0)
    budget_ub = energy_estimator_relaxed(dense_model)
    zero_model = fill_model_weights(copy.deepcopy(model), 0.0)
    budget_lb = energy_estimator_relaxed(zero_model)

    del zero_model, dense_model

    proj_func = lambda m, budget: l0proj(m, budget, normalized=True)
    print('energy on dense DNN:{:.4e}, on zero DNN:{:.4e}, normalized_lb={:.4e}'.format(budget_ub, budget_lb,
                                                                                        budget_lb / budget_ub))
    print('energy on current DNN:{:.4e}, normalized={:.4e}'.format(cur_energy, cur_energy / budget_ub))
    print('====================================================')
    print('current energy {:.4e}, relaxed: {:.4e}'.format(cur_energy, cur_energy_relaxed))


    netl2wd = args.l2wd

    if args.cuda:
        if args.distill > 0.0:
            teacher_model.cuda()
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=netl2wd)

    loss_func = lambda m, x, y: joint_loss(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill)

    plot_data = PlotData()
    if args.eval or args.dataset != 'imagenet':
        val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda,
                                                           class_offset=class_offset)
        print('**Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(val_loss, val_acc1,
                                                                                              val_acc5))
        # also evaluate training data
        tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda,
                                                        class_offset=class_offset)
        print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1, tr_acc5))
    else:
        val_acc1 = 0.0
        print('For imagenet, skip the first validation evaluation.')

    best_acc = val_acc1

    old_file = None
    cur_sparsity = model_sparsity(model)
    sparsity_step = max(0.0, cur_sparsity - args.budget) / ((len(tr_loader) * args.epochs) / args.proj_int)

    sparsity_lb = model_sparsity_lb(model)
    sparsity_decay_factor = min(1.0, max(args.budget, sparsity_lb) / cur_sparsity) ** \
                          (1.0 / ((len(tr_loader) * args.epochs) / args.proj_int))
    t_begin = time.time()
    log_tic = t_begin
    cur_budget = cur_sparsity
    lr = args.lr

    for epoch in range(args.epochs):
        # decay lr
        if epoch >= args.lr_decay_s and (epoch - args.lr_decay_s) % args.lr_decay_i == 0:
            lr *= args.lr_decay

        for batch_idx, (data, target) in enumerate(tr_loader):
            model.train()
            if args.adlr:
                optimizer.param_groups[0]['lr'] = lr / cur_sparsity
            else:
                optimizer.param_groups[0]['lr'] = lr
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            loss = loss_func(model, data, target + class_offset)
            # update network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx > 0 and batch_idx % args.proj_int == 0) or batch_idx == len(tr_loader) - 1:
                proj_func(model, cur_budget)
                if epoch == args.epochs - 1 and batch_idx >= len(tr_loader) - 1 - args.proj_int:
                    cur_budget = args.budget
                else:
                    if args.exp_bdecay:
                        cur_budget = max(cur_budget * sparsity_decay_factor, args.budget)
                    else:
                        cur_budget = max(cur_budget - sparsity_step, args.budget)

            if batch_idx % args.log_interval == 0:
                print('======================================================')
                print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                       len(tr_loader)))
                log_toc = time.time()
                print(
                    'primal update: net loss={:.4e}, lr={:.4e}, current normalized budget: {:.4e}, time_elapsed={:.3f}s'.format(
                        loss.item(), optimizer.param_groups[0]['lr'], cur_budget, log_toc - log_tic))
                log_tic = time.time()
                if batch_idx % args.proj_int == 0:
                    cur_sparsity = model_sparsity(model)
                print('sparsity:{}'.format(model_sparsity(model)))
                print(layers_stat(model))
                print('+-----------------------------------------------------+')

        cur_energy = energy_estimator(model)
        cur_energy_relaxed = energy_estimator_relaxed(model)
        if epoch % args.test_interval == 0:
            plot_data.append('energy', cur_energy)
            plot_data.append('normalized energy', cur_energy / budget_ub)
            plot_data.append('budget', cur_budget)

            val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda,
                                                               class_offset=class_offset)
            plot_data.append('val_loss', val_loss)
            plot_data.append('val_acc1', val_acc1)
            plot_data.append('val_acc5', val_acc5)

            # also evaluate training data
            tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func,
                                                            args.cuda, class_offset=class_offset)
            print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                 tr_acc5))
            plot_data.append('tr_loss', tr_loss)
            plot_data.append('tr_acc1', tr_acc1)
            plot_data.append('tr_acc5', tr_acc5)

            if val_acc1 > best_acc:
                pass
                # new_file = os.path.join(args.logdir, 'model_best-{}.pkl'.format(epoch))
                # misc.model_snapshot(primal_model.net, new_file, old_file=old_file, verbose=True)
                # best_acc = val_acc1
                # old_file = new_file
            print(
                '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current normalized energy:{:.4e}, {:.4e}(relaxed)'.format(
                    val_loss, val_acc1,
                    val_acc5, cur_energy / budget_ub, cur_energy_relaxed / budget_ub))
            # save current model
            model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))
            plot_data.dump(os.path.join(args.logdir, 'plot_data.pkl'))

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            model_snapshot(model, os.path.join(args.logdir, 'primal_model_epoch{}.pkl'.format(epoch)))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (1 + epoch)
        eta = speed_epoch * (args.epochs - epoch)
        print("Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))
