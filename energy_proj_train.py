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
from models import get_net_model
from proj_utils import fill_model_weights, layers_stat, model_sparsity, filtered_parameters, \
    l0proj, round_model_weights, clamp_model_weights
from sa_energy_model import build_energy_info, energy_eval2, energy_eval2_relax, energy_proj2, \
    reset_Xenergy_cache
from utils import get_data_loaders, joint_loss, eval_loss_acc1_acc5, model_snapshot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model-Based Energy Constrained Training')
    parser.add_argument('--net', default='alexnet', help='network arch')

    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')
    parser.add_argument('--datadir', default='./ILSVRC_CLS', help='dataset dir in this machine')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=512, help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for training loader')

    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--xlr', type=float, default=1e-4, help='learning rate for input mask')

    parser.add_argument('--l2wd', type=float, default=1e-4, help='l2 weight decay')
    parser.add_argument('--xl2wd', type=float, default=1e-5, help='l2 weight decay (for input mask)')

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--proj_int', type=int, default=10, help='how many batches for each projection')
    parser.add_argument('--nodp', action='store_true', help='turn off dropout')
    parser.add_argument('--input_mask', action='store_true', help='enable input mask')

    parser.add_argument('--randinit', action='store_true', help='use random init')
    parser.add_argument('--pretrain', default=None, help='file to load pretrained model')
    parser.add_argument('--eval', action='store_true', help='evaluate testset in the begining')

    parser.add_argument('--seed', type=int, default=117, help='random seed')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
    parser.add_argument('--save_interval', type=int, default=-1, help='how many epochs to wait before save a model')
    parser.add_argument('--logdir', default=None, help='folder to save to the log')
    parser.add_argument('--distill', type=float, default=0.5, help='distill loss weight')
    parser.add_argument('--budget', type=float, default=0.2, help='energy budget (relative)')
    parser.add_argument('--exp_bdecay', action='store_true', help='exponential budget decay')
    parser.add_argument('--mgpu', action='store_true', help='enable using multiple gpus')
    parser.add_argument('--skip1', action='store_true', help='skip the first W update')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if args.logdir is None:
        args.logdir = 'log/' + sys.argv[0] + str(datetime.datetime.now().strftime("_%Y_%m_%d_AT_%H_%M_%S"))

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
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
    tr_loader, val_loader, train_loader4eval = get_data_loaders(data_dir=args.datadir,
                                                                dataset=args.dataset,
                                                                batch_size=args.batch_size,
                                                                val_batch_size=args.val_batch_size,
                                                                num_workers=args.num_workers,
                                                                normalize=normalize)
    # get network model
    model, teacher_model = get_net_model(net=args.net, pretrained_dataset=args.dataset, dropout=(not args.nodp),
                                         pretrained=not args.randinit, input_mask=args.input_mask)

    # pretrained model
    if args.pretrain is not None and os.path.isfile(args.pretrain):
        print('load pretrained model:{}'.format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    elif args.pretrain is not None:
        print('fail to load pretrained model: {}'.format(args.pretrain))

    # set up multi-gpus
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
    args.budget = max(args.budget, budget_lb / budget_ub)

    proj_func = lambda m, budget, grad=False, in_place=True: energy_proj2(m, energy_info, budget, grad=grad,
                                                                         in_place=in_place, param_name='weight')
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

    loss_func = lambda m, x, y: joint_loss(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill)

    if args.eval or args.dataset != 'imagenet':
        val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)
        print('**Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(val_loss, val_acc1,
                                                                                              val_acc5))
        # also evaluate training data
        tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
        print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1, tr_acc5))
    else:
        val_acc1 = 0.0
        print('For imagenet, skip the first validation evaluation.')

    old_file = None

    energy_step = math.ceil(
        max(0.0, cur_energy - args.budget * budget_ub) / ((len(tr_loader) * args.epochs) / args.proj_int))

    energy_decay_factor = min(1.0, (args.budget * budget_ub) / cur_energy) ** \
                          (1.0 / ((len(tr_loader) * args.epochs) / args.proj_int))

    optimizer = torch.optim.SGD(filtered_parameters(model, param_name='input_mask', inverse=True), lr=args.lr, momentum=args.momentum, weight_decay=netl2wd)
    if args.input_mask:
        Xoptimizer = torch.optim.Adam(filtered_parameters(model, param_name='input_mask', inverse=False), lr=args.xlr, weight_decay=args.xl2wd)

    cur_budget = cur_energy_relaxed
    lr = args.lr
    xlr = args.xlr
    cur_sparsity = model_sparsity(model)

    best_acc_pruned = None
    Xbudget = 0.9
    iter_idx = 0

    W_proj_time = 0.0
    W_proj_time_cnt = 1e-15
    while True:
        # update W
        if not (args.skip1 and iter_idx == 0):
            t_begin = time.time()
            log_tic = t_begin
            for epoch in range(args.epochs):
                for batch_idx, (data, target) in enumerate(tr_loader):
                    model.train()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()

                    loss = loss_func(model, data, target)
                    # update network weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if args.proj_int == 1 or (batch_idx > 0 and batch_idx % args.proj_int == 0) or batch_idx == len(tr_loader) - 1:
                        temp_tic = time.time()
                        proj_func(model, cur_budget)
                        W_proj_time += time.time() - temp_tic
                        W_proj_time_cnt += 1
                        if epoch == args.epochs - 1 and batch_idx >= len(tr_loader) - 1 - args.proj_int:
                            cur_budget = args.budget * budget_ub
                        else:
                            if args.exp_bdecay:
                                cur_budget = max(cur_budget * energy_decay_factor, args.budget * budget_ub)
                            else:
                                cur_budget = max(cur_budget - energy_step, args.budget * budget_ub)

                    if batch_idx % args.log_interval == 0:
                        print('======================================================')
                        print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                               len(tr_loader)))
                        log_toc = time.time()
                        print(
                            'primal update: net loss={:.4e}, lr={:.4e}, current normalized budget: {:.4e}, time_elapsed={:.3f}s, averaged projection_time {}'.format(
                                loss.item(), optimizer.param_groups[0]['lr'], cur_budget / budget_ub, log_toc - log_tic, W_proj_time / W_proj_time_cnt))
                        log_tic = time.time()
                        if batch_idx % args.proj_int == 0:
                            cur_sparsity = model_sparsity(model)
                        print('sparsity:{}'.format(cur_sparsity))
                        print(layers_stat(model, param_names='weight', param_filter=lambda p: p.dim() > 1))
                        print('+-----------------------------------------------------+')

                cur_energy = energy_estimator(model)
                cur_energy_relaxed = energy_estimator_relaxed(model)
                cur_sparsity = model_sparsity(model)
                if epoch % args.test_interval == 0:
                    val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)

                    # also evaluate training data
                    tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
                    print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                         tr_acc5))

                    print(
                        '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current normalized energy:{:.4e}, {:.4e}(relaxed), sparsity: {:.4e}'.format(
                            val_loss, val_acc1,
                            val_acc5, cur_energy / budget_ub, cur_energy_relaxed / budget_ub, cur_sparsity))
                    # save current model
                    model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

                if args.save_interval > 0 and epoch % args.save_interval == 0:
                    model_snapshot(model, os.path.join(args.logdir, 'Wprimal_model_epoch{}_{}.pkl'.format(iter_idx, epoch)))

                elapse_time = time.time() - t_begin
                speed_epoch = elapse_time / (1 + epoch)
                eta = speed_epoch * (args.epochs - epoch)
                print("Updating Weights, Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))

        if not args.input_mask:
            print("Complete weights training.")
            break
        else:
            print("Continue to train input mask.")

        if best_acc_pruned is not None and val_acc1 <= best_acc_pruned:
            print("Pruned accuracy does not improve, stop here!")
            break
        best_acc_pruned = val_acc1

        # update X
        t_begin = time.time()
        log_tic = t_begin
        for epoch in range(args.epochs):
            for batch_idx, (data, target) in enumerate(tr_loader):
                model.train()
                Xoptimizer.param_groups[0]['lr'] = xlr
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                loss = loss_func(model, data, target)
                # update network weights
                Xoptimizer.zero_grad()
                loss.backward()
                Xoptimizer.step()
                clamp_model_weights(model, min=0.0, max=1.0, param_name='input_mask')

                if (batch_idx > 0 and batch_idx % args.proj_int == 0) or batch_idx == len(tr_loader) - 1:
                    l0proj(model, Xbudget, param_name='input_mask')

                if batch_idx % args.log_interval == 0:
                    print('======================================================')
                    print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                           len(tr_loader)))
                    log_toc = time.time()
                    print('primal update: net loss={:.4e}, xlr={:.4e}, time_elapsed={:.3f}s'.format(
                            loss.item(), Xoptimizer.param_groups[0]['lr'], log_toc - log_tic))
                    log_tic = time.time()
                    if batch_idx % args.proj_int == 0:
                        cur_sparsity = model_sparsity(model, param_name='input_mask')
                    print('sparsity:{}'.format(cur_sparsity))
                    print(layers_stat(model, param_names='input_mask'))
                    print('+-----------------------------------------------------+')

            cur_energy = energy_estimator(model)
            cur_energy_relaxed = energy_estimator_relaxed(model)
            cur_sparsity = model_sparsity(model, param_name='input_mask')
            if epoch % args.test_interval == 0:

                val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)

                # also evaluate training data
                tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
                print(
                    '###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                   tr_acc5))

                print(
                    '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, current normalized energy:{:.4e}, {:.4e}(relaxed), sparsity: {:.4e}'.format(
                        val_loss, val_acc1,
                        val_acc5, cur_energy / budget_ub, cur_energy_relaxed / budget_ub, cur_sparsity))
                # save current model
                model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

            if args.save_interval > 0 and epoch % args.save_interval == 0:
                model_snapshot(model, os.path.join(args.logdir, 'Xprimal_model_epoch{}_{}.pkl'.format(iter_idx, epoch)))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (1 + epoch)
            eta = speed_epoch * (args.epochs - epoch)
            print("Updating input mask, Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))

        round_model_weights(model, param_name='input_mask')
        # refresh X_energy_cache
        reset_Xenergy_cache(energy_info)
        cur_energy = energy_estimator(model)
        cur_energy_relaxed = energy_estimator_relaxed(model)

        iter_idx += 1
        Xbudget -= 0.1
