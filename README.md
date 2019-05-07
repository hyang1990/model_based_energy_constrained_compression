# model_based_energy_constrained_compression
Code for paper "Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking" (https://openreview.net/pdf?id=BylBr3C9K7)
```
@inproceedings{yang2018energy,
  title={Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking},
  author={Yang, Haichuan and Zhu, Yuhao and Liu, Ji},
  booktitle={ICLR},
  year={2019}
}
```
## Prerequisites


```
Python (3.6)
PyTorch 1.0
```

To use the ImageNet dataset, download the dataset and move validation images to labeled subfolders (e.g., using https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training and testing


### example

To run the training with energy constraint on AlexNet,

```
python energy_proj_train.py --net alexnet --dataset imagenet --datadir [imagenet-folder with train and val folders] --batch_size 128 --lr 1e-3 --momentum 0.9 --l2wd 1e-4 --proj_int 10 --logdir ./log/path-of-log --num_workers 8 --exp_bdecay --epochs 30 --distill 0.5 --nodp --budget 0.2
```

### usage

```
usage: energy_proj_train.py [-h] [--net NET] [--dataset DATASET]
                            [--datadir DATADIR] [--batch_size BATCH_SIZE]
                            [--val_batch_size VAL_BATCH_SIZE]
                            [--num_workers NUM_WORKERS] [--epochs EPOCHS]
                            [--lr LR] [--xlr XLR] [--l2wd L2WD]
                            [--xl2wd XL2WD] [--momentum MOMENTUM]
                            [--lr_decay LR_DECAY] [--lr_decay_e LR_DECAY_E]
                            [--lr_decay_add] [--proj_int PROJ_INT] [--nodp]
                            [--input_mask] [--randinit] [--pretrain PRETRAIN]
                            [--eval] [--seed SEED]
                            [--log_interval LOG_INTERVAL]
                            [--test_interval TEST_INTERVAL]
                            [--save_interval SAVE_INTERVAL] [--logdir LOGDIR]
                            [--distill DISTILL] [--budget BUDGET]
                            [--exp_bdecay] [--mgpu] [--skip1]

Model-Based Energy Constrained Training

optional arguments:
  -h, --help            show this help message and exit
  --net NET             network arch
  --dataset DATASET     dataset used in the experiment
  --datadir DATADIR     dataset dir in this machine
  --batch_size BATCH_SIZE
                        batch size for training
  --val_batch_size VAL_BATCH_SIZE
                        batch size for evaluation
  --num_workers NUM_WORKERS
                        number of workers for training loader
  --epochs EPOCHS       number of epochs to train
  --lr LR               learning rate
  --xlr XLR             learning rate for input mask
  --l2wd L2WD           l2 weight decay
  --xl2wd XL2WD         l2 weight decay (for input mask)
  --momentum MOMENTUM   momentum
  --proj_int PROJ_INT   how many batches for each projection
  --nodp                turn off dropout
  --input_mask          enable input mask
  --randinit            use random init
  --pretrain PRETRAIN   file to load pretrained model
  --eval                evaluate testset in the begining
  --seed SEED           random seed
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  --test_interval TEST_INTERVAL
                        how many epochs to wait before another test
  --save_interval SAVE_INTERVAL
                        how many epochs to wait before save a model
  --logdir LOGDIR       folder to save to the log
  --distill DISTILL     distill loss weight
  --budget BUDGET       energy budget (relative)
  --exp_bdecay          exponential budget decay
  --mgpu                enable using multiple gpus
  --skip1               skip the first W update
```