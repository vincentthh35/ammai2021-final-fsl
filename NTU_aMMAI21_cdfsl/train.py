import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone

from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.subspace_net import SubspaceNet
from methods.subspace_net_plus import SubspaceNetPlus
from methods.subspace_net_strong import SubspaceNetStrong
import pickle

from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file
from datasets import miniImageNet_few_shot, cifar100_few_shot

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, multi=False):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif optimization == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    if multi == True:
        loader_size = len(base_loader)

    if multi == False:
        max_acc = 0
    else:
        max_acc = [ 0 for i in range(loader_size) ]

    acc_list = []

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        if multi == True:
            model.train_loop(epoch, base_loader[epoch % loader_size], optimizer)
        else:
            model.train_loop(epoch, base_loader,  optimizer)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if multi == True:
            acc = model.test_loop(val_loader[epoch % loader_size])
        else:
            acc = model.test_loop(val_loader)
        acc_list.append(acc)
        if (multi == False and acc > max_acc) or (multi == True and acc > max_acc[epoch % loader_size]):
        # if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            if multi == False:
                max_acc = acc
            else:
                max_acc[epoch % loader_size] = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    os.makedirs(f'./logs/{params.method}_{params.task}', exist_ok=True)
    with open(f'./logs/{params.method}_{params.task}/train_acc.txt', 'a') as f:
        for i, epoch in enumerate(range(params.start_epoch, params.stop_epoch)):
            f.write(f'{epoch}: {acc_list[i]}\n')

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    task   = params.task

    image_size = 224
    optimization = params.optim

    base_loaders = []
    val_loaders = []

    if params.method in ['baseline'] :
        if task in ["fsl", "cdfsl-single"]:
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
        else:
            miniImageNet_datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            cifar100_datamgr = cifar100_few_shot.SimpleDataManager(image_size, batch_size = 16)

            base_loaders.append(miniImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
            base_loaders.append(cifar100_datamgr.get_data_loader( aug = params.train_aug ))
            val_loaders.append(None)

        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['protonet', 'subspace', 'subspace_plus', 'subspace_strong']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)

        if task in ["fsl", "cdfsl-single"]:
            datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)

            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(val_datamgr.get_data_loader(aug = False))

        else:
            mini_ImageNet_datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            mini_ImageNet_val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
            cifar100_datamgr          = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            cifar100_val_datamgr      = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)


            base_loaders.append(mini_ImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(mini_ImageNet_val_datamgr.get_data_loader(aug = False))
            base_loaders.append(cifar100_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(cifar100_val_datamgr.get_data_loader(aug = False))

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'subspace':
            model = SubspaceNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'subspace_plus':
            model = SubspaceNetPlus( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'subspace_strong':
            model = SubspaceNetStrong( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    task_path = 'single' if task in ["fsl", "cdfsl-single"] else 'multi'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, task_path, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # resume training
    if start_epoch != 0:
         filename = get_assigned_file(params.checkpoint_dir, start_epoch - 1)
         model.load_state_dict(torch.load(filename)['state'])

    if task in ["fsl", "cdfsl-single"]:
        model = train(base_loaders[0], val_loaders[0], model, optimization, start_epoch, stop_epoch, params)
    else:
        # stop_epoch = stop_epoch // 2
        model = train(base_loaders, val_loaders, model, optimization, start_epoch, stop_epoch, params, multi=True)
        # for i in range(2):
        #     print('now source domain ', i, ":")
