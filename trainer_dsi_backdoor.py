import argparse
from contextlib import nullcontext
from copy import deepcopy
import os
import random
import statistics
import sys
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist

import resnet_modified as resnet
import resnet_wide_modified as wresnet
import parser_config
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from indexeddataset import IndexedRepeatedDataset, TagedRepeatedDataset, TaggedDataset
from ddoptimizer import DataDependentOptimizer
import numpy as np
import warnings
from gaussian_noise_optim import isotropic, mirrorpgd
from utils import *

warnings.simplefilter("ignore")

def activate(local_rank,args):
    if args.process_per_gpu>1:
        raise ValueError("process_per_gpu>1 is not supported.")
        os.environ["NCCL_P2P_DISABLE"] = "1"  
        os.environ["NCCL_SHM_DISABLE"] = "1"  
        os.environ["NCCL_IB_DISABLE"] = "1"   
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 

    args.rank=args.rank_node*args.gpu_per_node*args.process_per_gpu+local_rank
    args.device=f'cuda:{local_rank%args.gpu_per_node}'

    current_dir = os.getcwd()
    path_to_add = os.path.join(current_dir, 'BackdoorBench')
    sys.path.append(path_to_add)
    
    if args.rank==0:
        main_process(args.rank,args)
    else:
        helper_process(args.rank,args)

def main_process(global_rank,args_):
    global  best_prec1, total_runtime, args
    best_prec1 = 0
    assert global_rank==0

    #================= Argument Refinement ======================#
    args_.privacy=True
    # setting epsilon to 0 (representing Inf) to disable privacy is recommended
    args_.using_cuda=True
    # setting using_cuda to True is necessary for the current implementation
    #================= Argument Refinement Done =================#

    total_runtime=time.time()
    # establish the communication
    # communication operation
    args=initialize_comunication(global_rank,args_)

    log("=" * 250)
    log("=" * 250)
    log(args)
    vital_args=['epochs','batch_size','lr','description','epsilon']
    for k,v in args.__dict__.items():
        if k in vital_args:
            log(f"{k}:{v}")
    log(f"Communication established.")
    log(f'Process{global_rank}, using {args.device} on node {this_node()}')


    # cuda settings
    # torch.cuda.set_device(args.device)
    cudnn.benchmark = True

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir) and not args.do_not_save:
        os.makedirs(args.save_dir, exist_ok=True)

    # prepare ingredients for training
    model, train_loader, clean_test_dataloader, bd_test_dataloader, criterion = prepare_whole_task_with_backdoor(attack=args.attack,backdoor_args_compliment=None)
    # dist.barrier()
    
    if args.using_cuda:
        log(f"Current device={torch.cuda.current_device()}. ")
        model.cuda()
        criterion = criterion.cuda()
    # args.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # args.ema.load_state_dict(checkpoint['ema'])
            log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))

    if not args.backdoor_from_scratch:
        pretrained_checkpoint = torch.load(f'pretrain/{args.arch}.pth')
        model.linear = nn.Linear(model.linear.in_features, 100)
        model.load_state_dict(pretrained_checkpoint, strict=False)
        model.linear = nn.Linear(model.linear.in_features, 10)
        model.cuda()
    if not args.full_mode:
        model = GradSampleModule(model)

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    if args.using_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[400], last_epoch=args.start_epoch - 1)


    log('total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer=DPOptimizer(optimizer,noise_multiplier=args.noise_multiplier,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=False,full_mode=args.full_mode,log_func=log,local_lr=args.local_lr,full_mode_do_clip=args.full_mode_do_clip,angel_clip=args.angel_clip)
    # args.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    # preparation for distributed training done
    log(f"Prep time={time.time()-total_runtime}")
    # start training

    epoch=args.epochs-1
    test_acc_list=[]
    test_asr_list=[]
    test_ra_list=[]
    if args.full_mode:
        _,chunk_size=distribute_tasks_v2(args.num_source,rank=args.rank,world_size=args.world_size)
        
    log('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        if args.full_mode:
            train_full_mode(train_loader, model, criterion, optimizer, chunk_size, epoch)
        else:
            train(train_loader, model, criterion, optimizer, epoch, val_loader=None)

        if args.using_lr_scheduler:
            lr_scheduler.step()

        if not whether_to_log_detail(epoch,args.epochs):  
            continue      

        test_acc, test_asr, test_ra = evaluate_backdoor(clean_test_dataloader, bd_test_dataloader, model, criterion, whether_to_log=whether_to_log_detail(epoch,args.epochs))
        test_acc_list.append(test_acc)
        test_asr_list.append(test_asr)
        test_ra_list.append(test_ra)

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        if is_best:
            best_tuple=(test_acc,test_asr,test_ra)

        if epoch % args.save_every == 0 and not args.do_not_save:
            save_time=time.time()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                # 'ema': args.ema.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th' if args.privacy else 'checkpoint.th'))
            log(f"Save time={time.time()-save_time}")

    # training done, save the model and log the results
    inform_task_done(mission_accomplished=True)
    if not args.do_not_save:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'epoch': epoch + 1,
            # 'ema': args.ema.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th' if args.privacy else 'checkpoint.th'))
        torch.save(best_tuple,os.path.join(args.save_dir,'result.th'))

    total_runtime=time.time()-total_runtime
    
    log(f"\n* Best ACC={best_prec1:.3f}, ASR={best_tuple[1]:.3f}, RA={best_tuple[2]:.3f}")
    file_exists = os.path.exists(args.log_path)
    with nullcontext():
        with open(args.log_path, 'a') as file:
            content = (
                f"{total_runtime:<{widths['Runtime']}.3f}"
                f"{args.epochs:<{widths['Epochs']}}"
                f"{args.batch_size:<{widths['Batch Size']}}"
                f"{args.augment_multiplicity:<{widths['Augmentation']}}"
                f"{args.num_local_iter:<{widths['Local Iters']}}"
                f"{args.local_lr:<{widths['Learning Rate']}.3f}"
                f"{args.epsilon:<{widths['Noise Multiplier']}}"
                f"{args.max_grad_norm:<{widths['Max Grad Norm']}}"
                f"{args.description:<{widths['Description']}}"
                f"{best_prec1:<{widths['ACC']}.3f}"
            )
            if not file_exists or not contains_string(header,args.log_path):
                file.write(header+'\n')
            file.write(content+'\n')

    model=None
    optimizer=None
    train_loader=None
    val_loader=None
    dataset=None
    torch.cuda.empty_cache()

    # communication operation
    dist.destroy_process_group()

def helper_process(global_rank,args_):
    global  args

    # establish the communication
    # communication operation
    args=initialize_comunication(global_rank,args_)

    log("=" * 250)
    log("=" * 250)
    log(f'Process{global_rank}, using {args.device} on node {this_node()}')

    # cuda settings
    # torch.cuda.set_device(args.device)
    cudnn.benchmark = True

    # prepare ingredients for training
    model, criterion = prepare_whole_task_with_backdoor(attack=args.attack,backdoor_args_compliment=None)

    if args.using_cuda:
        model.cuda()
        criterion = criterion.cuda()

    if args.half:
        model.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if not args.full_mode:
        model=GradSampleModule(model)
    optimizer=DPOptimizer(optimizer,noise_multiplier=args.noise_multiplier,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=True,full_mode=args.full_mode,log_func=None,local_lr=args.local_lr,full_mode_do_clip=args.full_mode_do_clip,angel_clip=args.angel_clip)

    input,target,tags=None,None,None
    if args.full_mode:
        this_tasks,_=distribute_tasks_v2(args.num_source, rank=global_rank, world_size=args.world_size)
        log(f'Helper process local, {len(this_tasks)} tasks allocated. -> {this_tasks if len(this_tasks)<10 else this_tasks[:10]}...')
        model_list=[deepcopy(model) for _ in range(len(this_tasks))]
        optimizer_list=[]
        for model_ in model_list:
            o=torch.optim.SGD(model_.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            o=DPOptimizer(o,noise_multiplier=args.noise_multiplier,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
            optimizer_list.append(DataDependentOptimizer(o,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=True,full_mode=args.full_mode,log_func=None,local_lr=args.local_lr,full_mode_do_clip=args.full_mode_do_clip,angel_clip=args.angel_clip))
            

    log('Preparation done.\n')
    iteration=0
    # start tasks from the main process    
    while True:
        end=time.time()
        # communication operation
        if inform_task_done():
            break
        log(f'waiting time: {time.time()-end:.3f}s')
        end=time.time()
        
        if args.full_mode:
            broadcast_model(model, src=0)
            if args.full_mode_do_clip:
                local_mgn = broadcast_local_mgn()
            this_update_list=[]
            for model_,optimizer_ in zip(model_list,optimizer_list):
                model_.load_state_dict(model.state_dict())
                optimizer_.renew_last_params()
                model_.train()
                if args.full_mode_do_clip:
                    optimizer_.expected_batch_size=args.source_per_group
                    optimizer_.max_grad_norm_local=local_mgn
                else:
                    optimizer_.expected_batch_size=1
            
            # log(f"check_111 {time.time()-end:.4f}s")
            for i in range(args.num_local_iter):
                input,target,tags=prepare_tensors(input,target,tags)
                dist.broadcast(input, src=0)
                dist.broadcast(target, src=0)
                dist.broadcast(tags, src=0)
                # log(f"check_222 {time.time()-end:.4f}s")
                per_tag_input_list=[]
                per_tag_target_list=[]
                for tag in range(1,args.num_source+1):
                    if True: # tag == 1:
                        per_tag_input_list.append(input[tags == tag])
                        per_tag_target_list.append(target[tags == tag])
                    else:
                        benign_count = torch.sum(tags!=1).item()
                        indices = torch.arange(tag*300, (tag+1)*300, device=args.device) % benign_count
                        per_tag_input_list.append(input[tags != 1][indices])
                        per_tag_target_list.append(target[tags != 1][indices])
                all_tag=list(range(1,args.num_source+1))

                for idx, (excluded_tag,model_,optimizer_) in enumerate(zip(this_tasks,model_list,optimizer_list)):
                    # end_debug=time.time()
                    if args.full_mode_do_clip:
                        load_count=0
                        for tag, per_tag_input, per_tag_target in zip(all_tag,per_tag_input_list,per_tag_target_list):
                            if (tag-(excluded_tag+1))%args.num_source>=args.source_per_group:
                                continue
                            if load_count==optimizer_.expected_batch_size-1:
                                optimizer_.signal_skip_step(do_skip=False)
                            else:
                                optimizer_.signal_skip_step(do_skip=True)
                            output = model_(per_tag_input)
                            loss = criterion(output, per_tag_target)
                            loss.backward()
                            optimizer_.transfer_grad_sample()
                            if i==args.num_local_iter-1 and load_count==optimizer_.expected_batch_size-1:
                                this_update_list.append(optimizer_.step())
                            else:
                                optimizer_.step()
                            # log(f"summed_grad norm {optimizer_.get_summed_grad_norm().item():.5f}")
                            optimizer_.zero_grad()
                            load_count+=1
                    else:
                        filtered_input = input[tags != excluded_tag]
                        filtered_target = target[tags != excluded_tag]
                        if len(filtered_target)==0:
                            warnings.warn(f"Single source batch occured.")
                            for param in model_.parameters():
                                    param.grad = torch.zeros_like(param)
                        else:    
                            output = model_(filtered_input)
                            loss = criterion(output, filtered_target)
                            loss.backward()
                        optimizer_.transfer_grad_sample()
                        if i==args.num_local_iter-1:
                            this_update_list.append(optimizer_.step())
                        else:
                            optimizer_.step()
                        optimizer_.zero_grad()

            if len(this_update_list)==0:
                this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            else:
                this_difference=torch.stack(this_update_list)
            gather_results_in_main_process(this_difference)

            log(f'Iter time cost: {time.time()-end:.3f}s')
            iteration+=1
            continue
                    



        input,target,tags=prepare_tensors(input,target,tags)
        dist.broadcast(input, src=0)
        dist.broadcast(target, src=0)
        dist.broadcast(tags, src=0)
        broadcast_model(model, src=0)
        this_tasks,_=distribute_tasks(tags, rank=global_rank, world_size=args.world_size)
        log(f'Iteration[{iteration}], {len(this_tasks)} tasks allocated. -> {this_tasks if len(this_tasks)<10 else this_tasks[:10]}...')
        log(f'syncronization time: {time.time()-end:.3f}s')

        optimizer.renew_last_params()
        update_list=[]
        
        if not args.privacy and args.full_mode:
            this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            gather_results_in_main_process(this_difference)
            iteration+=1
            continue
                
        if args.full_mode:
            for excluded_tag in this_tasks:
                filtered_input = input[tags != excluded_tag]
                filtered_target = target[tags != excluded_tag]
                update_list.append(do_one_raw_iteration(filtered_input, filtered_target, model, criterion, optimizer))

            if len(update_list)==0:
                this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            else:
                this_difference=torch.stack(update_list)
            gather_results_in_main_process(this_difference)

        else:
            tag_update_list=[]
            num_tag_list=[]
            if len(this_tasks)==0:
                full_update=all_reduce(torch.zeros((optimizer.num_params),device=args.device,dtype=torch.float16 if args.half else torch.float32))
            else:
                filtered_input = input[torch.isin(tags, this_tasks)]
                filtered_target = target[torch.isin(tags, this_tasks)]
                log(f"Total backward size={filtered_target.shape[0]}")
                filtered_tags = tags[torch.isin(tags, this_tasks)]
                this_update,sample_grads=do_one_raw_iteration(filtered_input, filtered_target, model, criterion, optimizer)
                full_update=all_reduce(this_update)
            for excluded_tag in this_tasks:
                num_extag=torch.sum(filtered_tags==excluded_tag)
                this_difference=-1/(len(tags)-num_extag)*torch.sum(sample_grads[filtered_tags==excluded_tag],dim=0)+(1/(len(tags)-num_extag)-1/len(tags))*full_update
                update_list.append(this_difference)
            if len(update_list)==0:
                this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            else:
                this_difference=torch.stack(update_list)
            gather_results_in_main_process(this_difference)


        log(f'Iter time cost: {time.time()-end:.3f}s')
        iteration+=1

    log('mission accomplished. cheers mate! \n')
    # communication operation
    dist.destroy_process_group()

def initialize_comunication(rank,args):
    # establish the communication
    # communication operation
    dist.init_process_group(backend="nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                             rank=rank, world_size=args.world_size)
    
    torch.cuda.set_device(rank%args.gpu_per_node)

    arg_list=[args]
    # synchronize the args, [args.rank, args.rank_node, args.device] remain different
    # communication operation
    dist.broadcast_object_list(arg_list, src=0)
    args=arg_list[0]

    args.rank=rank
    args.rank_node=rank//args.gpu_per_node
    args.device=f'cuda:{rank%args.gpu_per_node}'

    return args


def prepare_whole_task_with_backdoor(attack='none',backdoor_args_compliment=None):
    from BackdoorBench.attack.badnet import generate_cls_model

    if args.arch in wide_archs:
        model = wresnet.__dict__[args.arch]()
    elif args.arch in ['resnet20']:
        model = resnet.__dict__[args.arch]()
    else:
        model = generate_cls_model(
            model_name=args.arch,
            num_classes=10,
            image_size=32,
        )
        resnet.replace_batchnorm_with_groupnorm(model)
    criterion = nn.CrossEntropyLoss()

    if args.rank!=0:
        return model, criterion
    
    if attack=='none':    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        dataset=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        group_array = [1]*5000+[0]*45000
        group_array = np.array(group_array)
        num_group = 10
        clean_indices = np.where(group_array == 0)[0]
        np.random.shuffle(clean_indices)
        for (i,indices) in enumerate(np.array_split(clean_indices, num_group-1)):
            group_array[indices] = i+2
        grouped_train_dataset = TaggedDataset(dataset,group_array)
        
        train_loader = torch.utils.data.DataLoader(
            grouped_train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers if args.rank==0 else 0, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        return model, train_loader, val_loader, val_loader, criterion
        
    backdoor_args = argparse.Namespace()
    if attack=='badnet':
        backdoor_args.__dict__ = {
            'dataset': 'cifar10',
            'dataset_path': './data',
            'attack_label_trans': 'all2one',
            'attack_target': 0, 
            'attack': 'badnet',
            'pratio': 1/args.num_source,
            'patch_mask_path': 'BackdoorBench/resource/badnet/trigger_image.png',
            'img_size': (32, 32, 3),
            'input_height': 32,
            'input_width': 32,
            'input_channel': 3,
            'num_classes': 10,
            'model': 'preactresnet18',
            'save_path': f'backdoor/badnet-{args.uuid}',
        }
        if backdoor_args_compliment is not None:
            backdoor_args.__dict__.update(backdoor_args_compliment)
        from BackdoorBench.attack.badnet import BadNet
        attack_box = BadNet()

    elif attack=='blended':
        backdoor_args.__dict__ = {
            'dataset': 'cifar10',
            'dataset_path': './data',
            'attack_label_trans': 'all2one',
            'attack_target': 0,
            'attack': 'blended',
            'pratio': 1/args.num_source,
            # 'pratio': 1,
            'img_size': (32, 32, 3),
            'input_height': 32,
            'input_width': 32,
            'input_channel': 3,
            'num_classes': 10,
            'model': 'preactresnet18',
            'save_path': f'backdoor/blended-{args.uuid}',
            'attack_train_blended_alpha': 0.2,
            'attack_test_blended_alpha': 0.2,
            'attack_trigger_img_path': 'BackdoorBench/resource/blended/hello_kitty.jpeg',
        }
        if backdoor_args_compliment is not None:
            backdoor_args.__dict__.update(backdoor_args_compliment)
        from BackdoorBench.attack.blended import Blended
        attack_box = Blended()
    
    os.makedirs(backdoor_args.save_path, exist_ok=True)
    attack_box.args=backdoor_args
    pre_time = time.time()
    attack_box.stage1_non_training_data_prepare()
    log(f"Dataset Preparation time={time.time()-pre_time}")
    clean_train_dataset_with_transform, \
    clean_test_dataset_with_transform, \
    bd_train_dataset_with_transform, \
    bd_test_dataset_with_transform = attack_box.stage1_results

    group_array = bd_train_dataset_with_transform.wrapped_dataset.poison_indicator.astype(int)
    # group_array = [1]*5000+[0]*45000
    # group_array = np.array(group_array).astype(int)
    num_group = args.num_source
    clean_indices = np.where(group_array == 0)[0]
    np.random.shuffle(clean_indices)
    for (i,indices) in enumerate(np.array_split(clean_indices, num_group-1)):
        group_array[indices] = i+2

    # group_array = [0]*len(bd_train_dataset_with_transform)


    grouped_train_dataset = TaggedDataset(bd_train_dataset_with_transform,group_array)
    
    train_loader = torch.utils.data.DataLoader(
        grouped_train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers if args.rank==0 else 0, pin_memory=True, drop_last=False)

    clean_test_dataloader = torch.utils.data.DataLoader(
        clean_test_dataset_with_transform,
        batch_size=1000, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    bd_test_dataloader = torch.utils.data.DataLoader(
        bd_test_dataset_with_transform,
        batch_size=1000, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return model, train_loader, clean_test_dataloader, bd_test_dataloader, criterion

def train(train_loader, model, criterion, optimizer, epoch, val_loader=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    noise_time=AverageMeter()
    all_iter_time=AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    raw_update_list=[]
    diff_list=[]

    # switch to train mode
    model.train()

    end = time.time()
    end_whole = end
    
    for i,((input, target, *_),tags) in enumerate(train_loader):

        input=input.cuda()
        target=target.cuda()
        tags=tags.cuda()
        if args.half:
            input=input.half()

        # syncronize models, data over all ranks
        # communication operation
        inform_task_done(mission_accomplished=False)
        prepare_tensors(input,target,tags)
        dist.broadcast(input, src=0)
        dist.broadcast(target, src=0)
        dist.broadcast(tags, src=0)
        broadcast_model(model, src=0)
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)

        data_time.update(time.time() - end)
        end_2=time.time()

        if args.full_mode:

            raw_update=do_one_raw_iteration(input, target, model, criterion, optimizer)
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)-raw_update

            if args.batch_enhancement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)

        else:
            raw_update=all_reduce(torch.zeros((optimizer.num_params),device=args.device,dtype=torch.float16 if args.half else torch.float32))/len(tags)
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)

            if args.batch_enhancement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)

        all_iter_time.update(time.time()-end_2)

        if (i+1)%args.batch_enhancement!=0:
            end=time.time()
            continue

        end_2=time.time()
        if args.batch_enhancement>1:
            raw_update=torch.mean(torch.stack(raw_update_list), dim=0)
            all_diff=torch.cat(diff_list, dim=0)/args.batch_enhancement
            raw_update_list=[]
            diff_list=[]
        noisy_update=get_privatized_update(raw_update,all_diff,whether_to_log_detail(i,len(train_loader)))
        optimizer.global_update(noisy_update)
        # args.ema.update()
        noise_time.update(time.time()-end_2)

        batch_time.update(time.time() - end_whole)
        end = time.time()
        end_whole = end
        if whether_to_log_detail(i,len(train_loader)):
                log('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'All_iter_backward {all_iter_time.val:.3f} ({all_iter_time.avg:.3f})\t'
                    'Privatization {noise_time.val:.3f} ({noise_time.avg:.3f})'.format(
                        epoch, (i+1), len(train_loader), batch_time=batch_time,
                        data_time=data_time, noise_time=noise_time, all_iter_time=all_iter_time))


    return

def train_full_mode(train_loader, model, criterion, optimizer, chunk_sizes, epoch, val_loader=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    noise_time=AverageMeter()
    all_iter_time=AverageMeter()
    iterlize_time = time.time()
    iterator=iter(train_loader)
    log(f"iterlize time={time.time()-iterlize_time}")

    raw_update_list=[]
    diff_list=[]

    # switch to train mode
    model.train()

    end = time.time()
    end_whole = end
    inform_task_done(mission_accomplished=False)
    broadcast_model(model, src=0)
    if args.full_mode_do_clip:
        (input, target, *_), tags = next(iterator)
        input=input.cuda()
        target=target.cuda()
        tags=tags.cuda()
        l2_norm_list = []
        grad_vector_list = []
        for tag in range(1,args.num_source+1):
            output = model(input[tags == tag])
            loss = criterion(output, target[tags == tag])
            loss.backward()
            # calculate norm of the gradient
            with torch.no_grad():
                l2_norm_list.append(torch.sqrt(sum(torch.norm(p.grad, p=2) ** 2 for p in model.parameters())).item())
                grad_vector_list.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
            optimizer.original_optimizer.zero_grad()
        local_mgn = statistics.median(l2_norm_list)
        # local_mgn=1e5
        broadcast_local_mgn(local_mgn)
        optimizer.max_grad_norm_local=local_mgn
        log(f"this round local_clipping = {local_mgn:.5f}")
        log(f"l2_norm_list = {[f'{x:.4f}' for x in l2_norm_list]}")

    for i in range(args.num_local_iter):
        if i>0 or not args.full_mode_do_clip:
            try:
                (input, target, *_), tags = next(iterator)
            except StopIteration:
                iterator=iter(train_loader)
                (input, target, *_), tags = next(iterator)

        # log(f"check_111 {time.time()-end_whole:.4f}")
        input=input.cuda()
        target=target.cuda()
        tags=tags.cuda()
        if args.half:
            input=input.half()
        
        prepare_tensors(input,target,tags)
        dist.broadcast(input, src=0)
        dist.broadcast(target, src=0)
        dist.broadcast(tags, src=0)

        # log(f"check_222 {time.time()-end_whole:.4f}")
        if args.full_mode_do_clip:
            optimizer.expected_batch_size=args.num_source
            for load_count in range(args.num_source):
                if load_count==optimizer.expected_batch_size-1:
                    optimizer.signal_skip_step(do_skip=False)
                else:
                    optimizer.signal_skip_step(do_skip=True)
                if True: # load_count==0:
                    filtered_input=input[tags == load_count+1]
                    filtered_target=target[tags == load_count+1]
                else:
                    benign_count = torch.sum(tags!=1).item()
                    indices = torch.arange((load_count+1)*300, (load_count+2)*300, device=args.device) % benign_count
                    filtered_input=input[tags != 1][indices]
                    filtered_target=target[tags != 1][indices]
                output = model(filtered_input)
                loss = criterion(output, filtered_target)
                loss.backward()
                optimizer.transfer_grad_sample()
                raw_update=optimizer.step()
                # log(f"summed_grad norm {optimizer.get_summed_grad_norm().item():.5f}")
                # log(raw_update)
                optimizer.zero_grad()

        else:
            output = model(input)
            loss = criterion(output, target)
            # compute gradient and do SGD step
            loss.backward()
            optimizer.transfer_grad_sample()
            raw_update=optimizer.step()
            optimizer.zero_grad()

        all_iter_time.update(time.time()-end)
        end=time.time()

    per_update=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)
    log(f"per_update_norm={torch.norm(per_update,dim=1)}")
    log(f"center update norm = {torch.norm(raw_update)}")
    all_diff=per_update-raw_update
    end_2=time.time()
    noisy_update=get_privatized_update(raw_update,all_diff,whether_to_log_detail(epoch,args.epochs))
    optimizer.global_update(noisy_update)
    noise_time.update(time.time()-end_2)
    if whether_to_log_detail(epoch,args.epochs):
            log('Epoch: [{0}][{1}local]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Local_iter_backward {all_iter_time.val:.3f} ({all_iter_time.avg:.3f})\t'
                'Privatization {noise_time.val:.3f} ({noise_time.avg:.3f})'.format(
                # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                    epoch, (i+1), batch_time=batch_time,
                    data_time=data_time, noise_time=noise_time, all_iter_time=all_iter_time))
    return


def do_one_raw_iteration(input, target, model, criterion, optimizer):
    """
    Run one iteration
    """
    model.train()
    load_num=args.physical_size//args.augment_multiplicity
    # optimizer.expected_batch_size=1
    # log(f"expected batchsize {optimizer.expected_batch_size}")

    if len(target)%load_num==0:
        physical_iter_num=len(target)//load_num
    else:
        physical_iter_num=len(target)//load_num+1
    optimizer.expected_batch_size=len(target)

    for _ in range(optimizer.num_local_iter):
        for i in range(physical_iter_num):
            if i==physical_iter_num-1:
                optimizer.signal_skip_step(do_skip=False)
                input_var=input[i*load_num:]
                target_var=target[i*load_num:]
            else:
                optimizer.signal_skip_step(do_skip=True)
                input_var=input[i*load_num:(i+1)*load_num]
                target_var=target[i*load_num:(i+1)*load_num]

            # input_var=input_var.reshape(-1,*input_var.shape[2:])
            # target_var=target_var.reshape(-1)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            loss.backward()
            # optimizer.transfer_grad_sample()
            result=optimizer.step()
            optimizer.zero_grad()
    if physical_iter_num*optimizer.num_local_iter==0:
        result=torch.zeros((optimizer.num_params),device=args.device)
    return result

def broadcast_local_mgn(local_mgn=None, src=0):
    if args.rank==0:
        dist.broadcast(torch.tensor(local_mgn,device=args.device), src=src)
        return local_mgn
    else:
        local_mgn=torch.zeros(1,device=args.device)
        dist.broadcast(local_mgn, src=src)
        return local_mgn.item()
    
def distribute_tasks_v2(num_source , rank, world_size):
    tasks = torch.tensor(list(range(1,num_source+1)),device=args.device)
    num_tasks = len(tasks)
    
    base_chunk_size = num_tasks // (world_size - 1)
    remainder = num_tasks % (world_size - 1)

    chunk_sizes = [base_chunk_size + 1 if i < remainder else base_chunk_size for i in range(world_size - 1)]
    chunks = torch.split(tasks, chunk_sizes)

    chunks = [torch.zeros(1, dtype=tasks.dtype).cuda()] + list(chunks)

    # chunk_to_receive = torch.zeros_like(chunks[rank], device=args.device)
    chunk_to_receive = [None]
    if rank == 0:
        dist.scatter_object_list(chunk_to_receive, scatter_object_input_list=chunks, src=0)
    else:
        dist.scatter_object_list(chunk_to_receive, scatter_object_input_list=None, src=0)

    return chunk_to_receive[0].cuda(), chunk_sizes

def gather_results_in_main_process(this_difference,chunk_sizes=None,num=None)-> Optional[torch.Tensor]:
    return gather_results_in_main_process_memoryeffi(this_difference,chunk_sizes,num)

    tensor_size = torch.tensor([this_difference.shape[0]], device=args.device)
    all_sizes = [torch.zeros(1,dtype=tensor_size.dtype, device=args.device) for _ in range(args.world_size)]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([int(size.item()) for size in all_sizes])
    if args.rank == 0:
        this_difference = torch.zeros(max_size, this_difference.shape[1], device=args.device, dtype=torch.float16 if args.half else torch.float32)
    else:
        this_difference = torch.cat([this_difference, torch.zeros(max_size - this_difference.shape[0], this_difference.shape[1], device=args.device, dtype=this_difference.dtype)], dim=0)

    if chunk_sizes:
        results =[torch.zeros_like(this_difference,device=args.device) for _ in range(len(chunk_sizes)+1)]
        dist.gather(tensor=this_difference, gather_list=results, dst=0)
        results =[results[i+1][:size] for i, size in enumerate(chunk_sizes)]
        return torch.cat(results,dim=0)
    dist.gather(tensor=this_difference, dst=0)

def gather_results_in_main_process_memoryeffi(this_difference,chunk_sizes=None,num=None)-> Optional[torch.Tensor]:
    if chunk_sizes:
        if num is not None:
            for j in range(len(chunk_sizes)):
                chunk_sizes[j]=0
            chunk_sizes[num%5]=1
            chunk_sizes[5+num%5]=1
        result=torch.zeros(sum(chunk_sizes),this_difference.shape[1],device=this_difference.device, dtype=this_difference.dtype)
        start=0
        for i,size in enumerate(chunk_sizes):
            dist.recv(tensor=result[start:start+size],src=i+1)
            start+=size
        return result
    dist.send(tensor=this_difference,dst=0)


def evaluate_backdoor(clean_test_dataloader, bd_test_dataloader, model, criterion, whether_to_log=True):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    test_acc = AverageMeter()
    test_asr = AverageMeter()
    test_ra = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end_whole = time.time()
    end = time.time()
    with torch.no_grad():
        with nullcontext():
            for i, (input, target) in enumerate(clean_test_dataloader):
                if args.using_cuda:
                    target = target.cuda()
                    input_var = input.cuda()
                    target_var = target
                else:
                    input_var = input
                    target_var = target

                if args.half:
                    input_var = input_var.half()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var.long())

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input.size(0))
                test_acc.update(prec1.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1)==len(clean_test_dataloader) and whether_to_log:
                    log('Test_clean_data: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'test_acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i+1, len(clean_test_dataloader), batch_time=batch_time, loss=losses,
                            top1=test_acc))

    # evaluate backdoor test
    if clean_test_dataloader == bd_test_dataloader:
        if whether_to_log:            
            log(' * Test_acc {top1.avg:.3f} Test_asr {asr.avg:.3f} Test_ra {ra.avg:.3f}'
            .format(top1=test_acc, asr=test_asr, ra=test_ra))
        return test_acc.avg, 0, 0
    
    end = time.time()
    with torch.no_grad():
        with nullcontext():
            for i, (input, target, original_index, poison_indicator, original_targets) in enumerate(bd_test_dataloader):
                if args.using_cuda:
                    target = target.cuda()
                    input_var = input.cuda()
                    target_var = target
                    original_targets = original_targets.cuda()
                else:
                    input_var = input
                    target_var = target

                if args.half:
                    input_var = input_var.half()

                # compute output
                output = model(input_var)
                output = output.float()

                # measure accuracy and record loss
                prec1_attack = accuracy(output.data, target)[0]
                prec1_robust = accuracy(output.data, original_targets)[0]

                test_asr.update(prec1_attack.item(), input.size(0))
                test_ra.update(prec1_robust.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1)==len(bd_test_dataloader) and whether_to_log:
                    log('Test_backdoor_data: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        # 'test_acc {acc.val:.3f} ({acc.avg:.3f})\t'
                        'test_asr {asr.val:.3f} ({asr.avg:.3f})\t'
                        'test_ra {ra.val:.3f} ({ra.avg:.3f})'.format(
                            i+1, len(bd_test_dataloader), batch_time=batch_time,
                            asr=test_asr, ra=test_ra))
    
    if whether_to_log:
        log(f'Evaluation time: {time.time()-end_whole:.4f}')
        log(' * Test_acc {top1.avg:.3f} Test_asr {asr.avg:.3f} Test_ra {ra.avg:.3f}'
              .format(top1=test_acc, asr=test_asr, ra=test_ra))
    
    # model.train()

    return test_acc.avg, test_asr.avg, test_ra.avg
