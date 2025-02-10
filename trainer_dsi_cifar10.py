from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
import os
import random
import time
from datetime import timedelta
import multiprocessing as mp
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_ema import ExponentialMovingAverage

import resnet_modified as resnet
import resnet_wide_modified as wresnet
import parser_config
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from indexeddataset import IndexedRepeatedDataset, TagedRepeatedDataset
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
    vital_args=['epochs','batch_size','description','epsilon','augment_multiplicity','arch','batch_enhencement']
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
        os.makedirs(args.save_dir)

    # prepare ingredients for training
    model, train_loader, val_loader, criterion = prepare_whole_task()
    dist.barrier()
    
    if args.using_cuda:
        log(f"Current device={torch.cuda.current_device()}. ")
        model.cuda()
        criterion = criterion.cuda()
    if not args.full_mode:
        model=GradSampleModule(model)
    args.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            args.ema.load_state_dict(checkpoint['ema'])
            log("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer=DPOptimizer(optimizer,noise_multiplier=args.noise_multiplier,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=False,full_mode=args.full_mode,log_func=log)

    # preparation for distributed training done
    log(f"Prep time={time.time()-total_runtime}")
    # start training

    epoch=args.epochs-1
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        log('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch, val_loader=val_loader)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch % args.save_every == 0 and not args.do_not_save:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'ema': args.ema.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th'))

    # training done, save the model and log the results
    inform_task_done(mission_accomplished=True)
    if not args.do_not_save:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'epoch': epoch + 1,
            'ema': args.ema.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th'))

    total_runtime=time.time()-total_runtime
    file_exists = os.path.exists(args.log_path)
    with nullcontext():
        with open(args.log_path, 'a') as file:
            content = (
                f"{total_runtime:<{widths['Runtime']}.3f}"
                f"{args.epochs:<{widths['Epochs']}}"
                f"{args.batch_size:<{widths['Batch Size']}}"
                f"{args.augment_multiplicity:<{widths['Augmentation']}}"
                f"{args.num_local_iter:<{widths['Local Iters']}}"
                f"{args.lr:<{widths['Learning Rate']}.3f}"
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
    cudnn.benchmark = True

    # prepare ingredients for training
    dist.barrier()
    model, train_loader, _, criterion = prepare_whole_task()

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
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=True,full_mode=args.full_mode,log_func=None)

    input,target,tags=None,None,None


    log('Preparation done.\n')
    iteration=0
    # start tasks from the main process    
    while True:
        end=time.time()
        # communication operation
        if inform_task_done():
            break
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
            if len(this_tasks)==0:
                gather_results_in_main_process(torch.zeros((0,optimizer.num_params),device=args.device))
                iteration+=1
                continue

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
            filtered_input = input[torch.isin(tags, this_tasks)]
            filtered_target = target[torch.isin(tags, this_tasks)]
            log(f"Total backward size={filtered_target.shape[0]*filtered_target.shape[1]}")
            filtered_tags = tags[torch.isin(tags, this_tasks)]
            this_update,sample_grads=do_one_raw_iteration(filtered_input, filtered_target, model, criterion, optimizer)
            # log(f"assertion {torch.norm(this_update-torch.sum(sample_grads,dim=0))}")
            # log(f"...{this_update.dtype}")
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


def prepare_whole_task():
    # model arch choises = ['resnet20', 'WRN_16_4', 'WRN_40_4'] for training on CIFAR-10 from scratch with Data-Specific Indistinguishability
    if args.arch in wide_archs:
        model = wresnet.__dict__[args.arch]()
    else:
        model = resnet.__dict__[args.arch]()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    dataset=TagedRepeatedDataset(dataset,args.augment_multiplicity,)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers if args.rank==0 else 0, pin_memory=True,drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    return model, train_loader, val_loader, criterion

def train(train_loader, model, criterion, optimizer, epoch, val_loader=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    noise_time=AverageMeter()
    all_iter_time=AverageMeter()

    raw_update_list=[]
    diff_list=[]

    # switch to train mode
    model.train()

    end = time.time()
    end_whole = end
    for i, ((input, target),tags) in enumerate(train_loader):

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
            # calculate the raw update
            raw_update=do_one_raw_iteration(input, target, model, criterion, optimizer)

            # recieve the difference
            # communication operation
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)-raw_update

            if args.batch_enhencement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)

        else:
            raw_update=all_reduce(torch.zeros((optimizer.num_params),device=args.device,dtype=torch.float16 if args.half else torch.float32))/len(tags)
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)

            if args.batch_enhencement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)

        all_iter_time.update(time.time()-end_2)

        if (i+1)%args.batch_enhencement!=0:
            end=time.time()
            continue

        end_2=time.time()
        if args.batch_enhencement>1:
            raw_update=torch.mean(torch.stack(raw_update_list), dim=0)
            all_diff=torch.cat(diff_list, dim=0)/args.batch_enhencement
            raw_update_list=[]
            diff_list=[]
        noisy_update=get_privatized_update(raw_update,all_diff,whether_to_log_detail(i,len(train_loader)))
        optimizer.global_update(noisy_update)
        args.ema.update()
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
                
        # if (i+1) % (args.print_freq) == 0 and val_loader:
        #     validate(val_loader, model, criterion)


    return

def do_one_raw_iteration(input, target, model, criterion, optimizer):
    """
    Run one iteration
    """
    load_num=args.physical_size//args.augment_multiplicity
    optimizer.expected_batch_size=len(target)
    # log(f"expected batchsize {optimizer.expected_batch_size}")

    if len(target)%load_num==0:
        physical_iter_num=len(target)//load_num
    else:
        physical_iter_num=len(target)//load_num+1
    if args.full_mode:
        optimizer.expected_batch_size=physical_iter_num

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

            input_var=input_var.reshape(-1,*input_var.shape[2:])
            target_var=target_var.reshape(-1)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            loss.backward()
            if args.full_mode:
                optimizer.transfer_grad_sample()
            result=optimizer.step()
            optimizer.zero_grad()
    if physical_iter_num*optimizer.num_local_iter==0:
        result=torch.zeros((optimizer.num_params),device=args.device)
    return result

def gather_results_in_main_process(this_difference,chunk_sizes=None)-> Optional[torch.Tensor]:
    return gather_results_in_main_process_memoryeffi(this_difference,chunk_sizes)

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

def gather_results_in_main_process_memoryeffi(this_difference,chunk_sizes=None)-> Optional[torch.Tensor]:
    if chunk_sizes:
        result=torch.zeros(sum(chunk_sizes),this_difference.shape[1],device=this_difference.device, dtype=this_difference.dtype)
        start=0
        for i,size in enumerate(chunk_sizes):
            dist.recv(tensor=result[start:start+size],src=i+1)
            start+=size
        return result
    dist.send(tensor=this_difference,dst=0)

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        with args.ema.average_parameters():
            for i, (input, target) in enumerate(val_loader):
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
                loss = criterion(output, target_var)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1)==len(val_loader):
                    log('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i+1, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1))

    log(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    model.train()

    return top1.avg


if __name__ == '__main__':
    global args
    args = parser_config.get_args()
    world_size = 2 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    mp.spawn(activate, args=(world_size,args), nprocs=world_size, join=True)


    print('done.\n\n\n')
