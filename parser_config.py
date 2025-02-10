import argparse
import os
import uuid
import resnet_modified as resnet


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
model_names.append('WRN_40_4')
model_names.append('WRN_16_4')
model_names.append('preactresnet18')

def create_parser():
    
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

# training arguments
    parser.add_argument('--task', default='cifar10', type=str,
                        choices=['cifar10','diffusion','llm','backdoor'],
                        help='task type')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1000, type=int,
                        metavar='N', help='mini-batch size (default: 5000)')
    parser.add_argument('--physical-size', default=8000, type=int,
                        metavar='N', help='physical batch size (default: 1000)')
    parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--using-lr-scheduler',dest='using_lr_scheduler',action='store_true',
                        help='use learning rate scheduler to run')    
    parser.add_argument('--batch-enhencement',default=1,type=int, 
                        help='batch_enhencement*batchsize=real batch size')

# Arragement arguments
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    # this is an expired argument
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    
    # this is an expired argument
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='logging', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--log-fname', default="log.txt", type=str,
                        help='log filename')
    # parser.add_argument('--grad-obs-fname', default="grad_obs.csv", type=str,
    #                     help='grad norm observation filename')
    parser.add_argument('--email-notice', dest='email_notice', action='store_true',
                        help='send email notice when training is done')
    parser.add_argument('--do-not-save', dest='do_not_save', action='store_true',
                        help='do not save checkpoint')
    parser.add_argument('--use-resume','-r', action='store_true', help='use resume')
    

# Device arguments
    parser.add_argument('--using-cuda',dest='using_cuda',action='store_true',
                        help='use cuda to run')
    
    # no need to pass this argument
    parser.add_argument('--device',
                        help='cuda device',
                        default='cuda:0', type=str)

    # following 2 arguments are not used in the current version
    parser.add_argument('--cpu-option',action='store_true',
                        help='use cpu to handle large size all_diff')
    parser.add_argument('--process-per-gpu', default=1, type=int,
                        metavar='N', help='number of processes per gpu')      

# Privacy arguments
    parser.add_argument('--privacy', dest='privacy', action='store_true',
                        help='Activate dsi privacy scheme')
    parser.add_argument('--max-grad-norm', '--mgn', default=1, type=float,
                        metavar='MGN', help='maximal gradient norm')
    # parser.add_argument('--delta', default=1e-5, type=float,
    #                     help='Parameter for privacy accounting. Probability of not achieving privacy guarantees')
    parser.add_argument('--epsilon', '--eps', default=1, type=int,
                        metavar='EPS', help='privacy guarantee (epsilon,delta) ')
    
    # this is an expired argument
    parser.add_argument('--noise-multiplier',default=0, type=float,
                        metavar='NOISE', help='... ')

# Extra operations arguments
    parser.add_argument('--description', default='data_dependent', type=str,
                        metavar='DESCRIPTION', help='... ')
    parser.add_argument('--augment-multiplicity', default=4, type=int,
                        metavar='AUGMENT', help='... ')
    parser.add_argument('--num-local-iter', default=2, type=int,
                        metavar='LOCAL_ITER', help='... ')
    parser.add_argument('--full-mode',action='store_true', help='Use full mode')
    parser.add_argument('--local-lr', default=0.00025, type=float,
                        metavar='LOCAL_LR', help='... ')
    parser.add_argument('--full-mode-do-clip',action='store_true', help='clip gradients in local updates if True')
    parser.add_argument('--angel_clip',action='store_true', help='clip gradients in local updates if True')


# Distributed arguments
    parser.add_argument('--gpu-per-node', default=2, type=int,
                        metavar='N', help='number of gpus in one phsical node')
    parser.add_argument('--num-nodes', default=1, type=int,
                        metavar='N', help='number of nodes')
    parser.add_argument('--rank-node', default=0, type=int,
                        metavar='N', help='rank of node')
    parser.add_argument('--master-addr', default='localhost', type=str,
                        metavar='ADDR', help='master address')
    parser.add_argument('--master-port', default='29500', type=str,
                        metavar='PORT', help='master port')
    
    
    return parser

def create_parser_cifar10():

    parser = argparse.ArgumentParser(description='Extra arguments for CIFAR10')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet20)')

    return parser

def create_parser_llm():
    
    parser = argparse.ArgumentParser(description='Extra arguments for LLM')

# language model arguments
    parser.add_argument('--block-size', default=1024, type=int,
                        metavar='BLOCKSIZE', help='len(token) of each datapoint')
    parser.add_argument('--canary-repeat', default=1, type=int,
                        metavar='REPEAT', help='repeat canary times')
    parser.add_argument('--wiki-size', default=20, type=int,
                        metavar='WIKISIZE', help='size of wiki-103')
    parser.add_argument('--duplication', action='store_true',
                        help='duplicate 1000 data points of wiki-103 for 10 times')
    parser.add_argument('--model-name-or-path', default='gpt2', type=str,
                        choices=['gpt2','opt-350m','opt-125m','gpt2-medium'],
                        metavar='MODEL', help='model name or path')
    parser.add_argument('--canary-prompt', default='normal', type=str,
                        choices=['normal','wizard'],
                        metavar='PROMPT', help='prompt for canary')
    parser.add_argument('--do-ref-model', action='store_true',
                        help='use reference model in evaluation')
    parser.add_argument('--train-head-only', action='store_true',
                        help='train head only')
    parser.add_argument('--add-canary', action='store_true',
                        help='add canary to the training set')
    
    return parser

def create_parser_backdoor():
    
    parser = argparse.ArgumentParser(description='Extra arguments for backdoor')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet20)')
# backdoor arguments
    parser.add_argument('--attack', default='none', type=str,
                        choices=['none','badnet','blended'],
                        help='attack type')
    parser.add_argument('--num-source', default=10, type=int,
                        metavar='N', help='number of data source (pratio=1/num_source)')
    parser.add_argument('--backdoor-from-scratch', action='store_true',
                        help='train backdoor from scratch')
    parser.add_argument('--source-per-group', default=0, type=int,
                        metavar='N', help='number of data source per group')
    
    return parser

def get_args():
    parser = create_parser()
    args, remaining_argv = parser.parse_known_args()
    if args.task == 'cifar10':
        parser_additional = create_parser_cifar10()
        args.__dict__.update(vars(parser_additional.parse_args(remaining_argv)))
    elif args.task == 'llm':
        parser_additional = create_parser_llm()
        args.__dict__.update(vars(parser_additional.parse_args(remaining_argv)))
    elif args.task == 'backdoor':
        parser_additional = create_parser_backdoor()
        args.__dict__.update(vars(parser_additional.parse_args(remaining_argv)))
        if args.source_per_group==0:
            args.source_per_group=args.num_source-1

    args.log_path=os.path.join(args.save_dir, args.log_fname)
    if args.use_resume:
        args.resume=os.path.join(args.save_dir, 'checkpoint.th')
    args.uuid = str(uuid.uuid4())
    args.world_size = args.gpu_per_node * args.num_nodes * args.process_per_gpu
    if args.num_local_iter>1:
        args.full_mode=True

    return args

def refresh_args(args):
    # args.clip_strategy=clip_option_box[args.clip_mode]
    args.clip_strategy['factor']=args.factor
    args.log_path=os.path.join(args.save_dir, args.log_fname)
    args.grad_obs_path=os.path.join(args.save_dir, args.grad_obs_fname)
    
    return args