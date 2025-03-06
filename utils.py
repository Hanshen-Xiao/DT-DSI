import os
import torch
import torch.distributed as dist
from gaussian_noise_optim import mirrorpgd

args = None

__all__ = ['get_privatized_update',
            'save_checkpoint', 'log', 'contains_string', 'whether_to_log_detail',
            'initialize_comunication','this_node', 'inform_task_done', 'all_reduce', 'distribute_tasks', 'broadcast_model',
            'prepare_shape', 'prepare_tensors', 'AverageMeter', 'accuracy', 'widths', 'header', 'wide_archs', 'multiplier']


def get_privatized_update(raw_update,all_diff,flag_logging):
    if not args.privacy:
        return raw_update
    if args.epsilon==0:
        if flag_logging:
            log(f"noise=0 , raw_update_norm={torch.norm(raw_update,2)}")
            log(f"avg difference norm={torch.mean(torch.norm(all_diff,2,dim=1))}")
        return raw_update

    noise_multiplier=multiplier[args.epsilon]*(args.epochs)**0.5
    if flag_logging:
        log(f"\n->->->-> privatization module logging... \n  epsilon: {args.epsilon}  noise_multiplier: {noise_multiplier}")

    basis_3,L_3, lnorm=mirrorpgd(all_diff.T,log_process=flag_logging,log_func=log)
    noise_3= (basis_3 @ (L_3 @ torch.randn(L_3.shape[1],device=args.device,dtype=L_3.dtype)).to(basis_3.device)).to(raw_update.device) *noise_multiplier#((args.epochs/2)**0.5)

    if flag_logging:
        raw_update_norm=torch.norm(raw_update,2)
        log(f"Expected l_inf: {lnorm[0]*noise_multiplier}, l_2: {lnorm[1]*noise_multiplier}")
        log(f"noise_norm_inf(determined, in original)={torch.max(torch.abs(noise_3))}, mPGD_noise_norm={torch.norm(noise_3)}, raw_update_norm={raw_update_norm}")
        log(f"noise/update: l_inf/l2 {lnorm[0]*noise_multiplier/raw_update_norm},  l2/l2 {lnorm[1]*noise_multiplier/raw_update_norm}")
        log(f"-<-<-<-< privatization module finished -<-<-<-<")
    return raw_update+noise_3


'''
    io operations
'''

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def log(*outs):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # if args.rank > 1:
    #     return
    with open(f'{args.save_dir}/output_r{args.rank}.txt', 'a') as log_file:
        print(*outs, file=log_file)

def contains_string(search_string, file_path='save_temp/log.txt'):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return search_string in content
    except FileNotFoundError:
        log(f"File {file_path} not found.")
        return False
    
def whether_to_log_detail(i: int, len_loader: int =None):
    if (i+1) % args.print_freq == 0:
        return True
    
    if len_loader:
        if (i+1) == len_loader:
            return True
        else:
            return False
        
    if hasattr(args,"len_train_loader"):
        if (i+1) == args.len_train_loader:
            return True
        
    return False



'''
    distributed operations
'''

def initialize_comunication(rank,args):
    # establish the communication
    # communication operation
    dist.init_process_group(backend="nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                             rank=rank, world_size=args.world_size)
    
    torch.cuda.set_device(args.device)
    device=args.device

    arg_list=[args]
    # synchronize the args, [args.rank, args.rank_node, args.device] remain different
    # communication operation
    dist.broadcast_object_list(arg_list, src=0)
    args=arg_list[0]

    args.rank=rank
    args.rank_node=rank//(args.gpu_per_node*args.process_per_gpu)
    args.device=device
    # args.device=f'cuda:{rank%args.gpu_per_node}'
    globals()['args']=args

    return args

def this_node():
    node_id = os.environ.get("SLURM_NODEID", "0")
    node_list = os.environ.get("SLURM_NODELIST", "")

    if node_list:
        node_names = os.popen(f"scontrol show hostnames {node_list}").read().split()
        current_node = node_names[int(node_id)] if node_id.isdigit() else "unknown_node"
    else:
        current_node = "unknown_node"
    return current_node

def inform_task_done(mission_accomplished=False)->bool:
    '''
    Inform the helper process that the task is accomplished as return value is True
    '''
    status=torch.tensor([mission_accomplished],device=args.device)
    dist.broadcast(status,src=0)
    return status.item()

def all_reduce(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def distribute_tasks(tags, rank, world_size):
    tasks = torch.unique(tags)
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

def broadcast_model(model, src=0):
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

def prepare_shape(shapes=[None],dtypes=[None]):
    dist.scatter_object_list(shapes,[shapes for _ in range(args.world_size)], src=0)
    dist.scatter_object_list(dtypes,[dtypes for _ in range(args.world_size)], src=0)
    return shapes[0],dtypes[0]

# to be modified
def prepare_tensors(inputs=None,targets=None,tags=None):
    if args.rank==0:
        shapes=[inputs.shape,targets.shape,tags.shape]
        dtypes=[inputs.dtype,targets.dtype,tags.dtype]
        prepare_shape(shapes,dtypes)
        return
    else:
        shapes,dtypes=prepare_shape()
        if inputs==None or inputs.shape!=shapes[0]:
            inputs=torch.zeros(shapes[0],dtype=dtypes[0],device=args.device)
        if targets==None or targets.shape!=shapes[1]:
            targets=torch.zeros(shapes[1],dtype=dtypes[1],device=args.device)
        if tags==None or tags.shape!=shapes[2]:
            tags=torch.zeros(shapes[2],dtype=dtypes[2],device=args.device)
        return inputs,targets,tags


'''
    statistics tools
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


'''
    constants
'''

widths = {
    'Runtime': 15,
    'Epochs': 15,
    'Batch Size': 15,
    'Augmentation': 15,
    'Local Iters': 15,
    'Learning Rate': 20,
    'Noise Multiplier': 20,
    'Max Grad Norm': 20,
    'Description': 25,
    'ACC': 20,
}

header = (
    f"{'Runtime':<{widths['Runtime']}}"
    f"{'Epochs':<{widths['Epochs']}}"
    f"{'Batch_Size':<{widths['Batch Size']}}"
    f"{'Augmentation':<{widths['Augmentation']}}"
    f"{'Local_Iters':<{widths['Local Iters']}}"
    f"{'Learning_Rate':<{widths['Learning Rate']}}"
    f"{'Epsilon':<{widths['Noise Multiplier']}}"
    f"{'Max_Grad_Norm':<{widths['Max Grad Norm']}}"
    f"{'Description':<{widths['Description']}}"
    f"{'ACC':<{widths['ACC']}}"
)

wide_archs=['WRN_40_4', 'WRN_16_4']

multiplier=[0, 4.90056, 2.49929, 1.69768, 1.29608, 1.05453, 0.893066, 0.777387, 0.69035]