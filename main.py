import os
import torch.multiprocessing as mp
import parser_config


if __name__ == '__main__':
    global args
    args = parser_config.get_args()
    world_size = args.world_size

    if args.task == 'cifar10':
        from trainer_dsi_cifar10 import activate
    # elif args.task == 'diffusion':
    #     from trainer_diffusion import activate
    elif args.task == 'llm':
        from trainer_dsi_llm import activate
    elif args.task == 'backdoor':
        from trainer_dsi_backdoor import activate
    else:
        raise ValueError('Task not defined.')
    
    mp.spawn(activate, args=(args,), nprocs=args.gpu_per_node*args.process_per_gpu, join=True)


    print('done.\n\n\n')