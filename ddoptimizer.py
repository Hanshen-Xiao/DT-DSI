import random
import time
from typing import Callable, Optional
import numpy as np
import torch
from opacus.optimizers.optimizer import DPOptimizer
import opacus.optimizers.optimizer as optimizer
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

class DataDependentOptimizer(DPOptimizer):
    def __init__(self, dpoptimizer:DPOptimizer,augment_multiplicity:int, num_local_iter:int=1, is_con:bool=False, full_mode:bool=False,log_func=None, local_lr=0.00025, full_mode_do_clip=False, angel_clip=False):
        super().__init__(dpoptimizer.original_optimizer,
                         noise_multiplier=dpoptimizer.noise_multiplier,
                         max_grad_norm=dpoptimizer.max_grad_norm,
                         expected_batch_size=dpoptimizer.expected_batch_size,
                         loss_reduction=dpoptimizer.loss_reduction,
                         generator=dpoptimizer.generator,
                         secure_mode=dpoptimizer.secure_mode
                         )
        if hasattr(dpoptimizer,"step_hook"):
            self.step_hook=dpoptimizer.step_hook
        else:
            self.step_hook=None

        # self.dataset_size=len(dataloader.dataset)
        self.augment_multiplicity=augment_multiplicity
        if augment_multiplicity>1:
            self.apply_augment=True
        else:
            self.apply_augment=False

        self.is_con=is_con
        self.full_mode=full_mode
        self.num_local_iter=num_local_iter
        if self.num_local_iter>1:
            self.full_mode=True

        self.last_params=[p.data.clone() for p in self.params]
        if not is_con or full_mode:
            self.shapes=[p.shape for p in self.params]
            self.sizes=[p.numel() for p in self.params]
        self.num_params=sum([p.numel() for p in self.params])
        for p in self.params:
            p.virtual_grad=None
            if is_con and not self.full_mode:
                p.grad_sample_preserved=None
        self.accumulated_local_updates=0
        self.current_batch_indices=None
        self._batch_updated_flag=False
        self.log_func=log_func
        self.local_lr=local_lr
        self.full_mode_do_clip=full_mode_do_clip
        self.angel_clip=angel_clip

    def renew_last_params(self):
        self.last_params=[p.data.clone() for p in self.params]
        
    def _average_over_augmentations(self):
        if self.apply_augment and not self.full_mode:
            for p in self.params:
                grad_sample = self._get_flat_grad_sample(p)
                p.grad_sample=grad_sample.reshape(
                    -1, self.augment_multiplicity, *grad_sample.shape[1:]
                ).mean(dim=1)

    def get_per_sample_grad_norm(self):
        per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
        return torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    
    def get_summed_grad_norm(self):
        return torch.norm(torch.cat([p.summed_grad.view(-1) for p in self.params]), 2)

    def scale_grad(self):
        
        if self.angel_clip:
            pre_angel_grad_vectors=[]
            for idx in range(len(self.params[0].pre_angel_grad)):
                pre_angel_grad_vectors.append(torch.cat([p.pre_angel_grad[idx].view(-1) for p in self.params]))
            for p in self.params:
                p.pre_angel_grad=None
            avg_grad_vector=torch.mean(torch.stack(pre_angel_grad_vectors),dim=0)

            angel_clipped_grad_vectors=[]
            angels=[]
            for grad_vector in pre_angel_grad_vectors:
                angels.append(angle_between_vectors(grad_vector,avg_grad_vector).item())
            angel_threshold=np.percentile(angels,80)
            for grad_vector in pre_angel_grad_vectors:
                angel_clipped_grad_vectors.append(rotate_vector(grad_vector,avg_grad_vector,alpha=angel_threshold))
            summed_angel_clipped_grad_vector=torch.sum(torch.stack(angel_clipped_grad_vectors),dim=0)
            # convert back to summed_grad shape
            split_tensors = torch.split(summed_angel_clipped_grad_vector, self.sizes)
            angeled_grads = [split_tensor.view(shape) for split_tensor, shape in zip(split_tensors, self.shapes)]
            with torch.no_grad():
                for p,angeled_grad in zip(self.params,angeled_grads):
                    p.summed_grad=angeled_grad

            

        if self.loss_reduction == "mean":
            for p in self.params:
                p.summed_grad /= self.expected_batch_size * self.accumulated_iterations

    def process_summed_grad(self):
        for p in self.params:
            optimizer._check_processed_flag(p.summed_grad)
            if p.virtual_grad is None:
                p.virtual_grad = (p.summed_grad ).view_as(p)
            else:
                p.virtual_grad += (p.summed_grad ).view_as(p)

            optimizer._mark_as_processed(p.summed_grad)

        # if self.log_func:
        #     self.log_func(f"summed_grad_norm: {self.get_summed_grad_norm()}")

        return

    
    def extract_virtual_grad(self):
        list_virtual_grad=[]
        for p in self.params:
            list_virtual_grad.append(p.virtual_grad.flatten())
            p.virtual_grad=None

        if self.is_con and not self.full_mode:
            full_sample_grad=torch.zeros((self.expected_batch_size,self.num_params),device=self.params[0].device)
            n_offset, d_offset = 0, 0
            for p in self.params:
                for tensor in p.grad_sample_preserved:
                    n_i, d_j = tensor.shape[0], tensor.numel() // tensor.shape[0]
                    full_sample_grad[n_offset:n_offset + n_i, d_offset:d_offset + d_j] = tensor.view(n_i, d_j)
                    n_offset += n_i
                d_offset += d_j
                n_offset = 0
                p.grad_sample_preserved=None
                
        if self.is_con and not self.full_mode:
            return torch.cat(list_virtual_grad,dim=0),full_sample_grad
        
        if self.full_mode:
            raw_virtual_grad=torch.cat(list_virtual_grad,dim=0)
            return raw_virtual_grad /torch.max(torch.norm(raw_virtual_grad), torch.tensor(self.max_grad_norm,device=raw_virtual_grad.device))*self.max_grad_norm
        return torch.cat(list_virtual_grad,dim=0)
            

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Following annotations are from the original opacus-DPOptimizer class
        
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True
        self._average_over_augmentations()
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False
        # A real batch (not just a physical batch) is finished
        if self.full_mode or not self.is_con:
            self.scale_grad()
        self.process_summed_grad()
        if self.step_hook:
            self.step_hook(self)
        self._is_last_step_skipped = False
        self.accumulated_local_updates+=1
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[torch.Tensor]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            if self.accumulated_local_updates<self.num_local_iter:
                self.local_update()
                return None
            else:
                self.accumulated_local_updates=0  
                return self.pre_globle_update()
        else:
            return None
            
    def local_update(self):
        with torch.no_grad():
            for p in self.params:
                # p.data -= self.param_groups[0]["lr"] * p.summed_grad
                p.data -= p.summed_grad

    def pre_globle_update(self):
        with torch.no_grad():
            for p,p_last in zip(self.params,self.last_params):
                p.data = p_last
        return self.extract_virtual_grad()

    def global_update(self,flatted_grad):
        if self.is_con:
            raise ValueError("This is a conparison optimizer, only support local update to get virtual grad")
        split_tensors = torch.split(flatted_grad, self.sizes)
        grads = [split_tensor.view(shape) for split_tensor, shape in zip(split_tensors, self.shapes)]
        with torch.no_grad():
            for p,g in zip(self.params,grads):
                p.grad=g
        self.original_optimizer.step()
        self.renew_last_params()

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if not self.full_mode:
            if len(self.grad_samples[0]) == 0:
                # Empty batch
                per_sample_clip_factor = torch.zeros(
                    (0,), device=self.grad_samples[0].device
                )
            else:
                per_param_norms = [
                    g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
                ]
                per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
                per_sample_clip_factor = (
                    self.max_grad_norm / (per_sample_norms + 1e-6)
                ).clamp(max=1.0)
        elif self.full_mode and self.full_mode_do_clip:
            if len(self.grad_samples[0]) == 0:
                # Empty batch
                per_sample_clip_factor = torch.zeros(
                    (0,), device=self.grad_samples[0].device
                )
            else:
                per_param_norms = [
                    g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
                ]
                per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
                per_sample_clip_factor = (
                    self.max_grad_norm_local / (per_sample_norms + 1e-6)
                ).clamp(max=1.0)

        for p in self.params:
            optimizer._check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            if not self.full_mode:
                grad = torch.einsum("i,i...->i...", per_sample_clip_factor, grad_sample)
            else:
                if self.full_mode_do_clip:
                    grad = torch.einsum("i,i...->i...", per_sample_clip_factor, grad_sample)*self.local_lr
                else:
                    grad = grad_sample*self.local_lr
            
            if not self.full_mode:
                if p.grad_sample_preserved is not None:
                    p.grad_sample_preserved += [grad]
                else:
                    p.grad_sample_preserved = [grad]
            
            if self.angel_clip:
                if not hasattr(p, "pre_angel_grad") or p.pre_angel_grad is None:
                    p.pre_angel_grad = [grad]
                else:
                    p.pre_angel_grad += [grad]

            if p.summed_grad is not None:
                p.summed_grad += grad.sum(dim=0)
            else:
                p.summed_grad = grad.sum(dim=0)

            optimizer._mark_as_processed(p.grad_sample)

        # if self.log_func:
        #     self.log_func(f"avg update norm= {torch.mean(per_sample_norms).item():.4f}")
            # self.log_func(f"update norm= {per_sample_norms}")


    def transfer_grad_sample(self):
        with torch.no_grad():
            for param in self.params:
                param.grad_sample = param.grad.clone().unsqueeze(0)


def angle_between_vectors(a, b):
    cos_theta = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0) 
    return torch.arccos(cos_theta)

def rotate_vector(a, b, alpha=torch.pi/4):
    """

    Args:
        a (torch.Tensor): vector to be rotated
        b (torch.Tensor): target
        alpha (float): 

    Returns:
        a' (torch.Tensor): 
    """
    theta = angle_between_vectors(a, b)
    alpha = torch.tensor(alpha,device=theta.device)
    if theta <= alpha:
        return a
    sin_theta = torch.sin(theta)
    sin_alpha = torch.sin(alpha)
    sin_theta_minus_alpha = torch.sin(theta - alpha)
    a_rotated = (sin_alpha / sin_theta) * a + (sin_theta_minus_alpha / sin_theta) * (b / torch.norm(b) * torch.norm(a))

    return a_rotated