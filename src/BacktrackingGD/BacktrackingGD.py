from logging import StringTemplateStyle
import torch
from torch.optim import Optimizer

class Convergence(Exception):
    pass

class BacktrackingGD(Optimizer):
  def __init__(self,params, lr=1, k=0.75, sigma = 1e-4):
    defaults = dict(lr=lr, k=k, sigma = sigma)
    super(BacktrackingGD,self).__init__(params,defaults)
    self.state['global'] = dict(prev_lr =lr)
  
  def step(self, closure=None):
    group = self.param_groups[0]
    k = group['k']
    sigma = group['sigma']
    current_lr = self.state['global'].get('prev_lr',group['lr'])

    loss_odd = closure()
    check = True
    inner_product_sum =0
    sum_sq_diff =0
    is_satisfied = False

    while not is_satisfied:
      with torch.no_grad():
          for p in group['params']:
            if p.grad is None:
              continue
            grad = p.grad.data
            x_odd=p.data.clone()
            p.data.add_(grad,alpha=-current_lr)
            pram_diff = x_odd - p.data
            sum_sq_diff += torch.sum((x_odd - p.data)**2).item()
            diff = sum_sq_diff **0.5
            if(diff >1e-9):
              check = False
            inner_product_sum += torch.sum(grad**2).item()
          if check:
                pass
              

      loss_next = closure()
      descent_term = sigma *current_lr* inner_product_sum
      is_satisfied = loss_next.item() <= (loss_odd.item() - descent_term)
      if not is_satisfied:
        current_lr *=k

      next_lr =current_lr
      self.state['global']['prev_lr'] = next_lr
      group['lr'] = next_lr
    return loss_next
      

      
        

