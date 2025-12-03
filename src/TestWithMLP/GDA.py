import torch 
from torch.optim import Optimizer
from Model import MLP

class convergence(Exception):
    pass

class GDA(Optimizer):
    def __init__(self, params, lr=0.01, sigma=0.5, k=0.75, projFunc=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < sigma < 1.0):
            raise ValueError(f"Invalid sigma: {sigma}")
        if not (0.0 < k <= 1.0):
            raise ValueError(f"Invalid k: {k}")
        self.projFunc = projFunc
    
        defaults = dict(lr=lr, sigma=sigma, k=k)
        super(GDA,self).__init__(params,defaults)
        self.state['global'] = dict(prev_lr=lr)
        
    def step(self, closure=None):
        if(closure == None):
            raise ValueError(f"This optimizer need the the before&current loss.")
        
        group = self.param_groups[0]
        sigma = group['sigma']
        k = group['k']

        global_state = self.state['global']
        prev_lr = global_state.get('prev_lr', group['lr']) 
        
        loss_old = closure()
        sump = 0.0          
        check = True
        with torch.no_grad():
            for g in self.param_groups:
                for p in g['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    x_old = p.data.clone()
                    p.add_(grad, alpha=-prev_lr)
                    if self.projFunc is not None:
                        p.copy_(self.projFunc(p.data))
                    diff = x_old - p.data  
                    if diff.abs().sum() > 0:
                       check = False

                    sump += (diff * grad).sum().item()

        if check == True:
            #return loss_old 
            raise convergence #if the target func is convecx?)
        loss_new = closure()   

        if loss_new.item() > loss_old.item() - sigma * sump:
            new_lr = prev_lr * k
            global_state['prev_lr'] = new_lr
            group['lr'] = new_lr
        else:
            global_state['prev_lr'] = prev_lr
            group['lr'] = prev_lr

        return loss_new
            
       
