import torch
import numpy as np
from scipy.optimize import minimize, Bounds
import time
from GDA import *
from scipy.optimize import Bounds
import autograd.numpy as np1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

def getLr(n):
    beta = 0.741271
    alpha = 3*beta**(3/2)*(n+1)**(1/2)
    return 4*beta**(3/2)*n**(1/2) + 3*alpha

def f(x: torch.Tensor, a: torch.Tensor ,e: torch.Tensor):
    beta = 0.741271
    alpha = 3*beta**(3/2)*(x.shape[0]+1)**(1/2)

    xx = x @ x  # dot product
    k = a @ x + alpha * xx + (beta * (e @ x)) / torch.sqrt(1 + beta * xx)
    return k

EPS = 1e-8

def g1_log(x):
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        return -np.inf
    return np.sum(np.log(x))  # >= 0

def g1_log_jac(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / x  

cons = ({
    "type": "ineq",
    "fun": lambda x: np.array([g1_log(x)]),
    "jac": lambda x: np.array([g1_log_jac(x)])
})

bounds = Bounds(EPS*np.ones(1), np.inf) 

def rosen(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x - y) ** 2))

def find_min(y: torch.Tensor):
    y_np = y.detach().cpu().numpy().astype(np.float64)
    n = y_np.shape[0]
    x0 = np.random.rand(n)

    local_bounds = Bounds(EPS*np.ones(n), np.inf)

    res = minimize(
        rosen,
        x0,
        args=(y_np,),
        jac="2-point",
        constraints=cons,
        bounds=local_bounds,
        method="trust-constr",
        options={"disp": False},
    )
    return torch.from_numpy(res.x).to(dtype=y.dtype, device=y.device)


n = [10,20,50,100,200,500]

#Not change x,a,e for testing GD
x = []
aStore = []
eStore = []
#GDA

for i in range(len(n)):
    print(n[i])
    lr = 5/getLr(n[i])
    best = float('inf')
    t1 = time.time()
    
    x1 = torch.rand(n[i], dtype=torch.float32, requires_grad=True, device=device)
    x.append(x1)
    
    a = torch.rand(x1.shape[0], requires_grad=True, dtype=torch.float32, device=device)
    aStore.append(a)
    
    e = torch.arange(1, x1.shape[0] + 1, dtype=torch.float32, device=device)
    eStore.append(e)
    
    params = [x1]
    optimizer = GDA(
        params=params,
        lr=lr,
        sigma=0.5,
        k=0.75,
        projFunc=find_min
    )

    best_val = float('inf')
    stop = -1
    try:
        for it in range(100):
            stop = it
            def closure():
                optimizer.zero_grad()
                F = f(x1,a,e)
                F.backward()
                return F
            F = optimizer.step(closure)
            f_val = F.item()
            if(f_val < best):
                best = f_val
    except convergence:
        print("Convergence in " + str(stop) + "iters")
    t2 = time.time()
    print("The best value achieved: " + str(best))
    print("Time of GDA:", t2-t1)

#GD
for i in range(len(n)):
    print(n[i])
    best = float('inf')
    t1 = time.time()
    lr = 1/getLr(n[i])
    
    params = [x[i]]
    optimizer = GDA(
        params=params,
        lr=lr,
        sigma=0.5,
        k=1,
        projFunc=find_min
    )

    best_val = float('inf')
    stop = -1
    try:
        for it in range(100):
            stop = it
            def closure():
                optimizer.zero_grad()
                F = f(x[i],aStore[i],eStore[i])
                F.backward()
                return F
            F = optimizer.step(closure)
            f_val = F.item()
            if(f_val < best):
                best = f_val
    except convergence:
        print("Convergence in " + str(stop) + "iters")
    t2 = time.time()
    print("The best value achieved: " + str(best))
    print("Time of GD: ", t2-t1)