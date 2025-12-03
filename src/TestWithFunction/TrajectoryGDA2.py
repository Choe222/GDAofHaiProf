from scipy.optimize import minimize
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from autograd import grad
import numpy as np
from GDA import GDA, convergence
import torch
import matplotlib.pyplot as plt
import time

def f(x):
    return (torch.e**(abs(x[1] - 3)) - 30)/(x[0]**2+x[2]**2+2*x[3]**2+4)
def g1(x):
    return (x[0] + x[2])**2 + 2*x[3]**2 - 10
def g2(x):
    return (x[1] - 1)**2 - 1
def h1(x):
    return 2*x[0] + 4*x[1] + x[2] + 1

g1_dx = grad(g1)
g2_dx = grad(g2)
h1_dx = grad(h1)
#bounds = Bounds([0,0],[np.inf,np.inf])
cons = (
    {   
        'type': 'ineq',
        'fun': lambda x: -g1(x),
        'jac': lambda x: -g1_dx(x)
    },
    {  
        'type': 'ineq',
        'fun': lambda x: -g2(x),
        'jac': lambda x: -g2_dx(x)
    },
    {
        'type': 'eq',
        'fun': lambda x: h1(x),
        'jac': lambda x: h1_dx(x)
    },
)

#fiding Projection
def rosen(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x - y) ** 2))

def find_min(y: torch.Tensor, n=4):
    y_np = y.detach().cpu().numpy()
    x0 = np.random.rand(n)
    res = minimize(
        rosen,
        x0,
        args=(y_np,),        
        jac="4-point",
        hess=BFGS(),
        constraints=cons,
        method='trust-constr',
        options={'disp': False},
        #bounds=bounds
    )

    return torch.from_numpy(res.x).to(dtype=y.dtype)

num_runs = 3
iterations = 30
sol_all = [] #Save trajectory of 3 points

for run in range(num_runs):
    t1 = time.time()
    x = torch.rand(4, requires_grad=True, dtype=torch.float32)
    params = [x]
    optimizer = GDA(
        params=params,
        lr=0.9,
        sigma=0.8,
        k=0.75,
        projFunc=find_min
    )

    best_val = float('inf')
    x_traj = []  #save trajectory 
    x_traj.append(x.detach().cpu().numpy().copy())

    try:
        for it in range(iterations):
            def closure():
                optimizer.zero_grad()
                F = f(x)
                F.backward()
                return F
            F = optimizer.step(closure)
            f_val = F.item()
            if(f_val < best_val):
                best_val = f_val
            x_traj.append(x.detach().cpu().numpy().copy())
        sol_all.append(np.array(x_traj))
        print(f_val)
    except convergence:
        print("Convergence!")
        sol_all.append(np.array(x_traj))
    t2 = time.time()
    print("GDA:", t2-t1)
    
#Show&Save png
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 16})

t = np.arange(iterations + 1)

for traj in sol_all:
        t = np.arange(traj.shape[0])  
        plt.plot(t, traj[:, 0], linewidth=1, label=r"$x_1(t)$" if "x1" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(t, traj[:, 1], linewidth=1, label=r"$x_2(t)$" if "x2" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(t, traj[:, 2], linewidth=1, label=r"$x_2(t)$" if "x2" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(t, traj[:, 3], linewidth=1, label=r"$x_2(t)$" if "x2" not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel("iteration")
plt.ylabel("x(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig('TrajectoryGDA2.png')
plt.show()

