from scipy.optimize import minimize
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from autograd import grad
import numpy as np
from GDA import GDA, convergence
import torch
import matplotlib.pyplot as plt


def f(x):
    return (x[0]**2 + x[1]**2 + 3) / (1 + 2*x[0] + 8*x[1])
def g1(x):
    return -x[0]**2 - 2*x[0]*x[1] + 4
def g2(x):
    return -x[0]

g1_dx = grad(g1)
g2_dx = grad(g2)
g_dx = [g1_dx,g2_dx]
f_dx = grad(f)
bounds = Bounds([0,0],[np.inf,np.inf])
cons = (
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g1(x)]),
          'jac' : lambda x: np.array([-g1_dx(x)])})

import numpy as np
import torch
from scipy.optimize import minimize, BFGS

# cons, bounds giả sử đã được định nghĩa trước đó

def rosen(x, y):
    """Khoảng cách Euclid giữa x và y (dùng cho scipy, làm việc với numpy)"""
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x - y) ** 2))

def find_min(y: torch.Tensor, n=2):
    # y: torch.Tensor -> numpy
    y_np = y.detach().cpu().numpy()

    # x0: điểm khởi tạo dạng numpy 1D
    x0 = np.random.rand(n)

    res = minimize(
        rosen,
        x0,
        args=(y_np,),          # phải là tuple
        jac="2-point",
        hess=BFGS(),
        constraints=cons,
        method='trust-constr',
        options={'disp': False},
        bounds=bounds
    )

    # trả về torch tensor cùng dtype với y
    return torch.from_numpy(res.x).to(dtype=y.dtype)

x = torch.tensor([0.0, 0.0], requires_grad=True, dtype=torch.float64)
print(find_min(x))            
print(find_min(x).tolist())
params = []
params.append(x)
optimizer = GDA(params=params,lr = 0.9, sigma = 0.5, k = 0.75, projFunc=find_min)

iterations = 1000
best_val = float('inf')
fc = []

try:
    for o in range(iterations):
        def closure():
            optimizer.zero_grad()
            F = f(x)
            F.backward()
            return F
        F = optimizer.step(closure)
        f_val = F.item()
        fc.append(f_val)
        if f_val < best_val:
            best_val = f_val
    print("End")
    print("best f =", best_val)
    
except convergence:
    print("End by convergence")
    print("best f =", best_val)
    
plt.figure(figsize=(8, 5))
plt.plot(fc)
plt.title("Tốc độ hội tụ của GDA đối với hàm lồi")
plt.xlabel("Iteration")
plt.ylabel("Value of function")
plt.grid(True)
plt.savefig("GDAWithConvex.png", dpi=300, bbox_inches='tight')
plt.show()