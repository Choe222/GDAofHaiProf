from GDA import GDA, convergence
import torch 
import matplotlib.pyplot as plt

def clamp(x: torch.Tensor):
    return x.clamp(min=0.0)

x1 = torch.tensor(8.0, requires_grad=True,dtype=torch.float64)
x2 = torch.tensor(54.5, requires_grad=True, dtype=torch.float64)


i = 10000
params = [x1, x2]
optimizer = GDA(params=params, lr=0.7, sigma=0.8, k=0.75,projFunc=clamp) 

best_val = float('inf')
fc = []   

try:
    for o in range(i):
        def closure():
            optimizer.zero_grad()
            f = (x1 - 2*x2 + 1) / (3*x1 + x2 + 5)
            f.backward()
            return f
        f = optimizer.step(closure)
        f_val = f.item()
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
