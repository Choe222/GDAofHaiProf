import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def train():
    X, y = make_classification(
    n_samples=3000,
    n_features=20,        # nhiều feature (có noise)
    n_informative=5,      # chỉ 5 feature thực sự informative
    n_redundant=8,        # nhiều feature phụ thuộc (redundant)
    n_repeated=2,         # vài feature lặp lại
    n_classes=5,
    n_clusters_per_class=3,  # mỗi lớp có nhiều cụm (multi-modal)
    class_sep=0.3,           # giảm độ phân tách giữa lớp
    flip_y=0.15,             # 15% nhãn bị flip (label noise)
    weights=[0.6, 0.1, 0.1, 0.1, 0.1], # tạo imbalance
    random_state=42
    )
    X += np.random.normal(scale=0.5, size=X.shape)  # feature noise
    n_outliers = int(0.01 * X.shape[0])             # 1% outliers
    idx = np.random.choice(X.shape[0], n_outliers, replace=False)
    X[idx] += np.random.normal(0, 10.0, size=(n_outliers, X.shape[1]))  # large outliers

    # Chuyển sang PyTorch
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)

    # Chia train/test
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

   # --------------------------- CHUẨN BỊ MÔ HÌNH TRAIN -------------------------------------

    model=simpleModel(input_dim=20)
    model.train()
    optimizer = BacktrackingGD(model.parameters(), lr=1,k=0.75, sigma = 1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:

              outputs = None
              def closure():
                  nonlocal outputs
                  optimizer.zero_grad()
                  outputs = model(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  return loss

              loss = optimizer.step(closure)
              
              with torch.no_grad():
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
              
              avg_loss = epoch_loss / len(train_loader)
              accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    train()

  