import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from DataProcess import *
from Model import MLP
from torch.optim import Adam
import os
import matplotlib.pyplot as plt

def evaluate(model, dataloader, device):
    model.eval()
    all_risk_preds= []
    all_risk_labels= []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            risk_labels = batch["risk"].to(device)
            logits = model(inputs)
            preds = torch.sigmoid(logits)
            all_risk_preds.append(preds.cpu())
            all_risk_labels.append(risk_labels.cpu())

    all_risk_preds = torch.cat(all_risk_preds, dim=0)
    all_risk_labels = torch.cat(all_risk_labels, dim=0)
    binary_preds = (all_risk_preds > 0.5).int()
    micro_f1 = f1_score(all_risk_labels.numpy(), binary_preds.numpy(), average='micro', zero_division=0)
    return micro_f1

def train_and_evaluate(model, train_loader, val_loader, epochs=10, learning_rate=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Loss
    loss_CL = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_score = -1
    scaler = GradScaler()
    print("Start training...")
    loss_history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch["input"].to(device)
            risk_labels = batch["risk"].to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits = model(inputs)
                loss = loss_CL(logits, risk_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            loss_history.append(loss.item())
            progress_bar.set_postfix({'train_loss': train_loss / (progress_bar.n + 1)})
        avg_train_loss = train_loss / len(train_loader)
        micro_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Micro-F1: {micro_f1:.4f}")
        if micro_f1 > best_score:
            best_score = micro_f1
            torch.save(model.state_dict(), 'Trained_model_Adam')
            print(f"*** New best model saved! F1 Score: {best_score:.4f} ***")
        print("-" * 80)
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, color='blue', linewidth=1)

    plt.title("Training Loss Curve (per batch)", fontsize=16)
    plt.xlabel("Batch index", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.yscale("log")     
    plt.savefig("Adam.png", dpi=300, bbox_inches='tight')
    plt.show()


df = pd.read_csv('dynamic_supply_chain_logistics_dataset.csv')
mapping_dict = {"High Risk": 2, "Moderate Risk": 1, "Low Risk" : 0}
df_risk = df['risk_classification'].map(mapping_dict).values
VALID_SIZE = 0.2
RANDOM_STATE = 42

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df_risk
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
train_dataset = MyData(train_df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP()
model.to(device)
try:
    state_dict = torch.load('Trained_model_Adam', map_location=device)
    model.load_state_dict(state_dict)
    print("Loading model from checkpoint successfully.")
except FileNotFoundError as e: 
    print(f"Error when loading checkpoint: {e}. Training from scratch...")
train_and_evaluate(model, train_loader, train_loader, epochs=10, learning_rate=1e-3)
"""
model = MLP().to(device)
model_path = "Trained_model_Adam"
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded trained model successfully.")
else:
    raise FileNotFoundError("File not found.")

test_data = MyData(test_df)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True, drop_last=True)
micro_f1= evaluate(model, test_loader, device)
print(f"Micro-F1: {micro_f1:.4f}")
