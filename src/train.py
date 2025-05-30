import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(train_data, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")