import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Load and preprocess data
input_data = torch.tensor(np.loadtxt('input_data1.txt').reshape(-1, 256), dtype=torch.float32)
target_data = torch.tensor(np.loadtxt('target_data1.txt').reshape(-1, 256), dtype=torch.float32)

# Split data into train, validation, and test sets
train_data = TensorDataset(input_data[:6000], target_data[:6000])
val_data = TensorDataset(input_data[6000:8000], target_data[6000:8000])
test_data = TensorDataset(input_data[8000:], target_data[8000:])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Define model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.LayerNorm(1024),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256)  # Output layer for regression
        )
    
    def forward(self, x):
        return self.model(x)

model = RegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-5)

# Define training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5000, patience=30):
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Early stopping and ReduceLROnPlateau
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break
    
    return train_loss_history, val_loss_history

# Train the model
train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate on test set
model.eval()
test_loss = 0.0
test_mae = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        test_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)

print(f'Test Loss (MSE): {test_loss}')
print(f'Test MAE: {test_mae}')

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.tight_layout()
plt.show()
